"""Stage 4: world-grounded SMPL-X motion per track with GVHMR (runs in the `gvhmr` env).

Replaces GVHMR's demo script: instead of tracking the largest person itself, each Stage 3
track is processed separately with our boxes. Frames inside the track's span are written to
a work video (resampled to `input_fps`), then ViTPose 2D keypoints, HMR2 image features and
SimpleVO camera rotation are computed and fed to GVHMR.

Output data/smpl/<track_id>.npz holds GVHMR's native world frame (gravity-aligned, **Y-up**):
global_orient (T,3), body_pose (T,63), transl (T,3), betas (10,) [+ betas_per_frame],
fps, frame_idx (shot-relative source frames), the in-camera params, K and camera R_w2c.
"""
from __future__ import annotations

import logging
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .manifest import failed, ok, read_manifest, upsert

log = logging.getLogger(__name__)

COORD_CONVENTION = "gvhmr_ay_y_up"   # stored with every output


# ---------------------------------------------------------------- setup
def ensure_links(cfg: Config) -> Path:
    """Link checkpoints + body models into the GVHMR tree where its code expects them."""
    root = (cfg.root / cfg["reconstruct"]["gvhmr_root"]).resolve()
    ckpt = (cfg.root / cfg["reconstruct"]["checkpoints"]).resolve()
    body = cfg.path_of("body_models").resolve()
    (root / "inputs").mkdir(exist_ok=True)
    for link, target in [(root / "inputs" / "checkpoints", ckpt), (ckpt / "body_models", body)]:
        if link.is_symlink() and link.resolve() != target:
            link.unlink()
        if not link.exists():
            link.symlink_to(target, target_is_directory=True)
    missing = [p for p in [ckpt / "gvhmr/gvhmr_siga24_release.ckpt",
                           ckpt / "hmr2/epoch=10-step=25000.ckpt",
                           ckpt / "vitpose/vitpose-h-multi-coco.pth",
                           body / "smplx/SMPLX_NEUTRAL.npz"] if not p.exists()]
    if missing:
        raise FileNotFoundError("GVHMR inputs missing: " + ", ".join(map(str, missing)))
    return root


def resample_indices(n_src: int, src_fps: float, input_fps) -> tuple[np.ndarray, float]:
    """Source frame indices to feed GVHMR and the resulting fps."""
    if input_fps == "native" or float(input_fps) >= src_fps:
        return np.arange(n_src), src_fps
    tgt = float(input_fps)
    n = int(np.floor((n_src - 1) * tgt / src_fps)) + 1
    idx = np.round(np.arange(n) * src_fps / tgt).astype(int)
    return np.clip(idx, 0, n_src - 1), tgt


class Gvhmr:
    """Holds the GVHMR model + preprocessors so they load once per run."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.root = ensure_links(cfg)
        os.chdir(self.root)          # GVHMR loads checkpoints via relative paths
        import hydra
        import torch
        from hydra import compose, initialize_config_module
        from hmr4d.configs import register_store_gvhmr
        from hmr4d.utils.preproc import Extractor, VitPoseExtractor

        with initialize_config_module(version_base="1.3", config_module="hmr4d.configs"):
            register_store_gvhmr()
            gcfg = compose(config_name="demo", overrides=["video_name=wct2g1"])
        self.model = hydra.utils.instantiate(gcfg.model, _recursive_=False)
        self.model.load_pretrained_model(str(self.root / gcfg.ckpt_path))
        self.model = self.model.eval().cuda()
        self.vitpose = VitPoseExtractor(tqdm_leave=False)
        self.extractor = Extractor(tqdm_leave=False)
        self.torch = torch

    def write_work_video(self, shot_video: Path, idx: np.ndarray, out: Path, fps: float) -> None:
        from hmr4d.utils.video_io_utils import get_video_reader, get_writer

        want = set(idx.tolist())
        out.parent.mkdir(parents=True, exist_ok=True)
        writer = get_writer(str(out), fps=fps, crf=17)
        reader = get_video_reader(str(shot_video))
        last = int(idx.max())
        # idx may repeat a frame when resampling; write in order with repeats
        counts = pd.Series(idx).value_counts()
        for i, img in enumerate(reader):
            if i in want:
                for _ in range(int(counts[i])):
                    writer.write_frame(img)
            if i >= last:
                break
        writer.close()
        reader.close()

    def run_track(self, video: Path, bbx_xyxy: np.ndarray, static: bool) -> dict:
        torch = self.torch
        from hmr4d.utils.geo.hmr_cam import create_camera_sensor, estimate_K, get_bbx_xys_from_xyxy
        from hmr4d.utils.geo_transform import compute_cam_angvel
        from hmr4d.utils.net_utils import detach_to_cpu
        from hmr4d.utils.preproc import SimpleVO
        from hmr4d.utils.video_io_utils import get_video_lwh

        rc = self.cfg["reconstruct"]
        length, width, height = get_video_lwh(str(video))
        if length != len(bbx_xyxy):
            raise RuntimeError(f"work video has {length} frames, boxes {len(bbx_xyxy)}")
        bbx = torch.from_numpy(bbx_xyxy).float()
        bbx_xys = get_bbx_xys_from_xyxy(bbx, base_enlarge=rc["bbox_enlarge"]).float()
        with torch.no_grad():
            kp2d = self.vitpose.extract(str(video), bbx_xys)
            feats = self.extractor.extract_video_features(str(video), bbx_xys)
        if static:
            R_w2c = torch.eye(3).repeat(length, 1, 1)
        else:
            vo = SimpleVO(str(video), scale=rc["vo"]["scale"], step=rc["vo"]["step"],
                          method=rc["vo"]["method"], f_mm=rc["f_mm"])
            R_w2c = torch.from_numpy(vo.compute()[:, :3, :3]).float()
        if rc["f_mm"] is not None:
            K = create_camera_sensor(width, height, rc["f_mm"])[2].repeat(length, 1, 1)
        else:
            K = estimate_K(width, height).repeat(length, 1, 1)
        data = {"length": torch.tensor(length), "bbx_xys": bbx_xys, "kp2d": kp2d,
                "K_fullimg": K, "cam_angvel": compute_cam_angvel(R_w2c), "f_imgseq": feats}
        with torch.no_grad():
            pred = detach_to_cpu(self.model.predict(data, static_cam=static))
        g, c = pred["smpl_params_global"], pred["smpl_params_incam"]
        betas_pf = g["betas"].numpy()
        return dict(
            global_orient=g["global_orient"].numpy(), body_pose=g["body_pose"].numpy(),
            transl=g["transl"].numpy(), betas=betas_pf.mean(0), betas_per_frame=betas_pf,
            incam_global_orient=c["global_orient"].numpy(), incam_transl=c["transl"].numpy(),
            K_fullimg=K.numpy(), R_w2c=R_w2c.numpy(), kp2d=kp2d.numpy(),
            bbx_xys=bbx_xys.numpy())


# ---------------------------------------------------------------- review renders
def render_review(g: Gvhmr, work_video: Path, out_npz: Path, out_mp4: Path, fps: float) -> None:
    """Side-by-side: SMPL mesh over the footage | world view with ground plane.
    Mirrors GVHMR's demo render_incam/render_global."""
    import torch
    from einops import einsum
    from hmr4d.utils.geo.hmr_cam import create_camera_sensor
    from hmr4d.utils.geo_transform import apply_T_on_points, compute_T_ayfz2ay
    from hmr4d.utils.smplx_utils import make_smplx
    from hmr4d.utils.video_io_utils import get_video_lwh, get_video_reader, get_writer, merge_videos_horizontal
    from hmr4d.utils.vis.renderer import Renderer, get_global_cameras_static, get_ground_params_from_points

    d = np.load(out_npz)
    T = len(d["transl"])
    smplx = make_smplx("supermotion").cuda()
    smplx2smpl = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt").cuda()
    faces = make_smplx("smpl").faces
    J_reg = torch.load("hmr4d/utils/body_model/smpl_neutral_J_regressor.pt").cuda()
    betas = torch.from_numpy(d["betas_per_frame"]).float().cuda()

    def verts(go, tr):
        out = smplx(global_orient=torch.from_numpy(go).float().cuda(),
                    body_pose=torch.from_numpy(d["body_pose"]).float().cuda(),
                    transl=torch.from_numpy(tr).float().cuda(), betas=betas)
        return torch.stack([smplx2smpl @ v for v in out.vertices])

    L, W, H = get_video_lwh(str(work_video))
    tmp_a, tmp_b = out_mp4.with_suffix(".incam.mp4"), out_mp4.with_suffix(".global.mp4")
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    # in-camera
    vc = verts(d["incam_global_orient"], d["incam_transl"])
    r = Renderer(W, H, device="cuda", faces=faces, K=torch.from_numpy(d["K_fullimg"][0]))
    w = get_writer(str(tmp_a), fps=fps, crf=23)
    for i, img in enumerate(get_video_reader(str(work_video))):
        if i >= T:
            break
        w.write_frame(r.render_mesh(vc[i], img, [0.8, 0.8, 0.8]))
    w.close()
    # world view (start at origin, facing +Z, on the ground) — as in GVHMR demo
    vg = verts(d["global_orient"], d["transl"])
    off = einsum(J_reg, vg[0], "j v, v i -> j i")[0]
    off[1] = vg[:, :, 1].min()
    vg = vg - off
    Tf = compute_T_ayfz2ay(einsum(J_reg, vg[[0]], "j v, l v i -> l j i"), inverse=True)
    vg = apply_T_on_points(vg, Tf)
    joints = einsum(J_reg, vg, "j v, l v i -> l j i")
    gR, gT, gL = get_global_cameras_static(vg.cpu(), beta=2.0, cam_height_degree=20,
                                           target_center_height=1.0)
    _, _, K = create_camera_sensor(W, H, 24)
    r = Renderer(W, H, device="cuda", faces=faces, K=K)
    sc, cx, cz = get_ground_params_from_points(joints[:, 0], vg)
    r.set_ground(sc * 1.5, cx, cz)
    col = torch.ones(3).float().cuda() * 0.8
    w = get_writer(str(tmp_b), fps=fps, crf=23)
    for i in range(T):
        w.write_frame(r.render_with_ground(vg[[i]], col[None], r.create_camera(gR[i], gT[i]), gL))
    w.close()
    merge_videos_horizontal([str(tmp_a), str(tmp_b)], str(out_mp4))
    tmp_a.unlink()
    tmp_b.unlink()


# ---------------------------------------------------------------- entry point
def run(cfg: Config, limit: int | None = None, ids: list[str] | None = None) -> None:
    rc = cfg["reconstruct"]
    tracks = read_manifest(cfg, "tracks")
    tracks = tracks[tracks["status"] == "ok"]
    if ids:   # accept track ids or shot ids
        tracks = tracks[tracks["track_id"].isin(ids) | tracks["shot_id"].isin(ids)]
    smpl = read_manifest(cfg, "smpl")
    if not smpl.empty:
        done = set(smpl.loc[smpl["status"] == "ok", "track_id"])
        tracks = tracks[~tracks["track_id"].isin(done)]
    if limit:
        tracks = tracks.head(limit)
    log.info("reconstruct: %d tracks to process", len(tracks))
    if tracks.empty:
        return
    shots = read_manifest(cfg, "shots").set_index("shot_id")
    rng = random.Random(cfg["seed"])
    render_ids = set(ids or []) | set(rng.sample(list(tracks["track_id"]),
                                                 min(rc["review"]["sample_n"], len(tracks))))
    root = cfg.root          # resolve all our paths before GVHMR chdirs
    g = Gvhmr(cfg)
    work_root = cfg.path_of("smpl") / "_work"

    for _, t in tracks.iterrows():
        tid = t["track_id"]
        row = dict(track_id=tid, shot_id=t["shot_id"], path=None)
        try:
            tr = np.load(root / t["path"])
            shot = shots.loc[t["shot_id"]]
            src_fps = float(shot["fps"])
            local, fps = resample_indices(len(tr["frame_idx"]), src_fps, rc["input_fps"])
            frame_idx = tr["frame_idx"][local]
            bbx = tr["bbox_xyxy"][local]
            work = work_root / tid / "input.mp4"
            g.write_work_video(root / shot["path"], frame_idx, work, fps)
            static = t["shot_id"] in set(rc["static_shot_ids"])
            res = g.run_track(work, bbx, static)
            out = cfg.path_of("smpl") / f"{tid}.npz"
            np.savez_compressed(out, **res, fps=fps, frame_idx=frame_idx, src_fps=src_fps,
                                static_cam=static, coord=COORD_CONVENTION,
                                config_hash=cfg.hash)
            row.update(T=len(frame_idx), fps=fps, static_cam=static,
                       path=str(out.relative_to(root)), status=ok())
            if tid in render_ids:
                ov = cfg.path_of("review") / "smpl_renders" / f"{tid}.mp4"
                render_review(g, work, out, ov, fps)
                log.info("render: %s", ov)
        except Exception as e:  # noqa: BLE001
            log.exception("reconstruct failed for %s", tid)
            row["status"] = failed(e)
        upsert(cfg, "smpl", [row])
        log.info("%s: %s", tid, row["status"])
