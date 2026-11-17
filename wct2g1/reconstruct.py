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
from .manifest import shard, filter_ids, failed, ok, read_manifest, upsert

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
    """Source frame indices to feed GVHMR and the resulting fps. Uses an integer frame stride
    (the one closest to src_fps/input_fps): GVHMR assumes evenly spaced frames, and a
    fractional stride (50 -> 30 fps gives steps of 2,1,2,2,1...) turns into jitter."""
    if input_fps == "native":
        return np.arange(n_src), src_fps
    k = max(1, int(round(src_fps / float(input_fps))))
    return np.arange(0, n_src, k), src_fps / k


def rotation_from_homography(pts0: np.ndarray, pts1: np.ndarray, K: np.ndarray) -> np.ndarray | None:
    """Camera rotation (frame0 -> frame1) for a purely rotating camera: H = K R K^-1.
    Returns None if no homography can be fitted."""
    import cv2

    if len(pts0) < 4:
        return None
    H, _ = cv2.findHomography(pts0.astype(np.float64), pts1.astype(np.float64), cv2.RANSAC, 4.0)
    if H is None:
        return None
    M = np.linalg.inv(K) @ H @ K
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    return R * np.sign(np.linalg.det(R))


def patch_two_view_solver() -> dict:
    """pycolmap>=4 returns no relative pose for panning-only (homography) pairs, where
    GVHMR's solver crashes. Fall back to the homography rotation (zero translation; GVHMR
    only uses R_w2c), else identity. Returns a counter of fallbacks."""
    from hmr4d.utils.preproc.relpose import solver_two_view as stv

    cls = stv.PycolmapRansacTwoViewGeometrySolver
    stats = {"pairs": 0, "homography": 0, "identity": 0}
    if getattr(cls, "_wct2g1_patched", False):
        return cls._wct2g1_stats
    orig = cls.solve

    def solve(self, pts0, pts1):
        stats["pairs"] += 1
        try:
            return orig(self, pts0, pts1)
        except AttributeError:          # answer.cam2_from_cam1 is None
            T = np.eye(4)
            R = rotation_from_homography(pts0, pts1, self.camera_matrix)
            if R is None:
                stats["identity"] += 1
            else:
                stats["homography"] += 1
                T[:3, :3] = R
            return T

    cls.solve = solve
    cls._wct2g1_patched = True
    cls._wct2g1_stats = stats
    return stats


class Gvhmr:
    """Holds the GVHMR model + preprocessors so they load once per run."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.root = ensure_links(cfg)
        os.chdir(self.root)          # GVHMR loads checkpoints via relative paths
        import hydra
        import torch
        from hydra import compose, initialize_config_module
        import hmr4d.model.gvhmr.gvhmr_pl_demo  # noqa: F401  (registers model/gvhmr/gvhmr_pl_demo)
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
        self.vo_stats = patch_two_view_solver()

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

    def dpvo_rotations(self, video: Path, width: int, height: int, length: int):
        """World-to-camera rotations (L,3,3) from DPVO, as GVHMR's demo does with --use_dpvo.
        Frames are fed in-process with the same preprocessing as GVHMR's SLAMModel: its reader
        subprocess runs under DPVO's global 'spawn' start method, which re-imports our CLI in
        the child and loads every model a second time on the GPU."""
        import cv2
        from pytorch3d.transforms import quaternion_to_matrix

        torch = self.torch
        from dpvo.config import cfg as dpvo_cfg
        from dpvo.dpvo import DPVO
        from hmr4d.utils.geo.hmr_cam import convert_K_to_K4, estimate_K

        v = self.cfg["reconstruct"]["vo"]
        r = v["dpvo_resize"]
        dpvo_cfg.merge_from_file(str(self.root / "third-party/DPVO/config/default.yaml"))
        dpvo_cfg.BUFFER_SIZE = length + v["dpvo_buffer_margin"]
        intr = (convert_K_to_K4(estimate_K(width, height)) * r).cuda()
        cap = cv2.VideoCapture(str(video))
        slam, n = None, 0
        try:
            while True:
                ok_, img = cap.read()
                if not ok_:
                    break
                img = cv2.resize(img, None, fx=r, fy=r, interpolation=cv2.INTER_AREA)
                h, w = img.shape[:2]
                img = torch.from_numpy(img[: h - h % 16, : w - w % 16]).permute(2, 0, 1).cuda()
                if slam is None:
                    slam = DPVO(dpvo_cfg, "inputs/checkpoints/dpvo/dpvo.pth",
                                ht=img.shape[1], wd=img.shape[2], viz=False)
                with torch.no_grad():
                    slam(n, img, intr)
                n += 1
            with torch.no_grad():
                for _ in range(12):                              # as SLAMModel.process
                    slam.update()
                traj = slam.terminate()[0]                       # (L, 7): xyz, quat xyzw
        finally:
            cap.release()
            del slam
            torch.cuda.empty_cache()
        if len(traj) != length:
            raise RuntimeError(f"DPVO returned {len(traj)} poses for {length} frames")
        q = torch.from_numpy(traj[:, [6, 3, 4, 5]]).float()      # -> wxyz
        return quaternion_to_matrix(q).mT

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
            R_w2c = None
            if rc["vo"]["backend"] == "dpvo":
                R_w2c = self.dpvo_rotations(video, width, height, length)
                if not torch.isfinite(R_w2c).all():
                    log.warning("DPVO returned non-finite poses; falling back to SimpleVO")
                    R_w2c = None
            if R_w2c is None:
                vo = SimpleVO(str(video), scale=rc["vo"]["scale"], step=rc["vo"]["step"],
                              method=rc["vo"]["method"], f_mm=rc["f_mm"])
                before = dict(self.vo_stats)
                R_w2c = torch.from_numpy(vo.compute()[:, :3, :3]).float()
                log.info("SimpleVO pairs %d, homography fallbacks %d, identity fallbacks %d",
                         *(self.vo_stats[k] - before[k] for k in ("pairs", "homography", "identity")))
        if rc["f_mm"] is not None:
            K = create_camera_sensor(width, height, rc["f_mm"])[2].repeat(length, 1, 1)
        else:
            K = estimate_K(width, height).repeat(length, 1, 1)
        data = {"length": torch.tensor(length), "bbx_xys": bbx_xys, "kp2d": kp2d,
                "K_fullimg": K, "cam_angvel": compute_cam_angvel(R_w2c), "f_imgseq": feats}
        with torch.no_grad():
            pred = detach_to_cpu(self.model.predict(data, static_cam=static))
        for k, v in pred["smpl_params_global"].items():
            if not torch.isfinite(v).all():
                raise RuntimeError(f"GVHMR produced non-finite {k}")
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
        tracks = filter_ids(tracks, ids)
    smpl = read_manifest(cfg, "smpl")
    if not smpl.empty:
        done = set(smpl.loc[smpl["status"] == "ok", "track_id"])
        tracks = tracks[~tracks["track_id"].isin(done)]
    tracks = shard(tracks, "track_id")
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
                try:                    # a review render must not fail a good reconstruction
                    render_review(g, work, out, ov, fps)
                    log.info("render: %s", ov)
                except Exception:  # noqa: BLE001
                    log.exception("review render failed for %s", tid)
        except Exception as e:  # noqa: BLE001
            log.exception("reconstruct failed for %s", tid)
            row["status"] = failed(e)
        upsert(cfg, "smpl", [row])
        log.info("%s: %s", tid, row["status"])
