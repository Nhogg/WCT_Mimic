"""Stage 6: retarget clean SMPL-X clips to the Unitree G1 29-DoF with GMR (`gmr` env).

Writes data/g1/<clip_id>.csv in the format mjlab's csv_to_npz reads (Unitree convention):
one row per frame at 50 fps = root xyz (3), root quaternion **xyzw** (4), 29 joint angles in
the G1 MJCF order (identical in GMR and mjlab; checked). No header.
"""
from __future__ import annotations

import logging
import os
import random
from pathlib import Path

import numpy as np

from .config import Config
from .manifest import shard, filter_ids, failed, ok, read_manifest, rejected, upsert

log = logging.getLogger(__name__)
os.environ.setdefault("MUJOCO_GL", "egl")   # GPU nodes are headless; must precede mujoco import

N_DOF = 29


# ---------------------------------------------------------------- format helpers
def qpos_to_rows(qpos: np.ndarray) -> np.ndarray:
    """MuJoCo qpos (T, 7+29) [pos, quat wxyz, dofs] -> CSV rows [pos, quat xyzw, dofs].
    Quaternion signs are made continuous so interpolation never takes the long way."""
    q = qpos[:, 3:7].copy()
    for t in range(1, len(q)):
        if np.dot(q[t], q[t - 1]) < 0:
            q[t] = -q[t]
    return np.concatenate([qpos[:, :3], q[:, [1, 2, 3, 0]], qpos[:, 7:]], axis=1)


def write_csv(path: Path, rows: np.ndarray) -> None:
    if rows.shape[1] != 7 + N_DOF:
        raise ValueError(f"expected {7 + N_DOF} columns, got {rows.shape[1]}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, rows, delimiter=",", fmt="%.6f")


def read_csv_as_qpos(path: Path) -> np.ndarray:
    """Inverse of the writer, using mjlab's own reading convention (xyzw -> wxyz)."""
    m = np.loadtxt(path, delimiter=",", ndmin=2)
    return np.concatenate([m[:, :3], m[:, 3:7][:, [3, 0, 1, 2]], m[:, 7:]], axis=1)


def limit_fraction(dofs: np.ndarray, lo: np.ndarray, hi: np.ndarray, tol: float) -> float:
    at = (dofs <= lo + tol) | (dofs >= hi - tol)
    return float(at.any(1).mean())


def saturated_joints(dofs: np.ndarray, lo: np.ndarray, hi: np.ndarray, tol: float,
                     names: list[str], min_frac: float = 0.1) -> str:
    """'name:frac,...' for joints at a limit in >= min_frac of frames, worst first."""
    at = ((dofs <= lo + tol) | (dofs >= hi - tol)).mean(0)
    order = [i for i in np.argsort(-at) if at[i] >= min_frac]
    return ",".join(f"{names[i].removesuffix('_joint')}:{at[i]:.2f}" for i in order)


# ---------------------------------------------------------------- GMR
def retarget_clip(cfg: Config, clip_npz: Path, height: float) -> tuple[np.ndarray, float]:
    from general_motion_retargeting import GeneralMotionRetargeting as GMR
    from general_motion_retargeting.utils.smpl import get_smplx_data_offline_fast, load_smplx_file

    rc = cfg["retarget"]
    smplx_data, body_model, smplx_out, gmr_height = load_smplx_file(
        str(clip_npz), str(cfg.path_of("body_models")))
    frames, fps = get_smplx_data_offline_fast(smplx_data, body_model, smplx_out,
                                              tgt_fps=rc["fps"])
    h = height if rc["height_source"] == "measured" else gmr_height
    gmr = GMR(actual_human_height=h, src_human="smplx", tgt_robot=rc["robot"], verbose=False)
    qpos = np.stack([gmr.retarget(f) for f in frames])
    return qpos, float(fps)


def robot_model(cfg: Config):
    import mujoco
    from general_motion_retargeting.params import ROBOT_XML_DICT

    return mujoco.MjModel.from_xml_path(str(ROBOT_XML_DICT[cfg["retarget"]["robot"]]))


def joint_limits(model) -> tuple[np.ndarray, np.ndarray]:
    import mujoco

    hinge = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    r = model.jnt_range[hinge]
    return r[:, 0], r[:, 1]


def hinge_names(model) -> list[str]:
    import mujoco

    return [model.joint(j).name for j in range(model.njnt)
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]


def render_preview(model, qpos: np.ndarray, fps: float, out: Path, w: int, h: int) -> None:
    import imageio.v2 as iio
    import mujoco

    data = mujoco.MjData(model)
    r = mujoco.Renderer(model, h, w)
    cam = mujoco.MjvCamera()
    cam.distance, cam.elevation, cam.azimuth = 3.5, -15, 135
    out.parent.mkdir(parents=True, exist_ok=True)
    with iio.get_writer(str(out), fps=fps, codec="libx264", quality=7) as wr:
        for q in qpos:
            data.qpos[:] = q
            mujoco.mj_forward(model, data)
            cam.lookat[:] = [q[0], q[1], 0.8]
            r.update_scene(data, cam)
            wr.append_data(r.render())
    r.close()


# ---------------------------------------------------------------- entry point
def run(cfg: Config, limit: int | None = None, ids: list[str] | None = None) -> None:
    rc = cfg["retarget"]
    clips = read_manifest(cfg, "clean")
    clips = clips[clips["status"] == "ok"]
    if ids:
        clips = filter_ids(clips, ids)
    g1 = read_manifest(cfg, "g1")
    if not g1.empty:
        done = set(g1.loc[~g1["status"].str.startswith("failed:"), "clip_id"])
        clips = clips[~clips["clip_id"].isin(done)]
    clips = shard(clips, "clip_id")
    if limit:
        clips = clips.head(limit)
    log.info("retarget: %d clips to process", len(clips))
    if clips.empty:
        return
    model = robot_model(cfg)
    lo, hi = joint_limits(model)
    names = hinge_names(model)
    rng = random.Random(cfg["seed"])
    preview = set(ids or []) | set(rng.sample(list(clips["clip_id"]),
                                              min(rc["review"]["sample_n"], len(clips))))
    for _, c in clips.iterrows():
        cid = c["clip_id"]
        row = dict(clip_id=cid, track_id=c["track_id"], path=None)
        try:
            qpos, fps = retarget_clip(cfg, cfg.root / c["path"], float(c["human_height"]))
            if abs(fps - rc["fps"]) > 1e-6:
                raise RuntimeError(f"GMR returned {fps} fps, expected {rc['fps']}")
            lf = limit_fraction(qpos[:, 7:], lo, hi, rc["limit_tol_rad"])
            row.update(n_frames=len(qpos), fps=fps, limit_frac=lf,
                       limit_joints=saturated_joints(qpos[:, 7:], lo, hi, rc["limit_tol_rad"],
                                                     names),
                       min_root_z=float(qpos[:, 2].min()))
            out = cfg.path_of("g1") / f"{cid}.csv"        # written for rejected clips too (review)
            write_csv(out, qpos_to_rows(qpos))
            row["path"] = str(out.relative_to(cfg.root))
            mx = rc["max_limit_frac"]
            row["status"] = rejected("joint_limits") if mx is not None and lf > mx else ok()
            if cid in preview:
                pv = cfg.path_of("review") / "g1_previews" / f"{cid}.mp4"
                try:                    # a preview must not change the clip's status
                    render_preview(model, qpos, fps, pv, rc["review"]["width"],
                                   rc["review"]["height"])
                    log.info("preview: %s", pv)
                except Exception:  # noqa: BLE001
                    log.exception("preview failed for %s", cid)
        except Exception as e:  # noqa: BLE001
            log.exception("retarget failed for %s", cid)
            row["status"] = failed(e)
        upsert(cfg, "g1", [row])
        log.info("%s: %s", cid, row["status"])
