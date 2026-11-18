"""Stage 5: clean, retime and cut GVHMR motion into short real-time clips.

Per track: Y-up (GVHMR) -> Z-up (same rotation GMR uses), slow-mo retime by speed_factor,
resample to 50 fps (slerp rotations / lerp translation), zero-phase low-pass in 6D rotation
space, ground at z=0, foot-contact detection, foot-skate removal by root correction, cut at
low-velocity points into 2-10 s clips, reject implausible clips, flag likely obstacle use.

Output data/clean/<clip_id>.npz is AMASS-style SMPL-X (what GMR's smplx_to_robot reads):
root_orient (T,3), pose_body (T,63), trans (T,3), betas (10,), gender, mocap_frame_rate,
plus contact (T,2) [left, right], human_height, coord="z_up".
No height scaling here; GMR scales to the G1 in Stage 6.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from .config import Config
from .manifest import shard, filter_ids, failed, ok, read_manifest, rejected, upsert, write_manifest

log = logging.getLogger(__name__)

# GMR's GVHMR Y-up -> Z-up rotation (general_motion_retargeting/utils/smpl.py): +90 deg about X.
YUP_TO_ZUP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)

# SMPL-X body joint indices
PELVIS, L_ANKLE, R_ANKLE, L_FOOT, R_FOOT, L_WRIST, R_WRIST = 0, 7, 8, 10, 11, 20, 21


# ---------------------------------------------------------------- frames & time
def yup_to_zup(global_orient: np.ndarray, transl: np.ndarray, pelvis_rest: np.ndarray):
    """Rotate a SMPL-X root trajectory into the Z-up world.

    SMPL-X rotates the body about the rest pelvis j0, so world pelvis = transl + j0 and
    the new transl is R(transl + j0) - j0, not R transl.
    """
    rot = R.from_matrix(YUP_TO_ZUP)
    go = (rot * R.from_rotvec(global_orient)).as_rotvec()
    tr = (transl + pelvis_rest) @ YUP_TO_ZUP.T - pelvis_rest
    return go, tr


def timestamps(frame_idx: np.ndarray, src_fps: float, speed_factor: float | None) -> np.ndarray:
    """Real-world seconds for each sample. Slow-mo footage (speed_factor < 1) is compressed."""
    sf = 1.0 if speed_factor is None or not np.isfinite(speed_factor) else float(speed_factor)
    return (frame_idx - frame_idx[0]) / src_fps * sf


def resample(t: np.ndarray, rotvecs: np.ndarray, transl: np.ndarray, fps: float):
    """rotvecs (T,J,3), transl (T,3) at times t -> uniform grid at fps (slerp / lerp)."""
    n = int(np.floor(t[-1] * fps + 1e-6)) + 1
    grid = np.minimum(np.arange(n) / fps, t[-1])   # float rounding must not step past t[-1]
    out_r = np.empty((len(grid), rotvecs.shape[1], 3))
    for j in range(rotvecs.shape[1]):
        out_r[:, j] = Slerp(t, R.from_rotvec(rotvecs[:, j]))(grid).as_rotvec()
    out_t = np.stack([np.interp(grid, t, transl[:, k]) for k in range(3)], -1)
    return grid, out_r, out_t


# ---------------------------------------------------------------- smoothing
def rot_to_6d(rotvecs: np.ndarray) -> np.ndarray:
    m = R.from_rotvec(rotvecs.reshape(-1, 3)).as_matrix()
    return m[:, :, :2].reshape(*rotvecs.shape[:-1], 6)


def rot_from_6d(d6: np.ndarray) -> np.ndarray:
    a = d6.reshape(-1, 3, 2)
    x = a[:, :, 0] / np.linalg.norm(a[:, :, 0], axis=-1, keepdims=True)
    y = a[:, :, 1] - np.sum(x * a[:, :, 1], -1, keepdims=True) * x
    y /= np.linalg.norm(y, axis=-1, keepdims=True)
    z = np.cross(x, y)
    m = np.stack([x, y, z], -1)
    return R.from_matrix(m).as_rotvec().reshape(*d6.shape[:-1], 3)


def lowpass(x: np.ndarray, fps: float, cutoff: float, order: int) -> np.ndarray:
    """Zero-phase Butterworth along axis 0 (no-op if the clip is too short)."""
    if cutoff >= fps / 2:
        return x
    b, a = butter(order, cutoff / (fps / 2))
    padlen = 3 * max(len(a), len(b))
    if len(x) <= padlen:
        return x
    return filtfilt(b, a, x, axis=0)


def smooth_motion(rotvecs, transl, fps, cutoff, order):
    d6 = lowpass(rot_to_6d(rotvecs), fps, cutoff, order)
    return rot_from_6d(d6), lowpass(transl, fps, cutoff, order)


# ---------------------------------------------------------------- contact & skating
def _drop_short_runs(mask: np.ndarray, min_len: int) -> np.ndarray:
    out = mask.copy()
    n = len(mask)
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if j - i < min_len:
                out[i:j] = False
            i = j
        else:
            i += 1
    return out


def detect_contact(sole_z: np.ndarray, foot_pos: np.ndarray, fps: float,
                   height_m: float, speed_mps: float, min_frames: int) -> np.ndarray:
    """sole_z (T,2), foot_pos (T,2,3) -> contact (T,2) bool."""
    vel = np.zeros(sole_z.shape)
    vel[1:] = np.linalg.norm(np.diff(foot_pos, axis=0), axis=-1) * fps
    vel[0] = vel[1] if len(vel) > 1 else 0
    c = (sole_z < height_m) & (vel < speed_mps)
    return np.stack([_drop_short_runs(c[:, k], min_frames) for k in range(2)], -1)


def skate_correction(foot_pos: np.ndarray, sole_z: np.ndarray, contact: np.ndarray,
                     fps: float, z_cutoff: float) -> np.ndarray:
    """Root offset (T,3) that keeps planted feet still in xy and on the floor in z."""
    T = len(foot_pos)
    d_xy = np.zeros((T, 2))
    for t in range(1, T):
        planted = contact[t] & contact[t - 1]
        if planted.any():
            d_xy[t] = -np.mean(foot_pos[t, planted, :2] - foot_pos[t - 1, planted, :2], axis=0)
    corr_xy = np.cumsum(d_xy, axis=0)
    zc = np.full(T, np.nan)
    any_c = contact.any(1)
    for t in np.flatnonzero(any_c):
        zc[t] = -np.mean(sole_z[t, contact[t]])
    if any_c.any():
        valid = np.flatnonzero(any_c)
        zc = np.interp(np.arange(T), valid, zc[valid])
        zc = lowpass(zc, fps, z_cutoff, 2)
    else:
        zc = np.zeros(T)
    return np.c_[corr_xy, zc]


# ---------------------------------------------------------------- cutting & flags
def cut_points(speed: np.ndarray, fps: float, min_s: float, max_s: float) -> list[tuple[int, int]]:
    """Split [0,T) into segments of min_s..max_s, cutting at the slowest frame in range."""
    T = len(speed)
    lo, hi = int(round(min_s * fps)), int(round(max_s * fps))
    if T < lo:
        return []
    segs, a = [], 0
    while T - a > hi:
        # the cut must leave >= lo frames on both sides
        window = np.arange(a + lo, min(a + hi, T - lo) + 1)
        if len(window) == 0:
            break
        b = int(window[np.argmin(speed[window])])
        segs.append((a, b))
        a = b
    if T - a >= lo:
        segs.append((a, T))
    return segs


def rolling_floor(min_sole: np.ndarray, fps: float, window_s: float, pct: float) -> np.ndarray:
    """Slowly varying floor height (T,): rolling low percentile of the lowest sole, smoothed.
    Absorbs the vertical drift of monocular world trajectories (GVHMR on panning shots)."""
    from scipy.ndimage import percentile_filter, uniform_filter1d

    n = max(1, int(round(window_s * fps))) | 1
    return uniform_filter1d(percentile_filter(min_sole, pct, size=n, mode="nearest"), n,
                            mode="nearest")


def _runs_longer_than(mask: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros_like(mask)
    i, T = 0, len(mask)
    while i < T:
        if mask[i]:
            j = i
            while j < T and mask[j]:
                j += 1
            if j - i > n:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


def obstacle_mask(joints: np.ndarray, sole_above_floor: np.ndarray, pelvis_above_floor: np.ndarray,
                  standing_pelvis_z: float, fps: float, c: dict) -> np.ndarray:
    """Per-frame (T,) bool: frames that look like obstacle interaction, dilated by margin_s.
    sole_above_floor uses the drift-corrected floor; pelvis_above_floor the global one, so a
    real climb (which the rolling floor would flatten) still shows up."""
    from scipy.ndimage import binary_dilation

    hand_low = joints[:, [L_WRIST, R_WRIST], 2].min(1) < c["hand_height_m"]
    pelvis_high = pelvis_above_floor > standing_pelvis_z + c["pelvis_rise_m"]
    air = _runs_longer_than(sole_above_floor > c["air_height_m"], int(c["airborne_s"] * fps))
    m = hand_low | pelvis_high | air
    k = int(round(c["margin_s"] * fps))
    return binary_dilation(m, iterations=k) if k > 0 and m.any() else m


def runs(mask: np.ndarray) -> list[tuple[int, int, bool]]:
    """[(start, end, value)] for maximal constant runs of a bool array."""
    out, a = [], 0
    for i in range(1, len(mask) + 1):
        if i == len(mask) or mask[i] != mask[a]:
            out.append((a, i, bool(mask[a])))
            a = i
    return out


def export_betas(betas: np.ndarray, used: int, n_out: int) -> np.ndarray:
    """First `used` shape coefficients, zero-padded to `n_out` (same body shape)."""
    out = np.zeros(n_out, np.float32)
    k = min(used, n_out, len(betas))
    out[:k] = betas[:k]
    return out


# ---------------------------------------------------------------- body model
class Body:
    def __init__(self, cfg: Config, betas: np.ndarray):
        import smplx
        import torch

        c = cfg["clean"]
        self.torch = torch
        self.model = smplx.create(str(cfg.path_of("body_models")), model_type="smplx",
                                  gender=c["smplx_gender"], use_pca=False,
                                  num_betas=c["num_betas"], ext="npz")
        self.betas = torch.from_numpy(np.asarray(betas[: c["num_betas"]], np.float32))[None]
        rest = self._fk(np.zeros((1, 3)), np.zeros((1, 63)), np.zeros((1, 3)), verts=True)
        j, v = rest
        self.pelvis_rest = j[0, PELVIS].copy()
        # SMPL-X rest pose stands along +Y. Sole offsets: each foot joint's height above the
        # lowest rest vertex, so sole height = joint height - offset in the Z-up world.
        floor = v[0, :, 1].min()
        self.sole_off = {k: j[0, k, 1] - floor for k in (L_ANKLE, R_ANKLE, L_FOOT, R_FOOT)}
        self.standing_pelvis = j[0, PELVIS, 1] - floor
        self.height = float(v[0, :, 1].max() - floor)

    def _fk(self, go, body, tr, verts=False):
        t = self.torch
        n = len(go)
        f = lambda a: t.from_numpy(np.ascontiguousarray(a)).float()  # noqa: E731  (slices may be reversed views)
        with t.no_grad():
            z = lambda k: t.zeros(n, k)  # noqa: E731  (face/hands neutral, sized to the batch)
            m = self.model
            out = m(global_orient=f(go), body_pose=f(body), transl=f(tr),
                    jaw_pose=z(3), leye_pose=z(3), reye_pose=z(3),
                    left_hand_pose=z(m.left_hand_pose.shape[1]),
                    right_hand_pose=z(m.right_hand_pose.shape[1]),
                    expression=z(m.expression.shape[1]),
                    betas=self.betas.expand(n, -1), return_verts=verts)
        j = out.joints.numpy()
        return (j, out.vertices.numpy()) if verts else j

    def joints(self, go, body, tr):
        out = []
        for s in range(0, len(go), 512):
            out.append(self._fk(go[s:s + 512], body[s:s + 512], tr[s:s + 512]))
        return np.concatenate(out)

    def soles(self, j: np.ndarray) -> np.ndarray:
        """Per-frame sole height (T,2) for left/right foot from ankle + foot joints (Z-up)."""
        o = self.sole_off
        left = np.minimum(j[:, L_ANKLE, 2] - o[L_ANKLE], j[:, L_FOOT, 2] - o[L_FOOT])
        right = np.minimum(j[:, R_ANKLE, 2] - o[R_ANKLE], j[:, R_FOOT, 2] - o[R_FOOT])
        return np.stack([left, right], -1)


# ---------------------------------------------------------------- per track
def process_track(cfg: Config, row: pd.Series, speed_factor: float | None) -> list[dict]:
    c = cfg["clean"]
    fps = float(c["fps"])
    d = np.load(cfg.root / row["path"])
    betas = d["betas"]
    body = Body(cfg, betas)

    # raw-rate root-jump check (before any smoothing hides it)
    jumps = np.linalg.norm(np.diff(d["transl"], axis=0), axis=-1)

    go, tr = yup_to_zup(d["global_orient"], d["transl"], body.pelvis_rest)
    rot = np.concatenate([go[:, None], d["body_pose"].reshape(-1, 21, 3)], 1)
    t = timestamps(d["frame_idx"].astype(float), float(d["src_fps"]), speed_factor)
    grid, rot, tr = resample(t, rot, tr, fps)
    rot, tr = smooth_motion(rot, tr, fps, c["smooth"]["cutoff_hz"], c["smooth"]["order"])
    go, bp = rot[:, 0], rot[:, 1:].reshape(-1, 63)

    # trim the track ends, where reconstruction is least reliable
    k = int(round(c["cut"]["trim_edges_s"] * fps))
    if k > 0 and len(grid) > 2 * k:
        grid, go, bp, tr = grid[k:-k], go[k:-k], bp[k:-k], tr[k:-k]

    # ground: global floor (typical lowest sole at z = 0), then optionally a rolling floor
    g = c["ground"]
    j = body.joints(go, bp, tr)
    soles = body.soles(j)
    tr[:, 2] -= np.percentile(soles.min(1), g["floor_percentile"])
    j = body.joints(go, bp, tr)
    pelvis_global = j[:, PELVIS, 2].copy()           # for obstacle detection (keeps real climbs)
    if g["mode"] == "rolling":
        tr[:, 2] -= rolling_floor(body.soles(j).min(1), fps, g["window_s"], g["floor_percentile"])
        j = body.joints(go, bp, tr)
    elif g["mode"] != "global":
        raise ValueError(f"clean.ground.mode {g['mode']!r}")
    soles = body.soles(j)
    feet = j[:, [L_FOOT, R_FOOT]]
    cc = c["contact"]
    contact = detect_contact(soles, feet, fps, cc["height_m"], cc["speed_mps"], cc["min_frames"])
    tr = tr + skate_correction(feet, soles, contact, fps, cc["z_fix_cutoff_hz"])
    j = body.joints(go, bp, tr)
    soles = body.soles(j)

    root_v = np.r_[0.0, np.linalg.norm(np.diff(tr, axis=0), axis=-1) * fps]
    speed_s = lowpass(root_v, fps, 1.0 / max(c["cut"]["speed_smooth_s"], 1e-3), 2)
    obst = obstacle_mask(j, soles.min(1), pelvis_global, body.standing_pelvis, fps, c["obstacle"])
    if not c["obstacle"]["split"]:
        obst[:] = obst.any()
    segs = [(a + s0, a + s1, flag) for a, b, flag in runs(obst)
            for s0, s1 in cut_points(speed_s[a:b], fps, c["cut"]["min_s"], c["cut"]["max_s"])]
    if not segs:
        return [dict(clip_id=f"{row['track_id']}_c0", track_id=row["track_id"],
                     status=rejected("too_short"))]

    # map raw jump indices onto the resampled timeline
    sf = 1.0 if speed_factor is None or not np.isfinite(speed_factor) else float(speed_factor)
    jump_t = t[1:]
    rj = c["reject"]
    rows = []
    for k, (a, b, flag) in enumerate(segs):
        cid = f"{row['track_id']}_c{k}"
        seg = slice(a, b)
        t0, t1 = grid[a], grid[b - 1]
        in_seg = (jump_t >= t0) & (jump_t <= t1)
        max_jump = float(jumps[in_seg].max()) if in_seg.any() else 0.0
        below = float(np.mean(soles[seg].min(1) < -rj["below_floor_m"]))
        vmax = float(root_v[seg].max())
        base = dict(clip_id=cid, track_id=row["track_id"], shot_id=row["shot_id"],
                    start_s=float(t0), dur_s=float((b - a) / fps),
                    mean_speed=float(root_v[seg].mean()), max_speed=vmax, max_root_jump=max_jump,
                    below_floor_frac=below, contact_frac=float(contact[seg].any(1).mean()),
                    speed_factor=sf, human_height=body.height, path=None)
        if vmax > rj["max_root_speed_mps"]:
            base["status"] = rejected("root_speed")
        elif max_jump > rj["max_root_jump_m"]:
            base["status"] = rejected("root_jump")
        elif below > rj["max_below_floor_frac"]:
            base["status"] = rejected("below_floor")
        else:
            base["status"] = ok()
        base["has_obstacle_contact"] = flag
        if base["status"] == "ok":
            out = cfg.path_of("clean") / f"{cid}.npz"
            out.parent.mkdir(parents=True, exist_ok=True)
            trs = tr[seg].copy()
            trs[:, :2] -= trs[0, :2]          # clip starts at the xy origin
            np.savez_compressed(
                out, root_orient=go[seg].astype(np.float32), pose_body=bp[seg].astype(np.float32),
                trans=trs.astype(np.float32), betas=export_betas(betas, c["num_betas"], c["export_num_betas"]),
                gender=np.array(c["smplx_gender"]), mocap_frame_rate=np.array(fps),
                contact=contact[seg], human_height=np.array(body.height),
                coord=np.array("z_up"), source_track=np.array(row["track_id"]),
                config_hash=np.array(cfg.hash))
            base["path"] = str(out.relative_to(cfg.root))
        rows.append(base)
    return rows


def run(cfg: Config, limit: int | None = None, ids: list[str] | None = None) -> None:
    smpl = read_manifest(cfg, "smpl")
    smpl = smpl[smpl["status"] == "ok"]
    if ids:
        smpl = filter_ids(smpl, ids)
    cl = read_manifest(cfg, "clean")
    if not cl.empty:
        bad = set(cl.loc[cl["status"].str.startswith("failed:"), "track_id"])
        smpl = smpl[~smpl["track_id"].isin(set(cl["track_id"]) - bad)]
    smpl = shard(smpl, "track_id")
    if limit:
        smpl = smpl.head(limit)
    shots = read_manifest(cfg, "shots").set_index("shot_id")
    log.info("clean: %d tracks to process", len(smpl))
    for _, row in smpl.iterrows():
        sf = shots.loc[row["shot_id"], "speed_factor"] if "speed_factor" in shots.columns else None
        try:
            rows = process_track(cfg, row, None if pd.isna(sf) else float(sf))
        except Exception as e:  # noqa: BLE001
            log.exception("clean failed for %s", row["track_id"])
            rows = [dict(clip_id=f"{row['track_id']}_c0", track_id=row["track_id"],
                         status=failed(e))]
        upsert(cfg, "clean", rows, replace=("track_id", row["track_id"]))
        log.info("%s: %s", row["track_id"], [r["status"] for r in rows])
