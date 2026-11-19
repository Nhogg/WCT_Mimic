"""Milestone 5: obstacle boxes implied by the reference motion (`wct-data` env).

GVHMR reconstructs the athlete, not the course. Where a foot stays planted above the
floor (a box top, a platform), the motion only makes physical sense with something
underneath it. For every retargeted clip this stage finds such stances in the G1
reference (mjlab npz: body_pos_w / body_lin_vel_w), groups them into boxes standing on
the floor, checks that no part of the reference passes through a box, and writes
data/g1/obstacles/<clip_id>.json. The multi-clip tracking task places these boxes in
each env's world (wct2g1.multi_motion).

Hands are not used: a still hand at hip height is indistinguishable from a hand on a box.
"""
from __future__ import annotations

import json
import logging

import numpy as np

from .config import Config
from .manifest import failed, filter_ids, ok, read_manifest, shard, upsert

log = logging.getLogger(__name__)

# Body indices in the mjlab G1 npz (body order of the G1 MJCF).
FEET = (6, 12)          # left/right ankle_roll_link
LEG = {6: (4, 5, 6), 12: (10, 11, 12)}   # knee, ankle_pitch, ankle_roll of that foot


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index ranges where mask is True."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.flatnonzero(np.diff(m.astype(np.int8)))
    return list(zip(d[::2], d[1::2]))


def ground_level(body_pos: np.ndarray, pct: float) -> float:
    """Ankle-link height when standing on the floor (floor is z=0 in the sim)."""
    return float(np.percentile(body_pos[:, FEET, 2].min(axis=1), pct))


def elevated_stances(body_pos: np.ndarray, body_vel: np.ndarray, fps: float, c: dict
                     ) -> list[dict]:
    """Planted-foot intervals whose sole is at least c['min_height_m'] above the floor."""
    g = ground_level(body_pos, c["ground_percentile"])
    out = []
    for b in FEET:
        speed = np.linalg.norm(body_vel[:, b], axis=-1)
        for a, e in runs(speed < c["max_foot_speed"]):
            if (e - a) / fps < c["min_stance_s"]:
                continue
            h = float(np.median(body_pos[a:e, b, 2])) - g
            if h >= c["min_height_m"]:
                out.append(dict(body=int(b), start=int(a), end=int(e), height=h,
                                xy=body_pos[a:e, b, :2].copy()))
    return out


def cluster(stances: list[dict], c: dict) -> list[list[dict]]:
    """Group stances on the same surface: similar height and nearby footprints."""
    groups: list[list[dict]] = []
    for s in sorted(stances, key=lambda s: s["height"]):
        for grp in groups:
            pts = np.concatenate([q["xy"] for q in grp])
            near = np.min(np.linalg.norm(pts[:, None] - s["xy"][None], axis=-1))
            if (abs(np.median([q["height"] for q in grp]) - s["height"]) <= c["merge_dz_m"]
                    and near <= c["merge_dxy_m"]):
                grp.append(s)
                break
        else:
            groups.append([s])
    return groups


def box_from_group(grp: list[dict], margin: float, min_size: float) -> dict:
    """Axis-aligned box on the floor whose top is the stance height."""
    pts = np.concatenate([q["xy"] for q in grp])
    top = float(np.median([q["height"] for q in grp]))
    lo, hi = pts.min(0) - margin, pts.max(0) + margin
    half = np.maximum((hi - lo) / 2, min_size / 2)
    ctr = (lo + hi) / 2
    return dict(center=[float(ctr[0]), float(ctr[1]), top / 2],
                half_size=[float(half[0]), float(half[1]), top / 2], top=top,
                n_stances=len(grp), frames=sum(q["end"] - q["start"] for q in grp))


def penetration(box: dict, body_pos: np.ndarray, ground: float, tol: float,
                ignore: np.ndarray | None = None) -> float:
    """Fraction of frames in which some body of the reference is inside the box
    (body origins, heights relative to the floor, `tol` slack below the top).
    `ignore` is a (frames, bodies) mask of points not checked."""
    cx, cy, _ = box["center"]
    hx, hy, _ = box["half_size"]
    p = body_pos
    inside = ((np.abs(p[..., 0] - cx) < hx) & (np.abs(p[..., 1] - cy) < hy)
              & (p[..., 2] - ground < box["top"] - tol))
    if ignore is not None:
        inside &= ~ignore
    return float(inside.any(axis=1).mean())


def stepping_mask(grp: list[dict], shape: tuple[int, int], fps: float, window_s: float
                  ) -> np.ndarray:
    """The stance leg just before and after each stance: stepping onto or off an edge
    clips its corner, and the reconstruction is a few cm off anyway."""
    m = np.zeros(shape, bool)
    w = int(round(window_s * fps))
    for q in grp:
        a, e = max(q["start"] - w, 0), min(q["end"] + w, shape[0])
        m[a:e, list(LEG[q["body"]])] = True
    return m


def infer_boxes(body_pos: np.ndarray, body_vel: np.ndarray, fps: float, c: dict
                ) -> tuple[list[dict], list[dict]]:
    """Returns (kept boxes, dropped boxes with a reason)."""
    g = ground_level(body_pos, c["ground_percentile"])
    kept, dropped = [], []
    for grp in cluster(elevated_stances(body_pos, body_vel, fps, c), c):
        for margin in c["margins_m"]:            # try a generous footprint first
            box = box_from_group(grp, margin, c["min_size_m"])
            ign = stepping_mask(grp, body_pos.shape[:2], fps, c["step_window_s"])
            pen = penetration(box, body_pos, g, c["penetration_tol_m"], ign)
            if pen <= c["max_penetration_frac"]:
                box["penetration"] = pen
                kept.append(box)
                break
        else:
            box["penetration"], box["reason"] = pen, "reference passes through the box"
            dropped.append(box)
    return kept, dropped


def run(cfg: Config, limit: int | None = None, ids: list[str] | None = None) -> None:
    c = cfg["obstacles"]
    clips = read_manifest(cfg, "g1")
    clips = clips[clips["status"] == "ok"]
    if ids:
        clips = filter_ids(clips, ids)
    clips = shard(clips, "clip_id")
    if limit:
        clips = clips.head(limit)
    out_dir = cfg.path_of("g1") / "obstacles"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, r in clips.iterrows():
        cid = r["clip_id"]
        row = dict(clip_id=cid)
        try:
            npz = cfg.path_of("g1") / "npz" / f"{cid}.npz"
            d = np.load(npz)
            kept, dropped = infer_boxes(d["body_pos_w"], d["body_lin_vel_w"],
                                        float(d["fps"][0]), c)
            path = out_dir / f"{cid}.json"
            path.write_text(json.dumps(dict(clip_id=cid, boxes=kept, dropped=dropped),
                                       indent=1))
            row.update(path=str(path.relative_to(cfg.root)), n_boxes=len(kept),
                       n_dropped=len(dropped),
                       max_top_m=max([b["top"] for b in kept], default=0.0), status=ok())
        except Exception as e:  # noqa: BLE001
            log.exception("obstacles failed for %s", cid)
            row.update(status=failed(e))
        rows.append(row)
    upsert(cfg, "obstacles", rows)
    n = sum(r.get("n_boxes", 0) > 0 for r in rows)
    log.info("obstacles: %d clips, %d with boxes", len(rows), n)
