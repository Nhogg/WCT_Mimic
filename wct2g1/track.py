"""Stage 3: detect and track runners in each kept shot.

YOLO (person class) + BoT-SORT over every frame -> raw tracklets. Tracklets of the same
person split by occlusion are stitched, gaps interpolated, boxes Savitzky-Golay smoothed.
Each stitched track is scored on coverage, median box height and motion; the top
`max_tracks` that pass the hard rules are written to data/tracks/<shot_id>/<track_id>.npz
with shot-relative frame_idx, bbox_xyxy, conf and an `interpolated` mask.
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from scipy.signal import savgol_filter

from .config import Config
from .manifest import failed, ok, read_manifest, rejected, upsert, write_manifest

log = logging.getLogger(__name__)


@dataclass
class Track:
    frames: np.ndarray            # (N,) int, strictly increasing
    boxes: np.ndarray             # (N,4) xyxy
    conf: np.ndarray              # (N,)
    src_ids: list[int] = field(default_factory=list)
    interpolated: np.ndarray | None = None

    @property
    def start(self) -> int:
        return int(self.frames[0])

    @property
    def end(self) -> int:
        return int(self.frames[-1])


# ---------------------------------------------------------------- box helpers
def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0


def _h(b: np.ndarray) -> np.ndarray:
    return b[..., 3] - b[..., 1]


def _center(b: np.ndarray) -> np.ndarray:
    return np.stack([(b[..., 0] + b[..., 2]) / 2, (b[..., 1] + b[..., 3]) / 2], -1)


def can_join(a: Track, b: Track, fps: float, s: dict) -> float | None:
    """Cost of appending b after a, or None if they can't be the same person."""
    gap = b.start - a.end
    if gap <= 0 or gap > s["max_gap_s"] * fps:
        return None
    la, fb = a.boxes[-1], b.boxes[0]
    hr = _h(la) / max(_h(fb), 1e-6)
    if not (1 / s["max_height_ratio"] <= hr <= s["max_height_ratio"]):
        return None
    d = float(np.linalg.norm(_center(la) - _center(fb)) / max(_h(la), 1e-6))
    if iou(la, fb) >= s["iou_min"] or d <= s["max_center_dist_h"]:
        return d + gap / fps   # prefer closest in space and time
    return None


def stitch(tracks: list[Track], fps: float, s: dict) -> list[Track]:
    """Greedily join tracklets, cheapest pair first; each end/start used once."""
    tracks = sorted(tracks, key=lambda t: t.start)
    while True:
        best = None
        for i, a in enumerate(tracks):
            for j, b in enumerate(tracks):
                if i == j:
                    continue
                c = can_join(a, b, fps, s)
                if c is not None and (best is None or c < best[0]):
                    best = (c, i, j)
        if best is None:
            return tracks
        _, i, j = best
        a, b = tracks[i], tracks[j]
        merged = Track(np.r_[a.frames, b.frames], np.r_[a.boxes, b.boxes],
                       np.r_[a.conf, b.conf], a.src_ids + b.src_ids)
        tracks = [t for k, t in enumerate(tracks) if k not in (i, j)] + [merged]
        tracks.sort(key=lambda t: t.start)


def fill_and_smooth(t: Track, fps: float, s: dict) -> Track:
    """Linearly interpolate missing frames, then Savitzky-Golay smooth the boxes."""
    full = np.arange(t.start, t.end + 1)
    boxes = np.stack([np.interp(full, t.frames, t.boxes[:, k]) for k in range(4)], -1)
    conf = np.zeros(len(full))
    present = np.isin(full, t.frames)
    conf[present] = t.conf
    win = int(round(s["savgol_window_s"] * fps)) | 1      # odd
    if len(full) > win > s["savgol_polyorder"]:
        boxes = savgol_filter(boxes, win, s["savgol_polyorder"], axis=0)
    return Track(full, boxes, conf, t.src_ids, interpolated=~present)


def score_track(t: Track, n_shot_frames: int, frame_h: int, fps: float, s: dict) -> dict:
    present = ~t.interpolated if t.interpolated is not None else np.ones(len(t.frames), bool)
    hs = _h(t.boxes)
    med_h = float(np.median(hs[present])) / frame_h
    c = _center(t.boxes)
    moved_bh = float(np.sum(np.linalg.norm(np.diff(c, axis=0), axis=1) / np.maximum(hs[1:], 1)))
    dur = len(t.frames) / fps
    m = s["edge_margin_px"]
    trunc = float(np.mean((t.boxes[:, 1] <= m) | (t.boxes[:, 3] >= frame_h - m)))
    cov = float(present.sum()) / n_shot_frames
    motion = moved_bh / max(dur, 1e-6)
    rank = cov + s["w_height"] * med_h + s["w_motion"] * motion
    return dict(n_frames=len(t.frames), coverage=cov, median_box_h=med_h, motion=motion,
                truncated_frac=trunc, dur_s=dur, rank_score=rank)


def select_tracks(scored: list[tuple[Track, dict]], s: dict) -> list[tuple[Track, dict, str]]:
    """Apply hard rules, then keep the top `max_tracks` by rank_score."""
    out, passing = [], []
    for t, m in scored:
        if m["dur_s"] < s["min_dur_s"]:
            st = rejected("too_short")
        elif m["coverage"] < s["min_coverage"]:
            st = rejected("coverage")
        elif m["median_box_h"] < s["min_median_box_h"]:
            st = rejected("too_small")
        elif m["truncated_frac"] > s["max_truncated_frac"]:
            st = rejected("truncated")
        else:
            passing.append((t, m))
            continue
        out.append((t, m, st))
    passing.sort(key=lambda tm: -tm[1]["rank_score"])
    for k, (t, m) in enumerate(passing):
        out.append((t, m, ok() if k < s["max_tracks"] else rejected("not_top_ranked")))
    return out


# ---------------------------------------------------------------- detection
def run_tracker(cfg: Config, video: Path) -> tuple[list[Track], int, int, int]:
    """Returns raw tracklets, n_frames, frame_h, frame_w."""
    from ultralytics import YOLO

    c = cfg["track"]
    tracker_yaml = cfg.path_of("tracks") / "_botsort.yaml"
    tracker_yaml.parent.mkdir(parents=True, exist_ok=True)
    tracker_yaml.write_text(yaml.safe_dump(c["tracker"]))
    dev = c["device"]
    if dev == "auto":
        import torch
        dev = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(c["model"])
    per_id: dict[int, list] = {}
    n = h = w = 0
    for fi, r in enumerate(model.track(str(video), stream=True, persist=True, classes=[0],
                                       conf=c["conf"], imgsz=c["imgsz"], device=dev,
                                       tracker=str(tracker_yaml), verbose=False)):
        n = fi + 1
        h, w = r.orig_shape
        b = r.boxes
        if b is None or b.id is None:
            continue
        for tid, xyxy, cf in zip(b.id.int().tolist(), b.xyxy.cpu().numpy(), b.conf.cpu().numpy()):
            per_id.setdefault(tid, []).append((fi, xyxy, float(cf)))
    tracks = []
    for tid, rows in per_id.items():
        rows.sort(key=lambda r: r[0])
        tracks.append(Track(np.array([r[0] for r in rows]), np.array([r[1] for r in rows]),
                            np.array([r[2] for r in rows]), [tid]))
    return tracks, n, h, w


# ---------------------------------------------------------------- overlay
def write_overlay(cfg: Config, video: Path, out: Path, kept: list[tuple[str, Track]],
                  others: list[Track]) -> None:
    width = cfg["track"]["review"]["overlay_width"]
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sc = width / W
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".raw.mp4")
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, int(H * sc)))
    colors = [(0, 255, 0), (0, 200, 255)]
    lut = [(name, {int(f): (b, it) for f, b, it in zip(t.frames, t.boxes, t.interpolated)})
           for name, t in kept]
    olut = [{int(f): b for f, b in zip(t.frames, t.boxes)} for t in others]
    fi = 0
    while True:
        ok_, f = cap.read()
        if not ok_:
            break
        f = cv2.resize(f, (width, int(H * sc)))
        for d in olut:
            if fi in d:
                x1, y1, x2, y2 = (d[fi] * sc).astype(int)
                cv2.rectangle(f, (x1, y1), (x2, y2), (128, 128, 128), 1)
        for k, (name, d) in enumerate(lut):
            if fi in d:
                b, it = d[fi]
                x1, y1, x2, y2 = (b * sc).astype(int)
                col = colors[k % 2]
                cv2.rectangle(f, (x1, y1), (x2, y2), col, 1 if it else 2)
                cv2.putText(f, name, (x1, max(12, y1 - 4)), 0, 0.45, col, 1)
        cv2.putText(f, f"{fi}", (6, 18), 0, 0.5, (255, 255, 255), 1)
        vw.write(f)
        fi += 1
    vw.release()
    cap.release()
    import subprocess  # re-encode to H.264 so it plays in browsers
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(tmp), "-c:v", "libx264", "-crf", "23",
                    "-pix_fmt", "yuv420p", str(out)], check=True)
    tmp.unlink()


# ---------------------------------------------------------------- per shot
def process_shot(cfg: Config, shot: pd.Series, overlay: bool) -> list[dict]:
    c = cfg["track"]
    video = cfg.root / shot["path"]
    fps = float(shot["fps"])
    raw, n, h, w = run_tracker(cfg, video)
    stitched = stitch(raw, fps, c["stitch"])
    filled = [fill_and_smooth(t, fps, c["smooth"]) for t in stitched]
    scored = [(t, score_track(t, n, h, fps, c["select"])) for t in filled]
    decided = select_tracks(scored, c["select"])

    out_dir = cfg.path_of("tracks") / shot["shot_id"]
    rows, kept = [], []
    k = 0
    for t, m, st in sorted(decided, key=lambda x: -x[1]["rank_score"]):
        if m["dur_s"] < c["select"]["min_dur_s"]:
            continue   # don't bloat the manifest with fragments
        tid = f"{shot['shot_id']}_t{k}"
        k += 1
        row = dict(track_id=tid, shot_id=shot["shot_id"], video_id=shot["video_id"],
                   start_frame=t.start, end_frame=t.end + 1, fps=fps,
                   src_ids=",".join(map(str, t.src_ids)), path=None, status=st, **m)
        if st == "ok":
            p = out_dir / f"{tid}.npz"
            p.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(p, frame_idx=t.frames, bbox_xyxy=t.boxes.astype(np.float32),
                                conf=t.conf.astype(np.float32), interpolated=t.interpolated,
                                fps=fps, frame_hw=np.array([h, w]))
            row["path"] = str(p.relative_to(cfg.root))
            kept.append((tid.rsplit("_", 1)[1], t))
        rows.append(row)
    if not kept:
        rows.append(dict(track_id=f"{shot['shot_id']}_none", shot_id=shot["shot_id"],
                         video_id=shot["video_id"], status=rejected("no_runner_tracks")))
    if overlay:
        others = [t for t, _, st in decided if st != "ok"]
        ov = cfg.path_of("review") / "track_overlays" / f"{shot['shot_id']}.mp4"
        write_overlay(cfg, video, ov, kept, others)
        log.info("overlay: %s", ov)
    return rows


def run(cfg: Config, limit: int | None = None, ids: list[str] | None = None) -> None:
    shots = read_manifest(cfg, "shots")
    shots = shots[shots["status"] == "ok"]
    if ids:
        shots = shots[shots["shot_id"].isin(ids)]
    tr = read_manifest(cfg, "tracks")
    if not tr.empty:
        bad = set(tr.loc[tr["status"].str.startswith("failed:"), "shot_id"])
        done = set(tr["shot_id"]) - bad
        shots = shots[~shots["shot_id"].isin(done)]
    if limit:
        shots = shots.head(limit)
    rng = random.Random(cfg["seed"])
    n_ov = cfg["track"]["review"]["sample_n"]
    overlay_ids = set(ids or []) | set(rng.sample(list(shots["shot_id"]), min(n_ov, len(shots))))
    log.info("track: %d shots to process", len(shots))

    for _, shot in shots.iterrows():
        try:
            rows = process_shot(cfg, shot, shot["shot_id"] in overlay_ids)
        except Exception as e:  # noqa: BLE001
            log.exception("track failed for %s", shot["shot_id"])
            rows = [dict(track_id=f"{shot['shot_id']}_none", shot_id=shot["shot_id"],
                         video_id=shot["video_id"], status=failed(e))]
        cur = read_manifest(cfg, "tracks")
        if not cur.empty:
            write_manifest(cfg, "tracks", cur[cur["shot_id"] != shot["shot_id"]])
        upsert(cfg, "tracks", rows)
        n_ok = sum(r["status"] == "ok" for r in rows)
        log.info("%s: %d runner tracks", shot["shot_id"], n_ok)
