"""Stage 2: split videos into camera shots, keep wide/medium chase shots.

Per video: PySceneDetect AdaptiveDetector -> shots; open_clip zero-shot scores on a few
sampled frames per shot; hard duration rules; slow-mo flag (CLIP + optical flow relative
to neighbouring shots); kept shots cut to data/shots/<video_id>/<shot_idx>.mp4.

Human input lives in review/slowmo.csv (speed_factor column). Values already filled in
there are preserved across reruns and copied into the manifest.
"""
from __future__ import annotations

import html
import logging
import random
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .config import Config
from .manifest import failed, ok, read_manifest, rejected, upsert, write_manifest

log = logging.getLogger(__name__)


# ---------------------------------------------------------------- shot detection
def detect_shots(cfg: Config, video: Path) -> list[tuple[int, int]]:
    """Return [(start_frame, end_frame_exclusive), ...]."""
    from scenedetect import AdaptiveDetector, detect

    d = cfg["segment"]["detector"]
    det = AdaptiveDetector(
        adaptive_threshold=d["adaptive_threshold"],
        min_scene_len=d["min_scene_len_frames"],
        window_width=d["window_width"],
        min_content_val=d["min_content_val"],
    )
    scenes = detect(str(video), det)
    if not scenes:  # no cuts found: one shot covering the whole video
        n = int(cv2.VideoCapture(str(video)).get(cv2.CAP_PROP_FRAME_COUNT))
        return [(0, n)]
    return [(s.get_frames(), e.get_frames()) for s, e in scenes]


# ---------------------------------------------------------------- frame access
def read_frame(cap: cv2.VideoCapture, idx: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok_, f = cap.read()
    return f if ok_ else None


def sample_indices(start: int, end: int, n: int) -> list[int]:
    """n frames evenly inside [start, end), avoiding the cut boundaries."""
    return [int(start + (end - start) * (i + 1) / (n + 1)) for i in range(n)]


# ---------------------------------------------------------------- CLIP
class ClipScorer:
    def __init__(self, cfg: Config):
        import open_clip
        import torch

        c = cfg["segment"]["clip"]
        dev = c["device"]
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if dev == "auto" else dev
        self.torch = torch
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            c["model"], pretrained=c["pretrained"], device=self.device)
        self.model.eval()
        tok = open_clip.get_tokenizer(c["model"])
        self.n_keep = len(c["keep_prompts"])
        self.class_text = self._encode_text(tok, c["keep_prompts"] + c["drop_prompts"])
        self.slowmo_text = self._encode_text(tok, cfg["segment"]["slowmo"]["prompts"])
        log.info("CLIP %s/%s on %s", c["model"], c["pretrained"], self.device)

    def _encode_text(self, tok, prompts):
        with self.torch.no_grad():
            t = self.model.encode_text(tok(prompts).to(self.device))
        return t / t.norm(dim=-1, keepdim=True)

    def score(self, frames_bgr: list[np.ndarray]) -> tuple[float, float]:
        """Returns (keep_prob, slowmo_prob), each averaged over the frames."""
        from PIL import Image

        imgs = [self.preprocess(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
                for f in frames_bgr]
        with self.torch.no_grad():
            im = self.model.encode_image(self.torch.stack(imgs).to(self.device))
            im = im / im.norm(dim=-1, keepdim=True)
            scale = self.model.logit_scale.exp()
            p_cls = (scale * im @ self.class_text.T).softmax(-1)
            p_slo = (scale * im @ self.slowmo_text.T).softmax(-1)
        keep = p_cls[:, : self.n_keep].sum(-1).mean().item()
        slow = p_slo[:, 0].mean().item()
        return keep, slow


# ---------------------------------------------------------------- optical flow
def shot_flow(cap: cv2.VideoCapture, start: int, end: int, n_pairs: int, width: int) -> float:
    """Mean Farneback flow magnitude per frame (pixels at `width`), over sampled pairs."""
    mags = []
    for i in sample_indices(start, end - 1, n_pairs):
        a = read_frame(cap, i)
        ok_, b = cap.read()
        if a is None or not ok_:
            continue
        h = int(a.shape[0] * width / a.shape[1])
        ga = cv2.cvtColor(cv2.resize(a, (width, h)), cv2.COLOR_BGR2GRAY)
        gb = cv2.cvtColor(cv2.resize(b, (width, h)), cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(ga, gb, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mags.append(float(np.linalg.norm(flow, axis=-1).mean()))
    return float(np.mean(mags)) if mags else float("nan")


def neighbour_flow_ratio(flows: np.ndarray, k: int) -> np.ndarray:
    """Each shot's flow divided by the median flow of up to k shots on each side."""
    out = np.full(len(flows), np.nan)
    for i in range(len(flows)):
        nb = np.r_[flows[max(0, i - k): i], flows[i + 1: i + 1 + k]]
        nb = nb[np.isfinite(nb)]
        if len(nb) and np.median(nb) > 0:
            out[i] = flows[i] / np.median(nb)
    return out


# ---------------------------------------------------------------- cutting
def keyframe_set(video: Path) -> np.ndarray:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-skip_frame", "nokey",
         "-show_entries", "frame=pts_time", "-of", "csv=p=0", str(video)],
        capture_output=True, text=True, check=True).stdout
    times = []
    for line in out.splitlines():
        field = line.split(",")[0].strip()   # csv lines can carry trailing empty fields
        if field and field != "N/A":
            times.append(float(field))
    return np.array(sorted(times))


def cut_shot(cfg: Config, video: Path, out: Path, start: int, end: int, fps: float,
             keyframes: np.ndarray) -> str:
    """Cut frames [start, end). Stream copy if start is on a keyframe, else re-encode."""
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = start / fps
    on_key = len(keyframes) and np.min(np.abs(keyframes - t0)) < 0.5 / fps
    n = end - start
    if on_key:
        cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{t0:.6f}", "-i", str(video),
               "-frames:v", str(n), "-an", "-c:v", "copy", str(out)]
        mode = "copy"
    else:
        c = cfg["segment"]["cut"]
        cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{t0:.6f}", "-i", str(video),
               "-frames:v", str(n), "-an", "-c:v", c["vcodec"], "-crf", str(c["crf"]),
               "-preset", c["preset"], "-pix_fmt", "yuv420p", str(out)]
        mode = "reencode"
    tmp = out.with_suffix(".tmp.mp4")
    cmd[-1] = str(tmp)
    subprocess.run(cmd, check=True)
    tmp.replace(out)
    return mode


# ---------------------------------------------------------------- per video
def process_video(cfg: Config, clip: ClipScorer, row: pd.Series) -> list[dict]:
    s = cfg["segment"]
    video = cfg.root / row["path"]
    vid = row["video_id"]
    fps = float(row["fps"])
    shots = detect_shots(cfg, video)
    log.info("%s: %d shots", vid, len(shots))

    cap = cv2.VideoCapture(str(video))
    recs = []
    for idx, (a, b) in enumerate(shots):
        dur = (b - a) / fps
        frames = [f for f in (read_frame(cap, i) for i in
                              sample_indices(a, b, s["clip"]["samples_per_shot"])) if f is not None]
        keep_p, slow_p = clip.score(frames) if frames else (float("nan"), float("nan"))
        flow = shot_flow(cap, a, b, s["slowmo"]["flow_pairs"], s["slowmo"]["flow_width"]) \
            if b - a >= 2 else float("nan")
        recs.append(dict(shot_id=f"{vid}_{idx:04d}", video_id=vid, shot_idx=idx,
                         start_frame=a, end_frame=b, fps=fps, dur_s=dur,
                         clip_score=keep_p, slowmo_prob=slow_p, flow=flow))
    cap.release()

    ratio = neighbour_flow_ratio(np.array([r["flow"] for r in recs]), s["slowmo"]["neighbours"])
    keyframes = keyframe_set(video)
    for r, fr in zip(recs, ratio):
        r["flow_ratio"] = fr
        r["is_slowmo"] = bool(np.isfinite(fr) and fr <= s["slowmo"]["flow_ratio_max"]
                              and r["slowmo_prob"] >= s["slowmo"]["prob_min"])
        r["needs_review"] = r["dur_s"] > s["max_shot_s"]
        r["path"] = None
        if r["dur_s"] < s["min_shot_s"]:
            r["status"] = rejected("too_short")
        elif not r["clip_score"] >= s["clip"]["keep_threshold"]:
            r["status"] = rejected("clip_score")
        else:
            out = cfg.path_of("shots") / vid / f"{r['shot_idx']:04d}.mp4"
            try:
                r["cut_mode"] = cut_shot(cfg, video, out, r["start_frame"], r["end_frame"],
                                         fps, keyframes)
                r["path"] = str(out.relative_to(cfg.root))
                r["status"] = ok()
            except Exception as e:  # noqa: BLE001
                r["status"] = failed(f"cut: {e}")
    return recs


# ---------------------------------------------------------------- review files
def sync_slowmo_csv(cfg: Config, df: pd.DataFrame) -> pd.DataFrame:
    """Add newly flagged slow-mo shots to review/slowmo.csv, keep human-entered
    speed factors, and copy them into the manifest's speed_factor column."""
    p = cfg.path_of("review") / "slowmo.csv"
    old = pd.read_csv(p, dtype={"shot_id": str}) if p.exists() else \
        pd.DataFrame(columns=["shot_id", "speed_factor"])
    if "is_slowmo" not in df.columns:   # e.g. only failure rows so far
        df["is_slowmo"] = False
    cols = ["shot_id", "video_id", "start_frame", "end_frame", "dur_s",
            "slowmo_prob", "flow_ratio", "path"]
    flagged = df.loc[df["is_slowmo"].fillna(False).astype(bool) & (df["status"] == "ok")]
    flagged = flagged.reindex(columns=cols)
    merged = flagged.merge(old[["shot_id", "speed_factor"]], on="shot_id", how="outer")
    # keep human rows even if no longer auto-flagged
    p.parent.mkdir(parents=True, exist_ok=True)
    merged.sort_values("shot_id").to_csv(p, index=False)
    sf = merged.set_index("shot_id")["speed_factor"]
    df["speed_factor"] = df["shot_id"].map(sf).astype(float)
    return df


def write_contact_sheet(cfg: Config, df: pd.DataFrame) -> Path:
    r = cfg["segment"]["review"]
    rng = random.Random(cfg["seed"])
    kept = df[df["status"] == "ok"]
    rej = df[df["status"].str.startswith("rejected:clip")]
    sample_k = kept.iloc[sorted(rng.sample(range(len(kept)), min(r["sample_n"], len(kept))))]
    sample_r = rej.iloc[sorted(rng.sample(range(len(rej)), min(r["reject_sample_n"], len(rej))))]

    review = cfg.path_of("review")
    thumbs = review / "thumbs"
    thumbs.mkdir(parents=True, exist_ok=True)
    raw = read_manifest(cfg, "raw").set_index("video_id")

    def cells(rows: pd.DataFrame) -> str:
        out = []
        for vid, g in rows.groupby("video_id"):
            cap = cv2.VideoCapture(str(cfg.root / raw.loc[vid, "path"]))
            for _, s in g.iterrows():
                imgs = []
                for j, fi in enumerate(sample_indices(s.start_frame, s.end_frame, 3)):
                    tp = thumbs / f"{s.shot_id}_{j}.jpg"
                    if not tp.exists():
                        f = read_frame(cap, fi)
                        if f is None:
                            continue
                        h = int(f.shape[0] * r["thumb_width"] / f.shape[1])
                        cv2.imwrite(str(tp), cv2.resize(f, (r["thumb_width"], h)))
                    imgs.append(f'<img src="thumbs/{tp.name}">')
                tag = " slow-mo" if s.is_slowmo else ""
                tag += " LONG" if s.needs_review else ""
                out.append(
                    f'<div class="c"><div class="i">{"".join(imgs)}</div>'
                    f'<div class="t"><b>{html.escape(s.shot_id)}</b> {s.dur_s:.1f}s '
                    f'score {s.clip_score:.2f}{tag}<br>{html.escape(s.status)}</div></div>')
            cap.release()
        return "\n".join(out)

    p = review / "shots_sample.html"
    p.write_text(f"""<!doctype html><meta charset="utf-8"><title>Shot sample</title>
<style>body{{font:13px sans-serif;margin:16px}} .c{{margin:0 0 14px}}
.i img{{width:{r['thumb_width']}px;margin-right:4px}} h2{{margin-top:28px}}</style>
<h1>Kept shots: random sample of {len(sample_k)} / {len(kept)}</h1>
<p>Mark each as usable action footage or not; aim for 80%+ precision.</p>
{cells(sample_k)}
<h2>Rejected by CLIP: random sample of {len(sample_r)} / {len(rej)}</h2>
{cells(sample_r)}
""")
    return p


# ---------------------------------------------------------------- entry point
def run(cfg: Config, limit: int | None = None, ids: list[str] | None = None) -> None:
    raw = read_manifest(cfg, "raw")
    raw = raw[raw["status"] == "ok"]
    if ids:
        raw = raw[raw["video_id"].isin(ids)]
    shots = read_manifest(cfg, "shots")
    if not shots.empty:
        done_vids = set(shots.loc[~shots["status"].str.startswith("failed:"), "video_id"])
        failed_vids = set(shots.loc[shots["status"].str.startswith("failed:"), "video_id"])
        todo = raw[~raw["video_id"].isin(done_vids - failed_vids)]
    else:
        todo = raw
    if limit:
        todo = todo.head(limit)
    log.info("segment: %d videos to process", len(todo))

    clip = ClipScorer(cfg) if len(todo) else None
    for _, row in todo.iterrows():
        try:
            recs = process_video(cfg, clip, row)
        except Exception as e:  # noqa: BLE001
            log.exception("segment failed for %s", row["video_id"])
            recs = [dict(shot_id=f"{row['video_id']}_video", video_id=row["video_id"],
                         status=failed(e))]
        # replace any earlier rows for this video (e.g. a previous failure)
        cur = read_manifest(cfg, "shots")
        if not cur.empty:
            write_manifest(cfg, "shots", cur[cur["video_id"] != row["video_id"]])
        upsert(cfg, "shots", recs)
        n_ok = sum(r["status"] == "ok" for r in recs)
        log.info("%s: kept %d / %d shots", row["video_id"], n_ok, len(recs))

    df = read_manifest(cfg, "shots")
    if df.empty:
        return
    df = sync_slowmo_csv(cfg, df)
    write_manifest(cfg, "shots", df)
    if (df["status"] == "ok").any():
        log.info("contact sheet: %s", write_contact_sheet(cfg, df))
