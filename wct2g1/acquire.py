"""Stage 1: download source videos + metadata with yt-dlp.

Restartable: yt-dlp's download archive skips videos already fetched. The manifest
is rebuilt from what is actually on disk after every run, so it always matches
the files. Archive entries whose media file has gone missing are pruned before
downloading so they are fetched again.
"""
from __future__ import annotations

import json
import logging
import subprocess
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import pandas as pd

from .config import Config
from .manifest import failed, ok, write_manifest

log = logging.getLogger(__name__)

MEDIA_EXTS = {".mp4", ".mkv", ".webm", ".mov"}
# yt-dlp exit code when --max-downloads is reached; not an error.
YTDLP_MAX_DOWNLOADS_EXIT = 101


def _media_for(video_dir: Path, vid: str) -> Path | None:
    for p in sorted(video_dir.glob(f"{vid}.*")):
        if p.suffix in MEDIA_EXTS and not p.name.endswith(".part"):
            return p
    return None


def _ffprobe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate:format=duration",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    d = json.loads(out.stdout)
    s = d["streams"][0]
    rate = s.get("avg_frame_rate") or s.get("r_frame_rate") or "0/1"
    if rate in ("0/0", "0/1"):
        rate = s.get("r_frame_rate", "0/1")
    return {
        "width": int(s["width"]),
        "height": int(s["height"]),
        "fps": float(Fraction(rate)) if rate != "0/0" else float("nan"),
        "duration_s": float(d.get("format", {}).get("duration", "nan")),
    }


def _archive_ids(archive: Path) -> list[tuple[str, str]]:
    if not archive.exists():
        return []
    out = []
    for line in archive.read_text().splitlines():
        parts = line.split()
        if len(parts) == 2:
            out.append((parts[0], parts[1]))
    return out


def prune_archive(cfg: Config) -> list[str]:
    """Drop archive entries with no media on disk so yt-dlp re-fetches them."""
    raw = cfg.path_of("raw")
    archive = raw / cfg["acquire"]["archive_file"]
    entries = _archive_ids(archive)
    keep, dropped = [], []
    for extractor, vid in entries:
        if _media_for(raw / vid, vid) is None:
            dropped.append(vid)
        else:
            keep.append(f"{extractor} {vid}")
    if dropped:
        log.warning("archive lists %d videos missing on disk; will re-download: %s",
                    len(dropped), dropped[:10])
        archive.write_text("".join(f"{l}\n" for l in keep))
    return dropped


def ytdlp_command(cfg: Config, url: str, limit: int | None = None) -> list[str]:
    a = cfg["acquire"]
    raw = cfg.path_of("raw")
    cmd = [
        a["ytdlp_bin"],
        "-f", a["format"],
        "--merge-output-format", a["merge_output_format"],
        "--write-info-json", "--no-write-playlist-metafiles",
        "--download-archive", str(raw / a["archive_file"]),
        "--sleep-interval", str(a["sleep_interval_s"]),
        "--max-sleep-interval", str(a["max_sleep_interval_s"]),
        "--ignore-errors", "--no-overwrites", "--no-progress",
        "-o", str(raw / a["output_template"]),
    ]
    for flt in a.get("match_filters") or []:
        cmd += ["--match-filters", flt]
    if limit:
        cmd += ["--max-downloads", str(limit)]
    cmd += list(a.get("extra_args") or [])
    cmd.append(url)
    return cmd


def build_manifest(cfg: Config):
    """Scan paths.raw and write manifests/raw.parquet from what is on disk."""
    raw = cfg.path_of("raw")
    rows = []
    for info_path in sorted(raw.glob("*/*.info.json")):
        vid = info_path.name[: -len(".info.json")]
        try:
            info = json.loads(info_path.read_text())
        except json.JSONDecodeError as e:
            rows.append({"video_id": vid, "status": failed(f"bad info.json: {e}")})
            continue
        row = {
            "video_id": vid,
            "title": info.get("title"),
            "upload_date": info.get("upload_date"),
            "duration_s": info.get("duration"),
            "width": info.get("width"),
            "height": info.get("height"),
            "fps": info.get("fps"),
            "path": None,
        }
        media = _media_for(info_path.parent, vid)
        if media is None:
            row["status"] = failed("media file missing (incomplete download?)")
        else:
            row["path"] = str(media.relative_to(cfg.root)) if media.is_relative_to(cfg.root) else str(media)
            try:
                row.update(_ffprobe(media))  # trust the file over the metadata
                row["status"] = ok()
            except Exception as e:  # noqa: BLE001 - log and continue
                row["status"] = failed(f"ffprobe: {e}")
        rows.append(row)

    cols = ["video_id", "title", "upload_date", "duration_s", "width", "height",
            "fps", "path", "status"]
    df = pd.DataFrame(rows, columns=cols)
    df["config_hash"] = cfg.hash
    df["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    write_manifest(cfg, "raw", df)
    return df


def run(cfg: Config, limit: int | None = None, ids: list[str] | None = None,
        manifest_only: bool = False) -> None:
    raw = cfg.path_of("raw")
    raw.mkdir(parents=True, exist_ok=True)

    if not manifest_only:
        prune_archive(cfg)
        if ids:
            urls = [cfg["acquire"]["video_url_template"].format(id=i) for i in ids]
        else:
            urls = list(cfg["acquire"].get("sources") or [])
        if not urls:
            log.warning("no sources: set acquire.sources in %s or pass --ids", cfg.path)
        remaining = limit
        for url in urls:
            if remaining is not None and remaining <= 0:
                break
            before = len(_archive_ids(raw / cfg["acquire"]["archive_file"]))
            cmd = ytdlp_command(cfg, url, remaining)
            log.info("yt-dlp %s", url)
            rc = subprocess.run(cmd).returncode
            if rc not in (0, YTDLP_MAX_DOWNLOADS_EXIT):
                log.error("yt-dlp exited %d for %s (continuing)", rc, url)
            if remaining is not None:
                after = len(_archive_ids(raw / cfg["acquire"]["archive_file"]))
                remaining -= max(after - before, 0)

    df = build_manifest(cfg)
    n_ok = int((df["status"] == "ok").sum())
    log.info("raw manifest: %d videos, %d ok, %d not ok", len(df), n_ok, len(df) - n_ok)
    if n_ok:
        low = df[(df["status"] == "ok") & (df["fps"] < 49)]
        if len(low):
            log.info("%d videos below 50 fps (no high-fps format available)", len(low))
