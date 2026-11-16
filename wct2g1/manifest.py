"""Per-stage manifests: one parquet file per stage under paths.manifests.

Every row has a key column (e.g. video_id, shot_id), a `status` of
`ok`, `rejected:<reason>` or `failed:<error>`, the config hash that produced it,
and an `updated_at` timestamp. Writes are atomic (temp file + rename) so a crash
never leaves a half-written manifest.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import Config

STAGE_KEYS = {
    "raw": "video_id",
    "shots": "shot_id",
    "tracks": "track_id",
    "smpl": "track_id",
    "clean": "clip_id",
    "g1": "clip_id",
    "filter": "clip_id",
}
# Order used by `wct2g1 report`.
STAGE_ORDER = list(STAGE_KEYS)

META_COLS = ["status", "config_hash", "updated_at"]


def ok() -> str:
    return "ok"


def rejected(reason: str) -> str:
    return f"rejected:{reason}"


def failed(error: str | BaseException) -> str:
    msg = str(error).replace("\n", " ").strip() or type(error).__name__
    return f"failed:{msg[:200]}"


def manifest_path(cfg: Config, stage: str) -> Path:
    return cfg.path_of("manifests") / f"{stage}.parquet"


def read_manifest(cfg: Config, stage: str) -> pd.DataFrame:
    p = manifest_path(cfg, stage)
    if not p.exists():
        return pd.DataFrame(columns=[STAGE_KEYS[stage], *META_COLS])
    return pd.read_parquet(p)


def write_manifest(cfg: Config, stage: str, df: pd.DataFrame) -> Path:
    key = STAGE_KEYS[stage]
    if key not in df.columns:
        raise ValueError(f"manifest '{stage}' missing key column '{key}'")
    if df[key].duplicated().any():
        dups = df.loc[df[key].duplicated(), key].tolist()[:5]
        raise ValueError(f"manifest '{stage}' has duplicate keys: {dups}")
    if "status" not in df.columns:
        raise ValueError(f"manifest '{stage}' missing 'status' column")
    p = manifest_path(cfg, stage)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".parquet.tmp")
    df.sort_values(key).reset_index(drop=True).to_parquet(tmp, index=False)
    os.replace(tmp, p)
    return p


def upsert(cfg: Config, stage: str, rows: list[dict] | pd.DataFrame) -> pd.DataFrame:
    """Insert or replace rows by key; stamps config_hash and updated_at."""
    key = STAGE_KEYS[stage]
    new = pd.DataFrame(rows) if not isinstance(rows, pd.DataFrame) else rows.copy()
    if new.empty:
        return read_manifest(cfg, stage)
    new["config_hash"] = cfg.hash
    new["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    old = read_manifest(cfg, stage)
    if not old.empty:
        old = old[~old[key].isin(new[key])]
        merged = pd.concat([old, new], ignore_index=True) if not old.empty else new
    else:
        merged = new
    write_manifest(cfg, stage, merged)
    return merged


def done_keys(cfg: Config, stage: str) -> set:
    """Keys already finished (ok or rejected) — skipped on rerun. Failures are retried."""
    df = read_manifest(cfg, stage)
    if df.empty:
        return set()
    finished = df["status"].eq("ok") | df["status"].str.startswith("rejected:")
    return set(df.loc[finished, STAGE_KEYS[stage]])


def status_kind(status: str) -> str:
    return status.split(":", 1)[0]
