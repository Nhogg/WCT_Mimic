"""Per-stage manifests: one parquet file per stage under paths.manifests.

Every row has a key column (e.g. video_id, shot_id), a `status` of
`ok`, `rejected:<reason>` or `failed:<error>`, the config hash that produced it,
and an `updated_at` timestamp. Writes are atomic (temp file + rename) so a crash
never leaves a half-written manifest.
"""
from __future__ import annotations

import os
import time
import zlib
from contextlib import contextmanager
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
    "obstacles": "clip_id",
    "generalist": "clip_id",
    "parkour": "course",
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
    tmp = p.with_suffix(f".parquet.{os.getpid()}.tmp")
    df.sort_values(key).reset_index(drop=True).to_parquet(tmp, index=False)
    os.replace(tmp, p)
    return p


@contextmanager
def manifest_lock(cfg: Config, stage: str):
    """Exclusive lock for a manifest's read-modify-write, safe across SLURM array tasks on NFS
    (O_CREAT|O_EXCL lock file). A lock older than manifest.lock_stale_s is assumed abandoned."""
    m = cfg["manifest"]
    lock = manifest_path(cfg, stage).with_suffix(".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.uname().nodename}:{os.getpid()}".encode())
            os.close(fd)
            break
        except FileExistsError:
            try:
                if time.time() - lock.stat().st_mtime > m["lock_stale_s"]:
                    lock.unlink(missing_ok=True)
                    continue
            except FileNotFoundError:
                continue
            time.sleep(m["lock_poll_s"])
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def upsert(cfg: Config, stage: str, rows: list[dict] | pd.DataFrame,
           replace: tuple[str, object] | None = None) -> pd.DataFrame:
    """Insert or replace rows by key; stamps config_hash and updated_at. Locked, so parallel
    jobs (SLURM arrays) don't overwrite each other's rows. replace=(col, value) first drops
    every existing row with that value (e.g. all old clips of a re-cleaned track)."""
    key = STAGE_KEYS[stage]
    new = pd.DataFrame(rows) if not isinstance(rows, pd.DataFrame) else rows.copy()
    if new.empty:
        return read_manifest(cfg, stage)
    new["config_hash"] = cfg.hash
    new["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with manifest_lock(cfg, stage):
        old = read_manifest(cfg, stage)
        if replace is not None and not old.empty and replace[0] in old.columns:
            old = old[old[replace[0]] != replace[1]]
        if not old.empty:
            old = old[~old[key].isin(new[key])]
            merged = pd.concat([old, new], ignore_index=True) if not old.empty else new
        else:
            merged = new
        write_manifest(cfg, stage, merged)
    return merged


def shard(df: pd.DataFrame, key: str, spec: str | None = None) -> pd.DataFrame:
    """Rows for shard "i/n" (default: env WCT2G1_SHARD; unset = all), by a stable hash of key."""
    spec = spec if spec is not None else os.environ.get("WCT2G1_SHARD")
    if not spec or df.empty:
        return df
    i, n = (int(x) for x in spec.split("/"))
    if not 0 <= i < n:
        raise ValueError(f"bad shard {spec!r}")
    h = df[key].map(lambda k: zlib.crc32(str(k).encode()) % n)
    return df[h == i]


def done_keys(cfg: Config, stage: str) -> set:
    """Keys already finished (ok or rejected) — skipped on rerun. Failures are retried."""
    df = read_manifest(cfg, stage)
    if df.empty:
        return set()
    finished = df["status"].eq("ok") | df["status"].str.startswith("rejected:")
    return set(df.loc[finished, STAGE_KEYS[stage]])


def status_kind(status: str) -> str:
    return status.split(":", 1)[0]


def filter_ids(df: pd.DataFrame, ids, cols=("video_id", "shot_id", "track_id", "clip_id")):
    """Rows whose id in any of `cols` (those present) is in `ids`. Empty-safe: a manifest
    holding only failed rows may lack downstream id columns."""
    if not ids or df.empty:
        return df
    mask = pd.Series(False, index=df.index)
    for c in cols:
        if c in df.columns:
            mask |= df[c].isin(ids)
    return df[mask]
