"""Milestone 4: one tracking policy for every accepted clip (`mjlab` env).

generalist_train  trains a single PPO policy on all clips accepted by Stage 7, using the
                  multi-clip motion command (wct2g1.multi_motion). One GPU job.
generalist_eval   rolls the trained policy out from the start of every retargeted clip
                  (accepted or not), with the Stage 7 acceptance thresholds, and writes the
                  `generalist` manifest. Clips rejected by Stage 7 show whether training on
                  many clips helps the hard ones. Shardable (SLURM array).
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from .config import Config
from .manifest import failed, filter_ids, read_manifest, shard, upsert
from .train_filter import _checkpoints, _run, decide

log = logging.getLogger(__name__)
NAME = "generalist"


def run_dir(cfg: Config) -> Path:
    return cfg.path_of("policies") / NAME / cfg["generalist"]["run_name"]


def training_clips(cfg: Config, ids: list[str] | None = None) -> list[Path]:
    """Stage 7 accepted clips (+ clips with obstacle boxes when terrain is on)."""
    f = read_manifest(cfg, "filter")
    if f.empty:
        raise RuntimeError("no Stage 7 results: run train_filter first")
    f = f[f["accepted"].astype(bool)]
    if ids:
        f = filter_ids(f, ids)
    keep = set(f["clip_id"])
    t = cfg["generalist"].get("terrain") or {}
    if t.get("enabled") and t.get("add_obstacle_clips"):
        # Clips with inferred boxes: Stage 7 judged them on a flat floor, where a foot
        # planted on a box is impossible, so they are trained on regardless.
        o = read_manifest(cfg, "obstacles")
        extra = set(o.loc[o["n_boxes"].fillna(0) > 0, "clip_id"]) if not o.empty else set()
        if ids:
            extra &= set(ids)
        keep |= extra
    d = cfg.path_of("g1") / "npz"
    paths = [d / f"{c}.npz" for c in sorted(keep)]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} accepted clips lack npz, e.g. {missing[0]}")
    return paths


def latest_checkpoint(cfg: Config) -> Path | None:
    ck = _checkpoints(run_dir(cfg), NAME)
    return ck[-1] if ck else None


def train_stage(cfg: Config, limit: int | None = None, ids: list[str] | None = None) -> None:
    g = cfg["generalist"]
    rd = run_dir(cfg)
    rd.mkdir(parents=True, exist_ok=True)
    clips = training_clips(cfg, ids)
    if limit:
        clips = clips[:limit]
    lst = rd / "motions.txt"
    lst.write_text("".join(f"{p}\n" for p in clips))
    log.info("generalist: training on %d clips (%s)", len(clips), lst)
    done = [p for p in _checkpoints(rd, NAME)
            if int(p.stem.split("_")[1]) >= g["max_iterations"] - 1]
    if done:
        log.info("generalist: finished run exists (%s), nothing to do", done[-1])
        return
    _run([sys.executable, "-m", "wct2g1.mjlab_launch", "train", "--config", str(cfg.path),
          "--task", "multi",
          "--env.commands.motion.motion-file", str(lst),
          "--env.scene.num-envs", str(g["num_envs"]),
          "--agent.max-iterations", str(g["max_iterations"]),
          "--agent.logger", g["logger"], "--agent.experiment-name", NAME,
          "--agent.seed", str(cfg["seed"]),
          "--log-root", str(rd)], rd / "train.log", cfg.root)
    log.info("generalist: done, checkpoint %s", latest_checkpoint(cfg))


def eval_stage(cfg: Config, limit: int | None = None, ids: list[str] | None = None) -> None:
    g = cfg["generalist"]
    ckpt = Path(g["checkpoint"]) if g.get("checkpoint") else latest_checkpoint(cfg)
    if ckpt is None or not ckpt.exists():
        raise RuntimeError(f"no generalist checkpoint under {run_dir(cfg)}")
    ckpt = ckpt if ckpt.is_absolute() else cfg.root / ckpt
    trained = {Path(p).stem for p in (run_dir(cfg) / "motions.txt").read_text().split()}
    clips = read_manifest(cfg, "filter")
    if ids:
        clips = filter_ids(clips, ids)
    clips = shard(clips, "clip_id")
    if limit:
        clips = clips.head(limit)
    log.info("generalist_eval: %d clips with %s", len(clips), ckpt)
    a = cfg["train_filter"]["filter"]["accept"]
    out_root = run_dir(cfg) / "eval"
    for _, c in clips.iterrows():
        cid = c["clip_id"]
        motion = cfg.path_of("g1") / "npz" / f"{cid}.npz"
        rd = out_root / cid
        rd.mkdir(parents=True, exist_ok=True)
        row = dict(clip_id=cid, checkpoint=str(ckpt.relative_to(cfg.root)),
                   in_training=cid in trained, stage7_status=c["status"])
        try:
            vid = (cfg.path_of("review") / "generalist_rollouts" / g["run_name"] / f"{cid}.mp4"
                   if g["video"] else None)
            extra = []
            if vid is not None:
                vid.parent.mkdir(parents=True, exist_ok=True)
                extra = ["--video", str(vid)]
            out = rd / "eval.json"
            _run([sys.executable, "-m", "wct2g1.mjlab_launch", "eval", "--config",
                  str(cfg.path), "--task", "multi", "--motion-file", str(motion),
                  "--checkpoint", str(ckpt), "--num-envs", str(g["rollouts"]),
                  "--output-json", str(out), *extra], rd / "eval.log", cfg.root)
            m = json.loads(out.read_text())
            row.update(m, status=decide(m, a))
        except Exception as e:  # noqa: BLE001
            log.exception("generalist_eval failed for %s", cid)
            row.update(status=failed(e))
        row["accepted"] = row["status"] == "ok"
        upsert(cfg, "generalist", [row])
        log.info("%s: %s %s", cid, row["status"],
                 {k: row.get(k) for k in ("fall_rate", "mpjpe_rel_m", "completion")})
