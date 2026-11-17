"""Stage 7: per-clip filtering with mjlab BeyondMimic tracking (`mjlab` env).

For each retargeted clip: mjlab's csv_to_npz (patched for local output) computes body
kinematics; a short single-clip PPO policy is trained with the sim-only task; 20 rollouts
from the clip start measure fall rate, MPJPE and completion. Accepted clips are copied to
data/final_dataset/. Each step runs in its own subprocess so GPU state is reset per clip.

Multi-clip final training (Milestone 4) is not implemented yet: mjlab's motion command
takes a single motion file, so it needs a stitched motion or a small sampler patch.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .config import REPO_ROOT, Config
from .manifest import shard, filter_ids, failed, ok, read_manifest, rejected, upsert

log = logging.getLogger(__name__)


def _run(cmd: list[str], log_file: Path, cwd: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    pp = os.pathsep.join(filter(None, [str(REPO_ROOT), os.environ.get("PYTHONPATH")]))
    env = dict(os.environ, MUJOCO_GL="egl", PYTHONPATH=pp)   # wct2g1 importable from any cwd
    with open(log_file, "w") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=cwd, env=env)
    if r.returncode != 0:
        tail = log_file.read_text().splitlines()[-5:]
        raise RuntimeError(f"{cmd[2] if len(cmd) > 2 else cmd[0]} exited {r.returncode}: "
                           + " | ".join(tail))


def convert(cfg: Config, csv: Path, out_npz: Path, logs: Path) -> None:
    fps = cfg["train_filter"]["convert"]["fps"]
    _run([sys.executable, "-m", "mjlab.scripts.csv_to_npz", "--input-file", str(csv),
          "--output-name", out_npz.stem, "--input-fps", str(fps), "--output-fps", str(fps),
          "--output-file", str(out_npz)], logs / "convert.log", cfg.root)


def _checkpoints(run_dir: Path, name: str) -> list[Path]:
    return sorted(run_dir.glob(f"{name}/*/model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))


def train(cfg: Config, motion: Path, run_dir: Path, name: str) -> Path:
    f = cfg["train_filter"]["filter"]
    done = [p for p in _checkpoints(run_dir, name)
            if int(p.stem.split("_")[1]) >= f["max_iterations"] - 1]
    if done:                       # a finished run exists (e.g. re-evaluating): reuse it
        log.info("reusing %s", done[-1])
        return done[-1]
    _run([sys.executable, "-m", "wct2g1.mjlab_launch", "train", "--config", str(cfg.path),
          "--env.commands.motion.motion-file", str(motion),
          "--env.scene.num-envs", str(f["num_envs"]),
          "--agent.max-iterations", str(f["max_iterations"]),
          "--agent.logger", f["logger"], "--agent.experiment-name", name,
          "--agent.seed", str(cfg["seed"]),
          "--log-root", str(run_dir)], run_dir / "train.log", cfg.root)
    ckpts = _checkpoints(run_dir, name)
    if not ckpts:
        raise RuntimeError("training produced no checkpoint")
    return ckpts[-1]


def evaluate(cfg: Config, motion: Path, ckpt: Path, run_dir: Path,
             video: Path | None = None) -> dict:
    f = cfg["train_filter"]["filter"]
    out = run_dir / "eval.json"
    extra = [] if video is None else ["--video", str(video)]
    if video is not None:
        video.parent.mkdir(parents=True, exist_ok=True)
    _run([sys.executable, "-m", "wct2g1.mjlab_launch", "eval", "--config", str(cfg.path),
          "--motion-file", str(motion), "--checkpoint", str(ckpt),
          "--num-envs", str(f["rollouts"]), "--output-json", str(out), *extra],
         run_dir / "eval.log", cfg.root)
    return json.loads(out.read_text())


def decide(m: dict, a: dict) -> str:
    if m["fall_rate"] >= a["max_fall_rate"]:
        return rejected("falls")
    key = {"global": "mpjpe_m", "root_relative": "mpjpe_rel_m"}[a["mpjpe_frame"]]
    if m[key] >= a["max_mpjpe_m"]:
        return rejected("mpjpe")
    if m["completion"] <= a["min_completion"]:
        return rejected("incomplete")
    return ok()


def run(cfg: Config, limit: int | None = None, ids: list[str] | None = None) -> None:
    t = cfg["train_filter"]
    clips = read_manifest(cfg, "g1")
    clips = clips[clips["status"] == "ok"]
    if ids:
        clips = filter_ids(clips, ids)
    done = read_manifest(cfg, "filter")
    if not done.empty:
        fin = set(done.loc[~done["status"].str.startswith("failed:"), "clip_id"])
        clips = clips[~clips["clip_id"].isin(fin)]
    clips = shard(clips, "clip_id")
    if limit:
        clips = clips.head(limit)
    log.info("train_filter: %d clips to process", len(clips))
    final_dir = cfg.path_of("final_dataset")
    for _, c in clips.iterrows():
        cid = c["clip_id"]
        run_dir = cfg.path_of("policies") / "filter" / cid
        motion = cfg.path_of("g1") / "npz" / f"{cid}.npz"
        row = dict(clip_id=cid, run_dir=str(run_dir.relative_to(cfg.root)))
        try:
            motion.parent.mkdir(parents=True, exist_ok=True)
            if not motion.exists():
                convert(cfg, cfg.root / c["path"], motion, run_dir)
            ckpt = train(cfg, motion, run_dir, cid)
            vid = (cfg.path_of("review") / "policy_rollouts" / f"{cid}.mp4"
                   if t["filter"]["video"] else None)
            m = evaluate(cfg, motion, ckpt, run_dir, vid)
            row.update(m, checkpoint=str(ckpt.relative_to(cfg.root)),
                       torque_scale=t["sim_only"]["torque_scale"])
            row["status"] = decide(m, t["filter"]["accept"])
            row["accepted"] = row["status"] == "ok"
            if row["accepted"]:
                final_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(cfg.root / c["path"], final_dir / f"{cid}.csv")
                shutil.copy2(motion, final_dir / f"{cid}.npz")
        except Exception as e:  # noqa: BLE001
            log.exception("train_filter failed for %s", cid)
            row.update(status=failed(e), accepted=False)
        upsert(cfg, "filter", [row])
        log.info("%s: %s %s", cid, row["status"],
                 {k: row.get(k) for k in ("fall_rate", "mpjpe_m", "completion")})
