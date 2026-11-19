"""Milestone 6: a goal-directed parkour policy (`mjlab` env).

parkour_train  trains one PPO policy on random obstacle courses with a difficulty
               curriculum (wct2g1.course). One GPU job.
parkour_eval   rolls the policy out on (a) random courses at every difficulty level
               ("random") and (b) every hand-written course in parkour.courses_dir
               (courses/<name>.yaml), recording a video of each named course. Writes the
               `parkour` manifest (one row per run and course). `--ids NAME` runs only those
               courses, e.g. a new course you just wrote.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from .config import Config
from .manifest import failed, ok, upsert
from .train_filter import _checkpoints, _run

log = logging.getLogger(__name__)
NAME = "parkour"
RANDOM = "random"


def run_dir(cfg: Config) -> Path:
    return cfg.path_of("policies") / NAME / cfg["parkour"]["run_name"]


def latest_checkpoint(cfg: Config) -> Path | None:
    ck = _checkpoints(run_dir(cfg), NAME)
    return ck[-1] if ck else None


def courses(cfg: Config) -> dict[str, Path]:
    d = cfg.root / cfg["parkour"]["courses_dir"]
    return {p.stem: p for p in sorted(d.glob("*.yaml"))}


def train_stage(cfg: Config, limit: int | None = None, ids: list[str] | None = None) -> None:
    del limit, ids
    p = cfg["parkour"]
    rd = run_dir(cfg)
    rd.mkdir(parents=True, exist_ok=True)
    done = [c for c in _checkpoints(rd, NAME)
            if int(c.stem.split("_")[1]) >= p["max_iterations"] - 1]
    if done:
        log.info("parkour: finished run exists (%s), nothing to do", done[-1])
        return
    _run([sys.executable, "-m", "wct2g1.mjlab_launch", "train", "--config", str(cfg.path),
          "--task", "course",
          "--env.scene.num-envs", str(p["num_envs"]),
          "--agent.max-iterations", str(p["max_iterations"]),
          "--agent.logger", p["logger"], "--agent.experiment-name", NAME,
          "--agent.seed", str(cfg["seed"]),
          "--log-root", str(rd)], rd / "train.log", cfg.root)
    log.info("parkour: done, checkpoint %s", latest_checkpoint(cfg))


def eval_stage(cfg: Config, limit: int | None = None, ids: list[str] | None = None) -> None:
    p = cfg["parkour"]
    ckpt = Path(p["checkpoint"]) if p.get("checkpoint") else latest_checkpoint(cfg)
    if ckpt is None:
        raise RuntimeError(f"no parkour checkpoint under {run_dir(cfg)}")
    ckpt = ckpt if ckpt.is_absolute() else cfg.root / ckpt
    todo: dict[str, Path | None] = {RANDOM: None, **courses(cfg)}
    if ids:
        unknown = set(ids) - set(todo)
        if unknown:
            raise ValueError(f"unknown courses {sorted(unknown)}; have {sorted(todo)}")
        todo = {k: v for k, v in todo.items() if k in ids}
    if limit:
        todo = dict(list(todo.items())[:limit])
    vid_dir = cfg.path_of("review") / "parkour" / p["run_name"]
    for name, path in todo.items():
        rd = run_dir(cfg) / "eval" / name
        rd.mkdir(parents=True, exist_ok=True)
        out = rd / "eval.json"
        row = dict(eval_id=f"{p['run_name']}/{name}", course=name, run_name=p["run_name"],
                   checkpoint=str(ckpt.relative_to(cfg.root)))
        cmd = [sys.executable, "-m", "wct2g1.mjlab_launch", "course-eval",
               "--config", str(cfg.path), "--checkpoint", str(ckpt),
               "--output-json", str(out)]
        if path is None:
            lv = int(p["levels"])
            cmd += ["--levels", str(lv), "--num-envs", str(lv * int(p["eval_per_level"]))]
        else:
            vid_dir.mkdir(parents=True, exist_ok=True)
            cmd += ["--course", str(path), "--num-envs", str(p["rollouts"])]
            if p["video"]:
                cmd += ["--video", str(vid_dir / f"{name}.mp4")]
        try:
            _run(cmd, rd / "eval.log", cfg.root)
            m = json.loads(out.read_text())
            per_level = m.pop("per_level", None)
            if per_level is not None:
                row["per_level_json"] = json.dumps(per_level)
            row.update(m, status=ok())
        except Exception as e:  # noqa: BLE001
            log.exception("parkour_eval failed for %s", name)
            row.update(status=failed(e))
        upsert(cfg, NAME, [row])
        log.info("%s: %s", name, {k: row.get(k) for k in
                                  ("success_rate", "fall_rate", "progress", "time_to_goal_s")})
