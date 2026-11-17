"""Our mjlab task variant: BeyondMimic G1 tracking with sim-only settings (`mjlab` env).

Simulation only, so domain randomization and observation noise are switched off. Actuator
effort limits can be scaled with train_filter.sim_only.torque_scale (1.0 = real G1).
"""
from __future__ import annotations

import dataclasses

from .config import Config


def _sim_only(env_cfg, s: dict):
    for name in s["drop_events"]:
        env_cfg.events.pop(name, None)
    env_cfg.observations["actor"].enable_corruption = bool(s["obs_corruption"])
    scale = float(s["torque_scale"])
    if scale != 1.0:
        robot = env_cfg.scene.entities["robot"]
        art = robot.articulation
        acts = tuple(dataclasses.replace(a, effort_limit=a.effort_limit * scale)
                     for a in art.actuators)
        robot.articulation = dataclasses.replace(art, actuators=acts)
    return env_cfg


def register(cfg: Config) -> str:
    import mjlab.tasks  # noqa: F401  (populates the registry with the base task)
    from mjlab.tasks.registry import (list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls,
                                      register_mjlab_task)

    t = cfg["train_filter"]
    if t["task_id"] in list_tasks():
        return t["task_id"]
    base = t["base_task"]
    s = t["sim_only"]
    register_mjlab_task(
        task_id=t["task_id"],
        env_cfg=_sim_only(load_env_cfg(base), s),
        play_env_cfg=_sim_only(load_env_cfg(base, play=True), s),
        rl_cfg=load_rl_cfg(base),
        runner_cls=load_runner_cls(base),
    )
    return t["task_id"]
