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


def _add_terrain(e, cfg: Config, t: dict) -> None:
    """Milestone 5: per-env obstacle boxes for each env's (fixed) clip, plus a height
    scan around the pelvis in the actor and critic observations."""
    from mjlab.entity import EntityCfg
    from mjlab.envs.mdp.observations import height_scan
    from mjlab.managers.event_manager import EventTermCfg
    from mjlab.managers.observation_manager import ObservationTermCfg
    from mjlab.sensor import GridPatternCfg, ObjRef, RayCastSensorCfg

    from .multi_motion import obstacle_spec, place_obstacles

    k = int(t["max_boxes"])
    mc = e.commands["motion"]
    mc.fixed_env_clips = True
    mc.obstacles_dir = str(cfg.path_of("g1") / "obstacles")
    mc.max_boxes = k
    e.scene.entities["obstacles"] = EntityCfg(spec_fn=lambda: obstacle_spec(k))
    e.events["place_obstacles"] = EventTermCfg(func=place_obstacles, mode="startup")
    scan = RayCastSensorCfg(
        name="terrain_scan", frame=ObjRef(type="body", name="pelvis", entity="robot"),
        ray_alignment="yaw",
        pattern=GridPatternCfg(size=tuple(t["scan_size_m"]), resolution=float(t["scan_res_m"])),
        max_distance=float(t["scan_max_m"]), exclude_parent_body=True,
        include_geom_groups=(0,))
    e.scene.sensors = tuple(e.scene.sensors or ()) + (scan,)
    for group in ("actor", "critic"):
        e.observations[group].terms["height_scan"] = ObservationTermCfg(
            func=height_scan, params={"sensor_name": "terrain_scan"},
            scale=1.0 / float(t["scan_max_m"]))


def register_multi(cfg: Config) -> str:
    """Milestone 4 task: the sim-only tracking task with a multi-clip motion command
    (wct2g1.multi_motion), optional look-ahead observations (generalist.future_steps) and
    the network size from generalist.hidden_dims."""
    import mjlab.tasks  # noqa: F401
    from mjlab.tasks.registry import (list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls,
                                      register_mjlab_task)

    from .multi_motion import MultiMotionCommandCfg

    g = cfg["generalist"]
    if g["task_id"] in list_tasks():
        return g["task_id"]
    t = cfg["train_filter"]
    base, s = t["base_task"], t["sim_only"]

    def env(play: bool):
        e = _sim_only(load_env_cfg(base, play=play), s)
        mc = e.commands["motion"]
        kw = {f.name: getattr(mc, f.name) for f in dataclasses.fields(mc) if f.init}
        e.commands["motion"] = MultiMotionCommandCfg(**kw, bin_s=float(g["bin_s"]))
        e.commands["motion"].adaptive_uniform_ratio = float(g["adaptive_uniform_ratio"])
        terrain = g.get("terrain") or {}
        if terrain.get("enabled"):
            _add_terrain(e, cfg, terrain)
        offsets = tuple(int(k) for k in g.get("future_steps") or ())
        if offsets:
            from mjlab.managers.observation_manager import ObservationTermCfg

            from .multi_motion import future_reference
            for group in ("actor", "critic"):
                e.observations[group].terms["future_reference"] = ObservationTermCfg(
                    func=future_reference,
                    params={"command_name": "motion", "offsets": offsets})
        return e

    rl = load_rl_cfg(base)
    dims = tuple(int(d) for d in g["hidden_dims"])
    rl.actor = dataclasses.replace(rl.actor, hidden_dims=dims)
    rl.critic = dataclasses.replace(rl.critic, hidden_dims=dims)
    rl.save_interval = int(g["save_interval"])
    register_mjlab_task(task_id=g["task_id"], env_cfg=env(False), play_env_cfg=env(True),
                        rl_cfg=rl, runner_cls=load_runner_cls(base))
    return g["task_id"]
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


def _add_terrain(e, cfg: Config, t: dict) -> None:
    """Milestone 5: per-env obstacle boxes for each env's (fixed) clip, plus a height
    scan around the pelvis in the actor and critic observations."""
    from mjlab.entity import EntityCfg
    from mjlab.envs.mdp.observations import height_scan
    from mjlab.managers.event_manager import EventTermCfg
    from mjlab.managers.observation_manager import ObservationTermCfg
    from mjlab.sensor import GridPatternCfg, ObjRef, RayCastSensorCfg

    from .multi_motion import obstacle_spec, place_obstacles

    k = int(t["max_boxes"])
    mc = e.commands["motion"]
    mc.fixed_env_clips = True
    mc.obstacles_dir = str(cfg.path_of("g1") / "obstacles")
    mc.max_boxes = k
    e.scene.entities["obstacles"] = EntityCfg(spec_fn=lambda: obstacle_spec(k))
    e.events["place_obstacles"] = EventTermCfg(func=place_obstacles, mode="startup")
    scan = RayCastSensorCfg(
        name="terrain_scan", frame=ObjRef(type="body", name="pelvis", entity="robot"),
        ray_alignment="yaw",
        pattern=GridPatternCfg(size=tuple(t["scan_size_m"]), resolution=float(t["scan_res_m"])),
        max_distance=float(t["scan_max_m"]), exclude_parent_body=True,
        include_geom_groups=(0,))
    e.scene.sensors = tuple(e.scene.sensors or ()) + (scan,)
    for group in ("actor", "critic"):
        e.observations[group].terms["height_scan"] = ObservationTermCfg(
            func=height_scan, params={"sensor_name": "terrain_scan"},
            scale=1.0 / float(t["scan_max_m"]))


def register_multi(cfg: Config) -> str:
    """Milestone 4 task: the sim-only tracking task with a multi-clip motion command
    (wct2g1.multi_motion), optional look-ahead observations (generalist.future_steps) and
    the network size from generalist.hidden_dims."""
    import mjlab.tasks  # noqa: F401
    from mjlab.tasks.registry import (list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls,
                                      register_mjlab_task)

    from .multi_motion import MultiMotionCommandCfg

    g = cfg["generalist"]
    if g["task_id"] in list_tasks():
        return g["task_id"]
    t = cfg["train_filter"]
    base, s = t["base_task"], t["sim_only"]

    def env(play: bool):
        e = _sim_only(load_env_cfg(base, play=play), s)
        mc = e.commands["motion"]
        kw = {f.name: getattr(mc, f.name) for f in dataclasses.fields(mc) if f.init}
        e.commands["motion"] = MultiMotionCommandCfg(**kw, bin_s=float(g["bin_s"]))
        e.commands["motion"].adaptive_uniform_ratio = float(g["adaptive_uniform_ratio"])
        terrain = g.get("terrain") or {}
        if terrain.get("enabled"):
            _add_terrain(e, cfg, terrain)
        offsets = tuple(int(k) for k in g.get("future_steps") or ())
        if offsets:
            from mjlab.managers.observation_manager import ObservationTermCfg

            from .multi_motion import future_reference
            for group in ("actor", "critic"):
                e.observations[group].terms["future_reference"] = ObservationTermCfg(
                    func=future_reference,
                    params={"command_name": "motion", "offsets": offsets})
        return e

    rl = load_rl_cfg(base)
    dims = tuple(int(d) for d in g["hidden_dims"])
    rl.actor = dataclasses.replace(rl.actor, hidden_dims=dims)
    rl.critic = dataclasses.replace(rl.critic, hidden_dims=dims)
    rl.save_interval = int(g["save_interval"])
    register_mjlab_task(task_id=g["task_id"], env_cfg=env(False), play_env_cfg=env(True),
                        rl_cfg=rl, runner_cls=load_runner_cls(base))
    return g["task_id"]
