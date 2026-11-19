"""Milestone 6: "complete this parkour course" task (`mjlab` env).

Built on mjlab's G1 rough-terrain velocity task (rewards, sensors, actuators), with:
  CourseTerrainCfg   a terrain cell holding one straight course (wct2g1.course_profile):
                     random per cell for training, or a fixed hand-written course.
  CourseGoalCommand  goal = end of the course. The policy sees the goal position in its
                     yaw frame and a desired speed; the velocity-style rewards see the
                     twist that walks straight at the goal at that speed (zero once there).
  course_levels      terrain curriculum: promote envs that reached the goal, demote envs
                     that covered less than `down` of the course.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeometry, TerrainOutput
from mjlab.utils.lab_api.math import wrap_to_pi

from . import course_profile as cp

FLOOR_DEPTH = 1.0          # every segment box reaches this far below z=0


# --- terrain ---------------------------------------------------------------------------

def _color(h: float) -> tuple[float, float, float, float]:
    t = float(np.clip((h + 0.5) / 1.0, 0.0, 1.0))
    return (0.35 + 0.5 * t, 0.4 + 0.2 * t, 0.75 - 0.45 * t, 1.0)


@dataclass(kw_only=True)
class CourseTerrainCfg(SubTerrainCfg):
    """One course along +x across the whole cell; segments span the cell width, so there
    is no way around an obstacle. `segments` fixes the course; otherwise it is sampled
    with `params` (config parkour.course) at the cell's difficulty."""
    params: dict | None = None
    segments: list | None = None

    def function(self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
                 ) -> TerrainOutput:
        lx, ly = self.size
        if self.segments is not None:          # a full lane, e.g. from cp.from_spec
            segs = cp.merge([(float(a), float(b)) for a, b in self.segments])
            if cp.total_length(segs) < lx - 1e-6:
                segs = cp.merge(segs + cp.flat(lx - cp.total_length(segs)))
        else:
            assert self.params is not None
            segs = cp.sample_profile(rng, difficulty, lx, self.params)
        body = spec.body("terrain")
        geoms, x = [], 0.0
        for length, h in segs:
            g = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
                              size=(length / 2, ly / 2, (h + FLOOR_DEPTH) / 2),
                              pos=(x + length / 2, ly / 2, (h - FLOOR_DEPTH) / 2))
            geoms.append(TerrainGeometry(geom=g, color=_color(h)))
            x += length
        return TerrainOutput(origin=np.array([cp.START_M / 2, ly / 2, 0.0]), geometries=geoms)


# --- goal command ----------------------------------------------------------------------

class CourseGoalCommand(CommandTerm):
    cfg: CourseGoalCommandCfg

    def __init__(self, cfg: CourseGoalCommandCfg, env):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.entity_name]
        n, dev = self.num_envs, self.device
        self.goal_w = torch.zeros(n, 3, device=dev)
        self.start_w = torch.zeros(n, 3, device=dev)
        self.speed = torch.zeros(n, device=dev)
        self.vel_cmd_b = torch.zeros(n, 3, device=dev)
        self.goal_b = torch.zeros(n, 2, device=dev)
        self.dist = torch.zeros(n, device=dev)
        self.reached = torch.zeros(n, dtype=torch.bool, device=dev)
        self.progress = torch.zeros(n, device=dev)       # max fraction of the course covered
        self.metrics["reached"] = torch.zeros(n, device=dev)
        self.metrics["progress"] = torch.zeros(n, device=dev)

    @property
    def command(self) -> torch.Tensor:
        """Twist toward the goal (body frame: vx, vy, yaw rate) for the velocity rewards."""
        return self.vel_cmd_b

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        o = self._env.scene.env_origins[env_ids]
        self.start_w[env_ids] = o
        self.goal_w[env_ids] = o
        self.goal_w[env_ids, 0] += self.cfg.goal_offset_x
        if self.cfg.goal_y_jitter > 0:
            self.goal_w[env_ids, 1] += torch.empty(len(env_ids), device=self.device
                                                   ).uniform_(-1, 1) * self.cfg.goal_y_jitter
        self.speed[env_ids] = torch.empty(len(env_ids), device=self.device
                                          ).uniform_(*self.cfg.speed_range)
        self.reached[env_ids] = False
        self.progress[env_ids] = 0.0

    def _update_command(self, env_ids: torch.Tensor | None) -> None:
        del env_ids
        pos = self.robot.data.root_link_pos_w
        rel = self.goal_w[:, :2] - pos[:, :2]
        self.dist = rel.norm(dim=-1)
        yaw = self.robot.data.heading_w
        c, s = torch.cos(yaw), torch.sin(yaw)
        self.goal_b = torch.stack([c * rel[:, 0] + s * rel[:, 1],
                                   -s * rel[:, 0] + c * rel[:, 1]], dim=-1)
        self.reached |= self.dist < self.cfg.reach_radius
        along = (pos[:, 0] - self.start_w[:, 0]) / self.cfg.goal_offset_x
        self.progress = torch.maximum(self.progress, along.clamp(0.0, 1.0))
        self.progress = torch.where(self.reached, torch.ones_like(self.progress), self.progress)
        # Slow down over the last `slow_radius` metres; stand once the goal is reached.
        v = self.speed * (self.dist / self.cfg.slow_radius).clamp(max=1.0)
        v = torch.where(self.reached, torch.zeros_like(v), v)
        dir_b = self.goal_b / self.dist.clamp(min=1e-6).unsqueeze(-1)
        self.vel_cmd_b[:, :2] = dir_b * v.unsqueeze(-1)
        err = wrap_to_pi(torch.atan2(self.goal_b[:, 1], self.goal_b[:, 0]))
        w = (self.cfg.heading_gain * err).clamp(-self.cfg.max_yaw_rate, self.cfg.max_yaw_rate)
        self.vel_cmd_b[:, 2] = torch.where(self.reached, torch.zeros_like(w), w)

    def _update_metrics(self) -> None:
        self.metrics["reached"] = self.reached.float()
        self.metrics["progress"] = self.progress.clone()


@dataclass(kw_only=True)
class CourseGoalCommandCfg(CommandTermCfg):
    entity_name: str = "robot"
    goal_offset_x: float = 8.5          # spawn point -> goal along +x (course_profile)
    goal_y_jitter: float = 0.5
    speed_range: tuple[float, float] = (0.6, 1.5)
    reach_radius: float = 0.5
    slow_radius: float = 1.0
    heading_gain: float = 1.0
    max_yaw_rate: float = 1.0
    max_obs_dist: float = 5.0

    def build(self, env) -> CourseGoalCommand:
        return CourseGoalCommand(self, env)


# --- observations, rewards, curriculum -------------------------------------------------

def goal_obs(env, command_name: str = "goal") -> torch.Tensor:
    """Goal in the robot's yaw frame (clipped to max_obs_dist), its distance, and the
    desired speed (zero once reached)."""
    t: CourseGoalCommand = env.command_manager.get_term(command_name)
    m = t.cfg.max_obs_dist
    scale = (m / t.dist.clamp(min=m)).unsqueeze(-1)        # 1 inside m, shrink beyond
    v = torch.where(t.reached, torch.zeros_like(t.speed), t.speed)
    return torch.cat([t.goal_b * scale, t.dist.clamp(max=m).unsqueeze(-1), v.unsqueeze(-1)],
                     dim=-1)


def track_goal_velocity(env, std: float, command_name: str = "goal") -> torch.Tensor:
    """Like mjlab's track_linear_velocity but horizontal only: climbing needs vertical
    velocity, so it is not penalised."""
    cmd = env.command_manager.get_command(command_name)
    v = env.scene["robot"].data.root_link_lin_vel_b
    return torch.exp(-torch.sum(torch.square(cmd[:, :2] - v[:, :2]), dim=1) / std**2)


def goal_reached(env, command_name: str = "goal") -> torch.Tensor:
    t: CourseGoalCommand = env.command_manager.get_term(command_name)
    return t.reached.float()


def course_levels(env, env_ids: torch.Tensor, command_name: str = "goal",
                  down: float = 0.4) -> dict[str, torch.Tensor]:
    """Promote envs whose episode reached the goal; demote those that covered < `down`
    of the course. Levels above the hardest wrap to a random level (mjlab default)."""
    terrain = env.scene.terrain
    t: CourseGoalCommand = env.command_manager.get_term(command_name)
    prog = t.progress[env_ids]
    up = t.reached[env_ids]
    dn = (prog < down) & ~up
    if env.common_step_counter == 0:              # spawn reset: nothing walked yet
        up, dn = torch.zeros_like(up), torch.zeros_like(dn)
    terrain.update_env_origins(env_ids, up, dn)
    lv = terrain.terrain_levels.float()
    return {"mean": lv.mean(), "max": lv.max()}


# --- task --------------------------------------------------------------------------------

def make_env_cfg(cfg, segments: list | None = None, num_rows: int | None = None,
                 lane_x: float | None = None):
    """The course env cfg. `segments` (a full lane from course_profile.from_spec) gives a
    one-cell world holding that course; otherwise the random-course curriculum grid.
    `num_rows` overrides the number of difficulty levels and `lane_x` the lane length
    (evaluation)."""
    from mjlab.managers.curriculum_manager import CurriculumTermCfg
    from mjlab.managers.observation_manager import ObservationTermCfg
    from mjlab.managers.reward_manager import RewardTermCfg
    from mjlab.sensor import GridPatternCfg
    from mjlab.tasks.registry import load_env_cfg
    from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

    from .sim_task import _sim_only

    p = cfg["parkour"]
    e = _sim_only(load_env_cfg(p["base_task"]), cfg["train_filter"]["sim_only"])
    lane = (float(lane_x or p["lane_m"][0]), float(p["lane_m"][1]))

    if segments is not None:
        subs = {"course": CourseTerrainCfg(segments=[tuple(s) for s in segments])}
        rows = 1
    else:
        subs = {f"course_{i}": CourseTerrainCfg(params=dict(p["course"]))
                for i in range(int(p["variants"]))}
        rows = int(num_rows or p["levels"])
    e.scene.terrain.terrain_generator = TerrainGeneratorCfg(
        size=lane, border_width=float(p["border_m"]), num_rows=rows, num_cols=len(subs),
        curriculum=True, sub_terrains=subs, add_lights=True, seed=int(cfg["seed"]))
    e.scene.terrain.max_init_terrain_level = int(p["max_init_level"])

    for s in e.scene.sensors:
        if s.name == "terrain_scan":
            s.pattern = GridPatternCfg(size=tuple(p["scan_size_m"]),
                                       offset=tuple(p["scan_offset_m"]),
                                       resolution=float(p["scan_res_m"]))

    e.commands.pop("twist")
    e.commands["goal"] = CourseGoalCommandCfg(
        resampling_time_range=(1e9, 1e9), goal_offset_x=cp.goal_offset(lane[0]),
        goal_y_jitter=float(p["goal_y_jitter_m"]), speed_range=tuple(p["speed_mps"]),
        reach_radius=float(p["reach_radius_m"]))
    for g in e.observations.values():
        g.terms["command"] = ObservationTermCfg(func=goal_obs, params={"command_name": "goal"})
    for term in e.rewards.values():
        if (term.params or {}).get("command_name") == "twist":
            term.params["command_name"] = "goal"
    w = p["rewards"]
    e.rewards["track_linear_velocity"] = RewardTermCfg(
        func=track_goal_velocity, weight=float(w["goal_velocity"]),
        params={"command_name": "goal", "std": float(w["goal_velocity_std"])})
    e.rewards["track_angular_velocity"].weight = float(w["heading"])
    e.rewards["goal_reached"] = RewardTermCfg(func=goal_reached, weight=float(w["reached"]),
                                              params={"command_name": "goal"})
    for name in ("upright", "pose"):
        e.rewards[name].weight = float(w[name])
    if (p.get("style") or {}).get("enabled"):
        _add_style(e, cfg)
    e.curriculum = {"course_levels": CurriculumTermCfg(
        func=course_levels, params={"command_name": "goal", "down": float(p["demote_below"])})}
    e.episode_length_s = float(p["episode_s"])
    leftover = [n for n, t in {**e.rewards, **e.events, **e.terminations}.items()
                if "twist" in str(t.params or {})]
    if leftover:
        raise RuntimeError(f"terms still reference the twist command: {leftover}")
    return e


def _add_style(e, cfg) -> None:
    """WCT style reward (wct2g1.style) on the clips the M4 generalist trains on, plus
    discriminator statistics as metrics. Style replaces the default-pose reward."""
    from mjlab.managers.metrics_manager import MetricsTermCfg
    from mjlab.managers.reward_manager import RewardTermCfg

    from .generalist import training_clips
    from .style import StyleReward, save_list, style_stat

    st = cfg["parkour"]["style"]
    lst = save_list(training_clips(cfg),
                    cfg.path_of("policies") / "parkour" / cfg["parkour"]["run_name"]
                    / "style_motions.txt")
    e.rewards["style"] = RewardTermCfg(
        func=StyleReward, weight=float(st["weight"]),
        params=dict(motions=str(lst), hidden=list(st["hidden"]), lr=float(st["lr"]),
                    buffer_size=int(st["buffer_size"]), batch_size=int(st["batch_size"]),
                    updates=int(st["updates"]), update_every=int(st["update_every"]),
                    gp_weight=float(st["gp_weight"])))
    e.rewards["pose"].weight = float(st["pose_weight"])
    for stat in ("d_expert", "d_policy", "loss", "gp"):
        e.metrics[f"style_{stat}"] = MetricsTermCfg(func=style_stat, params={"stat": stat})


def register(cfg) -> str:
    """Register config parkour.task_id (training: random-course curriculum)."""
    import dataclasses

    import mjlab.tasks  # noqa: F401
    from mjlab.rl import MjlabOnPolicyRunner
    from mjlab.tasks.registry import list_tasks, load_rl_cfg, register_mjlab_task

    p = cfg["parkour"]
    if p["task_id"] in list_tasks():
        return p["task_id"]
    rl = load_rl_cfg(p["base_task"])
    dims = tuple(int(d) for d in p["hidden_dims"])
    rl.actor = dataclasses.replace(rl.actor, hidden_dims=dims)
    rl.critic = dataclasses.replace(rl.critic, hidden_dims=dims)
    rl.save_interval = int(p["save_interval"])
    register_mjlab_task(task_id=p["task_id"], env_cfg=make_env_cfg(cfg),
                        play_env_cfg=make_env_cfg(cfg), rl_cfg=rl,
                        runner_cls=MjlabOnPolicyRunner)
    return p["task_id"]


# --- evaluation ----------------------------------------------------------------------------

def load_course(path: str, cfg) -> tuple[list, float]:
    """(segments, lane length) for a course YAML: {course: [features...]}. The lane is
    the configured length or longer if the course needs it."""
    import yaml

    spec = yaml.safe_load(open(path))["course"]
    need = cp.total_length(cp.from_spec(spec))
    lane_x = max(float(cfg["parkour"]["lane_m"][0]), need)
    return cp.from_spec(spec, lane_x), lane_x


def evaluate(cfg, checkpoint: str, num_envs: int, device: str, course: str | None = None,
             video: str | None = None, levels: int | None = None) -> dict:
    """Roll out one episode per env. With `course` (a YAML file) every env runs that
    course from the start line, facing it; otherwise envs are spread evenly over `levels`
    random-course difficulty levels. success = reached the goal without falling.
    Metrics count each env's first episode only. The video follows env 0 for the whole
    evaluation window, so after a fall it shows the next attempt."""
    from dataclasses import asdict

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_rl_cfg

    p = cfg["parkour"]
    task = register(cfg)
    if course:
        segs, lane_x = load_course(course, cfg)
        e = make_env_cfg(cfg, segments=segs, lane_x=lane_x)
        e.commands["goal"].speed_range = (float(p["eval_speed_mps"]),) * 2
        # Start at the start line facing the course (training spawns at any heading).
        yaw = float(p["eval_spawn_yaw_rad"])
        e.events["reset_base"].params["pose_range"]["yaw"] = (-yaw, yaw)
    else:
        e = make_env_cfg(cfg, num_rows=levels)
    e.curriculum = {}
    e.rewards.pop("style", None)          # a training signal only
    e.metrics = {k: v for k, v in e.metrics.items() if not k.startswith("style_")}
    e.commands["goal"].goal_y_jitter = 0.0
    e.terminations.pop("out_of_terrain_bounds", None)
    e.episode_length_s = float(p["eval_episode_s"])
    e.scene.num_envs = num_envs
    if video:
        e.viewer.width, e.viewer.height = cfg["train_filter"]["filter"]["video_size"]
        e.viewer.max_extra_envs = 0          # only the followed robot, no overlapping ghosts
    env = ManagerBasedRlEnv(cfg=e, device=device, render_mode="rgb_array" if video else None)
    terrain = env.scene.terrain
    rows, cols = terrain.terrain_origins.shape[:2]
    ids = torch.arange(num_envs, device=device)
    terrain.terrain_levels[:] = ids % rows
    terrain.terrain_types[:] = (ids // rows) % cols
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels,
                                                     terrain.terrain_types]
    env.reset()
    agent = load_rl_cfg(task)
    wenv = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner = MjlabOnPolicyRunner(wenv, asdict(agent), device=device)
    runner.load(checkpoint, map_location=device)
    policy = runner.get_inference_policy(device=device)
    goal: CourseGoalCommand = env.command_manager.get_term("goal")
    level = terrain.terrain_levels.clone()

    done = torch.zeros(num_envs, dtype=torch.bool, device=device)
    fell = torch.zeros_like(done)
    reached = torch.zeros_like(done)
    t_reach = torch.full((num_envs,), float("nan"), device=device)
    progress = torch.zeros(num_envs, device=device)
    frames = []
    obs = wenv.get_observations()
    dt = env.step_dt
    for step in range(int(env.max_episode_length) + 5):
        with torch.no_grad():
            obs, _, dones, _ = wenv.step(policy(obs))
        dones = dones.bool()
        live = ~done & ~dones                  # goal state is reset for envs that ended
        new_reach = live & goal.reached & ~reached
        t_reach = torch.where(new_reach, torch.full_like(t_reach, (step + 1) * dt), t_reach)
        reached |= live & goal.reached
        progress = torch.where(live, torch.maximum(progress, goal.progress), progress)
        if video:            # the whole evaluation window: after a fall env 0 restarts
            frames.append(env.render())
        new = dones & ~done
        fell |= new & env.termination_manager.terminated
        done |= new
        if done.all() and not video:
            break
    env.close()
    if video and frames:
        import imageio.v2 as iio

        iio.mimwrite(video, frames, fps=round(1 / dt), macro_block_size=1)
    ok_ = reached & ~fell
    out = dict(rollouts=num_envs, success_rate=ok_.float().mean().item(),
               reach_rate=reached.float().mean().item(), fall_rate=fell.float().mean().item(),
               progress=progress.mean().item(),
               time_to_goal_s=(t_reach[reached].mean().item() if reached.any() else None))
    if not course:
        out["per_level"] = {int(lv): dict(n=int((level == lv).sum()),
                                          success=ok_[level == lv].float().mean().item(),
                                          progress=progress[level == lv].mean().item())
                            for lv in level.unique().tolist()}
    return out
