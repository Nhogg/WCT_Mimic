"""Multi-clip motion command for mjlab tracking (Milestone 4; runs in the `mjlab` env).

mjlab's MotionCommand tracks a single motion file. MultiMotionCommand loads many clips,
concatenates them along time, and keeps each environment inside the clip it started in:
when the reference reaches the clip's last frame, the env is resampled (same as the
single-clip wraparound). Adaptive sampling uses ~1 s bins that never span two clips, so
failure-weighted sampling picks hard sections across the whole dataset.

`motion_file` is either one .npz or a .txt listing one .npz path per line, so the
command is a drop-in replacement and the train CLI flag stays
`--env.commands.motion.motion-file`.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from mjlab.managers.event_manager import requires_model_fields
from mjlab.tasks.tracking.mdp.commands import MotionCommand, MotionCommandCfg, MotionLoader
from mjlab.utils.lab_api.math import matrix_from_quat, sample_uniform, subtract_frame_transforms

_FIELDS = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w",
           "body_ang_vel_w")


def motion_paths(motion_file: str) -> list[str]:
    p = Path(motion_file)
    if p.suffix == ".txt":
        base = p.parent
        lines = [ln.strip() for ln in p.read_text().splitlines()]
        return [str((base / ln) if not Path(ln).is_absolute() else ln)
                for ln in lines if ln and not ln.startswith("#")]
    return [motion_file]


def clip_bins(lengths: list[int], bin_frames: int) -> tuple[list[int], list[int]]:
    """Split each clip [start, start+len-1) into bins of ~bin_frames frames.

    The last frame of a clip is excluded because the reference must advance at least one
    step before wrapping. Returns (bin_start, bin_end) in concatenated frame indices."""
    starts, ends, s = [], [], 0
    for n in lengths:
        usable = max(n - 1, 1)
        nb = max(1, math.ceil(usable / bin_frames))
        edges = [s + round(i * usable / nb) for i in range(nb + 1)]
        starts += edges[:-1]
        ends += edges[1:]
        s += n
    return starts, ends


def allocate_envs(num_envs: int, lengths: list[int]) -> np.ndarray:
    """Fixed env -> clip assignment, proportional to clip length (largest remainder),
    at least one env per clip when there are enough envs. Envs of a clip are contiguous."""
    n = len(lengths)
    w = np.asarray(lengths, float)
    if num_envs < n:
        return np.arange(num_envs) % n
    base = np.ones(n, int)
    share = (num_envs - n) * w / w.sum()
    base += np.floor(share).astype(int)
    rest = num_envs - base.sum()
    base[np.argsort(-(share - np.floor(share)), kind="stable")[:rest]] += 1
    return np.repeat(np.arange(n), base)


def load_boxes(paths: list[str], obstacles_dir: str, max_boxes: int) -> np.ndarray:
    """(clips, max_boxes, 6) array of box center xyz + half sizes xyz, zeros = no box.
    Reads <obstacles_dir>/<clip stem>.json written by wct2g1.obstacles."""
    out = np.zeros((len(paths), max(max_boxes, 0), 6), np.float32)
    if not obstacles_dir or max_boxes <= 0:
        return out
    for i, p in enumerate(paths):
        f = Path(obstacles_dir) / f"{Path(p).stem}.json"
        if not f.exists():
            continue
        boxes = json.loads(f.read_text())["boxes"]
        if len(boxes) > max_boxes:
            raise ValueError(f"{f}: {len(boxes)} boxes > max_boxes {max_boxes}")
        for j, b in enumerate(boxes):
            out[i, j] = [*b["center"], *b["half_size"]]
    return out


class _ConcatMotion:
    """Same attributes as MotionLoader, concatenated over clips."""

    def __init__(self, loaders: list[MotionLoader]):
        for f in _FIELDS:
            setattr(self, f, torch.cat([getattr(m, f) for m in loaders], dim=0))
        self.time_step_total = int(self.joint_pos.shape[0])


class MultiMotionCommand(MotionCommand):
    cfg: MultiMotionCommandCfg

    def __init__(self, cfg: MultiMotionCommandCfg, env):
        paths = motion_paths(cfg.motion_file)
        if not paths:
            raise ValueError(f"no motion files in {cfg.motion_file}")
        # "start" = every env begins at the first frame of a clip (clip = env index % n).
        self._from_start = cfg.sampling_mode == "start"
        if self._from_start:
            cfg.sampling_mode = "uniform"
        single = cfg.motion_file
        cfg.motion_file = paths[0]          # base __init__ loads one file; replaced below
        super().__init__(cfg, env)
        cfg.motion_file = single

        loaders = [self.motion] + [MotionLoader(p, self.body_indexes, device=self.device)
                                   for p in paths[1:]]
        lengths = [m.time_step_total for m in loaders]
        self.motion = _ConcatMotion(loaders)
        self.motion_paths = paths
        dev = self.device
        self.clip_start = torch.tensor([0] + lengths[:-1], device=dev).cumsum(0)
        self.clip_end = self.clip_start + torch.tensor(lengths, device=dev)
        frame_clip = torch.repeat_interleave(torch.arange(len(lengths), device=dev),
                                             torch.tensor(lengths, device=dev))
        self.frame_clip = frame_clip

        bin_frames = max(1, round(cfg.bin_s / env.step_dt))
        bs, be = clip_bins(lengths, bin_frames)
        self.bin_start = torch.tensor(bs, device=dev)
        self.bin_end = torch.tensor(be, device=dev)
        frame_bin = torch.zeros(self.motion.time_step_total, dtype=torch.long, device=dev)
        for i, (a, b) in enumerate(zip(bs, be)):
            frame_bin[a:b] = i
        last = self.clip_end - 1            # last frame of each clip -> its last bin
        frame_bin[last] = frame_bin[(last - 1).clamp(min=0)]
        self.frame_bin = frame_bin
        self.bin_count = len(bs)
        self.bin_failed_count = torch.zeros(self.bin_count, device=dev)
        self._current_bin_failed = torch.zeros(self.bin_count, device=dev)
        bin_clip = self.frame_clip[self.bin_start]
        self.clip_first_bin = torch.searchsorted(bin_clip, torch.arange(len(lengths), device=dev))
        self.clip_nbins = torch.bincount(bin_clip, minlength=len(lengths))
        if cfg.fixed_env_clips:
            self.env_clip = torch.as_tensor(allocate_envs(self.num_envs, lengths), device=dev)
        else:
            self.env_clip = torch.arange(self.num_envs, device=dev) % len(lengths)
        self.env_end = self.clip_end[self.env_clip]
        self.clip_boxes = torch.as_tensor(
            load_boxes(paths, cfg.obstacles_dir, cfg.max_boxes), device=dev)

    # -- sampling -------------------------------------------------------------------
    def _set_frames(self, env_ids: torch.Tensor, frames: torch.Tensor) -> None:
        self.time_steps[env_ids] = frames
        if not self.cfg.fixed_env_clips:
            self.env_clip[env_ids] = self.frame_clip[frames]
            self.env_end[env_ids] = self.clip_end[self.env_clip[env_ids]]

    def _uniform_sampling(self, env_ids: torch.Tensor):
        if self._from_start:
            self._set_frames(env_ids, self.clip_start[self.env_clip[env_ids]])
        elif self.cfg.fixed_env_clips:
            c = self.env_clip[env_ids]
            u = torch.rand(len(env_ids), device=self.device)
            b = self.clip_first_bin[c] + (u * self.clip_nbins[c]).long().clamp(
                max=self.clip_nbins[c] - 1)
            self._set_frames(env_ids, self._frame_in_bin(b))
        else:
            b = torch.randint(0, self.bin_count, (len(env_ids),), device=self.device)
            self._set_frames(env_ids, self._frame_in_bin(b))
        self.metrics["sampling_entropy"][:] = 1.0
        self.metrics["sampling_top1_prob"][:] = 1.0 / self.bin_count
        self.metrics["sampling_top1_bin"][:] = 0.5

    def _frame_in_bin(self, b: torch.Tensor) -> torch.Tensor:
        u = sample_uniform(0.0, 1.0, (len(b),), device=self.device)
        width = (self.bin_end[b] - self.bin_start[b]).float()
        return (self.bin_start[b] + (u * width).long().clamp(max=width.long() - 1)).long()

    def _adaptive_sampling(self, env_ids: torch.Tensor):
        failed = self._env.termination_manager.terminated[env_ids]
        if torch.any(failed):
            steps = self.time_steps[env_ids][failed].clamp(0, self.motion.time_step_total - 1)
            self._current_bin_failed[:] = torch.bincount(self.frame_bin[steps],
                                                         minlength=self.bin_count)
        p = self.bin_failed_count + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
        p = p / p.sum()
        if self.cfg.fixed_env_clips:
            # Sample among the bins of each env's own clip: invert the CDF restricted to
            # that clip's contiguous bin range.
            c = self.env_clip[env_ids]
            cdf = torch.cumsum(p, 0)
            first, last = self.clip_first_bin[c], self.clip_first_bin[c] + self.clip_nbins[c] - 1
            lo = torch.where(first > 0, cdf[(first - 1).clamp(min=0)], torch.zeros_like(cdf[first]))
            u = lo + torch.rand(len(env_ids), device=self.device) * (cdf[last] - lo)
            b = torch.searchsorted(cdf, u, right=True)
            b = torch.maximum(torch.minimum(b, last), first)
        else:
            b = torch.multinomial(p, len(env_ids), replacement=True)
        self._set_frames(env_ids, self._frame_in_bin(b))
        H = -(p * (p + 1e-12).log()).sum()
        self.metrics["sampling_entropy"][:] = H / math.log(self.bin_count) if self.bin_count > 1 else 1.0
        pmax, imax = p.max(dim=0)
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count

    # -- stepping (base version, but wraps at each env's clip end) ---------------------
    def _update_command(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            self.time_steps += 1
        else:
            self.time_steps[env_ids] += 1
        wrap_ids = torch.where(self.time_steps >= self.env_end)[0]
        if wrap_ids.numel() > 0:
            self._resample_command(wrap_ids)
        if self._pending_forward:
            self._pending_forward = False
            self._env.sim.forward()
        self.update_relative_body_poses()
        if env_ids is None and self.cfg.sampling_mode == "adaptive":
            self.bin_failed_count = (self.cfg.adaptive_alpha * self._current_bin_failed
                                     + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count)
            self._current_bin_failed.zero_()


def future_reference(env, command_name: str, offsets: tuple[int, ...]) -> torch.Tensor:
    """Look-ahead observation: for each offset k (control steps), the reference joint
    positions and the reference anchor pose k steps ahead, expressed in the robot's
    current anchor frame (position + first two rotation-matrix columns). Clamped to the
    last frame of the env's clip. The single-clip task only sees the current frame, which
    is enough to memorise one clip but not to anticipate across many."""
    cmd: MultiMotionCommand = env.command_manager.get_term(command_name)
    m = cmd.motion
    a = cmd.motion_anchor_body_index
    origins = env.scene.env_origins
    out = []
    for k in offsets:
        idx = torch.minimum(cmd.time_steps + k, cmd.env_end - 1)
        pos, ori = subtract_frame_transforms(cmd.robot_anchor_pos_w, cmd.robot_anchor_quat_w,
                                             m.body_pos_w[idx, a] + origins,
                                             m.body_quat_w[idx, a])
        rot = matrix_from_quat(ori)[..., :2].reshape(env.num_envs, -1)
        out += [m.joint_pos[idx], pos, rot]
    return torch.cat(out, dim=-1)


HIDDEN_Z = -5.0   # unused obstacle boxes are parked below the floor


@requires_model_fields("geom_pos", "geom_size")
def place_obstacles(env, env_ids: torch.Tensor | None, command_name: str = "motion",
                    entity_name: str = "obstacles") -> None:
    """Startup event: size and place each env's clip boxes (wct2g1.obstacles) in its own
    world; unused boxes are parked under the floor. Needs fixed_env_clips, since an
    env's boxes never change after startup. The EventManager recomputes the collision
    bounds afterwards because geom_size changed."""
    cmd: MultiMotionCommand = env.command_manager.get_term(command_name)
    if not cmd.cfg.fixed_env_clips:
        raise ValueError("place_obstacles needs fixed_env_clips=True")
    ent = env.scene[entity_name]
    gids = ent.indexing.geom_ids.long()
    ids = (torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids).long()
    boxes = cmd.clip_boxes[cmd.env_clip[ids]]                    # (E, K, 6)
    k = min(boxes.shape[1], len(gids))
    present = (boxes[..., 3:] > 0).all(-1)[:, :k]
    pos = boxes[:, :k, :3] + env.scene.env_origins[ids][:, None, :]
    pos[..., 2] = torch.where(present, pos[..., 2], torch.full_like(pos[..., 2], HIDDEN_Z))
    size = torch.where(present[..., None], boxes[:, :k, 3:],
                       torch.full_like(boxes[:, :k, 3:], 0.01))
    eg, gg = torch.meshgrid(ids, gids[:k], indexing="ij")
    env.sim.model.geom_pos[eg, gg] = pos
    env.sim.model.geom_size[eg, gg] = size


def obstacle_spec(max_boxes: int, rgba=(0.55, 0.45, 0.35, 1.0)):
    """Static entity holding `max_boxes` box geoms (terrain collision group 0)."""
    import mujoco

    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="course")
    for i in range(max_boxes):
        body.add_geom(name=f"box{i}", type=mujoco.mjtGeom.mjGEOM_BOX, size=(0.01,) * 3,
                      pos=(0.0, 0.0, HIDDEN_Z), rgba=rgba, contype=1, conaffinity=1,
                      group=0)
    return spec


@dataclass(kw_only=True)
class MultiMotionCommandCfg(MotionCommandCfg):
    bin_s: float = 1.0     # adaptive-sampling bin length (bins never span two clips)
    fixed_env_clips: bool = False   # each env tracks one clip for the whole run
    obstacles_dir: str = ""         # <clip>.json boxes from wct2g1.obstacles
    max_boxes: int = 0

    def build(self, env) -> MultiMotionCommand:
        return MultiMotionCommand(self, env)
