"""Milestone 6b: WCT motion style reward (adversarial motion prior) (`mjlab` env).

A discriminator learns to tell one-step transitions of the retargeted WCT clips (the
"expert") from transitions of the policy; the policy is rewarded for transitions it
cannot tell apart (least-squares GAN form of AMP, Peng et al. 2021):

    r_style = max(0, 1 - 0.25 (D(s, s') - 1)^2)

mjlab's PPO (rsl_rl) has no AMP, so the discriminator lives in the env as a class-based
reward term: every step it scores the current transitions and stores them in a replay
buffer; every `update_every` steps (one PPO rollout) it takes a few gradient steps.

Features (per frame, all in the pelvis frame, so they do not depend on heading, position
or terrain height): joint positions and velocities, pelvis linear and angular velocity,
gravity direction, and feet/hand positions relative to the pelvis.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

KEY_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link",
              "left_wrist_yaw_link", "right_wrist_yaw_link")


def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate v by the inverse of unit quaternion q (w, x, y, z); broadcasts."""
    w, u = q[..., :1], -q[..., 1:]
    t = 2.0 * torch.cross(u, v, dim=-1)
    return v + w * t + torch.cross(u, t, dim=-1)


def frame_features(joint_pos, joint_vel, root_pos, root_quat, root_lin_vel, root_ang_vel,
                   key_pos) -> torch.Tensor:
    """(N, F) features from world-frame quantities; key_pos is (N, K, 3)."""
    grav = torch.zeros_like(root_lin_vel)
    grav[..., 2] = -1.0
    key = quat_rotate_inverse(root_quat.unsqueeze(1), key_pos - root_pos.unsqueeze(1))
    return torch.cat([joint_pos, joint_vel,
                      quat_rotate_inverse(root_quat, root_lin_vel),
                      quat_rotate_inverse(root_quat, root_ang_vel),
                      quat_rotate_inverse(root_quat, grav),
                      key.flatten(1)], dim=-1)


def clip_transitions(npz: dict, key_ids: list[int]) -> torch.Tensor:
    """(T-1, 2F) consecutive-frame feature pairs of one mjlab motion npz."""
    t = {k: torch.as_tensor(np.asarray(npz[k]), dtype=torch.float32)
         for k in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w",
                   "body_ang_vel_w")}
    f = frame_features(t["joint_pos"], t["joint_vel"], t["body_pos_w"][:, 0],
                       t["body_quat_w"][:, 0], t["body_lin_vel_w"][:, 0],
                       t["body_ang_vel_w"][:, 0], t["body_pos_w"][:, key_ids])
    return torch.cat([f[:-1], f[1:]], dim=-1)


def expert_transitions(paths: list[str], key_ids: list[int], fps: float) -> torch.Tensor:
    out = []
    for p in paths:
        d = np.load(p)
        if abs(float(d["fps"][0]) - fps) > 1e-6:
            raise ValueError(f"{p}: {float(d['fps'][0])} fps, the policy runs at {fps}")
        out.append(clip_transitions(d, key_ids))
    return torch.cat(out)


class Discriminator(nn.Module):
    def __init__(self, dim: int, hidden: tuple[int, ...], mean: torch.Tensor,
                 std: torch.Tensor):
        super().__init__()
        layers, d = [], dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(d, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body((x - self.mean) / self.std)).squeeze(-1)


def style_reward(d: torch.Tensor) -> torch.Tensor:
    return torch.clamp(1.0 - 0.25 * torch.square(d - 1.0), min=0.0)


def discriminator_loss(disc: Discriminator, expert: torch.Tensor, policy: torch.Tensor,
                       gp_weight: float) -> tuple[torch.Tensor, dict]:
    """LSGAN targets +1 (expert) / -1 (policy), gradient penalty on expert samples."""
    expert = expert.clone().requires_grad_(True)
    de, dp = disc(expert), disc(policy)
    loss = torch.mean(torch.square(de - 1.0)) + torch.mean(torch.square(dp + 1.0))
    g, = torch.autograd.grad(de.sum(), expert, create_graph=True)
    gp = torch.mean(torch.sum(torch.square(g), dim=-1))
    stats = dict(loss=loss.detach(), gp=gp.detach(), d_expert=de.detach().mean(),
                 d_policy=dp.detach().mean())
    return loss + gp_weight * gp, stats


class StyleReward:
    """Class-based mjlab reward term (see module docstring). params: motions (a .txt list
    of npz files), hidden, lr, buffer_size, batch_size, updates, update_every,
    gp_weight."""

    def __init__(self, cfg, env):
        from .multi_motion import motion_paths

        p = cfg.params
        self.env = env
        self.robot = env.scene["robot"]
        names = list(self.robot.body_names)
        self.key_ids = [names.index(b) for b in KEY_BODIES]
        dev = env.device
        paths = motion_paths(p["motions"])
        self.expert = expert_transitions(paths, self.key_ids, 1.0 / env.step_dt).to(dev)
        dim = self.expert.shape[1]
        mean, std = self.expert.mean(0), self.expert.std(0).clamp(min=1e-2)
        self.disc = Discriminator(dim, tuple(p["hidden"]), mean, std).to(dev)
        self.opt = torch.optim.Adam(self.disc.parameters(), lr=float(p["lr"]))
        n = env.num_envs
        self.buf = torch.zeros(int(p["buffer_size"]), dim, device=dev)
        self.ptr, self.filled = 0, 0
        self.prev = torch.zeros(n, dim // 2, device=dev)
        self.valid = torch.zeros(n, dtype=torch.bool, device=dev)
        self.steps = 0
        self.p = p
        self.stats = {k: torch.zeros((), device=dev) for k in
                      ("loss", "gp", "d_expert", "d_policy")}
        print(f"[style] {len(paths)} clips, {len(self.expert)} expert transitions, "
              f"feature dim {dim}")

    def _features(self) -> torch.Tensor:
        d = self.robot.data
        return frame_features(d.joint_pos, d.joint_vel, d.root_link_pos_w, d.root_link_quat_w,
                              d.root_link_lin_vel_w, d.root_link_ang_vel_w,
                              d.body_link_pos_w[:, self.key_ids])

    def _store(self, x: torch.Tensor) -> None:
        k = min(len(x), len(self.buf))
        x = x[torch.randperm(len(x), device=x.device)[:k]]
        idx = (self.ptr + torch.arange(k, device=x.device)) % len(self.buf)
        self.buf[idx] = x
        self.ptr = int((self.ptr + k) % len(self.buf))
        self.filled = min(self.filled + k, len(self.buf))

    def _update(self) -> None:
        b = int(self.p["batch_size"])
        st = self.stats
        for _ in range(int(self.p["updates"])):
            e = self.expert[torch.randint(len(self.expert), (b,), device=self.expert.device)]
            q = self.buf[torch.randint(self.filled, (b,), device=self.buf.device)]
            total, st = discriminator_loss(self.disc, e, q, float(self.p["gp_weight"]))
            self.opt.zero_grad()
            total.backward()
            self.opt.step()
        self.stats = st

    def __call__(self, env, **kwargs) -> torch.Tensor:
        del env, kwargs
        cur = self._features()
        trans = torch.cat([self.prev, cur], dim=-1)
        ok = self.valid.clone()
        with torch.no_grad():
            r = torch.where(ok, style_reward(self.disc(trans)), torch.zeros_like(ok, dtype=torch.float))
        # Rollouts run under torch.inference_mode(); the buffer and the discriminator
        # are ordinary tensors, so writes and training happen outside it.
        with torch.inference_mode(False):
            if ok.any():
                self._store(trans[ok].clone())
            self.steps += 1
            if self.steps % int(self.p["update_every"]) == 0 and self.filled >= int(
                    self.p["batch_size"]):
                with torch.enable_grad():
                    self._update()
        self.prev = cur.clone()
        self.valid[:] = True
        return r

    def reset(self, env_ids=None) -> None:
        self.valid[slice(None) if env_ids is None else env_ids] = False

    def state_dict(self) -> dict:
        return dict(disc=self.disc.state_dict(), opt=self.opt.state_dict())


def style_stat(env, stat: str, term: str = "style") -> torch.Tensor:
    """Metrics term: the style discriminator's latest training statistic."""
    t: StyleReward = env.reward_manager.get_term_cfg(term).func
    return t.stats[stat].float().expand(env.num_envs).clone()


def save_list(paths: list[Path], out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(f"{p}\n" for p in paths))
    return out
