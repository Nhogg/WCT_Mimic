"""Entry points run in the `mjlab` env (one process per clip, launched by train_filter).

  python -m wct2g1.mjlab_launch train  --config C <mjlab train CLI args...>
  python -m wct2g1.mjlab_launch eval   --config C --motion-file M --checkpoint P
                                       --num-envs N --output-json J
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .config import load_config
from .sim_task import register


def train(cfg, argv: list[str]) -> None:
    from mjlab.scripts import train as mj_train

    task = register(cfg)
    sys.argv = ["train", task, *argv]
    mj_train.main()


def evaluate(cfg, motion_file: str, checkpoint: str, num_envs: int, device: str,
             video: str | None = None) -> dict:
    """Roll out `num_envs` episodes from the clip start; fall rate, MPJPE (world-frame and
    root-relative), completion. Optionally records env 0 to `video` (mp4).
    Follows mjlab.tasks.tracking.scripts.evaluate but loads a local checkpoint."""
    from types import SimpleNamespace
    from typing import cast

    import numpy as np
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.tasks.tracking.mdp.commands import MotionCommand
    from mjlab.tasks.tracking.mdp.metrics import compute_mpkpe, compute_root_relative_mpkpe

    task = register(cfg)
    env_cfg = load_env_cfg(task, play=False)
    agent_cfg = load_rl_cfg(task)
    mc = env_cfg.commands["motion"]
    mc.motion_file = motion_file
    mc.sampling_mode = "start"
    mc.pose_range = {}
    mc.velocity_range = {}
    n_frames = int(np.load(motion_file)["joint_pos"].shape[0])
    fps = float(np.load(motion_file)["fps"][0])
    # Episode = exactly one pass through the clip; a time-out means it was completed.
    env_cfg.episode_length_s = (n_frames - 1) / fps
    env_cfg.scene.num_envs = num_envs
    if video:
        env_cfg.viewer.width, env_cfg.viewer.height = cfg["train_filter"]["filter"]["video_size"]

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device,
                            render_mode="rgb_array" if video else None)
    frames = []
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = (load_runner_cls(task) or MjlabOnPolicyRunner)(env, asdict(agent_cfg), device=device)
    runner.load(checkpoint, map_location=device)
    policy = runner.get_inference_policy(device=device)
    command = cast(MotionCommand, env.unwrapped.command_manager.get_term("motion"))

    done = torch.zeros(num_envs, dtype=torch.bool, device=device)
    fell = torch.zeros(num_envs, dtype=torch.bool, device=device)
    steps_alive = torch.zeros(num_envs, device=device)
    mpkpe_sum = torch.zeros(num_envs, device=device)
    rel_sum = torch.zeros(num_envs, device=device)
    obs = env.get_observations()
    max_steps = int(env.unwrapped.max_episode_length) + 5
    for _ in range(max_steps):
        ref = SimpleNamespace(num_envs=command.num_envs, device=command.device, cfg=command.cfg,
                              body_pos_w=command.body_pos_w.clone(),
                              body_pos_relative_w=command.body_pos_relative_w.clone())
        with torch.no_grad():
            obs, _, dones, _ = env.step(policy(obs))
        ref.robot_body_pos_w = command.robot_body_pos_w
        active = ~done & ~dones.bool()
        mpkpe_sum += torch.where(active, compute_mpkpe(cast(MotionCommand, ref)), 0.0)
        rel_sum += torch.where(active, compute_root_relative_mpkpe(cast(MotionCommand, ref)), 0.0)
        if video and not done[0]:
            frames.append(env.unwrapped.render())
        steps_alive += active.float()
        new = dones.bool() & ~done
        term = env.unwrapped.termination_manager.terminated
        fell |= new & term
        done |= new
        if done.all():
            break
    env.close()
    if video and frames:
        import imageio.v2 as iio

        iio.mimwrite(video, frames, fps=round(fps), macro_block_size=1)
    total = max(int(env_cfg.episode_length_s * fps), 1)
    return {
        "fall_rate": fell.float().mean().item(),
        "mpjpe_m": (mpkpe_sum / steps_alive.clamp(min=1)).mean().item(),
        "mpjpe_rel_m": (rel_sum / steps_alive.clamp(min=1)).mean().item(),
        "completion": (steps_alive / total).clamp(max=1.0).mean().item(),
        "n_frames": n_frames,
        "rollouts": num_envs,
    }


def main() -> None:
    mode = sys.argv[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    if mode == "train":
        args, rest = ap.parse_known_args(sys.argv[2:])
        train(load_config(args.config), rest)
    elif mode == "eval":
        ap.add_argument("--motion-file", required=True)
        ap.add_argument("--checkpoint", required=True)
        ap.add_argument("--num-envs", type=int, default=20)
        ap.add_argument("--device", default="cuda:0")
        ap.add_argument("--output-json", required=True)
        ap.add_argument("--video", default=None)
        a = ap.parse_args(sys.argv[2:])
        m = evaluate(load_config(a.config), a.motion_file, a.checkpoint, a.num_envs, a.device,
                     a.video)
        Path(a.output_json).write_text(json.dumps(m, indent=2))
        print(json.dumps(m))
    else:
        raise SystemExit(f"unknown mode {mode}")


if __name__ == "__main__":
    main()
