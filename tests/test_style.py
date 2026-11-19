import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mjlab")   # runs in the `mjlab` env only

from wct2g1 import style  # noqa: E402


def _yaw_quat(a):
    return torch.tensor([math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)])


def test_quat_rotate_inverse():
    q = _yaw_quat(math.pi / 2)          # +90 deg about z: world +y is body +x
    v = torch.tensor([0.0, 1.0, 0.0])
    assert torch.allclose(style.quat_rotate_inverse(q, v), torch.tensor([1.0, 0, 0]), atol=1e-6)


def _state(n=5, k=4, yaw=0.0, shift=(0.0, 0.0, 0.0)):
    g = torch.Generator().manual_seed(0)
    jp, jv = torch.randn(n, 29, generator=g), torch.randn(n, 29, generator=g)
    pos = torch.randn(n, 3, generator=g)
    lin, ang = torch.randn(n, 3, generator=g), torch.randn(n, 3, generator=g)
    key = pos.unsqueeze(1) + torch.randn(n, k, 3, generator=g)
    q0 = torch.nn.functional.normalize(torch.randn(n, 4, generator=g), dim=-1)
    # Rotate the whole scene by `yaw` about z and translate it.
    c, s = math.cos(yaw), math.sin(yaw)
    R = torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    qy = _yaw_quat(yaw).expand(n, 4)
    w1, v1 = qy[:, :1], qy[:, 1:]
    w2, v2 = q0[:, :1], q0[:, 1:]
    q = torch.cat([w1 * w2 - (v1 * v2).sum(-1, keepdim=True),
                   w1 * v2 + w2 * v1 + torch.cross(v1, v2, dim=-1)], dim=-1)
    t = torch.tensor(shift)
    return (jp, jv, pos @ R.T + t, q, lin @ R.T, ang @ R.T, key @ R.T + t)


def test_features_ignore_heading_and_position():
    a = style.frame_features(*_state())
    b = style.frame_features(*_state(yaw=1.3, shift=(4.0, -2.0, 0.5)))
    assert a.shape == (5, 29 + 29 + 3 + 3 + 3 + 12)
    assert torch.allclose(a, b, atol=1e-5)


def test_discriminator_separates_and_reward_tracks_it():
    torch.manual_seed(0)
    expert, policy = torch.randn(512, 6) + 2.0, torch.randn(512, 6) - 2.0
    d = style.Discriminator(6, (32, 32), torch.zeros(6), torch.ones(6))
    opt = torch.optim.Adam(d.parameters(), lr=1e-2)
    for _ in range(200):
        loss, st = style.discriminator_loss(d, expert, policy, gp_weight=0.1)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert st["d_expert"] > 0.5 and st["d_policy"] < -0.5
    with torch.no_grad():
        assert style.style_reward(d(expert)).mean() > style.style_reward(d(policy)).mean() + 0.5


def _fake_env(n=8):
    data = SimpleNamespace(
        joint_pos=torch.zeros(n, 29), joint_vel=torch.zeros(n, 29),
        root_link_pos_w=torch.zeros(n, 3), root_link_quat_w=torch.tensor([1.0, 0, 0, 0]).repeat(n, 1),
        root_link_lin_vel_w=torch.zeros(n, 3), root_link_ang_vel_w=torch.zeros(n, 3),
        body_link_pos_w=torch.zeros(n, 30, 3))
    robot = SimpleNamespace(data=data, body_names=[f"b{i}" for i in range(30)])
    for i, b in zip((6, 12, 22, 29), style.KEY_BODIES):
        robot.body_names[i] = b
    return SimpleNamespace(scene={"robot": robot}, device="cpu", num_envs=n, step_dt=0.02)


def test_style_reward_term_runs_under_inference_mode(tmp_path):
    import numpy as np
    T = 40
    np.savez(tmp_path / "a.npz", fps=np.array([50.0]), joint_pos=np.random.randn(T, 29),
             joint_vel=np.random.randn(T, 29), body_pos_w=np.random.randn(T, 30, 3),
             body_quat_w=np.tile([1.0, 0, 0, 0], (T, 30, 1)),
             body_lin_vel_w=np.random.randn(T, 30, 3), body_ang_vel_w=np.random.randn(T, 30, 3))
    (tmp_path / "m.txt").write_text("a.npz\n")
    env = _fake_env()
    p = dict(motions=str(tmp_path / "m.txt"), hidden=[16], lr=1e-3, buffer_size=64,
             batch_size=8, updates=2, update_every=3, gp_weight=1.0)
    term = style.StyleReward(SimpleNamespace(params=p), env)
    w0 = term.disc.head.weight.detach().clone()
    with torch.inference_mode():
        r0 = term(env)
        assert (r0 == 0).all()                 # no previous frame yet
        term.reset(torch.tensor([0, 1]))
        for _ in range(6):
            env.scene["robot"].data.joint_pos += 0.1
            r = term(env)
        term.reset(None)
    assert r.shape == (8,) and (r >= 0).all() and (r <= 1).all()
    assert term.filled == 6 + 5 * 8            # envs 0 and 1 were reset before the loop
    assert not torch.equal(w0, term.disc.head.weight)   # it trained
