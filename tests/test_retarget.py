import numpy as np
from scipy.spatial.transform import Rotation as R

from wct2g1.retarget import limit_fraction, qpos_to_rows, read_csv_as_qpos, write_csv


def _qpos(T=5):
    yaw = np.linspace(0, 1.0, T)
    quat_xyzw = R.from_euler("z", yaw[:, None]).as_quat()   # scipy is xyzw
    q = np.zeros((T, 36))
    q[:, :3] = np.c_[np.arange(T) * 0.1, np.zeros(T), np.full(T, 0.78)]
    q[:, 3:7] = quat_xyzw[:, [3, 0, 1, 2]]                  # MuJoCo is wxyz
    q[:, 7:] = np.arange(29) * 0.01
    return q, quat_xyzw


def test_csv_column_order_and_xyzw(tmp_path):
    q, quat_xyzw = _qpos()
    rows = qpos_to_rows(q)
    assert rows.shape == (5, 36)
    assert np.allclose(rows[:, :3], q[:, :3])
    assert np.allclose(rows[:, 3:7], quat_xyzw)              # xyzw, as mjlab's csv_to_npz expects
    assert np.allclose(rows[:, 7:], q[:, 7:])
    p = tmp_path / "m.csv"
    write_csv(p, rows)
    assert np.allclose(read_csv_as_qpos(p), q, atol=1e-6)    # round trip through the file


def test_quaternion_sign_continuity():
    q, _ = _qpos()
    q[2, 3:7] *= -1                                          # same rotation, flipped sign
    rows = qpos_to_rows(q)
    assert np.all(np.sum(rows[1:, 3:7] * rows[:-1, 3:7], axis=1) > 0)


def test_limit_fraction():
    lo, hi = np.full(29, -1.0), np.full(29, 1.0)
    d = np.zeros((10, 29))
    d[:3, 5] = 0.995
    assert limit_fraction(d, lo, hi, 0.02) == 0.3


def test_saturated_joints_lists_worst_first():
    from wct2g1.retarget import saturated_joints

    lo, hi = np.full(3, -1.0), np.full(3, 1.0)
    d = np.zeros((10, 3))
    d[:2, 0] = 1.0
    d[:6, 2] = -1.0
    assert saturated_joints(d, lo, hi, 0.02, ["a_joint", "b_joint", "c_joint"]) == "c:0.60,a:0.20"


def test_train_filter_decide_uses_configured_mpjpe_frame():
    from wct2g1.train_filter import decide

    a = dict(max_fall_rate=0.1, max_mpjpe_m=0.1, min_completion=0.95, mpjpe_frame="root_relative")
    m = dict(fall_rate=0.0, mpjpe_m=0.15, mpjpe_rel_m=0.05, completion=1.0)
    assert decide(m, a) == "ok"
    assert decide(m, {**a, "mpjpe_frame": "global"}) == "rejected:mpjpe"
    assert decide({**m, "fall_rate": 0.5}, a) == "rejected:falls"
