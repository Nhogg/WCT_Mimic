import numpy as np
from scipy.spatial.transform import Rotation as R

from wct2g1 import clean


def test_timestamps_slowmo_and_realtime():
    fi = np.array([10, 12, 13, 15])
    assert np.allclose(clean.timestamps(fi, 50.0, None), [0, 0.04, 0.06, 0.10])
    assert np.allclose(clean.timestamps(fi, 50.0, 0.25), [0, 0.01, 0.015, 0.025])
    assert np.allclose(clean.timestamps(fi, 50.0, float("nan")), [0, 0.04, 0.06, 0.10])


def test_resample_30_to_50_lerp_and_slerp():
    t = np.arange(31) / 30.0                     # 1 s at 30 fps
    ang = np.linspace(0, np.pi / 2, 31)          # yaw 0 -> 90 deg
    rot = np.stack([np.zeros(31), np.zeros(31), ang], -1)[:, None]
    tr = np.stack([t * 3.0, np.zeros(31), np.ones(31)], -1)   # 3 m/s along x
    grid, r2, t2 = clean.resample(t, rot, tr, 50.0)
    assert len(grid) == 51 and np.isclose(grid[-1], 1.0)
    assert np.allclose(t2[:, 0], grid * 3.0)
    yaw = R.from_rotvec(r2[:, 0]).as_euler("xyz")[:, 2]
    assert np.allclose(yaw, grid * np.pi / 2, atol=1e-6)


def test_6d_roundtrip_and_lowpass_keeps_slow_signal():
    rv = R.random(40, random_state=0).as_rotvec().reshape(20, 2, 3)
    assert np.allclose(R.from_rotvec(clean.rot_from_6d(clean.rot_to_6d(rv)).reshape(-1, 3)).as_matrix(),
                       R.from_rotvec(rv.reshape(-1, 3)).as_matrix(), atol=1e-6)
    t = np.arange(200) / 50
    slow = np.sin(2 * np.pi * 1.0 * t)
    noisy = slow + 0.2 * np.sin(2 * np.pi * 20 * t)
    out = clean.lowpass(noisy, 50, 8.0, 4)
    assert np.abs(out - slow)[20:-20].max() < 0.02


def test_yup_to_zup_moves_pelvis_like_a_point():
    j0 = np.array([0.0, 0.9, 0.05])              # rest pelvis offset
    go = R.from_euler("y", [[0.0], [0.7]]).as_rotvec()
    tr = np.array([[1.0, 0.1, 2.0], [1.5, 0.2, 2.5]])
    go2, tr2 = clean.yup_to_zup(go, tr, j0)
    pelvis_y = tr + j0
    assert np.allclose(tr2 + j0, pelvis_y @ clean.YUP_TO_ZUP.T)
    # Y-up "up" axis of the root maps to +Z
    up = R.from_rotvec(go2).apply([0, 1, 0])
    assert np.allclose(up[0], [0, 0, 1])


def _walk_feet(T=100, fps=50):
    """Left foot planted for frames 0-49, swinging 50-99; right the opposite."""
    pos = np.zeros((T, 2, 3))
    z = np.zeros((T, 2))
    t = np.arange(T)
    pos[:, 0, 0] = np.where(t < 50, 0.0, (t - 50) * 0.04)
    pos[:, 1, 0] = np.where(t < 50, t * 0.04, 2.0)
    z[:, 0] = np.where(t < 50, 0.01, 0.15)
    z[:, 1] = np.where(t < 50, 0.15, 0.01)
    return pos, z


def test_detect_contact():
    pos, z = _walk_feet()
    c = clean.detect_contact(z, pos, 50, height_m=0.05, speed_mps=0.6, min_frames=3)
    assert c[:50, 0].all() and not c[50:, 0].any()
    assert c[51:, 1].all() and not c[:50, 1].any()


def test_skate_correction_cancels_planted_drift():
    pos, z = _walk_feet()
    c = np.zeros((100, 2), bool)
    c[:, 0] = True
    pos[:, 0, 0] = np.arange(100) * 0.01          # planted foot slides 0.5 m/s
    z[:, 0] = 0.01                                # 1 cm above the floor throughout
    corr = clean.skate_correction(pos, z, c, 50, 4.0)
    fixed = pos[:, 0, 0] + corr[:, 0]
    assert np.allclose(fixed, 0.0)
    assert np.allclose(corr[:, 2], -0.01, atol=1e-3)   # sole pulled down onto the floor


def test_cut_points_respect_bounds():
    fps = 50
    speed = np.ones(25 * fps)
    speed[[9 * fps, 17 * fps]] = 0.0               # natural pauses
    segs = clean.cut_points(speed, fps, 2.0, 10.0)
    assert segs[0] == (0, 9 * fps) and segs[1] == (9 * fps, 17 * fps) and segs[-1][1] == 25 * fps
    assert all(2 * fps <= b - a <= 10 * fps for a, b in segs)
    assert clean.cut_points(np.ones(fps), fps, 2.0, 10.0) == []


def test_export_betas_pads_with_zeros():
    from wct2g1.clean import export_betas

    b = export_betas(np.arange(1, 11, dtype=float), 10, 16)
    assert b.shape == (16,) and np.allclose(b[:10], np.arange(1, 11)) and not b[10:].any()


def test_rolling_floor_follows_slow_drift():
    fps = 50
    t = np.arange(5 * fps) / fps
    drift = 0.06 * t                                    # 6 cm/s vertical drift
    gait = 0.15 * np.abs(np.sin(np.pi * 2 * t))         # feet lift and return to the floor
    fl = clean.rolling_floor(drift + gait, fps, 1.0, 5)
    mid = slice(fps, 4 * fps)
    assert np.max(np.abs(fl[mid] - drift[mid])) < 0.05


def test_runs_and_obstacle_mask():
    assert clean.runs(np.array([0, 0, 1, 1, 1, 0], bool)) == [(0, 2, False), (2, 5, True), (5, 6, False)]
    fps, T = 50, 200
    j = np.zeros((T, 24, 3))
    j[:, clean.L_WRIST, 2] = j[:, clean.R_WRIST, 2] = 1.0
    pelvis = np.full(T, 0.9)
    pelvis[150:] = 1.5                                  # climbs onto something
    c = dict(hand_height_m=0.25, pelvis_rise_m=0.3, air_height_m=0.1, airborne_s=0.5, margin_s=0.2)
    m = clean.obstacle_mask(j, np.zeros(T), pelvis, 0.9, fps, c)
    assert not m[:140].any() and m[141:].all()          # 150 dilated back by 10 frames
