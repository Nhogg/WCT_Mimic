import numpy as np

from wct2g1.reconstruct import resample_indices


def test_resample_uses_even_integer_stride():
    idx, fps = resample_indices(101, 50.0, 30)        # 50/30 -> stride 2 -> 25 fps
    assert fps == 25.0
    assert idx[0] == 0 and idx[-1] == 100 and len(idx) == 51
    assert set(np.diff(idx)) == {2}                    # evenly spaced, no 2,1,2,2,1 pattern
    assert resample_indices(120, 60.0, 30)[1] == 30.0


def test_resample_native_and_no_upsample():
    assert resample_indices(10, 50.0, "native")[1] == 50.0
    idx, fps = resample_indices(10, 25.0, 30)
    assert fps == 25.0 and list(idx) == list(range(10))


def test_rotation_from_homography_recovers_pan():
    from scipy.spatial.transform import Rotation as R

    from wct2g1.reconstruct import rotation_from_homography

    rng = np.random.default_rng(0)
    K = np.array([[1000.0, 0, 640], [0, 1000.0, 360], [0, 0, 1]])
    R_true = R.from_euler("yx", [4.0, -1.5], degrees=True).as_matrix()
    rays = np.c_[rng.uniform(-0.5, 0.5, (200, 2)), np.ones(200)]
    p0 = (K @ rays.T).T
    p1 = (K @ R_true @ rays.T).T
    pts0, pts1 = p0[:, :2] / p0[:, 2:], p1[:, :2] / p1[:, 2:]
    assert np.allclose(rotation_from_homography(pts0, pts1, K), R_true, atol=1e-6)
