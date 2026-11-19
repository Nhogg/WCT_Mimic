import numpy as np

from wct2g1.obstacles import infer_boxes, runs

C = dict(ground_percentile=5, max_foot_speed=0.4, min_stance_s=0.08, min_height_m=0.2,
         merge_dz_m=0.1, merge_dxy_m=0.3, margins_m=[0.2, 0.12, 0.06], min_size_m=0.25,
         penetration_tol_m=0.05, max_penetration_frac=0.05, step_window_s=0.25)
FPS = 50.0


def _motion(n=150):
    """Standing still with both ankles 0.03 m above the floor; body 0 is the pelvis."""
    p = np.zeros((n, 30, 3))
    p[:, :, 2] = 0.5
    p[:, 0, 2] = 0.75
    p[:, [4, 5, 6], :] = [0.0, 0.1, 0.03]
    p[:, [10, 11, 12], :] = [0.0, -0.1, 0.03]
    return p


def _vel(p):
    return np.gradient(p, 1 / FPS, axis=0)


def test_runs():
    assert runs(np.array([0, 1, 1, 0, 1], bool)) == [(1, 3), (4, 5)]
    assert runs(np.zeros(3, bool)) == []


def test_flat_motion_has_no_boxes():
    p = _motion()
    kept, dropped = infer_boxes(p, _vel(p), FPS, C)
    assert kept == [] and dropped == []


def test_foot_on_box_gives_box_under_it():
    p = _motion()
    p[50:100, [4, 5, 6], :] = [0.5, -0.4, 0.03 + 0.35]   # left foot up on a 0.35 m box
    kept, _ = infer_boxes(p, _vel(p), FPS, C)
    assert len(kept) == 1
    b = kept[0]
    assert abs(b["top"] - 0.35) < 1e-6
    assert abs(b["center"][0] - 0.5) < 1e-6 and abs(b["center"][1] + 0.4) < 1e-6
    assert b["half_size"][2] == b["top"] / 2


def test_box_dropped_when_reference_walks_through_it():
    p = _motion()
    p[50:100, [4, 5, 6], :] = [0.5, -0.4, 0.38]
    p[:, 0, :2] = [0.5, -0.4]          # pelvis stays over the box, below its top
    p[:, 0, 2] = 0.2
    kept, dropped = infer_boxes(p, _vel(p), FPS, C)
    assert kept == [] and len(dropped) == 1
