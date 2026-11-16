import numpy as np

from wct2g1.reconstruct import resample_indices


def test_resample_50_to_30():
    idx, fps = resample_indices(101, 50.0, 30)
    assert fps == 30.0
    assert idx[0] == 0 and idx[-1] == 100 and len(idx) == 61   # 2 s span -> 61 frames
    assert np.all(np.diff(idx) >= 1)


def test_resample_native_and_no_upsample():
    assert resample_indices(10, 50.0, "native")[1] == 50.0
    idx, fps = resample_indices(10, 25.0, 30)
    assert fps == 25.0 and list(idx) == list(range(10))
