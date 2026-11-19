import numpy as np
import pytest

from wct2g1 import course_profile as cp

C = dict(height_m=[0.05, 0.5], gap_m=[0.1, 0.8], run_m=[0.4, 1.4], hurdle_max_m=0.35,
         pit_from=0.5, weights=dict(platform=1, hurdle=1, gap=1, stairs=1, pit=0.5))


def test_merge_joins_equal_heights_and_drops_empty():
    assert cp.merge([(1, 0), (0.5, 0), (0, 0.3), (1, 0.2)]) == [(1.5, 0.0), (1.0, 0.2)]


@pytest.mark.parametrize("d", [0.0, 0.5, 1.0])
def test_sampled_course_fills_lane_with_flat_ends(d):
    rng = np.random.default_rng(0)
    for _ in range(20):
        segs = cp.sample_profile(rng, d, 10.0, C)
        assert abs(cp.total_length(segs) - 10.0) < 1e-9
        assert segs[0][1] == 0 and segs[0][0] >= cp.START_M - 1e-9
        assert segs[-1][1] == 0 and segs[-1][0] >= cp.END_M - 1e-9
        hs = [h for _, h in segs]
        assert max(hs) <= 1.25 * C["height_m"][1] + 1e-9
        if d < C["pit_from"]:
            assert min(hs) >= 0


def test_difficulty_raises_obstacles():
    rng = np.random.default_rng(1)
    top = {d: np.mean([max(h for _, h in cp.sample_profile(rng, d, 10.0, C))
                       for _ in range(50)]) for d in (0.0, 1.0)}
    assert top[0.0] < 0.1 < 0.3 < top[1.0]


def test_from_spec_builds_features_and_checks_length():
    spec = [{"platform": {"height": 0.3, "length": 1.0}}, {"flat": 0.5},
            {"gap": {"width": 0.4, "height": 0.2, "length": 0.6}}]
    segs = cp.from_spec(spec)
    want = [(cp.START_M, 0.0), (1.0, 0.3), (0.5, 0.0), (0.6, 0.2), (0.4, 0.0), (0.6, 0.2),
            (cp.END_M, 0.0)]
    assert np.allclose(segs, want)
    assert cp.total_length(cp.from_spec(spec, 12.0)) == pytest.approx(12.0)
    with pytest.raises(ValueError):
        cp.from_spec(spec, 5.0)
    with pytest.raises(ValueError):
        cp.from_spec([{"wall": 1.0}])


def test_stairs_go_up_and_down():
    s = cp.stairs(0.3, steps=3, tread=0.3, top=0.5)
    assert [h for _, h in s] == pytest.approx([0.1, 0.2, 0.3, 0.3, 0.2, 0.1])


def test_example_courses_parse():
    import pathlib

    import yaml
    for p in sorted(pathlib.Path(__file__).parents[1].glob("courses/*.yaml")):
        segs = cp.from_spec(yaml.safe_load(p.read_text())["course"])
        assert cp.total_length(segs) > cp.START_M + cp.END_M
