import numpy as np
import pandas as pd

from wct2g1 import segment


def test_sample_indices_inside_shot():
    idx = segment.sample_indices(100, 200, 3)
    assert idx == [125, 150, 175]
    assert all(100 <= i < 102 for i in segment.sample_indices(100, 102, 3))


def test_neighbour_flow_ratio():
    flows = np.array([10.0, 10.0, 2.0, 10.0, np.nan, 10.0])
    r = segment.neighbour_flow_ratio(flows, k=2)
    assert r[2] == 0.2           # slow shot vs neighbours
    assert r[0] == 10.0 / 6.0       # neighbours 10, 2 -> median 6
    assert np.isnan(r[4])         # own flow unknown


def _shots(flags):
    return pd.DataFrame({
        "shot_id": [f"v_{i:04d}" for i in range(len(flags))], "video_id": "v",
        "start_frame": 0, "end_frame": 100, "dur_s": 2.0, "slowmo_prob": 0.9,
        "flow_ratio": 0.2, "path": "p", "status": "ok", "is_slowmo": flags})


def test_slowmo_csv_preserves_human_speed_factor(cfg):
    df = segment.sync_slowmo_csv(cfg, _shots([True, False, True]))
    p = cfg.path_of("review") / "slowmo.csv"
    csv = pd.read_csv(p)
    assert list(csv["shot_id"]) == ["v_0000", "v_0002"]
    assert csv["speed_factor"].isna().all() and df["speed_factor"].isna().all()

    csv.loc[csv["shot_id"] == "v_0002", "speed_factor"] = 0.25   # human fills it in
    csv.to_csv(p, index=False)
    df = segment.sync_slowmo_csv(cfg, _shots([True, False, True]))
    assert df.set_index("shot_id").loc["v_0002", "speed_factor"] == 0.25
    assert pd.read_csv(p).set_index("shot_id").loc["v_0002", "speed_factor"] == 0.25


def test_slowmo_csv_tolerates_failure_only_manifest(cfg):
    df = pd.DataFrame({"shot_id": ["v_video"], "video_id": ["v"], "status": ["failed:x"]})
    out = segment.sync_slowmo_csv(cfg, df)
    assert out["speed_factor"].isna().all()
