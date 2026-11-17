import pandas as pd
import pytest

from wct2g1 import manifest as m


def test_read_missing_manifest_is_empty_with_key(cfg):
    df = m.read_manifest(cfg, "shots")
    assert df.empty and "shot_id" in df.columns and "status" in df.columns


def test_upsert_roundtrip_and_replace(cfg):
    m.upsert(cfg, "raw", [{"video_id": "a", "status": m.ok(), "fps": 50.0},
                          {"video_id": "b", "status": m.failed("boom")}])
    m.upsert(cfg, "raw", [{"video_id": "b", "status": m.ok(), "fps": 60.0}])
    df = m.read_manifest(cfg, "raw").set_index("video_id")
    assert list(df.index) == ["a", "b"]
    assert df.loc["b", "status"] == "ok" and df.loc["b", "fps"] == 60.0
    assert (df["config_hash"] == cfg.hash).all()
    assert not m.manifest_path(cfg, "raw").with_suffix(".parquet.tmp").exists()


def test_done_keys_retries_failures(cfg):
    m.upsert(cfg, "clean", [
        {"clip_id": "ok1", "status": m.ok()},
        {"clip_id": "rej", "status": m.rejected("root_speed")},
        {"clip_id": "bad", "status": m.failed(RuntimeError("x"))},
    ])
    assert m.done_keys(cfg, "clean") == {"ok1", "rej"}


def test_write_rejects_duplicates_and_missing_columns(cfg):
    with pytest.raises(ValueError, match="duplicate"):
        m.write_manifest(cfg, "g1", pd.DataFrame({"clip_id": ["x", "x"], "status": ["ok", "ok"]}))
    with pytest.raises(ValueError, match="status"):
        m.write_manifest(cfg, "g1", pd.DataFrame({"clip_id": ["x"]}))
    with pytest.raises(ValueError, match="key column"):
        m.write_manifest(cfg, "g1", pd.DataFrame({"status": ["ok"]}))


def test_status_helpers():
    assert m.status_kind(m.rejected("too_short")) == "rejected"
    assert m.failed("line1\nline2") == "failed:line1 line2"
    assert m.failed(ValueError()) == "failed:ValueError"


def test_filter_ids_tolerates_missing_columns():
    import pandas as pd

    from wct2g1.manifest import filter_ids

    df = pd.DataFrame({"track_id": ["a_t1", "b_t1"], "status": ["failed:x", "ok"]})
    assert list(filter_ids(df, ["a_t1"])["track_id"]) == ["a_t1"]
    assert filter_ids(df, ["zzz"]).empty
    assert filter_ids(pd.DataFrame(), ["a"]).empty


def test_upsert_replace_and_shard(cfg):
    import pandas as pd

    from wct2g1.manifest import read_manifest, shard, upsert

    upsert(cfg, "clean", [dict(clip_id="t1_c0", track_id="t1", status="ok"),
                          dict(clip_id="t1_c1", track_id="t1", status="ok"),
                          dict(clip_id="t2_c0", track_id="t2", status="ok")])
    upsert(cfg, "clean", [dict(clip_id="t1_c0", track_id="t1", status="rejected:x")],
           replace=("track_id", "t1"))
    d = read_manifest(cfg, "clean")
    assert sorted(d["clip_id"]) == ["t1_c0", "t2_c0"]
    assert not (cfg.path_of("manifests") / "clean.lock").exists()

    df = pd.DataFrame({"k": [f"id{i}" for i in range(50)]})
    parts = [shard(df, "k", f"{i}/4") for i in range(4)]
    assert sum(len(p) for p in parts) == 50
    assert set().union(*(set(p["k"]) for p in parts)) == set(df["k"])
