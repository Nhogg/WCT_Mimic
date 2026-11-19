import pytest

pytest.importorskip("mjlab")   # runs in the `mjlab` env only

from wct2g1.multi_motion import allocate_envs, clip_bins, motion_paths  # noqa: E402


def test_clip_bins_stay_inside_clips_and_skip_last_frame():
    lengths = [220, 51, 2]
    starts, ends = clip_bins(lengths, 50)
    clip_edges = [(0, 220), (220, 271), (271, 273)]
    covered = set()
    for a, b in zip(starts, ends):
        assert b > a
        lo, hi = next(e for e in clip_edges if e[0] <= a < e[1])
        assert b <= hi - 1          # never the clip's last frame, never the next clip
        covered |= set(range(a, b))
    for lo, hi in clip_edges:      # every other frame is sampleable
        assert set(range(lo, hi - 1)) <= covered


def test_motion_paths_reads_list_relative_to_file(tmp_path):
    (tmp_path / "a.npz").touch()
    lst = tmp_path / "m.txt"
    lst.write_text("# comment\na.npz\n/abs/b.npz\n\n")
    assert motion_paths(str(lst)) == [str(tmp_path / "a.npz"), "/abs/b.npz"]
    assert motion_paths("x.npz") == ["x.npz"]


def test_allocate_envs_proportional_contiguous_and_covering():
    a = allocate_envs(100, [300, 100, 5])
    assert len(a) == 100 and set(a) == {0, 1, 2}
    assert (a[:-1] <= a[1:]).all()                     # contiguous blocks
    counts = [int((a == i).sum()) for i in range(3)]
    assert counts[0] > counts[1] > counts[2] >= 1
    assert list(allocate_envs(2, [10, 10, 10])) == [0, 1]   # fewer envs than clips
