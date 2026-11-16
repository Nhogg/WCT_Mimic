import numpy as np

from wct2g1.config import load_config
from wct2g1.track import Track, fill_and_smooth, score_track, select_tracks, stitch

CFG = load_config()
S = CFG["track"]
FPS = 50.0


def _walk(f0, f1, x0, dx=2.0, h=300.0, y=100.0):
    f = np.arange(f0, f1)
    x = x0 + dx * (f - f0)
    boxes = np.stack([x, np.full_like(x, y), x + h / 2.5, np.full_like(x, y + h)], -1)
    return Track(f, boxes, np.full(len(f), 0.9), [int(f0)])


def test_stitch_joins_occlusion_gap_only():
    a = _walk(0, 100, 0)
    b = _walk(110, 200, 220)            # 10-frame gap, continues the same path
    far = _walk(105, 200, 1500)         # someone else across the frame
    late = _walk(160, 250, 420)         # starts too late (gap > 0.5 s after a)
    out = stitch([a, b, far, late], FPS, S["stitch"])
    starts = sorted((t.start, t.end) for t in out)
    assert (0, 199) in starts and len(out) == 3


def test_fill_marks_interpolated_and_keeps_length():
    t = stitch([_walk(0, 50, 0), _walk(60, 100, 120)], FPS, S["stitch"])[0]
    full = fill_and_smooth(t, FPS, S["smooth"])
    assert len(full.frames) == 100 and full.interpolated.sum() == 10
    assert np.all(np.diff(full.frames) == 1)


def test_select_rules_and_top2():
    sel = S["select"]
    H = 1080
    runner1 = fill_and_smooth(_walk(0, 300, 0, dx=4), FPS, S["smooth"])
    runner2 = fill_and_smooth(_walk(0, 290, 600, dx=3), FPS, S["smooth"])
    ref = fill_and_smooth(_walk(0, 300, 1200, dx=0.5), FPS, S["smooth"])       # slow mover
    tiny = fill_and_smooth(_walk(0, 300, 100, h=80), FPS, S["smooth"])         # crowd member
    short = fill_and_smooth(_walk(0, 60, 50), FPS, S["smooth"])
    scored = [(t, score_track(t, 300, H, FPS, sel)) for t in [runner1, runner2, ref, tiny, short]]
    res = {t.boxes[0, 0]: st for t, _, st in select_tracks(scored, sel)}
    assert res[runner1.boxes[0, 0]] == "ok" and res[runner2.boxes[0, 0]] == "ok"
    assert res[ref.boxes[0, 0]] == "rejected:not_top_ranked"
    assert res[tiny.boxes[0, 0]] == "rejected:too_small"
    assert res[short.boxes[0, 0]] == "rejected:too_short"
