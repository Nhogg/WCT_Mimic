"""Milestone 6: parkour course height profiles (pure numpy, importable from any env).

A course is a straight lane along +x, described as a list of (length_m, height_m)
segments. Each segment becomes a box spanning the lane width from below the floor up to
`height` (0 = floor, > 0 = box top, < 0 = pit). Every course starts and ends with a flat
zone: the robot spawns in the middle of the start zone and the goal is the middle of the
end zone.

Courses come from two places:
  sample_profile  random courses for training; obstacle size grows with `difficulty`.
  from_spec       a hand-written course (YAML list of features), e.g. courses/demo.yaml.
"""
from __future__ import annotations

import numpy as np

START_M = 1.5        # flat start zone; the robot spawns at START_M / 2
END_M = 1.5          # flat end zone; the goal is END_M / 2 before the end

Segments = list[tuple[float, float]]


# --- features: each returns segments -------------------------------------------------

def flat(length: float) -> Segments:
    return [(length, 0.0)]


def platform(height: float, length: float = 1.2) -> Segments:
    """Climb onto a box, cross it, step down."""
    return [(length, height)]


def hurdle(height: float, length: float = 0.2) -> Segments:
    """A thin box to step or hop over."""
    return [(length, height)]


def gap(width: float, height: float = 0.3, length: float = 0.8) -> Segments:
    """Two platforms with floor between them: jump across or drop down and climb out."""
    return [(length, height), (width, 0.0), (length, height)]


def stairs(height: float, steps: int = 3, tread: float = 0.35, top: float = 0.6) -> Segments:
    """Steps up to `height`, a landing, steps down."""
    up = [(tread, height * (i + 1) / steps) for i in range(steps)]
    down = [(tread, height * (steps - 1 - i) / steps) for i in range(steps - 1)]
    return up + [(top, height)] + down


def pit(width: float, depth: float = 0.4) -> Segments:
    """A trench below the floor."""
    return [(width, -depth)]


FEATURES = dict(flat=flat, platform=platform, hurdle=hurdle, gap=gap, stairs=stairs, pit=pit)


# --- helpers ---------------------------------------------------------------------------

def merge(segs: Segments, eps: float = 1e-4) -> Segments:
    """Join neighbours of equal height (no seams to catch a foot) and drop empty ones."""
    out: Segments = []
    for length, h in segs:
        if length <= eps:
            continue
        if out and abs(out[-1][1] - h) <= eps:
            out[-1] = (out[-1][0] + length, out[-1][1])
        else:
            out.append((float(length), float(h)))
    return out


def total_length(segs: Segments) -> float:
    return float(sum(length for length, _ in segs))


def goal_offset(lane_length: float) -> float:
    """Distance along x from the spawn point to the goal."""
    return lane_length - START_M / 2 - END_M / 2


def fit(segs: Segments, lane_length: float) -> Segments:
    """Start zone + segs + flat padding so the lane is exactly `lane_length` long
    (the end zone is part of the padding)."""
    body = merge(segs)
    need = START_M + total_length(body) + END_M
    if need > lane_length + 1e-6:
        raise ValueError(f"course needs {need:.2f} m but the lane is {lane_length:.2f} m")
    return merge(flat(START_M) + body + flat(lane_length - START_M - total_length(body)))


def spec_segments(spec: list) -> Segments:
    """Segments of a hand-written course (no start/end zones): a list of
    {feature: params} items, e.g.
        - flat: 1.0
        - platform: {height: 0.3, length: 1.2}
        - gap: {width: 0.5, height: 0.3}
    A scalar parameter is passed as the feature's first argument."""
    segs: Segments = []
    for item in spec:
        if not isinstance(item, dict) or len(item) != 1:
            raise ValueError(f"course item must be a one-key mapping, got {item!r}")
        (kind, p), = item.items()
        if kind not in FEATURES:
            raise ValueError(f"unknown course feature {kind!r}; known: {sorted(FEATURES)}")
        segs += FEATURES[kind](**p) if isinstance(p, dict) else FEATURES[kind](p)
    return merge(segs)


def from_spec(spec: list, lane_length: float | None = None) -> Segments:
    """Full lane for a hand-written course; `lane_length` None = as short as possible."""
    body = spec_segments(spec)
    need = START_M + total_length(body) + END_M
    return fit(body, need if lane_length is None else lane_length)


def _lerp(r, d: float) -> float:
    return float(r[0] + (r[1] - r[0]) * d)


def sample_profile(rng: np.random.Generator, difficulty: float, lane_length: float,
                   c: dict) -> Segments:
    """A random course for `difficulty` in [0, 1]. `c` is config parkour.course:
    height_m, gap_m, run_m ([easy, hard] ranges), hurdle_max_m, pit_from, weights."""
    d = float(np.clip(difficulty, 0.0, 1.0))
    h_max, g_max = _lerp(c["height_m"], d), _lerp(c["gap_m"], d)
    run = (_lerp([c["run_m"][1], c["run_m"][0]], d), c["run_m"][1])   # shorter run-ups later
    kinds = [k for k in c["weights"] if k != "pit" or d >= c["pit_from"]]
    w = np.array([c["weights"][k] for k in kinds], float)
    room = lane_length - START_M - END_M
    segs: Segments = []
    while True:
        k = kinds[rng.choice(len(kinds), p=w / w.sum())]
        u = rng.uniform(0.5, 1.0)
        if k == "platform":
            f = platform(u * h_max, rng.uniform(0.8, 1.6))
        elif k == "hurdle":
            f = hurdle(u * min(h_max, c["hurdle_max_m"]), rng.uniform(0.15, 0.3))
        elif k == "gap":
            f = gap(rng.uniform(0.3, 1.0) * g_max, u * h_max, rng.uniform(0.6, 1.0))
        elif k == "stairs":
            f = stairs(u * h_max * 1.25, int(rng.integers(2, 4)))
        else:
            f = pit(rng.uniform(0.3, 1.0) * g_max, u * h_max)
        f = f + flat(rng.uniform(*run))
        if total_length(segs) + total_length(f) > room:
            break
        segs += f
    return fit(segs, lane_length)
