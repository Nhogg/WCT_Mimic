# wct2g1

World Chase Tag YouTube footage → Unitree G1 (29-DoF) reference motions → BeyondMimic-style
tracking policies, in simulation only. **Non-commercial research use**: the footage belongs
to World Chase Tag and SMPL/SMPL-X have their own licenses; `data/` is git-ignored and
must not be published.

## Stages

| # | Stage | Module | Env | Status |
|---|-------|--------|-----|--------|
| 1 | Acquire (yt-dlp) | `acquire.py` | wct-data | run on 1 video |
| 2 | Segment (PySceneDetect + open_clip) | `segment.py` | wct-data | run; ~75% precision, prompts need tuning |
| 3 | Track (YOLO26 + BoT-SORT/ReID) | `track.py` | wct-data | run on all shots |
| 4 | Reconstruct (GVHMR, own bbox wrapper) | `reconstruct.py` | gvhmr | run on all 130 tracks of 1 video (L40S only) |
| 5 | Clean / retime / contacts / cut | `clean.py` | wct-data | 89 clips (377 s); rolling floor, obstacle split |
| 6 | Retarget (GMR → G1 CSV) | `retarget.py` | gmr | 89 clips; joint limits report-only |
| 7 | Train + filter (mjlab tracking) | `train_filter.py` | mjlab | 74/89 clips accepted (329 s); rollout videos |
| M4 | One policy for all accepted clips | `generalist.py`, `multi_motion.py` | mjlab | v3 (look-ahead + terrain boxes + height scan): 73/74 clips pass (v2: 66, v1: 54) |
| M5 | Obstacle boxes inferred from planted feet | `obstacles.py`, `multi_motion.py` | wct-data / mjlab | boxes in 5 clips; 1/5 box clips tracked without falls |
| M6 | Reach the end of an obstacle course (+ WCT style reward) | `course.py`, `course_profile.py`, `parkour.py`, `style.py` | mjlab | v1 (no style) and v2 (style) training |

Each stage reads the previous manifest (`data/manifests/<stage>.parquet`), skips items
already `ok`/`rejected:*`, retries `failed:*`, and stamps rows with the config hash.
All paths, thresholds and URLs live in `config.yaml`.

## Deviations from the original spec

- **Stage 7 uses mjlab, not Isaac Sim + Isaac Lab.** HybridRobotics/whole_body_tracking is
  retired upstream in favour of mjlab's tracking task; the cluster's GPU nodes run RHEL 10
  with no containers, which Isaac Sim 4.5 doesn't target. `wct2g1/sim_task.py` registers
  `WCT-Tracking-Flat-G1-SimOnly` (no domain randomization or observation noise,
  `torque_scale` option). `patches/mjlab-csv_to_npz.patch` lets mjlab's
  `csv_to_npz` write a local file instead of uploading to W&B (and loosens a float32
  velocity self-check to 1e-4); apply with
  `scripts/apply_patches.sh`.
- **GVHMR runs at ~30 fps with an integer frame stride** (`reconstruct.input_fps`; 50 fps
  footage -> 25 fps): it is trained at 30 fps, assumes evenly spaced frames, and predicts
  per-frame motion; Stage 5 resamples to 50 fps.
- **Coordinates:** GVHMR output (Stage 4) is stored Y-up as GVHMR produces it; Stage 5
  converts to Z-up with GMR's own rotation and stores AMASS-style SMPL-X so Stage 6 uses
  GMR's standard SMPL-X path. CSVs are `pos(3), quat xyzw(4), 29 joints`, joint order
  identical in GMR and mjlab (checked).

## Setup

```bash
conda env create -f envs/wct-data.yml
# gvhmr: python 3.10, pip install chumpy with --no-build-isolation, then
#        third_party/GVHMR/requirements.txt and -e third_party/GVHMR
# gmr:   python 3.10, pip install -e third_party/GMR pandas pyarrow pyyaml
# mjlab: python 3.12, pip install -e third_party/mjlab pandas pyarrow pyyaml
scripts/apply_patches.sh
```

Model files (not in git), under `data/models/`:
- `body_models/smplx/SMPLX_NEUTRAL.{npz,pkl}`, `body_models/smpl/SMPL_NEUTRAL.pkl` (licensed, manual download)
- `gvhmr/checkpoints/{gvhmr,hmr2,vitpose,...}` from GVHMR's Google Drive

## Usage

```bash
scripts/run_stage.sh <stage> [--limit N] [--ids ...]   # on this machine
scripts/submit.sh <stage> [--limit N] [--ids ...]      # as a SLURM GPU job via the head node
scripts/run_stage.sh report
```

Large stages shard over a SLURM array: `ARRAY=12 scripts/submit.sh train_filter` runs
12 tasks, each taking the items whose id hashes to its shard.

Milestone 4 (multi-clip policy), after Stage 7:

```bash
scripts/submit.sh generalist_train            # one GPU job, ~8 h for 30k iterations
ARRAY=12 scripts/submit.sh generalist_eval    # per-clip rollouts -> manifests/generalist.parquet
```

`generalist.run_name` in `config.yaml` names the run (`data/policies/generalist/<run_name>`,
`review/generalist_rollouts/<run_name>/`). The multi-clip motion command takes a `.txt`
list of npz files in place of a single motion file.

Milestone 6 (parkour courses):

```bash
scripts/submit.sh parkour_train               # random courses with a difficulty curriculum
scripts/submit.sh parkour_eval                # every level of random courses + courses/*.yaml
scripts/submit.sh parkour_eval --ids demo     # one hand-written course: courses/demo.yaml
```

A course is a YAML list of features (`flat`, `platform`, `hurdle`, `gap`, `stairs`, `pit`;
see `wct2g1/course_profile.py`). The policy walks from the start zone to the goal at the
end; results go to `manifests/parkour.parquet`, videos to `review/parkour/<run_name>/`.

`parkour.style` adds a WCT style reward: a discriminator trained during PPO rollouts
(inside the env, `wct2g1/style.py`) scores how much each one-step transition of the robot
looks like the retargeted WCT clips (adversarial motion prior). A run's config can be
snapshotted outside the repo with `root:` pointing back at it, e.g.
`WCT2G1_CONFIG=data/policies/parkour/v1/config.yaml scripts/submit.sh parkour_eval`.

Tests: `conda activate wct-data && python -m pytest`.
