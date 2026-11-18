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
| 4 | Reconstruct (GVHMR, own bbox wrapper) | `reconstruct.py` | gvhmr | run on M1 track (L40S only) |
| 5 | Clean / retime / contacts / cut | `clean.py` | wct-data | run on M1; rolling floor, obstacle split |
| 6 | Retarget (GMR → G1 CSV) | `retarget.py` | gmr | run on M1; joint limits report-only |
| 7 | Train + filter (mjlab tracking) | `train_filter.py` | mjlab | M1 clip accepted; rollout videos; multi-clip final policy TODO |

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

Tests: `conda activate wct-data && python -m pytest`.
