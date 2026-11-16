# wct2g1

World Chase Tag YouTube footage → Unitree G1 (29-DoF) reference motions → BeyondMimic
tracking policies in Isaac Lab. Simulation only. **Non-commercial research use**: the
footage belongs to World Chase Tag and SMPL-X has its own license; `data/` is git-ignored
and must not be published.

## Stages

| # | Stage | Module | Env | Status |
|---|-------|--------|-----|--------|
| 1 | Acquire (yt-dlp) | `wct2g1/acquire.py` | wct-data | done |
| 2 | Segment (PySceneDetect + open_clip) | `segment.py` | wct-data | todo |
| 3 | Track (YOLO + BoT-SORT) | `track.py` | wct-data | todo |
| 4 | Reconstruct (GVHMR) | `reconstruct.py` | gvhmr | todo |
| 5 | Clean / retime / cut | `clean.py` | wct-data | todo |
| 6 | Retarget (GMR → G1) | `retarget.py` | gmr | todo |
| 7 | Train + filter (BeyondMimic) | `train_filter.py` | isaaclab | todo |

Each stage reads the previous manifest (`data/manifests/<stage>.parquet`), skips items
already `ok`/`rejected:*`, retries `failed:*`, and stamps rows with the config hash.

## Usage

```bash
conda env create -f envs/wct-data.yml
# put channel/playlist URLs in config.yaml -> acquire.sources
scripts/run_stage.sh acquire --limit 1          # or --ids <youtube_id> ...
scripts/run_stage.sh acquire --manifest-only    # rebuild manifest from disk
scripts/run_stage.sh report
```

Tests: `conda activate wct-data && python -m pytest`.

All paths, thresholds and URLs live in `config.yaml`.
