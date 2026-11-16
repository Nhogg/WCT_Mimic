#!/usr/bin/env bash
# Usage: scripts/run_stage.sh <stage|report> [--limit N] [--ids ...]
# Activates the conda env configured for the stage (config.yaml: envs) and runs it.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${WCT2G1_CONFIG:-$ROOT/config.yaml}"
STAGE="${1:?usage: run_stage.sh <stage|report> [args...]}"
shift

ENV_NAME="$(python3 - "$CONFIG" "$STAGE" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
env = cfg.get("envs", {}).get(sys.argv[2])
if not env:
    sys.exit(f"unknown stage '{sys.argv[2]}' (no entry under envs: in config)")
print(env)
PY
)"

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
cd "$ROOT"
exec python -m wct2g1 --config "$CONFIG" "$STAGE" "$@"
