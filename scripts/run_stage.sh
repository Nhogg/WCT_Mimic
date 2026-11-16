#!/usr/bin/env bash
# Usage: scripts/run_stage.sh <stage|report> [--limit N] [--ids ...]
# Activates the conda env configured for the stage (config.yaml: envs) and runs it.
# Works without a conda install (e.g. on compute nodes): envs are found by path.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${WCT2G1_CONFIG:-$ROOT/config.yaml}"
STAGE="${1:?usage: run_stage.sh <stage|report> [args...]}"
shift

# Read envs.<stage> from the flat `envs:` mapping without needing PyYAML.
ENV_NAME="$(awk -v s="$STAGE" '
  /^envs:/ {in_envs=1; next}
  in_envs && /^[^ #]/ {in_envs=0}
  in_envs { gsub(/#.*/, ""); split($0, kv, ":"); k=kv[1]; v=kv[2];
            gsub(/[ \t]/, "", k); gsub(/[ \t]/, "", v); if (k==s) print v }' "$CONFIG")"
[[ -n "$ENV_NAME" ]] || { echo "unknown stage '$STAGE' (no entry under envs: in $CONFIG)" >&2; exit 2; }

ENV_DIR=""
for d in ${CONDA_ENVS_DIRS:+${CONDA_ENVS_DIRS//:/ }} "$HOME/.conda/envs" "${CONDA_BASE:-/nonexistent}/envs"; do
  [[ -x "$d/$ENV_NAME/bin/python" ]] && { ENV_DIR="$d/$ENV_NAME"; break; }
done
[[ -n "$ENV_DIR" ]] || { echo "conda env '$ENV_NAME' not found" >&2; exit 2; }

if [[ -n "${CONDA_BASE:-}" && -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate "$ENV_DIR"
else
  export CONDA_PREFIX="$ENV_DIR" PATH="$ENV_DIR/bin:$PATH"
  for f in "$ENV_DIR"/etc/conda/activate.d/*.sh; do
    # shellcheck disable=SC1090
    [[ -f "$f" ]] && source "$f"
  done
fi
cd "$ROOT"
exec python -m wct2g1 --config "$CONFIG" "$STAGE" "$@"
