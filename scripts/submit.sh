#!/usr/bin/env bash
# Usage: scripts/submit.sh <stage> [stage args...]
# Submits `run_stage.sh <stage> ...` as a SLURM job via the head node.
# ARRAY=N runs an N-task SLURM array, task i processing shard i/N (stage --shard).
# Override resources with env vars: PARTITION GRES CPUS MEM TIME (defaults: config.yaml slurm.defaults, then slurm.stages.<stage>).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${WCT2G1_CONFIG:-$ROOT/config.yaml}"
STAGE="${1:?usage: submit.sh <stage> [args...]}"
CONDA_BASE="$(conda info --base)"
read -r HEAD D_PART D_GRES D_CPUS D_MEM D_TIME < <("$CONDA_BASE/bin/python" - "$CONFIG" "$STAGE" <<'PY'
import sys, yaml
s = yaml.safe_load(open(sys.argv[1]))["slurm"]
d = {**s["defaults"], **(s.get("stages") or {}).get(sys.argv[2], {})}
print(s["head_host"], d["partition"], d["gres"], d["cpus_per_task"], d["mem"], d["time"])
PY
)
LOGDIR="$ROOT/data/logs/slurm"; mkdir -p "$LOGDIR"
ARGS="$(printf '%q ' "$@")"
ARRAY_OPT=""; OUT="$LOGDIR/%x-%j.out"
if [[ -n "${ARRAY:-}" ]]; then
  ARRAY_OPT="--array=0-$((ARRAY - 1))"; OUT="$LOGDIR/%x-%A_%a.out"
  ARGS="$ARGS --shard \$SLURM_ARRAY_TASK_ID/$ARRAY"   # expanded inside the job
fi
ssh "$HEAD" "sbatch --parsable -J wct2g1-$STAGE $ARRAY_OPT \
  -p ${PARTITION:-$D_PART} --gres=${GRES:-$D_GRES} -c ${CPUS:-$D_CPUS} \
  --mem=${MEM:-$D_MEM} -t ${TIME:-$D_TIME} -o $OUT \
  --wrap 'cd $ROOT && WCT2G1_CONFIG=$CONFIG scripts/run_stage.sh $ARGS'"
