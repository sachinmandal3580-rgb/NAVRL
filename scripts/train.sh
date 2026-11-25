#!/usr/bin/env bash
# Train a NavRL stage (CPU, tensorboard logging).
#
#   scripts/train.sh                                        # s1_empty, background (nohup)
#   FOREGROUND=1 scripts/train.sh                           # watch it
#   TASK=s2_static NUM_ENVS=4 MAX_ITER=300 WARM_START=logs/navrl/curriculum/s1/model_100.pt scripts/train.sh
#   scripts/train.sh --seed 7                               # extra args pass through to train.py
#
# One iteration = NUM_ENVS x NUM_STEPS env steps + one PPO update
# (4 x 1024 = 4096 steps/iter by default).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PY="${PY:-/home/sachin/miniconda3/envs/dhav/bin/python}"
TASK="${TASK:-s1_empty}"
# 4 envs = benchmarked throughput plateau on this 12-thread laptop (6 and 8 were no faster)
# and keeps RAM low (7 GB machine).
NUM_ENVS="${NUM_ENVS:-4}"
NUM_STEPS="${NUM_STEPS:-1024}"
MAX_ITER="${MAX_ITER:-100}"
N_STACK="${N_STACK:-3}"
PHYSICS_DT="${PHYSICS_DT:-0.01}"   # 0.02 = ~1.8x faster env stepping
EXPERIMENT="${EXPERIMENT:-curriculum}"
RUN_NAME="${RUN_NAME:-${TASK}_$(date +%Y%m%d_%H%M%S)}"
WARM_START="${WARM_START:-}"

LOG_DIR="$REPO/logs/navrl/$EXPERIMENT"
mkdir -p "$LOG_DIR"
STDOUT="$LOG_DIR/${RUN_NAME}.stdout.log"

ARGS=(
  scripts/train.py
  --task "$TASK"
  --num_envs "$NUM_ENVS"
  --num_steps_per_env "$NUM_STEPS"
  --max_iterations "$MAX_ITER"
  --n_stack "$N_STACK"
  --physics_dt "$PHYSICS_DT"
  --experiment_name "$EXPERIMENT"
  --run_name "$RUN_NAME"
)
[[ -n "$WARM_START" ]] && ARGS+=(--warm_start "$WARM_START")
ARGS+=("$@")

# ROS Jazzy on PYTHONPATH is not needed here
unset PYTHONPATH || true

if [[ "${FOREGROUND:-0}" == "1" ]]; then
  exec "$PY" -u "${ARGS[@]}"
else
  nohup "$PY" -u "${ARGS[@]}" > "$STDOUT" 2>&1 &
  echo "started PID $! -> logs/navrl/$EXPERIMENT/$RUN_NAME"
  echo "follow:  tail -f $STDOUT"
  echo "stop:    kill $!"
fi
