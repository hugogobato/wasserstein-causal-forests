#!/bin/bash
# Production WCF fits for the primary state-minimum-wage study.
# Usage: bash run_production.sh <core|robust|all>
# Every fit skips when its output JSON already exists, so batches resume safely.
# Parallelism is fixed at 2 (machine policy: at most 2 python processes).
set -u
cd "$(dirname "$0")/../../.."
ROOT="$(pwd)"
STUDY="$ROOT/results/applied_study_exploration/primary_state_minwage"
export PYTHONPATH="$ROOT/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export ROOT STUDY

run() {
  local out="$1"; shift
  if [ -f "$out" ]; then echo "skip $out"; return 0; fi
  echo "START $(basename "$out")"
  if python3 "$ROOT/research/applied/primary_state_minwage/fit_wcf.py" "$@" --out "$out" \
      > "${out%.json}.log" 2>&1; then
    echo "DONE $(basename "$out")"
  else
    echo "FAIL $(basename "$out")"
  fi
}
export -f run

RES="$STUDY/results"
DATA="$STUDY/data"

CORE=(
  "run $RES/fit_primary_seed0.json --data-dir $DATA --seed 0"
  "run $RES/fit_primary_seed1.json --data-dir $DATA --seed 1"
  "run $RES/fit_primary_seed2.json --data-dir $DATA --seed 2"
  "run $RES/placebo_primary_seed0.json --data-dir $DATA --seed 0 --placebo"
  "run $RES/placebo_primary_seed1.json --data-dir $DATA --seed 1 --placebo"
  "run $RES/placebo_primary_seed2.json --data-dir $DATA --seed 2 --placebo"
)

ROBUST=(
  "run $RES/fit_primary_A2_seed0.json --data-dir $DATA --seed 0 --treatment A2"
  "run $RES/fit_primary_A3_seed0.json --data-dir $DATA --seed 0 --treatment A3"
  "run $RES/fit_ext_1980_2016_seed0.json --data-dir $STUDY/data_1980_2016 --seed 0"
  "run $RES/fit_ext_1980_2016_seed1.json --data-dir $STUDY/data_1980_2016 --seed 1"
  "run $RES/fit_ext_2017_2022_seed0.json --data-dir $STUDY/data_2017_2022 --seed 0"
  "run $RES/fit_yearfe_seed0.json --data-dir $STUDY/data_yearfe_2000_2016 --seed 0"
  "run $RES/fit_fd_seed0.json --data-dir $STUDY/data_fd_2000_2016 --seed 0"
  "run $RES/fit_fd_seed1.json --data-dir $STUDY/data_fd_2000_2016 --seed 1"
)

STABILITY=(
  "run $RES/lo_state_CA.json --data-dir $DATA --seed 0 --drop-states 6"
  "run $RES/lo_state_TX.json --data-dir $DATA --seed 0 --drop-states 48"
  "run $RES/lo_state_NY.json --data-dir $DATA --seed 0 --drop-states 36"
)

batch="${1:-core}"
jobs=()
case "$batch" in
  core) jobs=("${CORE[@]}") ;;
  robust) jobs=("${ROBUST[@]}") ;;
  stability) jobs=("${STABILITY[@]}") ;;
  all) jobs=("${CORE[@]}" "${ROBUST[@]}" "${STABILITY[@]}") ;;
  *) echo "unknown batch $batch"; exit 1 ;;
esac

printf '%s\n' "${jobs[@]}" | xargs -P 2 -I CMD bash -c CMD
echo "BATCH $batch COMPLETE"
