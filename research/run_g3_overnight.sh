#!/usr/bin/env bash
# Run Phase G3 end to end, unattended.
#
#   bash research/run_g3_overnight.sh
#
# Executes the frozen tournament, then merges, analyses, recomputes the gate
# flags independently, and writes the memo. Each stage runs only if the one
# before it succeeded, except the tournament, which is resumable: rerunning this
# script skips cells already recorded in the per-worker execution logs.
#
# Worker count is set from physical cores rather than logical, and from the RAM
# each worker actually needs. Every numerical library is pinned to one thread by
# research/run_g3.py; without that pinning six workers took thirty-nine threads
# each and lost roughly a factor of forty to oversubscription.

set -u
cd "$(dirname "$0")/.." || exit 1

WORKERS="${WORKERS:-9}"
LOGS="logs/g3"
mkdir -p "$LOGS"

STAMP="$(date +%Y%m%d_%H%M%S)"
MAIN_LOG="$LOGS/overnight_$STAMP.log"

exec > >(tee -a "$MAIN_LOG") 2>&1

echo "=== Phase G3 overnight run, started $(date) ==="
echo "workers: $WORKERS"
free -m | sed -n '1,2p'
echo

# A RAM watchdog rather than a RAM killer: it records pressure so a shortfall is
# diagnosable afterwards, and only intervenes if the machine is genuinely about
# to start swapping hard.
(
  while true; do
    AVAIL=$(free -m | awk '/^Mem:/ {print $7}')
    echo "$(date +%H:%M:%S) available_mb=$AVAIL" >> "$LOGS/memory_$STAMP.log"
    if [ "$AVAIL" -lt 800 ]; then
      echo "$(date +%H:%M:%S) CRITICAL: available RAM ${AVAIL}MB, stopping the run" \
        | tee -a "$LOGS/memory_$STAMP.log"
      pkill -f "run_g3.py run"
      break
    fi
    sleep 60
  done
) &
WATCHDOG=$!
trap 'kill $WATCHDOG 2>/dev/null' EXIT

stage() {
  echo
  echo "--- $1 :: $(date +%H:%M:%S) ---"
  shift
  "$@"
  local status=$?
  echo "--- exit $status :: $(date +%H:%M:%S) ---"
  return $status
}

stage "WP3-B1 execute tournament" \
  nice -n 5 python3 research/run_g3.py run --workers "$WORKERS" --resume
TOURNAMENT=$?

DONE=$(cat results/manifests/execution_log*.jsonl 2>/dev/null | wc -l)
echo
echo "cells recorded: $DONE"

if [ "$TOURNAMENT" -ne 0 ]; then
  echo "tournament did not exit cleanly; stopping before the merge so that a"
  echo "partial run is never analysed as if it were complete."
  exit 1
fi

# The merge is the reconciliation gate. It fails on a duplicate, an unknown key,
# or a missing cell, and a failure here must stop the pipeline: analysing an
# incomplete manifest would silently report a smaller tournament.
stage "WP3-B2 merge and reconcile" python3 research/run_g3.py merge || {
  echo "merge FAILED; see results/merged/merge_audit.json"
  exit 1
}

stage "WP3-B3 tables and figures" python3 research/checks/g3_report.py || exit 1
stage "WP3-B3 independent gate flags" python3 research/checks/g3_gate_flags.py
stage "WP3-B3 gate memo" python3 research/checks/g3_write_memo.py || exit 1

echo
echo "=== finished $(date) ==="
echo "verdict:"
python3 -c "
import json
p = json.load(open('results/merged/analysis_payload.json'))
print(json.dumps(p['gate_flags']['summary'], indent=2))
"
echo
echo "artefacts:"
ls -1 results/merged/ tables/simulation/ figures/simulation/ 2>/dev/null
echo "memo: research/gates/G3_simulation_memo.md"
echo "peak memory pressure:"
sort -t= -k2 -n "$LOGS/memory_$STAMP.log" 2>/dev/null | head -3
