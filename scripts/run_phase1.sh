#!/bin/bash
# Full Phase 1 matrix in a new write-once run folder.
#   cd /data/project-01 && nohup bash scripts/run_phase1.sh > /dev/null 2>&1 &
#
# Design: S3 is fixed on the iGPU for end-to-end runs (fastest in the smoke
# test); a reduced S3 sweep on CPU/NPU is kept as evidence for that choice.
set -u
SELF=$(realpath "$0")
cd "$(dirname "$SELF")"
P=$(python3 -c "import yaml;print(yaml.safe_load(open('../config.yaml'))['python'])")
RUN=$($P -c "from rag_common import new_run_dir; print(new_run_dir('phase1'))")
O=$RUN/results.jsonl
exec > "$RUN/run.log" 2>&1

# provenance
mkdir "$RUN/code_snapshot"
cp ../config.yaml "$RUN/"
cp "$SELF" rag_common.py harness.py "$RUN/code_snapshot/"
git -C .. rev-parse HEAD > "$RUN/git_rev.txt"
git -C .. status --short >> "$RUN/git_rev.txt"
$P -m pip freeze > "$RUN/pip_freeze.txt"
{ uptime; free -h; df -h /; } > "$RUN/env_start.txt"
echo "start $(date)"

# A. Per-stage sweeps
for D in CPU GPU NPU; do
  $P harness.py stage S1 --device $D --batch 1,8,32 --seq 64 --reps 10 --out $O
  $P harness.py stage S2 --device $D --cands 10,50,100 --seq 256 --reps 10 --out $O
done
$P harness.py stage S3 --device GPU --prompt-len 256,1024,4096 --out-len 32,128,256 --reps 5 --out $O
for D in CPU NPU; do   # reduced: evidence for fixing S3 on GPU
  $P harness.py stage S3 --device $D --prompt-len 256,1024,4096 --out-len 128 --reps 3 --out $O
done
echo "stages done $(date)"

# B. End-to-end, dev split. S3 fixed on GPU; S1 x S2 over all devices.
# Rounds are interleaved across mappings so machine-load drift hits all equally.
E2E="--split dev --n 100 --cands 20"
for R in 1 2 3; do
  for S1 in CPU GPU NPU; do
    for S2 in CPU GPU NPU; do
      $P harness.py e2e --map S1=$S1,S2=$S2,S3=GPU $E2E --round $R --out $O
      $P harness.py e2e --map S1=$S1,S2=$S2,S3=GPU $E2E --round $R --overlap --out $O
    done
  done
  echo "e2e round $R done $(date)"
done

# Single-device references (one round, sequential)
for D in CPU NPU; do
  $P harness.py e2e --map S1=$D,S2=$D,S3=$D $E2E --round 1 --out $O
done

{ uptime; free -h; } > "$RUN/env_end.txt"
echo "PHASE1_DONE $(date)"
