#!/bin/bash
# Phase 1d: rerun the Phase 1c experiments that failed because two NPU models
# (S3 LLM + another stage) cannot coexist in one process. S3 runs in its own
# process (harness --s3-process auto/always; RemoteLLM in rag_common.py).
#   cd /data/project-01 && START_AT=08:00 nohup bash scripts/run_phase1d.sh > /dev/null 2>&1 &
#   B. the 5 mappings with S3=NPU and S1 or S2 on NPU (incl. all-NPU), sequential, 100 q
#   B2. overhead control: S1=GPU,S2=GPU,S3=NPU with S3 in its own process (compare Phase 1c,
#       same mapping in-process)
#   C. Poisson load sweep for S1=GPU,S2=NPU,S3=NPU (3 rounds) + closed-loop 1/2/4 clients
# START_AT=HH:MM waits until that local time (tomorrow if already past).
set -u
SELF=$(realpath "$0")
cd "$(dirname "$SELF")"
if [ -n "${START_AT:-}" ]; then
  T=$(date -d "$START_AT" +%s); N=$(date +%s)
  [ "$T" -le "$N" ] && T=$(date -d "tomorrow $START_AT" +%s)
  sleep $((T - N))
fi
P=$(python3 -c "import yaml;print(yaml.safe_load(open('../config.yaml'))['python'])")
RUN=$($P -c "from rag_common import new_run_dir; print(new_run_dir('phase1d-npu-subprocess'))")
O=$RUN/results.jsonl
exec > "$RUN/run.log" 2>&1

mkdir "$RUN/code_snapshot"
cp ../config.yaml "$RUN/"
cp "$SELF" rag_common.py harness.py analyze_phase1c.py "$RUN/code_snapshot/"
git -C .. rev-parse HEAD > "$RUN/git_rev.txt"
git -C .. status --short >> "$RUN/git_rev.txt"
$P -m pip freeze > "$RUN/pip_freeze.txt"
{ date; uptime; free -h; df -h /; who; ps -eo user:14,pcpu,pmem,etime,cmd --sort=-pcpu | head -10 | cut -c1-160; } > "$RUN/env_start.txt"
echo "start $(date)"
H="$P harness.py"
E2E="--split dev --cands 20"

# B. sequential, 100 q, 1 round
for M in S1=NPU,S2=NPU,S3=NPU S1=NPU,S2=GPU,S3=NPU S1=NPU,S2=CPU,S3=NPU S1=GPU,S2=NPU,S3=NPU S1=CPU,S2=NPU,S3=NPU; do
  $H e2e --map $M $E2E --n 100 --round 1 --out $O
done
$H e2e --map S1=GPU,S2=GPU,S3=NPU $E2E --n 100 --s3-process always --round 1 --out $O
echo "B done $(date)"

# C. load sweep, same n rule as Phase 1c
n_for() { awk -v r="$1" 'BEGIN{n=100*r; if (n<150) n=150; printf "%d", n}'; }
M=S1=GPU,S2=NPU,S3=NPU
for R in 1 2 3; do
  for RATE in 0.5 1.0 1.5 2.0; do
    $H e2e --map $M $E2E --n $(n_for $RATE) --arrival poisson --rate $RATE --seed $R --round $R --out $O
  done
  if [ $R = 1 ]; then
    for C in 1 2 4; do
      $H e2e --map $M $E2E --n 100 --arrival closed --clients $C --round 1 --out $O
    done
  fi
  echo "C round $R done $(date)"
done

{ date; uptime; free -h; who; ps -eo user:14,pcpu,pmem,etime,cmd --sort=-pcpu | head -10 | cut -c1-160; } > "$RUN/env_end.txt"
$P analyze_phase1c.py "$RUN" || echo "analysis failed"
echo "PHASE1D_DONE $(date)"
