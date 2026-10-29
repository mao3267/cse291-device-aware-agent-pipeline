#!/bin/bash
# Phase 1 supplement in a new write-once run folder: memory, actual placement,
# transfer breakdown, true cold start, concurrency, contention.
#   cd /data/project-01 && nohup bash scripts/run_phase1b.sh > /dev/null 2>&1 &
# Complements runs/2026-10-27_155708_wuklab-NUC16_phase1 (same models, shapes, corpus).
set -u
SELF=$(realpath "$0")
cd "$(dirname "$SELF")"
P=$(python3 -c "import yaml;print(yaml.safe_load(open('../config.yaml'))['python'])")
RUN=$($P -c "from rag_common import new_run_dir; print(new_run_dir('phase1-supplement'))")
O=$RUN/results.jsonl
exec > "$RUN/run.log" 2>&1

# provenance
mkdir "$RUN/code_snapshot"
cp ../config.yaml "$RUN/"
cp "$SELF" rag_common.py harness.py analyze_phase1.py "$RUN/code_snapshot/"
git -C .. rev-parse HEAD > "$RUN/git_rev.txt"
git -C .. status --short >> "$RUN/git_rev.txt"
$P -m pip freeze > "$RUN/pip_freeze.txt"
{ uptime; free -h; df -h /; } > "$RUN/env_start.txt"
echo "start $(date)"
H="$P harness.py"

# A. Stage sweeps with memory peaks, device activity and S1/S2 host-side breakdown
for D in CPU GPU NPU; do
  $H stage S1 --device $D --batch 1,8,32 --seq 64 --reps 10 --out $O
  if [ $D = NPU ]; then C=10,20,50; else C=10,20,50,100; fi   # NPU 100: 21-min compile, known
  $H stage S2 --device $D --cands $C --seq 256 --reps 10 --out $O
  $H stage S3 --device $D --prompt-len 256,1024,4096 --out-len 128 --reps 3 --out $O
done
# NPU prefill is compiled for MAX_PROMPT_LEN: one compile per prompt length
for PL in 256 1024 4096; do
  $H stage S3 --device NPU --prompt-len $PL --out-len 128 --reps 3 --out $O
done
echo "A stages done $(date)"

# B. True cold start: NPU driver cache bypassed, GPU kernel cache off. 3 repeats.
for R in 1 2 3; do
  for D in CPU GPU NPU; do
    NEO_CACHE_PERSISTENT=0 $H stage S1 --device $D --batch 1 --reps 2 --cold --out $O
    NEO_CACHE_PERSISTENT=0 $H stage S2 --device $D --cands 20 --reps 2 --cold --out $O
    NEO_CACHE_PERSISTENT=0 $H stage S3 --device $D --prompt-len 1024 --out-len 32 --reps 1 --cold --out $O
  done
done
echo "B cold done $(date)"

# C. Concurrency (dev, 100 q, 20 cands): closed-loop clients and open-loop Poisson
#    arrivals; 2 rounds, interleaved across mappings
E2E="--split dev --n 100 --cands 20"
MAPS="S1=GPU,S2=GPU,S3=GPU S1=NPU,S2=NPU,S3=GPU S1=NPU,S2=GPU,S3=GPU"
for R in 1 2; do
  for M in $MAPS; do
    for C in 1 2 4; do
      $H e2e --map $M $E2E --arrival closed --clients $C --round $R --out $O
    done
    for RATE in 1.0 1.5 2.0 2.3; do
      $H e2e --map $M $E2E --arrival poisson --rate $RATE --seed $R --round $R --out $O
    done
  done
  echo "C round $R done $(date)"
done

# D. Contention: S3 decode measured while an S2 aggressor runs elsewhere.
#    thread vs process separates GIL effects from shared-memory/bandwidth contention.
CT="--prompt-len 1024 --out-len 128 --reps 5"
for R in 1 2 3; do
  $H contend --device GPU $CT --aggressor none --round $R --out $O
  $H contend --device GPU $CT --agg-device NPU --aggressor rerank --mode thread --round $R --out $O
  $H contend --device GPU $CT --agg-device NPU --aggressor rerank --mode process --round $R --out $O
  $H contend --device GPU $CT --agg-device NPU --aggressor tokenize --mode thread --round $R --out $O
  $H contend --device GPU $CT --agg-device CPU --aggressor rerank --mode process --round $R --out $O
  $H contend --device NPU $CT --aggressor none --round $R --out $O
  $H contend --device NPU $CT --agg-device GPU --aggressor rerank --mode process --round $R --out $O
done
echo "D contention done $(date)"

{ uptime; free -h; } > "$RUN/env_end.txt"
$P analyze_phase1.py "$RUN" || echo "analysis failed"
echo "PHASE1B_DONE $(date)"
