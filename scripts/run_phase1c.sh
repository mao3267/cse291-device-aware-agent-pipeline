#!/bin/bash
# Phase 1c: fill the gaps left by Phase 1/1b, in a new write-once run folder.
#   cd /data/project-01 && START_AT=17:00 nohup bash scripts/run_phase1c.sh > /dev/null 2>&1 &
#
#   A. RAG prompt size: e2e with top-1 / top-8 passages (top-3 is in Phase 1)
#   B. Mapping coverage: the 18 S1 x S2 x S3 mappings with S3 on CPU or NPU
#      (S3 on GPU was measured in Phase 1); with it, all 27 mappings are covered
#   C. Load sweep: Poisson arrivals up to and past capacity, 4 candidate mappings,
#      3 interleaved rounds; plus closed-loop clients for the two S3=NPU mappings
# START_AT=HH:MM waits until that local time before creating the run folder.
set -u
SELF=$(realpath "$0")
cd "$(dirname "$SELF")"
if [ -n "${START_AT:-}" ]; then
  T=$(date -d "today $START_AT" +%s); N=$(date +%s)
  [ "$T" -gt "$N" ] && sleep $((T - N))
fi
P=$(python3 -c "import yaml;print(yaml.safe_load(open('../config.yaml'))['python'])")
RUN=$($P -c "from rag_common import new_run_dir; print(new_run_dir('phase1c-load-coverage'))")
O=$RUN/results.jsonl
exec > "$RUN/run.log" 2>&1

# provenance
mkdir "$RUN/code_snapshot"
cp ../config.yaml "$RUN/"
cp "$SELF" rag_common.py harness.py analyze_phase1c.py "$RUN/code_snapshot/"
git -C .. rev-parse HEAD > "$RUN/git_rev.txt"
git -C .. status --short >> "$RUN/git_rev.txt"
$P -m pip freeze > "$RUN/pip_freeze.txt"
{ date; uptime; free -h; df -h /; ps aux --sort=-%cpu | head -8 | cut -c1-160; } > "$RUN/env_start.txt"
echo "start $(date)"
H="$P harness.py"
E2E="--split dev --cands 20"

# A. RAG prompt size (sequential, 100 q). top-8 prompts need a larger NPU MAX_PROMPT_LEN.
for M in S1=GPU,S2=GPU,S3=GPU S1=GPU,S2=NPU,S3=GPU S1=GPU,S2=GPU,S3=NPU; do
  $H e2e --map $M $E2E --n 100 --topn 1 --round 1 --out $O
  $H e2e --map $M $E2E --n 100 --topn 8 --max-prompt-len 3072 --round 1 --out $O
done
echo "A rag-n done $(date)"

# B. Mapping coverage (sequential, 100 q, 1 round). S3=NPU first: faster.
for S3 in NPU CPU; do
  for S2 in GPU NPU CPU; do
    for S1 in GPU NPU CPU; do
      [ "$S1$S2" = "$S3$S3" ] && continue   # all-CPU / all-NPU are in Phase 1
      $H e2e --map S1=$S1,S2=$S2,S3=$S3 $E2E --n 100 --round 1 --out $O
    done
  done
done
echo "B coverage done $(date)"

# C. Load sweep. n = max(150, 100 x rate) queries per point; offered rates reach past
#    each mapping's capacity (all-GPU ~2.4 q/s, S3=NPU ~1.2-1.4 q/s).
#    S2=CPU mappings (capacity ~0.47 q/s) are dominated and left out.
n_for() { awk -v r="$1" 'BEGIN{n=100*r; if (n<150) n=150; printf "%d", n}'; }
sweep() {   # map rates...
  local M=$1; shift
  for RATE in "$@"; do
    $H e2e --map $M $E2E --n $(n_for $RATE) --arrival poisson --rate $RATE --seed $R --round $R --out $O
  done
}
for R in 1 2 3; do
  sweep S1=GPU,S2=GPU,S3=GPU 1.0 1.5 2.0 2.5 3.0
  sweep S1=GPU,S2=NPU,S3=GPU 1.0 1.5 2.0 2.5 3.0
  sweep S1=GPU,S2=GPU,S3=NPU 0.5 1.0 1.5 2.0
  sweep S1=GPU,S2=NPU,S3=NPU 0.5 1.0 1.5 2.0
  if [ $R = 1 ]; then
    for M in S1=GPU,S2=GPU,S3=NPU S1=GPU,S2=NPU,S3=NPU; do
      for C in 1 2 4; do
        $H e2e --map $M $E2E --n 100 --arrival closed --clients $C --round 1 --out $O
      done
    done
  fi
  echo "C round $R done $(date)"
done

{ date; uptime; free -h; ps aux --sort=-%cpu | head -8 | cut -c1-160; } > "$RUN/env_end.txt"
$P analyze_phase1c.py "$RUN" || echo "analysis failed"
echo "PHASE1C_DONE $(date)"
