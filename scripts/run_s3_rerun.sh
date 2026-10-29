#!/bin/bash
# S3 sweep rerun with a fresh prompt per rep (the first Phase 1 run reused one
# prompt per config, so CPU/GPU warm TTFT measured prefix-cache hits).
#   cd /data/project-01 && nohup bash scripts/run_s3_rerun.sh > /dev/null 2>&1 &
set -u
SELF=$(realpath "$0")
cd "$(dirname "$SELF")"
P=$(python3 -c "import yaml;print(yaml.safe_load(open('../config.yaml'))['python'])")
RUN=$($P -c "from rag_common import new_run_dir; print(new_run_dir('phase1-s3-fresh-prompts'))")
O=$RUN/results.jsonl
exec > "$RUN/run.log" 2>&1
mkdir "$RUN/code_snapshot"
cp ../config.yaml "$RUN/"
cp "$SELF" rag_common.py harness.py "$RUN/code_snapshot/"
{ uptime; free -h; } > "$RUN/env_start.txt"

$P harness.py stage S3 --device GPU --prompt-len 256,1024,4096 --out-len 32,128,256 --reps 5 --out $O
for D in CPU NPU; do
  $P harness.py stage S3 --device $D --prompt-len 256,1024,4096 --out-len 128 --reps 3 --out $O
done
{ uptime; free -h; } > "$RUN/env_end.txt"
echo S3_RERUN_DONE
