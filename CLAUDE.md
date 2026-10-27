# project-01 — CSE 291A: Profile-Driven Device Placement for a RAG Pipeline

This file **overrides `/data/CLAUDE.md`** for all work under `/data/project-01`.
Where the two conflict, follow this file. Anything not covered here falls back
to the global file.

**Standing rule (from the user):** whenever you change anything this file
describes — models, datasets, env, code layout, findings, TODOs, disk state —
update this file in the same step. Do not leave it stale.

---

## Project

- Proposal: `proposal.md` (Option 2). RAG only — the Whisper/Stable Diffusion
  alternative was dropped; do not reintroduce it.
- Pipeline: `Query → S1 embed (bge-small-en-v1.5) → vector search → S2 rerank (bge-reranker-base) → S3 generate (Qwen2.5-1.5B-Instruct, 3B as sensitivity)`.
- Goal: per-stage CPU / iGPU / NPU profiles (Phase 1), then a profile-driven
  stage→device mapping vs. single-device baselines and an exhaustive oracle (Phase 2).
- Deadlines: **Phase 1 — Oct 28**, Phase 2 / final report — Dec 10 (2026).

## Machine facts (verified 2026-10-27, supersede the global file)

- Kernel is **7.0.0-31-generic** (global file's 6.17 is stale). `xe` and `intel_vpu` load fine.
- 16 CPUs, no HT · AVX2 + AVX-VNNI · **no AVX-512, no AMX**.
- 96 GB DDR5 shared by CPU, iGPU, NPU (bandwidth contention is in scope).
- iGPU compute **is** installed (Level Zero + OpenCL, `intel-opencl-icd 26.18`).
- NPU user-space **is** installed (`intel-level-zero-npu 1.33`). OpenVINO lists CPU, GPU, NPU.
- Disk: `/` is **99% full (~8.6 GB free, 2026-10-27)**. Check `df -h /` before any download or export.
  `/data/scratch` holds ~64 GB (pre-existing, not yet reviewed) — the obvious place to reclaim space.

## Environment

- Python venv: **`/home/wuklab/ph1`** (OpenVINO / GenAI / tokenizers 2026.4.1,
  optimum-intel 2.2.0, nncf 3.4.0, transformers 5.5.4, datasets 5.0.1).
  Use `/home/wuklab/ph1/bin/python`; do not install into other envs in `/data/envs/`.
- Missing from ph1: `matplotlib` (needed for figures), `faiss-cpu` (optional —
  numpy brute force is enough at ~26k × 384), `evaluate` (EM/F1 can be hand-written).
  torch is the CUDA build (`+cu130`, ~3 GB wasted) — prefer CPU torch on rebuild.
- llama.cpp: `/data/repos/llama.cpp/build-vulkan/bin` (Vulkan iGPU) and `build-cpu`.
  **No NPU backend** — llama.cpp is a CPU/iGPU cross-check only.

## Runtime rules

1. **OpenVINO is the primary runtime** — it is the only one covering all three
   devices. All cross-device comparisons are made within OpenVINO with model and
   precision held fixed.
2. Record every stage×device result in the support matrix, including failures
   and silent CPU fallback. A wrong-output device is a finding, not a bug to hide.
3. Separate cold start (load/compile) from warm runs in every measurement.
4. Do not change model or precision between devices within one comparison.

## Known status / findings

- Qwen2.5-1.5B OV int8 (`/data/models/qwen25_1_5b_ov_int8`, asymmetric uint8 weights):
  CPU ✓ (0.5 s/16 tok), iGPU ✓ (0.3 s/16 tok), **NPU compiles but emits garbage**
  (OV 2026.3 and 2026.4.1).
- **Fixed by symmetric INT4 channel-wise export** (`qwen25_1_5b_ov_int4_sym_cw`,
  `--weight-format int4 --sym --ratio 1.0 --group-size -1`; lm_head stays int8_asym).
  Smoke test (greedy, 32 tok, warm): CPU 0.16 s, iGPU 0.13 s, NPU 0.50 s — all three
  answer correctly. Load: CPU 0.7 s, iGPU 1.7 s, NPU 1.3 s. This is the S3 model.
- bge-small-en-v1.5 and bge-reranker-base OV FP16, static shape [2,128]: run on
  CPU / iGPU / NPU with no fallback (EXECUTION_DEVICES checked); embeddings cos = 1.000
  vs CPU, reranker logits agree within 0.01. Compile: embed 0.4 / 0.7 / 1.2 s,
  rerank 2.9 / 3.7 / 3.7 s (CPU / GPU / NPU). NPU needs static shapes — reshape first.
- llama.cpp Vulkan, Qwen2.5-3B Q4_K_M: pp256 ≈ 377 t/s, tg32 ≈ 14.5 t/s.

## TODO before Phase 1 profiling

- [x] Download + export bge-small-en-v1.5 and bge-reranker-base to OV IR (FP16).
- [x] Export Qwen2.5-1.5B to INT4 symmetric OV IR; NPU now correct.
- [ ] Export Qwen2.5-3B INT4 sym (needs ~6 GB of source weights in the HF cache — free disk first).
- [x] Fetch Wikipedia passages, SQuAD v1.1, HotpotQA → `rag-corpus/v1`.
- [ ] Install matplotlib in ph1 (vector index: numpy brute force).
- [ ] Pipeline runner, support matrix, Phase 1 sweeps.

## Paths

| What | Where |
|---|---|
| Project code + docs | `/data/project-01/` |
| Model weights / exports | `/data/models/` (append only) |
| Datasets | `/data/datasets/rag-<name>/` (append only) |
| Experiment outputs | `/data/runs/<date>_<host>_p01-<name>/` (write-once, global run-folder recipe) |
| Throwaway files | `/data/scratch/` (HF download cache: `/data/scratch/hf-cache`) |

### Artifacts on disk

| Stage | Path (under `/data/models`) | Source |
|---|---|---|
| S1 embed | `bge_small_en_v15_ov_fp16` (66 MB) | BAAI/bge-small-en-v1.5, `feature-extraction`, CLS pooling |
| S2 rerank | `bge_reranker_base_ov_fp16` (557 MB) | BAAI/bge-reranker-base, `text-classification` |
| S3 generate | `qwen25_1_5b_ov_int4_sym_cw` (870 MB) | Qwen/Qwen2.5-1.5B-Instruct |

Datasets (under `/data/datasets`): `rag-squad/raw` (SQuAD v1.1 train+dev parquet),
`rag-hotpotqa/raw` (HotpotQA distractor dev parquet), `rag-corpus/v1` built by
`scripts/build_corpus.py` (seed 0): **25,880 Wikipedia passages** (all SQuAD
train+dev paragraphs + the 10 paragraphs of each sampled HotpotQA question),
500 SQuAD-dev questions (1 gold pid each), 500 HotpotQA held-out questions
(2 gold pids each). Hashes in `manifest.json`. Rebuild = new version dir, never overwrite.

### Code layout

| Path | What |
|---|---|
| `config.yaml` | All paths, model names, dataset locations, corpus params. Scripts read only this. |
| `scripts/build_corpus.py` | Raw SQuAD/HotpotQA → `rag-corpus/<ver>`. Refuses to overwrite. |

- Resolve these in code from one place (e.g. a `config.yaml` at project root);
  do not scatter hardcoded `/data/...` paths across scripts.
- This directory is a **standalone git repo** (initialized 2026-10-27, branch `main`,
  no remote yet), separate from `edge-ai`. Use the global commit prefixes
  (`docs:` `scripts:` `configs:` `exp:` `fix:` `chore:` `paper:`), one per commit.
  Local author is set to `chenvincent610 <chenvincent610@gmail.com>`.

## Still in force from the global file

Never delete from `/data/{models,datasets,runs,backups}` without explicit user
confirmation · run folders are write-once · only `/data/scratch` is free to clear ·
no secrets in files (use `.env`).
