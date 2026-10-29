# project-01 — CSE 291A: Profile-Driven Device Placement for a RAG Pipeline

This file **overrides `/data/CLAUDE.md`** for all work under `/data/project-01`.
Where the two conflict, follow this file. Anything not covered here falls back
to the global file.

**Standing rule (from the user):** whenever you change anything this file
describes — models, datasets, env, code layout, findings, TODOs, disk state —
update this file in the same step. Do not leave it stale.

**Self-contained rule (from the user, 2026-10-27):** every file this project
creates or uses lives **inside `/data/project-01/`**: code, models, datasets,
the Python venv, HF caches, run outputs, scratch. Do not write project files to
`/data/models`, `/data/datasets`, `/data/envs`, `/data/runs`, `/data/scratch`,
`~/.cache` or `/home/wuklab`. This overrides the global `/data` layout table.
Set `HF_HOME=/data/project-01/cache/hf` for any Hugging Face download or export,
so nothing lands in `~/.cache/huggingface`.

---

## Project

- Proposal: `proposal.md` (Option 2). RAG only — the Whisper/Stable Diffusion
  alternative was dropped; do not reintroduce it.
- Pipeline: `Query → S1 embed (bge-small-en-v1.5) → vector search → S2 rerank (bge-reranker-base) → S3 generate (Qwen2.5-1.5B-Instruct, 3B as sensitivity)`.
- Goal: per-stage CPU / iGPU / NPU profiles (Phase 1), then a profile-driven
  stage→device mapping vs. single-device baselines and an exhaustive oracle (Phase 2).
- Deadlines: **Phase 1 — Nov 4** (extended from Oct 28), Phase 2 / final report — Dec 10 (2026).

## Machine facts (verified 2026-10-27, supersede the global file)

- Kernel is **7.0.0-31-generic** (global file's 6.17 is stale). `xe` and `intel_vpu` load fine.
- 16 CPUs, no HT · AVX2 + AVX-VNNI · **no AVX-512, no AMX**.
- 96 GB DDR5 shared by CPU, iGPU, NPU (bandwidth contention is in scope).
- iGPU compute **is** installed (Level Zero + OpenCL, `intel-opencl-icd 26.18`).
- NPU user-space **is** installed (`intel-level-zero-npu 1.33`). OpenVINO lists CPU, GPU, NPU.
- Disk: `/` is **99% full (~8.6 GB free, 2026-10-27)**. Check `df -h /` before any download or export.
  `/data/scratch` holds ~64 GB (pre-existing, not yet reviewed, not this project's) — the obvious
  place to reclaim space, but ask the user before clearing it.

## Environment

- Python venv: **`/data/project-01/.venv`** (OpenVINO / GenAI / tokenizers 2026.4.1,
  optimum-intel 2.2.0, nncf 3.4.0, transformers 5.5.4, datasets 5.0.1).
  Use `.venv/bin/python`; do not install into envs in `/data/envs/`.
  Moved from `/home/wuklab/ph1` on 2026-10-27 (shebangs + `pyvenv.cfg` rewritten;
  imports and `optimum-cli` verified). If it ever breaks, recreate it in place, not elsewhere.
- Missing from `.venv`: `matplotlib` (needed for figures), `faiss-cpu` (optional —
  numpy brute force is enough at ~26k × 384), `evaluate` (EM/F1 can be hand-written).
  torch is the CUDA build (`+cu130`, ~3 GB wasted) — prefer CPU torch on rebuild.
- llama.cpp: `/data/repos/llama.cpp/build-vulkan/bin` (Vulkan iGPU) and `build-cpu`.
  **No NPU backend** — llama.cpp is a CPU/iGPU cross-check only. These shared
  binaries are a read-only system tool (exception to the self-contained rule);
  any GGUF files this project uses go in `models/`.

## Runtime rules

1. **OpenVINO is the primary runtime** — it is the only one covering all three
   devices. All cross-device comparisons are made within OpenVINO with model and
   precision held fixed.
2. Record every stage×device result in the support matrix, including failures
   and silent CPU fallback. A wrong-output device is a finding, not a bug to hide.
3. Separate cold start (load/compile) from warm runs in every measurement.
4. Do not change model or precision between devices within one comparison.

## Known status / findings

- Qwen2.5-1.5B OV int8 (`/data/models/qwen25_1_5b_ov_int8` — a pre-existing shared export,
  not part of this project; history only, asymmetric uint8 weights):
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
- **Phase 1 results (2026-10-27, FP16 encoders, INT4 LLM, OV 2026.4.1):**
  - No silent fallback for S1/S2 (EXECUTION_DEVICES). S1 is negligible (2–8 ms at batch 1).
  - S2 rerank (seq 256), warm median: 10 cands CPU 1071 / GPU 72 / NPU 82 ms; 100 cands CPU 10.3 s /
    GPU 519 ms / **NPU 14.2 s with a 21-min compile** (50 cands: 242 s compile). NPU breaks down at large shapes.
  - S2 on NPU changes rerank order: recall@3 0.83 vs 0.87, F1 0.34 vs 0.36 (GPU = CPU outputs exactly).
  - S3 (fresh prompts): TTFT 256/1k/4k tok — GPU 39/146/669 ms, NPU 374/785/2578, CPU 814/2988/13634;
    decode GPU 69–76 tok/s, NPU 47–58, CPU 40–57. GenAI prefix cache makes repeated prompts look instant
    on CPU/GPU — always vary prompts when measuring TTFT.
  - e2e (dev, 20 cands, top-3): best sequential = all-GPU 2.23 q/s, median 437 ms/query. Best overlap =
    S2 on NPU + S3 on GPU, 2.50 q/s (+5% vs all-GPU overlap 2.38) at the cost of the recall/F1 drop above.
    S2 on CPU caps at ~0.4 q/s. Std across 3 rounds ≤ 0.01 q/s.
  - NPU compile cache lives in `~/.cache/ze_intel_npu_cache` (system-managed); clear it to measure true NPU cold start.
- **Sequence length, padding and truncation (S1/S2), measured 2026-10-27 21:31** (ad-hoc check, not in
  a run folder; load avg 1.3). Encoders use static `[batch, seq]`, `padding="max_length"`, hard truncation
  (S2: `only_second`, so the passage is cut and the query is kept). Pad positions are fully computed.
  - S1 queries (seq 64): token length median 22 / max 39 (SQuAD dev), median 28 / max 61 (HotpotQA);
    0% truncated, 54–65% of compute is padding. Batch=1 latency is overhead-bound on GPU/NPU
    (GPU ~2.1 ms, NPU 1.8→2.4 ms, CPU 4.8→7.9 ms for seq 32→64). **Keep seq 64.**
  - S2 pairs (seq 256), the real e2e candidates (100 dev q × 20): median 176, p90 267, **12.6% truncated,
    30% of compute is padding**. Longest pair per query: median 313, so picking one shape per batch saves nothing.
  - S2 latency, 20 cands, seq 128 / 192 / 256: CPU 994 / 1510 / 2044 ms, GPU 48 / 73 / 99 ms,
    NPU 66 / 109 / 158 ms (NPU compile 27 s / 47 s at 128 / 192). Linear in seq with ~0 fixed cost per call
    (NPU worse than linear), so padding waste ≈ the same share of S2 latency.
  - Phase 1 keeps seq 256 on all devices for a fair comparison; report padding/truncation as a known property.
  - S1 sweep `batch` = number of **queries** per call (dev questions, seq 64), not passages. The e2e pipeline
    always runs S1 at batch 1; passages were embedded offline by `build_index.py` (not timed).
- llama.cpp Vulkan, Qwen2.5-3B Q4_K_M: pp256 ≈ 377 t/s, tg32 ≈ 14.5 t/s.

## TODO before Phase 1 profiling

- [x] Download + export bge-small-en-v1.5 and bge-reranker-base to OV IR (FP16).
- [x] Export Qwen2.5-1.5B to INT4 symmetric OV IR; NPU now correct.
- [ ] Export Qwen2.5-3B INT4 sym (needs ~6 GB of source weights in the HF cache — free disk first).
- [x] Fetch Wikipedia passages, SQuAD v1.1, HotpotQA → `rag-corpus/v1`.
- [ ] Install matplotlib in `.venv` (vector index: numpy brute force).
- [x] Passage index `datasets/rag-corpus-index/v1_bge-small-fp16_seq256` (`scripts/build_index.py`).
- [x] Pipeline runner + harness (`scripts/harness.py`), smoke-tested on all three devices.
- [x] Full Phase 1 matrix: `runs/2026-10-27_155708_wuklab-NUC16_phase1/` (15:57–18:28), analysis in its
  `analysis/` (read `analysis/NOTES.md` first). S3 sweep re-done with fresh prompts per rep:
  `runs/2026-10-27_182944_wuklab-NUC16_phase1-s3-fresh-prompts/`.
  Design decision (user, 2026-10-27): **S3 fixed on iGPU** for e2e; e2e = 9 mappings
  (S1, S2 ∈ CPU/GPU/NPU; S3=GPU) × sequential/overlap × 3 interleaved rounds + all-CPU/all-NPU refs.
- [ ] Phase 1 writeup (due Nov 4).
- [ ] Phase 2 idea: S2 length-bucketed sub-batches — sort the 20 candidates by length, split into
  2–3 sub-batches with pre-compiled static shapes (e.g. 128 + 256). Up to ~30% less S2 time, NPU-compatible;
  cost = extra compiles per shape. Re-run the seq check in a proper run folder before reporting numbers.

## Paths

Everything is under `/data/project-01/`. Heavy state is gitignored.

| What | Where (relative to project root) | Git | Rule |
|---|---|---|---|
| Code, docs, config | `scripts/`, `config.yaml`, `proposal.md`, `CLAUDE.md` | yes | normal commits |
| Python venv | `.venv/` | no | recreate in place |
| Model weights / exports | `models/<name>/` | no | append only; delete only with user confirmation |
| Datasets | `datasets/rag-<name>/` | no | append only; rebuild = new version dir |
| Experiment outputs | `runs/<date>_<host>_<name>/` | no | write-once (global run-folder recipe, with `RUN` under `runs/`) |
| HF download cache | `cache/hf/` (`HF_HOME`) | no | safe to delete |
| Throwaway files | `scratch/` | no | safe to delete |

### Artifacts on disk

| Stage | Path (under `models/`) | Source |
|---|---|---|
| S1 embed | `bge_small_en_v15_ov_fp16` (66 MB) | BAAI/bge-small-en-v1.5, `feature-extraction`, CLS pooling |
| S2 rerank | `bge_reranker_base_ov_fp16` (557 MB) | BAAI/bge-reranker-base, `text-classification` |
| S3 generate | `qwen25_1_5b_ov_int4_sym_cw` (870 MB) | Qwen/Qwen2.5-1.5B-Instruct |

Datasets (under `datasets/`): `rag-squad/raw` (SQuAD v1.1 train+dev parquet),
`rag-hotpotqa/raw` (HotpotQA distractor dev parquet), `rag-corpus/v1` built by
`scripts/build_corpus.py` (seed 0): **25,880 Wikipedia passages** (all SQuAD
train+dev paragraphs + the 10 paragraphs of each sampled HotpotQA question),
500 SQuAD-dev questions (1 gold pid each), 500 HotpotQA held-out questions
(2 gold pids each). Hashes in `manifest.json`. Rebuild = new version dir, never overwrite.

Models and datasets were moved here from `/data/models` and `/data/datasets`
on 2026-10-27 (same filesystem, plain `mv`; contents unchanged).

### Code layout

| Path | What |
|---|---|
| `config.yaml` | All paths (absolute, all under `/data/project-01`), model names, dataset locations, corpus params. Scripts read only this. |
| `scripts/build_corpus.py` | Raw SQuAD/HotpotQA → `rag-corpus/<ver>`. Refuses to overwrite. |
| `scripts/rag_common.py` | Shared pipeline code: config loading, static-shape encoders (S1/S2), numpy index, LLM (S3), EM/F1, `new_run_dir()`. Each model is loaded once on one device. |
| `scripts/build_index.py` | Embeds every corpus passage (`title: text`, seq 256, CLS+L2) → `datasets/<index>/` (`emb.npy`, `pids.npy`, `manifest.json`). Refuses to overwrite. |
| `scripts/harness.py` | `stage S1/S2/S3` sweeps and `e2e --map S1=..,S2=..,S3=.. [--overlap]`. JSONL rows with env info (load avg, free mem) and `EXECUTION_DEVICES`; rep −1 = cold. Default `--out` is `scratch/`. |
| `scripts/run_phase1.sh` | Full Phase 1 matrix into a new `runs/<date>_<host>_phase1/` with config, git rev, pip freeze. |
| `scripts/smoke.py` | Support-matrix smoke test (dummy inputs). |

- Resolve paths in code only through `config.yaml` (`cfg["paths"][...]`, `cfg["python"]`).
  Do not hardcode `/data/...` paths in scripts. Every path in `config.yaml` must
  point inside `/data/project-01/`.
- This directory is a **standalone git repo** (initialized 2026-10-27, branch `main`,
  no remote yet), separate from `edge-ai`. Use the global commit prefixes
  (`docs:` `scripts:` `configs:` `exp:` `fix:` `chore:` `paper:`), one per commit.
  Local author is set to `chenvincent610 <chenvincent610@gmail.com>`.

## Still in force from the global file

Run folders are write-once · never delete models / datasets / runs without explicit
user confirmation (now meaning the dirs inside this project) · no secrets in files
(use a gitignored `.env`) · commit-prefix and tidiness rules.

## Home-dir cleanup (2026-10-27)

Files an earlier session left in `/home/wuklab` were moved into the project:
`runs/2026-10-27_145900_wuklab-NUC16_smoke-fp32-encoders/` (first support-matrix
smoke test; FP32 encoders, see its README) and
`scratch/home-leftovers-2026-10-27/` (duplicate model exports ~2.1 GB, an older
11,836-passage corpus, old harness code). That scratch folder was deleted on
2026-10-27 with user confirmation (superseded; freed ~2.1 GB).
