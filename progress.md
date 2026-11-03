# project-01 progress vs. the Option 2 spec

Spec: CSE 291A "Option 2: CPU/GPU/NPU-Aware Agent Pipelines" (Phase 1 due **Nov 4**, Phase 2 + final
report due **Dec 10**, 2026). This file tracks every requirement in the spec: status, evidence, and
what is left. Update it together with `CLAUDE.md` whenever a run finishes or a gap closes.

Status: ✅ done · 🔄 running / measured but not yet analyzed · ⚠️ partial · ❌ not started · ⏭️ skipped by decision

Run folders (under `runs/`):
- **P1** = `2026-10-27_155708_wuklab-NUC16_phase1` — stage sweeps + 9-mapping e2e (seq/overlap, 3 rounds).
- **S3R** = `2026-10-27_182944_wuklab-NUC16_phase1-s3-fresh-prompts` — S3 sweep, fresh prompt per rep.
- **P1b** = `2026-10-28_212940_wuklab-NUC16_phase1-supplement` — ✅ 21:29–22:46, no errors
  (`scripts/run_phase1b.sh`): memory, placement, breakdown, true cold, concurrency, contention. Findings in `CLAUDE.md`.

## Phase 1 — measurement and profiling

| Requirement | Status | Evidence / notes | Remaining |
|---|---|---|---|
| Pipeline with ≥ 2 meaningful stages | ✅ | S1 embed → search → S2 rerank → S3 generate | — |
| Profile supported stages on every device | ✅ | P1, S3R: 3 stages × CPU/GPU/NPU | — |
| Vary prompt / output length | ✅ | S3: prompt 256/1k/4k × out 32/128/256 (GPU), out 128 (CPU/NPU) | — |
| Vary batch size | ⚠️ | S1 batch 1/8/32 = **queries** per call; e2e always uses batch 1 | Optional: passage-embedding batch (indexing workload) |
| Vary candidate count | ✅ | S2: 10/50/100 (P1), 10/20/50/100 (P1b) | — |
| Vary concurrent requests | ✅ | P1b: closed-loop 1/2/4 clients, Poisson 1.0/1.5/2.0/2.3 q/s, 3 mappings × 2 rounds. Capacity all-GPU 2.38 q/s, S2=NPU 2.49. Light load → all-GPU has lower latency; heavy load (≥ 2 q/s) → S2 on NPU | — |
| Per-stage and e2e latency | ✅ | P1 stage tables + e2e per-stage timing | — |
| Throughput | ✅ | e2e qps; S3 tok/s | — |
| Memory | ✅ | P1b `memory_load.csv`, `e2e_memory.csv`: S2 GPU 0.6–1.3 GB GTT, S2 NPU 0.63–0.98 GB, S3 GPU 1.0 GB, S3 NPU 1.35–2.03 GB (grows with MAX_PROMPT_LEN); e2e process 1.7–2.4 GB RSS | — |
| Initialization / compilation time | ✅ | P1b true cold ×3 (spread ≤ 0.3 s) vs warm cache: S2 NPU 68 vs 0.12 s, S3 NPU 10.8 vs 4.9 s, S3 GPU 3.0 vs 1.6 s, S1 GPU/NPU ≈ 1.0 s cold; CPU has no cache | — |
| Data-transfer overhead | ✅ | P1b breakdown: GPU S2 input copy 3–13 ms (≤ 10% of S2), NPU/S1 < 0.4 ms; S2 tokenization 8–35 ms is the larger host cost; S3 tokenize/detokenize < 2 ms; inter-stage payload 1.5 / 14.6 / 2.4 KB per query | — |
| Warm vs cold separated | ✅ | rep −1 = first run after compile; `load_s` separate | — |
| Energy (only if reliable) | ⏭️ skipped | **Out of scope by user decision (2026-10-28).** RAPL `energy_uj` is root-only and `sudo` needs a password. The harness still records `energy_j` (null) if it ever becomes readable | Report states energy was not measured (no reliable non-root instrumentation) |
| Actual device placement / fallback | ✅ | S1/S2: `EXECUTION_DEVICES`, no fallback. P1b `placement.csv`: every stage × device ran where asked (GPU cycles only on GPU, NPU busy 91–95% of latency only on NPU, neither on CPU), S3 included | — |
| Support matrix | ✅ | `P1/analysis/support_matrix.csv`; P1b adds the activity-based placement column | — |
| Profiles: which device under which condition | ✅ | Device-preference table in `paper/merged/phase1_report.tex` (Table 8) | — |

## Phase 2 — profile-driven placement

| Requirement | Status | Notes |
|---|---|---|
| Placement policy (lookup table / perf model) | ❌ | Inputs: prompt length, candidate count, device queue state, transfer cost, model residency, memory |
| Baselines: fixed placements, best static config on dev | ⚠️ | P1 e2e has all 9 S1×S2 mappings (S3=GPU) + all-CPU/all-NPU on dev = the baseline set |
| Evaluate on new workload mixes (HotpotQA held-out, mixed prompt lengths) | ❌ | `--split eval` exists; no mixed workload yet |
| Concurrent load | ⚠️ | P1b arrival modes (closed/Poisson) work; the crossover (all-GPU at light load, S2=NPU at heavy load) is a policy input |
| Include switching and transfer overhead | ❌ | P1b breakdown gives the transfer term; switching = model load_s (P1b true cold) |
| Policy overhead | ❌ | Time the policy decision per query |
| Idea: S2 length-bucketed sub-batches | ❌ | ~30% of S2 compute is padding (seq check, CLAUDE.md); NPU-compatible |
| Idea: NPU S3 MAX_PROMPT_LEN per workload class | ❌ | NPU TTFT depends on the compiled MAX_PROMPT_LEN (256-token prompt: 99 ms at 384 vs 374 ms at 4224); P1b measures it |
| Sensitivity: Qwen2.5-3B | ❌ | Needs INT4-sym export (disk now 38 GB free) |

## Evaluation and deliverables

| Requirement | Status | Notes |
|---|---|---|
| e2e latency, throughput | ✅ (dev) | Eval split not run |
| Memory use | ✅ | P1b |
| Policy overhead | ❌ | Phase 2 |
| Output quality | ✅ | EM/F1, recall@20/@3; S2 on NPU lowers recall@3 0.87 → 0.83, F1 0.36 → 0.34 |
| Model / precision matched across devices | ✅ | Same IRs and shapes on every device; 3B kept separate |
| Negative results | ✅ | NPU S2 breaks at large shapes (100 cands: 14 s, 21-min compile); NPU changes rerank order; overlap on one device gives ~no gain |
| Causes of device differences | ✅ | GenAI prefix cache, NPU compile, padding waste, NPU MAX_PROMPT_LEN (TTFT 99.6 vs 366 ms), and **shared-memory contention, not the GIL** (P1b, 3 rounds: GPU decode 14.0 → 22.1 ms/token with NPU rerank in a thread or a process, 14.2 with a tokenize-only thread, 15.2 with CPU rerank; NPU decode 18.6 → 31.5 with GPU rerank) |
| Deliverables: harness, profiles, policy, pipeline, analysis | ⚠️ | Harness + pipeline in `scripts/` (committed); profiles in run folders; Phase 1 report `paper/merged/phase1_report.{tex,pdf}`; policy not yet |

## Next steps (in order)
1. Phase 1 writeup (Nov 4): merged draft done (`paper/merged/`, 5 pages incl. timeline appendix); team review and final page budget left.
2. Phase 2: policy design on dev profiles → evaluate on HotpotQA + mixed workloads + concurrent load.
