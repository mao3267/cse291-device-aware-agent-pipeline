# project-01 progress vs. the Option 2 spec

Spec: CSE 291A "Option 2: CPU/GPU/NPU-Aware Agent Pipelines" (Phase 1 due **Nov 4**, Phase 2 + final
report due **Dec 10**, 2026). This file tracks every requirement in the spec: status, evidence, and
what is left. Update it together with `CLAUDE.md` whenever a run finishes or a gap closes.

Status: ✅ done · 🔄 running / measured but not yet analyzed · ⚠️ partial · ❌ not started

Run folders (under `runs/`):
- **P1** = `2026-10-27_155708_wuklab-NUC16_phase1` — stage sweeps + 9-mapping e2e (seq/overlap, 3 rounds).
- **S3R** = `2026-10-27_182944_wuklab-NUC16_phase1-s3-fresh-prompts` — S3 sweep, fresh prompt per rep.
- **P1b** = `2026-10-28_212940_wuklab-NUC16_phase1-supplement` — 🔄 started 2026-10-28 21:29
  (`scripts/run_phase1b.sh`, ~1.5–2 h): memory, placement, breakdown, true cold, concurrency, contention.

## Phase 1 — measurement and profiling

| Requirement | Status | Evidence / notes | Remaining |
|---|---|---|---|
| Pipeline with ≥ 2 meaningful stages | ✅ | S1 embed → search → S2 rerank → S3 generate | — |
| Profile supported stages on every device | ✅ | P1, S3R: 3 stages × CPU/GPU/NPU | — |
| Vary prompt / output length | ✅ | S3: prompt 256/1k/4k × out 32/128/256 (GPU), out 128 (CPU/NPU) | — |
| Vary batch size | ⚠️ | S1 batch 1/8/32 = **queries** per call; e2e always uses batch 1 | Optional: passage-embedding batch (indexing workload) |
| Vary candidate count | ✅ | S2: 10/50/100 (P1), 10/20/50/100 (P1b) | — |
| Vary concurrent requests | 🔄 | P1: seq vs overlap only. P1b: closed-loop 1/2/4 clients, Poisson 1.0/1.5/2.0/2.3 q/s, 3 mappings × 2 rounds | Analyze P1b |
| Per-stage and e2e latency | ✅ | P1 stage tables + e2e per-stage timing | — |
| Throughput | ✅ | e2e qps; S3 tok/s | — |
| Memory | 🔄 | Was one RSS sample per config. P1b: per stage × device load footprint and run peaks of process RSS, system used, GPU GTT (xe fdinfo), NPU memory (intel_vpu fdinfo) | Analyze P1b |
| Initialization / compilation time | ✅/🔄 | `load_s` everywhere. P1 NPU numbers partly hit the driver cache | P1b true cold (NPU cache bypass, GPU `NEO_CACHE_PERSISTENT=0`) ×3 |
| Data-transfer overhead | 🔄 | P1b: S1/S2 split into tokenize / set_input / infer / get_output, plus infer with zero-copy inputs (difference = plugin input copy); bytes handed between stages; S3 GenAI tokenize/infer/detokenize. Smoke test: S2 GPU input copy ≈ 13 ms of 107 ms; S2 tokenization 11–16 ms on every device | Analyze P1b |
| Warm vs cold separated | ✅ | rep −1 = first run after compile; `load_s` separate | — |
| Energy (only if reliable) | ❌ | RAPL `energy_uj` is root-only; `sudo` needs a password. Code records `energy_j` automatically once readable | User: `sudo chmod o+r /sys/class/powercap/intel-rapl:*/energy_uj` (resets at reboot), then re-run; else report "no reliable instrumentation" (package-level only) |
| Actual device placement / fallback | ✅/🔄 | S1/S2: `EXECUTION_DEVICES`, no fallback. **S3 now verified** (smoke test): GPU compute cycles only on GPU runs, NPU busy ≈ 96% of NPU S3 wall time, zero activity on CPU runs. P1b `placement` table covers all stage × device | Analyze P1b |
| Support matrix | ✅ | `P1/analysis/support_matrix.csv`; P1b adds the activity-based placement column | — |
| Profiles: which device under which condition | ⚠️ | Numbers exist (P1, S3R, seq-length check in CLAUDE.md) | Write the condition → device summary for the Phase 1 report |

## Phase 2 — profile-driven placement

| Requirement | Status | Notes |
|---|---|---|
| Placement policy (lookup table / perf model) | ❌ | Inputs: prompt length, candidate count, device queue state, transfer cost, model residency, memory |
| Baselines: fixed placements, best static config on dev | ⚠️ | P1 e2e has all 9 S1×S2 mappings (S3=GPU) + all-CPU/all-NPU on dev = the baseline set |
| Evaluate on new workload mixes (HotpotQA held-out, mixed prompt lengths) | ❌ | `--split eval` exists; no mixed workload yet |
| Concurrent load | 🔄 | P1b arrival modes are the base for this |
| Include switching and transfer overhead | ❌ | P1b breakdown gives the transfer term; switching = model load_s (P1b true cold) |
| Policy overhead | ❌ | Time the policy decision per query |
| Idea: S2 length-bucketed sub-batches | ❌ | ~30% of S2 compute is padding (seq check, CLAUDE.md); NPU-compatible |
| Idea: NPU S3 MAX_PROMPT_LEN per workload class | ❌ | NPU TTFT depends on the compiled MAX_PROMPT_LEN (256-token prompt: 99 ms at 384 vs 374 ms at 4224); P1b measures it |
| Sensitivity: Qwen2.5-3B | ❌ | Needs INT4-sym export (disk now 38 GB free) |

## Evaluation and deliverables

| Requirement | Status | Notes |
|---|---|---|
| e2e latency, throughput | ✅ (dev) | Eval split not run |
| Memory use | 🔄 | P1b |
| Policy overhead | ❌ | Phase 2 |
| Output quality | ✅ | EM/F1, recall@20/@3; S2 on NPU lowers recall@3 0.87 → 0.83, F1 0.36 → 0.34 |
| Model / precision matched across devices | ✅ | Same IRs and shapes on every device; 3B kept separate |
| Negative results | ✅ | NPU S2 breaks at large shapes (100 cands: 14 s, 21-min compile); NPU changes rerank order; overlap on one device gives ~no gain |
| Causes of device differences | ⚠️/🔄 | Known: GenAI prefix cache, NPU compile, padding waste, NPU MAX_PROMPT_LEN. Smoke test: S3 slowdown under overlap is **shared-memory contention, not the GIL** (GPU decode 14.0 → 22.1 ms/token with NPU rerank in a thread *or* a separate process; tokenize-only thread: 14.2) — P1b confirms with 3 rounds + CPU/NPU variants |
| Deliverables: harness, profiles, policy, pipeline, analysis | ⚠️ | Harness + pipeline in `scripts/` (committed); profiles in run folders; policy and report not yet |

## Next steps (in order)
1. When P1b finishes: read `P1b/analysis/summary.md`, check every 🔄 row above, update this file and `CLAUDE.md`.
2. Phase 1 writeup (Nov 4): support matrix + placement evidence, per-stage profiles, memory, cold start,
   transfer breakdown, concurrency, contention cause, negative results, energy statement.
3. Phase 2: policy design on dev profiles → evaluate on HotpotQA + mixed workloads + concurrent load.
