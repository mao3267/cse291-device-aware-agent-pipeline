# CSE 291A Project Proposal: Profile-Driven Device Placement for a RAG Agent Pipeline (Option 2)

## Research Question

An edge agent combines embedding, reranking, and generation, stages that differ in model size, input shape, and data movement. For a single-user edge device with low concurrency, which processor (CPU, iGPU, or NPU) should each pipeline stage run on, and does a heterogeneous stage-to-device mapping beat the best single device? Each model is loaded once on one processor; we do not route different requests to different devices.

## Pipeline and Setup

We use a retrieval-augmented question-answering pipeline with three stages:

```
Query → S1: Embedding → vector search → S2: Rerank → S3: LLM generation
```

- **Models (4-bit LLM, fixed across devices):** bge-small-en-v1.5 (S1), bge-reranker-base (S2), Qwen2.5-1.5B-Instruct (S3; 3B as a sensitivity check).
- **Data:** Wikipedia passage corpus (~10–50k passages) with SQuAD v1.1 questions for development and HotpotQA for held-out evaluation (several hundred questions each).
- **Hardware and runtimes:** one lab edge machine (ASUS NUC 16 Pro and/or GMKtec EVO-X2), using llama.cpp and OpenVINO. Device comparisons are made within one runtime, and model and precision are held fixed.
- **Candidate stage-to-device mappings:** each stage is assigned to exactly one device and its model stays resident there (at most 3³ = 27 mappings, pruned by the support matrix). Our initial hypotheses, to be tested rather than assumed:
  - S1 (small, batch-friendly) on NPU or CPU
  - S2 (rerank cost grows with candidates) on iGPU or NPU
  - S3 (decode is memory-bandwidth bound) on iGPU or CPU

  Since CPU, iGPU, and NPU share system memory, we also measure bandwidth contention when stages overlap.

## Phase 1: Measurement and Profiling (due Oct 28)

- **Support matrix:** for every stage and device, record whether it runs and whether the runtime silently falls back (e.g., NPU request executed on CPU). Unsupported NPU execution is reported as a finding.
- **Controlled sweeps (one factor at a time):**
  - prompt length (256 / 1k / 4k tokens)
  - output length (32 / 128 / 256)
  - embedding batch size (1 / 8 / 32)
  - rerank candidate count (10 / 50 / 100)
  - request overlap (sequential vs. lightly pipelined, 1–2 in flight)
- **Metrics:** per-stage and end-to-end latency, throughput, memory, initialization/compilation time, and data-transfer overhead, with cold start separated from warm runs. The deliverable is a set of per-stage device profiles that drive Phase 2.

## Phase 2: Profile-Driven Placement (due Dec 10)

- **Placement policy:** a simple performance model (sum of measured per-stage latencies plus inter-device transfer cost, plus a pipelining term when stages on different devices overlap) that selects one stage-to-device mapping per workload class (e.g., by candidate count and context length). The mapping is fixed at deployment; each model is loaded once, so there is no per-request routing or duplicate model loading, and memory footprint is part of the objective.
- **Baselines:** all-CPU, all-iGPU, all-NPU (where supported), the best single device, and an exhaustive-search oracle over all supported mappings, which tests whether the model picks a near-best mapping.
- **Evaluation:** held-out HotpotQA workloads in a single-user setting (sequential queries, plus light overlap of query *i+1*'s S1/S2 with query *i*'s S3). We report end-to-end latency (median/tail), throughput, memory, and answer quality (EM/F1, to confirm placement does not change outputs). Runs are repeated and variability is reported.
- **Negative results** (e.g., the best single device matches the best mapping) are reported as findings.

## Timeline and Deliverables

| Period | Milestone |
|---|---|
| Through Oct 28 | Pipeline, runner, lab profiling, Phase 1 writeup |
| Oct 29 – Nov 18 | Placement model and all-mapping runs |
| Nov 19 – Dec 10 | Extra runs, analysis, final report |
