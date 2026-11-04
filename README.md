# Profile-Driven Device Placement for a RAG Pipeline

CSE 291A (Fall 2026), Option 2. Which processor should each stage of an edge RAG pipeline run on, when the CPU, integrated GPU (iGPU) and NPU share one DRAM?

```
Query → S1 embed → vector search → S2 rerank → S3 generate
```

| Stage | Model | Format |
|---|---|---|
| S1 | bge-small-en-v1.5 | OpenVINO FP16, static shape `[b, 64]` |
| Search | exact cosine similarity (numpy, CPU, about 4 ms) | not a placement choice |
| S2 | bge-reranker-base | OpenVINO FP16, static shape `[k, 256]` |
| S3 | Qwen2.5-1.5B-Instruct | OpenVINO INT4, symmetric channel-wise |

Each model is loaded once on one device; requests are not routed per query.

**Status:** Phase 1 (measurement and profiling) is done. Phase 2 (placement policy, due Dec 10, 2026) has not started.

## Phase 1 results

Report: [`paper/phase1_4page/phase1_report.pdf`](paper/phase1_4page/phase1_report.pdf). Requirement checklist: [`progress.md`](progress.md).

Sequential end-to-end runs on 100 SQuAD dev questions (20 candidates, top-3 passages, S1 on the iGPU):

| S2 | S3 | Median (ms) | q/s | recall@3 | F1 |
|---|---|---:|---:|---:|---:|
| iGPU | iGPU | **437** | **2.23** | 0.87 | 0.36 |
| NPU | iGPU | 573 | 1.73 | 0.82 | 0.34 |
| iGPU | NPU | 843 | 1.19 | 0.87 | 0.38 |
| iGPU | CPU | 1866 | 0.52 | 0.87 | 0.34 |
| CPU | CPU | 3845 | 0.26 | 0.87 | 0.34 |

- All 9 stage × device pairs run with no silent fallback, and all 27 mappings were measured.
- All-iGPU is the best mapping overall. The S1 device changes latency by at most 22 ms.
- An NPU reranker lowers recall@3 and helps only near saturation: +4% closed-loop throughput, with the same Poisson capacity within noise (about 2.4 q/s).
- Stages running on the iGPU and NPU at once slow each other through shared memory bandwidth, not the Python GIL (S3 decode +58%).
- Quality differences come from 100 questions and have no confidence intervals. Energy was not measured.

## Repository layout

| Path | Contents |
|---|---|
| `config.yaml` | Paths, model names and dataset locations; scripts read only this file |
| `scripts/rag_common.py` | Shared pipeline code: encoders, index, LLM, `RemoteLLM`, EM/F1, run folders |
| `scripts/harness.py` | Measurement harness |
| `scripts/build_corpus.py`, `scripts/build_index.py` | Build the corpus and the passage index |
| `scripts/smoke.py` | Support-matrix smoke test |
| `scripts/run_phase1*.sh`, `scripts/run_s3_rerun.sh` | Experiment drivers, one write-once run folder each |
| `scripts/analyze_phase1.py`, `scripts/analyze_phase1c.py` | Tables and figures from a run folder |
| `paper/` | Phase 1 report: `phase1_4page/` (current), `merged/` (5-page draft) |
| `proposal.md`, `progress.md`, `CLAUDE.md` | Proposal, requirement checklist, full project notes |

Models, datasets, caches and `runs/` are gitignored and stay on the lab machine.

## Setup

Tested on an Intel Core Ultra 7 356H (Arc iGPU, NPU, 96 GB DDR5) with Ubuntu 24.04, kernel 7.0 and NPU driver 1.33.0.

```bash
python3 -m venv .venv
.venv/bin/pip install openvino==2026.4.1 openvino-genai==2026.4.1.0 openvino-tokenizers==2026.4.1.0 \
  optimum-intel==2.2.0 nncf==3.4.0 transformers==5.5.4 datasets==5.0.1 numpy pandas matplotlib
```

There is no requirements file; these are the main packages. Each run folder holds the exact `pip_freeze.txt`.

`config.yaml` uses absolute paths under `/data/project-01`. Change them to run elsewhere.

**Models.** These commands are reconstructed from the project notes, as there is no export script. S3 must be symmetric: an asymmetric export generates garbage on the NPU.

```bash
.venv/bin/optimum-cli export openvino -m BAAI/bge-small-en-v1.5 --task feature-extraction --weight-format fp16 models/bge_small_en_v15_ov_fp16
.venv/bin/optimum-cli export openvino -m BAAI/bge-reranker-base --task text-classification --weight-format fp16 models/bge_reranker_base_ov_fp16
.venv/bin/optimum-cli export openvino -m Qwen/Qwen2.5-1.5B-Instruct --weight-format int4 --sym --ratio 1.0 --group-size -1 models/qwen25_1_5b_ov_int4_sym_cw
```

**Data.** Put the SQuAD v1.1 and HotpotQA (distractor) parquet files in `datasets/rag-squad/raw` and `datasets/rag-hotpotqa/raw`, then build the 25,880-passage corpus and its index:

```bash
.venv/bin/python scripts/build_corpus.py
.venv/bin/python scripts/build_index.py GPU
```

## Running

```bash
# One stage on one device
.venv/bin/python scripts/harness.py stage S2 --device NPU --cands 20

# End to end: sequential, overlapped, or under load
.venv/bin/python scripts/harness.py e2e --map S1=GPU,S2=GPU,S3=GPU
.venv/bin/python scripts/harness.py e2e --map S1=GPU,S2=NPU,S3=GPU --overlap
.venv/bin/python scripts/harness.py e2e --map S1=GPU,S2=GPU,S3=GPU --arrival poisson --rate 2.0

# S3 decode next to a reranker on another device
.venv/bin/python scripts/harness.py contend --device GPU --agg-device NPU --aggressor rerank --mode process

# Full experiments and analysis
bash scripts/run_phase1.sh
.venv/bin/python scripts/analyze_phase1.py runs/<run-folder>
```

Results are JSONL rows, written to `scratch/` unless `--out` is set. Run scripts create `runs/<date>_<host>_<name>/` with the code snapshot, environment and raw results; treat these folders as write-once.

## Known issue

An NPU `LLMPipeline` and any other NPU model in one process fail at first inference (`zeMemAllocHost`, `ZE_RESULT_ERROR_INVALID_NULL_HANDLE`). This worked on Oct 27 and the cause of the change is unknown. As a workaround, `harness.py e2e --s3-process auto` runs S3 in its own process when needed, which adds 0.4–0.6 ms per call.

## Authors

Ting-Yu Yu, Cheng-Yang Wu, Jonas Tsai, Vincent Chen
