"""Support-matrix smoke test: can each stage run on CPU / GPU / NPU?

Dummy inputs, single short prompt; only answers "does it run" and rough
magnitudes. Records EXECUTION_DEVICES to catch silent fallback.
Usage: smoke.py OUT_JSON
"""
import json, sys, time

import numpy as np
import openvino_genai as ov_genai

from rag_common import Encoder, model_path

DEVICES = ["CPU", "GPU", "NPU"]
results = []


def rec(stage, dev, **kw):
    results.append(dict(stage=stage, device=dev, **kw))
    print(results[-1], flush=True)


for dev in DEVICES:  # S3 LLM
    try:
        t = time.time()
        pipe = ov_genai.LLMPipeline(str(model_path("llm")), dev)
        load = time.time() - t
        cfg = ov_genai.GenerationConfig(max_new_tokens=64)
        prompt = "Explain what an NPU is in two sentences."
        pipe.generate(prompt, cfg)  # first run
        m = pipe.generate([prompt], cfg).perf_metrics
        rec("S3_llm", dev, ok=True, load_s=load,
            ttft_ms=m.get_ttft().mean, tok_per_s=m.get_throughput().mean)
        del pipe
    except Exception as e:
        rec("S3_llm", dev, ok=False, err=str(e)[:300])

for stage, key in [("S1_embed", "embed"), ("S2_rerank", "rerank")]:
    for dev in DEVICES:
        try:
            enc = Encoder(key, dev, batch=1, seq=128)
            feed = {n: np.ones((1, 128), dtype=np.int64) for n in enc.names}
            t = time.time(); enc.infer(feed); first = time.time() - t
            ts = []
            for _ in range(10):
                t = time.time(); enc.infer(feed); ts.append(time.time() - t)
            rec(stage, dev, ok=True, exec=enc.exec_devices, load_s=enc.load_s,
                first_ms=first * 1e3, warm_ms=float(np.mean(ts)) * 1e3)
        except Exception as e:
            rec(stage, dev, ok=False, err=str(e)[:300])

json.dump(results, open(sys.argv[1], "w"), indent=1)
