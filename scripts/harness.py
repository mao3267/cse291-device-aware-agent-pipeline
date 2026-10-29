"""Phase 1 profiling harness.

  stage S1  --device GPU --batch 1,8,32 --seq 64
  stage S2  --device NPU --cands 10,50,100 --seq 256
  stage S3  --device CPU --prompt-len 256,1024,4096 --out-len 32,128,256
  e2e       --map S1=GPU,S2=NPU,S3=CPU --split dev --n 100 --cands 20 [--overlap]

Every row is appended to the JSONL file given by --out (default: scratch), with
the config, environment info (load average, available memory) and the
measurements. The first inference after compiling a shape is recorded as cold
(rep=-1); later reps are warm. Full matrices go through run_phase1.sh, which
creates a write-once run folder.
"""
import argparse, json, queue, threading, time
from pathlib import Path

import numpy as np

from rag_common import (CFG, Embedder, Reranker, Index, LLM, QUERY_PREFIX,
                        load_questions, model_path, qa_prompt, em_f1, env_info)

ints = lambda s: [int(x) for x in s.split(",")]


def writer(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "a")
    def w(row):
        f.write(json.dumps(row) + "\n"); f.flush()
        print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()},
              flush=True)
    return w


def timed(fn, *a, **kw):
    t = time.perf_counter(); out = fn(*a, **kw)
    return out, (time.perf_counter() - t) * 1e3


# ---------------------------------------------------------------- stage sweeps
def stage_S1(a, w):
    qs = [QUERY_PREFIX + q["question"] for q in load_questions("dev")]
    for b in ints(a.batch):
        emb = Embedder(a.device, batch=b, seq=a.seq)
        base = dict(exp="stage", stage="S1", device=a.device, exec=emb.exec_devices,
                    batch=b, seq=a.seq, load_s=emb.load_s, **env_info())
        for rep in range(-1, a.reps):
            s = ((rep + 1) * b) % len(qs)
            _, ms = timed(emb.embed, (qs[s:] + qs)[:b])
            w(dict(base, rep=rep, latency_ms=ms, per_item_ms=ms / b))
        del emb


def stage_S2(a, w):
    idx = Index(); qs = load_questions("dev")
    emb = Embedder("CPU")  # only to obtain realistic candidates; not measured
    for k in ints(a.cands):
        rr = Reranker(a.device, batch=k, seq=a.seq)
        base = dict(exp="stage", stage="S2", device=a.device, exec=rr.exec_devices,
                    cands=k, seq=a.seq, load_s=rr.load_s, **env_info())
        for rep in range(-1, a.reps):
            q = qs[(rep + 1) % len(qs)]["question"]
            cand = [idx.text[p] for p in idx.search(emb.embed([QUERY_PREFIX + q])[0], k)]
            _, ms = timed(rr.score, q, cand)
            w(dict(base, rep=rep, latency_ms=ms, per_item_ms=ms / k))
        del rr


def make_prompt(tok, texts, n_tokens, start=0):
    """~n_tokens of corpus text. A different `start` per rep gives a fresh prompt,
    so warm reps measure real prefill instead of a prefix-cache hit."""
    ids, i = [], start
    while len(ids) < n_tokens:
        ids += tok.encode(texts[i] + "\n\n"); i += 1
    return tok.decode(ids[:max(n_tokens - 24, 8)]) + "\n\nSummarize the text above."


def stage_S3(a, w):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path("llm"))
    texts = list(Index().text.values())
    plens, olens = ints(a.prompt_len), ints(a.out_len)
    llm = LLM(a.device, max_prompt_len=max(plens) + 128, min_response_len=max(olens))
    base0 = dict(exp="stage", stage="S3", device=a.device, load_s=llm.load_s)
    llm.generate(make_prompt(tok, texts, 64, start=10**4), max_new_tokens=4)  # first-call warmup
    seed = 0
    for p in plens:
        for o in olens:
            base = dict(base0, prompt_len=p, out_len=o, **env_info())
            for rep in range(-1, a.reps):
                seed += 1
                prompt = make_prompt(tok, texts, p, start=seed * 40)
                try:
                    r = llm.generate(prompt, max_new_tokens=o, ignore_eos=True)
                    r.pop("text")
                    w(dict(base, rep=rep, ok=True, **r))
                except Exception as e:
                    w(dict(base, rep=rep, ok=False, err=str(e)[:300])); break


# ---------------------------------------------------------------- end to end
def e2e(a, w):
    m = dict(kv.split("=") for kv in a.map.split(","))
    idx = Index()
    emb = Embedder(m["S1"], batch=1, seq=64)
    rr = Reranker(m["S2"], batch=a.cands, seq=256)
    llm = LLM(m["S3"], max_prompt_len=1024, min_response_len=64)
    qs = load_questions(a.split)[:a.n]
    base = dict(exp="e2e", map=a.map, split=a.split, cands=a.cands, topn=a.topn,
                overlap=a.overlap, round=a.round,
                exec={"S1": emb.exec_devices, "S2": rr.exec_devices},
                load_s={"S1": emb.load_s, "S2": rr.load_s, "S3": llm.load_s},
                **env_info())

    def front(q):
        t0 = time.perf_counter()
        v, s1 = timed(emb.embed, [QUERY_PREFIX + q["question"]])
        cand, srch = timed(idx.search, v[0], a.cands)
        sc, s2 = timed(rr.score, q["question"], [idx.text[p] for p in cand])
        top = [cand[i] for i in np.argsort(-sc)[:a.topn]]
        gold = set(q["gold_pids"])
        times = dict(s1_ms=s1, search_ms=srch, s2_ms=s2,
                     recall_cands=len(gold & set(cand)) / len(gold),
                     recall_top=len(gold & set(top)) / len(gold))
        return t0, times, qa_prompt(q["question"], [idx.text[p] for p in top])

    def back(i, q, t0, times, prompt):
        r = llm.generate(prompt, max_new_tokens=32)
        em, f1 = em_f1(r["text"].strip(), q["answers"])
        w(dict(base, qi=i, qid=q["qid"], **times, s3_ms=r["wall_s"] * 1e3,
               ttft_ms=r["ttft_ms"], tpot_ms=r["tpot_ms"], in_tokens=r["in_tokens"],
               out_tokens=r["out_tokens"], e2e_ms=(time.perf_counter() - t0) * 1e3,
               em=em, f1=f1, pred=r["text"].strip()[:100]))

    front(qs[0]); llm.generate("warmup", max_new_tokens=4)  # warm all devices
    t_start = time.perf_counter()
    if not a.overlap:
        for i, q in enumerate(qs):
            back(i, q, *front(q))
    else:
        # query i+1's S1/S2 overlap with query i's S3 (at most 2 in flight);
        # per-query e2e_ms then includes queueing, so compare makespan / qps.
        qq = queue.Queue(maxsize=1)
        def producer():
            for i, q in enumerate(qs):
                qq.put((i, q, *front(q)))
            qq.put(None)
        th = threading.Thread(target=producer); th.start()
        while (item := qq.get()) is not None:
            back(*item)
        th.join()
    total = time.perf_counter() - t_start
    w(dict(base, summary=True, n=len(qs), makespan_s=total, qps=len(qs) / total,
           **{f"end_{k}": v for k, v in env_info().items()}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["stage", "e2e"])
    ap.add_argument("stage", nargs="?")
    ap.add_argument("--device", default="CPU")
    ap.add_argument("--batch", default="1,8,32")
    ap.add_argument("--seq", type=int, default=None)
    ap.add_argument("--cands", default="10,50,100")
    ap.add_argument("--prompt-len", default="256,1024,4096")
    ap.add_argument("--out-len", default="32,128,256")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--map", default="S1=CPU,S2=CPU,S3=CPU")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--topn", type=int, default=3)
    ap.add_argument("--overlap", action="store_true")
    ap.add_argument("--round", type=int, default=0, help="repeat index, recorded in e2e rows")
    ap.add_argument("--out", default=f"{CFG['paths']['scratch']}/p01-harness-test/results.jsonl")
    a = ap.parse_args()
    w = writer(a.out)
    if a.cmd == "stage":
        a.seq = a.seq or {"S1": 64, "S2": 256}.get(a.stage)
        {"S1": stage_S1, "S2": stage_S2, "S3": stage_S3}[a.stage](a, w)
    else:
        a.cands = int(a.cands.split(",")[0])
        e2e(a, w)
