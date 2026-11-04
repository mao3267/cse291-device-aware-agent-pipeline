"""Phase 1 profiling harness.

  stage S1  --device GPU --batch 1,8,32 --seq 64 [--cold]
  stage S2  --device NPU --cands 10,50,100 --seq 256 [--cold]
  stage S3  --device CPU --prompt-len 256,1024,4096 --out-len 32,128,256 [--cold]
  e2e       --map S1=GPU,S2=NPU,S3=CPU --split dev --n 100 --cands 20 [--overlap]
  e2e       --map ... --arrival closed --clients 2      (N clients, each waits for its answer)
  e2e       --map ... --arrival poisson --rate 1.5     (open loop, queries/s)
  contend   --device GPU --agg-device NPU --aggressor rerank|tokenize|none --mode thread|process

Every row is appended to the JSONL file given by --out (default: scratch), with
the config, environment info (load average, available memory) and the
measurements. The first inference after compiling a shape is recorded as cold
(rep=-1); later reps are warm. --cold additionally bypasses the NPU driver's
compile cache, so load_s is a true compile (GPU: the runner sets
NEO_CACHE_PERSISTENT=0). Full matrices go through run_phase1*.sh, which create a
write-once run folder.

Stage rows also carry: a latency breakdown (tokenize / set_input / infer /
get_output, and infer_zc_ms with inputs in device-accessible memory), memory
peaks (process RSS, system used, GPU GTT, NPU memory; load_peak_* during
compile, peak_* during the rep), and device activity during the rep
(active_gpu_cycles from this process's DRM fdinfo, active_npu_ms from the
device-wide NPU busy counter), which shows where the work actually ran.
"""
import argparse, json, os, queue, subprocess, sys, threading, time
from pathlib import Path

import numpy as np

from rag_common import (CFG, Embedder, Reranker, Index, LLM, RemoteLLM, QUERY_PREFIX, ResourceMonitor,
                        counters, delta, load_passages, load_questions, model_path,
                        passage_text, qa_prompt, em_f1, env_info)

ints = lambda s: [int(x) for x in s.split(",")]


def writer(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "a")
    lock = threading.Lock()
    def w(row):
        with lock:
            f.write(json.dumps(row) + "\n"); f.flush()
            print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()},
                  flush=True)
    return w


def timed(fn, *a, **kw):
    t = time.perf_counter(); out = fn(*a, **kw)
    return out, (time.perf_counter() - t) * 1e3


def cache_info(a):
    return dict(cold=a.cold, npu_cache="bypass" if a.cold else "default",
                neo_cache_persistent=os.environ.get("NEO_CACHE_PERSISTENT"))


# ---------------------------------------------------------------- stage sweeps
def encoder_sweep(a, w, stage, param, values, make, inputs):
    """Shared loop for S1/S2: compile one static shape per value, then cold + warm reps."""
    mon, enc = ResourceMonitor(), None
    for v in values:
        enc = None                                   # free the previous shape first
        base_mem = mon.now("base_"); mon.take()
        enc = make(v)
        load_peak = mon.take("load_peak_")
        base = dict(exp="stage", stage=stage, device=a.device, exec=enc.exec_devices,
                    **{param: v}, seq=a.seq, load_s=enc.load_s, **cache_info(a),
                    **base_mem, **load_peak, **env_info())
        for rep in range(-1, a.reps):
            texts, pairs = inputs(v, rep)
            c0 = counters(); r = enc.infer_breakdown(texts, pairs); c1 = counters()
            peaks = mon.take()
            zc = enc.infer_zero_copy_ms(texts, pairs) if rep >= 0 else None
            w(dict(base, rep=rep, per_item_ms=r["latency_ms"] / v, infer_zc_ms=zc,
                   **r, **delta(c0, c1), **peaks))
    mon.stop()


def stage_S1(a, w):
    qs = [QUERY_PREFIX + q["question"] for q in load_questions("dev")]
    def inputs(b, rep):
        s = ((rep + 1) * b) % len(qs)
        return (qs[s:] + qs)[:b], None
    encoder_sweep(a, w, "S1", "batch", ints(a.batch),
                  lambda b: Embedder(a.device, batch=b, seq=a.seq, cold=a.cold), inputs)


def stage_S2(a, w):
    idx = Index(); qs = load_questions("dev")
    emb = Embedder("CPU")  # only to obtain realistic candidates; not measured
    def inputs(k, rep):
        q = qs[(rep + 1) % len(qs)]["question"]
        cand = [idx.text[p] for p in idx.search(emb.embed([QUERY_PREFIX + q])[0], k)]
        return [q] * k, cand
    encoder_sweep(a, w, "S2", "cands", ints(a.cands),
                  lambda k: Reranker(a.device, batch=k, seq=a.seq, cold=a.cold), inputs)


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
    mon = ResourceMonitor()
    base_mem = mon.now("base_"); mon.take()
    # NPU compiles the prefill for a fixed MAX_PROMPT_LEN; record it, since NPU TTFT
    # depends on it rather than on the actual prompt length
    max_prompt = max(plens) + 128
    llm = LLM(a.device, max_prompt_len=max_prompt, min_response_len=max(olens), cold=a.cold)
    base0 = dict(exp="stage", stage="S3", device=a.device, load_s=llm.load_s, **cache_info(a),
                 max_prompt_len=max_prompt if a.device == "NPU" else None,
                 **base_mem, **mon.take("load_peak_"))
    llm.generate(make_prompt(tok, texts, 64, start=10**4), max_new_tokens=4)  # first-call warmup
    seed = 0
    for p in plens:
        for o in olens:
            base = dict(base0, prompt_len=p, out_len=o, **env_info())
            for rep in range(-1, a.reps):
                seed += 1
                prompt = make_prompt(tok, texts, p, start=seed * 40)
                mon.take()
                try:
                    c0 = counters()
                    r = llm.generate(prompt, max_new_tokens=o, ignore_eos=True)
                    c1 = counters()
                    r.pop("text")
                    w(dict(base, rep=rep, ok=True, **r, **delta(c0, c1), **mon.take()))
                except Exception as e:
                    w(dict(base, rep=rep, ok=False, err=str(e)[:300])); break
    mon.stop()


# ---------------------------------------------------------------- end to end
def e2e(a, w):
    m = dict(kv.split("=") for kv in a.map.split(","))
    mon = ResourceMonitor(interval_s=0.05)
    idx = Index()
    emb = Embedder(m["S1"], batch=1, seq=64)
    rr = Reranker(m["S2"], batch=a.cands, seq=256)
    npu_other = m["S3"] == "NPU" and "NPU" in (m["S1"], m["S2"])
    remote = a.s3_process == "always" or (a.s3_process == "auto" and npu_other)
    llm = (RemoteLLM if remote else LLM)(m["S3"], max_prompt_len=a.max_prompt_len, min_response_len=64)
    qs = load_questions(a.split)[:a.n]
    mode = a.arrival or ("overlap" if a.overlap else "sequential")
    base = dict(exp="e2e", map=a.map, split=a.split, cands=a.cands, topn=a.topn,
                overlap=a.overlap, round=a.round, mode=mode, s3_process=remote, clients=a.clients, rate=a.rate,
                exec={"S1": emb.exec_devices, "S2": rr.exec_devices},
                load_s={"S1": emb.load_s, "S2": rr.load_s, "S3": llm.load_s},
                **mon.take("load_peak_"), **env_info())

    def front(q):
        t0 = time.perf_counter()
        v, s1 = timed(emb.embed, [QUERY_PREFIX + q["question"]])
        cand, srch = timed(idx.search, v[0], a.cands)
        ctext = [idx.text[p] for p in cand]
        sc, s2 = timed(rr.score, q["question"], ctext)
        top = [cand[i] for i in np.argsort(-sc)[:a.topn]]
        gold = set(q["gold_pids"])
        prompt = qa_prompt(q["question"], [idx.text[p] for p in top])
        # data handed between stages (all in shared DRAM on this machine)
        times = dict(s1_ms=s1, search_ms=srch, s2_ms=s2,
                     recall_cands=len(gold & set(cand)) / len(gold),
                     recall_top=len(gold & set(top)) / len(gold),
                     bytes_qvec=v.nbytes, bytes_cand_text=sum(len(t.encode()) for t in ctext),
                     bytes_prompt=len(prompt.encode()))
        return t0, times, prompt

    def back(i, q, t0, times, prompt):
        r = llm.generate(prompt, max_new_tokens=32)
        em, f1 = em_f1(r["text"].strip(), q["answers"])
        w(dict(base, qi=i, qid=q["qid"], **times, s3_ms=r["wall_s"] * 1e3,
               ttft_ms=r["ttft_ms"], tpot_ms=r["tpot_ms"], ipc_ms=r.get("ipc_ms"), in_tokens=r["in_tokens"],
               out_tokens=r["out_tokens"], e2e_ms=(time.perf_counter() - t0) * 1e3,
               em=em, f1=f1, pred=r["text"].strip()[:100]))

    front(qs[0]); llm.generate("warmup", max_new_tokens=4)  # warm all devices
    mon.take()
    c0 = counters()
    t_start = time.perf_counter()
    if a.arrival:
        t_end = run_arrival(a, qs, front, back)
    elif not a.overlap:
        for i, q in enumerate(qs):
            back(i, q, *front(q))
        t_end = time.perf_counter()
    else:
        # query i+1's S1/S2 overlap with query i's S3 (at most 2 stages computing at
        # once); per-query e2e_ms then includes queueing, so compare makespan / qps.
        qq = queue.Queue(maxsize=1)
        def producer():
            for i, q in enumerate(qs):
                qq.put((i, q, *front(q)))
            qq.put(None)
        th = threading.Thread(target=producer); th.start()
        while (item := qq.get()) is not None:
            back(*item)
        th.join()
        t_end = time.perf_counter()
    total = t_end - t_start
    act = delta(c0, counters())
    act.pop("window_ms")
    w(dict(base, summary=True, n=len(qs), makespan_s=total, qps=len(qs) / total,
           **act, **mon.take(), **{f"end_{k}": v for k, v in env_info().items()}))
    mon.stop()
    if remote:
        llm.close()


def run_arrival(a, qs, front, back):
    """Two-stage pipeline fed by an arrival process. One worker per stage (front =
    S1 -> search -> S2, back = S3), each model loaded once. e2e_ms = response time
    from arrival to answer; queue waits are recorded per stage."""
    in_q, mid_q, done = queue.Queue(), queue.Queue(), {}

    def front_worker():
        while (it := in_q.get()) is not None:
            i, q, t_arr = it
            t_fs = time.perf_counter()
            _, times, prompt = front(q)
            mid_q.put((i, q, t_arr, t_fs, time.perf_counter(), times, prompt))
        mid_q.put(None)

    def back_worker():
        while (it := mid_q.get()) is not None:
            i, q, t_arr, t_fs, t_fe, times, prompt = it
            times = dict(times, wait_front_ms=(t_fs - t_arr) * 1e3, front_ms=(t_fe - t_fs) * 1e3,
                         wait_back_ms=(time.perf_counter() - t_fe) * 1e3)
            back(i, q, t_arr, times, prompt)
            done[i].set()

    workers = [threading.Thread(target=front_worker), threading.Thread(target=back_worker)]
    for t in workers: t.start()
    for i in range(len(qs)): done[i] = threading.Event()

    if a.arrival == "closed":
        it, lock = iter(enumerate(qs)), threading.Lock()
        def client():
            while True:
                with lock:
                    nxt = next(it, None)
                if nxt is None:
                    return
                i, q = nxt
                in_q.put((i, q, time.perf_counter())); done[i].wait()
        cl = [threading.Thread(target=client) for _ in range(a.clients)]
        for t in cl: t.start()
        for t in cl: t.join()
    else:  # poisson, open loop: arrival time is the scheduled time
        rng = np.random.default_rng(a.seed)
        t_next = time.perf_counter()
        for i, q in enumerate(qs):
            t_next += rng.exponential(1 / a.rate)
            time.sleep(max(0.0, t_next - time.perf_counter()))
            in_q.put((i, q, t_next))
        for ev in done.values(): ev.wait()
    t_end = time.perf_counter()
    in_q.put(None)
    for t in workers: t.join()
    return t_end


# ---------------------------------------------------------------- contention
AGG_Q = "Which city hosted the 1992 Summer Olympics?"


def aggressor_inputs():
    ps = load_passages()[:20]
    return [AGG_Q] * 20, [passage_text(p) for p in ps]


def aggress_loop(rr, kind, stop):
    """Repeat S2 work until stop is set. kind=tokenize does only the Python/CPU part
    (holds the GIL, no accelerator work); kind=rerank does tokenize + inference."""
    texts, pairs = aggressor_inputs()
    n, t = 0, time.perf_counter()
    while not stop.is_set():
        if kind == "rerank":
            rr.infer_breakdown(texts, pairs)
        else:
            rr.tokenize(texts, pairs)
        n += 1
    return n, time.perf_counter() - t


def aggress(a, w):
    """Child process for `contend --mode process`: prints 'ready', loops until stdin closes."""
    rr = Reranker(a.agg_device, batch=20, seq=256)
    stop = threading.Event()
    threading.Thread(target=lambda: (sys.stdin.read(), stop.set()), daemon=True).start()
    print("ready", flush=True)
    n, s = aggress_loop(rr, a.aggressor, stop)
    print(json.dumps(dict(iters=n, secs=s)), flush=True)


def contend(a, w):
    """S3 decode on --device measured while an S2 aggressor runs on --agg-device,
    either in a thread of this process or in a separate process. Comparing the two
    separates Python-GIL interference from shared-memory/bandwidth contention."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path("llm"))
    texts = list(Index().text.values())
    p, o = ints(a.prompt_len)[0], ints(a.out_len)[0]
    llm = LLM(a.device, max_prompt_len=p + 128, min_response_len=o)
    llm.generate(make_prompt(tok, texts, 64, start=10**4), max_new_tokens=4)
    base = dict(exp="contend", device=a.device, prompt_len=p, out_len=o, aggressor=a.aggressor,
                agg_device=a.agg_device, mode=a.mode, round=a.round, **env_info())

    stop, th, proc, res = threading.Event(), None, None, {}
    if a.aggressor != "none":
        if a.mode == "thread":
            rr = Reranker(a.agg_device, batch=20, seq=256)
            th = threading.Thread(target=lambda: res.update(zip(("iters", "secs"),
                                                                 aggress_loop(rr, a.aggressor, stop))))
            th.start()
        else:
            proc = subprocess.Popen([sys.executable, __file__, "aggress", "--agg-device", a.agg_device,
                                     "--aggressor", a.aggressor], stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE, text=True)
            while proc.stdout.readline().strip() != "ready":
                pass
        time.sleep(1.0)  # let the aggressor reach steady state

    for rep in range(a.reps):
        prompt = make_prompt(tok, texts, p, start=(rep + 1) * 40 + a.round * 4000)
        c0 = counters()
        r = llm.generate(prompt, max_new_tokens=o, ignore_eos=True)
        r.pop("text")
        w(dict(base, rep=rep, **r, **delta(c0, counters())))

    if th:
        stop.set(); th.join()
    if proc:
        proc.stdin.close()
        res = json.loads(proc.stdout.read().strip().splitlines()[-1]); proc.wait()
    if res:
        w(dict(base, summary=True, agg_iters=res["iters"], agg_secs=res["secs"],
               agg_iters_per_s=res["iters"] / res["secs"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["stage", "e2e", "contend", "aggress"])
    ap.add_argument("stage", nargs="?")
    ap.add_argument("--device", default="CPU")
    ap.add_argument("--batch", default="1,8,32")
    ap.add_argument("--seq", type=int, default=None)
    ap.add_argument("--cands", default="10,50,100")
    ap.add_argument("--prompt-len", default="256,1024,4096")
    ap.add_argument("--out-len", default="32,128,256")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--cold", action="store_true", help="bypass the NPU compile cache")
    ap.add_argument("--map", default="S1=CPU,S2=CPU,S3=CPU")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--topn", type=int, default=3)
    ap.add_argument("--s3-process", choices=["auto", "always", "never"], default="auto",
                    help="e2e: run S3 in its own process (auto: when S3=NPU and S1 or S2 is also NPU)")
    ap.add_argument("--max-prompt-len", type=int, default=1024, help="e2e: NPU MAX_PROMPT_LEN for S3 (raise for large --topn)")
    ap.add_argument("--overlap", action="store_true")
    ap.add_argument("--arrival", choices=["closed", "poisson"], default=None)
    ap.add_argument("--clients", type=int, default=1, help="closed-loop clients")
    ap.add_argument("--rate", type=float, default=None, help="poisson arrivals, queries/s")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--agg-device", default="NPU")
    ap.add_argument("--aggressor", choices=["none", "rerank", "tokenize"], default="rerank")
    ap.add_argument("--mode", choices=["thread", "process"], default="thread")
    ap.add_argument("--round", type=int, default=0, help="repeat index, recorded in rows")
    ap.add_argument("--out", default=f"{CFG['paths']['scratch']}/p01-harness-test/results.jsonl")
    a = ap.parse_args()
    w = writer(a.out) if a.cmd != "aggress" else None
    if a.cmd == "stage":
        a.seq = a.seq or {"S1": 64, "S2": 256}.get(a.stage)
        {"S1": stage_S1, "S2": stage_S2, "S3": stage_S3}[a.stage](a, w)
    elif a.cmd == "e2e":
        a.cands = int(a.cands.split(",")[0])
        if a.arrival == "poisson" and not a.rate:
            ap.error("--arrival poisson needs --rate")
        e2e(a, w)
    elif a.cmd == "contend":
        contend(a, w)
    else:
        aggress(a, w)
