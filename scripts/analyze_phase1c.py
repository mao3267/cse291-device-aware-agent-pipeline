"""Summarize a Phase 1c run folder (optionally merged with earlier run folders).

Usage: analyze_phase1c.py RUN_DIR [EXTRA_RUN_DIR ...]
Writes RUN_DIR/analysis/ (a new subfolder): summary.md, coverage.csv, rag_n.csv,
load_poisson.csv, load_closed.csv, fig_load.png. Extra run folders (e.g. the Phase 1
run) are merged so coverage.csv spans all 27 mappings.
"""
import json, sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

run = Path(sys.argv[1])
out = run / "analysis"
out.mkdir(exist_ok=False)
rows = []
for d in [run] + [Path(x) for x in sys.argv[2:]]:
    rows += [dict(json.loads(l), src=d.name) for l in open(d / "results.jsonl")]
df = pd.DataFrame(rows)
df = df[df.exp == "e2e"].copy()
for c in ["mode", "rate", "clients", "wait_front_ms", "wait_back_ms"]:
    if c not in df:
        df[c] = None
df["mode"] = df["mode"].fillna(df.overlap.map({True: "overlap", False: "sequential"}))
q, s = df[df.summary.isna()], df[df.summary == True]
md = [f"# Phase 1c summary — `{run.name}`\n"]


def pct(p):
    return lambda x: x.quantile(p)


def table(d, name, title):
    d.to_csv(out / f"{name}.csv")
    md.append(f"## {title}\n\n{d.round(2).to_markdown()}\n")


# all sequential top-3 mappings, any run: the 27-mapping coverage table
sq, ss = q[(q["mode"] == "sequential") & (q.topn == 3)], s[(s["mode"] == "sequential") & (s.topn == 3)]
lat = sq.groupby("map").agg(e2e_med=("e2e_ms", "median"), e2e_p90=("e2e_ms", pct(.9)),
                            s1=("s1_ms", "median"), s2=("s2_ms", "median"), s3=("s3_ms", "median"),
                            ttft=("ttft_ms", "median"), recall3=("recall_top", "mean"),
                            f1=("f1", "mean"), n=("qi", "count"))
thr = ss.groupby("map").agg(qps=("qps", "mean"), rounds=("qps", "count"), load1=("load1", "mean"))
table(lat.join(thr).sort_values("qps", ascending=False), "coverage",
      "Sequential, top-3: all mappings measured (qps = mean over rounds)")

# RAG prompt size
sq, ss = q[q["mode"] == "sequential"], s[s["mode"] == "sequential"]
lat = sq.groupby(["map", "topn"]).agg(in_tok_med=("in_tokens", "median"), in_tok_max=("in_tokens", "max"),
                                      ttft=("ttft_ms", "median"), s3=("s3_ms", "median"),
                                      e2e_med=("e2e_ms", "median"), recall3=("recall_top", "mean"),
                                      f1=("f1", "mean"), em=("em", "mean"), n=("qi", "count"))
thr = ss.groupby(["map", "topn"]).agg(qps=("qps", "mean"))
d = lat.join(thr)
d = d[d.index.get_level_values("map").isin(
    d.reset_index().groupby("map").topn.nunique().loc[lambda x: x > 1].index)]
table(d, "rag_n", "RAG prompt size: top-N passages given to S3 (maps measured with more than one N)")

# load
pq, ps = q[q["mode"] == "poisson"], s[s["mode"] == "poisson"]
if len(pq):
    lat = pq.groupby(["map", "rate"]).agg(p50=("e2e_ms", "median"), p95=("e2e_ms", pct(.95)),
                                          p99=("e2e_ms", pct(.99)), wait_front=("wait_front_ms", "median"),
                                          wait_back=("wait_back_ms", "median"), n=("qi", "count"))
    thr = ps.groupby(["map", "rate"]).agg(qps=("qps", "mean"), qps_std=("qps", "std"),
                                          rounds=("qps", "count"), load1=("load1", "mean"))
    d = lat.join(thr)
    table(d, "load_poisson", "Open-loop Poisson arrivals: response time (ms, pooled over rounds) and achieved q/s")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for m, g in d.reset_index().groupby("map"):
        ax[0].plot(g.rate, g.p95, marker="o", label=m)
        ax[1].errorbar(g.rate, g.qps, yerr=g.qps_std.fillna(0), marker="o", label=m)
    ax[0].set_yscale("log"); ax[0].set_ylabel("p95 response time (ms)")
    ax[1].plot([0, 3], [0, 3], "k:", lw=1); ax[1].set_ylabel("achieved q/s")
    for a in ax:
        a.set_xlabel("offered q/s")
    ax[0].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(out / "fig_load.png", dpi=150)

cq, cs = q[q["mode"] == "closed"], s[s["mode"] == "closed"]
if len(cq):
    lat = cq.groupby(["map", "clients"]).agg(p50=("e2e_ms", "median"), p95=("e2e_ms", pct(.95)),
                                             n=("qi", "count"))
    thr = cs.groupby(["map", "clients"]).agg(qps=("qps", "mean"), rounds=("qps", "count"))
    table(lat.join(thr), "load_closed", "Closed-loop clients: response time (ms) and q/s")

(out / "summary.md").write_text("\n".join(md))
print("wrote", out)
