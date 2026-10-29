"""Summarize a Phase 1 run folder into tables and figures.

Usage: analyze_phase1.py RUN_DIR
Writes RUN_DIR/analysis/ (a new subfolder; existing run files are not touched):
  summary.md, stage_*.csv, e2e_*.csv, fig_*.png
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
rows = [json.loads(l) for l in open(run / "results.jsonl")]
df = pd.DataFrame(rows)
DEV_ORDER = ["CPU", "GPU", "NPU"]
md = [f"# Phase 1 summary — `{run.name}`\n"]


def q(x, p):
    return x.quantile(p)


def table(d, name, title):
    d.to_csv(out / f"{name}.csv")
    md.append(f"## {title}\n\n{d.round(2).to_markdown()}\n")


# ------------------------------------------------------------ stage sweeps
st = df[df.exp == "stage"].copy()
st["exec"] = st["exec"].astype(str)

for stage, param in [("S1", "batch"), ("S2", "cands")]:
    d = st[st.stage == stage]
    warm = d[d.rep >= 0].groupby(["device", param]).latency_ms.agg(
        median="median", p90=lambda x: q(x, .9), n="count")
    cold = d[d.rep == -1].groupby(["device", param]).agg(
        cold_ms=("latency_ms", "first"), load_s=("load_s", "first"), exec=("exec", "first"))
    table(warm.join(cold), f"stage_{stage}", f"{stage} latency (ms), warm reps; cold = first run after compile")

d = st[(st.stage == "S3") & (st.get("ok", True) == True)]
if len(d):
    warm = d[d.rep >= 0].groupby(["device", "prompt_len", "out_len"]).agg(
        ttft_ms=("ttft_ms", "median"), tpot_ms=("tpot_ms", "median"),
        tok_per_s=("tok_per_s", "median"), wall_s=("wall_s", "median"), n=("rep", "count"))
    table(warm, "stage_S3", "S3 LLM (medians over warm reps)")
fails = st[st.get("ok", True) == False]
if len(fails):
    md.append("## Failed stage runs\n\n" + fails[["stage", "device", "prompt_len", "out_len", "err"]]
              .to_markdown() + "\n")

# support matrix
sm = st.groupby(["stage", "device"]).agg(exec=("exec", "first"),
                                         load_s_min=("load_s", "min"),
                                         load_s_max=("load_s", "max"))
table(sm, "support_matrix", "Support matrix (EXECUTION_DEVICES; S3 = LLMPipeline, not reported)")

# ------------------------------------------------------------ end to end
e = df[(df.exp == "e2e")].copy()
qrows = e[e.summary.isna()] if "summary" in e else e
srows = e[e.summary == True] if "summary" in e else e.iloc[0:0]

lat = qrows.groupby(["map", "overlap"]).agg(
    e2e_med=("e2e_ms", "median"), e2e_p90=("e2e_ms", lambda x: q(x, .9)),
    s1_med=("s1_ms", "median"), s2_med=("s2_ms", "median"), s3_med=("s3_ms", "median"),
    ttft_med=("ttft_ms", "median"), f1=("f1", "mean"), em=("em", "mean"),
    recall20=("recall_cands", "mean"), recall3=("recall_top", "mean"), n=("qi", "count"))
thr = srows.groupby(["map", "overlap"]).agg(
    qps_mean=("qps", "mean"), qps_std=("qps", "std"), makespan_s=("makespan_s", "mean"),
    rounds=("qps", "count"), load1=("load1", "mean"))
table(lat.join(thr).sort_values("qps_mean", ascending=False), "e2e",
      "End-to-end (dev, 100 q per round). Overlap rows: per-query e2e includes queueing — compare qps")

# answer consistency across mappings (placement must not change outputs)
cons = qrows.groupby("qid").pred.nunique()
md.append(f"## Output consistency\n\nQuestions with identical predictions across all runs: "
          f"{(cons == 1).sum()} / {len(cons)}\n")

# ------------------------------------------------------------ figures
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
for i, (stage, param) in enumerate([("S1", "batch"), ("S2", "cands")]):
    d = st[(st.stage == stage) & (st.rep >= 0)].groupby(["device", param]).latency_ms.median()
    for dev in DEV_ORDER:
        if dev in d.index.get_level_values(0):
            s = d[dev]; ax[i].plot(s.index, s.values, marker="o", label=dev)
    ax[i].set_yscale("log"); ax[i].set_xlabel(param); ax[i].set_ylabel("warm latency (ms)")
    ax[i].set_title(f"{stage}"); ax[i].legend()
fig.tight_layout(); fig.savefig(out / "fig_stage_S1_S2.png", dpi=150)

d = st[(st.stage == "S3") & (st.rep >= 0) & (st.out_len == 128)]
if len(d):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    g = d.groupby(["device", "prompt_len"])[["ttft_ms", "tok_per_s"]].median()
    for dev in DEV_ORDER:
        if dev in g.index.get_level_values(0):
            s = g.loc[dev]
            ax[0].plot(s.index, s.ttft_ms, marker="o", label=dev)
            ax[1].plot(s.index, s.tok_per_s, marker="o", label=dev)
    ax[0].set_xscale("log", base=2); ax[1].set_xscale("log", base=2)
    ax[0].set_ylabel("TTFT (ms)"); ax[1].set_ylabel("decode tok/s")
    for a in ax: a.set_xlabel("prompt tokens"); a.legend()
    fig.suptitle("S3 (out_len 128)"); fig.tight_layout(); fig.savefig(out / "fig_stage_S3.png", dpi=150)

if len(srows):
    t = thr.reset_index()
    maps = sorted(t["map"].unique(), key=lambda m: -t[t["map"] == m].qps_mean.max())
    fig, ax = plt.subplots(figsize=(10, 4))
    w = 0.4
    for j, ov in enumerate([False, True]):
        s = t[t.overlap == ov].set_index("map").reindex(maps)
        ax.bar([i + (j - .5) * w for i in range(len(maps))], s.qps_mean, w,
               yerr=s.qps_std.fillna(0), label="overlap" if ov else "sequential", capsize=3)
    ax.set_xticks(range(len(maps)))
    ax.set_xticklabels([m.replace(",", "\n") for m in maps], fontsize=7)
    ax.set_ylabel("queries / s (mean ± std over rounds)"); ax.legend()
    fig.tight_layout(); fig.savefig(out / "fig_e2e_qps.png", dpi=150)

(out / "summary.md").write_text("\n".join(md))
print("wrote", out)
