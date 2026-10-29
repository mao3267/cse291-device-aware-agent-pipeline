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
# NPU compiles S3 for a fixed MAX_PROMPT_LEN; 0 = not applicable (CPU/GPU, older runs)
st["mpl"] = st["max_prompt_len"].fillna(0) if "max_prompt_len" in st else 0
st_all = st                                  # incl. true-cold (--cold) runs
if "cold" in st:
    st = st[st.cold != True]                 # latency tables use the default-cache runs

for stage, param in [("S1", "batch"), ("S2", "cands")]:
    d = st[st.stage == stage]
    warm = d[d.rep >= 0].groupby(["device", param]).latency_ms.agg(
        median="median", p90=lambda x: q(x, .9), n="count")
    cold = d[d.rep == -1].groupby(["device", param]).agg(
        cold_ms=("latency_ms", "first"), load_s=("load_s", "first"), exec=("exec", "first"))
    table(warm.join(cold), f"stage_{stage}", f"{stage} latency (ms), warm reps; cold = first run after compile")

d = st[(st.stage == "S3") & (st.get("ok", True) == True)]
if len(d):
    warm = d[d.rep >= 0].groupby(["device", "mpl", "prompt_len", "out_len"]).agg(
        ttft_ms=("ttft_ms", "median"), tpot_ms=("tpot_ms", "median"),
        tok_per_s=("tok_per_s", "median"), wall_s=("wall_s", "median"), n=("rep", "count"))
    table(warm, "stage_S3", "S3 LLM (medians over warm reps; mpl = NPU MAX_PROMPT_LEN, 0 = n/a)")
fails = st[st.get("ok", True) == False]
if len(fails):
    md.append("## Failed stage runs\n\n" + fails[["stage", "device", "prompt_len", "out_len", "err"]]
              .to_markdown() + "\n")

# support matrix
sm = st_all.groupby(["stage", "device"]).agg(exec=("exec", "first"),
                                         load_s_min=("load_s", "min"),
                                         load_s_max=("load_s", "max"))
table(sm, "support_matrix", "Support matrix (EXECUTION_DEVICES; S3 = LLMPipeline, not reported)")

# ------------------------------------------------------------ end to end
e = df[(df.exp == "e2e")].copy()
if len(e):
    # mode: sequential / overlap (old runs: from the overlap flag) / closed / poisson
    if "mode" not in e:
        e["mode"] = e.overlap.map({True: "overlap", False: "sequential"})
    e["mode"] = e["mode"].fillna(e.overlap.map({True: "overlap", False: "sequential"}))
    for c in ("clients", "rate"):
        if c not in e:
            e[c] = None
    e["mkey"] = [m if m in ("sequential", "overlap") else
                 (f"closed c={int(c)}" if m == "closed" else f"poisson r={r:g}")
                 for m, c, r in zip(e["mode"], e["clients"], e["rate"])]
qrows = e[e.summary.isna()] if "summary" in e else e
srows = e[e.summary == True] if "summary" in e else e.iloc[0:0]
classic = lambda d: d[d.mkey.isin(["sequential", "overlap"])] if len(d) else d

lat = classic(qrows).groupby(["map", "mkey"]).agg(
    e2e_med=("e2e_ms", "median"), e2e_p90=("e2e_ms", lambda x: q(x, .9)),
    s1_med=("s1_ms", "median"), s2_med=("s2_ms", "median"), s3_med=("s3_ms", "median"),
    ttft_med=("ttft_ms", "median"), f1=("f1", "mean"), em=("em", "mean"),
    recall20=("recall_cands", "mean"), recall3=("recall_top", "mean"), n=("qi", "count"))
thr = srows.groupby(["map", "mkey"]).agg(
    qps_mean=("qps", "mean"), qps_std=("qps", "std"), makespan_s=("makespan_s", "mean"),
    rounds=("qps", "count"), load1=("load1", "mean"))
if len(lat):
    table(lat.join(thr).sort_values("qps_mean", ascending=False), "e2e",
          "End-to-end (dev, 100 q per round). Overlap rows: per-query e2e includes queueing — compare qps")

# concurrency: closed-loop clients / open-loop Poisson arrivals (e2e_ms = response time)
cq = qrows[~qrows.mkey.isin(["sequential", "overlap"])] if len(qrows) else qrows
if len(cq):
    conc = cq.groupby(["map", "mkey"]).agg(
        resp_p50=("e2e_ms", "median"), resp_p90=("e2e_ms", lambda x: q(x, .9)),
        resp_p99=("e2e_ms", lambda x: q(x, .99)), wait_front_med=("wait_front_ms", "median"),
        wait_back_med=("wait_back_ms", "median"), s2_med=("s2_ms", "median"),
        s3_med=("s3_ms", "median"), f1=("f1", "mean"), n=("qi", "count"))
    table(conc.join(thr, how="left"), "e2e_concurrency",
          "Concurrency: response time from arrival (ms), per-stage queue waits, throughput")

# memory and device activity over whole e2e runs
if len(srows) and "peak_rss_gb" in srows:
    em_ = srows.groupby(["map", "mkey"])[[c for c in srows.columns if c.startswith(("peak_", "load_peak_"))]
                                         + ["active_gpu_cycles", "active_npu_ms"]].max()
    table(em_, "e2e_memory", "End-to-end memory peaks (GB / MB) and device activity per run")
if "bytes_prompt" in qrows:
    md.append("## Data handed between stages (median bytes per query)\n\n" +
              qrows[["bytes_qvec", "bytes_cand_text", "bytes_prompt"]].median().to_frame("bytes")
              .to_markdown() + "\n")

# answer consistency across mappings (placement must not change outputs)
if len(qrows):
    cons = qrows.groupby("qid").pred.nunique()
    md.append(f"## Output consistency\n\nQuestions with identical predictions across all runs: "
              f"{(cons == 1).sum()} / {len(cons)}\n")

# ------------------------------------------------------------ supplement sections
# (only when the run has the new columns, so older run folders still analyze)
if "active_gpu_cycles" in st:
    warm = st[st.rep >= 0].copy()
    lat_col = warm["latency_ms"].fillna(warm.get("wall_s", 0) * 1e3) if "wall_s" in warm else warm["latency_ms"]
    warm["npu_share"] = warm.active_npu_ms / lat_col
    pl = warm.groupby(["stage", "device"]).agg(
        exec=("exec", "first"), gpu_cycles_med=("active_gpu_cycles", "median"),
        npu_busy_ms_med=("active_npu_ms", "median"), npu_busy_share=("npu_share", "median"))
    pl["ran_on"] = ["GPU" if g > 0 else ("NPU" if (n or 0) > 0.05 else "CPU")
                    for g, n in zip(pl.gpu_cycles_med, pl.npu_busy_share)]
    table(pl, "placement", "Where the work actually ran (DRM fdinfo GPU cycles; NPU busy time / latency)")

    mem = st_all.copy()
    for k in ("rss_gb", "gpu_gtt_mb", "npu_mem_mb"):
        mem[f"model_{k}"] = mem[f"load_peak_{k}"] - mem[f"base_{k}"]
    key = {"S1": "batch", "S2": "cands"}
    mem["shape"] = [f"{key[s]}={int(r[key[s]])}" if s in key else f"mpl={int(r['mpl'])}"
                    for s, (_, r) in zip(mem.stage, mem.iterrows())]
    mt = mem.groupby(["stage", "device", "shape", "cold"]).agg(
        load_s=("load_s", "first"), model_rss_gb=("model_rss_gb", "max"),
        model_gpu_gtt_mb=("model_gpu_gtt_mb", "max"), model_npu_mem_mb=("model_npu_mem_mb", "max"),
        peak_rss_gb=("peak_rss_gb", "max"), peak_gpu_gtt_mb=("peak_gpu_gtt_mb", "max"),
        peak_npu_mem_mb=("peak_npu_mem_mb", "max"))
    table(mt, "memory_load", "Memory added by loading each model (load peak − baseline) and run peaks; "
          "load_s by compile-cache mode (cold = NPU cache bypassed)")

    enc = warm[warm.stage.isin(["S1", "S2"])].copy()
    if len(enc):
        enc["param"] = enc.get("batch", pd.Series(index=enc.index, dtype=float)).fillna(
            enc.get("cands", pd.Series(index=enc.index, dtype=float)))
        bd = enc.groupby(["stage", "device", "param"])[
            ["tokenize_ms", "set_input_ms", "infer_ms", "infer_zc_ms", "get_output_ms", "latency_ms",
             "in_bytes", "out_bytes"]].median()
        bd["input_transfer_est_ms"] = bd.infer_ms - bd.infer_zc_ms
        table(bd, "breakdown_S1_S2", "S1/S2 host-side breakdown, warm medians (ms). "
              "input_transfer_est = infer − infer with zero-copy inputs")
    s3 = warm[(warm.stage == "S3")]
    if len(s3) and "infer_ms" in s3:
        table(s3.groupby(["device", "prompt_len", "out_len"])[
            ["ttft_ms", "tpot_ms", "tokenize_ms", "infer_ms", "detokenize_ms", "wall_s"]].median(),
            "breakdown_S3", "S3 breakdown from GenAI perf metrics (warm medians, ms; wall in s)")
    if st.get("energy_j") is not None and st.energy_j.notna().any():
        en = warm.groupby(["stage", "device"]).agg(energy_j_med=("energy_j", "median"),
                                                   window_ms=("window_ms", "median"))
        table(en, "energy", "RAPL package energy per warm rep (J) — processor package, not whole machine")
    else:
        md.append("## Energy\n\nNot measured: RAPL energy_uj was not readable in this run.\n")

c = df[df.exp == "contend"] if "exp" in df else df.iloc[0:0]
if len(c):
    cr = c[c.summary.isna()] if "summary" in c else c
    cs = c[c.summary == True] if "summary" in c else c.iloc[0:0]
    ct = cr.groupby(["device", "agg_device", "aggressor", "mode"]).agg(
        ttft_med=("ttft_ms", "median"), tpot_med=("tpot_ms", "median"),
        npu_busy_ms=("active_npu_ms", "median"), n=("rep", "count"))
    if len(cs):
        ct = ct.join(cs.groupby(["device", "agg_device", "aggressor", "mode"])
                     .agg_iters_per_s.mean())
    table(ct, "contention", "Contention: S3 on device while an S2 aggressor runs on agg_device "
          "(thread vs process separates GIL from memory/bandwidth contention)")

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
d = d[d.mpl == d.groupby("device").mpl.transform("max")]  # NPU: the single-compile config
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

t = thr.reset_index()
t = t[t.mkey.isin(["sequential", "overlap"])] if len(t) else t
if len(t):
    maps = sorted(t["map"].unique(), key=lambda m: -t[t["map"] == m].qps_mean.max())
    fig, ax = plt.subplots(figsize=(10, 4))
    w = 0.4
    for j, ov in enumerate([False, True]):
        s = t[t.mkey == ("overlap" if ov else "sequential")].set_index("map").reindex(maps)
        ax.bar([i + (j - .5) * w for i in range(len(maps))], s.qps_mean, w,
               yerr=s.qps_std.fillna(0), label="overlap" if ov else "sequential", capsize=3)
    ax.set_xticks(range(len(maps)))
    ax.set_xticklabels([m.replace(",", "\n") for m in maps], fontsize=7)
    ax.set_ylabel("queries / s (mean ± std over rounds)"); ax.legend()
    fig.tight_layout(); fig.savefig(out / "fig_e2e_qps.png", dpi=150)

(out / "summary.md").write_text("\n".join(md))
print("wrote", out)
