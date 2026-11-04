"""Shared pieces of the RAG pipeline: config, encoders, retrieval, LLM, scoring.

Each model is loaded once on one device. Encoders are compiled with static
shapes, and the same shapes are used on CPU, GPU, and NPU (NPU requires static
shapes), so device comparisons are controlled. All paths come from config.yaml.
"""
import atexit, collections, json, multiprocessing as mp, os, re, socket, string, time
from pathlib import Path

import numpy as np
import openvino as ov
import openvino_genai as ov_genai
import yaml
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load(open(ROOT / "config.yaml"))
MODELS = Path(CFG["paths"]["models"])
DATASETS = Path(CFG["paths"]["datasets"])
CORPUS = DATASETS / CFG["datasets"]["corpus"]
INDEX = DATASETS / CFG["datasets"]["index"]
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

core = ov.Core()


def model_path(key):
    return MODELS / CFG["models"][key]


def passage_text(p):
    return f"{p['title']}: {p['text']}"


def load_passages():
    return [json.loads(l) for l in open(CORPUS / "passages.jsonl")]


def load_questions(split):
    name = {"dev": "questions_squad_dev.jsonl", "eval": "questions_hotpot.jsonl"}[split]
    return [json.loads(l) for l in open(CORPUS / name)]


def compile_props(device, cold=False):
    """cold=True bypasses the NPU driver's compiled-blob cache (~/.cache/ze_intel_npu_cache),
    so load_s is a true NPU compile. GPU kernel caching is controlled by the env var
    NEO_CACHE_PERSISTENT=0, set by the runner; CPU has no persistent cache."""
    return {"NPU_BYPASS_UMD_CACHING": "YES"} if (cold and device == "NPU") else {}


class Encoder:
    """Static-shape OpenVINO encoder (bge embedding or reranker)."""

    def __init__(self, key, device, batch, seq, cold=False):
        self.path = model_path(key)
        self.device, self.batch, self.seq = device, batch, seq
        self.tok = AutoTokenizer.from_pretrained(self.path)
        t = time.perf_counter()
        model = core.read_model(self.path / "openvino_model.xml")
        model.reshape({i.get_any_name(): [batch, seq] for i in model.inputs})
        self.names = [i.get_any_name() for i in model.inputs]
        self.compiled = core.compile_model(model, device, compile_props(device, cold))
        self.req = self.compiled.create_infer_request()
        self.load_s = time.perf_counter() - t
        self.exec_devices = self.compiled.get_property("EXECUTION_DEVICES")
        self._zc = None

    def infer(self, enc):
        self.req.infer({n: enc[n].astype(np.int64) for n in self.names})
        return self.req.get_output_tensor(0).data.copy()

    def _zero_copy_request(self):
        """A second request whose inputs live in device-accessible host memory
        (RemoteContext.create_host_tensor). None on CPU or if unsupported."""
        if self._zc is None:
            self._zc = False
            if self.device != "CPU":
                try:
                    ctx = core.get_default_context(self.device)
                    req = self.compiled.create_infer_request()
                    shape = ov.Shape([self.batch, self.seq])
                    for n in self.names:
                        req.set_tensor(n, ctx.create_host_tensor(ov.Type.i64, shape))
                        req.get_tensor(n).data[:] = 0
                    req.infer()  # warm the second request before it is timed
                    self._zc = req
                except Exception as e:
                    print(f"zero-copy request unavailable on {self.device}: {e}", flush=True)
        return self._zc or None

    def infer_breakdown(self, texts, pairs=None):
        """Same work as embed/score, split into host-side phases (ms):
        tokenize, set_input (numpy -> ov.Tensor copy + set_tensor), infer (with inputs
        already set; on GPU/NPU this still includes the plugin's input transfer),
        get_output (copy back)."""
        t0 = time.perf_counter()
        enc, n = self.tokenize(texts, pairs)
        t1 = time.perf_counter()
        for name in self.names:
            self.req.set_tensor(name, ov.Tensor(enc[name].astype(np.int64), shared_memory=False))
        t2 = time.perf_counter()
        self.req.infer()
        t3 = time.perf_counter()
        out = self.req.get_output_tensor(0).data.copy()
        t4 = time.perf_counter()
        return dict(tokenize_ms=(t1 - t0) * 1e3, set_input_ms=(t2 - t1) * 1e3,
                    infer_ms=(t3 - t2) * 1e3, get_output_ms=(t4 - t3) * 1e3,
                    latency_ms=(t4 - t0) * 1e3, n_real=n,
                    in_bytes=sum(enc[k].size * 8 for k in self.names), out_bytes=out.nbytes)

    def infer_zero_copy_ms(self, texts, pairs=None):
        """Inference time with inputs already in device-accessible host memory
        (RemoteContext host tensors). infer_ms - this estimates the input transfer
        done inside the GPU/NPU plugin. None on CPU or when unsupported."""
        zc = self._zero_copy_request()
        if zc is None:
            return None
        enc, _ = self.tokenize(texts, pairs)
        for name in self.names:
            zc.get_tensor(name).data[:] = enc[name]
        t = time.perf_counter(); zc.infer()
        return (time.perf_counter() - t) * 1e3

    def tokenize(self, texts, pairs=None):
        n = len(texts)
        assert n <= self.batch
        texts = list(texts) + [""] * (self.batch - n)
        kw = dict(padding="max_length", truncation=True, max_length=self.seq,
                  return_tensors="np")
        if pairs is not None:
            pairs = list(pairs) + [""] * (self.batch - n)
            kw["truncation"] = "only_second"
            return self.tok(texts, pairs, **kw), n
        return self.tok(texts, **kw), n


class Embedder(Encoder):
    def __init__(self, device, batch=1, seq=64, cold=False):
        super().__init__("embed", device, batch, seq, cold)

    def embed(self, texts):
        enc, n = self.tokenize(texts)
        out = self.infer(enc)[:n, 0, :]          # CLS pooling
        return out / np.linalg.norm(out, axis=1, keepdims=True)


class Reranker(Encoder):
    def __init__(self, device, batch=20, seq=256, cold=False):
        super().__init__("rerank", device, batch, seq, cold)

    def score(self, query, passages):
        enc, n = self.tokenize([query] * len(passages), passages)
        return self.infer(enc)[:n, 0]


class Index:
    """Brute-force cosine search over precomputed passage embeddings (CPU)."""

    def __init__(self):
        self.emb = np.load(INDEX / "emb.npy")
        self.pids = np.load(INDEX / "pids.npy")
        ps = {p["pid"]: p for p in load_passages()}
        self.text = {pid: passage_text(ps[pid]) for pid in self.pids.tolist()}

    def search(self, qvec, k):
        scores = self.emb @ qvec
        idx = np.argpartition(-scores, k)[:k]
        return self.pids[idx[np.argsort(-scores[idx])]].tolist()


class LLM:
    def __init__(self, device, max_prompt_len=1024, min_response_len=256, cold=False):
        cfg = compile_props(device, cold)
        if device == "NPU":
            cfg.update(MAX_PROMPT_LEN=max_prompt_len, MIN_RESPONSE_LEN=min_response_len)
        t = time.perf_counter()
        self.pipe = ov_genai.LLMPipeline(str(model_path("llm")), device, **cfg)
        self.load_s = time.perf_counter() - t
        self.device = device

    def generate(self, prompt, max_new_tokens=32, ignore_eos=False):
        cfg = ov_genai.GenerationConfig(max_new_tokens=max_new_tokens)
        cfg.ignore_eos = ignore_eos
        cfg.do_sample = False
        t = time.perf_counter()
        res = self.pipe.generate([prompt], cfg)
        wall = time.perf_counter() - t
        m = res.perf_metrics
        return dict(text=res.texts[0], wall_s=wall,
                    ttft_ms=m.get_ttft().mean, tpot_ms=m.get_tpot().mean,
                    tok_per_s=m.get_throughput().mean,
                    in_tokens=m.get_num_input_tokens(),
                    out_tokens=m.get_num_generated_tokens(),
                    tokenize_ms=m.get_tokenization_duration().mean,
                    infer_ms=m.get_inference_duration().mean,
                    detokenize_ms=m.get_detokenization_duration().mean)


def _llm_worker(device, kwargs, q_in, q_out):
    llm = LLM(device, **kwargs)
    q_out.put(("ready", llm.load_s))
    while (job := q_in.get()) is not None:
        try:
            q_out.put(("ok", llm.generate(*job[0], **job[1])))
        except Exception as e:
            q_out.put(("err", repr(e)))


class RemoteLLM:
    """LLM in its own process, same interface as LLM. Needed when S3 is on the NPU
    together with another NPU model: in one process, the second NPU model compiled
    invalidates the first one (L0 zeMemAllocHost INVALID_NULL_HANDLE, 2026-11-01).
    Adds one queue round trip per call, recorded as ipc_ms."""

    def __init__(self, device, **kwargs):
        ctx = mp.get_context("spawn")
        self.q_in, self.q_out, self.device = ctx.Queue(), ctx.Queue(), device
        self.proc = ctx.Process(target=_llm_worker, args=(device, kwargs, self.q_in, self.q_out),
                                daemon=True)
        self.proc.start()
        self.load_s = self._get()
        atexit.register(self.close)

    def _get(self):
        while True:
            try:
                tag, val = self.q_out.get(timeout=5)
            except Exception:
                if not self.proc.is_alive():
                    raise RuntimeError("LLM worker process died")
                continue
            if tag == "err":
                raise RuntimeError(val)
            return val

    def generate(self, prompt, max_new_tokens=32, ignore_eos=False):
        t = time.perf_counter()
        self.q_in.put(((prompt,), dict(max_new_tokens=max_new_tokens, ignore_eos=ignore_eos)))
        r = self._get()
        r["ipc_ms"] = (time.perf_counter() - t) * 1e3 - r["wall_s"] * 1e3
        return r

    def close(self):
        if self.proc.is_alive():
            self.q_in.put(None)
            self.proc.join(timeout=30)


def qa_prompt(question, passages):
    ctx = "\n\n".join(f"[{i+1}] {p}" for i, p in enumerate(passages))
    return (f"Answer the question using the passages. Reply with a short phrase only.\n\n"
            f"{ctx}\n\nQuestion: {question}\nAnswer:")


# SQuAD-style EM / F1
def _norm(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def em_f1(pred, golds):
    p = _norm(pred)
    em = max(float(p == _norm(g)) for g in golds)
    best = 0.0
    for g in golds:
        pt, gt = p.split(), _norm(g).split()
        same = sum((collections.Counter(pt) & collections.Counter(gt)).values())
        if same:
            pr, rc = same / len(pt), same / len(gt)
            best = max(best, 2 * pr * rc / (pr + rc))
    return em, best


def env_info():
    import psutil
    vm = psutil.virtual_memory()
    return dict(load1=os.getloadavg()[0], mem_avail_gb=vm.available / 2**30,
                rss_gb=psutil.Process().memory_info().rss / 2**30,
                time=time.strftime("%Y-%m-%d %H:%M:%S"))


# ---------------------------------------------------------------- monitoring
# All sources are readable without root:
#  - DRM fdinfo of this process (/proc/self/fdinfo): xe -> GPU memory (resident GTT)
#    and compute-engine busy cycles; intel_vpu -> NPU memory.
#  - NPU busy counter in sysfs (device-wide, so other NPU users would show up too).
#  - RAPL package energy, only if energy_uj is readable (root-only by default).
NPU_BUSY = Path("/sys/class/accel/accel0/device/npu_busy_time_us")
RAPL = Path("/sys/class/powercap/intel-rapl:0")


def drm_snapshot():
    """Per-process GPU/NPU counters summed over this process's DRM clients."""
    seen, s = set(), dict(gpu_gtt_kib=0, gpu_ccs_cycles=0, npu_mem_kib=0, npu_resident_kib=0)
    for fd in os.listdir("/proc/self/fdinfo"):
        try:
            txt = open(f"/proc/self/fdinfo/{fd}").read()
        except OSError:
            continue
        if "drm-driver" not in txt:
            continue
        kv = dict(l.split(":", 1) for l in txt.splitlines() if ":" in l)
        kv = {k.strip(): v.strip() for k, v in kv.items()}
        key = (kv["drm-driver"], kv.get("drm-client-id"))
        if key in seen:
            continue
        seen.add(key)
        num = lambda k: int(kv.get(k, "0").split()[0])
        if kv["drm-driver"] == "xe":
            s["gpu_gtt_kib"] += num("drm-resident-gtt")
            s["gpu_ccs_cycles"] += num("drm-cycles-ccs")
        elif kv["drm-driver"] == "intel_vpu":
            s["npu_mem_kib"] += num("drm-total-memory")
            s["npu_resident_kib"] += num("drm-resident-memory")
    return s


def _read_int(p):
    try:
        return int(p.read_text())
    except (OSError, ValueError):
        return None


def counters():
    """Cumulative counters; take one before and one after a block and pass both to delta()."""
    return dict(t=time.perf_counter(), ccs=drm_snapshot()["gpu_ccs_cycles"],
                npu_us=_read_int(NPU_BUSY), rapl_uj=_read_int(RAPL / "energy_uj"))


def delta(a, b):
    """active_gpu: GPU compute cycles used by this process; active_npu_ms: device-wide NPU
    busy time; energy_j: RAPL package energy (None when not readable)."""
    e = None
    if a["rapl_uj"] is not None and b["rapl_uj"] is not None:
        d = b["rapl_uj"] - a["rapl_uj"]
        if d < 0:  # counter wrapped
            d += _read_int(RAPL / "max_energy_range_uj") or 0
        e = d / 1e6
    npu = None if a["npu_us"] is None or b["npu_us"] is None else (b["npu_us"] - a["npu_us"]) / 1e3
    return dict(active_gpu_cycles=b["ccs"] - a["ccs"], active_npu_ms=npu, energy_j=e,
                window_ms=(b["t"] - a["t"]) * 1e3)


class ResourceMonitor:
    """Background sampler of memory peaks: process RSS, system used memory, GPU GTT,
    NPU memory. take() returns the peaks since the last take() and resets them."""

    def __init__(self, interval_s=0.02):
        import psutil, threading
        self._proc, self._vm = psutil.Process(), psutil.virtual_memory
        self.interval_s = interval_s
        self._lock, self._stop = threading.Lock(), threading.Event()
        self._peak = {}
        self._sample()
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    def _now(self):
        vm, d = self._vm(), drm_snapshot()
        return dict(rss_gb=self._proc.memory_info().rss / 2**30,
                    sys_used_gb=(vm.total - vm.available) / 2**30,
                    gpu_gtt_mb=d["gpu_gtt_kib"] / 1024, npu_mem_mb=d["npu_mem_kib"] / 1024)

    def _sample(self):
        now = self._now()
        with self._lock:
            for k, v in now.items():
                self._peak[k] = max(self._peak.get(k, v), v)
        return now

    def _loop(self):
        while not self._stop.wait(self.interval_s):
            self._sample()

    def take(self, prefix="peak_"):
        now = self._sample()
        with self._lock:
            out = {prefix + k: v for k, v in self._peak.items()}
            self._peak = dict(now)
        return out

    def now(self, prefix="now_"):
        return {prefix + k: v for k, v in self._now().items()}

    def stop(self):
        self._stop.set(); self._th.join()


def new_run_dir(name):
    """Create a fresh write-once run folder: <runs>/<date>_<host>_<name>/."""
    d = Path(CFG["paths"]["runs"]) / f"{time.strftime('%Y-%m-%d_%H%M%S')}_{socket.gethostname()}_{name}"
    d.mkdir(parents=True, exist_ok=False)
    return d
