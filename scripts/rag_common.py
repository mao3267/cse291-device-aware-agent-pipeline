"""Shared pieces of the RAG pipeline: config, encoders, retrieval, LLM, scoring.

Each model is loaded once on one device. Encoders are compiled with static
shapes, and the same shapes are used on CPU, GPU, and NPU (NPU requires static
shapes), so device comparisons are controlled. All paths come from config.yaml.
"""
import collections, json, os, re, socket, string, time
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


class Encoder:
    """Static-shape OpenVINO encoder (bge embedding or reranker)."""

    def __init__(self, key, device, batch, seq):
        self.path = model_path(key)
        self.device, self.batch, self.seq = device, batch, seq
        self.tok = AutoTokenizer.from_pretrained(self.path)
        t = time.perf_counter()
        model = core.read_model(self.path / "openvino_model.xml")
        model.reshape({i.get_any_name(): [batch, seq] for i in model.inputs})
        self.names = [i.get_any_name() for i in model.inputs]
        self.compiled = core.compile_model(model, device)
        self.req = self.compiled.create_infer_request()
        self.load_s = time.perf_counter() - t
        self.exec_devices = self.compiled.get_property("EXECUTION_DEVICES")

    def infer(self, enc):
        self.req.infer({n: enc[n].astype(np.int64) for n in self.names})
        return self.req.get_output_tensor(0).data.copy()

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
    def __init__(self, device, batch=1, seq=64):
        super().__init__("embed", device, batch, seq)

    def embed(self, texts):
        enc, n = self.tokenize(texts)
        out = self.infer(enc)[:n, 0, :]          # CLS pooling
        return out / np.linalg.norm(out, axis=1, keepdims=True)


class Reranker(Encoder):
    def __init__(self, device, batch=20, seq=256):
        super().__init__("rerank", device, batch, seq)

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
    def __init__(self, device, max_prompt_len=1024, min_response_len=256):
        cfg = {}
        if device == "NPU":
            cfg = {"MAX_PROMPT_LEN": max_prompt_len, "MIN_RESPONSE_LEN": min_response_len}
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
                    out_tokens=m.get_num_generated_tokens())


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


def new_run_dir(name):
    """Create a fresh write-once run folder: <runs>/<date>_<host>_<name>/."""
    d = Path(CFG["paths"]["runs"]) / f"{time.strftime('%Y-%m-%d_%H%M%S')}_{socket.gethostname()}_{name}"
    d.mkdir(parents=True, exist_ok=False)
    return d
