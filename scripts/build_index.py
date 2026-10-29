"""Embed every corpus passage with the S1 model and save a brute-force index.

Offline preprocessing, not part of any measured pipeline. Writes a new
directory <datasets>/<index>/ and refuses to overwrite an existing one.
Usage: build_index.py [DEVICE]   (default GPU)
"""
import hashlib, json, sys, time

import numpy as np

from rag_common import CFG, CORPUS, INDEX, Embedder, load_passages, passage_text

BATCH, SEQ = 32, 256


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "GPU"
    if INDEX.exists():
        sys.exit(f"{INDEX} exists; bump the index version in config.yaml instead")
    ps = load_passages()
    emb = Embedder(device, batch=BATCH, seq=SEQ)
    out, t = [], time.perf_counter()
    for s in range(0, len(ps), BATCH):
        out.append(emb.embed([passage_text(p) for p in ps[s:s + BATCH]]))
        if s % (BATCH * 100) == 0:
            print(f"embedded {s}/{len(ps)}", flush=True)
    INDEX.mkdir(parents=True)
    np.save(INDEX / "emb.npy", np.concatenate(out).astype(np.float32))
    np.save(INDEX / "pids.npy", np.array([p["pid"] for p in ps], dtype=np.int64))
    sha = {f: hashlib.sha256((INDEX / f).read_bytes()).hexdigest()
           for f in ("emb.npy", "pids.npy")}
    json.dump({"corpus": CFG["datasets"]["corpus"], "model": CFG["models"]["embed"],
               "n_passages": len(ps), "seq": SEQ, "batch": BATCH, "device": device,
               "pooling": "cls+l2", "passage_format": "title: text",
               "build_s": round(time.perf_counter() - t, 1), "sha256": sha},
              open(INDEX / "manifest.json", "w"), indent=1)
    print("INDEX_DONE", INDEX, flush=True)


if __name__ == "__main__":
    main()
