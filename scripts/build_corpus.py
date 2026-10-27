"""Build the RAG passage corpus + question sets from raw SQuAD v1.1 / HotpotQA.

Corpus = every unique SQuAD (train + dev) paragraph + the 10 distractor-setting
paragraphs of each sampled HotpotQA question. All are Wikipedia paragraphs, and
every sampled question's gold paragraph(s) are guaranteed to be in the corpus.

Outputs (in <datasets>/<corpus>/):
  passages.jsonl          {pid, title, text, source}
  questions_squad_dev.jsonl   {qid, question, answers, gold_pids}
  questions_hotpot.jsonl      {qid, question, answers, gold_pids, type, level}
  manifest.json           counts, seed, sha256 of each output
"""
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    ds = ROOT / cfg["paths"]["datasets"]
    c = cfg["corpus"]
    out = ds / cfg["datasets"]["corpus"]
    if out.exists():
        sys.exit(f"{out} exists; datasets are append-only — bump the version instead")

    squad = ds / cfg["datasets"]["squad_raw"]
    train = pd.read_parquet(squad / "train-00000-of-00001.parquet")
    dev = pd.read_parquet(squad / "validation-00000-of-00001.parquet")
    hotpot = pd.read_parquet(ds / cfg["datasets"]["hotpot_raw"] / "validation-00000-of-00001.parquet")

    passages, key2pid = [], {}

    def add(title, text, source):
        key = (title, text)
        if key not in key2pid:
            key2pid[key] = len(passages)
            passages.append({"pid": len(passages), "title": title, "text": text, "source": source})
        return key2pid[key]

    for df, src in [(dev, "squad_dev"), (train, "squad_train")]:
        for t, ctx in df[["title", "context"]].drop_duplicates().itertuples(index=False):
            add(t.replace("_", " "), ctx, src)

    sq = dev.sample(n=c["n_squad_dev_questions"], random_state=c["seed"])
    squad_qs = [{
        "qid": r.id, "question": r.question,
        "answers": sorted(set(r.answers["text"].tolist())),
        "gold_pids": [key2pid[(r.title.replace("_", " "), r.context)]],
    } for r in sq.itertuples()]

    hq = hotpot.sample(n=c["n_hotpot_questions"], random_state=c["seed"])
    hotpot_qs = []
    for r in hq.itertuples():
        pids = {t: add(t, "".join(s), "hotpot_dev")
                for t, s in zip(r.context["title"], r.context["sentences"])}
        gold = sorted({pids[t] for t in r.supporting_facts["title"]})
        hotpot_qs.append({"qid": r.id, "question": r.question, "answers": [r.answer],
                          "gold_pids": gold, "type": r.type, "level": r.level})

    out.mkdir(parents=True)
    files = {"passages.jsonl": passages, "questions_squad_dev.jsonl": squad_qs,
             "questions_hotpot.jsonl": hotpot_qs}
    for name, rows in files.items():
        write_jsonl(out / name, rows)
    manifest = {
        "seed": c["seed"],
        "counts": {name: len(rows) for name, rows in files.items()},
        "passages_by_source": pd.Series([p["source"] for p in passages]).value_counts().to_dict(),
        "sha256": {name: sha256(out / name) for name in files},
    }
    json.dump(manifest, open(out / "manifest.json", "w"), indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
