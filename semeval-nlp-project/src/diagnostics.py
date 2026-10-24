"""Diagnostika baseline B1: ktoré príznaky nesú signál a z ktorej časti textu pochádza.
Všetko sa fituje na train, hodnotí na dev (test sa nepoužíva)."""
from __future__ import annotations
import json
import numpy as np
from .utils import ROOT, load_config, set_seed, save_json
from .data_loader import load_json
from .baselines import build_tfidf_lr
from .evaluate import official_scores
from .preprocessing import split_sentences


def part(x: dict, which: str) -> str:
    s = split_sentences(x["syllogism"])
    return {"full": " ".join(s), "premises_only": " ".join(s[:-1]), "conclusion_only": s[-1]}[which]


def main():
    cfg = load_config(); set_seed(cfg["seed"])
    train = load_json("data/processed/train.json"); dev = load_json("data/processed/dev.json")
    out = {"part_ablation": [], "top_features": {}}
    for which in ("full", "premises_only", "conclusion_only"):
        for an, ng in (("word", (1, 2)), ("char_wb", (2, 4))):
            m = build_tfidf_lr(an, ng, 10.0, 2000, cfg["seed"])
            m.fit([part(x, which) for x in train], [x["validity"] for x in train])
            pred = [{"id": x["id"], "validity": bool(p)} for x, p in zip(dev, m.predict([part(x, which) for x in dev]))]
            s = official_scores(dev, pred)
            out["part_ablation"].append({"input": which, "analyzer": an, "dev_accuracy": s["accuracy"], "dev_tce": s["tce"]})
    m = build_tfidf_lr("word", (1, 1), 10.0, 2000, cfg["seed"])
    m.fit([x["syllogism"] for x in train], [x["validity"] for x in train])
    names = np.array(m.named_steps["tfidf"].get_feature_names_out()); w = m.named_steps["lr"].coef_[0]
    o = np.argsort(w)
    out["top_features"] = {"towards_invalid": [(names[i], round(float(w[i]), 2)) for i in o[:12]],
                           "towards_valid": [(names[i], round(float(w[i]), 2)) for i in o[::-1][:12]]}
    save_json(out, ROOT / cfg["paths"]["outputs"] / "diagnostics.json")
    print(json.dumps(out, indent=1, ensure_ascii=False))

if __name__ == "__main__":
    main()
