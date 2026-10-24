"""B2 (NESPUSTENÉ v tomto prostredí - chýba torch/HF prístup): doladenie enkódera na surovom texte.

Určené pre Google Colab (GPU). Použitie:  python -m src.finetune_encoder --model microsoft/deberta-v3-base --seeds 0 1 2 3 4
Rovnaký split ako B0/B1 (data/processed/train.json, dev.json), výber epochy podľa dev, test sa vyhodnotí raz.
Pred použitím v správe je nutné kód spustiť a overiť; zatiaľ NEOBSAHUJE žiadne zmerané výsledky.
"""
from __future__ import annotations
import argparse, numpy as np
from .utils import load_config, set_seed, save_json, ROOT
from .data_loader import load_json
from .evaluate import official_scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/deberta-v3-base")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=5); ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
    cfg = load_config()
    train, dev = load_json("data/processed/train.json"), load_json("data/processed/dev.json")
    test = load_json(cfg["paths"]["test_st1"])
    tok = AutoTokenizer.from_pretrained(a.model); dev_ = "cuda" if torch.cuda.is_available() else "cpu"

    def enc(d):
        e = tok([x["syllogism"] for x in d], truncation=True, max_length=128, padding=True, return_tensors="pt")
        return e, torch.tensor([int(x["validity"]) for x in d])

    def predict(model, d):
        model.eval(); e, _ = enc(d); out = []
        with torch.no_grad():
            for i in range(0, len(d), 64):
                b = {k: v[i:i + 64].to(dev_) for k, v in e.items()}
                out += model(**b).logits.argmax(-1).tolist()
        return [{"id": x["id"], "validity": bool(p)} for x, p in zip(d, out)]

    results = []
    for seed in a.seeds:
        set_seed(seed); torch.manual_seed(seed)
        model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=2).to(dev_)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
        e, y = enc(train); steps = a.epochs * ((len(train) + a.batch - 1) // a.batch)
        sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
        best, best_state = -1, None
        for ep in range(a.epochs):
            model.train(); perm = torch.randperm(len(train))
            for i in range(0, len(train), a.batch):
                idx = perm[i:i + a.batch]
                b = {k: v[idx].to(dev_) for k, v in e.items()}
                loss = model(**b, labels=y[idx].to(dev_)).loss
                loss.backward(); opt.step(); sch.step(); opt.zero_grad()
            s = official_scores(dev, predict(model, dev))["accuracy"]   # výber epochy podľa dev accuracy
            if s > best: best, best_state = s, {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        results.append({"seed": seed, "dev_acc": best, **official_scores(test, predict(model, test))})
    save_json({"model": a.model, "runs": results}, ROOT / cfg["paths"]["outputs"] / f"b2_{a.model.split('/')[-1]}.json")


if __name__ == "__main__":
    main()
