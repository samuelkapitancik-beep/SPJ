"""B2 pre ST2: DeBERTa-v3-base s výberom premís, tréning rovnaký ako B2 v ST1 (src/finetune_encoder.py).

Rovnaké ako ST1 (prevzaté, nie skopírované): model, seedy, epochy, lr, dávka, akumulácia a max_length sú predvolené
hodnoty `finetune_encoder.parser()`; gradient checkpointing cez `finetune_encoder.enable_checkpointing`. Rovnaký
postup: AdamW (weight decay 0,01), lineárny plán s 10 % warmup krokov optimalizátora, FP16 + GradScaler na GPU
(krok plánovača sa preskočí, ak GradScaler preskočil krok), orezanie gradientu na 1,0, pomalý tokenizér
(use_fast=False), pevná revízia (B2 v ST1), checkpoint najlepšej epochy, test v samostatnom režime nad zmrazeným
checkpointom.

Rozdiely oproti ST1, ktoré vyžaduje ST2 (zdôvodnenie v správe):
  * párový cross-encoder: vstup "premisa_i premisa_j záver" (rovnaký 3-vetný text, aký dostáva parser v S2),
    štítok 1 = relevantná dvojica platného sylogizmu, 0 = ostatné (aj pôvodná dvojica NEPLATNÉHO sylogizmu = ST1 negatív);
    predikcia: platné <=> max pravdepodobnosť dvojice > 0,5 (pevný prah), premisy = argmax;
  * tréningové dáta: syntetické ST2 príklady z ST1 TRAIN (st2_devset, varianty easy/mixed/hard), negatívne dvojice
    podvzorkované, nevyváženosť tried kompenzuje váha v krížovej entropii (ST1 má triedy vyvážené, váhu nemá);
  * výber epochy: (ACC + F1)/2 na jednej dev množine, pri zhode skoršia (ST1: dev ACC, pri zhode skoršia).

Colab (GPU):  python -m src.st2_encoder                -> <output>/seed_N/ (checkpoint, history, dev_pair_scores.json)
              python -m src.st2_encoder --mode test    -> <output>/seed_N/test_pair_scores.json (iba skóre, NEHODNOTÍ sa)
Lokálne:      `load_seed_scores`, `scores_to_predictions` (bez torch).
"""
from __future__ import annotations
import argparse, datetime, gc, json, math, os, random
from itertools import combinations
from pathlib import Path
os.environ.setdefault("USE_TF", "0")       # ako finetune_encoder: bez TensorFlow backendu z Colabu
from .utils import ROOT, set_seed, save_json
from .data_loader import load_json, prepare_splits
from .preprocessing import split_sentences
from .st2_devset import make_st2_dev
from .st2_config import load_st2_config
from .st2_llm import text_sha, eval_sets
from .evaluate import official_scores_st2
from .neural_common import environment, fingerprint, resolve_revision
from .finetune_encoder import enable_checkpointing, parser as st1_parser


def pair_text(premises: list[str], i: int, j: int, conclusion: str) -> str:
    return f"{premises[i]} {premises[j]} {conclusion}"


def all_pairs(x: dict) -> list[tuple[int, int, str]]:
    sents = split_sentences(x["syllogism"])
    prem, concl = sents[:-1], sents[-1]
    return [(i, j, pair_text(prem, i, j, concl)) for i, j in combinations(range(len(prem)), 2)]


def build_pair_examples(data: list[dict], neg_per_example: int, seed: int) -> list[dict]:
    """Tréningové dvojice. Platný príklad: správna dvojica (1) + 1 dvojica zdieľajúca jednu správnu premisu + náhodné.
    Neplatný príklad: pôvodná dvojica (0, to je ST1 negatív) + náhodné. Potrebuje pole `orig_premises` zo st2_devset."""
    rng = random.Random(seed)
    out = []
    for x in data:
        pairs = {(i, j): t for i, j, t in all_pairs(x)}
        gold = tuple(sorted(x["orig_premises"]))
        chosen = [(gold, int(x["validity"]))]
        rest = [p for p in pairs if p != gold]
        if x["validity"]:
            near = [p for p in rest if set(p) & set(gold)]
            if near:
                h = rng.choice(near); chosen.append((h, 0)); rest.remove(h)
        k = max(neg_per_example - (len(chosen) - 1), 0)
        chosen += [(p, 0) for p in rng.sample(rest, min(k, len(rest)))]
        out += [{"id": x["id"], "pair": list(p), "text": pairs[p], "label": y} for p, y in chosen]
    return out


def scores_to_predictions(data: list[dict], scores: dict, threshold: float = 0.5) -> list[dict]:
    """Skóre dvojíc {id: {"pairs": [[i, j, p], ...]}} -> predikcie ST2 (platné <=> max p > prah)."""
    preds = []
    for x in data:
        i, j, p = max(scores[x["id"]]["pairs"], key=lambda r: (r[2], -r[0], -r[1]))
        v = p > threshold
        preds.append({"id": x["id"], "validity": bool(v), "relevant_premises": [i, j] if v else []})
    return preds


def mean_scores(per_seed: list[dict]) -> dict:
    """Priemer pravdepodobností dvojíc cez seedy (ensemble; používa sa iba ako záloha v hybride S2 + B2)."""
    out = {}
    for xid in per_seed[0]:
        rows = [{(i, j): p for i, j, p in s[xid]["pairs"]} for s in per_seed]
        out[xid] = {"pairs": [[i, j, sum(r[(i, j)] for r in rows) / len(rows)] for (i, j) in rows[0]]}
    return out


# ------------------------------------------------------------------------------------------------ lokálne čítanie
def load_seed_scores(output: str) -> dict[int, dict]:
    """Lokálne: {seed: {"threshold", "scores": {množina: {id: ...}}}} pre seedy s hotovým dev behom."""
    out = {}
    for d in sorted((ROOT / output).glob("seed_*")):
        tr, dev = d / "training.json", d / "dev_pair_scores.json"
        if not (d.is_dir() and d.name.split("_")[1].isdigit() and tr.exists() and dev.exists()):
            continue
        meta = load_json(tr)
        scores = dict(load_json(dev))
        if (d / "test_pair_scores.json").exists():
            scores["test"] = load_json(d / "test_pair_scores.json")
        out[meta["seed"]] = {"seed": meta["seed"], "threshold": meta["threshold"], "scores": scores}
    return out


def set_scores(seed_data: dict, set_name: str, data: list[dict]) -> dict | None:
    """Skóre jednej množiny z jedného seedu; None, ak chýbajú; chyba, ak text nesedí (iná verzia množiny)."""
    s = seed_data["scores"].get(set_name)
    if s is None or any(x["id"] not in s for x in data):
        return None
    for x in data:
        if s[x["id"]]["text_sha"] != text_sha(x["syllogism"]):
            raise ValueError(f"B2 seed {seed_data['seed']} / {set_name}: text príkladu {x['id']} nesedí")
    return s


# ------------------------------------------------------------------------------------------------ Colab
def parser():
    d = st1_parser().parse_args([])                     # hyperparametre B2 zo ST1 (jediný zdroj)
    b = load_st2_config()["st2"]["b2"]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["train", "test"], default="train")
    p.add_argument("--model", default=d.model)
    p.add_argument("--revision", default=b["revision"], help="predvolene revízia B2 v ST1")
    p.add_argument("--seeds", nargs="+", type=int, default=d.seeds)
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--batch", type=int, default=d.batch)
    p.add_argument("--accumulation", type=int, default=d.accumulation)
    p.add_argument("--max-length", type=int, default=d.max_length)
    p.add_argument("--output", default=b["output"])
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="skúška: iba N príkladov, výstup do <output>_skuska")
    return p


def main(argv=None):   # pragma: no cover - vyžaduje GPU (overené smoke testom s malým modelom na CPU)
    a = parser().parse_args(argv)
    if min(a.epochs, a.batch, a.accumulation, a.max_length) < 1 or a.lr <= 0:
        raise ValueError("Training parameters must be positive.")
    if len(set(a.seeds)) != len(a.seeds):
        raise ValueError("Seeds must be unique.")
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and not a.allow_cpu:
        raise RuntimeError("Enable a GPU runtime in Colab (or explicitly pass --allow-cpu).")
    cfg = load_st2_config(); b = cfg["st2"]["b2"]
    out = ROOT / (a.output + ("_skuska" if a.limit else ""))
    out.mkdir(parents=True, exist_ok=True)
    cut = (lambda d: d[: a.limit]) if a.limit else (lambda d: d)

    def score(model, tok, data):
        """Pravdepodobnosť triedy 1 pre všetky dvojice (ako predict v ST1: fp32, inference_mode, dávka a.batch)."""
        model.eval()
        items = [(x, i, j, t) for x in data for i, j, t in all_pairs(x)]
        probs = []
        with torch.inference_mode():
            for start in range(0, len(items), a.batch):
                rows = items[start:start + a.batch]
                inputs = tok([t for *_, t in rows], padding=True, truncation=True, max_length=a.max_length,
                             return_tensors="pt").to(device)
                probs += torch.softmax(model(**inputs).logits.float(), -1)[:, 1].tolist()
        res = {}
        for (x, i, j, _), p in zip(items, probs):
            res.setdefault(x["id"], {"text_sha": text_sha(x["syllogism"]), "pairs": []})["pairs"].append([i, j, round(p, 6)])
        return res

    def release():
        gc.collect()
        if device == "cuda": torch.cuda.empty_cache()

    if a.mode == "test":
        test = cut(load_json(cfg["paths"]["test_st2"]))
        for seed in a.seeds:
            dest = out / f"seed_{seed}"
            if not (dest / "training.json").exists():
                raise FileNotFoundError(f"Seed {seed} nemá dokončený tréning: {dest}")
            if (dest / "test_pair_scores.json").exists():
                raise FileExistsError(f"Test already scored: {dest}")
            tok = AutoTokenizer.from_pretrained(dest / "checkpoint")
            model = AutoModelForSequenceClassification.from_pretrained(dest / "checkpoint").to(device)
            save_json(score(model, tok, test), dest / "test_pair_scores.json")
            del model
            release()
        return

    train = load_json("data/processed/train.json")
    dev = load_json("data/processed/dev.json")
    expected_train, expected_dev, _, _, _ = prepare_splits(cfg)
    if train != expected_train or dev != expected_dev:
        raise ValueError("Processed splits differ from the baseline protocol.")
    train = cut(train)
    train_st2 = [x for v in b["train_variants"] for x in make_st2_dev(train, train, v, b["train_seed"])]
    sets = {k: cut(v) for k, v in eval_sets(cfg, "dev").items()}
    select = sets[b["select_set"]]
    revision = resolve_revision(a.model, a.revision)
    tok = AutoTokenizer.from_pretrained(a.model, revision=revision, use_fast=False)
    for seed in a.seeds:
        dest = out / f"seed_{seed}"
        if (dest / "training.json").exists() and (dest / "dev_pair_scores.json").exists():
            print(f"Seed {seed}: hotový, preskakujem", flush=True); continue
        if dest.exists():                                  # prerušený seed: odložiť (nemazať) a začať odznova
            dest.rename(out / f"seed_{seed}_prerusene_{datetime.datetime.now():%Y%m%d_%H%M%S}")
        dest.mkdir(parents=True)
        set_seed(seed)
        torch.manual_seed(seed)
        if device == "cuda": torch.cuda.manual_seed_all(seed)
        ex = build_pair_examples(train_st2, b["neg_per_example"], seed)
        n_pos = sum(e["label"] for e in ex)
        weight = torch.tensor([1.0, (len(ex) - n_pos) / max(n_pos, 1)], device=device)
        model = AutoModelForSequenceClassification.from_pretrained(a.model, revision=revision, num_labels=2).to(device)
        enable_checkpointing(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
        batches = math.ceil(len(ex) / a.batch)
        steps = a.epochs * math.ceil(batches / a.accumulation)
        scheduler = get_linear_schedule_with_warmup(optimizer, int(steps * 0.1), steps)
        use_amp = device == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        best, best_epoch, history = float("-inf"), None, []
        for epoch in range(1, a.epochs + 1):
            model.train()
            order = torch.randperm(len(ex)).tolist()
            optimizer.zero_grad(set_to_none=True)
            loss_total = 0.0
            for batch_index, start in enumerate(range(0, len(ex), a.batch)):
                rows = [ex[i] for i in order[start:start + a.batch]]
                inputs = tok([r["text"] for r in rows], padding=True, truncation=True,
                             max_length=a.max_length, return_tensors="pt").to(device)
                labels = torch.tensor([r["label"] for r in rows], device=device)
                group_start = (batch_index // a.accumulation) * a.accumulation * a.batch
                group_size = min(a.accumulation * a.batch, len(ex) - group_start)
                with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                    logits = model(**inputs).logits
                loss = torch.nn.functional.cross_entropy(logits.float(), labels, weight=weight)
                loss_total += loss.item() * len(rows)
                scaler.scale(loss * len(rows) / group_size).backward()
                if (batch_index + 1) % a.accumulation == 0 or batch_index + 1 == batches:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    previous_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    if scaler.get_scale() >= previous_scale: scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
            s = official_scores_st2(select, scores_to_predictions(select, score(model, tok, select), b["threshold"]))
            sel = (s["accuracy"] + s["f1_premises"]) / 2
            history.append({"epoch": epoch, "train_loss": loss_total / len(ex), "select_avg": sel,
                            **{k: s[k] for k in ("accuracy", "f1_premises", "tce", "combined_score")}})
            print(f"Seed {seed}, epoch {epoch}: {b['select_set']} (ACC+F1)/2={sel:.2f}, TCE={s['tce']:.2f}", flush=True)
            if sel > best:                                 # pri zhode ostáva skoršia epocha (ako ST1)
                best, best_epoch = sel, epoch
                model.save_pretrained(dest / "checkpoint", safe_serialization=True)
                tok.save_pretrained(dest / "checkpoint")
            save_json(history, dest / "history.json")
        del model, optimizer, scheduler, scaler
        release()
        best_model = AutoModelForSequenceClassification.from_pretrained(dest / "checkpoint").to(device)
        save_json({name: score(best_model, tok, data) for name, data in sets.items()}, dest / "dev_pair_scores.json")
        del best_model
        release()
        save_json({"experiment": "B2-ST2", "split": "dev", "model": a.model, "revision": revision,
                   "gradient_checkpointing": {"use_reentrant": False}, "seed": seed,
                   "selection": f"{b['select_set']}_acc_f1_avg_earliest_tie", "best_epoch": best_epoch,
                   "best_select_avg": best, "max_length": a.max_length, "threshold": b["threshold"],
                   "n_train_pairs": len(ex), "n_pos": n_pos, "class_weight": weight.tolist(),
                   "arguments": vars(a), "st2_config": b, "environment": environment(),
                   "train_sha256": fingerprint(train_st2),
                   "dev_sets_sha256": {k: fingerprint(v) for k, v in sets.items()}}, dest / "training.json")
        print(f"Seed {seed}: uložené {dest}", flush=True)


if __name__ == "__main__":
    main()
