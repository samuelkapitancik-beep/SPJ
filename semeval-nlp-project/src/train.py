"""Hlavný beh Etapy 1: split -> B0 -> B1 (grid na dev) -> refit train+dev -> test (jednorazovo).

Použitie: python -m src.train --config configs/experiment.yaml
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from .utils import ROOT, load_config, set_seed, save_json, environment_info
from .data_loader import prepare_splits
from .baselines import (majority_baseline, plausibility_heuristic, random_premises,
                        build_tfidf_lr, predict_with, grid_search_cv)
from .evaluate import official_scores, official_scores_st2, bootstrap_ci, confusion, mcnemar_exact, tce_noise_floor
from .predict import export_predictions, run_official_file_scoring


def summarize(name, gold, pred, cfg, extra=None):
    s = official_scores(gold, pred)
    ci = bootstrap_ci(gold, pred, cfg["bootstrap"]["n_resamples"], cfg["bootstrap"]["alpha"], cfg["seed"])
    return {"model": name, **{k: s[k] for k in ("accuracy", "tce", "tce_intra", "tce_inter", "combined_score",
            "acc_plausible_valid", "acc_implausible_valid", "acc_plausible_invalid", "acc_implausible_invalid")},
            "ci95": ci, "confusion": confusion(gold, pred), **(extra or {})}


def main(config_path: str):
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    out = ROOT / cfg["paths"]["outputs"]
    train, dev, test1, test2, report = prepare_splits(cfg)
    save_json(report, out / "data_report.json")
    for n, d in (("train", train), ("dev", dev)):
        save_json(d, ROOT / cfg["paths"]["processed"] / f"{n}.json")

    res = {"config": cfg, "environment": environment_info(), "data_report": report}

    # ---- B0 ----
    dev_rows = [summarize("B0a majority", dev, majority_baseline(train, dev), cfg),
                summarize("B0b plausibility==validity (diagnostic)", dev, plausibility_heuristic(dev), cfg)]

    # ---- B1: výber hyperparametrov CV na train (dev ani test sa nepoužívajú) ----
    g = cfg["tfidf_lr"]
    best, grid_results = grid_search_cv(train, g["grid"], g["max_iter"], cfg["seed"], g["selection"]["folds"])
    res["b1_best_params"], res["b1_grid"] = best, grid_results
    m = build_tfidf_lr(best["analyzer"], best["ngram_range"], best["C"], g["max_iter"], cfg["seed"])
    m.fit([x["syllogism"] for x in train], [x["validity"] for x in train])
    dev_rows.append(summarize("B1 TF-IDF+LR (train only)", dev, predict_with(m, dev), cfg))
    res["dev"] = dev_rows

    # ---- Finálny beh: refit na train+dev, TEST sa vyhodnocuje jednorazovo ----
    full = train + dev
    mf = build_tfidf_lr(best["analyzer"], best["ngram_range"], best["C"], g["max_iter"], cfg["seed"])
    mf.fit([x["syllogism"] for x in full], [x["validity"] for x in full])
    preds1 = {"B0a_majority": majority_baseline(full, test1), "B0b_plaus_heuristic": plausibility_heuristic(test1),
              "B1_tfidf_lr": predict_with(mf, test1)}
    test_rows = [summarize(n, test1, p, cfg) for n, p in preds1.items()]
    for n, p in preds1.items():
        export_predictions(p, out / "predictions" / f"st1_{n}.json")
    # kontrola formátu priamo oficiálnym skriptom
    res["official_file_check_b1"] = run_official_file_scoring(test1, preds1["B1_tfidf_lr"])
    res["test_st1"] = test_rows
    res["mcnemar_b1_vs_majority"] = mcnemar_exact(test1, preds1["B1_tfidf_lr"], preds1["B0a_majority"])

    # ---- ST2: validita + (referenčný) výber premís. B1 nebol na ST2 trénovaný ----
    pv = predict_with(mf, test2)
    rp = random_premises(test2, cfg["seed"])
    st2_pred = [{"id": p["id"], "validity": p["validity"],
                 "relevant_premises": (r if p["validity"] else [])} for p, r in zip(pv, rp)]
    res["test_st2_B1_validity_plus_random_premises"] = official_scores_st2(test2, st2_pred)
    export_predictions(st2_pred, out / "predictions" / "st2_B1_randompremises.json")

    res["tce_noise_floor"] = {str(a): tce_noise_floor(a) for a in (50, 61, 70, 80, 90, 97)}
    save_json(res, out / "baseline_results.json")
    print(json.dumps({"best_B1": best, "dev": [(r["model"], round(r["accuracy"], 2), round(r["tce"], 2), round(r["combined_score"], 2)) for r in dev_rows],
                      "test": [(r["model"], round(r["accuracy"], 2), round(r["tce"], 2), round(r["combined_score"], 2)) for r in test_rows]}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/experiment.yaml")
    main(ap.parse_args().config)
