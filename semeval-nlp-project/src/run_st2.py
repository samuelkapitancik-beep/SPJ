"""Subtask 2: vyhodnotenie všetkých modelov (lokálne, bez GPU). Štruktúra kopíruje ST1:
  B0a/B0b (+ B0c dolná hranica výberu premís), B1 TF-IDF+LR, B2 DeBERTa (cache z Colabu),
  N1 parser -> abstrakcia -> Llama 3 8B (cache z Colabu), S2 symbolika + hybridy so zálohou.

Dev (predvolené): 9 syntetických ST2 množín z ST1 dev (`st2_devset`); B0/B1 sa fitujú iba na ST1 train.
    python -m src.run_st2                       -> outputs/st2_dev_results.json
Test: jednorazové vyhodnotenie zafixovanej konfigurácie (st2.symbolic); B0/B1 sa fitujú na ST1 train+dev
(ako src/train.py). Log v st2.test_log bráni opakovanému behu. B2/N1 sa zahrnú, ak je k dispozícii cache pre test.
    python -m src.run_st2 --split test --final  -> outputs/st2_test_results.json, outputs/predictions/st2_*.json
"""
from __future__ import annotations
import argparse, datetime, hashlib, itertools, json, os, statistics, tempfile
from .utils import ROOT, set_seed, save_json, environment_info
from .st2_config import load_st2_config
from .data_loader import load_json
from .st2_devset import make_st2_dev
from .st2_symbolic import predict as s2_predict, TIE_BREAKS, INVALID_RULES
from .st2_llm import load_predictions as n1_load, REPRESENTATIONS
from .st2_encoder import load_seed_scores, set_scores, scores_to_predictions, mean_scores
from .baselines import (majority_baseline, plausibility_heuristic, random_premises,
                        build_tfidf_lr, predict_with, grid_search_cv)
from .st2_baselines import overlap_premises, st2_with_premises
from .st2_evaluate import st2_breakdown, bootstrap_ci_st2, official_24
from .predict import export_predictions

GRID_FALLBACKS = ("invalid", "valid", "B1")
METRICS = ("accuracy", "f1_premises", "tce", "combined_score")
CELLS = ("acc_plausible_valid", "acc_implausible_valid", "acc_plausible_invalid", "acc_implausible_invalid")


def fit_b1(train: list[dict], cfg: dict, best: dict | None = None):
    """B1 (TF-IDF + LR) s hyperparametrami z 5-násobnej CV na `train` (rovnaká procedúra ako src/train.py)."""
    g = cfg["tfidf_lr"]
    if best is None:
        best, _ = grid_search_cv(train, g["grid"], g["max_iter"], cfg["seed"], g["selection"]["folds"])
    m = build_tfidf_lr(best["analyzer"], best["ngram_range"], best["C"], g["max_iter"], cfg["seed"])
    m.fit([x["syllogism"] for x in train], [x["validity"] for x in train])
    return m, best


def baseline_preds(data: list[dict], train: list[dict], b1, cfg: dict) -> dict:
    """ST2 baseliny. Výber premís pri 'platné': lexikálny prekryv so záverom (B0c: náhodná dvojica = dolná hranica)."""
    ov = overlap_premises(data)
    maj = majority_baseline(train, data)
    return {
        "B0a majority+overlap": st2_with_premises(maj, data, ov),
        "B0b plaus==valid+overlap (diag)": st2_with_premises(plausibility_heuristic(data), data, ov),
        "B0c majority+random": st2_with_premises(maj, data, random_premises(data, cfg["seed"])),
        "B1 tfidf+overlap": st2_with_premises(predict_with(b1, data), data, ov),
    }


def neural_preds(set_name: str, data: list[dict], b2_seeds: dict, cfg: dict) -> tuple[dict, dict, dict]:
    """Predikcie z cache (Colab): N1 pre obe reprezentácie, B2 po seedoch + ensemble (iba pre hybrid).
    Vráti (single, b2_per_seed, extra_info); chýbajúca cache = model sa vynechá."""
    single, info = {}, {}
    for rep in REPRESENTATIONS:
        r = n1_load(set_name, data, rep, cfg["st2"]["n1"]["output"])
        if r is not None:
            single[f"N1 {rep}"], info[f"N1 {rep}"] = r
    per_seed, scores = {}, []
    for s, d in b2_seeds.items():
        sc = set_scores(d, set_name, data)
        if sc is not None:
            per_seed[s] = scores_to_predictions(data, sc, d["threshold"]); scores.append(sc)
    if scores:
        single["B2 ensemble (iba záloha)"] = scores_to_predictions(data, mean_scores(scores), 0.5)
    return single, per_seed, info


def fallback_source(fb: str, base: dict, single: dict):
    """Záloha pre S2: 'invalid'/'valid' alebo predikcie iného modelu (B1, B2 ensemble, N1 ...)."""
    key = {"B1": "B1 tfidf+overlap", "B2": "B2 ensemble (iba záloha)",
           "N1-abstract": "N1 abstract", "N1-original": "N1 original"}.get(fb)
    if key is None:
        return fb
    src = base.get(key) or single.get(key)
    return None if src is None else {p["id"]: p for p in src}


def evaluate_set(set_name, data, train, b1, cfg, b2_seeds, grid=True) -> dict:
    base = baseline_preds(data, train, b1, cfg)
    single, b2_per_seed, info = neural_preds(set_name, data, b2_seeds, cfg)
    res = {n: st2_breakdown(data, p) for n, p in base.items()}
    for n, p in single.items():
        if not n.startswith("B2 ensemble"):
            res[n] = {**st2_breakdown(data, p), **info.get(n, {})}
    if b2_per_seed:                          # B2 ako v ST1: priemer metrík cez seedy (+ sd)
        runs = [st2_breakdown(data, p) for p in b2_per_seed.values()]
        res["B2 deberta"] = {k: statistics.mean(r[k] for r in runs) for k in METRICS + CELLS}
        res["B2 deberta"]["seed_sd_combined"] = statistics.stdev(r["combined_score"] for r in runs) if len(runs) > 1 else 0.0
        res["B2 deberta"]["n_seeds"] = len(runs)
    sc = cfg["st2"]["symbolic"]
    combos = list(itertools.product(TIE_BREAKS, INVALID_RULES, GRID_FALLBACKS)) if grid else []
    # hybridy so zvoleným tie/rule a každou dostupnou neurónovou zálohou
    combos += [(sc["tie_break"], sc["invalid_rule"], fb) for fb in ("B2", "N1-abstract", "N1-original")]
    for tie, rule, fb in combos:
        f = fallback_source(fb, base, single)
        if f is None:
            continue
        p, dec = s2_predict(data, f, tie, rule)
        res[f"S2 tie={tie} inv={rule} fb={fb}"] = st2_breakdown(data, p, dec)
    return res


def summarize(per_set: dict, variants, seeds) -> dict:
    """Priemer (a sd) cez dev seedy pre každý model a variant; modely, ktoré nie sú vo všetkých množinách, sa vynechajú.
    Model s abstenciou v niektorej množine nemá oficiálne metriky - uvedie sa iba pokrytie (ako N1 v ST1)."""
    models = [m for m in next(iter(next(iter(per_set.values())).values()))
              if all(m in per_set[v][s] for v in variants for s in seeds)]
    out = {}
    for m in models:
        out[m] = {}
        if any(per_set[v][s][m].get("official_withheld") for v in variants for s in seeds):
            for v in variants:
                rs = [per_set[v][s][m] for s in seeds]
                out[m][v] = {"official_withheld": True,
                             "answer_coverage": round(statistics.mean(r.get("answer_coverage", 1.0) for r in rs), 4),
                             "accuracy_with_abstentions_as_errors": round(statistics.mean(
                                 r.get("accuracy_with_abstentions_as_errors", r.get("accuracy")) for r in rs), 2)}
            continue
        for v in variants:
            vals = {k: [per_set[v][s][m][k] for s in seeds] for k in METRICS}
            out[m][v] = {k: round(statistics.mean(x), 2) for k, x in vals.items()}
            out[m][v]["combined_sd"] = round(statistics.stdev(vals["combined_score"]), 2) if len(seeds) > 1 else 0.0
    return out


def tie_summary(per_set: dict, variants, seeds) -> dict:
    """Porovnanie pravidiel pre remízy: koľkokrát pravidlo vybralo správnu dvojicu (inv=any_parsed, fb=invalid)."""
    out = {}
    for tie in TIE_BREAKS:
        name = f"S2 tie={tie} inv=any_parsed fb=invalid"
        agg = {"n": 0, "gold_valid": 0, "gold_pair_among_valid": 0, "chosen_pair_correct": 0}
        for v in variants:
            for s in seeds:
                for k in agg:
                    agg[k] += per_set[v][s][name]["ties"][k]
        out[tie] = agg
    return out


def select_fallback(summary: dict, cfg: dict) -> dict:
    """Vopred zapísané pravidlo (configs/st2.yaml, st2.fallback_selection): spomedzi kandidátov s oficiálnymi
    výsledkami na VŠETKÝCH dev množinách vyhrá najvyššie priemerné combined_score cez varianty; pri rozdiele
    < `tie_tolerance` vyhrá kandidát skôr v zozname (jednoduchší). Kandidát s abstenciou (nejednoznačná odpoveď LLM)
    nemá oficiálne metriky, preto sa vylúči (`excluded`); `missing` = výsledky ešte neexistujú. Iba dev, nikdy test."""
    sc, sel = cfg["st2"]["symbolic"], cfg["st2"]["fallback_selection"]
    ranking, missing, excluded = [], [], []
    for fb in sel["candidates"]:
        name = f"S2 tie={sc['tie_break']} inv={sc['invalid_rule']} fb={fb}"
        if name not in summary:
            missing.append(fb)
        elif any(r.get("official_withheld") for r in summary[name].values()):
            excluded.append(fb)
        else:
            ranking.append((fb, round(statistics.mean(v["combined_score"] for v in summary[name].values()), 2)))
    if not ranking:
        return {"ranking": [], "selected": None, "missing": missing, "excluded": excluded, "matches_config": False}
    top = max(s for _, s in ranking)
    chosen = next(fb for fb, s in ranking if top - s < sel["tie_tolerance"])
    return {"ranking": sorted(ranking, key=lambda r: -r[1]), "selected": chosen, "missing": missing,
            "excluded": excluded, "matches_config": chosen == sc["fallback"]}


def print_table(summary: dict, variants, only=None) -> None:
    print(f"{'model':48s}" + "".join(f" | {v:^28s}" for v in variants))
    print(f"{'':48s}" + " | ACC    F1     TCE    CS(sd)  " * len(variants))
    for m, r in summary.items():
        if only and not only(m):
            continue
        row = f"{m:48s}"
        for v in variants:
            x = r[v]
            if x.get("official_withheld"):
                row += f" | pokrytie {100 * x['answer_coverage']:5.1f} %, bez ofic. metrík "
                continue
            row += f" | {x['accuracy']:5.1f}  {x['f1_premises']:5.1f}  {x['tce']:5.1f}  {x['combined_score']:4.1f}({x['combined_sd']:.1f})"
        print(row)


def run_dev(cfg: dict) -> None:
    c = cfg["st2"]
    train, base, pool = load_json("data/processed/train.json"), load_json(c["dev_base"]), load_json(c["dev_pool"])
    b1, best = fit_b1(train, cfg)                              # fit iba na ST1 train; dev ST1 je báza dev množín
    b2_seeds = load_seed_scores(c["b2"]["output"])
    per_set = {}
    for v in c["dev_variants"]:
        per_set[v] = {}
        for s in c["dev_seeds"]:
            data = make_st2_dev(base, pool, v, s)
            save_json(data, ROOT / c["dev_dir"] / f"dev_{v}_s{s}.json")
            per_set[v][s] = evaluate_set(f"dev_{v}_s{s}", data, train, b1, cfg, b2_seeds)
    summary = summarize(per_set, c["dev_variants"], c["dev_seeds"])
    ties = tie_summary(per_set, c["dev_variants"], c["dev_seeds"])
    fb_sel = select_fallback(summary, cfg)
    save_json({"config": cfg, "environment": environment_info(), "b1_best_params": best,
               "b2_seeds_available": sorted(b2_seeds), "summary_mean_over_seeds": summary,
               "tie_break_comparison": ties, "fallback_selection": fb_sel, "per_set": per_set},
              ROOT / cfg["paths"]["outputs"] / "st2_dev_results.json")
    sc = c["symbolic"]
    main_rows = lambda m: (not m.startswith("S2") or m.startswith(f"S2 tie={sc['tie_break']} inv={sc['invalid_rule']}"))
    print_table(summary, c["dev_variants"], main_rows)
    print("\n(celá mriežka S2 vrátane ablácií je v outputs/st2_dev_results.json)")
    print("remízy (súčet cez varianty a seedy):", json.dumps(ties))
    if not b2_seeds:
        print(f"B2: výstupy {c['b2']['output']}/seed_*/ nenájdené - riadky B2 chýbajú.")
    print(f"\nvoľba zálohy S2 (pravidlo st2.fallback_selection, iba dev): poradie {fb_sel['ranking']}")
    print(f"  -> vybraná: {fb_sel['selected']} | v konfigurácii: {sc['fallback']} | "
          f"{'SEDÍ' if fb_sel['matches_config'] else 'NESEDÍ - uprav st2.symbolic.fallback pred testom'}"
          + (f" | chýbajú výsledky pre: {fb_sel['missing']}" if fb_sel["missing"] else "")
          + (f" | vylúčené (abstencie, bez ofic. metrík): {fb_sel['excluded']}" if fb_sel["excluded"] else ""))


def run_test(cfg: dict) -> None:
    c = cfg["st2"]; log_path = ROOT / c["test_log"]; sc = c["symbolic"]
    if log_path.exists():
        raise SystemExit(f"Test ST2 už bol vyhodnotený ({log_path}). Opakovaný beh je zablokovaný; "
                         "ak je naozaj potrebný, log zmaž a dôvod zapíš do správy.")
    dev_path = ROOT / cfg["paths"]["outputs"] / "st2_dev_results.json"
    sel = load_json(dev_path).get("fallback_selection") if dev_path.exists() else None
    if not sel or sel["missing"] or not sel["matches_config"]:
        raise SystemExit("Voľba zálohy S2 nie je uzavretá na dev (chýbajú výsledky kandidátov alebo konfigurácia "
                         f"nesedí s pravidlom): {sel}. Spusti `python -m src.run_st2` s cache z Colabu a uprav config.")
    full = load_json("data/processed/train.json") + load_json("data/processed/dev.json")
    test2 = load_json(cfg["paths"]["test_st2"])
    b1, best = fit_b1(full, cfg)
    b2_seeds = load_seed_scores(c["b2"]["output"])
    base = baseline_preds(test2, full, b1, cfg)
    single, b2_per_seed, info = neural_preds("test", test2, b2_seeds, cfg)
    main_fb = fallback_source(sc["fallback"], base, single)
    if main_fb is None:                     # kontrola PRED zápisom logu - inak by sa test „minul“ bez výsledku
        raise SystemExit(f"Záloha {sc['fallback']!r} nemá predikcie pre test (chýba cache z Colabu). Test sa nespustil.")
    preds, decisions = dict(base), {}
    for n, p in single.items():
        if not n.startswith("B2 ensemble"):
            preds[n] = p
    for s, p in b2_per_seed.items():
        preds[f"B2 deberta seed{s}"] = p
    for fb in dict.fromkeys(["invalid", sc["fallback"]]):     # čistá symbolika + zvolený hybrid
        p, dec = s2_predict(test2, fallback_source(fb, base, single), sc["tie_break"], sc["invalid_rule"])
        name = f"S2 tie={sc['tie_break']} inv={sc['invalid_rule']} fb={fb}"
        preds[name], decisions[name] = p, dec
    res = {}
    for n, p in preds.items():
        res[n] = {**st2_breakdown(test2, p, decisions.get(n)), **info.get(n, {})}
        if not res[n].get("official_withheld"):
            res[n]["ci95"] = bootstrap_ci_st2(test2, p, cfg["bootstrap"]["n_resamples"], cfg["bootstrap"]["alpha"], cfg["seed"])
        export_predictions(p, ROOT / cfg["paths"]["outputs"] / "predictions" / f"st2_{n.replace(' ', '_').replace('=', '-')}.json")
    if b2_per_seed:
        runs = [res[f"B2 deberta seed{s}"] for s in b2_per_seed]
        res["B2 deberta (priemer seedov)"] = {k: statistics.mean(r[k] for r in runs) for k in METRICS}
        res["B2 deberta (priemer seedov)"]["seed_sd_combined"] = (
            statistics.stdev(r["combined_score"] for r in runs) if len(runs) > 1 else 0.0)
    main_name = f"S2 tie={sc['tie_break']} inv={sc['invalid_rule']} fb={sc['fallback']}"
    with tempfile.TemporaryDirectory() as d:             # kontrola formátu priamo oficiálnym skriptom
        g, pp, o = (os.path.join(d, k) for k in ("gold.json", "pred.json", "out.json"))
        for path, obj in ((g, test2), (pp, preds[main_name])):
            with open(path, "w", encoding="utf-8") as f:
                json.dump(obj, f)
        official_24.run_full_scoring(g, pp, o)
        with open(o, encoding="utf-8") as f:
            res["official_file_check_main"] = json.load(f)
    save_json({"config": cfg, "environment": environment_info(), "b1_best_params": best,
               "b2_seeds_available": sorted(b2_seeds), "main_model": main_name, "results": res},
              ROOT / cfg["paths"]["outputs"] / "st2_test_results.json")
    cfg_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
    save_json({"evaluated_at": datetime.datetime.now().isoformat(timespec="seconds"), "config_sha256_12": cfg_hash,
               "models": list(res)}, log_path)
    for n, r in res.items():
        if n.startswith("official"):
            continue
        if r.get("official_withheld"):
            print(f"{n:48s}", f"bez oficiálnych metrík (abstencie: {r['unparsable_answers']}, pokrytie {r['answer_coverage']:.3f})")
        else:
            print(f"{n:48s}", {k: round(r[k], 2) for k in METRICS})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--st2-config", default="configs/st2.yaml")
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--final", action="store_true", help="povinné pre --split test (jednorazové vyhodnotenie)")
    a = ap.parse_args()
    cfg = load_st2_config(a.config, a.st2_config); set_seed(cfg["seed"])
    if a.split == "test":
        if not a.final:
            raise SystemExit("Test ST2 sa vyhodnocuje iba raz, so zafixovanou konfiguráciou: pridaj --final.")
        run_test(cfg)
    else:
        run_dev(cfg)


if __name__ == "__main__":
    main()
