"""Vyhodnotenie deterministickej (symbolickej) časti N1: pokrytie parsera, správnosť tabuľky platnosti a export vstupov pre LLM.

Pravidlá parsera boli vyvíjané výhradne na train+dev. Test ST1 sa vyhodnocuje raz, po zafixovaní pravidiel.
Použitie: python -m src.run_parser_eval
"""
from __future__ import annotations
import collections, json
from .utils import ROOT, load_config, save_json
from .data_loader import load_json
from .neurosymbolic import build_item, PROMPT_SHA
from .evaluate import official_scores, bootstrap_ci, confusion
from .symbolic import count_valid_forms


def evaluate_split(name: str, data: list[dict], cfg: dict, out_dir):
    items = [build_item(x) for x in data]
    save_path = out_dir / f"neurosymbolic_items_{name}.jsonl"
    with open(save_path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    parsed = [(x, it) for x, it in zip(data, items) if it["parsed"]]
    cell = lambda x: f"{'valid' if x['validity'] else 'invalid'}_{'plaus' if x['plausibility'] else 'implaus'}"
    cov_cells = collections.Counter(cell(x) for x, _ in parsed); all_cells = collections.Counter(cell(x) for x in data)
    # S0: symbolická tabuľka platnosti pre rozparsované, nerozparsované -> "neplatné" (iba referencia, nie navrhované riešenie)
    pred = [{"id": x["id"], "validity": bool(it["symbolic_valid"]) if it["parsed"] else False} for x, it in zip(data, items)]
    s = official_scores(data, pred)
    acc_parsed = sum(it["symbolic_valid"] == x["validity"] for x, it in parsed) / max(len(parsed), 1)
    res = {
        "n": len(data), "n_parsed": len(parsed), "coverage": len(parsed) / len(data),
        "coverage_by_cell": {k: cov_cells[k] / all_cells[k] for k in sorted(all_cells)},
        "accuracy_on_parsed": acc_parsed,
        "failure_reasons": dict(collections.Counter(it["parse_failure"].split(":")[0] for it in items if not it["parsed"])),
        "S0_reference_all_examples": {k: s[k] for k in ("accuracy", "tce", "combined_score")},
        "S0_ci95": bootstrap_ci(data, pred, cfg["bootstrap"]["n_resamples"], cfg["bootstrap"]["alpha"], cfg["seed"]),
        "gold_valid_share_unparsed": (sum(x["validity"] for x, it in zip(data, items) if not it["parsed"]) /
                                      max(sum(not it["parsed"] for it in items), 1)),
    }
    return res


def main():
    cfg = load_config(); out = ROOT / cfg["paths"]["outputs"]
    splits = {"train": load_json("data/processed/train.json"), "dev": load_json("data/processed/dev.json"),
              "test_st1": load_json(cfg["paths"]["test_st1"])}
    res = {"valid_forms_with_existential_import": count_valid_forms(), "prompt_sha256_12": PROMPT_SHA,
           **{k: evaluate_split(k, v, cfg, out) for k, v in splits.items()}}
    save_json(res, out / "parser_eval.json")
    for k in splits:
        r = res[k]; print(k, "pokrytie", round(r["coverage"], 3), "ACC(parsed)", round(r["accuracy_on_parsed"], 4),
                          "| S0 ACC/TCE/CS", [round(r["S0_reference_all_examples"][m], 2) for m in ("accuracy", "tce", "combined_score")])


if __name__ == "__main__":
    main()
