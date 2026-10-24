# SemEval-2026 Task 11 (ST1) – baseline (Etapa 1)

Projekt z predmetu Spracovanie prirodzeného jazyka. Úloha: predikcia **logickej platnosti** sylogizmu nezávisle od
**vierohodnosti** obsahu. Hlavná podúloha: ST1 (EN, binárna klasifikácia). ST2 slúži ako test odolnosti.

Stav: **Etapa 1 hotová pre B0 a B1.** B2 (doladený enkóder) je pripravený ako kód pre Colab a **nebol spustený**.

## Rýchly štart
```bash
pip install -r requirements.txt
python -m pytest -q tests          # 12 testov, ~2 s
python -m src.train                # B0 + B1, ~6 s na CPU, zapíše outputs/baseline_results.json
python -m src.diagnostics          # diagnostika B1 -> outputs/diagnostics.json
```
Dáta sú skopírované z oficiálneho repozitára (commit `6f71ea8`, https://github.com/neuro-symbolic-ai/semeval_2026_task_11)
do `data/raw/`. Oficiálne hodnotiace skripty sú nezmenené v `src/vendor/`.

## Metodika (stručne)
* 960 tréningových príkladov -> 766 train / 192 dev (stratifikované podľa 4 buniek validity x plausibility, seed 42).
  Z trénovacích dát sa odstránia 2 texty, ktoré sú aj v teste.
* Hyperparametre B1 sa volia **5-násobnou CV na train** (pooled out-of-fold). Dev je kontrolná množina, test sa
  vyhodnocuje raz po refite na train+dev.
* Metriky: oficiálne ACC, TCE a CS = ACC/(1+ln(1+TCE)); k tomu 95 % bootstrap CI (5000 resamplov), presný McNemar
  a „šumové dno“ TCE (`evaluate.tce_noise_floor`).
* Testovacie štítky sú verejné; neexistuje skrytý test, preto je disciplína „test iba raz“ **procedurálna**.
  Prvý beh (výber podľa dev) je uložený v `outputs/baseline_results_run1_dev_selection.json`.

## Štruktúra
`src/` data_loader, preprocessing, baselines, evaluate, predict, train, diagnostics, finetune_encoder (B2, nespustené) ·
`configs/experiment.yaml` · `tests/` · `outputs/` · `reports/` (správa + príloha o AI).


## Neuro-symbolický baseline N1 (parser -> abstrakcia -> LLM)
* `src/symbolic.py` - vlastný regex tokenizér, typovanie výrokov (A/E/I/O), identifikácia termov, abstrakcia (X, Y, Z), tabuľka platnosti (256 modelov, 24 platných foriem).
* `src/neurosymbolic.py` - reprezentácie vstupu (original / abstract / canonical), zero-shot prompt (v1, SHA-256 uložený), extrakcia odpovede, `run_llm` (GPU; v tomto prostredí NESPUSTENÉ).
* `python -m src.run_parser_eval` - pokrytie parsera a presnosť symbolického rozhodnutia -> `outputs/parser_eval.json`, `outputs/neurosymbolic_items_*.jsonl`.
* `outputs/parser_audit.json` - ručný audit 25 nezhôd (jeden hodnotiteľ).
Pravidlá parsera boli ladené iba na train+dev; test ST1 bol vyhodnotený jedenkrát po zafixovaní pravidiel.
