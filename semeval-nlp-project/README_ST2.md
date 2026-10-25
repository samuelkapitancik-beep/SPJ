# Subtask 2 (ST2)

ST2 je oddelený od ST1: žiadny existujúci súbor sa nemenil (overené voči commitu b419a57) - ST2 z nich iba importuje
(parser, metriky, baseliny, `run_llama.LlamaRunner`, `finetune_encoder.parser`/`enable_checkpointing`,
`neural_common`). Nastavenia ST2 sú v `configs/st2.yaml`
(načítava sa spolu s `experiment.yaml` cez `src/st2_config.py`).

```bash
python -m pytest -q tests          # všetky testy (ST1 + ST2)
python -m src.run_st2              # ST2 na 9 dev množinách, ~1 min na CPU
```

## Súbory
`src/st2_devset.py` (dev množiny) · `st2_symbolic.py` (S2) · `st2_abstraction.py` (lemmy, abstrakcia pre N1) ·
`st2_baselines.py` · `st2_evaluate.py` (metriky ST2) · `st2_encoder.py` (B2, Colab) · `st2_llm.py` (N1, Colab) ·
`st2_config.py` · `run_st2.py` · `configs/st2.yaml` · `tests/test_st2.py` · `COLAB_ST2.md` (návod pre Colab)

## Metodika
* ST2 nemá trénovacie ani vývojové dáta. `src/st2_devset.py` vytvára syntetické dev množiny z ST1 dev (192 príkladov):
  k dvom pôvodným premisám pridá 3–5 rušivých premís **iba z ST1 train** a premieša ich (záver ostáva posledný).
  Varianty `easy` (náhodné rušivé premisy), `mixed` (≈ polovica zdieľa slovo so sylogizmom; = pôvodné `make_st2_like(hard=True)`)
  a `hard` (prednostne zdieľajú slovo so záverom), seedy 42/43/44 -> `data/processed/st2/`.
* `src/st2_symbolic.py` (S2): každá dvojica premís + záver -> parser zo ST1 -> tabuľka platnosti. Stavy `valid-pair`,
  `invalid-sure`, `undecided` (rozhoduje fallback: `invalid`, `valid` + lexikálny prekryv, `B1`, neskôr LLM).
  Remízy (viac platných dvojíc): `lowest` / `jaccard` / `struct` / `union` (iba ablácia).
* B2 pre ST2 (`src/st2_encoder.py`, Colab): DeBERTa-v3-base ako párový klasifikátor - vstup „premisa_i premisa_j záver“
  (rovnaký 3-vetný text ako pre parser), štítok 1 = relevantná dvojica platného sylogizmu; platné <=> max p > 0.5,
  premisy = argmax. Trénuje sa na syntetických ST2 dátach z ST1 train. Hyperparametre a postup tréningu sú prevzaté
  z B2 v ST1 (`finetune_encoder.parser()`, revízia z `outputs/neural/b2/*/training.json`); navyše váha tried v strate
  (ST2 dvojice sú nevyvážené) a výber epochy podľa (ACC + F1)/2. Výstupy `outputs/neural/st2_b2/seed_N/`.
* N1 pre ST2 (`src/st2_llm.py`, Colab): lexikálna abstrakcia (`st2_abstraction.abstract_lexical`, obsahové lemmy -> umelé slová;
  zachováva rozhodnutie parsera na všetkých rozparsovateľných príkladoch ST1) -> Llama 3 8B načítaná cez
  `run_llama.LlamaRunner` (NF4 + double quant, revízia ako N1 v ST1), prompt ST1 N1 + číslované premisy, odpoveď
  „VALID i j“ / „INVALID“ (max 16 tokenov). Generovanie je overene zhodné s `LlamaRunner.predict`. Nejednoznačná
  odpoveď ostáva `null` ako v ST1. Reprezentácie `abstract` (N1) a `original` (zero-shot ablácia).
  Výstupy `outputs/neural/st2_llama/<množina>/<reprezentácia>/`.
* Colab iba generuje výstupy (návod: `COLAB_ST2.md`); hodnotí sa lokálne. Model alebo hybrid s abstenciou nemá
  oficiálne metriky (ako N1 v ST1), uvádza sa pokrytie a diagnostická accuracy.
* `python -m src.run_st2` - baseliny B0a/B0b/B0c/B1, B2 a N1 (ak je cache) a celá mriežka volieb S2 vrátane hybridov na 9 dev množinách
  -> `outputs/st2_dev_results.json` (priemer a sd cez seedy, rozpad podľa vetiev a buniek, porovnanie remíz).
* `python -m src.run_st2 --split test --final` - **jednorazové** vyhodnotenie konfigurácie `st2.symbolic` na teste ST2;
  spustí sa iba ak je voľba zálohy uzavretá na dev (pravidlo `st2.fallback_selection`), zapíše
  `outputs/st2_test_run_log.json` a opakovaný beh odmietne. **Spustené 25. 10. 2026** (S2 + baseliny; B2/N1 neskôr samostatne) – výsledky a
  zdôvodnenie v `reports/st2_sprava.md`, príloha AI v `reports/priloha_AI_ST2.md`.
* Výhrada: pravidlá parsera boli ladené na ST1 train+dev, takže S2 je na týchto dev množinách optimistická;
  všetkých 9 množín zdieľa tých istých 192 základných príkladov, sd cez seedy preto podhodnocuje neistotu.
