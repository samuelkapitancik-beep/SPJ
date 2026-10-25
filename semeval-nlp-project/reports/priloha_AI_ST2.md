# Príloha: použitie LLM a AI agentov – Subtask 2
*Záznam M. Kelemena pre ST2. Dopĺňa `priloha_AI.md` (ST1), ktorú tento súbor nemení.*

| Pole | Záznam |
|---|---|
| Nástroj | Claude (Anthropic), aplikácia Claude (Cowork) – dva rozhovory: (A) plánovanie, kontrola a integrácia do repa, (B) implementácia ST2 |
| Model | Claude Opus 5.5 |
| Dátum | 25. 10. 2026 |
| Účel | návrh rozdelenia práce, návrh a implementácia ST2 (dev množina, S2, baseliny, B2, N1), testy, vyhodnotenie, koncept správy `st2_sprava.md` |

## Prompty (rozhovor A, skrátene, v poradí)
1. „musíme spraviť všetky subtasky, navrhni mi rozdelenie práce“ → ST1 robia Adam a Samo, ST2 celý M. Kelemen v rovnakej štruktúre (B0, B1, B2, N1 + symbolika).
2. Príprava úvodného promptu pre rozhovor B (uvedenie do úlohy SemEval-2026 Task 11, ST2, repozitár tímu); doslovné znenie je v histórii rozhovoru B.
3. „potrebujem na druhý subtask výsledky z prvého?“ → nie, ST2 iba importuje kód ST1 (parser, metriky), nepotrebuje jeho výsledky.
4. „spustiť všetky seedy“ (B2: 5 seedov ako v ST1); „Samo svoju vec nemení“ → ST2 iba v nových súboroch `st2_*`.
5. „naklonovať repo“ → klon commitu `b419a57`; požiadavka preniesť ST2 z neaktuálneho zipu (`9c4cd93`) na `b419a57`.
6. Aplikovanie patchu, kontrola, commit `7f58b88`, push cez GitHub Desktop.
7. „dnes to treba odovzdať“ → zoškrtaný plán: test ST2 so symbolikou a baselinmi, B2/N1 presunuté (pozri `st2_sprava.md`, kap. 6).

Prompty v rozhovore B (implementácia) písal M. Kelemen ako krátke pokyny po slovensky; **doslovné znenie treba doplniť z histórie rozhovoru B.** [DOPLNIŤ]

## Čo AI vytvorila
* Kód: `src/st2_devset.py`, `st2_symbolic.py`, `st2_abstraction.py`, `st2_baselines.py`, `st2_evaluate.py`, `st2_encoder.py`, `st2_llm.py`,
  `st2_config.py`, `run_st2.py`; `configs/st2.yaml`; `tests/test_st2.py` (31 testov); `README_ST2.md`, `COLAB_ST2.md`.
* Syntetické dev množiny `data/processed/st2/` (9 súborov) a výsledky `outputs/st2_dev_results.json`, `outputs/st2_test_results.json`.
* Koncept správy `reports/st2_sprava.md` a táto príloha.

## Ako bolo overené
* Patch pridáva iba nové súbory; overené `git apply --stat` a `git status` (žiadny sledovaný súbor ST1 sa nezmenil).
* Všetky testy prešli (62 + 1 preskočený, vyžaduje torch) lokálne aj v čistom klone.
* Metriky ST2 sa počítajú oficiálnou implementáciou; oficiálny skript bol spustený aj nad exportovaným súborom hlavného modelu s rovnakým výsledkom.
* Kód B2 a N1 bol overený iba testami a skúšobnými behmi s malým `--limit`; **plné behy neexistujú, žiadne výsledky B2/N1 sa neuvádzajú.**

## Prijaté / zamietnuté / opravené
* **Náhodné vyhodnotenie na teste ST2** počas skúšobného behu v rozhovore B: AI to sama ohlásila, výsledky sa zmazali a nepoužili (správa, kap. 6.1).
* AI pôvodne upravila Samove súbory (`run_llm`, `neurosymbolic.py`); na pokyn M. Kelemena bolo všetko vrátené a ST2 je úplne oddelený.
  Chyba kompatibility s transformers 5 (`return_dict`) sa nahlásila Samovi; jeho nový `run_llama.py` ju už nemá.
* Remízy: pôvodný návrh M. Kelemena `jaccard` nahradilo `struct` na základe dev porovnania (53 vs. 50 správnych z 87).
* B2: pridaná vážená strata (odchýlka od ST1, zdôvodnená nevyváženosťou dvojíc ~1:9). N1: `max_new_tokens` 16 namiesto 8.
* Zúženie kandidátov zálohy na {invalid, valid, B1} kvôli termínu, pred vyhodnotením testu (správa, kap. 6.2).

## Doplní tím
* Doslovné prompty z rozhovoru B. [DOPLNIŤ]
* Ktoré návrhy tím prijal alebo zamietol nad rámec uvedeného. [DOPLNIŤ]
