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

## Prompty (rozhovor B – implementácia ST2, doslovne)
Prepísané zo snímok obrazovky rozhovoru B (`reports/ai_usage/st2/prompt_*.png`). Dlhé správy aplikácia na mobile skracuje
(„Show more“); skrátené miesto je označené `[…]`, plné znenie je v histórii rozhovoru B.

1. *(úvodný prompt, pripravený v rozhovore A)* „Ahoj, robíme v trojici školský projekt na Spracovanie prirodzeného jazyka (FEI TUKE, Ing. UI).
   Najprv ti vysvetlím, o čo ide, kód ešte nepíš. Riešime SemEval-2026 Task 11 – Disentangling Content and Formal Reasoning in LLMs.
   stránka: https://sites.google.com/view/semeval-2026-task-11 dáta a hodnotiace skripty: https://github.com/neuro-symbolic-ai/semeval_2026_task_11
   V skratke: model má povedať, či je […]“
2. „super, s vacsinou suhlasim. repo davam ako zip (SPJ-main.zip), data/raw si stiahni z oficialneho repa. tie tri veci som si pozrel
   v oficialnom skripte: CE je v percentach (bunky su 0-100), takze ta penalizacia je dost velka; F1 je macro, pre kazdy platny priklad sa
   zrata precision a recall, spriemeruju sa a F1 sa rata z tych priemerov; na poradi indexov nezalezi, […]“ *(+ príloha SPJ-main.zip)*
3. „jj generator mam, je to st2_augment.py (funkcia make_st2_like), prikladam aj st2_symbolic.py. to 89/81 vyslo na dev.json cez
   make_st2_like(dev, train, seed=42, hard=True). hard tam bere asi polovicu rusivych viet so slovami zo zaveru a zvysok nahodne, rusive vety
   idu len z train. kludne to zober ako zaklad do st2_devset.py a easy/hard si tam dorob po svojom. ked vyjde viac platnych dvojic naraz,
   ja som vyberal tu, ktorej slova sa najviac prekryvaju so zaverom […]“ *(+ prílohy st2_augment.py, st2_symbolic.py)*
4. „este jedna vec, ST2 robim cely sam, takze nic nebude spustat spoluziak, colab si pustim ja. a chcem, aby ST2 kopiroval strukturu, ako to
   zacali pri ST1, nech sa to da v sprave porovnat 1:1: B0 – vacsinova trieda + B0b validity = plausibility (to uz mame); B1 – TF-IDF + LR
   (to uz mame); B2 – doladeny DeBERTa ako pri ST1, len pre ST2 (navrhni, ako to spravit aj s vyberom premis, trenovat na syntetickych datach
   z train) […]“
5. „ok diky ze si to napisal, do prilohy o AI to dame presne tak ako si to popisal. aby to bolo ciste, zvol teraz tu poslednu vec (zalohu pre
   S2) len podla dev a zapis ju do configu s poznamkou kedy a preco, este predtym nez pustim ostry test. par veci k colabu: 4bit pri N1 –
   musime to mat rovnako aj pri ST1, inak sa to neda porovnat. dam vediet samovi, aby to pustil tiez s --load-4bit, alebo napis, ci sa to da
   spravit inak; B2 pustime vsetkych 5 seedov, cas […]“ *(„to“ = AI ohlásila náhodné vyhodnotenie na teste, pozri nižšie)*
6. „este jedna vec, nechcem menit samovu cast. vrat prosim vsetky zmeny v run_llm / neurosymbolic.py (ST1) do povodneho stavu, presne ako to
   bolo v zipe. ST2 nech je samostatne: nacitanie modelu (aj ten load_4bit) a generovanie daj do vlastneho modulu pre ST2, neimportuj to
   z run_llm; tu chybu s transformers 5 v ST1 neopravuj, len mi napis, co presne pada a ako to opravit, poslem to samovi a on si to upravi
   sam […]“
7. „pozor, repo sa medzitym zmenilo – samo uz pustil B2 a N1 pre ST1 a pushol to (commit b419a57). tvoje ST2 je postavene na starom zipe
   (9c4cd93). prikladam aktualne repo. co sa zmenilo: model sa nacitava cez src/run_llama.py (LlamaRunner) – 4bit NF4 s double quant,
   compute dtype bf16/fp16 podla GPU, revizia 8afb486c1db24fe5011ec46dfbe5b5dccdb575c2, transformers 4.57.1; run_llm v neurosymbolic.py
   je uz len wrapper […]“
8. „https://github.com/samuelkapitancik-beep/SPJ.git“ → AI preniesla ST2 na b419a57 (patch `st2_na_b419a57.patch`).

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
* Plné znenie skrátených promptov (`[…]`) z rozhovoru B, ak to vyučujúci vyžaduje. [DOPLNIŤ]
* Ktoré návrhy tím prijal alebo zamietol nad rámec uvedeného. [DOPLNIŤ]
