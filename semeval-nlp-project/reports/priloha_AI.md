# Príloha: použitie LLM a AI agentov (Etapa 1)
*Priebežný záznam. Obsahuje iba to, čo je doložené v tomto rozhovore. Použitie AI inými členmi tímu musí doplniť tím.*

| Pole | Záznam |
|---|---|
| Nástroj | Claude (Anthropic) v aplikácii Claude Code; model Claude Opus 5.5, od druhého zadania Claude Sonnet 5.5 (prepnutie používateľom) |
| Dátum | 21. 10. 2026 |
| Účel | audit zadania a dát SemEval, návrh metodiky, implementácia a testy baseline, koncept správy |
| Prompty | 1) `prompt_01_master_prompt_VERBATIM.md` (doslovné znenie, priložené); 2) „začnime s Etapou 1 – návrh a baseline“; 3) kapitoly 1–2 (formulácia problému, prehľad literatúry; šablóna TUKE); 4) doplnenie zistení z literatúry a bibliografických údajov; 5) kapitoly 3–4 (metodika, návrh a implementácia baseline, neuro-symbolický prístup). Prompty 3 až 5 sú uvedené skrátene; doslovné znenie doplní tím z histórie rozhovoru. |
| Prílohy poskytnuté AI | Projektove_zadanie.pdf, testik-1.jpg, cviko4.pdf (projektové súbory) |

## Čo AI vytvorila
* Audit a porovnanie podúloh, odporúčanie ST1 (+ ST2 ako OOD test).
* Kód: `src/*.py`, `tests/test_pipeline.py`, `configs/experiment.yaml`, README, `src/finetune_encoder.py` (**nespustený a neoverený**).
* Koncept správy `reports/etapa1_sprava.md`.

* Kapitoly 1 až 4 správy (DOCX v šablóne TUKE), literatúra, bibliografia.
* Vlastný regex tokenizér a parser sylogizmov (`src/symbolic.py`), modul neuro-symbolického reťazca (`src/neurosymbolic.py`), vyhodnotenie parsera (`src/run_parser_eval.py`), ručný audit 25 nezhôd (`outputs/parser_audit.json`, jeden hodnotiteľ, AI).
* **LLM časť N1 (Llama 3 8B, zero-shot) je pripravená, ale nebola spustená a nemá žiadne výsledky.**

## Ako bolo overené
* Dáta a oficiálne skripty boli načítané a analyzované priamo (počty, bunky, duplicity, prekryv train/test).
* Rýchla implementácia metriky bola testom porovnaná s oficiálnym skriptom; oficiálny skript bol spustený aj nad exportovanými predikciami.
* 22 jednotkových testov prešlo (vrátane testov parsera, tabuľky 24 platných foriem a extrakcie odpovede).
* Citácie z kapitol 3–4 (Llama 3, Transformers, McNemar, scikit-learn) sme overili iba vyhľadaním ich existencie; úplné bibliografické údaje má pred odovzdaním skontrolovať tím.
* Pravidlá parsera boli vyvíjané iba na trénovacej a vývojovej časti; testovacia množina podúlohy 1 bola vyhodnotená raz po ich zafixovaní. Čísla v správe pochádzajú z `outputs/*.json`.
* Citácie: overené sme len články, ktorých text AI načítala (overview 2026.semeval-1.450, ITLC arXiv:2603.02676); ostatné sú podľa README organizátorov.

## Prijaté / zamietnuté / opravené
* Zmena výberu hyperparametrov z dev na CV po zistení zašumeného dev (AI navrhla, v správe je zverejnená vrátane skutočnosti, že test bol už vyhodnotený).
* AI v audite nevedela otvoriť oficiálnu Google Sites stránku; údaje sú z GitHub README a overview článku.
* Časť údajov z internetu mohla AI prevziať z abstraktov alebo súhrnov; *neoverené nezávisle ľudskou kontrolou* [TÍM: skontrolovať].

* AI pri audite zistila chybu v tokenizéri (ticho zahadzoval znaky s diakritikou), opravila ju a overila, že anglické výsledky sa nezmenili.
* Zadanie požadovalo parser a LLM časť ako baseline; skutočne vykonaný a zmeraný baseline bol TF-IDF + LR. AI to v kapitole 4 uviedla v tabuľke stavu a LLM výsledky nevymýšľala.
* Ručný audit nezhôd (7 chýb parsera, 18 nezhôd s oficiálnym štítkom) je jedno subjektívne posúdenie; nepovažovať za overené bez druhého hodnotiteľa.

## Doplní tím
* Ktoré návrhy tím prijal alebo zamietol, aké úpravy urobil, kto z tímu AI používal, ďalšie prompty mimo tohto rozhovoru. [DOPLNIŤ]
