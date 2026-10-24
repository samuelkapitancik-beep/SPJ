# Etapa 1 – Návrh projektu a baseline riešenie
**SemEval-2026 Task 11: oddelenie obsahu a formálneho usudzovania v jazykových modeloch (ST1, EN)**
*Predmet: Spracovanie prirodzeného jazyka · Tím: [DOPLNIŤ mená] · Dátum: 21. 10. 2026*

> Stav správy: koncept. Všetky čísla pochádzajú zo súborov `outputs/*.json` a dajú sa reprodukovať príkazom `python -m src.train`.
> Miesta označené [DOPLNIŤ] musí doplniť tím. Model B2 (doladený enkóder) **nebol spustený**, preto žiadne jeho výsledky neuvádzame.

## 1. Úvod a motivácia
Veľké jazykové modely často zamieňajú **logickú platnosť** argumentu (záver nutne vyplýva z premís) s jeho **vierohodnosťou**
(záver súhlasí so znalosťou sveta). Tento jav sa nazýva *content effect*; organizátori úlohy ho opisujú ako dôsledok toho,
že predtrénovanie spája usudzovanie s obsahom (Valentino et al., 2026). Príklad z tréningových dát:
„*All cars are vehicles. No animal is a car. Therefore, no animal can be a vehicle.*“ – záver je vierohodný, ale argument
je **neplatný**. Spoľahlivé usudzovanie nezávislé od obsahu je nutné všade, kde sa model používa na kontrolu argumentov.

## 2. Formulácia problému a ciele
**Problém (ST1).** Pre anglický sylogizmus (3 vety: 2 premisy + záver) predikovať binárny štítok `validity`.
**Ciele Etapy 1:** (i) zdôvodniť metriky a metodiku; (ii) vytvoriť reprodukovateľné baseline; (iii) zistiť, čo baseline
skutočne zachytáva. **Ciele Etapy 2** (návrh, zatiaľ nerealizované): symbolické/abstrahované riešenie, ktoré je voči obsahu odolné.

## 3. Súvisiace práce
* **Úloha a dáta:** Valentino et al. (2026), *SemEval-2026 Task 11*, ACL Anthology 2026.semeval-1.450. Dáta vznikli z 24 platných
  aristotelovských schém (s existenčným predpokladom); obsah vytvoril a parafrázoval LLM (Gemini), nasledovala manuálna validácia.
* **Výsledky účastníkov (z overview článku):** ~250 účastníkov, 35 systémových článkov; 10 systémov dosiahlo na ST1 ACC = 100, TCE = 0.
  Najspoľahlivejšie boli neuro-symbolické pipeline, kde LLM prevedie argument do formálnej podoby a o platnosti rozhoduje deterministický solver.
* **Konkrétny systém:** ITLC (arXiv:2603.02676) – LLM normalizuje text do kategorického tvaru A/E/I/O, regexový parser určí mód a figúru
  a platnosť sa nájde v tabuľke; uvádza 100 % na ST1 aj ST3.
* **Content effect u ľudí a modelov:** Dasgupta et al. (2022), arXiv:2207.07051; Valentino et al. (2025), arXiv:2505.12189 (activation steering).
  Ďalšie zdroje (Bertolazzi et al. 2024; Eisape et al. 2024; Ranaldi et al. 2025; Kim et al. 2024) uvádzame podľa zoznamu v README organizátorov;
  ich plné texty sme **nečítali**, preto z nich nepreberáme konkrétne tvrdenia. [TÍM: doplniť po prečítaní]
* **Zero-shot referencie (overview, Tab. 2):** Qwen3-14B ACC 97,38 / TCE 4,23 / CS 36,62; Llama-3.1-8B ACC 58,33 / CS 11,87.

**Dôsledok:** keďže ACC = 100 už bolo dosiahnuté, nemôže byť prínosom ďalšie „vyššie skóre“; prínos musí vysvetľovať *prečo a za akých podmienok* riešenie funguje.

## 4. Dataset a jeho analýza
| Množina | n | Poznámka |
|---|---|---|
| train (oficiálny) | 960 | 480 platných / 480 neplatných; bunky 240/240/246/234 |
| → train (naše) / dev | 766 / 192 | stratifikovaný split, seed 42; z trainu odstránené 2 texty prítomné v teste |
| test ST1 | 191 | bunky 48/48/48/47; štítky sú **verejné** |
| test ST2 | 190 | 6–8 viet; platné príklady majú vždy 2 relevantné premisy (0-indexované) |

Kontroly: 0 duplicít v train; 2 normalizované texty v train aj teste ST1 (odstránené z trainu); žiadne spoločné `id`.
Texty majú vždy 3 vety, 14–40 slov, a až 350 rôznych úvodných fráz záveru (veľká povrchová variabilita).
**Obmedzenia datasetu:** syntetický (generovaný LLM), anglický, malý test (≈48 príkladov na bunku).

## 5. Metodika a metriky
* **ACC** – podiel správnych predikcií. **TCE** – priemer dvoch zložiek: (intra) rozdiel presnosti platných/neplatných príkladov medzi
  vierohodnými a nevierohodnými, (inter) rozdiel presnosti platných vs. neplatných v rámci rovnakej vierohodnosti.
  **CS = ACC / (1 + ln(1 + TCE))** je oficiálna rankingová metrika. Implementáciu sme prevzali **nezmenenú** z oficiálneho skriptu a vlastnú
  rýchlu verziu overili testom voči nemu.
* **Prečo nestačí iba CS (naše zistenie):** pri ~48 príkladoch na bunku posunie jedna chyba presnosť bunky o 2,1 p. b. a môj výpočet podľa oficiálneho vzorca dáva:
  0 chýb → CS 100; 1 chyba → 58,1. Preto navyše reportujeme bootstrap CI (ACC), McNemarov test a **šumové dno TCE**
  (simulácia modelu s nezávislými chybami: pri presnosti 65 % je očakávané TCE ≈ 8, 95. percentil ≈ 14).
* **Upozornenie k CI pre TCE/CS:** TCE je súčet absolútnych rozdielov zašumených veličín, preto je pri bootstrap resamplingu
  kladne skreslené; percentilový interval nemusí obsahovať bodový odhad. Na robustné závery používame ACC a McNemarov test.
* **Protokol:** výber hyperparametrov 5-násobnou CV na train, dev ako kontrola, test jednorazovo po refite na train+dev.

## 6. Baseline riešenia
| ID | Model | Úloha v experimentoch |
|---|---|---|
| B0a | väčšinová trieda | dolná hranica ACC |
| B0b | `validity := plausibility` (používa zlatý štítok → *diagnostika*, nie model) | ukotvenie TCE = 100 |
| B1 | TF-IDF (word/char n-gramy, sublinear TF) + logistická regresia | lexikálny baseline; **hypotéza:** blízko náhody |
| B2 | DeBERTa-v3-base, doladenie, 5 seedov | *pripravený kód, nespustený* |

Mriežka B1: analyzátor {word, char_wb} × n-gramy {(1,1), (1,2), (2,4)} × C {0,1; 1; 10; 100} = 24 konfigurácií.
Vybraná: `char_wb`, n-gramy (2,4), C = 100 (CV: ACC 70,4; TCE 0,59; CS 48,1).

## 7. Výsledky (ST1)
| Model | Dev ACC | Test ACC [95 % CI] | Test TCE | Test CS |
|---|---|---|---|---|
| B0a väčšinová | 50,0 | 50,3 [42,9; 57,6] | 50,0 | 10,2 |
| B0b plausibility (diagnostika) | 50,5 | 50,3 [43,5; 57,6] | 100,0 | 9,0 |
| **B1 TF-IDF + LR** | 67,7 | **65,4 [58,6; 71,7]** | 10,9 | 18,8 |
| *Referencia:* Qwen3-14B zero-shot (overview) | – | 97,4 | 4,2 | 36,6 |

* B1 vs. väčšinová trieda: McNemar presný p = 0,0022 (57 vs. 28 diskordantných párov) – **B1 je štatisticky lepší ako náhoda.**
* Presnosti buniek B1 na teste (PV / IV / PI / II): 75,0 / 66,7 / 53,2 / 66,7.
* Oficiálny skript spustený priamo nad exportovanými predikciami dáva rovnaké hodnoty (ACC 65,445; TCE 10,904; CS 18,823).
* **Dôležité:** pri tomto počte príkladov je TCE 10,9 v rámci šumového dna (≈ 8 ± 6). **Z testu nemožno tvrdiť, že B1 má
  preukázateľný content effect;** nízke CS pochádza hlavne z nízkej presnosti.

### 7.1 Transparentnosť protokolu (zmena po prvom behu)
Prvý beh vyberal hyperparametre podľa dev CS. Dev CS je však na 192 príkladoch veľmi zašumené: vybraná konfigurácia mala dev ACC 61,5 %,
hoci iné dosahovali 71 %, a vyhrala vďaka náhodne nízkemu TCE. Navyše bodový odhad TCE ležal mimo vlastného bootstrap intervalu.
Výber sme preto zmenili na CV nad train. **Test bol pred touto zmenou už vyhodnotený** (ACC 61,3; TCE 8,8; CS 18,65), takže zmena protokolu
nie je úplne nezávislá od testu; motiváciou bol pozorovaný šum na dev, nie výsledok na teste. Prvý beh je uložený v
`outputs/baseline_results_run1_dev_selection.json`.

### 7.2 Diagnostika: čo B1 skutočne využíva (fit train, hodnotené na dev)
| Vstup modelu | word 1–2 ACC | char 2–4 ACC |
|---|---|---|
| celý text | 67,2 | 66,1 |
| iba premisy | 64,6 | 64,6 |
| **iba záver** | **71,9** | 72,4 |

* Samotný záver predikuje lepšie než celý text → model využíva **prior nad formou záveru**, nie logický vzťah premís a záveru.
  Najsilnejšie príznaky smerom k „platné“: `not`, `few`, `certain`, `none`; k „neplatné“: `all`, `circle`, `cats`, `educator`.
* Prítomnosť obsahových slov (`circle`, `cats`, `chair`) medzi najvýraznejšími príznakmi ukazuje lexikálnu memorizáciu termínov.
* Hodnoty 64–72 % sú teda **povrchové skratky**, nie usudzovanie. (Ide o jeden split a jednu hodnotu C; je to diagnostika, nie dôkaz.)

### 7.3 Test odolnosti na ST2 (B1 netrénovaný na ST2)
ACC 51,1; TCE 48,0; presnosť platných 96–98 %, neplatných 2–9 %; F1 premís (náhodná dvojica) 33,7; CS 8,7.
B1 predikuje takmer všetko ako „platné“ – dlhšie texty s rušivými premisami sú mimo jeho rozdelenia. Toto **nie je férové porovnanie modelov**,
iba ilustrácia, že lexikálny baseline sa neprenáša.

## 8. Diskusia a obmedzenia
* Test má 191 príkladov, preto CI presnosti ≈ ±6,5 p. b.; rozdiely menšie než ~10 p. b. medzi dvoma modelmi nie sú spoľahlivo rozlíšiteľné.
* Testové štítky sú verejné; disciplína „test iba raz“ je procedurálna. Dev a CV sú odvodené z jedného 958-príkladového zdroja.
* Dáta sú generované LLM – skratky (forma záveru, slovník) môžu byť artefakty generovania, nie vlastnosť usudzovania.
* B2 nie je vyhodnotený; bez neho nevieme, ako ďaleko je štandardný neurónový prístup od 97 % zero-shot referencie.
* Ľudskú kontrolu, ktorú predpisuje zadanie pre analýzu chýb, sme zatiaľ nevykonali.

## 9. Návrh Etapy 2 (nerealizované)
Vlastný prínos: **(C1)** abstrakcia obsahových termínov (X/Y/Z), **(C2)** deterministický parser A/E/I/O + tabuľka 24 platných módov bez LLM,
**(C3)** hybrid s enkóderom, plus **kontrafaktuálna evaluácia** (rovnaká forma, odlišný obsah) so silnejším štatistickým testom než 48 príkladov na bunku.
Hypotézy: H1 – abstrakcia znižuje lexikálnu memorizáciu (pokles váhy obsahových slov a rozdielu ACC medzi vierohodnými/nevierohodnými);
H2 – parser dosiahne ACC ≥ 95 % na ST1 a prenesie sa na ST2; H3 – hybrid je robustnejší než samotný parser na nepokrytých parafrázach.
Experimenty: E3 enkóder na abstrahovanom texte · E4 parser · E5 hybrid · E6 ablácia C1/C2/C3 · E7 kontrafaktuálna sada · E8 ST2 ako OOD.
Poznámka: väčšina účastníkov dosiahla ACC 100 aj na ST1, preto formulujeme prínos ako analýzu mechanizmu, nie ako súťaž o skóre.

## 10. Reprodukovateľnosť
Seed 42; Python 3.13, scikit-learn 1.9.1, numpy 2.5.3, pandas 3.0.5, scipy 1.18.1; konfigurácia `configs/experiment.yaml`;
12 jednotkových testov; príkazy v `README.md`. [DOPLNIŤ po overení čistej inštalácie]

## 11. Literatúra
* Valentino, M., Ranaldi, L., Pucci, G., Ranaldi, F., Freitas, A. (2026). SemEval-2026 Task 11: Disentangling Content and Formal Reasoning in Large Language Models. *Proc. SemEval-2026*. https://aclanthology.org/2026.semeval-1.450/
* Muhamad, W. L. et al. (2026). ITLC at SemEval-2026 Task 11: Normalization and Deterministic Parsing for Formal Reasoning in LLMs. arXiv:2603.02676.
* Dasgupta, I. et al. (2022). Language models show human-like content effects on reasoning tasks. arXiv:2207.07051.
* Valentino, M., Kim, G., Dalal, D., Zhao, Z., Freitas, A. (2025). Mitigating Content Effects on Reasoning in Language Models through Fine-Grained Activation Steering. arXiv:2505.12189.
* Oficiálny repozitár úlohy: https://github.com/neuro-symbolic-ai/semeval_2026_task_11 (commit 6f71ea8).
* Ďalšie tituly (Bertolazzi 2024; Eisape 2024; Ranaldi 2025; Kim 2024) – podľa README organizátorov, text nečítaný, pred odovzdaním doplniť/overiť.
