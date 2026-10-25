# Subtask 2 – výber relevantných premís a platnosť (EN)
**SemEval-2026 Task 11, ST2** · *Predmet: Spracovanie prirodzeného jazyka · Autor ST2: Marek Kelemen · Dátum: 25. 10. 2026*

> Všetky čísla pochádzajú zo súborov `outputs/st2_dev_results.json` a `outputs/st2_test_results.json` a dajú sa reprodukovať:
> `python -m src.run_st2` (dev, ~1 min na CPU). Test bol vyhodnotený **raz** (`--split test --final`, log `outputs/st2_test_run_log.json`).
> ST2 je oddelený od ST1: žiadny súbor ST1 sa nemenil, ST2 z nich iba importuje (parser, metriky, baseliny, LlamaRunner).

## 1. Úloha
Vstupom je anglický sylogizmus s **5 až 7 premisami** a záverom (posledná veta). Najviac dve premisy sú relevantné, ostatné sú rušivé.
Výstup: `validity` (true/false) a pri platnom argumente `relevant_premises` – indexy dvoch premís (od 0, záver sa nečísluje);
pri neplatnom `[]`. Poradie indexov nehrá rolu.

**Metriky (oficiálny skript):** ACC platnosti; F1 výberu premís (makro cez príklady, počíta sa iba nad zlatými platnými príkladmi);
TCE ako v ST1; **CS = ((ACC + F1) / 2) / (1 + ln(1 + TCE))**.

**Test ST2:** 190 príkladov, bunky PV/IV/PI/II = 48/47/45/50, počet premís 5/6/7 = 70/67/53. ST2 **nemá trénovacie ani dev dáta.**

## 2. Vývojová množina (syntetická)
Aby sme nič neladili na teste, `src/st2_devset.py` vytvára ST2 príklady z ST1 dev (192 sylogizmov): k dvom pôvodným premisám pridá
3–5 rušivých premís **iba z ST1 train**, premieša ich, záver ostáva posledný. Počet premís sa losuje v pomere testu (70/67/53).
Varianty: `easy` (náhodné rušivé premisy), `mixed` (asi polovica zdieľa slovo so sylogizmom), `hard` (prednostne zdieľajú slovo so záverom);
seedy 42/43/44 → **9 dev množín** (`data/processed/st2/`). Štítky sa preberajú z pôvodného príkladu.

*Výhrada:* všetkých 9 množín zdieľa tých istých 192 základných sylogizmov, takže sd cez seedy podhodnocuje neistotu,
a pravidlá parsera boli ladené na ST1 train+dev, takže S2 je na týchto množinách optimistická.

## 3. Modely (štruktúra ako v ST1)
| ID | Model | Výber premís pri „platné“ | Stav |
|---|---|---|---|
| B0a | väčšinová trieda | lexikálny prekryv so záverom | hotové |
| B0b | `validity := plausibility` (diagnostika, používa zlatý štítok) | prekryv | hotové |
| B0c | väčšinová trieda | náhodná dvojica (dolná hranica F1) | hotové |
| B1 | TF-IDF + LR (fit na ST1, mriežka z CV ako v ST1) | prekryv | hotové |
| **S2** | symbolika: každá dvojica premís + záver → parser zo ST1 → tabuľka 24 platných foriem | dvojica, ktorú parser uznal za platnú | **hotové, hlavný model** |
| B2 | DeBERTa-v3-base, párový klasifikátor „premisa_i premisa_j záver“ | argmax pravdepodobnosti dvojice | kód hotový, **nespustené** (GPU) |
| N1 | Llama 3 8B (4-bit NF4) na lexikálne abstrahovanom texte | indexy z odpovede „VALID i j“ | kód hotový, **nespustené** (GPU) |

**S2 podrobnejšie** (`src/st2_symbolic.py`):
1. Pre každú dvojicu premís (i, j) sa zostaví 3-vetný sylogizmus a rozparsuje parserom zo ST1 (typy A/E/I/O, termíny X/Y/Z).
2. Ak je niektorá dvojica platná → `valid-pair`, platné, premisy = táto dvojica. Pri viacerých platných dvojiciach (remíza) rozhoduje
   pravidlo `struct` (štrukturálna zhoda termínov so záverom). Na dev trafilo správnu dvojicu v 53 z 87 remíz (`jaccard` 50, `lowest` 39).
3. Ak sa rozparsovala aspoň jedna dvojica a žiadna nie je platná → `invalid-sure`, neplatné (pravidlo `any_parsed`;
   prísnejšie `strict` na dev nezachytilo žiadnu chybu a 28 správnych odpovedí poslalo do zálohy).
4. Ak sa nerozparsovala žiadna dvojica alebo záver → `undecided`, rozhoduje **záloha** (fallback).

**Voľba zálohy** bola vopred zapísaná v `configs/st2.yaml` (pred akýmkoľvek výsledkom B2/N1): kandidát s najvyšším priemerným CS
cez 9 dev množín, pri rozdiele < 0,5 bodu vyhrá jednoduchší. Výsledok na dev: **B1 28,23** > invalid 26,05 > valid 24,18.

**B2 a N1 (pripravené, nespustené).** B2 preberá hyperparametre z B2 v ST1 (`finetune_encoder.parser()`: 5 seedov, 5 epôch, lr 2e-5,
dávka 4 × akumulácia 4, max_length 256, rovnaká revízia modelu). **Odchýlka od ST1:** vážená strata, lebo relevantných dvojíc je asi 1 z 10.
N1 používa `run_llama.LlamaRunner` zo ST1 (NF4 4-bit, double quant, rovnaká revízia, greedy); **odchýlka:** `max_new_tokens` = 16 namiesto 8,
lebo odpoveď obsahuje aj dva indexy. Lexikálna abstrakcia (obsahové lemy → umelé slová) zachováva rozhodnutie parsera na všetkých
823 rozparsovateľných príkladoch ST1.

## 4. Výsledky
### 4.1 Dev (priemer 9 syntetických množín)
| Model | ACC | F1 | TCE | CS |
|---|---|---|---|---|
| B0a väčšinová + prekryv | 50,0 | 86,6 | 50,0 | 13,8 |
| B0b plausibility (diagnostika) | 50,5 | 42,8 | 100,0 | 8,3 |
| B0c väčšinová + náhodná dvojica | 50,0 | 34,1 | 50,0 | 8,5 |
| B1 TF-IDF + prekryv | 60,9 | 59,4 | 13,5 | 16,6 |
| S2, záloha invalid | 89,0 | 81,4 | 8,8 | 26,1 |
| S2, záloha valid | 86,9 | 91,7 | 13,8 | 24,2 |
| **S2, záloha B1 (zvolená)** | **89,9** | **89,6** | **8,3** | **28,2** |

### 4.2 Test ST2 (190 príkladov, jednorazovo)
| Model | ACC [95 % CI] | F1 [95 % CI] | TCE | CS [95 % CI] |
|---|---|---|---|---|
| B0a väčšinová + prekryv | 50,0 [42,6; 57,4] | 97,4 | 50,0 | 14,9 |
| B0b plausibility (diagnostika) | 51,6 | 48,4 | 100,0 | 8,9 |
| B0c väčšinová + náhodná dvojica | 50,0 | 35,8 | 50,0 | 8,7 |
| B1 TF-IDF + prekryv | 53,2 [45,8; 60,0] | 93,2 | 45,9 | 15,1 |
| S2, záloha invalid (čistá symbolika) | 92,1 [87,9; 95,8] | 83,2 [75,5; 90,3] | 9,4 | 26,2 [21,8; 32,4] |
| **S2, záloha B1 (hlavný model)** | **92,1 [88,4; 95,8]** | **95,3 [90,9; 98,8]** | **5,9** | **31,9 [25,9; 40,1]** |

* Oficiálny skript spustený nad exportovaným súborom hlavného modelu dáva rovnaké hodnoty (ACC 92,1053; F1 95,2632; TCE 5,9362; CS 31,9006).
* Presnosť buniek hlavného modelu (PV / IV / PI / II): 95,8 / 97,9 / 88,9 / 86,0.
* S2 + B1 vs. B1: McNemar (platnosť) 76 vs. 2 diskordantných párov, p < 10⁻¹⁹ – symbolika je jednoznačne lepšia než lexikálny baseline.
* S2 + B1 vs. S2 + invalid: 12 vs. 12 diskordantných párov, **ACC je rovnaká**. Záloha B1 zlepšila CS cez vyššie F1 (záloha častejšie povie
  „platné“ a vyberie premisy prekryvom) a nižšie TCE, nie cez lepšie rozhodovanie o platnosti.

### 4.3 Rozpad podľa vetiev S2 (test)
| Vetva | n | správne | presnosť |
|---|---|---|---|
| `valid-pair` (parser našiel platnú dvojicu) | 80 | 80 | 100,0 |
| `invalid-sure` (rozparsované, žiadna platná) | 85 | 83 | 97,6 |
| `undecided` (rozhoduje záloha) | 25 | 12 | 48,0 |

Takmer všetky chyby sú v 25 príkladoch, ktoré parser nevedel rozparsovať (21× žiadna dvojica, 4× záver). Tam je B1 na úrovni náhody.
Toto je presne miesto, kde má zmysel neurónová záloha (B2 alebo N1).

## 5. Diskusia a obmedzenia
* **Lexikálny prekryv je na teste prekvapivo silný** (F1 97,4 pri B0a). Rušivé premisy v teste ST2 zjavne menej zdieľajú slová so záverom
  než v našej `hard` dev množine (tam F1 72). F1 baselinov s prekryvom je preto nadhodnotené a výber premís sám osebe nie je ťažký;
  ťažké je rozhodnúť platnosť, čo ukazuje ACC 50–53 % baselinov.
* B1 sa na ST2 neprenáša (ACC 53,2; TCE 45,9) – takmer všetko označí za platné, rovnako ako v teste odolnosti v Etape 1 (kap. 7.3).
* Test má 190 príkladov (~48 na bunku); jedna chyba posunie presnosť bunky o ~2 p. b. Rozdiely CS v rámci ~5 bodov nie sú spoľahlivé (pozri CI).
* B2 a N1 nie sú vyhodnotené, preto nevieme, či by neurónová záloha zlepšila 25 nerozhodnutých príkladov.

## 6. Transparentnosť protokolu
1. **Náhodné nahliadnutie do testu.** Pri vývoji ST2 AI agent počas skúšobného behu (smoke test) omylom vyhodnotil pipeline na skutočnom
   teste ST2. Agent to sám ohlásil, výsledky boli zmazané a nikde sa nepoužili. Pravidlá S2 (remízy, `any_parsed`) a pravidlo voľby zálohy
   boli zvolené iba na dev; pravidlo voľby zálohy bolo zapísané pred dev výsledkami B2/N1. Úplnú nezávislosť od testu to však nezaručuje.
2. **Zmena zoznamu kandidátov zálohy (25. 10. 2026, pred vyhodnotením testu).** Pôvodný zoznam bol {invalid, valid, B1, B2, N1-abstract}.
   B2 (asi 15× viac krokov na epochu než v ST1) a N1 sa do termínu nestihli spustiť, preto sa voľba uzavrela nad {invalid, valid, B1}.
   Vybraná záloha (B1) je rovnaká ako predtým s chýbajúcimi kandidátmi. Zmena je zapísaná s dátumom v `configs/st2.yaml`.
3. Test bol vyhodnotený jedným behom `python -m src.run_st2 --split test --final`; opakovaný beh skript odmietne.
4. B2 a N1 sa na teste vyhodnotia neskôr **samostatne** ako ďalšie modely; voľbu zálohy pre hlavný model už nemenia.

## 7. Ďalší postup (Etapa 2)
* Spustiť N1 (dev + test) a B2 (aspoň 1 seed, cieľovo 5) v Colabe podľa `COLAB_ST2.md`; porovnať ich ako samostatné modely
  a ako zálohu na 25 nerozhodnutých príkladoch.
* Rozšíriť parser o vzory z nerozhodnutých príkladov (iba z dev, nie z testu).
* Vytvoriť ťažšiu dev množinu s rušivými premisami, ktoré zdieľajú termíny so záverom aj štruktúrou, aby F1 nebolo nadhodnotené.

## 8. Reprodukovateľnosť
Seed 42 (B1, B0c), dev seedy 42/43/44; konfigurácia `configs/experiment.yaml` + `configs/st2.yaml`; 63 testov (`python -m pytest -q tests`,
1 test vyžaduje torch); vyhodnotenie bez GPU. Commit s kódom ST2: `7f58b88`.
