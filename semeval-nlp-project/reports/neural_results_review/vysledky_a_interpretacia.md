# Overené výsledky SemEval-2026 Task 11 / ST1

Analýza dodaného archívu z 25. 10. 2026. Nejde o nové spustenie modelov.

## Hlavné zistenia

- DeBERTa: test ACC **76,54 ± 3,06 %** (priemer ± výberová SD piatich seedov), oproti B1 +11,10 p. b.
- Llama canonical: **79,58 %**, oproti original +12,57 p. b.; CS však zostáva 18,27 oproti B1 18,82.
- Llama abstract: **52,36 %**, 185/191 odpovedí „platný“; hypotéza o automatickom prínose abstrakcie sa nepotvrdila.
- S0: **90,58 %** na celom teste, najvyššia ACC v porovnaní; ide o existujúcu symbolickú referenciu.

## Testové výsledky (191 príkladov)

| Model | ACC (%) | TCE | CS |
|---|---:|---:|---:|
| B1 TF-IDF + LR | 65,45 | 10,90 | 18,82 |
| B2 priemer ± SD | 76,54 ± 3,06 | 9,37 ± 4,77 | 24,13 ± 3,81 |
| S0 | 90,58 | 10,42 | 26,37 |
| Llama original | 67,02 | 43,62 | 13,97 |
| Llama abstract | 52,36 | 48,96 | 10,66 |
| Llama canonical | 79,58 | 27,66 | 18,27 |

ACC: vyššie je lepšie. TCE: nižšie je lepšie. CS = ACC / (1 + ln(1 + TCE)). TCE kombinuje rozdiely podľa platnosti aj vierohodnosti; nie je čistou mierou belief bias.

## B2: jednotlivé seedy

| Seed | Vybraná epocha | Dev ACC | Test ACC | Test TCE | Test CS |
|---|---:|---:|---:|---:|---:|
| 0 | 4 | 82,29 | 79,58 | 7,58 | 25,27 |
| 1 | 3 | 78,65 | 72,77 | 3,12 | 30,11 |
| 2 | 5 | 77,08 | 74,35 | 11,46 | 21,11 |
| 3 | 4 | 77,60 | 76,44 | 8,67 | 23,39 |
| 4 | 5 | 79,17 | 79,58 | 16,00 | 20,76 |

Checkpoint sa vyberal podľa dev ACC, nie podľa testového skóre. Seed 0 má najvyššiu dev ACC. Päť seedov nie je päť nezávislých testovacích množín. SD medzi seedmi nie je bootstrap CI.

## Llama: rozklad podľa platnosti a vierohodnosti

| Vstup | Platné/vierohodné | Platné/nevierohodné | Neplatné/vierohodné | Neplatné/nevierohodné |
|---|---:|---:|---:|---:|
| original | 100,00 | 87,50 | 12,77 | 66,67 |
| abstract | 100,00 | 97,92 | 8,51 | 2,08 |
| canonical | 100,00 | 100,00 | 44,68 | 72,92 |

Skupiny majú 48, 48, 47 a 48 príkladov. Llama canonical správne klasifikuje všetkých 96 platných argumentov, ale iba 56/95 neplatných. Abstract správne rozpozná len 5/95 neplatných.
Parser pokryl 160/191 testových vstupov. Na týchto vstupoch má Llama original ACC 67,50 %, abstract 50,00 % a canonical 82,50 %. Na ostatných 31 príkladoch dostali všetky varianty pôvodný text a majú totožné odpovede (ACC 64,52 %). Nejde teda o tri odlišné vstupy na celom teste.

## Validačné výsledky

DeBERTa: 78,96 ± 2,04 %. Llama original: 64,06 %, abstract: 53,13 %, canonical: 66,67 %. Vyššia testová ACC kanonickej Llamy oproti dev sama osebe nie je dôkazom úniku dát.

## Audit a štatistika

Prešlo 1308 kontrol; bez zistených nezrovnalostí. Overili sa úplnosť ID, hashe dát, metriky, bootstrap CI, confusion matrix, výber epochy a zhoda exportov. Všetky Llama odpovede sú interpretovateľné.
Exploračné presné McNemarove testy; Holmova korekcia sa vzťahuje na nasledujúce tri porovnania:

| Porovnanie A vs B | Len A správne | Len B správne | p | p po Holmovi |
|---|---:|---:|---:|---:|
| Llama_canonical vs Llama_original | 28 | 4 | 1.93012e-05 | 3.86024e-05 |
| Llama_abstract vs Llama_original | 5 | 33 | 4.25596e-06 | 1.27679e-05 |
| B2_0 vs B1 | 39 | 12 | 0.000198039 | 0.000198039 |

B2 seed 0 je tu porovnaný s B1, pretože má najvyššiu dev ACC; nevyberáme seed podľa testu.

## Hranice interpretácie

- Archív neobsahuje váhy, kompletné logy, snapshot kódu ani celý pip freeze; nezávislá GPU reprodukcia neprebehla.
- Absenciu ladenia podľa testu nemožno z exportov dokázať. Verejný test už bol použitý pri baseline.
- Llama používa NF4 double quantization, prompt v1 a max. 8 tokenov; nejde o všeobecný výsledok všetkých Llama modelov.
- B1 používa finálny refit na train+dev (958), B2 trénuje na train (766) a vyberá epochu cez dev (192).
- B1 a S0 sú pôvodné lokálne referencie a sú aj v dodanej comparison.json; nie sú novými neurónovými behmi.
- Archív neobsahuje nové výsledky ST2–ST4.

## Text na ústne prezentovanie

> Porovnali sme textový baseline, doladenú DeBERTu, symbolický systém a tri reprezentácie vstupu Llamy. DeBERTa zlepšila priemernú presnosť oproti baseline o približne jedenásť percentuálnych bodov. Pri Llame pomohol jednotný kanonický zápis, ale jednoduché nahradenie pojmov symbolmi viedlo k takmer neustálej odpovedi „platný“. Výsledky preto ukazujú, že odstránenie obsahu samo osebe nestačí. Najvyššiu presnosť na tomto teste dosiahla symbolická referencia; jej obmedzením zostáva pokrytie parsera.

## Zdroj a reprodukovateľnosť auditu

Vstup: `C:\Users\samoo\Downloads\semeval_neural_results.zip`.
SHA-256: `35e36847cc5e61f4e4640238f183770c5cfca4af611bf4365868f72d4710f743`.
Audit: `audit_neural_results.py`; podrobné metriky a všetky kontroly: `audit.json`.
