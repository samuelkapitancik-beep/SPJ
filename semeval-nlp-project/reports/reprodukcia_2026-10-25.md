# Protokol lokálnej reprodukcie – 25. 10. 2026

Reprodukcia existujúcich CPU experimentov bola úspešná. Všetkých 22 testov prešlo
(16,63 s). Všetkých 13 porovnaných artefaktov sa obsahovo presne zhoduje
s pôvodnými: metriky vrátane intervalov spoľahlivosti a CV mriežky, diagnostika,
predikcie ST1/ST2, parser a rozdelenie train/dev. Pri baseline sa z porovnania
vynechali iba metadáta prostredia a konfigurácia (obsahuje aj novší blok N1).
Nejde o bajtovú zhodu súborov, ale o zhodu načítaných JSON/JSONL hodnôt.

## Overené vstupy a prostredie

- Zdroj: https://github.com/neuro-symbolic-ai/semeval_2026_task_11
- Commit: `6f71ea846770f9418c7d2845ec6c25e59f885bd7`.
- Tréning: 960 záznamov; po odstránení 2 textových prekryvov s testom 958.
- Rozdelenie: 766 train / 192 dev, seed 42.
- Oficiálne súbory v tomto commite: 191 príkladov ST1 a 190 príkladov ST2.
  Tieto počty majú pre tento experiment prednosť pred údajom 192 v priloženom kontexte.
- Oba prevzaté evaluátory sa zhodujú s upstreamom po normalizácii koncov riadkov.
- Windows, Python 3.12.7; pôvodný beh uvádza Linux, Python 3.13.16.
- Primárne knižnice sú z `requirements-lock.txt`; úplný zoznam vrátane
  tranzitívnych závislostí je v `requirements-reproduced.txt`.

## Zopakované výsledky ST1

| Metóda | ACC (%) | TCE | CS |
|---|---:|---:|---:|
| B0a väčšinová trieda | 50,26 | 50,00 | 10,19 |
| B0b vierohodnosť ako platnosť (diagnostika so zlatým štítkom) | 50,26 | 100,00 | 8,95 |
| B1 TF-IDF + logistická regresia | 65,45 | 10,90 | 18,82 |
| S0 symbolická referencia, neparsované → neplatné | 90,58 | 10,42 | 26,37 |

B1 vybrala päťnásobná CV na train: `char_wb`, n-gramy `(2,4)`, `C=100`.
CV ACC = 70,37 %, TCE = 0,5880, CS = 48,11. Dev ACC = 67,71 %.
Symbolický parser pokryl 160/191 testovacích príkladov (83,77 %); na tejto
podmnožine mal presnosť 98,75 %. Táto podmnožinová presnosť nie je výsledok
na celom teste. S0 nie je plný neuro-symbolický model N1 s LLM.

## Rozsah a interpretácia

Nezmenili sa klasifikátory, parser, hyperparametre ani pravidlá výberu modelu.
Opätovné použitie verejného testu slúžilo výhradne na reprodukciu zmrazeného
postupu. Nejde o nový nezávislý experiment ani o dôkaz všeobecnej robustnosti.
Historický prvý beh s výberom podľa dev a ručný audit parsera boli zachované,
ale neboli opätovne vykonané a nie sú započítané do 13 porovnaní.

B2 (doladenie DeBERTa) a LLM časť N1 neboli spustené: pôvodný projekt pre ne
neobsahuje referenčné namerané výsledky. Viacjazyčné podúlohy a nové experimenty
navrhované v priloženej analýze tiež nie sú súčasťou tejto reprodukcie.
Literárne tvrdenia priloženej analýzy sa týmto behom neoverovali.

## Opakovanie a dôkazy

V priečinku projektu spustite:

```powershell
powershell -ExecutionPolicy Bypass -File .\reproduce.ps1
```

Skript kontroluje úspešné ukončenie každého kroku. Pri prvom spustení zálohuje
existujúce výstupy; ďalšie behy referenčnú zálohu nemenia.

- `reproduction/reference/`: pôvodné výstupy a rozdelenia.
- `reproduction/data_manifest.json`: presný zdroj a SHA-256 vstupov.
- `reproduction/logs/`: testy, tréning, diagnostika a parser.
- `reproduction/comparison.json`: výsledky všetkých 13 porovnaní.
- `outputs/`: nanovo vypočítané výsledky a predikcie.

Pridané boli spúšťacie skripty, tento protokol, úplný zoznam verzií a ignorovanie
lokálneho prostredia. README a komentáre boli opravené na 22 testov, výber cez CV
a rozdelenie 766/192. Pôvodná akademická správa zostala zachovaná.
