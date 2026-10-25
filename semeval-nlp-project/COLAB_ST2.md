# ST2 v Colabe: B2 (DeBERTa) a N1 (Llama 3 8B)

ST2 používa rovnaké prostredie a nastavenia ako Samov B2/N1 pre ST1 (`reports/etapa2_navod.md`):
* **N1:** model sa načíta importovaným `run_llama.LlamaRunner` – NF4 4-bit + double quant, compute dtype bf16/fp16
  podľa GPU, revízia `8afb486c…` (ako ST1), greedy, po jednom príklade. Pokračovanie z journalu + kontrola `run.json`.
* **B2:** hyperparametre sa berú z `finetune_encoder.parser()` (5 seedov, 5 epôch, lr 2e-5, dávka 4 × akumulácia 4,
  max_length 256), revízia `8ccc9b6f…`, gradient checkpointing, FP16, checkpoint najlepšej epochy, test samostatne.

V Colabe sa výstupy iba **generujú** (odpovede N1, skóre dvojíc B2) pre 9 dev množín aj pre test ST2; nič sa tam
nehodnotí na teste. Vyhodnotenie beží lokálne bez GPU (`python -m src.run_st2`, test raz cez `--split test --final`).

## 1. Príprava (každá nová session)
GPU runtime (T4 stačí – 4-bit Llama sa zmestí, ako v ST1). V Colab Secrets token `HF_TOKEN` s prístupom k
https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct (ten istý ako pre ST1).
```python
!git clone -q https://github.com/samuelkapitancik-beep/spj /content/spj      # po pushnutí ST2
%cd /content/spj/semeval-nlp-project
!pip install -q -r requirements.txt -r requirements-neural.txt               # rovnaké verzie ako ST1 (transformers 4.57.1)
import os
from google.colab import userdata, drive
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
drive.mount("/content/drive")
OUT = "/content/drive/MyDrive/SPJ_SemEval2026/st2_v1"     # nový adresár, nemieša sa so Samovým neural_v*
!python -m pytest -q tests                                # všetky testy (ST1 + ST2)
```
Dev množiny ST2 (`data/processed/st2/`) sú v repe, nič sa negeneruje.

## 2. B2 – DeBERTa pre ST2
```python
!python -m src.st2_encoder --output {OUT}/st2_b2               # dev: 5 seedov -> seed_N/ na Drive
!python -m src.st2_encoder --output {OUT}/st2_b2 --mode test   # test: iba skóre dvojíc zo zmrazených checkpointov
```
* Každý hotový seed (`training.json` + `dev_pair_scores.json`) sa pri ďalšom spustení preskočí. Prerušený seed sa
  **nemaže** – odloží sa ako `seed_N_prerusene_<čas>` a ten seed sa začne odznova (rovnako ako v ST1 sa tréning
  z polovice neobnovuje, checkpoint nemá stav optimalizátora).
* Čas: na epochu je 11 490 tréningových dvojíc oproti 766 príkladom v ST1 (~15× viac krokov) a po tréningu sa
  skóruje 25 583 dvojíc dev množín. Odhad podľa Samovho behu B2 v ST1 × ~15; reálny čas uvidíš po prvom seede.
* Skúška bez dopadu na výsledky: `--seeds 0 --epochs 1 --limit 20` (výstup ide do `{OUT}/st2_b2_skuska`).

## 3. N1 – Llama 3 8B pre ST2
```python
!python -m src.st2_llm --output {OUT}/st2_llama                 # dev: 9 množín × (abstract, original)
!python -m src.st2_llm --output {OUT}/st2_llama --split test    # test ST2: iba odpovede
```
* Po odpojení stačí ten istý príkaz – pokračuje z `responses.jsonl`, ale iba ak sa zhoduje model, revízia, prompt,
  dáta aj prostredie (inak chyba a nový `--output`, rovnako ako v ST1).
* Nejednoznačná odpoveď ostáva `null` (ako v ST1): model alebo hybrid s abstenciou nemá oficiálne metriky
  a z voľby zálohy pre S2 sa vylúči.
* Skúška: `--limit 3` (výstup do `{OUT}/st2_llama_skuska`).
* Odpovede na teste si nepozeraj.

## Miesto
| čo | kde | veľkosť |
|---|---|---|
| B2 checkpoint, 1 seed | Drive (`seed_N/checkpoint/`) | odhad ~0,74 GB (184 M parametrov × 4 B, fp32) → 5 seedov ~3,7 GB |
| B2 skóre dvojíc, 1 seed | Drive | ~1,9 MB dev + ~0,2 MB test (odmerané na smoke teste) |
| N1 odpovede, všetky behy | Drive | pod 1 MB (smoke test s dlhšími odpoveďami 0,8 MB) |
| Llama 3 8B, DeBERTa | disk Colab VM (cache HF), nie Drive | ~16 GB a rádovo stovky MB (podľa HF, neoverené) |

Checkpointy na stiahnutie netreba; Samov `.gitignore` ich v `outputs/neural/**/checkpoint/` aj tak vynecháva.

## Skrátená verzia (keď je málo času)
Najprv N1 iba v reprezentácii `abstract` (rádovo desiatky minút na T4), potom B2 s jedným seedom:
```python
!python -m src.st2_llm --output {OUT}/st2_llama --representations abstract
!python -m src.st2_llm --output {OUT}/st2_llama --representations abstract --split test
!python -m src.st2_encoder --output {OUT}/st2_b2 --seeds 0
!python -m src.st2_encoder --output {OUT}/st2_b2 --seeds 0 --mode test
```
Test hlavného modelu S2 už bol vyhodnotený (25. 10. 2026, `outputs/st2_test_run_log.json`), preto sa B2 a N1 na teste
vyhodnotia samostatne a voľbu zálohy S2 nemenia (zdôvodnenie v `reports/st2_sprava.md`, kap. 6).

## 4. Stiahnuť späť a vyhodnotiť lokálne
```python
!cd {OUT} && zip -qr /content/st2_vysledky.zip st2_b2 st2_llama -x "*/checkpoint/*"
from google.colab import files; files.download("/content/st2_vysledky.zip")
```
Lokálne rozbaľ do `semeval-nlp-project/outputs/neural/` (vzniknú `outputs/neural/st2_b2/` a `outputs/neural/st2_llama/`) a:
```bash
python -m src.run_st2      # dev: B2, N1 abstract/original a hybridy S2 + B2 / N1; na konci voľba zálohy S2
```
Ak voľba vypíše `NESEDÍ`, prepíš `st2.symbolic.fallback` v `configs/st2.yaml` (s dátumom a dôvodom) a pusti dev znova.
Až pri `SEDÍ` a bez chýbajúcich kandidátov raz: `python -m src.run_st2 --split test --final`.
