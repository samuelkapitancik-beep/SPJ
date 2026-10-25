# Etapa 2 — B2 a Llama v Google Colab

Experimenty B2 a Llama 3 8B boli vykonané v Colabe a ich exporty sú v
`outputs/neural/`. Metriky boli následne lokálne prepočítané z predikcií;
nejde o nezávislé opakovanie GPU tréningu. Aktuálne tabuľky a interpretáciu obsahuje
hlavné README. Tento dokument opisuje postup na nové spustenie.

## Spustenie

1. Otvorte https://colab.research.google.com/ a cez Upload notebook nahrajte
   `colab/SemEval_B2_Llama.ipynb`.
2. Zvoľte GPU runtime (T4 alebo výkonnejší).
3. Na Hugging Face požiadajte o prístup k
   https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct, ak ho ešte nemáte.
   Do Colab Secrets pridajte read token `HF_TOKEN` a povoľte notebooku prístup.
4. Spustite prvú bunku a nahrajte `colab/semeval_colab.zip`. Pripojte vlastný Drive.
5. Postupne spustite inštaláciu, testy, kontrolu prístupu a tréning/inferenciu na dev.
6. Po zmrazení postupu nastavte `RUN_FINAL_TEST=True` a spustite testovaciu bunku.
7. Posledná bunka stiahne archív nameraných výsledkov. Checkpointy ostanú na Drive.

Výstupy sa ukladajú do `MyDrive/SPJ_SemEval2026/neural_v2`. Pre iný experiment
použite nový názov adresára, aby sa nezmiešali konfigurácie. Notebook môže trvať
hodiny a dostupnosť GPU závisí od Colabu; runtime tu zatiaľ nebol spustený.

## B2: microsoft/deberta-v3-base

- Plné doladenie klasifikátora na 766 príkladoch; validačných príkladov je 192.
- Päť seedov 0–4, päť epoch, AdamW, learning rate 2e-5, weight decay 0,01.
- Dávka 4 × akumulácia 4 = efektívna dávka 16; gradient checkpointing a FP16 na GPU.
- Checkpointing explicitne používa `use_reentrant=False` kvôli zdieľaným
  relatívnym pozičným reprezentáciám DeBERTy. Nepoužíva sa `retain_graph=True`.
  Opravený notebook používa nový adresár `neural_v2`, aby nekolidoval s prerušeným
  behom v `neural_v1`. Pôvodné súbory sa nemažú.
- Maximálna dĺžka 256 tokenov, dynamické paddingovanie, dlhšie vstupy sa skrátia.
- Najlepšia epocha podľa dev ACC; pri zhode skoršia. Tento výber zachováva pôvodný
  návrh B2. TCE/CS sa tiež reportujú, ale nevyberajú epochu.
- Najlepší checkpoint sa uloží na disk; finálny test ho načíta bez refitu.
  To sa líši od B1, ktorý po výbere hyperparametrov refituje na train+dev.
- Ukladajú sa predikcie, história epoch, checkpoint, verzie knižníc, revízia modelu,
  hashe datasetov, ACC/TCE/CS, bootstrap intervaly a confusion matrix.
- Hotové seedy možno preskočiť. Prerušený tréning sa automaticky neobnovuje:
  checkpoint neobsahuje optimizer/scheduler/RNG stav. Nový pokus patrí do nového adresára.

## N1: meta-llama/Meta-Llama-3-8B-Instruct

- Zero-shot inferencia bez doladenia, rovnaký fixný prompt v1 pre všetky varianty.
- Tri vstupy: original, abstract a canonical; nespracovaný argument používa original.
- Greedy decoding, max. 8 nových tokenov; ukončenie EOS alebo Llama end-of-turn.
- NF4 4-bit + double quantization; toto je súčasť experimentálneho nastavenia
  a výsledok nemožno automaticky stotožniť s modelom v plnej presnosti.
- Model rozhoduje o platnosti. Nie je to LLM formalizácia nasledovaná symbolickým solverom.
- Surové odpovede sa ukladajú po jednom príklade; opakovaný beh pokračuje z journalu,
  iba ak sa zhoduje model, prompt, dáta, nastavenie a prostredie.
- Odpoveď bez jednoznačného VALID/INVALID zostáva `null`, nikdy sa automaticky
  nepovažuje za INVALID. V diagnostickej accuracy je chybou. Oficiálne metriky sa
  v takom behu nevytvoria: oficiálny evaluator by null vynechal z menovateľa.
  Oficiálny submission.json vznikne iba pri 100 % pokrytí binárnymi odpoveďami.

## Lokálne/CLI príkazy po inštalácii GPU prostredia

```bash
pip install -r requirements.txt -r requirements-neural.txt
python -m src.finetune_encoder --mode train
python -m src.run_llama --split dev
# Až po uzavretí vývoja:
python -m src.finetune_encoder --mode test
python -m src.run_llama --split test_st1
python -m src.compare_neural
```

`--revision` umožňuje uviesť presný HF commit. Notebook revízie zmrazí pred prvým
behom a použije ich aj pre test. `--output` určuje samostatný adresár experimentu.
Žiadne testovacie výsledky ani jednotlivé chyby sa nemajú používať na ladenie parsera,
promptu alebo parametrov. Verejný test bol už použitý pri reprodukcii baseline.

## Súbory a opätovné vytvorenie balíka

- `src/finetune_encoder.py`: tréning a samostatné testovanie B2.
- `src/run_llama.py`: načítanie Llamy a inferencia troch reprezentácií.
- `src/neural_common.py`: proveniencia, pokrytie odpovedí a spoločné metriky.
- `src/compare_neural.py`: tabuľka B0/B1/S0/B2/N1 a priemer/SD cez dostupné seedy B2.
- `build_colab_bundle.py`: znovu vytvorí notebook a ZIP po úprave zdrojov.

Balík obsahuje potrebné dáta, testy a evaluátory, nie `.venv`, tokeny ani váhy modelov.

## Primárne zdroje

- DeBERTa: https://huggingface.co/microsoft/deberta-v3-base
- Llama a prístupové podmienky: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
- Kvantizácia: https://huggingface.co/docs/transformers/quantization/bitsandbytes
