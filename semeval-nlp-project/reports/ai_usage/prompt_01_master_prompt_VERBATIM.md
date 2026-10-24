# PROMPT: Vypracovanie NLP projektu podľa univerzitného zadania a SemEval-2026 Task 11

## 1. Tvoja rola

Správaj sa ako skúsený NLP Researcher, Machine Learning Engineer a výskumný konzultant. Máš skúsenosti s jazykovými modelmi, logickým uvažovaním, klasifikáciou textu, experimentálnym výskumom a reprodukovateľnými ML projektmi.

Pomôž mi navrhnúť, implementovať, experimentálne vyhodnotiť a zdokumentovať univerzitný projekt z predmetu Spracovanie prirodzeného jazyka.

Projekt bude vychádzať z úlohy:

**SemEval-2026 Task 11: Disentangling Content and Formal Reasoning in Large Language Models.**

Oficiálna stránka:
https://sites.google.com/view/semeval-2026-task-11

Oficiálny repozitár:
https://github.com/neuro-symbolic-ai/semeval_2026_task_11

Zároveň dostaneš PDF s názvom `Projektove_zadanie.pdf`, ktoré obsahuje požiadavky môjho vyučujúceho. Tento dokument je záväzný pre štruktúru a hodnotenie univerzitného projektu.

Tvojou úlohou nie je iba napísať návrh riešenia. Chcem vytvoriť funkčný, reprodukovateľný a metodologicky správny projekt, ktorý spĺňa univerzitné požiadavky a má aj vlastný výskumný prínos.

## 2. Hierarchia požiadaviek

Pri rozhodovaní postupuj podľa tejto priority:

1. Požiadavky uvedené v univerzitnom PDF.
2. Oficiálna špecifikácia vybranej podúlohy SemEval-2026 Task 11.
3. Oficiálne datasety, repozitár a hodnotiace skripty.
4. Relevantná vedecká literatúra.
5. Vlastné návrhy a experimentálne hypotézy.

Ak sa požiadavky líšia, identifikuj rozdiel a vysvetli jeho dôsledky. Nepredpokladaj, že pravidlá súťaže automaticky nahrádzajú požiadavky univerzitného projektu.

Nevymýšľaj pravidlá, datasety, metriky ani výsledky. Každé dôležité tvrdenie o oficiálnej úlohe over v dostupných zdrojoch.

## 3. Povinné požiadavky univerzitného projektu

Riešenie musí pokrývať obe etapy uvedené v zadaní.

### Etapa 1: Návrh projektu a baseline riešenie

Priprav:

- Jasnú formuláciu problému, motiváciu a ciele.
- Prehľad relevantnej vedeckej literatúry.
- Analýzu dátových zdrojov a existujúcich riešení.
- Zdôvodnenie vybraných metrík a experimentálnej metodiky.
- Funkčný baseline model.
- Reprodukovateľnú implementáciu baseline.
- Kvalitnú technickú správu so zdrojmi a citáciami.
- Dokumentáciu použitia LLM a AI agentov.

### Etapa 2: Vlastné riešenie a experimentálne vyhodnotenie

Priprav:

- Vlastný výskumný prínos alebo podstatné, odôvodnené rozšírenie baseline.
- Jasne definované výskumné otázky a hypotézy.
- Experimentálny plán.
- Implementáciu navrhnutého riešenia.
- Porovnanie s baseline.
- Vyhodnotenie pomocou vhodných metrík.
- Analýzu chýb a interpretáciu výsledkov.
- Diskusiu obmedzení.
- Reprodukovateľný finálny projekt.

Vyučujúci hodnotí aj primeranú náročnosť, technickú správnosť, kvalitu experimentov a schopnosť interpretovať výsledky.

Preto nenavrhuj iba jednoduchý klasifikátor bez experimentálneho porovnania. Zároveň sa vyhni zbytočne komplikovanému riešeniu, ktorého prínos nebude možné spoľahlivo overiť.

## 4. Najskôr analyzuj zadanie SemEval

Preskúmaj oficiálnu stránku, GitHub repozitár, dostupné datasety, dokumentáciu a hodnotiace skripty.

Identifikuj:

- Výskumný problém a jeho motiváciu.
- Definíciu všetkých štyroch podúloh.
- Formát vstupných dát a očakávaných výstupov.
- Dostupné tréningové a validačné dáta.
- Dostupnosť testovacích štítkov.
- Oficiálne hodnotiace metriky.
- Existujúce baseline riešenia.
- Obmedzenia jednotlivých podúloh.
- Aktuálnu dostupnosť dát a možnosť vykonávať vlastné experimenty.

Vysvetli základné pojmy, najmä logickú platnosť sylogizmu, vierohodnosť obsahu a content effect.

Ak niektoré dáta alebo hodnotiace nástroje nie sú dostupné, jasne to uveď a navrhni realizovateľnú alternatívu, ktorá zachová metodologickú korektnosť.

## 5. Výber konkrétnej podúlohy

Porovnaj štyri podúlohy podľa:

- Náročnosti implementácie.
- Dostupnosti dát.
- Potrebného hardvéru.
- Možností vytvoriť vlastný prínos.
- Vhodnosti pre univerzitný projekt.
- Možnosti správneho experimentálneho vyhodnotenia.

Na základe porovnania odporuč jednu hlavnú podúlohu.

Preferuj riešenie, ktoré umožňuje kvalitný baseline, niekoľko zmysluplných experimentov a dôkladnú analýzu výsledkov v rámci realistických časových a výpočtových možností študenta.

Pred implementáciou mi predstav odporúčanú podúlohu, jej výhody a riziká. Ak výber zásadne mení ďalší postup, počkaj na moje potvrdenie.

## 6. Výskumný návrh

Po výbere podúlohy priprav kompletný výskumný návrh.

Musí obsahovať:

### 6.1 Výskumný problém

Definuj, čo presne budeme skúmať a prečo je to dôležité pre spracovanie prirodzeného jazyka.

### 6.2 Výskumné otázky

Formuluj 2 až 4 konkrétne výskumné otázky, na ktoré bude možné odpovedať experimentálne.

Príklad smerovania:

- Do akej miery jazykový model rozlišuje formálnu logickú platnosť od vierohodnosti obsahu?
- Zlepší explicitná reprezentácia logickej štruktúry výsledky oproti štandardnému textovému klasifikátoru?
- Ako sa mení výkon modelu pri odlišnej vierohodnosti argumentov?
- Ak sa skúma viac jazykov, ako dobre sa model generalizuje mimo angličtiny?

Tieto otázky ber ako inšpiráciu. Vyber ich podľa skutočných vlastností datasetu a zvolenej podúlohy.

### 6.3 Hypotézy

Ku každej výskumnej otázke formuluj testovateľnú hypotézu a vysvetli, aký výsledok by ju podporoval alebo vyvracal.

### 6.4 Očakávaný prínos

Presne rozlišuj medzi:

- Základnou implementáciou.
- Štandardnými experimentmi.
- Vlastným prínosom projektu.
- Potenciálnym rozšírením.

Za vlastný prínos nepovažuj automaticky samotné použitie predtrénovaného modelu alebo jeho fine-tuning.

## 7. Baseline a vlastné riešenie

Navrhni minimálne dva porovnateľné prístupy.

### Baseline

Vyber jednoduchý, ale metodologicky korektný model, napríklad TF-IDF s logistickou regresiou, ak je vhodný pre danú úlohu.

Alternatívne môžeš použiť jednoduchý predtrénovaný model, ak to lepšie zodpovedá oficiálnej metodike.

Vysvetli jeho predpoklady, výhody, slabiny a úlohu v experimentoch.

### Vlastné riešenie

Navrhni rozšírenie, ktoré má jasný dôvod a overiteľný prínos.

Zváž napríklad:

- Explicitnú extrakciu logickej štruktúry.
- Oddelenú reprezentáciu premís a záveru.
- Analýzu vplyvu plausibility.
- Symbolické overovanie logickej platnosti.
- Kombináciu neurónového modelu so symbolickými pravidlami.
- Cielené experimenty zamerané na odolnosť voči obsahovým skresleniam.

Vyber iba také rozšírenie, ktoré je realizovateľné a vhodné pre skutočné dáta.

Musíš vysvetliť, prečo by navrhované riešenie mohlo fungovať, ako ho implementujeme a akým experimentom overíme jeho prínos.

## 8. Implementácia projektu

Vytvor modulárny Python projekt. Prispôsob jeho štruktúru skutočným potrebám vybranej podúlohy.

Odporúčaná štruktúra:

```text
semeval-nlp-project/
├── README.md
├── requirements.txt
├── configs/
│   └── experiment.yaml
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── exploratory_analysis.ipynb
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── baselines.py
│   ├── models.py
│   ├── train.py
│   ├── predict.py
│   ├── evaluate.py
│   └── utils.py
├── tests/
├── outputs/
└── reports/
```

Implementuj:

1. Načítanie a validáciu dát.
2. Exploratory Data Analysis.
3. Kontrolu duplicít a dátových únikov.
4. Predspracovanie vhodné pre logické texty.
5. Baseline model.
6. Vlastný model alebo rozšírenie.
7. Tréning a validáciu.
8. Výpočet metrík.
9. Export predikcií.
10. Analýzu chýb.
11. Ukladanie konfigurácií a výsledkov.
12. Testy kritických častí pipeline.

Nepíš celý projekt do jedného súboru. Používaj funkcie a moduly s jasne definovanou zodpovednosťou.

Ak môžeš pracovať so súborovým systémom, vytváraj skutočné súbory a overuj ich funkčnosť. Ak nemáš možnosť súbory vytvárať alebo kód spúšťať, túto skutočnosť oznám.

## 9. Experimentálny plán

Pred tréningom priprav tabuľku experimentov.

Pri každom experimente uveď:

- ID experimentu.
- Výskumnú otázku.
- Použitý model.
- Rozdiel oproti baseline.
- Použité dáta.
- Konfiguráciu.
- Hodnotiace metriky.
- Očakávaný informačný prínos.

Experimenty majú umožniť zistiť, prečo riešenie funguje, nie iba to, či dosiahne vysoké skóre.

Použi rovnaké rozdelenie dát a porovnateľné podmienky pri modeloch, ktoré chceš férovo porovnať.

Zabráň úniku informácií z testovacej množiny. Hyperparametre vyberaj podľa tréningových a validačných dát, nie podľa výsledkov na skrytom teste.

Ak je to vhodné, vykonaj ablačný experiment, ktorý ukáže prínos jednotlivých komponentov navrhnutého riešenia.

## 10. Metriky a vyhodnotenie

Over oficiálne metriky vybranej podúlohy v hodnotiacich skriptoch.

Podľa potreby použi:

- Accuracy.
- Precision, Recall a F1-score.
- Confusion matrix.
- Total Content Effect.
- F1-score pre výber relevantných premís.
- Oficiálnu primárnu ranking metriku.

Nevyberaj metriky mechanicky. Zdôvodni ich vzhľadom na cieľ úlohy.

Porovnaj výsledky baseline a vlastného riešenia v jednej prehľadnej tabuľke.

Vysvetli:
- Čo jednotlivé hodnoty znamenajú.
- Ktorý model je lepší podľa oficiálneho kritéria.
- Či zlepšenie presahuje náhodné výkyvy.
- Aké kompromisy vznikli.
- Čo sa z výsledkov dá a nedá vyvodiť.

Nevymýšľaj namerané výsledky. Ak experiment nebol spustený, uveď to a priprav kód na jeho vyhodnotenie.

## 11. Analýza chýb

Vytvor systematickú analýzu nesprávnych predikcií.

Navrhni kategórie chýb na základe skutočných dát, napríklad:

- Nesprávne pochopenie negácie.
- Nesprávne pochopenie kvantifikátorov.
- Zámena plausibility a validity.
- Zlyhanie pri určitej logickej schéme.
- Citlivosť na rušivé premisy.
- Jazykovo špecifické problémy.

Kategórie prispôsob skutočným chybám. Ak je analýza vykonávaná manuálne, zdokumentuj spôsob výberu vzorky a pravidlá klasifikácie chýb.

Na základe analýzy navrhni jedno konkrétne zlepšenie a experimentálne over, či skutočne funguje.

## 12. Technická správa

Priprav správu zodpovedajúcu požiadavkám univerzitného zadania.

Odporúčaná štruktúra:

1. Úvod a motivácia.
2. Formulácia problému a ciele.
3. Súvisiace práce.
4. Dataset a jeho analýza.
5. Metodika.
6. Baseline riešenie.
7. Navrhnuté rozšírenie.
8. Experimentálne nastavenie.
9. Výsledky.
10. Analýza chýb.
11. Diskusia a obmedzenia.
12. Záver.
13. Zoznam literatúry.

Správu píš v slovenčine, pokiaľ si nevyžiadam angličtinu alebo oficiálne pravidlá nevyžadujú iný jazyk.

Používaj odborný, ale zrozumiteľný jazyk. Každé tvrdenie o existujúcom výskume podlož overiteľnou citáciou.

Nevymýšľaj výsledky ani bibliografické údaje. Jasne odlišuj vlastné zistenia od tvrdení prevzatých z literatúry.

## 13. Povinná príloha o použití AI

Univerzitné zadanie vyžaduje samostatnú prílohu dokumentujúcu použitie LLM a AI agentov.

Preto počas celého projektu priebežne eviduj:

- Názov použitého nástroja alebo modelu.
- Dátum a účel použitia.
- Presné použité prompty.
- Relevantné výstupy AI.
- Ktoré návrhy boli prijaté a ktoré zamietnuté.
- Aké úpravy vykonal autorský tím.
- Ako boli výstupy overené.
- Ktoré časti kódu, analýzy alebo dokumentácie vznikli s pomocou AI.

Na konci priprav samostatnú prílohu, ktorú možno pripojiť k odovzdávanej správe.

Neuvádzaj fiktívne konverzácie ani nevytváraj spätné záznamy o činnostiach, ktoré sa nestali. Ak nemáš kompletnú históriu použitia AI, označ chýbajúce informácie a požiadaj ma o ich doplnenie.

## 14. Reprodukovateľnosť a finálne odovzdanie

Priprav README s jasnými príkazmi na:

- Inštaláciu závislostí.
- Stiahnutie alebo prípravu dát.
- Spustenie analýzy dát.
- Tréning baseline.
- Tréning vlastného modelu.
- Evaluáciu.
- Generovanie predikcií.
- Reprodukciu hlavných výsledkov.

Zaznamenávaj random seeds, verzie knižníc, konfigurácie a identifikáciu použitých modelov.

Spusti dostupné testy a oprav chyby, ktoré sa dajú reprodukovať.

Ak chceš navrhovať bonusové rozšírenia, vyber ich podľa kritérií z univerzitného PDF. Neoznačuj jednoduchý fine-tuning, vytvorenie GUI alebo základnú ablačnú tabuľku automaticky za bonusový prínos.

## 15. Spôsob spolupráce

Pracuj v šiestich etapách:

**Etapa A – Audit:** analyzuj univerzitné zadanie a oficiálne zdroje SemEval.

**Etapa B – Návrh:** vyber podúlohu, sformuluj výskumné otázky, navrhni baseline a vlastné riešenie.

**Etapa C – Dáta:** priprav dataset, EDA a kontrolu dátových únikov.

**Etapa D – Implementácia:** vytvor funkčný kód a testy.

**Etapa E – Experimenty:** vykonaj vyhodnotenie, porovnaj modely a analyzuj chyby.

**Etapa F – Dokumentácia:** priprav správu, README a prílohu o použití AI.

Na konci každej etapy uveď:
- Čo je hotové.
- Čo bolo skutočne overené.
- Aké výsledky máme.
- Čo ešte treba urobiť.
- Aké rozhodnutie alebo vstup potrebuješ odo mňa.

Nevytváraj všetko naraz bez kontroly kvality. Pri zásadných rozhodnutiach mi predstav možnosti a odporúčanie.

## 16. Začni teraz

Začni Etapou A.

Prečítaj si priložené PDF univerzitného zadania a analyzuj oficiálnu dokumentáciu SemEval-2026 Task 11.

Vytvor:
1. Zoznam povinných požiadaviek univerzitného projektu.
2. Prehľad štyroch podúloh SemEval.
3. Porovnanie ich náročnosti a realizovateľnosti.
4. Odporúčanie jednej podúlohy.
5. Návrh baseline modelu a vlastného experimentálneho prínosu.
6. Predbežný plán implementácie a hodnotenia.
7. Zoznam otvorených otázok, ktoré treba vyriešiť pred začatím implementácie.

**Zatiaľ nezačni trénovať model ani nevytváraj vymyslené výsledky. Najprv zabezpeč, aby navrhované riešenie spĺňalo univerzitné zadanie a bolo realizovateľné na dostupných dátach a hardvéri.**