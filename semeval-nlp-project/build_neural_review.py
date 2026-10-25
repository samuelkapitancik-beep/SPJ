"""Produce a presentation-format PDF and an editable narrative from audited results."""
from pathlib import Path
import json
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor, white
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'reports/neural_results_review'
A=json.loads((OUT/'audit.json').read_text(encoding='utf-8'))
PDF=ROOT/'output/pdf/SemEval_ST1_vysledky.pdf'
PDF.parent.mkdir(parents=True,exist_ok=True)
pdfmetrics.registerFont(TTFont('Arial','C:/Windows/Fonts/arial.ttf'))
pdfmetrics.registerFont(TTFont('ArialB','C:/Windows/Fonts/arialbd.ttf'))
pdfmetrics.registerFontFamily('Arial',normal='Arial',bold='ArialB',italic='Arial',boldItalic='ArialB')
W,H=960,540
c=canvas.Canvas(str(PDF),pagesize=(W,H))
c.setTitle('SemEval-2026 Task 11 - overené výsledky ST1')
c.setAuthor('Analýza dodaných experimentálnych výsledkov')
NAVY='#152c43'; TEAL='#168773'; MUTED='#55677a'; LIGHT='#f1f5f8'; PURPLE='#6456a1'
page_no=0
def txt(s,x,y,size=14,color=NAVY,bold=False):
 c.setFont('ArialB' if bold else 'Arial',size);c.setFillColor(HexColor(color));c.drawString(x,y,s)
def para(s,x,y,width,size=15,color=NAVY):
 style=ParagraphStyle('body',fontName='Arial',fontSize=size,leading=size*1.45,textColor=HexColor(color))
 p=Paragraph(s,style); _,h=p.wrap(width,500);p.drawOn(c,x,y-h);return y-h
def start(kicker,title,sub):
 global page_no
 page_no+=1;c.setFillColor(white);c.rect(0,0,W,H,fill=1,stroke=0)
 c.setFillColor(HexColor(TEAL));c.rect(38,494,38,4,fill=1,stroke=0)
 txt(kicker.upper(),88,491,10,MUTED,True);txt(title,38,449,28,bold=True)
 para(sub,38,427,884,12,MUTED)
def end():
 c.setStrokeColor(HexColor('#dce4eb'));c.line(38,35,922,35)
 txt('SemEval-2026 Task 11 | ST1 | audit dodaných výsledkov, 25. 10. 2026',38,19,9,MUTED)
 txt(f'{page_no:02d} / 06',875,19,9,MUTED);c.showPage()
def card(x,y,w,value,label,detail,color=TEAL):
 c.setFillColor(HexColor(LIGHT));c.roundRect(x,y,w,132,9,fill=1,stroke=0)
 txt(value,x+18,y+84,29,color,True);txt(label,x+18,y+55,13,bold=True)
 para(detail,x+18,y+40,w-36,11,MUTED)
def table(headers,rows,x,y,widths,row_h=30):
 total=sum(widths)
 c.setFillColor(HexColor(NAVY));c.rect(x,y-row_h,total,row_h,fill=1,stroke=0)
 pos=x
 for h,w in zip(headers,widths):txt(h,pos+10,y-row_h+10,11,'#ffffff',True);pos+=w
 y-=row_h
 for i,row in enumerate(rows):
  c.setFillColor(HexColor(LIGHT if i%2==0 else '#ffffff'));c.rect(x,y-row_h,total,row_h,fill=1,stroke=0)
  pos=x
  for cell,w in zip(row,widths):txt(str(cell),pos+10,y-row_h+10,11);pos+=w
  y-=row_h
 return y
f=lambda x:f'{x:.2f}'.replace('.',',')
def run(rep,split='test_st1'):
 return next(r for r in A['runs'] if r['method']=='Llama' and r['representation']==rep and r['split']==split)

start('Výsledok experimentov','Formálna štruktúra pomáha. Samotné symboly nestačia.',
      'B1, B2, S0 a Llama 3 8B | Anglické sylogizmy | 191 testovacích príkladov')
card(38,236,282,'76,54 %','DeBERTa: priemer 5 seedov','SD 3,06 p. b.; +11,10 p. b. oproti B1.')
card(339,236,282,'+12,57 p. b.','Llama: kanonický zápis','79,58 % oproti 67,02 % na pôvodnom texte.',PURPLE)
card(640,236,282,'90,58 %','S0: najvyššia presnosť','Symbolická referencia na celom teste; nie LLM.')
para('<b>Negatívny, ale dôležitý výsledok:</b> jednoduchá abstrakcia znížila presnosť Llamy na 52,36 %. '
     'Model v tomto variante označil 185 zo 191 argumentov za platné.',48,198,862,18)
para('Všetkých 16 exportovaných neurónových vyhodnotení bolo prepočítaných z predikcií. '
     'Nenašli sa nezrovnalosti v kontrolovaných metrikách, rozdeleniach dát ani exportoch.',48,110,862,13,MUTED)
end()

start('01 / Porovnanie modelov','Presnosť: symbolický systém vedie, DeBERTa prekonáva B1',
      'B2: priemer ± smerodajná odchýlka medzi seedmi. Ostatné metódy: jeden beh; bez odhadu variability medzi behmi.')
c.drawImage(str(OUT/'accuracy.png'),38,180,width=884,height=220,mask='auto',preserveAspectRatio=True,anchor='c')
table(['Metóda','ACC (%)','TCE (nižšie lepšie)','CS (vyššie lepšie)'],[
 ['B1 / TF-IDF + LR','65,45','10,90','18,82'],
 ['B2 / DeBERTa, priemer','76,54','9,37','24,13'],
 ['S0 / symbolická referencia','90,58','10,42','26,37']],38,160,[340,150,195,199],25)
para('S0 má najvyššiu ACC, nie najnižšie TCE vo všetkých jednotlivých behoch. ACC, TCE a CS sú odlišné kritériá.',38,54,884,10,MUTED)
end()

start('02 / Llama: rozklad chýb','Abstrakcia vedie k odpovedi „platný“ takmer vždy',
      'Llama 3 8B Instruct, NF4 4-bit, rovnaký prompt v1, greedy decoding; žiadne nejednoznačné odpovede.')
c.drawImage(str(OUT/'llama_cells.png'),38,199,width=884,height=202,mask='auto',preserveAspectRatio=True,anchor='c')
table(['Vstup Llamy','ACC (%)','TCE','CS','Predikcie „platný“'],[
 ['Pôvodný text','67,02','43,62','13,97','147 / 191'],
 ['Abstrakcia','52,36','48,96','10,66','185 / 191'],
 ['Kanonický zápis','79,58','27,66','18,27','135 / 191']],38,180,[295,125,110,110,244],26)
para('Kanonický vstup zlepšil ACC, ale jeho CS (18,27) zostáva pod B1 (18,82). '
     'Vysoké TCE abstrakcie súvisí najmä s rozdielom medzi platnými a neplatnými argumentmi, nie iba s vierohodnosťou.',38,72,884,10,MUTED)
end()

start('03 / B2: stabilita tréningu','DeBERTa sa učí; výsledok závisí od náhodného seedu',
      '5 epoch, learning rate 2e-5, efektívna dávka 16. Checkpoint podľa najvyššej dev ACC, pri zhode skoršia epocha.')
rows=[]
for seed in range(5):
 d=next(r for r in A['runs'] if r['method']=='B2' and r['seed']==seed and r['split']=='dev')
 t=next(r for r in A['runs'] if r['method']=='B2' and r['seed']==seed and r['split']=='test_st1')
 rows.append([seed,t['best_epoch'],f(d['official_scores']['accuracy']),f(t['official_scores']['accuracy']),f(t['official_scores']['tce']),f(t['official_scores']['combined_score'])])
table(['Seed','Epocha','Dev ACC','Test ACC','Test TCE','Test CS'],rows,38,393,[65,75,90,90,90,90],29)
para('<b>Test ACC: 76,54 ± 3,06 %</b><br/>Rozsah: 72,77 až 79,58 %.<br/><br/>SD opisuje rozdiely medzi seedmi, nie 95 % interval spoľahlivosti.',578,380,330,16)
para('Seed 0 má najvyššiu dev ACC. Seed 1 má najvyššie testové CS (30,11), ale výber víťaza podľa testu by bol metodicky nesprávny.',578,245,330,13)
c.drawImage(str(OUT/'b2_learning.png'),38,52,width=515,height=161,mask='auto',preserveAspectRatio=True,anchor='c')
para('Priemer CS cez seedy = 24,13 ± 3,81.<br/>Priemer jednotlivých CS nie je CS vypočítané z priemernej ACC a TCE.',578,123,330,12,MUTED)
end()

start('04 / Kontrola výsledkov','Čísla sú konzistentné. Audit neznamená nový GPU experiment.',
      f'{len(A["checks"])} automatických kontrol; {len(A["issues"])} zistených nezrovnalostí v kontrolovaných položkách.')
para('<b>Čo sa overilo</b><br/>• Úplnosť ID: 192 dev a 191 test príkladov v každom behu.<br/>'
     '• ACC, TCE, CS, confusion matrix a 95 % bootstrap intervaly.<br/>'
     '• Hashe dát a vstupov Llamy; štítky zo surových odpovedí.<br/>'
     '• Päť epoch každého seedu a výber najlepšej podľa dev.<br/>'
     '• Zhoda journalu, predikcií, submission a súhrnnej tabuľky.',38,386,434,15)
para('<b>Čo archív nedokazuje</b><br/>• Neobsahuje váhy modelov ani kompletné konzolové logy.<br/>'
     '• Neobsahuje snapshot tréningového kódu ani úplný pip freeze.<br/>'
     '• Nemožno nezávisle potvrdiť presný GPU typ alebo absenciu ladenia podľa testu.<br/>'
     '• Preskúmali sa predikcie; tréning sa tu znovu nespúšťal.',500,386,422,15)
para('<b>Párové porovnania na teste (exploračné):</b> kanonická Llama oproti pôvodnej opravila 28 príkladov a pokazila 4. '
     'Presný McNemarov test: p = 0,0000193; po Holmovej korekcii troch porovnaní p = 0,0000386.',38,164,884,14)
para('Výsledok podporuje rozdiel na tejto vzorke; nezaručuje prenos na iné prompty, modely alebo jazyky. '
     'Intervaly spoľahlivosti a ostatné párové porovnania sú v audit.json.',38,90,884,12,MUTED)
end()

start('05 / Záver a interpretácia','Čo môžeme povedať pri prezentácii',
      'Výsledky sa týkajú ST1. Archív neobsahuje nové neurónové experimenty pre ST2, ST3 ani ST4.')
para('<b>1. Doladenie pomohlo.</b> DeBERTa prekonala jednoduchý textový baseline v každom z piatich seedov. '
     'Priemerný rozdiel v testovej presnosti je +11,10 percentuálneho bodu.',38,385,884,18)
para('<b>2. Záleží na forme vstupu.</b> Kanonický zápis zvýšil presnosť Llamy, kým samotná abstrakcia ju znížila. '
     'Hypotéza „odstránenie konkrétnych pojmov automaticky pomáha“ sa v tomto experimente nepotvrdila.',38,292,884,18)
para('<b>3. Formálna dedukcia zostáva silnou referenciou.</b> S0 dosiahol najvyššiu ACC. '
     'Parser však pokrýva len 160/191 príkladov; S0 pri ostatných predikuje neplatnosť.',38,192,884,18)
para('Ďalší krok: na dev preskúmať príčiny nadmernej predikcie „platný“, potom doplniť ST2 s výberom premís. '
     'B1 bol refitovaný na train+dev, B2 trénoval na train a vyberal epochu cez dev; porovnanie má túto hranicu.',38,95,884,12,MUTED)
end();c.save()

# Editable, presentation-ready text with the complete primary result table.
lines=['# Overené výsledky SemEval-2026 Task 11 / ST1','',
 'Analýza dodaného archívu z 25. 10. 2026. Nejde o nové spustenie modelov.', '',
 '## Hlavné zistenia','',
 '- DeBERTa: test ACC **76,54 ± 3,06 %** (priemer ± výberová SD piatich seedov), oproti B1 +11,10 p. b.',
 '- Llama canonical: **79,58 %**, oproti original +12,57 p. b.; CS však zostáva 18,27 oproti B1 18,82.',
 '- Llama abstract: **52,36 %**, 185/191 odpovedí „platný“; hypotéza o automatickom prínose abstrakcie sa nepotvrdila.',
 '- S0: **90,58 %** na celom teste, najvyššia ACC v porovnaní; ide o existujúcu symbolickú referenciu.', '',
 '## Testové výsledky (191 príkladov)','', '| Model | ACC (%) | TCE | CS |','|---|---:|---:|---:|',
 '| B1 TF-IDF + LR | 65,45 | 10,90 | 18,82 |',
 '| B2 priemer ± SD | 76,54 ± 3,06 | 9,37 ± 4,77 | 24,13 ± 3,81 |',
 '| S0 | 90,58 | 10,42 | 26,37 |']
for rep,label in [('original','Llama original'),('abstract','Llama abstract'),('canonical','Llama canonical')]:
 s=run(rep)['official_scores'];lines.append(f'| {label} | {f(s["accuracy"])} | {f(s["tce"])} | {f(s["combined_score"])} |')
lines += ['', 'ACC: vyššie je lepšie. TCE: nižšie je lepšie. CS = ACC / (1 + ln(1 + TCE)). '
          'TCE kombinuje rozdiely podľa platnosti aj vierohodnosti; nie je čistou mierou belief bias.',
          '', '## B2: jednotlivé seedy','', '| Seed | Vybraná epocha | Dev ACC | Test ACC | Test TCE | Test CS |','|---|---:|---:|---:|---:|---:|']
lines += ['| '+' | '.join(map(str,row))+' |' for row in rows]
lines += ['', 'Checkpoint sa vyberal podľa dev ACC, nie podľa testového skóre. Seed 0 má najvyššiu dev ACC. '
          'Päť seedov nie je päť nezávislých testovacích množín. SD medzi seedmi nie je bootstrap CI.',
          '', '## Llama: rozklad podľa platnosti a vierohodnosti','',
          '| Vstup | Platné/vierohodné | Platné/nevierohodné | Neplatné/vierohodné | Neplatné/nevierohodné |',
          '|---|---:|---:|---:|---:|']
for rep in ['original','abstract','canonical']:
 s=run(rep)['official_scores'];lines.append('| '+rep+' | '+' | '.join(f(s[k]) for k in ['acc_plausible_valid','acc_implausible_valid','acc_plausible_invalid','acc_implausible_invalid'])+' |')
lines += ['', 'Skupiny majú 48, 48, 47 a 48 príkladov. Llama canonical správne klasifikuje všetkých 96 platných '
          'argumentov, ale iba 56/95 neplatných. Abstract správne rozpozná len 5/95 neplatných.',
          'Parser pokryl 160/191 testových vstupov. Na týchto vstupoch má Llama original ACC 67,50 %, '
          'abstract 50,00 % a canonical 82,50 %. Na ostatných 31 príkladoch dostali všetky varianty pôvodný '
          'text a majú totožné odpovede (ACC 64,52 %). Nejde teda o tri odlišné vstupy na celom teste.',
          '', '## Validačné výsledky','',
          'DeBERTa: 78,96 ± 2,04 %. Llama original: 64,06 %, abstract: 53,13 %, canonical: 66,67 %. '
          'Vyššia testová ACC kanonickej Llamy oproti dev sama osebe nie je dôkazom úniku dát.',
          '', '## Audit a štatistika','',
          f'Prešlo {len(A["checks"])} kontrol; bez zistených nezrovnalostí. Overili sa úplnosť ID, hashe dát, '
          'metriky, bootstrap CI, confusion matrix, výber epochy a zhoda exportov. Všetky Llama odpovede sú interpretovateľné.',
          'Exploračné presné McNemarove testy; Holmova korekcia sa vzťahuje na nasledujúce tri porovnania:','',
          '| Porovnanie A vs B | Len A správne | Len B správne | p | p po Holmovi |','|---|---:|---:|---:|---:|']
for p in A['exploratory_mcnemar']:lines.append(f'| {p["a"]} vs {p["b"]} | {p["a_only_correct"]} | {p["b_only_correct"]} | {p["p_value"]:.6g} | {p["p_holm"]:.6g} |')
lines += ['', 'B2 seed 0 je tu porovnaný s B1, pretože má najvyššiu dev ACC; nevyberáme seed podľa testu.',
          '', '## Hranice interpretácie','',
          '- Archív neobsahuje váhy, kompletné logy, snapshot kódu ani celý pip freeze; nezávislá GPU reprodukcia neprebehla.',
          '- Absenciu ladenia podľa testu nemožno z exportov dokázať. Verejný test už bol použitý pri baseline.',
          '- Llama používa NF4 double quantization, prompt v1 a max. 8 tokenov; nejde o všeobecný výsledok všetkých Llama modelov.',
          '- B1 používa finálny refit na train+dev (958), B2 trénuje na train (766) a vyberá epochu cez dev (192).',
          '- B1 a S0 sú pôvodné lokálne referencie a sú aj v dodanej comparison.json; nie sú novými neurónovými behmi.',
          '- Archív neobsahuje nové výsledky ST2–ST4.',
          '', '## Text na ústne prezentovanie','',
          '> Porovnali sme textový baseline, doladenú DeBERTu, symbolický systém a tri reprezentácie vstupu Llamy. '
          'DeBERTa zlepšila priemernú presnosť oproti baseline o približne jedenásť percentuálnych bodov. '
          'Pri Llame pomohol jednotný kanonický zápis, ale jednoduché nahradenie pojmov symbolmi viedlo k '
          'takmer neustálej odpovedi „platný“. Výsledky preto ukazujú, že odstránenie obsahu samo osebe nestačí. '
          'Najvyššiu presnosť na tomto teste dosiahla symbolická referencia; jej obmedzením zostáva pokrytie parsera.',
          '', '## Zdroj a reprodukovateľnosť auditu','',f'Vstup: `{A["archive"]}`.',f'SHA-256: `{A["archive_sha256"]}`.',
          'Audit: `audit_neural_results.py`; podrobné metriky a všetky kontroly: `audit.json`.']
(OUT/'vysledky_a_interpretacia.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(PDF)
