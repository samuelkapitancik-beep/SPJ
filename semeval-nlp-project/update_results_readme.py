"""Regenerate README tables and figures from versioned JSON results (no training)."""
from pathlib import Path
import json
import statistics
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
FIG = ROOT / 'docs/figures'
read = lambda p: json.loads((ROOT / p).read_text(encoding='utf-8'))
fmt = lambda n: f'{n:.2f}'.replace('.', ',')
KEYS = ['accuracy', 'tce', 'combined_score']


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    baseline, symbolic = read('outputs/baseline_results.json'), read('outputs/parser_eval.json')
    rows = []
    for i, label in enumerate(['B0a · väčšinová trieda', 'B0b · vierohodnosť = platnosť', 'B1 · TF-IDF + LR']):
        rows.append({'label': label, 'dev': baseline['dev'][i], 'test': baseline['test_st1'][i]})
    seeds = {split: [read(f'outputs/neural/b2/seed_{s}/{split}_metrics.json') for s in range(5)] for split in ['dev','test']}
    b2 = {'label': 'B2 · DeBERTa, priemer ± SD'}
    for split in seeds:
        b2[split] = {k: statistics.mean(r['official_scores'][k] for r in seeds[split]) for k in seeds[split][0]['official_scores']}
        b2[split+'_sd'] = {k: statistics.stdev(r['official_scores'][k] for r in seeds[split]) for k in KEYS}
    rows.append(b2)
    rows.append({'label': 'S0 · symbolická referencia', 'dev': symbolic['dev']['S0_reference_all_examples'], 'test': symbolic['test_st1']['S0_reference_all_examples']})
    for rep in ['original','abstract','canonical']:
        rows.append({'label': 'N1 · Llama / '+rep,
                     **{split: read(f'outputs/neural/llama/{folder}/{rep}/metrics.json')['official_scores'] for split,folder in [('dev','dev'),('test','test_st1')]}})
    def table(headers, content):
        return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in content])
    result_rows = []
    for r in rows:
        values=[]
        for split in ['dev','test']:
            for k in KEYS:
                values.append(fmt(r[split][k]) + (' ± '+fmt(r[split+'_sd'][k]) if r is b2 else ''))
        result_rows.append([r['label'],*values])
    blocks={'RESULTS': table(['Model','Dev ACC ↑','Dev TCE ↓','Dev CS ↑','Test ACC ↑','Test TCE ↓','Test CS ↑'],result_rows)}
    blocks['SEEDS']=table(['Seed','Vybraná epocha','Dev ACC','Test ACC','Test TCE','Test CS'],
        [[s,read(f'outputs/neural/b2/seed_{s}/training.json')['best_epoch'],fmt(seeds['dev'][s]['official_scores']['accuracy']),
          *[fmt(seeds['test'][s]['official_scores'][k]) for k in KEYS]] for s in range(5)])
    keys=['acc_plausible_valid','acc_implausible_valid','acc_plausible_invalid','acc_implausible_invalid']
    blocks['CELLS']=table(['Llama vstup','Platné / vierohodné (48)','Platné / nevierohodné (48)',
                           'Neplatné / vierohodné (47)','Neplatné / nevierohodné (48)','Predikcie platný'],
        [[rep,*[fmt(read(f'outputs/neural/llama/test_st1/{rep}/metrics.json')['official_scores'][k]) for k in keys],
          str(sum(p['validity'] for p in read(f'outputs/neural/llama/test_st1/{rep}/predictions.json')))+' / 191'] for rep in ['original','abstract','canonical']])
    s=baseline['test_st2_B1_validity_plus_random_premises']
    blocks['ST2']=table(['Metóda','n','ACC','F1 premís','TCE','CS'],[['B1 + náhodný výber premís',190,*[fmt(s[k]) for k in ['accuracy','f1_premises','tce','combined_score']]]])
    blocks['DIAGNOSTICS']=table(['Vstup B1','Analyzátor','Dev ACC','Dev TCE'],[[r['input'],r['analyzer'],fmt(r['dev_accuracy']),fmt(r['dev_tce'])] for r in read('outputs/diagnostics.json')['part_ablation']])
    path=ROOT/'README.md'; text=path.read_text(encoding='utf-8')
    for name, body in blocks.items():
        pattern=f'<!-- BEGIN {name} -->.*?<!-- END {name} -->'
        replacement=f'<!-- BEGIN {name} -->\n{body}\n<!-- END {name} -->'
        text,n=re.subn(pattern,lambda _:replacement,text,flags=re.S)
        if n!=1: raise ValueError(f'Missing/duplicate README block: {name}')
    path.write_text(text,encoding='utf-8')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    labels=['B0a: väčšinová trieda','B0b: diagnostická heuristika','B1: TF-IDF + LR','B2: DeBERTa (5 seedov)','S0: symbolická referencia','N1: Llama original','N1: Llama abstract','N1: Llama canonical']
    colors=['#a0a9b5','#bec4cc','#3478ab','#164d78','#168773','#9b8abe','#c07850','#6654a0']
    fig,axes=plt.subplots(1,3,figsize=(15,5.2),sharey=True)
    for ax,key,title,limit in zip(axes,KEYS,['Presnosť ACC ↑ (%)','Obsahový efekt TCE ↓','Kombinované skóre CS ↑'],[112,112,36]):
        vals=[r['test'][key] for r in rows]; y=np.arange(len(rows))
        ax.barh(y,vals,color=colors,height=.65)
        ax.errorbar(vals[3],3,xerr=b2['test_sd'][key],fmt='none',ecolor='#101e30',capsize=4)
        ax.set_xlim(0,limit);ax.set_title(title,pad=14);ax.set_axisbelow(True);ax.xaxis.grid(alpha=.15)
        for i,v in enumerate(vals):ax.text(v+limit*.025+(b2['test_sd'][key] if i==3 else 0),i,fmt(v),va='center',fontsize=9)
    axes[0].set_yticks(range(len(labels)),labels);axes[0].invert_yaxis()
    fig.suptitle('ST1: výsledky na 191 testovacích príkladoch',fontsize=16)
    fig.text(.5,.015,'B2: priemer ± SD medzi piatimi seedmi; ostatné metódy: jeden beh. B0b používa zlatý štítok vierohodnosti.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.04,1,.95]);fig.savefig(FIG/'st1_results.png',dpi=180);plt.close(fig)
    matrix=np.array([[r['test'][k] for k in keys] for r in rows if r['label'].startswith('N1')])
    fig,ax=plt.subplots(figsize=(10,3.6));im=ax.imshow(matrix,vmin=0,vmax=100,cmap='YlGnBu',aspect='auto')
    ax.set_xticks(range(4),['Platné\nvierohodné (48)','Platné\nnevierohodné (48)','Neplatné\nvierohodné (47)','Neplatné\nnevierohodné (48)'])
    ax.set_yticks(range(3),['Original','Abstract','Canonical']);ax.set_title('Llama: presnosť v štyroch skupinách ST1',pad=14)
    for i in range(3):
        for j in range(4):ax.text(j,i,fmt(matrix[i,j])+' %',ha='center',va='center',color='white' if matrix[i,j]>55 else '#142b41',fontweight='bold')
    fig.colorbar(im,ax=ax,label='ACC (%)');fig.tight_layout();fig.savefig(FIG/'llama_cells.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.2))
    for seed in range(5):
        h=read(f'outputs/neural/b2/seed_{seed}/history.json')
        axes[0].plot([x['epoch'] for x in h],[x['accuracy'] for x in h],marker='o',label=f'Seed {seed}')
    axes[0].set(title='B2: vývoj validačnej presnosti',xlabel='Epocha',ylabel='Dev ACC (%)',xticks=range(1,6));axes[0].legend(ncol=3,fontsize=8)
    vals=[r['official_scores']['accuracy'] for r in seeds['test']]
    axes[1].bar(range(5),vals,color='#3478ab');axes[1].axhline(b2['test']['accuracy'],color='#168773',ls='--',label='Priemer B2')
    axes[1].axhline(rows[2]['test']['accuracy'],color='#b26843',ls=':',label='B1')
    axes[1].set(title='B2: testová presnosť podľa seedu',xlabel='Seed',ylabel='Test ACC (%)',xticks=range(5),ylim=(0,100));axes[1].legend(fontsize=9)
    for i,v in enumerate(vals):axes[1].text(i,v+1.2,fmt(v),ha='center',fontsize=9)
    for ax in axes:ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.tight_layout();fig.savefig(FIG/'b2_seeds.png',dpi=180);plt.close(fig)
    (ROOT/'outputs/results_summary.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    print('README tables, three charts and outputs/results_summary.json regenerated.')


if __name__=='__main__':main()
