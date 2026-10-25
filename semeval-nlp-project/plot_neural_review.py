import json, zipfile
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'reports/neural_results_review'
a=json.loads((OUT/'audit.json').read_text(encoding='utf-8'))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
names=['B1: TF-IDF + LR','B2: DeBERTa (5 seedov)','S0: symbolická referencia','Llama: pôvodný text','Llama: abstrakcia','Llama: kanonický zápis']
baseline=json.loads((ROOT/'outputs/baseline_results.json').read_text(encoding='utf-8'))
symbolic=json.loads((ROOT/'outputs/parser_eval.json').read_text(encoding='utf-8'))
values=[next(x['accuracy'] for x in baseline['test_st1'] if x['model']=='B1_tfidf_lr'),
        a['b2_summary']['test_st1']['accuracy']['mean'],symbolic['test_st1']['S0_reference_all_examples']['accuracy']]
values += [next(r['official_scores']['accuracy'] for r in a['runs'] if r['method']=='Llama' and r['split']=='test_st1' and r['representation']==rep) for rep in ['original','abstract','canonical']]
colors=['#8a99ad','#3273ac','#168773','#a795c6','#d18259','#6456a1']
fig,ax=plt.subplots(figsize=(11,4.4))
y=np.arange(len(names))
ax.barh(y,values,color=colors,height=.63)
ax.errorbar(values[1],1,xerr=a['b2_summary']['test_st1']['accuracy']['sd'],fmt='none',ecolor='#132b41',capsize=5,lw=2)
ax.set(yticks=y,yticklabels=names,xlim=(0,103),xlabel='Presnosť ACC (%) na ST1, n = 191')
ax.invert_yaxis(); ax.xaxis.grid(True,alpha=.15); ax.set_axisbelow(True)
for i,v in enumerate(values): ax.text(v+1.3 if i!=1 else v+4.8,i,f'{v:.2f} %'.replace('.',','),va='center',fontsize=11)
fig.tight_layout();fig.savefig(OUT/'accuracy.png',dpi=180,bbox_inches='tight');plt.close(fig)
ll=[next(r for r in a['runs'] if r['method']=='Llama' and r['split']=='test_st1' and r['representation']==rep) for rep in ['original','abstract','canonical']]
keys=['acc_plausible_valid','acc_implausible_valid','acc_plausible_invalid','acc_implausible_invalid']
matrix=np.array([[r['official_scores'][k] for k in keys] for r in ll])
fig,ax=plt.subplots(figsize=(10,3.25))
im=ax.imshow(matrix,cmap='YlGnBu',vmin=0,vmax=100,aspect='auto')
ax.set(xticks=range(4),xticklabels=['Platné\nvierohodné (n = 48)','Platné\nnevierohodné (n = 48)','Neplatné\nvierohodné (n = 47)','Neplatné\nnevierohodné (n = 48)'],yticks=range(3),yticklabels=['Pôvodný text','Abstrakcia','Kanonický zápis'])
ax.tick_params(axis='x',labelsize=9)
for i in range(3):
 for j in range(4): ax.text(j,i,f'{matrix[i,j]:.1f} %'.replace('.',','),ha='center',va='center',color='white' if matrix[i,j]>55 else '#132b41',fontsize=14,fontweight='bold')
fig.colorbar(im,ax=ax,label='ACC (%)',shrink=.8);fig.tight_layout();fig.savefig(OUT/'llama_cells.png',dpi=180,bbox_inches='tight');plt.close(fig)
with zipfile.ZipFile(ROOT/'reproduction/semeval_neural_results.zip') as z:
 fig,ax=plt.subplots(figsize=(9,3.5))
 for seed in range(5):
  h=json.loads(z.read(f'b2/seed_{seed}/history.json'))
  ax.plot([x['epoch'] for x in h],[x['accuracy'] for x in h],marker='o',label=f'Seed {seed}')
 ax.set(xticks=range(1,6),xlabel='Epocha',ylabel='Dev ACC (%)',ylim=(45,90));ax.grid(alpha=.15);ax.legend(ncol=5,loc='lower right',fontsize=9)
 fig.tight_layout();fig.savefig(OUT/'b2_learning.png',dpi=180,bbox_inches='tight');plt.close(fig)
print('Three charts saved.')
