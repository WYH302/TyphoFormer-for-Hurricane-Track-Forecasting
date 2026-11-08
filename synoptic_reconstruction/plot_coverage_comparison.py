"""Coverage figure from archived window manifests; no model fitting."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import prepare_revision as p


def main():
    old=pd.read_csv(p.HERE/'generated/historical_window_manifest.csv')
    frozen=json.loads((p.REV/'followup/window_selection/window_manifest.json').read_text(encoding='utf-8'))
    counts=[dict(period='1980–2003',raw_block=int(old.accepted_original.sum()),synoptic_first=len(old))]
    for key,period in [('test2022_2024','2022–2024'),('extension2025','2025')]:
        rows=[r for r in frozen if r['period']==key]
        counts.append(dict(period=period,raw_block=sum(bool(r['accepted_original']) for r in rows),synoptic_first=len(rows)))
    assert [(r['raw_block'],r['synoptic_first']) for r in counts]==[(3642,4624),(630,905),(172,187)]
    out=p.HERE/'generated';out.mkdir(exist_ok=True)
    pd.DataFrame(counts).to_csv(out/'coverage_comparison_counts.csv',index=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,3,figsize=(6.5,3.1),layout='constrained')
    for ax,row in zip(axes,counts):
        vals=[row['raw_block'],row['synoptic_first']]
        bars=ax.bar([0,1],vals,width=.61,color=['#b7c2c7','#27647b'],edgecolor='#36464f',linewidth=.6)
        bars[0].set_hatch('///')
        for bar,value in zip(bars,vals):
            ax.text(bar.get_x()+bar.get_width()/2,value+max(vals)*.035,f'{value:,}',ha='center',va='bottom',fontsize=12)
        ax.set_title(row['period'],fontsize=13,pad=10)
        ax.set_ylim(0,max(vals)*1.26)
        ax.set_xticks([])
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4,integer=True))
        ax.grid(axis='y',alpha=.22,linewidth=.6);ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('24-hour windows',fontsize=12)
    fig.legend(handles=[Patch(facecolor='#b7c2c7',edgecolor='#36464f',hatch='///',label='Raw-block'),Patch(facecolor='#27647b',edgecolor='#36464f',label='Synoptic-first')],
        loc='outside lower center',ncol=2,frameon=False,fontsize=12)
    for ext in ['pdf','svg','eps','png']:
        fig.savefig(out/f'Fig2_coverage_counts.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)
    p.dump(out/'coverage_comparison_provenance.json',dict(counts=counts,
        inputs={'historical_manifest_sha256':p.sha(p.HERE/'generated/historical_window_manifest.csv'),
                'frozen_window_manifest_sha256':p.sha(p.REV/'followup/window_selection/window_manifest.json')},
        scope='Overlapping 24-hour windows; independent vertical axes for each period. No model fit or additional observations.'))
    print(json.dumps(counts,ensure_ascii=False))

if __name__=='__main__':main()
