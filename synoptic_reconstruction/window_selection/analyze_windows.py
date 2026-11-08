"""Frozen-model population/feature sensitivity; no fitting or model selection."""
from pathlib import Path
import sys,json,csv,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
REV=HERE.parents[1]
sys.path.insert(0,str(REV/'experiments'))
from ridge_baseline import reference_functions,predict


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def writecsv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def paired(a,b,ids):
    unique=np.unique(ids);diff=a-b
    counts=np.asarray([sum(ids==s) for s in unique])
    sums=np.asarray([diff[ids==s].sum() for s in unique]);means=sums/counts
    draws=np.random.default_rng(20260913).integers(0,len(unique),(10000,len(unique)))
    boot_s=means[draws].mean(1);boot_w=sums[draws].sum(1)/counts[draws].sum(1)
    return {'window_delta':float(diff.mean()),'storm_delta':float(means.mean()),
            'window_ci_low':float(np.percentile(boot_w,2.5)) if len(unique)>1 else None,'window_ci_high':float(np.percentile(boot_w,97.5)) if len(unique)>1 else None,
            'storm_ci_low':float(np.percentile(boot_s,2.5)) if len(unique)>1 else None,'storm_ci_high':float(np.percentile(boot_s,97.5)) if len(unique)>1 else None,
            'improved_storms':int(sum(means<0)),'storms':len(unique),'windows':len(ids)}


def main():
    assert json.loads((HERE/'neural_evaluation_checks.json').read_text(encoding='utf-8'))['status']=='passed'
    lib=reference_functions();summaries=[];comparisons=[];baseline_checks=[]
    feature_count=len(json.loads((REV/'data/controlled_12to1_lead4/metadata.json').read_text(encoding='utf-8'))['feature_names'])
    for profile in ('raw_features','synoptic_features'):
        for period in ('test2022_2024','extension2025'):
            root=HERE/'data'/profile/period/'test';out=HERE/'results'/profile/period
            data=lib.load_records(root,[4]);rows=[np.load(p,allow_pickle=True).item() for p in sorted(root.glob('*.npy'))]
            ids=np.asarray(data['storm_ids']);windows=data['window']
            flags={'original':np.asarray([r['accepted_original'] for r in rows]),
                   'landfall':np.asarray([r['recorded_future_landfall'] for r in rows])}
            groups={}
            designs={'CV':None}
            for rmw in ('on','off'):
                designs[f'Ridge_{rmw}']=REV/f'results/ridge/lead4_rmw_{rmw}_model.npz'
                expanded=REV/f'followup/ridge_sensitivity/results/lead4_rmw_{rmw}_model.npz'
                assert expanded.exists(),expanded
                designs[f'Ridge_expanded_{rmw}']=expanded
            for name,path in designs.items():
                pred=data['cv_raw'] if path is None else predict(dict(np.load(path,allow_pickle=False)),data,feature_count)
                latlon=lib.disp_to_latlon_np(pred,data['origin']);err=lib.haversine_np(latlon,data['future'])[:,-1]
                np.savez_compressed(out/f'{name}.npz',error_km=err,pred_latlon=latlon,future_latlon=data['future'],
                                    storm_id=ids,window=windows,**flags)
                groups[name]=err.astype(float)
                if profile=='raw_features' and name in ('Ridge_on','Ridge_off'):
                    rmw=name.split('_')[-1]
                    old=np.load(REV/f'results/ridge/lead4_rmw_{rmw}_{period}_predictions.npz',allow_pickle=False)
                    oldmap={(str(s),int(w)):i for i,(s,w) in enumerate(zip(old['storm_id'],old['window']))}
                    newix=np.flatnonzero(flags['original']);oldix=[oldmap[(str(ids[i]),int(windows[i]))] for i in newix]
                    delta=float(np.abs(old['error_km'][oldix]-err[newix]).max())
                    assert delta<0.01,delta
                    baseline_checks.append({'period':period,'model':name,'original_max_error_difference_km':delta})
            for head,label in [('direct','Direct'),('residual','Single'),('adaptive','Adaptive')]:
                for rmw in ('on','off'):
                    seederr=[]
                    for seed in (42,123,2024):
                        p=np.load(out/f'lead4_{head}_rmw_{rmw}_seed{seed}.npz',allow_pickle=False)
                        assert np.array_equal(p['storm_id'],ids) and np.array_equal(p['window'],windows)
                        assert np.array_equal(p['future_latlon'],data['future'])
                        seederr.append(p['pred_error_km'].reshape(-1).astype(float))
                    groups[f'{label}_{rmw}']=np.mean(seederr,axis=0)
            subsets={'original':flags['original'],'added':~flags['original'],
                     'combined':np.ones(len(ids),dtype=bool),'recorded_future_landfall':flags['landfall']}
            for subset,mask in subsets.items():
                u=np.unique(ids[mask]);assert len(u)>0
                for name,err in groups.items():
                    e=err[mask];si=ids[mask]
                    row={'profile':profile,'period':period,'subset':subset,'model':name,
                         'windows':int(sum(mask)),'storms':len(u),'window_mean':float(e.mean()),
                         'storm_mean':float(np.mean([e[si==s].mean() for s in u]))}
                    summaries.append(row)
                pairs=[]
                for rmw in ('on','off'):
                    pairs += [(f'Single_{rmw}',f'Direct_{rmw}'),(f'Adaptive_{rmw}',f'Single_{rmw}'),
                              (f'Ridge_{rmw}','CV'),(f'Adaptive_{rmw}',f'Ridge_{rmw}'),
                              (f'Ridge_expanded_{rmw}',f'Ridge_{rmw}'),
                              (f'Adaptive_{rmw}',f'Ridge_expanded_{rmw}')]
                for a,b in pairs:
                    comparisons.append({'profile':profile,'period':period,'subset':subset,'A':a,'B':b,
                                        **paired(groups[a][mask],groups[b][mask],ids[mask])})
            np.savez_compressed(out/'aggregate_errors.npz',**groups,storm_id=ids,window=windows,**flags)
    writecsv(HERE/'summary.csv',summaries);writecsv(HERE/'paired_comparisons.csv',comparisons)
    (HERE/'analysis.json').write_text(json.dumps({'status':'completed','summary':summaries,'comparisons':comparisons,
        'baseline_original_recovery_checks':baseline_checks,'replicates':10000,'seed':20260913,
        'estimand':'Three-seed mean of errors, not ensemble coordinates; whole-storm paired resampling within each subset; pointwise intervals.'},indent=2),encoding='utf-8')
    # Main readable tables use separate W/S columns and keep feature policies separate.
    show=['CV','Ridge_on','Ridge_expanded_on','Direct_on','Single_on','Adaptive_on','Adaptive_off']
    labels={'CV':'CV','Ridge_on':'Ridge (original grid)','Ridge_expanded_on':'Ridge (expanded grid)',
            'Direct_on':'A: direct','Single_on':'B: single residual','Adaptive_on':'D: adaptive, RMW on','Adaptive_off':'D: adaptive, RMW off'}
    for profile in ('raw_features','synoptic_features'):
        for period in ('test2022_2024','extension2025'):
            selected=[r for r in summaries if r['profile']==profile and r['period']==period]
            tex=[r'\begin{table}[!htbp]\centering\small',
                 '\\caption{Frozen 24 h model errors (km), '+period.replace('_','--')+', '+profile.replace('_',' ')+'. Cells are window / equal-storm means. Models retain RMW except the labeled off model; CV uses positions only. Expanded Ridge is a separately labeled post-review sensitivity. Future-L means an observed raw L flag in $(T,T+24h]$, not a landfall-location/time prediction score.}',
                 '\\label{tab:selection-'+profile+'-'+period+'}',r'\setlength{\tabcolsep}{3pt}',
                 r'\begin{tabular}{lrrrr}\toprule',r'Model & Original & Added & Combined & Future-L \\\midrule']
            for name in show:
                row={r['subset']:r for r in selected if r['model']==name}
                tex.append(labels[name]+' & '+' & '.join(f"{row[s]['window_mean']:.1f}/{row[s]['storm_mean']:.1f}" for s in subsets)+r' \\')
            row={r['subset']:r for r in selected if r['model']=='CV'}
            tex.extend([r'\midrule', 'Windows / storms & '+' & '.join(f"{row[s]['windows']}/{row[s]['storms']}" for s in subsets)+r' \\',r'\bottomrule\end{tabular}\end{table}'])
            (HERE/f'table_{profile}_{period}.tex').write_text('\n'.join(tex),encoding='utf-8')
    fig,axes=plt.subplots(2,2,figsize=(11,7.2),sharey=False)
    for col,period in enumerate(('test2022_2024','extension2025')):
        for row,(a,b) in enumerate([('Single_on','Direct_on'),('Adaptive_on','Single_on')]):
            ax=axes[row,col]
            for offset,(profile,color) in enumerate([('raw_features','#27658b'),('synoptic_features','#d47b35')]):
                rs=[next(r for r in comparisons if r['profile']==profile and r['period']==period and r['subset']==sub and r['A']==a and r['B']==b) for sub in subsets]
                x=np.arange(4)+(offset-.5)*.15;y=np.array([r['storm_delta'] for r in rs])
                ax.errorbar(x,y,yerr=np.array([[r['storm_delta']-r['storm_ci_low'] if r['storm_ci_low'] is not None else np.nan for r in rs],[r['storm_ci_high']-r['storm_delta'] if r['storm_ci_high'] is not None else np.nan for r in rs]]),fmt='o',capsize=3,color=color,label=profile.replace('_',' '))
            ax.axhline(0,color='gray',lw=.7);ax.set_xticks(range(4),['Original','Added','Combined','Future L']);ax.set_ylabel('Equal-storm difference (km)')
            ax.set_title(period.replace('_','–')+'\n'+a.replace('_on','')+' minus '+b.replace('_on',''))
    axes[0,0].legend(fontsize=8);fig.tight_layout()
    fig.savefig(HERE/'selection_effects.pdf',bbox_inches='tight');fig.savefig(HERE/'selection_effects.png',dpi=180,bbox_inches='tight');plt.close(fig)
    print(json.dumps({'status':'completed','summary_rows':len(summaries),'paired_comparisons':len(comparisons)}))


if __name__=='__main__':main()
