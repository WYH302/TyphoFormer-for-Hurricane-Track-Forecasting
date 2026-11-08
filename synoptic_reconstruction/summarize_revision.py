"""Recompute the added sensitivity from paired saved endpoint errors."""
from pathlib import Path
import json
import numpy as np
import prepare_revision as p
HERE=p.HERE


def paired(a,b,ids):
    unique=np.unique(ids);d=np.asarray(a,dtype=np.float64)-b
    values=np.asarray([d[ids==s].mean() for s in unique])
    draws=np.random.default_rng(20260913).integers(0,len(unique),(10000,len(unique)))
    low,high=np.percentile(values[draws].mean(1),[2.5,97.5])
    return dict(delta_km=float(values.mean()),ci_low=float(low),ci_high=float(high),
        improved_storms=int(sum(values<0)),storms=len(unique))


def summarize():
    assert json.loads((HERE/'training_complete.json').read_text())['status']=='completed'
    periods={}
    for period in ('test','extension'):
        groups={};keys=None
        for head in ('direct','residual'):
            errors=[]
            for seed in (42,123,2024):
                data=np.load(HERE/f'runs/{head}_seed{seed}/{period}_predictions.npz')
                current=list(zip(data['storm_id'].tolist(),data['window'].tolist()))
                if keys is None:keys=current
                assert keys==current
                errors.append(data['pred_error_km'].reshape(-1).astype(np.float64))
                status=json.loads((HERE/f'runs/{head}_seed{seed}/status.json').read_text())
                history=json.loads((HERE/f'runs/{head}_seed{seed}/training_history.json').read_text())
                assert len(history)==40 and status['selected_epoch']==min(history,key=lambda r:r['validation_mean_km'])['epoch']
            groups[head]=np.mean(errors,axis=0)
        for model in ('ridge','cv'):
            data=np.load(HERE/f'{model}_{period}_predictions.npz')
            assert list(zip(data['storm_id'].tolist(),data['window'].tolist()))==keys
            groups[model]=data['error_km'].reshape(-1).astype(np.float64)
        ids=np.asarray([r[0] for r in keys]);unique=np.unique(ids)
        summary={name:dict(window_mean_km=float(e.mean()),storm_mean_km=float(np.mean([e[ids==s].mean() for s in unique]))) for name,e in groups.items()}
        pairs={f'{a}_minus_{b}':paired(groups[a],groups[b],ids) for a,b in
            [('residual','direct'),('residual','ridge'),('ridge','cv')]}
        periods[period]=dict(windows=len(ids),storms=len(unique),models=summary,pairs=pairs)
    report=dict(analysis='Post-test-visible synoptic-first retraining; RMW off, 24 h only',
        periods=periods,ridge_alpha=json.loads((HERE/'ridge_selection.json').read_text())['alpha'],
        intervals='Pointwise percentile intervals, 10000 paired whole-storm draws, seed 20260913; neural errors averaged over seeds before resampling.')
    p.dump(HERE/'sensitivity_results.json',report)
    tex=[r'\begin{table}[!htbp]\centering',r'\caption{Added 24-hour synoptic-first retraining sensitivity with RMW removed. Entries are window / equal-storm endpoint errors (km). Neural entries average per-window errors over seeds 42, 123 and 2024. All configurations and checkpoints use training/validation data; both test periods had been examined before this analysis was designed.}',r'\label{tab:synoptic-retraining}',r'\begin{tabular}{lrr}\toprule',r'Model & 2022--2024 & 2025 \\\midrule']
    for name,label in [('cv','CV'),('ridge','Ridge residual'),('direct','Direct'),('residual','Single residual')]:
        cells=[f"{periods[t]['models'][name]['window_mean_km']:.2f} / {periods[t]['models'][name]['storm_mean_km']:.2f}" for t in periods]
        tex.append(label+' & '+' & '.join(cells)+r' \\')
    tex.extend([r'\midrule','Windows / storms & '+' & '.join(f"{periods[t]['windows']} / {periods[t]['storms']}" for t in periods)+r' \\',r'\bottomrule\end{tabular}\end{table}'])
    (HERE/'generated/retraining_table.tex').write_text('\n'.join(tex),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':summarize()
