"""Analyze all fixed capacity sensitivity configurations without reselection."""
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
REV=HERE.parents[1]
sys.path.insert(0,str(REV/'experiments'))
import analyze_revision as analysis
from provenance import source_hashes
HEADS=('direct','residual','adaptive')
SEEDS=(42,123,2024)


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p,x):
    p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def match(a,b):
    assert a['keys']==b['keys'] and np.array_equal(a['future_latlon'],b['future_latlon'])


def metric(e,ids):
    return dict(window_mean_km=float(e.mean()),storm_equal_km=float(np.mean([e[ids==sid].mean() for sid in sorted(set(ids))])))


def interval(r,p):
    return f"{r[p+'_mean']:+.2f} [{r[p+'_ci_low']:+.2f}, {r[p+'_ci_high']:+.2f}]"


def main():
    state=read(HERE/'queue_status.json')
    assert state['status']=='completed' and state['completed_fits']==9 and state['holdout2025_evaluated']
    plan=read(HERE/'plan_lock.json')
    locked=read(HERE/'all_selections_frozen.json')
    assert plan['created_utc']<locked['frozen_utc']<state['completed_utc']
    expected=plan['fixed']['code_sha256']
    assert source_hashes(tuple(expected))==expected
    sources=[dict(path=str(p),sha256=sha(p)) for p in (Path(__file__),HERE/'plan_lock.json',HERE/'all_selections_frozen.json',Path(analysis.__file__))]
    curve_rows,curves=[],{}
    for head in HEADS:
        for seed in SEEDS:
            name=f'lead4_{head}_rmw_on_seed{seed}'
            configs=[]
            for size,root in [('small',HERE),('full',REV/'results/controlled_matrix')]:
                run=root/'runs'/name
                config=read(run/'frozen_config.json')
                configs.append(config)
                status=read(run/'status.json')
                hist=read(run/'training_history.json')
                assert len(hist)==40 and status['status']=='completed'
                best=min(hist,key=lambda x:x['validation_mean_km'])
                assert status['selected_epoch']==best['epoch']
                assert config['d_model']==(64 if size=='small' else 256)
                assert config['num_layers']==(1 if size=='small' else 3)
                if size=='small':
                    assert sha(run/'best.pt')==locked['selected'][name]['checkpoint_sha256']
                row=dict(size=size,head=head,seed=seed,parameters=config['parameters'],selected_epoch=best['epoch'],
                    selected_val_km=best['validation_mean_km'],first_val_km=hist[0]['validation_mean_km'],final_val_km=hist[-1]['validation_mean_km'],
                    final_minus_best_val_km=hist[-1]['validation_mean_km']-best['validation_mean_km'],
                    first_train_loss=hist[0]['train_loss'],final_train_loss=hist[-1]['train_loss'],
                    training_loss_reduction_pct=100*(1-hist[-1]['train_loss']/hist[0]['train_loss']),
                    train_plus_validation_seconds=status['total_seconds_train_plus_validation'],
                    cuda_peak_allocated_bytes=status['cuda_peak_allocated_bytes'])
                curve_rows.append(row)
                curves[size,head,seed]=hist
                sources.extend(dict(path=str(run/n),sha256=sha(run/n)) for n in ('frozen_config.json','training_history.json','status.json'))
            for key in ('seed','epochs','batch_size','lr','weight_decay','patience','num_heads','dropout','cv_convention',
                        'aux_direct','aux_residual','aux_gate','features','data_sha256','code_sha256'):
                assert configs[0][key]==configs[1][key],(name,key)
    results,pairs=[],[]
    for period,subdir in [('test2022_2024','runs'),('extension2025','holdout2025')]:
        p=REV/f'results/ridge/lead4_rmw_on_{period}_predictions.npz'
        ridge=analysis.load_npz(p)
        sources.append(dict(path=str(p),sha256=sha(p)))
        errors={'ridge_on':analysis.vector(ridge['error_km']).astype(np.float64)}
        ids=ridge['ids']
        for size,root in [('small',HERE),('full',REV/'results/controlled_matrix')]:
            for head in HEADS:
                group=[]
                for seed in SEEDS:
                    name=f'lead4_{head}_rmw_on_seed{seed}'
                    run=root/subdir/name
                    p=run/'test_predictions.npz'
                    data=analysis.load_npz(p)
                    match(data,ridge)
                    assert np.isfinite(data['pred_error_km']).all()
                    if size=='small' and period=='extension2025':
                        lock=read(run/'evaluation_lock.json')
                        assert lock['checkpoint_sha256']==locked['selected'][name]['checkpoint_sha256']
                        assert datetime.fromtimestamp(p.stat().st_mtime,timezone.utc).isoformat()>locked['frozen_utc']
                        sources.extend(dict(path=str(run/n),sha256=sha(run/n)) for n in ('evaluation_lock.json','development_identity_audit.json'))
                    group.append(analysis.vector(data['pred_error_km']))
                    sources.append(dict(path=str(p),sha256=sha(p)))
                errors[size+'_'+head]=np.mean(group,axis=0).astype(np.float64)
        for label,e in errors.items():
            results.append(dict(population=period,model=label,windows=len(e),storms=len(set(ids)),**metric(e,ids)))
        comparisons=[('small_'+h,'full_'+h) for h in HEADS]+[('small_'+h,'ridge_on') for h in HEADS]+[
            ('small_residual','small_direct'),('small_adaptive','small_direct'),('small_adaptive','small_residual')]
        for a,b in comparisons:
            stats=analysis.cluster(errors[a]-errors[b],ids,10000)
            pairs.append(dict(population=period,A=a,B=b,**stats))
    fig,axes=plt.subplots(3,2,figsize=(11,10),sharex=True)
    for h,head in enumerate(HEADS):
        for col,(key,ylabel) in enumerate([('train_loss','Training normalized MSE'),('validation_mean_km','Validation endpoint error (km)')]):
            ax=axes[h,col]
            for size,color in [('small','#2067a0'),('full','#b65329')]:
                data=np.asarray([[x[key] for x in curves[size,head,seed]] for seed in SEEDS])
                ax.plot(range(1,41),data.mean(axis=0),color=color,label=size,lw=1.6)
                ax.fill_between(range(1,41),data.min(axis=0),data.max(axis=0),color=color,alpha=.12)
                if col==1:
                    for seed in SEEDS:
                        r=next(r for r in curve_rows if r['size']==size and r['head']==head and r['seed']==seed)
                        ax.scatter(r['selected_epoch'],r['selected_val_km'],color=color,s=14)
            ax.set_title(head);ax.set_ylabel(ylabel);ax.set_xlabel('Epoch');ax.grid(alpha=.15);ax.legend()
    fig.suptitle('Post-review 24 h capacity sensitivity: all fixed RMW-on fits\nMean and range across seeds 42/123/2024; dots indicate validation-selected checkpoints')
    fig.tight_layout(rect=(0,0,1,.95))
    for ext in ('png','pdf'):
        fig.savefig(HERE/('capacity_learning_curves.'+ext),dpi=180)
    plt.close(fig)
    aggregates=[]
    for head in HEADS:
        for size in ('small','full'):
            group=[r for r in curve_rows if r['head']==head and r['size']==size]
            assert len({r['parameters'] for r in group})==1
            aggregates.append(dict(head=head,size=size,parameters=group[0]['parameters'],selected_epochs=[r['selected_epoch'] for r in group],
                selected_val_mean_km=float(np.mean([r['selected_val_km'] for r in group])),
                final_val_mean_km=float(np.mean([r['final_val_km'] for r in group])),
                final_minus_best_mean_km=float(np.mean([r['final_minus_best_val_km'] for r in group])),
                mean_train_plus_validation_seconds=float(np.mean([r['train_plus_validation_seconds'] for r in group]))))
    report=dict(completed_utc=datetime.now(timezone.utc).isoformat(),post_review_after_test_visibility=True,no_reselection=True,
        small_configuration=dict(d_model=64,num_layers=1,trajectory_layers=2,rmw='on',lead_hours=24),
        bootstrap_replicates=10000,bootstrap_seed=20260913,sign='A minus B; negative favors A',
        uncertainty='Seed errors averaged per window before storm bootstrap; pointwise percentile95% intervals condition on fixed seeds and configurations, not selection/multiplicity/post-review uncertainty.',
        runtime_limit='Small fits used up to2GPUworkers; historical fits up to4. Training-plus-validation elapsed times include contention and do not measure isolated inference speed.',
        groups=aggregates,training_runs=curve_rows,results=results,comparisons=pairs,sources=sources)
    write(HERE/'analysis.json',report)
    for name,rows in [('training_runs.csv',curve_rows),('results.csv',results),('paired_comparisons.csv',pairs)]:
        with (HERE/name).open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    md=['# Fixed 24 h capacity sensitivity','',
        'This diagnostic was specified after test results and early-overfitting curves were visible. All nine fits are reported; no configuration is chosen based on these outcomes. The small encoder has width64 and one backbone layer, with the existing two trajectory layers retained. Data, losses, optimizer, dropout,40epochs and validation checkpoint selection match the full-size comparisons.', '',
        '|Head|Size|Parameters|Chosen epochs (42/123/2024)|Best val km|Final val km|Train+val seconds/fit|',
        '|---|---|---:|---|---:|---:|---:|']
    for r in aggregates:
        md.append(f"|{r['head']}|{r['size']}|{r['parameters']}|{r['selected_epochs']}|{r['selected_val_mean_km']:.2f}|{r['final_val_mean_km']:.2f}|{r['mean_train_plus_validation_seconds']:.2f}|")
    md+=['',report['runtime_limit'],'','|Period|Model|Window mean km|Storm equal km|Windows/storms|','|---|---|---:|---:|---|']
    for r in results:
        md.append(f"|{r['population']}|{r['model']}|{r['window_mean_km']:.2f}|{r['storm_equal_km']:.2f}|{r['windows']}/{r['storms']}|")
    md+=['','|Period|A|B|W difference [95% CI]|S difference [95% CI]|Improved storms|','|---|---|---|---|---|---:|']
    for r in pairs:
        md.append(f"|{r['population']}|{r['A']}|{r['B']}|{interval(r,'window')}|{interval(r,'storm')}|{100*r['fraction_storms_negative']:.1f}%|")
    md+=['',report['uncertainty'], '',
        'Interpretation must distinguish capacity sensitivity from universal architecture claims: both width and backbone depth changed together. A smaller model can alter overfitting and test accuracy, but this single setting does not establish optimal capacity. In all groups the evaluated model is the early validation-selected checkpoint, not necessarily epoch40. Original choices/results remain intact.', '']
    (HERE/'report.md').write_text('\n'.join(md),encoding='utf-8')
    tex=[r'\begin{table}[t]',r'\centering\small',r'\caption{Post-review 24 h capacity sensitivity, RMW retained. All nine reduced-capacity fits are reported. Test errors (km) average each window across three seeds; both capacities use identical training and validation-selection rules.}',r'\label{tab:capacity24}',r'\begin{tabular}{llrrrr}',r'\toprule',r'Head & Capacity & Parameters & Val & 2022--24 & 2025 \\',r'\midrule']
    for r in aggregates:
        label=r['size']+'_'+r['head']
        old=next(x for x in results if x['model']==label and x['population']=='test2022_2024')
        new=next(x for x in results if x['model']==label and x['population']=='extension2025')
        tex.append(f"{r['head']} & {r['size']} & {r['parameters']/1e6:.3f}M & {r['selected_val_mean_km']:.2f} & {old['window_mean_km']:.2f} & {new['window_mean_km']:.2f} "+r'\\')
    tex += [r'\bottomrule',r'\end{tabular}',r'\end{table}', '',
        r'This fixed capacity diagnostic was specified after test visibility. Width and backbone depth were reduced jointly from 256/3 to64/1 while retaining two trajectory layers. It is not a blind test or a selection of an optimal reduced model. All configurations use only the original validation set to choose epochs.', '']
    (HERE/'capacity24.tex').write_text('\n'.join(tex),encoding='utf-8')
    assert all(sha(Path(r['path']))==r['sha256'] for r in sources)
    write(HERE/'verification.json',dict(checked_utc=datetime.now(timezone.utc).isoformat(),
        input_sha256_unchanged=len(sources),source_code_unchanged=True,all9_checkpoint_hashes_match=True,
        all18_small_prediction_groups_ground_truth_aligned=True,development_config_matches_except_capacity=True,
        all9_sealed_before_this_runs_2025=True,primary_comparisons=len(pairs)))
    print(json.dumps(dict(groups=aggregates,results=results,comparisons=pairs),indent=2))


if __name__=='__main__':
    main()
