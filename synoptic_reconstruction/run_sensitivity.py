"""Post-review fixed-grid Ridge sensitivity. Original artifacts are read-only."""
import csv
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
REV = HERE.parents[1]
sys.path.insert(0, str(REV / 'experiments'))
import ridge_baseline as ridge
import analyze_revision as analysis

OUT = HERE / 'results'
SEEDS = (42, 123, 2024)


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def now():
    return datetime.now(timezone.utc).isoformat()


def identity(a, b):
    assert a['keys'] == b['keys'], 'Window identities differ'
    assert np.array_equal(a['future_latlon'], b['future_latlon']), 'Targets differ'


def ci(row, prefix):
    return f"{row[prefix+'_mean']:+.2f} [{row[prefix+'_ci_low']:+.2f}, {row[prefix+'_ci_high']:+.2f}]"


def main():
    started = time.perf_counter()
    plan = read(HERE / 'plan_lock.json')
    assert plan['plan_sha256'] == sha(HERE / 'POST_REVIEW_PLAN.md')
    assert plan['post_review_after_test_visibility']
    grid = np.asarray(plan['alpha_grid'])
    assert len(grid) == 29 and np.allclose(np.log10(grid), np.arange(-4, 10.01, .5))
    OUT.mkdir(exist_ok=True)
    if (OUT / 'run_summary.json').exists():
        raise RuntimeError('Completed sensitivity results already exist; do not overwrite')
    dependencies = [Path(__file__), Path(ridge.__file__), Path(analysis.__file__), ridge.REFERENCE,
                    REV / 'audit/extension_training_identity.json', HERE / 'plan_lock.json']
    provenance = [{'path': str(p), 'sha256': sha(p)} for p in dependencies]
    write(OUT / 'execution_lock.json', dict(started_utc=now(), sources=provenance, plan=plan))
    ridge.ALPHAS = grid  # In-memory override only; original module source is unchanged.
    lib = ridge.reference_functions()
    designs, selections, inputs = {}, [], []
    with threadpool_limits(limits=4):
        for lead in range(1, 5):
            folder = REV / f'data/controlled_12to1_lead{lead}'
            meta = read(folder / 'metadata.json')
            inputs.append(dict(lead=lead, metadata_sha256=sha(folder / 'metadata.json'), splits={
                split: hashlib.sha256(''.join(p.name+':'+sha(p)+'\n' for p in sorted((folder/split).glob('*.npy'))).encode()).hexdigest()
                for split in ('train', 'val')}))
            train, val = (lib.load_records(folder / split, [lead]) for split in ('train', 'val'))
            for rmw in ('on', 'off'):
                label = f'lead{lead}_rmw_{rmw}'
                keep = [i for i, name in enumerate(meta['feature_names']) if rmw == 'on' or name != 'rmax']
                model = ridge.fit_design(train, val, keep, len(meta['feature_names']), lib, lead == 1 and rmw == 'on')
                model_path = OUT / f'{label}_model.npz'
                np.savez_compressed(model_path, **{k: model[k] for k in ('coef', 'intercept', 'x_mean', 'x_std', 'keep')},
                                    alpha=model['alpha'], feature_names=np.asarray([meta['feature_names'][i] for i in keep]))
                selection = dict(label=label, lead_hours=6*lead, rmw=rmw, alpha=model['alpha'],
                    val=model['val'], tuning=model['tuning'], fit_seconds=model['fit_seconds'],
                    model_sha256=sha(model_path), boundary_selected=bool(model['alpha'] in (grid[0], grid[-1])),
                    equivalence_check=model['equivalence_check'])
                write(OUT / f'{label}_selection.json', selection)
                selections.append(selection)
                designs[label] = (model, meta, lead, rmw)
                print(f"Selected {label}: alpha={model['alpha']:g}, val={model['val']['fde']:.4f}", flush=True)
        write(OUT / 'all_selections_frozen.json', dict(frozen_utc=now(), post_review_after_test_visibility=True,
            all_eight_selected_before_this_runs_test_loading=True, selections=selections, training_input_manifests=inputs))
        rows, pairs, saved = [], [], {}
        for label, (model, meta, lead, rmw) in designs.items():
            old_sel_path = REV / f'results/ridge/{label}_selection.json'
            old_sel = read(old_sel_path)
            provenance.append(dict(path=str(old_sel_path), sha256=sha(old_sel_path)))
            # The overlap exactly reproduces the pre-existing validation search.
            old_values = np.asarray([r['val_fde_km'] for r in old_sel['tuning']])
            new_values = np.asarray([r['val_fde_km'] for r in model['tuning'][:17]])
            assert np.allclose(old_values, new_values, atol=1e-4, rtol=0), (label, np.max(abs(old_values-new_values)))
            for period, dirname in [('test2022_2024', f'controlled_12to1_lead{lead}'), ('extension2025', f'extension2025_lead{lead}')]:
                folder = REV / 'data' / dirname
                eval_meta = read(folder / 'metadata.json')
                assert meta['feature_names'] == eval_meta['feature_names']
                assert meta['stats'] == eval_meta['stats']
                data = lib.load_records(folder / 'test', [lead])
                assert set(data['storm_year']).issubset({2025} if period == 'extension2025' else {2022, 2023, 2024})
                path = OUT / f'{label}_{period}_predictions.npz'
                pred = ridge.predict(model, data, len(meta['feature_names']))
                metrics = ridge.save_predictions(path, pred, data, lib)
                with np.load(OUT / f'{label}_model.npz', allow_pickle=False) as stored:
                    assert np.array_equal(ridge.predict(stored, data, len(meta['feature_names'])), pred)
                values = analysis.load_npz(path)
                old_path = REV / f'results/ridge/{label}_{period}_predictions.npz'
                old = analysis.load_npz(old_path)
                identity(values, old)
                provenance.append(dict(path=str(old_path), sha256=sha(old_path)))
                new_err, old_err = (analysis.vector(z['error_km']).astype(np.float64) for z in (values, old))
                stats = analysis.cluster(new_err-old_err, values['ids'], 10000)
                pairs.append(dict(population=period, lead_hours=lead*6, A='expanded_ridge_'+rmw,
                    B='original_ridge_'+rmw, sign='A minus B; negative favors A', **stats))
                row = dict(population=period, lead_hours=6*lead, rmw=rmw, original_alpha=old_sel['alpha'],
                    expanded_alpha=model['alpha'], original_val_km=old_sel['val']['fde'], expanded_val_km=model['val']['fde'],
                    original_window_mean_km=float(old_err.mean()), expanded_window_mean_km=float(new_err.mean()),
                    expanded_storm_equal_km=metrics['storm_mean'], windows=metrics['windows'], storms=metrics['storms'],
                    max_overlap_validation_difference_km=float(np.max(abs(old_values-new_values))))
                rows.append(row)
                saved[period, lead, rmw] = values
                write(OUT / f'{label}_{period}_metrics.json', dict(**row, metrics=metrics))
        for period, subdir in [('test2022_2024', 'runs'), ('extension2025', 'holdout2025')]:
            for neural_rmw in ('on', 'off'):
                group = []
                for seed in SEEDS:
                    run = REV / f'results/controlled_matrix/{subdir}/lead4_adaptive_rmw_{neural_rmw}_seed{seed}'
                    status = read(run / 'status.json')
                    config = read(run / 'frozen_config.json')
                    assert status['status'] == 'completed' and status['test_evaluated']
                    assert config['head'] == 'adaptive' and config['rmw'] == neural_rmw and config['seed'] == seed
                    assert all(config.get(k, 0) == 0 for k in ('aux_direct','aux_residual','aux_gate'))
                    p = run / 'test_predictions.npz'
                    group.append(analysis.load_npz(p))
                    provenance.append(dict(path=str(p), sha256=sha(p)))
                for rmw in ('on', 'off'):
                    ref = saved[period, 4, rmw]
                    for values in group:
                        identity(values, ref)
                    err = np.mean([analysis.vector(z['pred_error_km']) for z in group], axis=0).astype(np.float64)
                    stats = analysis.cluster(err-analysis.vector(ref['error_km']).astype(np.float64), ref['ids'], 10000)
                    pairs.append(dict(population=period, lead_hours=24, A='adaptive_'+neural_rmw,
                        B='expanded_ridge_'+rmw, sign='A minus B; negative favors A', **stats))
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.5), sharex=True)
    for selection in selections:
        ax = axes[0 if selection['rmw']=='on' else 1, selection['lead_hours']//6-1]
        curve = selection['tuning']
        ax.semilogx([x['alpha'] for x in curve], [x['val_fde_km'] for x in curve], marker='.', lw=1.2)
        ax.axvline(1e4, color='gray', ls=':', label='Original upper limit')
        ax.scatter(selection['alpha'], selection['val']['fde'], color='darkred', zorder=4)
        ax.set_title(f"{selection['lead_hours']} h, RMW {selection['rmw']}")
        ax.set_xlabel('Ridge alpha')
        ax.set_ylabel('Validation endpoint error (km)')
        ax.grid(alpha=.2)
    fig.suptitle('Post-review fixed-grid sensitivity: selection uses original validation only')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / ('validation_curves.'+ext), dpi=180)
    plt.close(fig)
    for name, records in [('summary.csv', rows), ('paired_comparisons.csv', pairs)]:
        with (OUT / name).open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(records)
    summary = dict(completed_utc=now(), elapsed_seconds=time.perf_counter()-started, status='completed',
        post_review_after_test_visibility=True, bootstrap_replicates=10000, bootstrap_seed=20260913,
        uncertainty='Pointwise storm percentile intervals conditional on fixed training seeds and selections; no multiplicity or post-review-selection uncertainty correction.',
        boundary_designs=[x['label'] for x in selections if x['boundary_selected']], results=rows, comparisons=pairs, sources=provenance)
    write(OUT / 'run_summary.json', summary)
    md = ['# Post-review Ridge alpha sensitivity', '',
        'This analysis was specified after both test periods were visible. The 29-point grid was fixed before computing these outcomes, and all eight models were selected solely on original validation endpoint error and sealed before this run loaded tests. These are sensitivity results, not a new blind validation. Original results are preserved.', '',
        '|Period|h|RMW|Original alpha|Expanded alpha|Original val km|Expanded val km|Original test km|Expanded test km|',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        md.append(f"|{r['population']}|{r['lead_hours']}|{r['rmw']}|{r['original_alpha']:g}|{r['expanded_alpha']:g}|{r['original_val_km']:.3f}|{r['expanded_val_km']:.3f}|{r['original_window_mean_km']:.3f}|{r['expanded_window_mean_km']:.3f}|")
    md += ['', 'Boundary selections: '+(', '.join(summary['boundary_designs']) or 'none')+'. No test-guided expansion was performed.', '',
        'All original 17 validation grid points reproduce within 0.0001 km; reloaded saved models produce identical forecasts. Prediction identities/targets and original/extension normalization match. Numeric RMW-off removes every historical RMW coordinate; these estimators use no text-derived features.', '',
        '|Period|h|A|B|W difference [95% CI] km|S difference [95% CI] km|Storm fraction A improves|',
        '|---|---:|---|---|---|---|---:|']
    for r in pairs:
        md.append(f"|{r['population']}|{r['lead_hours']}|{r['A']}|{r['B']}|{ci(r,'window')}|{ci(r,'storm')}|{100*r['fraction_storms_negative']:.1f}%|")
    md += ['', 'W denotes pooled windows and S equal storms. Negative differences favor A. Neural errors are averaged per matched window across seeds 42/123/2024 before 10,000 whole-storm draws (seed 20260913); seeds/windows are not independent storms. Both adaptive configurations are fixed sensitivity comparators; this analysis does not replace the prior validation-locked neural choice. Intervals are pointwise and do not address multiplicity, model-selection uncertainty or the post-review decision to extend the grid.', '',
        'Interpretation: a validation optimum outside the original grid identifies sensitivity to its upper bound. Improved validation performance need not improve every test population. This does not prove an implementation defect or justify revising hyperparameters using test outcomes.', '']
    (HERE / 'report.md').write_text('\n'.join(md), encoding='utf-8')
    tex = [r'\begin{table}[t]', r'\centering\small',
        r'\caption{Post-review Ridge sensitivity after test visibility. All expanded-grid alphas are selected only on original validation data. Test entries are pooled-window endpoint errors (km); original results are retained.}',
        r'\label{tab:ridge-expanded-alpha}',r'\begin{tabular}{rrlrrrr}',r'\toprule',
        r'h & RMW & $\alpha$ (old/new) & Val old & Val new & 2022--24 & 2025 \\',r'\midrule']
    for r in rows[::2]:
        ext = next(v for v in rows if v['population']=='extension2025' and v['lead_hours']==r['lead_hours'] and v['rmw']==r['rmw'])
        tex.append(f"{r['lead_hours']} & {r['rmw']} & {r['original_alpha']:.2g}/{r['expanded_alpha']:.2g} & {r['original_val_km']:.2f} & {r['expanded_val_km']:.2f} & {r['expanded_window_mean_km']:.2f} & {ext['expanded_window_mean_km']:.2f} "+r'\\')
    tex += [r'\bottomrule',r'\end{tabular}',r'\end{table}', '',
        r'The fixed post-review grid comprised 29 half-decade values from $10^{-4}$ to $10^{10}$. This sensitivity analysis was conducted after test visibility and is not an untouched holdout experiment. All original validation-grid scores were reproduced before comparing the expanded-grid forecasts. Boundary selections: '+('none.' if not summary['boundary_designs'] else ', '.join(summary['boundary_designs']).replace('_',r'\_')+'.'), '']
    (HERE / 'ridge_sensitivity.tex').write_text('\n'.join(tex), encoding='utf-8')
    print(json.dumps(dict(status='completed', seconds=summary['elapsed_seconds'], boundary=summary['boundary_designs'], results=rows), indent=2))


if __name__ == '__main__':
    main()
