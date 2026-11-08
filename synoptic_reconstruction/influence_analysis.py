"""Descriptive eight-storm 2025 influence analysis from frozen predictions only."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def load_predictions(path):
    with np.load(path, allow_pickle=False) as data:
        ids = data['storm_id' if 'storm_id' in data.files else 'storm_ids'].astype(str)
        window = data['window'].astype(int)
        keys = list(zip(ids, window))
        assert len(set(keys)) == len(keys)
        order = np.asarray(sorted(range(len(keys)), key=lambda i: keys[i]))
        errors = data['pred_error_km' if 'pred_error_km' in data.files else 'error_km']
        return dict(keys=[keys[i] for i in order], ids=ids[order], window=window[order],
                    truth=data['future_latlon'][order], errors=errors.reshape(len(ids), -1)[order, -1])


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(error, ridge, ids):
    storms = sorted(set(ids))
    candidate_storm = np.asarray([error[ids == sid].mean() for sid in storms])
    ridge_storm = np.asarray([ridge[ids == sid].mean() for sid in storms])
    return dict(windows=len(ids), storms=len(storms),
                candidate_window_mean_km=float(error.mean()), ridge_window_mean_km=float(ridge.mean()),
                window_delta_km=float((error - ridge).mean()),
                candidate_storm_equal_km=float(candidate_storm.mean()), ridge_storm_equal_km=float(ridge_storm.mean()),
                storm_equal_delta_km=float((candidate_storm - ridge_storm).mean()))


def influence(error, ridge, ids, label):
    error, ridge = np.asarray(error, dtype=np.float64), np.asarray(ridge, dtype=np.float64)
    full = aggregate(error, ridge, ids)
    per_storm, leave_one_out = [], []
    for sid in sorted(set(ids)):
        keep = ids != sid
        n = int((~keep).sum())
        row = dict(model=label, storm_id=sid, windows=n,
                   candidate_mean_km=float(error[~keep].mean()), ridge_mean_km=float(ridge[~keep].mean()),
                   delta_km=float((error[~keep] - ridge[~keep]).mean()))
        per_storm.append(row)
        reduced = aggregate(error[keep], ridge[keep], ids[keep])
        assert reduced['storms'] == 7 and reduced['windows'] == len(ids) - n
        leave_one_out.append(dict(model=label, omitted_storm=sid, omitted_windows=n, **reduced,
                                  equal_delta_change_from_full_km=reduced['storm_equal_delta_km'] - full['storm_equal_delta_km'],
                                  window_delta_change_from_full_km=reduced['window_delta_km'] - full['window_delta_km'],
                                  equal_delta_sign_reversal=(reduced['storm_equal_delta_km'] < 0) != (full['storm_equal_delta_km'] < 0),
                                  window_delta_sign_reversal=(reduced['window_delta_km'] < 0) != (full['window_delta_km'] < 0)))
    assert sum(r['windows'] for r in per_storm) == full['windows']
    assert np.isclose(np.mean([r['delta_km'] for r in per_storm]), full['storm_equal_delta_km'], atol=1e-12)
    assert np.isclose(sum(r['windows'] * r['delta_km'] for r in per_storm) / full['windows'], full['window_delta_km'], atol=1e-12)
    return full, per_storm, leave_one_out


def figures(output, full, per_storm, leave_one_out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
                         'font.size': 9, 'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
    ids = sorted(set(r['storm_id'] for r in per_storm))
    labels = [sid.split('_')[-1] + f' ({next(r["windows"] for r in per_storm if r["storm_id"] == sid)})' for sid in ids]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.8), sharey=True, constrained_layout=True)
    colors = {'adaptive_rmw_off': '#0072B2', 'adaptive_rmw_on': '#E76F51'}
    for model, color in colors.items():
        offset = -.11 if model.endswith('off') else .11
        y = np.arange(len(ids)) + offset
        one = {r['storm_id']: r for r in per_storm if r['model'] == model}
        loo = {r['omitted_storm']: r for r in leave_one_out if r['model'] == model}
        label = 'Adaptive, RMW ' + model.split('_')[-1]
        axes[0].plot([one[sid]['delta_km'] for sid in ids], y, 'o', color=color, label=label)
        for ax, key in [(axes[1], 'storm_equal_delta_km'), (axes[2], 'window_delta_km')]:
            ax.plot([loo[sid][key] for sid in ids], y, 'o', color=color, label=label)
            ax.axvline(full[model][key], color=color, linestyle='--', linewidth=1)
    for ax in axes:
        ax.axvline(0, color='.55', linewidth=.7)
        ax.set_xlabel('Adaptive − Ridge-on error (km)')
        ax.grid(axis='x', alpha=.15)
    axes[0].set_yticks(np.arange(len(ids)), labels)
    axes[0].invert_yaxis()
    axes[0].set_title('Individual storm mean difference')
    axes[1].set_title('Omit indicated storm: equal-storm')
    axes[2].set_title('Omit indicated storm: pooled windows')
    axes[0].set_ylabel('Storm (accepted windows)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=2)
    fig.savefig(output / 'influence_2025_24h.pdf')
    fig.savefig(output / 'influence_2025_24h.png', dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    revision = args.revision_root.resolve()
    output = args.output_dir or revision / 'followup/influence_results'
    output.mkdir(parents=True, exist_ok=True)
    lock_path = revision / 'results/controlled_matrix/holdout_selection_lock.json'
    lock = read_json(lock_path)
    assert lock['selected_per_lead_hours']['24']['selected_configuration'] == 'adaptive_rmw_off'
    ridge_path = revision / 'results/ridge/lead4_rmw_on_extension2025_predictions.npz'
    ridge = load_predictions(ridge_path)
    assert len(ridge['keys']) == 172 and len(set(ridge['ids'])) == 8
    assert all(sid.split('_')[0].endswith('2025') for sid in ridge['ids'])
    full, per_storm, leave_one_out, hashes = {}, [], [], {}
    all_errors = {}
    for mode in ('off', 'on'):
        errors, configs = [], []
        for seed in (42, 123, 2024):
            run = revision / f'results/controlled_matrix/holdout2025/lead4_adaptive_rmw_{mode}_seed{seed}'
            config = read_json(run / 'frozen_config.json')
            status = read_json(run / 'status.json')
            assert status['status'] == 'completed' and status['test_evaluated']
            assert config['head'] == 'adaptive' and config['rmw'] == mode and config['seed'] == seed
            config = {k: v for k, v in config.items() if k not in ('seed', 'device', 'device_name', 'threads')}
            configs.append(config)
            path = run / 'test_predictions.npz'
            data = load_predictions(path)
            assert data['keys'] == ridge['keys'] and np.array_equal(data['truth'], ridge['truth'])
            assert np.isfinite(data['errors']).all()
            errors.append(data['errors'])
            hashes[str(path.relative_to(revision))] = hashlib.sha256(path.read_bytes()).hexdigest()
        assert all(c == configs[0] for c in configs)
        label = 'adaptive_rmw_' + mode
        # Match the original analysis: average float32 per-window seed errors, then promote for aggregation.
        all_errors[label] = np.mean(np.stack(errors), axis=0)
        full[label], individual, omitted = influence(all_errors[label], ridge['errors'], ridge['ids'], label)
        per_storm.extend(individual)
        leave_one_out.extend(omitted)
    hashes[str(ridge_path.relative_to(revision))] = hashlib.sha256(ridge_path.read_bytes()).hexdigest()
    hashes[str(lock_path.relative_to(revision))] = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    write_csv(output / 'per_storm.csv', per_storm)
    write_csv(output / 'leave_one_storm_out.csv', leave_one_out)
    np.savez_compressed(output / 'paired_window_errors.npz', storm_id=ridge['ids'], window=ridge['window'],
                        ridge_on=ridge['errors'], **all_errors)
    summary = dict(period='2025', lead_hours=24, models=full, source_sha256=hashes,
                    selection='Adaptive/RMW-off was selected on validation; adaptive/RMW-on is the separately predeclared conditional configuration. Both use fixed Ridge/RMW-on as the reference.',
                    estimands='First average the three seed errors per window. Equal-storm averages each storm equally; pooled-window averages every retained window equally.',
                    scope='Descriptive influence only. Every leave-one-out row removes all windows of exactly one storm. No exclusion is adopted, no model is retrained, no result replaces the primary full-eight-storm analysis.',
                    no_rounded_inputs=True, n_primary_storms=8, n_primary_windows=172,
                    influence={model: dict(equal_delta_range_km=[min(r['storm_equal_delta_km'] for r in leave_one_out if r['model'] == model), max(r['storm_equal_delta_km'] for r in leave_one_out if r['model'] == model)],
                                           window_delta_range_km=[min(r['window_delta_km'] for r in leave_one_out if r['model'] == model), max(r['window_delta_km'] for r in leave_one_out if r['model'] == model)],
                                           equal_sign_reversal_storms=[r['omitted_storm'] for r in leave_one_out if r['model'] == model and r['equal_delta_sign_reversal']],
                                           window_sign_reversal_storms=[r['omitted_storm'] for r in leave_one_out if r['model'] == model and r['window_delta_sign_reversal']]) for model in full})
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    md = ['# Eight-storm influence: 2025, 24 h', '',
          'These descriptive calculations preserve all eight storms as the primary analysis. No leave-one-out result is selected to replace it. Positive differences mean the adaptive model has higher error than fixed Ridge-on. Calculations use saved per-window values, not rounded paper tables.', '',
          '| Model | Full equal-storm delta (km) | Full pooled-window delta (km) | Leave-one-out equal-storm range | Leave-one-out pooled-window range |',
          '|---|---:|---:|---:|---:|']
    for model, values in full.items():
        change = summary['influence'][model]
        md.append(f'| {model} | {values["storm_equal_delta_km"]:+.6f} | {values["window_delta_km"]:+.6f} | {change["equal_delta_range_km"][0]:+.6f} to {change["equal_delta_range_km"][1]:+.6f} | {change["window_delta_range_km"][0]:+.6f} to {change["window_delta_range_km"][1]:+.6f} |')
    md += ['', 'Per-storm errors, deltas and counts are in `per_storm.csv`; all 16 omissions, retained counts and both weighting schemes are in `leave_one_storm_out.csv`. `summary.json` records sign reversals and exact source hashes. Dashed plot lines are full-sample differences; the plotted omission is not an endorsed alternative population. The two RMW modes are not reselected from these diagnostics.', '']
    (output / 'README.md').write_text('\n'.join(md), encoding='utf-8')
    tex = [r'\begin{table}[t]\centering\scriptsize',
           r'\caption{Descriptive leave-one-storm-out sensitivity at 24 h in 2025. Differences are adaptive minus fixed Ridge-on, in km. Each row retains seven storms; the full eight-storm result remains primary. W: pooled windows; S: equal storms.}',
           r'\label{tab:influence-2025}', r'\begin{tabular}{lrrrrr}\toprule',
           r'Omitted storm & Retained windows & RMW-off S & RMW-off W & RMW-on S & RMW-on W \\\midrule']
    for sid in sorted(set(ridge['ids'])):
        off = next(r for r in leave_one_out if r['model'] == 'adaptive_rmw_off' and r['omitted_storm'] == sid)
        on = next(r for r in leave_one_out if r['model'] == 'adaptive_rmw_on' and r['omitted_storm'] == sid)
        tex.append(f'{sid.split("_")[-1]} & {off["windows"]} & {off["storm_equal_delta_km"]:+.3f} & {off["window_delta_km"]:+.3f} & {on["storm_equal_delta_km"]:+.3f} & {on["window_delta_km"]:+.3f}' + r' \\')
    tex += [r'\bottomrule\end{tabular}\end{table}', '']
    (output / 'influence_table.tex').write_text('\n'.join(tex), encoding='utf-8')
    if not args.no_plots:
        figures(output, full, per_storm, leave_one_out)
    print(json.dumps(dict(models=full, influence=summary['influence']), indent=2))


if __name__ == '__main__':
    main()
