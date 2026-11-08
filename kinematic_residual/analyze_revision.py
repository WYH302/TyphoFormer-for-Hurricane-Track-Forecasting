"""Storm-cluster comparisons and observed-history diagnostics for saved predictions.

Usage: python analyze_revision.py --replicates 10000
Safe to repeat after the training queue completes: all inputs are read-only.
Only complete, identical-window three-seed neural configurations are included.
"""
import argparse
import csv
import hashlib
import json
import itertools
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from ridge_baseline import reference_functions

REV = Path(__file__).resolve().parents[1]
OUT = REV / 'results/diagnostics'
FIG = REV / 'figures'
LIB = reference_functions()
EXPECTED_SEEDS = {42, 123, 2024}
BOOTSTRAP_SEED = 20260913


def json_write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def csv_write(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def vector(values):
    values = np.asarray(values)
    return values.reshape(len(values), -1)[:, -1]


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    ids = arrays.get('storm_ids', arrays.get('storm_id')).astype(str)
    keys = list(zip(ids, arrays['window'].astype(int)))
    assert len(set(keys)) == len(keys), f'Duplicate prediction keys: {path}'
    order = np.asarray(sorted(range(len(keys)), key=lambda i: keys[i]))
    arrays = {k: v[order] if v.ndim and len(v) == len(ids) else v for k, v in arrays.items()}
    arrays['keys'] = [keys[i] for i in order]
    arrays['ids'] = np.asarray([k[0] for k in arrays['keys']])
    return arrays


def local(points, origin):
    points, origin = np.asarray(points, dtype=np.float64), np.asarray(origin, dtype=np.float64)
    return LIB.latlon_to_disp_np(points, origin)


def history_motion(history):
    last3 = np.asarray(history, dtype=np.float64)[:, -3:]
    q = local(last3, last3[:, -1:, :])
    prev, current = q[:, 1] - q[:, 0], q[:, 2] - q[:, 1]
    norm_prev, norm_current = np.linalg.norm(prev, axis=1), np.linalg.norm(current, axis=1)
    cross = prev[:, 0] * current[:, 1] - prev[:, 1] * current[:, 0]
    dot = (prev * current).sum(axis=1)
    turn = np.rad2deg(np.abs(np.arctan2(cross, dot)))
    turn[(norm_prev < 1e-8) | (norm_current < 1e-8)] = np.nan
    return turn, (norm_current - norm_prev) / 6.0, current


def load_context(folder):
    metadata = json.loads((folder.parent / 'metadata.json').read_text(encoding='utf-8-sig'))
    rows = [np.load(p, allow_pickle=True).item() for p in sorted(folder.glob('*.npy'))]
    rows.sort(key=lambda row: (str(row['storm_id']), int(row['window'])))
    names = metadata['feature_names']
    status_indices = [i for i, name in enumerate(names) if name.startswith('status_')]
    status_names = np.asarray([names[i][7:] for i in status_indices])
    raw_init = np.stack([row['input_num'][-1] for row in rows]) * np.asarray(metadata['stats']['x_std']) + np.asarray(metadata['stats']['x_mean'])
    context = dict(keys=[(str(r['storm_id']), int(r['window'])) for r in rows],
                   history=np.stack([r['history_latlon'] for r in rows]),
                   future=np.stack([r['future_latlon'] for r in rows]),
                   origin=np.stack([r['origin'] for r in rows]),
                   year=np.asarray([int(r['storm_year']) for r in rows]),
                   init_status=status_names[raw_init[:, status_indices].argmax(axis=1)],
                   init_date=np.asarray([int(r['history_date'][-1]) for r in rows]),
                   init_time=np.asarray([int(r['history_time'][-1]) for r in rows]),
                   target_date=np.asarray([int(r['future_date'][-1]) for r in rows]),
                   target_time=np.asarray([int(r['future_time'][-1]) for r in rows]))
    context['turn'], context['speed_change'], context['velocity'] = history_motion(context['history'])
    context['ids'] = np.asarray([k[0] for k in context['keys']])
    return context


def raw_tracks():
    path = REV / 'data/hurdat2_frozen2024_plus2025.txt'
    lines = path.read_text(encoding='utf-8-sig').splitlines()
    tracks, index, i = {}, {}, 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        header = [s.strip() for s in lines[i].split(',')]
        count = int(header[2])
        if 2004 <= int(header[0][4:8]) <= 2025:
            ident = header[0] + '_' + header[1]
            track = []
            for line in lines[i + 1:i + count + 1]:
                p = [s.strip() for s in line.split(',')]
                lat = float(p[4][:-1]) * (-1 if p[4][-1] == 'S' else 1)
                lon = float(p[5][:-1]) * (-1 if p[5][-1] == 'W' else 1)
                row = dict(date=int(p[0]), time=int(p[1]), status=p[3], latlon=[lat, lon])
                track.append(row)
                index[ident, row['date'], row['time']] = row
            tracks[ident] = track
        i += count + 1
    return tracks, index


def cluster(values, ids, replicates):
    """Whole-storm percentile bootstrap; seeds and windows are never clusters."""
    values = np.asarray(values, dtype=np.float64)
    mask = np.isfinite(values)
    values, ids = values[mask], np.asarray(ids)[mask]
    unique = sorted(set(ids))
    if not unique:
        return dict(windows=0, storms=0, window_mean=None, storm_mean=None,
                    window_ci_low=None, window_ci_high=None, storm_ci_low=None, storm_ci_high=None,
                    fraction_storms_negative=None)
    sums = np.asarray([values[ids == sid].sum() for sid in unique])
    count = np.asarray([(ids == sid).sum() for sid in unique])
    means = sums / count
    result = dict(windows=len(values), storms=len(unique), window_mean=float(values.mean()),
                  storm_mean=float(means.mean()), fraction_storms_negative=float((means < 0).mean()))
    if len(unique) < 2:
        return dict(**result, window_ci_low=None, window_ci_high=None, storm_ci_low=None, storm_ci_high=None)
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(0, len(unique), (replicates, len(unique)))
    win_draws = sums[draws].sum(axis=1) / count[draws].sum(axis=1)
    storm_draws = means[draws].mean(axis=1)
    result.update(window_ci_low=float(np.quantile(win_draws, .025)), window_ci_high=float(np.quantile(win_draws, .975)),
                  storm_ci_low=float(np.quantile(storm_draws, .025)), storm_ci_high=float(np.quantile(storm_draws, .975)))
    return result


def diagnostic_groups(ctx, thresholds):
    turn = np.full(len(ctx['turn']), 'undefined', dtype='<U18')
    valid = np.isfinite(ctx['turn'])
    turn[valid] = np.asarray(['low', 'middle', 'high'])[np.digitize(ctx['turn'][valid], thresholds['turn_tertiles_deg'])]
    speed = np.asarray(['lower_change', 'middle_change', 'upper_change'])[np.digitize(ctx['speed_change'], thresholds['speed_change_tertiles_kmh'])]
    sign = np.where(ctx['speed_change'] > 1e-6, 'accelerating', np.where(ctx['speed_change'] < -1e-6, 'decelerating', 'unchanged'))
    columns = dict(year=ctx['year'].astype(str), init_status=ctx['init_status'], target_status=ctx['target_status'],
                   historical_turn=turn, historical_speed_change=speed, historical_acceleration_sign=sign)
    result = [('overall', 'all', np.ones(len(turn), dtype=bool))]
    for kind, labels in columns.items():
        result.extend((kind, label, labels == label) for label in sorted(set(labels)))
    return result


def ridge_models(lead, population, ctx):
    result = {}
    for mode in ('on', 'off'):
        path = REV / f'results/ridge/lead{lead}_rmw_{mode}_{population}_predictions.npz'
        saved = load_npz(path)
        assert saved['keys'] == ctx['keys']
        assert np.array_equal(saved['future_latlon'], ctx['future'])
        selection = json.loads((REV / f'results/ridge/lead{lead}_rmw_{mode}_selection.json').read_text(encoding='utf-8-sig'))
        cost = dict(seed=None, parameters=2 * selection['input_dim'] + 2,
                    seconds=selection['fit_seconds'], selected_epoch=None, device='CPU (four BLAS threads)',
                    cost_scope='One SVD and validation of 17 alphas; excludes data loading and inference')
        result[f'ridge_rmw_{mode}'] = dict(error=vector(saved['error_km']), latlon=saved['pred_latlon'],
                                         seed_error=vector(saved['error_km'])[None], costs=[cost],
                                         seed_latlon=saved['pred_latlon'][None], seeds=[], paths=[str(path)], branches={})
        if mode == 'on':
            cv_ll = LIB.disp_to_latlon_np(saved['cv_raw_km'], ctx['origin'])
            result['cv'] = dict(error=vector(LIB.haversine_np(cv_ll, ctx['future'])), latlon=cv_ll,
                                seed_error=vector(LIB.haversine_np(cv_ll, ctx['future']))[None], costs=[],
                                seed_latlon=cv_ll[None], seeds=[], paths=[str(path)], branches={})
    return result


def neural_models(lead, population, ctx, omitted, family='neural'):
    family_root = 'controlled_matrix' if family == 'neural' else 'mlp_residual'
    root = REV / 'results' / family_root / ('runs' if population == 'test2022_2024' else 'holdout2025')
    candidates = defaultdict(list)
    for path in sorted(root.glob(f'lead{lead}_*/test_predictions.npz')):
        config_path = path.parent / 'frozen_config.json'
        status_path = path.parent / 'status.json'
        if not config_path.exists() or not status_path.exists():
            omitted.append(dict(path=str(path), reason='Missing completed status/configuration'))
            continue
        config = json.loads(config_path.read_text(encoding='utf-8-sig'))
        status = json.loads(status_path.read_text(encoding='utf-8-sig'))
        if status['status'] != 'completed' or not status.get('test_evaluated'):
            omitted.append(dict(path=str(path), reason='Run not completed and evaluated'))
            continue
        aux = any(float(config.get(k, 0)) != 0 for k in ('aux_direct', 'aux_residual', 'aux_gate'))
        label = (f'neural_{config["head"]}' if family == 'neural' else 'mlp_residual') + f'_rmw_{config["rmw"]}' + ('_aux' if aux else '')
        if family == 'mlp':
            assert config['head'] == 'mlp_residual', 'Unexpected MLP comparator head'
        candidates[label].append((int(config['seed']), path, config))
    models = {}
    for label, runs in candidates.items():
        seeds = [seed for seed, _, _ in runs]
        if set(seeds) != EXPECTED_SEEDS or len(seeds) != 3:
            omitted.append(dict(model=label, lead_hours=lead * 6, population=population, seeds=seeds,
                                reason='Requires the complete predeclared three distinct seeds'))
            continue
        arrays, costs, reference_config = [], [], None
        for seed, path, config in sorted(runs):
            protocol = {k: v for k, v in config.items() if k not in {'seed', 'device', 'device_name', 'threads'}}
            if reference_config is None:
                reference_config = protocol
            else:
                assert protocol == reference_config, f'Mixed configuration in {label}'
            saved = load_npz(path)
            assert saved['keys'] == ctx['keys'], path
            assert np.array_equal(saved['future_latlon'], ctx['future']), path
            assert saved['lead_hours'].tolist() == [lead * 6], path
            assert np.isfinite(saved['pred_error_km']).all(), path
            arrays.append(saved)
            status = json.loads((path.parent / 'status.json').read_text(encoding='utf-8-sig'))
            costs.append(dict(seed=seed, parameters=config.get('parameters'),
                              seconds=status.get('total_seconds_train_plus_validation'),
                              selected_epoch=status.get('selected_epoch'),
                              device=config.get('device_name', config.get('device')),
                              cost_scope='Training plus validation through stopping epoch; excludes test evaluation'))
        branches = {}
        for branch in ('direct', 'cv_residual', 'cv'):
            key = branch + '_error_km'
            if all(key in a for a in arrays):
                branches[branch] = np.mean([vector(a[key]) for a in arrays], axis=0)
        gates = np.stack([vector(a['gate']) for a in arrays]) if all('gate' in a for a in arrays) else None
        branch_seed_errors = {branch: np.stack([vector(a[branch + '_error_km']) for a in arrays])
                              for branch in ('direct', 'cv_residual') if all(branch + '_error_km' in a for a in arrays)}
        models[label] = dict(error=np.mean([vector(a['pred_error_km']) for a in arrays], axis=0),
                             seed_error=np.stack([vector(a['pred_error_km']) for a in arrays]), costs=costs,
                             latlon=arrays[0]['pred_latlon'], seed_latlon=np.stack([a['pred_latlon'] for a in arrays]),
                             seeds=sorted(seeds), paths=[str(path) for _, path, _ in sorted(runs)],
                             branches=branches, gate=gates, branch_seed_errors=branch_seed_errors)
    return models


def directional_errors(model, ctx):
    velocity = ctx['velocity']
    norm = np.linalg.norm(velocity, axis=1)
    axis = velocity / np.maximum(norm[:, None], 1e-8)
    delta = local(model['seed_latlon'], ctx['origin'][None, :, None, :]) - local(ctx['future'], ctx['origin'][:, None, :])[None]
    delta = delta[:, :, -1]
    along = (delta * axis[None]).sum(axis=-1)
    cross = -delta[:, :, 0] * axis[None, :, 1] + delta[:, :, 1] * axis[None, :, 0]
    along[:, norm < 1e-8] = np.nan
    cross[:, norm < 1e-8] = np.nan
    return dict(along_signed_km=along.mean(axis=0), cross_signed_km=cross.mean(axis=0),
                along_absolute_km=np.abs(along).mean(axis=0), cross_absolute_km=np.abs(cross).mean(axis=0))


def strict_seed_average(values, strict):
    """Average each window over exactly the same non-neutral-gate seed support."""
    counts = strict.sum(axis=0)
    return np.divide(np.where(strict, values, 0).sum(axis=0), counts,
                     out=np.full(strict.shape[1], np.nan), where=counts > 0)


def gate_policy_values(gate, direct_error, residual_error, fused_error):
    strict = np.abs(gate - .5) > 1e-8
    residual_correct = residual_error <= direct_error
    direct_correct = direct_error <= residual_error
    gate_correct = np.where(gate > .5, residual_correct, direct_correct)
    preference = strict_seed_average(gate_correct, strict)
    cv_preference = strict_seed_average(residual_correct, strict)
    direct_preference = strict_seed_average(direct_correct, strict)
    return {
        'fraction_strict_gate_prefers_lower_error_branch': preference,
        'fraction_always_cv_residual_prefers_lower_error_branch_on_strict_gate_support': cv_preference,
        'fraction_always_direct_prefers_lower_error_branch_on_strict_gate_support': direct_preference,
        'strict_gate_accuracy_minus_always_cv_residual': preference - cv_preference,
        'strict_gate_accuracy_minus_always_direct': preference - direct_preference,
        'gate_induced_single_branch_error_on_strict_support_km': strict_seed_average(np.where(gate > .5, residual_error, direct_error), strict),
        'always_cv_residual_error_on_strict_gate_support_km': strict_seed_average(residual_error, strict),
        'always_direct_error_on_strict_gate_support_km': strict_seed_average(direct_error, strict),
        'fused_error_on_strict_gate_support_km': strict_seed_average(fused_error, strict),
        'best_single_branch_hindsight_error_on_strict_gate_support_km': strict_seed_average(np.minimum(residual_error, direct_error), strict),
    }


def case_context(ctx, candidate, reference, tracks, ix):
    """Describe an already-selected window; never participate in case selection."""
    sid = ctx['ids'][ix]
    begin = (int(ctx['init_date'][ix]), int(ctx['init_time'][ix]))
    end = (int(ctx['target_date'][ix]), int(ctx['target_time'][ix]))
    raw_init = next(r for r in tracks[sid] if (r['date'], r['time']) == begin)
    raw_target = next(r for r in tracks[sid] if (r['date'], r['time']) == end)
    assert raw_init['status'] == ctx['init_status'][ix]
    assert raw_target['status'] == ctx['target_status'][ix]
    assert np.allclose(raw_init['latlon'], ctx['origin'][ix], rtol=0, atol=1e-5)
    assert np.allclose(raw_target['latlon'], ctx['future'][ix, -1], rtol=0, atol=1e-5)
    turn, speed_change, velocity = history_motion(ctx['history'][ix:ix + 1])
    assert np.allclose(velocity[0], ctx['velocity'][ix])
    assert np.isclose(speed_change[0], ctx['speed_change'][ix])
    assert np.isclose(turn[0], ctx['turn'][ix], equal_nan=True)
    endpoint = local(ctx['future'][ix, -1], ctx['origin'][ix])
    last_velocity = velocity[0]
    posthoc = None
    if np.linalg.norm(endpoint) >= 1e-8 and np.linalg.norm(last_velocity) >= 1e-8:
        cross = last_velocity[0] * endpoint[1] - last_velocity[1] * endpoint[0]
        posthoc = float(np.rad2deg(np.arctan2(abs(cross), np.dot(last_velocity, endpoint))))
    candidate_error = float(candidate['seed_error'][:, ix].mean())
    plotted_error = float(candidate['seed_error'][0, ix])
    recomputed_plotted_error = float(LIB.haversine_np(candidate['latlon'][ix:ix + 1], ctx['future'][ix:ix + 1]).reshape(-1)[-1])
    assert np.isclose(candidate_error, candidate['error'][ix], atol=1e-6, rtol=1e-6)
    assert np.isclose(plotted_error, recomputed_plotted_error, atol=.005, rtol=2e-5)
    return dict(init_status=str(ctx['init_status'][ix]), target_status=str(ctx['target_status'][ix]),
                target_timestamp=f'{end[0]} {end[1]:04d}',
                observed_last_speed_kmh=float(np.linalg.norm(last_velocity) / 6.0),
                observed_speed_change_kmh=float(speed_change[0]),
                observed_last_turn_deg=float(turn[0]) if np.isfinite(turn[0]) else None,
                posthoc_true_endpoint_bearing_deviation_deg=posthoc,
                posthoc_definition='Absolute angle between last historical velocity and initialization-to-true-target displacement; uses future truth, not an available-at-initialization predictor or cumulative trajectory curvature.',
                population_windows=len(ctx['keys']), population_storms=len(set(ctx['ids'])),
                selected_storm_windows=int((ctx['ids'] == sid).sum()),
                candidate_seed_count=candidate['seed_error'].shape[0], candidate_seeds=candidate['seeds'],
                candidate_seed_mean_window_error_km=candidate_error,
                candidate_seed_sd_window_error_km=float(candidate['seed_error'][:, ix].std(ddof=1)) if candidate['seed_error'].shape[0] > 1 else 0.0,
                ridge_window_error_km=float(reference['error'][ix]),
                candidate_mean_minus_ridge_window_error_km=candidate_error - float(reference['error'][ix]),
                plotted_seed_window_error_km=plotted_error,
                plotted_error_recomputation_difference_km=abs(plotted_error - recomputed_plotted_error),
                context_array_checks_passed=True)


def plot_cases(population, ctx, methods, tracks, chosen_label):
    candidate, reference = methods[chosen_label], methods['ridge_rmw_on']
    difference = candidate['error'] - reference['error']
    storm_means = {sid: float(difference[ctx['ids'] == sid].mean()) for sid in sorted(set(ctx['ids']))}
    ordered = sorted(storm_means, key=lambda sid: (storm_means[sid], sid))
    cases = []
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.35), constrained_layout=True)
    for ax, sid, kind in zip(axes, [ordered[0], ordered[-1]], ['Best relative storm', 'Worst relative storm']):
        ids = np.flatnonzero(ctx['ids'] == sid)
        # Median window delta represents this selected storm; ties use earliest raw window.
        ix = sorted(ids, key=lambda i: (abs(difference[i] - np.median(difference[ids])), ctx['keys'][i][1]))[0]
        origin = ctx['origin'][ix]
        history = local(ctx['history'][ix], origin)
        begin = (int(ctx['init_date'][ix]), int(ctx['init_time'][ix]))
        end = (int(ctx['target_date'][ix]), int(ctx['target_time'][ix]))
        future = [r['latlon'] for r in tracks[sid] if begin <= (r['date'], r['time']) <= end and r['time'] in (0, 600, 1200, 1800)]
        future = local(future, origin)
        pred = local(candidate['latlon'][ix, -1], origin)
        ridge = local(reference['latlon'][ix, -1], origin)
        ax.plot(history[:, 0], history[:, 1], '-o', color='#707070', markersize=2.8, label='Observed history')
        ax.plot(future[:, 0], future[:, 1], '--o', color='#264653', markersize=3, label='True future track')
        ax.scatter([0], [0], marker='s', color='black', s=22, label='Initialization', zorder=5)
        ax.scatter([ridge[0]], [ridge[1]], marker='x', color='#0072B2', s=42, label='Ridge RMW-on endpoint')
        candidate_name = 'Ridge RMW-off' if chosen_label == 'ridge_rmw_off' else chosen_label.replace('neural_', '').replace('_', ' ')
        if candidate['seeds']:
            candidate_name += f' (seed {candidate["seeds"][0]})'
        ax.scatter([pred[0]], [pred[1]], marker='^', color='#E76F51', s=35, label=candidate_name + ' endpoint')
        ax.set_title(f'{kind}: {sid.split("_")[-1]} {sid.split("_")[0][-4:]}\nStorm mean Δ = {storm_means[sid]:+.2f} km', fontsize=9)
        ax.set_xlabel('East of initialization (km)')
        ax.set_ylabel('North of initialization (km)')
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(alpha=.18)
        cases.append(dict(population=population, lead_hours=24, model=chosen_label, reference='ridge_rmw_on',
                          selection=kind, storm_id=sid, storm_mean_delta_km=storm_means[sid],
                          window=int(ctx['keys'][ix][1]), initialization=f'{begin[0]} {begin[1]:04d}',
                          window_rule='Closest to median window delta within selected storm; tie uses earliest raw window.',
                          plotted_neural_seed=candidate['seeds'][0] if candidate['seeds'] else None,
                          future_track_is_ground_truth_only=True,
                          **case_context(ctx, candidate, reference, tracks, ix)))
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=2, fontsize=7)
    stem = FIG / f'fig_cases_{population}'
    fig.savefig(stem.with_suffix('.pdf'))
    fig.savefig(stem.with_suffix('.png'), dpi=300)
    plt.close(fig)
    return cases


def figure_rmw(comparisons):
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8), constrained_layout=True)
    for ax, population, title in zip(axes, ('test2022_2024', 'extension2025'), ('Historical test (2022–2024)', 'Frozen extension (2025)')):
        rows = [r for r in comparisons if r['population'] == population and r['A'] == 'ridge_rmw_off' and r['B'] == 'ridge_rmw_on' and r['group_type'] == 'overall']
        rows.sort(key=lambda r: r['lead_hours'])
        x = np.asarray([r['lead_hours'] for r in rows])
        y = np.asarray([r['storm_mean'] for r in rows])
        lo, hi = np.asarray([r['storm_ci_low'] for r in rows]), np.asarray([r['storm_ci_high'] for r in rows])
        ax.errorbar(x, y, yerr=[y - lo, hi - y], marker='o', color='#264653', capsize=3, linewidth=1.4)
        ax.axhline(0, color='#909090', linestyle='--', linewidth=.8)
        ax.set_xticks([6, 12, 18, 24])
        ax.set_xlabel('Lead (h)')
        ax.set_title(title)
        ax.set_ylabel('RMW-off − RMW-on (km)')
    fig.savefig(FIG / 'fig_rmw_storm_bootstrap.pdf')
    fig.savefig(FIG / 'fig_rmw_storm_bootstrap.png', dpi=300)
    plt.close(fig)


def extension_is_sealed():
    """Inspect only locks/status files; never peek at unsealed extension arrays."""
    matrix = REV / 'results/controlled_matrix'
    plan = json.loads((matrix / 'queue_plan.json').read_text(encoding='utf-8-sig'))
    if not (matrix / 'holdout_selection_lock.json').exists():
        raise ValueError('Neural 2025 analysis requires the all-checkpoint selection lock')
    for job in plan['jobs']:
        name = f'lead{job["lead"]}_{job["head"]}_rmw_{job["rmw"]}_seed{job["seed"]}' + ('_aux' if job['aux'] else '')
        for subdir in ('runs', 'holdout2025'):
            status_path = matrix / subdir / name / 'status.json'
            if not status_path.exists():
                raise ValueError(f'2025 analysis waits for the complete matrix: {name}/{subdir}')
            status = json.loads(status_path.read_text(encoding='utf-8-sig'))
            if status.get('status') != 'completed' or not status.get('test_evaluated'):
                raise ValueError(f'2025 analysis waits for completed run: {name}/{subdir}')
    mlp = REV / 'results/mlp_residual'
    if not (mlp / 'all_selections_frozen.json').exists():
        raise ValueError('MLP 2025 analysis requires all_selections_frozen.json')
    for lead, rmw, seed in itertools.product(range(1, 5), ('on', 'off'), sorted(EXPECTED_SEEDS)):
        for subdir in ('runs', 'holdout2025'):
            name = f'lead{lead}_mlp_residual_rmw_{rmw}_seed{seed}'
            path = mlp / subdir / name / 'status.json'
            if not path.exists():
                raise ValueError(f'MLP extension not completed: {name}/{subdir}')
            status = json.loads(path.read_text(encoding='utf-8-sig'))
            if status.get('status') != 'completed' or not status.get('test_evaluated'):
                raise ValueError(f'MLP extension not completed: {name}/{subdir}')


def comparison_specifications(methods):
    """Predeclared head, RMW and simple-baseline comparisons; no test ranking."""
    pairs = [('ridge_rmw_off', 'ridge_rmw_on'), ('ridge_rmw_on', 'cv'), ('ridge_rmw_off', 'cv')]
    for rmw in ('on', 'off'):
        ridge, mlp = f'ridge_rmw_{rmw}', f'mlp_residual_rmw_{rmw}'
        pairs += [(mlp, ridge), (mlp, 'cv')]
        for head in ('direct', 'residual', 'fixed', 'adaptive'):
            name = f'neural_{head}_rmw_{rmw}'
            pairs += [(name, ridge), (name, mlp), (name, 'cv')]
        for a, b in (('residual', 'direct'), ('fixed', 'residual'), ('adaptive', 'residual'), ('adaptive', 'fixed')):
            pairs.append((f'neural_{a}_rmw_{rmw}', f'neural_{b}_rmw_{rmw}'))
    for family in ('mlp_residual', 'neural_direct', 'neural_residual', 'neural_fixed', 'neural_adaptive'):
        pairs.append((f'{family}_rmw_off', f'{family}_rmw_on'))
    pairs.append(('neural_adaptive_rmw_on_aux', 'neural_adaptive_rmw_on'))
    return list(dict.fromkeys((a, b) for a, b in pairs if a in methods and b in methods))


def effect_figures(pairs):
    colors = {'on': '#0072B2', 'off': '#E76F51'}
    contrasts = [('residual', 'direct'), ('fixed', 'residual'), ('adaptive', 'residual'), ('adaptive', 'fixed')]
    for population in ('test2022_2024', 'extension2025'):
        data = [r for r in pairs if r['population'] == population and r['group_type'] == 'overall']
        if not any(r['A'].startswith('neural_') and r['B'].startswith('neural_') for r in data):
            continue
        fig, axes = plt.subplots(2, 2, figsize=(7, 5), constrained_layout=True)
        for ax, (a, b) in zip(axes.flat, contrasts):
            for rmw in ('on', 'off'):
                rows = sorted([r for r in data if r['A'] == f'neural_{a}_rmw_{rmw}' and r['B'] == f'neural_{b}_rmw_{rmw}'], key=lambda r: r['lead_hours'])
                if not rows:
                    continue
                y = np.asarray([r['storm_mean'] for r in rows])
                low = np.asarray([r['storm_ci_low'] for r in rows])
                high = np.asarray([r['storm_ci_high'] for r in rows])
                ax.errorbar([r['lead_hours'] for r in rows], y, yerr=[y - low, high - y],
                            marker='o' if rmw == 'on' else 's', color=colors[rmw], capsize=3, label='RMW ' + rmw)
            ax.axhline(0, color='.5', linestyle='--', linewidth=.8)
            ax.set(title=f'{a.capitalize()} − {b}', xlabel='Lead (h)', ylabel='Storm-equal error Δ (km)', xticks=[6, 12, 18, 24])
        handles, labels = axes.flat[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc='outside lower center', ncol=2)
        fig.savefig(FIG / f'fig_head_effects_{population}.pdf')
        fig.savefig(FIG / f'fig_head_effects_{population}.png', dpi=300)
        plt.close(fig)
        rmw_pairs = [r for r in data if '_rmw_off' in r['A'] and r['A'].replace('_rmw_off', '_rmw_on') == r['B']]
        families = sorted(set(r['A'].split('_rmw_')[0] for r in rmw_pairs))
        if families:
            ncols = min(3, len(families))
            nrows = (len(families) + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(7, 2.4 * nrows), squeeze=False, constrained_layout=True)
            for ax, family in zip(axes.flat, families):
                rows = sorted([r for r in rmw_pairs if r['A'] == family + '_rmw_off'], key=lambda r: r['lead_hours'])
                y = np.asarray([r['storm_mean'] for r in rows])
                low, high = np.asarray([r['storm_ci_low'] for r in rows]), np.asarray([r['storm_ci_high'] for r in rows])
                ax.errorbar([r['lead_hours'] for r in rows], y, yerr=[y - low, high - y], marker='o', capsize=3, color='#264653')
                ax.axhline(0, color='.5', linestyle='--', linewidth=.8)
                ax.set(title=friendly_model(family + '_rmw_on')[0], xlabel='Lead (h)',
                       ylabel='RMW-off − RMW-on (km)', xticks=[6, 12, 18, 24])
            for ax in list(axes.flat)[len(families):]:
                ax.set_visible(False)
            fig.savefig(FIG / f'fig_all_rmw_effects_{population}.pdf')
            fig.savefig(FIG / f'fig_all_rmw_effects_{population}.png', dpi=300)
            plt.close(fig)


def friendly_model(name):
    if name == 'cv':
        return 'CV', '--'
    base, rest = name.split('_rmw_', 1)
    rmw = rest.split('_')[0]
    names = {'ridge': 'Ridge residual', 'mlp_residual': 'MLP residual', 'neural_direct': 'Direct',
             'neural_residual': 'Single residual', 'neural_fixed': 'Fixed dual', 'neural_adaptive': 'Adaptive dual'}
    return names.get(base, base) + (' + aux' if rest.endswith('_aux') else ''), rmw


def tex_escape(value):
    text = str(value).replace('test2022_2024', '2022--2024 test').replace('extension2025', '2025 extension')
    return text.replace('&', r'\&').replace('%', r'\%').replace('_', r'\_').replace('#', r'\#')


def latex_table(path, caption, label, headers, rows):
    content = [r'\begin{table}[t]', r'\centering\scriptsize', r'\caption{' + caption + '}',
               r'\label{' + label + '}', r'\resizebox{\textwidth}{!}{%', r'\begin{tabular}{' + 'l' * len(headers) + '}', r'\toprule',
               ' & '.join(headers) + r' \\', r'\midrule']
    content += [' & '.join(row) + r' \\' for row in rows]
    content += [r'\bottomrule', r'\end{tabular}}', r'\end{table}', '']
    path.write_text('\n'.join(content), encoding='utf-8')


def context_tables(generated, cases, subgroup_errors, branches):
    written = []
    for population in ('test2022_2024', 'extension2025'):
        selected = [c for c in cases if c['population'] == population]
        if selected:
            headers = ['Context'] + [tex_escape(c['selection'].replace(' relative storm', '') + ': ' + c['storm_id']) for c in selected]
            items = [('Model', lambda c: tex_escape(friendly_model(c['model'])[0] + ', RMW ' + friendly_model(c['model'])[1])),
                     ('Initialization (UTC)', lambda c: c['initialization']),
                     ('Raw window index', lambda c: str(c['window'])),
                     ('Status: initialization / target', lambda c: c['init_status'] + ' / ' + c['target_status']),
                     ('Last observed speed (km/h)', lambda c: f'{c["observed_last_speed_kmh"]:.2f}'),
                     ('Observed speed change (km/h)', lambda c: f'{c["observed_speed_change_kmh"]:+.2f}'),
                     ('Last observed turn (deg)', lambda c: 'Undefined' if c['observed_last_turn_deg'] is None else f'{c["observed_last_turn_deg"]:.2f}'),
                     ('True endpoint bearing deviation (deg; post hoc)', lambda c: 'Undefined' if c['posthoc_true_endpoint_bearing_deviation_deg'] is None else f'{c["posthoc_true_endpoint_bearing_deviation_deg"]:.2f}'),
                     ('Population windows / storms', lambda c: f'{c["population_windows"]} / {c["population_storms"]}'),
                     ('Selected-storm windows', lambda c: str(c['selected_storm_windows'])),
                     ('Storm-mean error difference vs Ridge (km)', lambda c: f'{c["storm_mean_delta_km"]:+.2f}'),
                     ('Candidate seed count', lambda c: str(c['candidate_seed_count'])),
                     ('Candidate seed-mean window error (km)', lambda c: f'{c["candidate_seed_mean_window_error_km"]:.2f}'),
                     ('Ridge window error (km)', lambda c: f'{c["ridge_window_error_km"]:.2f}'),
                     ('Plotted seed', lambda c: 'Deterministic' if c['plotted_neural_seed'] is None else str(c['plotted_neural_seed'])),
                     ('Plotted-seed window error (km)', lambda c: f'{c["plotted_seed_window_error_km"]:.2f}')]
            rows = [[name] + [formatter(c) for c in selected] for name, formatter in items]
            path = generated / f'revision_case_context_{population}.tex'
            latex_table(path, 'Observed context and measured errors of the fixed-rule 24 h cases, ' + tex_escape(population) + '. Speed change is the difference between the last two six-hour speeds. Future bearing deviation uses the true endpoint and is explicitly post hoc. Case ranking uses storm-mean errors and window selection uses the median within-storm difference; plotted endpoints use one seed, not an ensemble.',
                        f'tab:case-context-{population}', headers, rows)
            written.append(str(path.relative_to(REV)))
        models = ['ridge_rmw_on', 'mlp_residual_rmw_on'] + [f'neural_{h}_rmw_on' for h in ('direct', 'residual', 'fixed', 'adaptive')]
        groups = [('historical_turn', 'low', 'Low historical turn'), ('historical_turn', 'high', 'High historical turn'),
                  ('historical_acceleration_sign', 'accelerating', 'Accelerating'),
                  ('historical_acceleration_sign', 'decelerating', 'Decelerating')]
        rows = []
        def summary_cell(row, precision=1):
            if row is None or row['storm_mean'] is None:
                return '---'
            value = f'{row["storm_mean"]:.{precision}f}'
            if row['storm_ci_low'] is None:
                return value + ' [---]'
            return value + f' [{row["storm_ci_low"]:.{precision}f}, {row["storm_ci_high"]:.{precision}f}]'
        for model in models:
            for kind, group, name in groups:
                error = next((r for r in subgroup_errors if r['population'] == population and r['lead_hours'] == 24 and r['model'] == model and r['group_type'] == kind and r['group'] == group), None)
                if error is None:
                    continue
                gate = next((r for r in branches if r['population'] == population and r['lead_hours'] == 24 and r['model'] == model and r['group_type'] == kind and r['group'] == group and r['statistic'] == 'gate_mean'), None)
                rows.append([tex_escape(friendly_model(model)[0]), name, str(error['windows']), str(error['storms']),
                             summary_cell(error), summary_cell(gate, 2)])
        if rows:
            path = generated / f'revision_motion_context_24h_{population}.tex'
            latex_table(path, 'Measured 24 h motion-context diagnostics with RMW retained, ' + tex_escape(population) + '. Values are equal-storm means [95\\% storm-cluster intervals]; error is in km. Historical turn thresholds are fixed from training windows. Acceleration/deceleration uses the sign of the last observed speed change. Gates are available only for actual dual-head outputs and are not physical probabilities. One-storm intervals and nonexistent gates are not estimated.',
                        f'tab:motion-context-{population}', ['Method', 'Historical context', 'Windows', 'Storms', 'Error [CI]', 'Gate [CI]'], rows)
            written.append(str(path.relative_to(REV)))
    return written


def yearly_directional_tables(generated, subgroup_errors, directional):
    """Render existing measured summaries; do not open any prediction arrays."""
    methods = ['cv', 'ridge_rmw_on', 'mlp_residual_rmw_on'] + [f'neural_{h}_rmw_on' for h in ('direct', 'residual', 'fixed', 'adaptive')]
    metrics = ('along_signed_km', 'cross_signed_km', 'along_absolute_km', 'cross_absolute_km')
    written = []
    for population, years in [('test2022_2024', ('2022', '2023', '2024')), ('extension2025', ('2025',))]:
        rows = []
        for year in years:
            for method in methods:
                found = [r for r in subgroup_errors if r['population'] == population and r['lead_hours'] == 24 and r['model'] == method and r['group_type'] == 'year' and r['group'] == year]
                assert len(found) <= 1, 'Duplicate yearly aggregate'
                if not found:
                    continue
                r = found[0]
                rows.append([year, tex_escape(friendly_model(method)[0]), str(r['windows']), str(r['storms']),
                             f'{r["window_mean"]:.2f}', f'{r["storm_mean"]:.2f}'])
        if rows:
            path = generated / f'revision_yearly_24h_{population}.tex'
            latex_table(path, 'Measured 24 h endpoint errors by year, ' + tex_escape(population) + '. All learned methods retain RMW; CV has no learned feature input. Window weighting averages all accepted windows, whereas storm weighting first averages within each storm and then weights storms equally. Neural per-window errors are averaged across the three completed seeds before either aggregation; seeds do not increase the storm count.',
                        f'tab:yearly-24h-{population}', ['Year', 'Method', 'Windows', 'Storms', 'Window mean (km)', 'Storm mean (km)'], rows)
            written.append(str(path.relative_to(REV)))
        rows = []
        for method in methods:
            records = {r['metric']: r for r in directional if r['population'] == population and r['lead_hours'] == 24 and r['model'] == method and r['group_type'] == 'overall'}
            if not records:
                continue
            assert set(metrics) <= set(records), 'Missing directional component aggregate'
            support = {(records[m]['windows'], records[m]['storms']) for m in metrics}
            assert len(support) == 1, 'Directional components do not share identical valid support'
            windows, storms = next(iter(support))
            row = [tex_escape(friendly_model(method)[0]), str(windows), str(storms)]
            for metric in metrics:
                r = records[metric]
                row.append('---' if r['window_mean'] is None else f'{r["window_mean"]:+.2f} / {r["storm_mean"]:+.2f}' if 'signed' in metric else f'{r["window_mean"]:.2f} / {r["storm_mean"]:.2f}')
            rows.append(row)
        if rows:
            path = generated / f'revision_along_cross_24h_{population}.tex'
            latex_table(path, 'Measured 24 h directional errors (km), ' + tex_escape(population) + '. Each cell reports window-weighted / equal-storm values. The reference is the last observed six-hour heading in the initialization tangent plane, not the future track. Positive along error points forward along that historical heading; positive cross error points left. Signed errors and absolute component errors are computed per seed before seed averaging. Counts include only windows with a defined historical direction and their represented storms. All learned methods retain RMW.',
                        f'tab:along-cross-24h-{population}', ['Method', 'Valid windows', 'Valid storms', 'Along bias W/S', 'Cross bias W/S', 'Along MAE W/S', 'Cross MAE W/S'], rows)
            written.append(str(path.relative_to(REV)))
    return written


def integration_artifacts(overall_errors, pairs, branches, costs, included, omitted, extension_enabled, cases, subgroup_errors, directional):
    """Compact actual-result tables; missing groups are explicit dashes, never zeros."""
    generated = REV / 'manuscript/generated'
    generated.mkdir(parents=True, exist_ok=True)
    overall_pairs = [r for r in pairs if r['group_type'] == 'overall']
    summary = dict(bootstrap_seed=BOOTSTRAP_SEED, extension_neural_analysis_enabled=extension_enabled,
                   outcome_status='Actual complete groups only; availability is not proof all requested work is complete.',
                   historical_b4_predictions_available=False, overall_errors=overall_errors,
                   paired_comparisons=overall_pairs, branch_gate=[r for r in branches if r['group_type'] == 'overall'],
                   gate_diagnostics_present=any(r['statistic'] == 'gate_mean' for r in branches),
                   training_costs=costs, included_groups=[{k: r[k] for k in ('population', 'lead_hours', 'model', 'seeds')} for r in included],
                   omitted_groups=omitted, case_context=cases,
                   yearly_24h=[r for r in subgroup_errors if r['lead_hours'] == 24 and r['group_type'] == 'year'],
                   directional_24h=[r for r in directional if r['lead_hours'] == 24 and r['group_type'] == 'overall'],
                   tables=context_tables(generated, cases, subgroup_errors, branches) + yearly_directional_tables(generated, subgroup_errors, directional))
    for population in ('test2022_2024', 'extension2025'):
        population_rows = [r for r in overall_errors if r['population'] == population]
        methods = sorted(set(r['model'] for r in population_rows))
        lookup = {(r['model'], r['lead_hours']): r for r in population_rows}
        table = []
        for method in methods:
            name, rmw = friendly_model(method)
            row = [tex_escape(name), rmw]
            for lead in (6, 12, 18, 24):
                r = lookup.get((method, lead))
                if r is None:
                    row.append('---')
                elif r['seed_count'] > 1:
                    row.append(f'${r["window_mean"]:.2f} \\pm {r["seed_window_sd"]:.2f}$')
                else:
                    row.append(f'{r["window_mean"]:.2f}')
            table.append(row)
        path = generated / f'revision_main_results_{population}.tex'
        latex_table(path, 'Measured endpoint error (km), ' + tex_escape(population) + '. Neural entries show three-seed mean and sample standard deviation of window-mean errors. A dash in the RMW column indicates that the feature setting does not apply to CV.',
                    f'tab:measured-{population}', ['Method', 'RMW', '6 h', '12 h', '18 h', '24 h'], table)
        summary['tables'].append(str(path.relative_to(REV)))
        pop_pairs = [r for r in overall_pairs if r['population'] == population]
        for category in ('heads', 'rmw', 'baselines'):
            chosen = []
            for r in pop_pairs:
                a, b = r['A'], r['B']
                rmw = a.replace('_rmw_off', '_rmw_on') == b and '_rmw_off' in a
                heads = a.startswith('neural_') and b.startswith('neural_') and not rmw
                belongs = rmw if category == 'rmw' else heads if category == 'heads' else not rmw and not heads
                # Main simple comparisons focus on numeric single-residual/adaptive and MLP/Ridge vs CV.
                if belongs and (category != 'baselines' or not a.startswith(('neural_direct', 'neural_fixed'))):
                    chosen.append(r)
            identities = sorted(set((r['A'], r['B']) for r in chosen))
            rows = []
            for a, b in identities:
                an, ar = friendly_model(a)
                bn, br = friendly_model(b)
                row = [tex_escape(an + ' - ' + bn), ar + '/' + br]
                for lead in (6, 12, 18, 24):
                    match = next((r for r in chosen if r['A'] == a and r['B'] == b and r['lead_hours'] == lead), None)
                    row.append('---' if match is None else f'{match["storm_mean"]:+.1f} [{match["storm_ci_low"]:+.1f}, {match["storm_ci_high"]:+.1f}]')
                rows.append(row)
            if rows:
                path = generated / f'revision_paired_{category}_{population}.tex'
                latex_table(path, tex_escape(category.capitalize() + ' comparisons, ' + population) + '. Entries are A minus B, equal-storm mean difference [95\\% paired storm-bootstrap interval] in km; negative favors A. Intervals are pointwise.',
                            f'tab:paired-{category}-{population}', ['A - B', 'RMW A/B', '6 h', '12 h', '18 h', '24 h'], rows)
                summary['tables'].append(str(path.relative_to(REV)))
    for lead in (6, 12, 18, 24):
        rows = []
        for c in [r for r in costs if r['population'] == 'test2022_2024' and r['lead_hours'] == lead]:
            name, rmw = friendly_model(c['model'])
            rows.append([tex_escape(name), rmw, str(c['parameters']) if c['parameters'] is not None else '---',
                         f'{c["seconds_mean"]:.1f}' if c['seconds_mean'] is not None else '---',
                         str(c['seed_count']), c['selected_epoch_range'], tex_escape(c['device_class'])])
        if rows:
            path = generated / f'revision_training_cost_{lead}h.tex'
            latex_table(path, f'Measured {lead} h training cost. Seconds are mean per fit (training plus validation for neural methods; 17-alpha tuning for Ridge). CPU/GPU times are not hardware-matched speed comparisons.',
                        f'tab:training-cost-{lead}', ['Method', 'RMW', 'Parameters', 'Seconds', 'Fits', 'Best epoch', 'Device'], rows)
            summary['tables'].append(str(path.relative_to(REV)))
    for population in ('test2022_2024', 'extension2025'):
        rows, accuracy_rows, policy_error_rows = [], [], []
        for r in branches:
            if r['population'] != population or r['statistic'] != 'gate_mean' or r['group_type'] != 'overall':
                continue
            name, rmw = friendly_model(r['model'])
            related = {b['statistic']: b for b in branches if b['population'] == population and b['model'] == r['model'] and b['lead_hours'] == r['lead_hours'] and b['group_type'] == 'overall'}
            def cell(stat):
                value = related.get(stat, {}).get('storm_mean')
                return '---' if value is None else f'{value:.3f}'
            rows.append([str(r['lead_hours']), tex_escape(name), rmw, cell('direct_error_km'), cell('cv_residual_error_km'),
                         cell('fused_error_km'), cell('gate_mean'), cell('fraction_strict_gate_prefers_lower_error_branch')])
            preference = related.get('fraction_strict_gate_prefers_lower_error_branch')
            if preference is not None and preference['storm_mean'] is not None:
                gain = related['strict_gate_accuracy_minus_always_cv_residual']
                gain_cell = f'{gain["storm_mean"]:+.3f}'
                if gain['storm_ci_low'] is not None:
                    gain_cell += f' [{gain["storm_ci_low"]:+.3f}, {gain["storm_ci_high"]:+.3f}]'
                accuracy_rows.append([str(r['lead_hours']), tex_escape(name), rmw,
                                      cell('fraction_strict_gate_prefers_lower_error_branch'),
                                      cell('fraction_always_cv_residual_prefers_lower_error_branch_on_strict_gate_support'),
                                      cell('fraction_always_direct_prefers_lower_error_branch_on_strict_gate_support'), gain_cell])
                policy_error_rows.append([str(r['lead_hours']), tex_escape(name), rmw,
                                          cell('fused_error_on_strict_gate_support_km'),
                                          cell('gate_induced_single_branch_error_on_strict_support_km'),
                                          cell('always_cv_residual_error_on_strict_gate_support_km'),
                                          cell('always_direct_error_on_strict_gate_support_km'),
                                          cell('best_single_branch_hindsight_error_on_strict_gate_support_km')])
        if rows:
            path = generated / f'revision_gate_branches_{population}.tex'
            latex_table(path, 'Measured branch diagnostics, ' + tex_escape(population) + '. Errors (km), gate values, and correct branch-preference fractions are storm-equal summaries. Gates exactly 0.5 are excluded from preference accuracy. These descriptive associations do not identify physical mechanisms.',
                        f'tab:gate-branches-{population}', ['h', 'Method', 'RMW', 'Direct', 'Residual', 'Fused', 'Gate', 'Preference'], rows)
            summary['tables'].append(str(path.relative_to(REV)))
        if accuracy_rows:
            path = generated / f'revision_gate_constant_policy_accuracy_{population}.tex'
            latex_table(path, 'Branch-choice accuracy against constant-choice references, ' + tex_escape(population) + '. All fractions use exactly the same non-neutral-gate seed/window support and storm-equal aggregation; tied branch errors count as correct for either choice. The last column gives gate accuracy minus always-residual accuracy [95\\% storm-bootstrap interval]. This checks descriptive incremental branch-choice value, not physical semantics.',
                        f'tab:gate-policy-accuracy-{population}', ['h', 'Method', 'RMW', 'Gate', 'Always residual', 'Always direct', 'Gain over residual [CI]'], accuracy_rows)
            summary['tables'].append(str(path.relative_to(REV)))
            path = generated / f'revision_gate_constant_policy_errors_{population}.tex'
            latex_table(path, 'Endpoint error (km) of branch policies on identical strict-gate support, ' + tex_escape(population) + '. Hard gate chooses one branch using whether the gate exceeds 0.5. Best single branch in hindsight chooses the lower true error per seed/window after observing the target; it is explicitly post hoc. It is not a lower bound on fused error: a blend can outperform both individual branches.',
                        f'tab:gate-policy-errors-{population}', ['h', 'Method', 'RMW', 'Fused', 'Hard gate', 'Always residual', 'Always direct', 'Best single in hindsight'], policy_error_rows)
            summary['tables'].append(str(path.relative_to(REV)))
    json_write(OUT / 'integration_summary.json', summary)
    json_write(generated / 'revision_table_manifest.json', dict(source='results/diagnostics/integration_summary.json', tables=summary['tables']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replicates', type=int, default=10000)
    parser.add_argument('--include-2025-neural', action='store_true', help='Only after every formal matrix and MLP holdout run is complete and sealed')
    args = parser.parse_args()
    assert args.replicates >= 1000
    if args.include_2025_neural:
        extension_is_sealed()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
                         'font.size': 9, 'axes.titlesize': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'legend.frameon': False})
    tracks, raw_index = raw_tracks()
    thresholds = {}
    for lead in range(1, 5):
        train = load_context(REV / f'data/controlled_12to1_lead{lead}/train')
        thresholds[str(lead * 6)] = dict(train_windows=len(train['keys']),
                                         turn_tertiles_deg=np.nanquantile(train['turn'], [1 / 3, 2 / 3]).tolist(),
                                         speed_change_tertiles_kmh=np.quantile(train['speed_change'], [1 / 3, 2 / 3]).tolist())
    json_write(OUT / 'training_only_thresholds.json', thresholds)
    errors, pairs, direction, branches, cases, omitted, included, overall_errors, costs = [], [], [], [], [], [], [], [], []
    for population in ('test2022_2024', 'extension2025'):
        for lead in range(1, 5):
            folder = REV / 'data' / (f'controlled_12to1_lead{lead}' if population == 'test2022_2024' else f'extension2025_lead{lead}')
            ctx = load_context(folder / 'test')
            ctx['target_status'] = np.asarray([raw_index[sid, int(date), int(tm)]['status']
                                                for sid, date, tm in zip(ctx['ids'], ctx['target_date'], ctx['target_time'])])
            raw_init_status = np.asarray([raw_index[sid, int(date), int(tm)]['status']
                                          for sid, date, tm in zip(ctx['ids'], ctx['init_date'], ctx['init_time'])])
            assert np.array_equal(raw_init_status, ctx['init_status']), 'Decoded numeric status disagrees with original snapshot'
            methods = ridge_models(lead, population, ctx)
            if population == 'test2022_2024' or args.include_2025_neural:
                methods.update(neural_models(lead, population, ctx, omitted))
                methods.update(neural_models(lead, population, ctx, omitted, family='mlp'))
            else:
                omitted.append(dict(population=population, lead_hours=lead * 6,
                                    reason='Neural/MLP extension arrays not read without --include-2025-neural and all-run completion checks'))
            groups = diagnostic_groups(ctx, thresholds[str(lead * 6)])
            for name, model in methods.items():
                included.append(dict(population=population, lead_hours=lead * 6, model=name, seeds=model['seeds'], paths=model['paths']))
                for kind, label, mask in groups:
                    stats = cluster(model['error'][mask], ctx['ids'][mask], args.replicates)
                    errors.append(dict(population=population, lead_hours=lead * 6, model=name, group_type=kind, group=label, **stats))
                    if kind == 'overall':
                        seed_means = model['seed_error'].mean(axis=1)
                        overall_errors.append(dict(population=population, lead_hours=lead * 6, model=name,
                                                   seed_count=len(seed_means), seeds=model['seeds'],
                                                   seed_window_sd=float(seed_means.std(ddof=1)) if len(seed_means) > 1 else 0.0,
                                                   **stats))
                if population == 'test2022_2024' and model['costs']:
                    entries = model['costs']
                    seconds = [float(c['seconds']) for c in entries if c['seconds'] is not None]
                    epochs = [int(c['selected_epoch']) for c in entries if c['selected_epoch'] is not None]
                    params = sorted({int(c['parameters']) for c in entries if c['parameters'] is not None})
                    assert len(params) <= 1, 'Parameter count differs within seed group'
                    devices = sorted({str(c['device']) for c in entries})
                    gpu = any('cuda' in d.lower() or 'nvidia' in d.lower() for d in devices)
                    costs.append(dict(population=population, lead_hours=lead * 6, model=name,
                                      parameters=params[0] if params else None, seed_count=len(entries),
                                      seconds_mean=float(np.mean(seconds)) if len(seconds) == len(entries) else None,
                                      seconds_total=float(np.sum(seconds)) if len(seconds) == len(entries) else None,
                                      selected_epoch_range=f'{min(epochs)}--{max(epochs)}' if epochs else '---',
                                      device_class='GPU' if gpu else 'CPU', device_names=devices,
                                      measured_runs=entries))
                for metric, values in directional_errors(model, ctx).items():
                    for kind, label, mask in [g for g in groups if g[0] in ('overall', 'year', 'historical_turn')]:
                        stats = cluster(values[mask], ctx['ids'][mask], args.replicates)
                        direction.append(dict(population=population, lead_hours=lead * 6, model=name, metric=metric, group_type=kind, group=label, **stats))
                measured_branches = dict(model['branches'])
                if measured_branches:
                    measured_branches['fused'] = model['error']
                branch_values = {branch + '_error_km': values for branch, values in measured_branches.items()}
                if model.get('gate') is not None:
                    gate = model['gate']
                    branch_values['gate_mean'] = gate.mean(axis=0)
                    bse = model['branch_seed_errors']
                    if 'direct' in bse and 'cv_residual' in bse:
                        branch_values.update(gate_policy_values(gate, bse['direct'], bse['cv_residual'], model['seed_error']))
                for statistic, values in branch_values.items():
                    for kind, label, mask in [g for g in groups if g[0] in ('overall', 'historical_turn', 'historical_acceleration_sign')]:
                        branches.append(dict(population=population, lead_hours=lead * 6, model=name, statistic=statistic,
                                             group_type=kind, group=label,
                                             **cluster(values[mask], ctx['ids'][mask], args.replicates)))
            comparisons = comparison_specifications(methods)
            for a, b in comparisons:
                difference = methods[a]['error'] - methods[b]['error']
                for kind, label, mask in groups:
                    stats = cluster(difference[mask], ctx['ids'][mask], args.replicates)
                    pairs.append(dict(population=population, lead_hours=lead * 6, A=a, B=b, group_type=kind, group=label, **stats))
            if lead == 4:
                chosen = 'ridge_rmw_off'
                seal = REV / 'results/controlled_matrix/holdout_selection_lock.json'
                if seal.exists():
                    selection = json.loads(seal.read_text(encoding='utf-8-sig'))['selected_per_lead_hours']['24']['selected_configuration']
                    if 'neural_' + selection in methods:
                        chosen = 'neural_' + selection
                cases.extend(plot_cases(population, ctx, methods, tracks, chosen))
            print(f'Analyzed {population} lead {lead * 6} h: {len(methods)} methods', flush=True)
    csv_write(OUT / 'subgroup_errors.csv', errors)
    csv_write(OUT / 'paired_cluster_bootstrap.csv', pairs)
    csv_write(OUT / 'along_cross_diagnostics.csv', direction)
    csv_write(OUT / 'branch_gate_diagnostics.csv', branches)
    csv_write(OUT / 'overall_errors.csv', overall_errors)
    csv_write(OUT / 'training_cost.csv', costs)
    json_write(OUT / 'case_selection.json', cases)
    figure_rmw(pairs)
    effect_figures(pairs)
    integration_artifacts(overall_errors, pairs, branches, costs, included, omitted, args.include_2025_neural, cases, errors, direction)
    protocol = dict(completed_utc=datetime.now(timezone.utc).isoformat(), replicates=args.replicates, bootstrap_seed=20260913,
                    differences='A minus B; negative means A improves. Fractions count storms with negative mean difference.',
                    uncertainty='95% percentile bootstrap, resampling complete storms. Both equal-storm and pooled-window estimands. No CI reported with fewer than two storms.',
                    seed_aggregation='Arithmetic mean of per-window errors across the three seeds, then storm resampling. Not an ensemble forecast; seeds are not independent storms.',
                    motion_definition='Project p[T-2],p[T-1],p[T] into the local tangent plane at p[T]. v0=q[T-1]-q[T-2], v1=q[T]-q[T-1]. Turn=atan2(abs(cross(v0,v1)),dot(v0,v1)) in degrees. Speed change=(norm(v1)-norm(v0))/6 in km/h per 6h step. Degenerate vector turn is undefined.',
                    groups='Lead-specific 1/3 and 2/3 quantiles fitted only on train windows; bins <q1, q1<=x<q2, >=q2. Acceleration sign uses +/-1e-6 km/h numerical tolerance. Status and yearly groups are descriptive.',
                    directional_definition='Predicted minus true endpoint displacement projected at initialization. Along axis is last observed 6h velocity; positive cross is left of that axis. This historical-heading reference differs from future-track-oriented metrics. Direction undefined for zero last speed. Compute signed and absolute errors per seed before averaging seeds.',
                    mechanism_limit='Gate statistics are descriptive branch associations and do not establish unique meteorological semantics. Always-residual, always-direct and gate-choice accuracies use the identical strict-gate seed/window support. Equal branch errors count as correct for either policy. Best-single-branch hindsight uses future errors after prediction; it is not a lower bound on fused error because blending can beat both individual branches.',
                    multiplicity='All subgroup confidence intervals are exploratory pointwise intervals, with no multiplicity adjustment; do not rank significance across many subgroups.',
                    case_rule='For 24h, use validation-selected complete neural configuration if available, otherwise Ridge RMW-off. Rank storms by mean error delta versus Ridge RMW-on; show best and worst. Select median-delta window within storm, ties earliest. Neural figure uses seed 42 endpoint, never mean coordinates presented as the average-error model. Lines beyond initialization are true observed future trajectory; predictions are single endpoints.',
                    included=included, omitted=omitted, branch_diagnostics_present=bool(branches),
                    gate_diagnostics_present=any(r['statistic'] == 'gate_mean' for r in branches),
                    extension_neural_analysis_enabled=args.include_2025_neural,
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    json_write(OUT / 'analysis_manifest.json', protocol)
    overall = [r for r in pairs if r['group_type'] == 'overall']
    json_write(OUT / 'overall_comparisons.json', overall)
    report = ['# Paired storm analysis and motion diagnostics', '',
              f'Differences below are A minus B, so negative values favor A. Intervals resample complete storms {args.replicates:,} times (seed 20260913); overlapping windows and training seeds are not independent units. Neural methods are included only once all three predeclared seeds have completed, matching identical storm/window keys and target labels. Errors are averaged across seeds before storm aggregation, not recomputed from averaged endpoints.', '',
              '| Population | Lead | A | B | Storm-equal Δ km [95% CI] | Improving storms |', '|---|---:|---|---|---:|---:|']
    for r in overall:
        report.append(f'| {r["population"]} | {r["lead_hours"]} h | {r["A"]} | {r["B"]} | {r["storm_mean"]:+.3f} [{r["storm_ci_low"]:+.3f}, {r["storm_ci_high"]:+.3f}] | {100 * r["fraction_storms_negative"]:.1f}% |')
    report += ['', 'Training-only thresholds and complete mathematical definitions are saved in `training_only_thresholds.json` and `analysis_manifest.json`. Year, initialization/target status, observed historical turn, and observed speed-change groups are in `subgroup_errors.csv`; matched subgroup differences and both bootstrap estimands are in `paired_cluster_bootstrap.csv`.', '',
               'Along/cross-track components use the last observed historical motion direction as reference. Signed bias and absolute component error are both reported. A long-lead positive along-track bias therefore means overshoot in that historical direction, not necessarily overspeed along a future turning track.', '',
               'Subgroup intervals are exploratory and pointwise without multiplicity correction. Small groups (especially the eight-storm 2025 24 h task) provide limited precision; no interval is given for a one-storm group. Gate and branch analyses are created only when actual saved neural outputs exist. Their association with lower branch error does not prove physical interpretability.', '',
               'The case-selection manifest documents the best and worst storms by mean relative error. Cases currently use the validation-selected complete neural model if its selection lock exists, otherwise explicitly use Ridge RMW-off against Ridge RMW-on. True future tracks are shown with dashed lines; each forecast is a single 24 h endpoint. No multi-step forecast path is implied.', '',
               f'Currently included Transformer groups: {sum(r["model"].startswith("neural_") for r in included)}; MLP groups: {sum(r["model"].startswith("mlp_") for r in included)}. Actual branch diagnostics available: {bool(branches)}; gate diagnostics: {any(r["statistic"] == "gate_mean" for r in branches)}. This script does not reconstruct missing historical B4 predictions.', '',
               'Rerun after the complete matrix and MLP holdout queue finish: `python journal_revision_20260913/experiments/analyze_revision.py --replicates 10000 --include-2025-neural`. Without the extension flag, 2025 neural/MLP arrays are not read.', '',
               'Compact integration summary: `integration_summary.json`. Measured main tables, head/RMW/simple-baseline intervals, training costs, and gate/branch tables are written to `manuscript/generated/revision_*.tex`; manuscript prose is never changed by this script.', '']
    (OUT / 'report.md').write_text('\n'.join(report), encoding='utf-8')


if __name__ == '__main__':
    main()
