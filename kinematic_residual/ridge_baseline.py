"""Frozen numeric Ridge residual-on-CV comparisons, with/without RMW.

One SVD per training design computes the same 17 solutions as sklearn's SVD
Ridge solver; a full-data sklearn fit checks the first design numerically.
All 8 alphas are selected and persisted before any 2025 evaluation is loaded.
"""
import ast
import csv
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType

import numpy as np
from scipy import linalg
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
REV = ROOT / 'journal_revision_20260913'
OUT = REV / 'results/ridge'
REFERENCE = ROOT / 'code_experiments/TyphoFormerPlus/train_simple_residual_baselines.py'
ALPHAS = np.logspace(-4, 4, 17)


def reference_functions():
    """Reuse the checked numeric functions without loading unrelated Torch code."""
    wanted = {'wrap_lon_delta', 'latlon_to_disp_np', 'disp_to_latlon_np', 'haversine_np',
              'constant_velocity_disp', 'load_records', 'standardize', 'metrics_from_raw'}
    parsed = ast.parse(REFERENCE.read_text(encoding='utf-8-sig'))
    selected = ast.Module(body=[n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name in wanted], type_ignores=[])
    module = ModuleType('historical_numeric_functions')
    module.__dict__.update(np=np, Path=Path, EARTH_RADIUS_KM=6371.0)
    exec(compile(selected, str(REFERENCE), 'exec'), module.__dict__)
    return module


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def model_x(data, keep, feature_count):
    full = data['x'].reshape(len(data['x']), 12, feature_count)
    return full[:, :, keep].reshape(len(full), -1)


def fit_design(train, val, keep, feature_count, lib, verify):
    train_x, val_x = model_x(train, keep, feature_count), model_x(val, keep, feature_count)
    (standardized, mean, std) = lib.standardize(train_x, train_x, val_x)
    x_train, x_val = standardized
    y_train = (train['y_raw'] - train['cv_raw']).reshape(len(train_x), -1)
    x_offset = x_train.mean(axis=0)
    y_offset = y_train.mean(axis=0)
    centered_x, centered_y = x_train - x_offset, y_train - y_offset
    started = time.perf_counter()
    u, singular, vt = linalg.svd(centered_x, full_matrices=False)
    projected_y = u.T @ centered_y
    valid = singular > 1e-15
    best, tuning, verified = None, [], None
    for alpha in ALPHAS:
        d = np.zeros((len(singular), 1), dtype=x_train.dtype)
        s = singular[valid]
        d[valid] = (s / (s ** 2 + float(alpha)))[:, None]
        coef = (vt.T @ (d * projected_y)).T
        intercept = y_offset - x_offset @ coef.T
        residual = x_val @ coef.T + intercept
        pred = val['cv_raw'] + residual.reshape(val['y_raw'].shape)
        metrics = lib.metrics_from_raw(pred, val)
        tuning.append(dict(alpha=float(alpha), val_fde_km=metrics['fde'], val_storm_equal_km=metrics['storm_mean']))
        if verify and float(alpha) == 1.0:
            reference = Ridge(alpha=1.0, fit_intercept=True, solver='svd').fit(x_train, y_train)
            ref_pred = reference.predict(x_val)
            max_difference = float(np.max(np.abs(ref_pred - residual)))
            assert np.allclose(ref_pred, residual, rtol=1e-5, atol=1e-3), max_difference
            verified = dict(alpha=1.0, full_training_design=True,
                            max_validation_residual_difference_km=max_difference,
                            rtol=1e-5, atol_km=1e-3, passed=True)
        if best is None or metrics['fde'] < best['val']['fde']:
            best = dict(alpha=float(alpha), val=metrics, coef=coef.copy(), intercept=intercept.copy())
    best.update(x_mean=mean, x_std=std, keep=np.asarray(keep), tuning=tuning,
                fit_seconds=time.perf_counter() - started, equivalence_check=verified)
    return best


def predict(model, data, feature_count):
    x = model_x(data, model['keep'], feature_count)
    normalized = (x - model['x_mean']) / model['x_std']
    residual = normalized @ model['coef'].T + model['intercept']
    return data['cv_raw'] + residual.reshape(data['y_raw'].shape)


def save_predictions(path, pred, data, lib):
    latlon = lib.disp_to_latlon_np(pred, data['origin'])
    err = lib.haversine_np(latlon, data['future'])[:, -1]
    np.savez_compressed(path, pred_raw_km=pred, pred_disp_km=pred, pred_latlon=latlon,
                        target_raw_km=data['y_raw'], future_latlon=data['future'],
                        cv_raw_km=data['cv_raw'], origin=data['origin'], error_km=err,
                        storm_ids=np.asarray(data['storm_ids']), storm_id=np.asarray(data['storm_ids']),
                        storm_year=data['storm_year'], window=data['window'],
                        filenames=np.asarray([Path(p).name for p in data['files']]))
    return lib.metrics_from_raw(pred, data)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    lib = reference_functions()
    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    reference_sha = hashlib.sha256(REFERENCE.read_bytes()).hexdigest()
    policy = dict(started_utc=datetime.now(timezone.utc).isoformat(), source_sha256=source_sha,
                  reference_sha256=reference_sha, alpha_grid=ALPHAS.tolist(),
                  selection='Minimum validation window-mean geodesic endpoint error; first alpha wins exact ties.',
                  input='12 x numeric features, no text or analog features; remove rmax in every step for RMW-off.',
                  standardization='Original train-only prepared statistics, then train-only column standardization of flattened numeric input.',
                  cv='Historical constant velocity helper: last displacement projected at penultimate location times lead steps.',
                  extension_rule='All leads and RMW modes selected and persisted before loading any 2025 test records. No tuning after test evaluation.',
                  user_confirmation='Parent reports user confirmed 2025 had not previously been used for development or tuning.')
    write_json(OUT / 'protocol_lock.json', policy)
    designs, frozen = {}, []
    with threadpool_limits(limits=4):
        for lead in range(1, 5):
            folder = REV / f'data/controlled_12to1_lead{lead}'
            metadata = json.loads((folder / 'metadata.json').read_text(encoding='utf-8-sig'))
            names = metadata['feature_names']
            train = lib.load_records(folder / 'train', [lead])
            val = lib.load_records(folder / 'val', [lead])
            for rmw in ('on', 'off'):
                label = f'lead{lead}_rmw_{rmw}'
                keep = [i for i, name in enumerate(names) if rmw == 'on' or name != 'rmax']
                model = fit_design(train, val, keep, len(names), lib, lead == 1 and rmw == 'on')
                model_path = OUT / f'{label}_model.npz'
                np.savez_compressed(model_path, **{k: model[k] for k in ('coef', 'intercept', 'x_mean', 'x_std', 'keep')},
                                    alpha=model['alpha'], feature_names=np.asarray([names[i] for i in keep]))
                selection = dict(label=label, lead_hours=lead * 6, rmw=rmw, alpha=model['alpha'],
                                 val=model['val'], tuning=model['tuning'], fit_seconds=model['fit_seconds'],
                                 input_dim=12 * len(keep), equivalence_check=model['equivalence_check'],
                                 model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
                                 metadata_sha256=hashlib.sha256((folder / 'metadata.json').read_bytes()).hexdigest())
                write_json(OUT / f'{label}_selection.json', selection)
                designs[label] = (model, metadata, lead, rmw)
                frozen.append(selection)
                print(f'Selected {label}: alpha={model["alpha"]:g}, val={model["val"]["fde"]:.4f} km', flush=True)
        lock = dict(frozen_utc=datetime.now(timezone.utc).isoformat(), source_sha256=source_sha,
                    all_eight_models_selected_before_2025_loading=True, designs=frozen)
        write_json(OUT / 'all_selections_frozen.json', lock)
        summary = []
        for label, (model, metadata, lead, rmw) in designs.items():
            extension_folder = REV / f'data/extension2025_lead{lead}'
            ext_metadata = json.loads((extension_folder / 'metadata.json').read_text(encoding='utf-8-sig'))
            assert metadata['feature_names'] == ext_metadata['feature_names']
            for key in metadata['stats']:
                assert np.array_equal(metadata['stats'][key], ext_metadata['stats'][key]), (lead, key)
            for population, folder in [('test2022_2024', REV / f'data/controlled_12to1_lead{lead}'),
                                       ('extension2025', extension_folder)]:
                data = lib.load_records(folder / 'test', [lead])
                expected_years = (2025,) if population == 'extension2025' else (2022, 2023, 2024)
                assert set(data['storm_year']).issubset(expected_years)
                pred = predict(model, data, len(metadata['feature_names']))
                metrics = save_predictions(OUT / f'{label}_{population}_predictions.npz', pred, data, lib)
                cv_metrics = lib.metrics_from_raw(data['cv_raw'], data)
                result = dict(label=label, population=population, lead_hours=lead * 6, rmw=rmw,
                              alpha=model['alpha'], metrics=metrics, cv_metrics=cv_metrics,
                              extension_statistics_identical_to_original_train=True)
                write_json(OUT / f'{label}_{population}_metrics.json', result)
                summary.append(dict(label=label, population=population, lead_hours=lead * 6, rmw=rmw,
                                    alpha=model['alpha'], windows=metrics['windows'], storms=metrics['storms'],
                                    window_mean_km=metrics['fde'], storm_equal_mean_km=metrics['storm_mean'],
                                    cv_window_mean_km=cv_metrics['fde'], fit_seconds=model['fit_seconds']))
                print(f'Evaluated {label} {population}: {metrics["fde"]:.4f} km / {metrics["storm_mean"]:.4f} storm-equal', flush=True)
    seconds = time.perf_counter() - start
    with (OUT / 'summary.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    write_json(OUT / 'run_summary.json', dict(completed_utc=datetime.now(timezone.utc).isoformat(),
                                              elapsed_seconds=seconds, results=summary,
                                              all_extension_stats_equal=True,
                                              b4_paired_comparison_completed=False))
    report = ['# Ridge residual-on-CV: current frozen revision', '',
              'These are newly fitted numeric-only Ridge results. Historical publication values are not substituted. RMW-off removes the RMW coordinate at every historical step before fitting; no text or analog inputs are used in either mode. Both modes are independently retrained and choose alpha solely on historical validation storms.', '',
              'One SVD per design evaluates the original 17-alpha grid. A full-design sklearn SVD Ridge fit verifies the optimized solve on lead 1 / RMW-on / alpha 1. Every selected model and its checksum is frozen in `all_selections_frozen.json` before loading 2025 evaluation records. All prepared extension normalization statistics must exactly equal historical train statistics.', '',
              '| Population | Lead | RMW | Alpha | Window mean km | Storm-equal mean km |',
              '|---|---:|---|---:|---:|---:|']
    for row in summary:
        report.append(f'| {row["population"]} | {row["lead_hours"]} h | {row["rmw"]} | {row["alpha"]:g} | {row["window_mean_km"]:.3f} | {row["storm_equal_mean_km"]:.3f} |')
    report += ['', f'Total elapsed wall time: {seconds:.1f} seconds (four BLAS threads). Models, tuning curves, per-window NPZ predictions, per-storm mean errors, and CSV summaries are saved in this directory.', '',
               'This completes the Ridge RMW sensitivity comparison and produces compatible per-window predictions for later paired storm-level analysis. It does not complete a comparison against historical B4 because its original predictions/checkpoints are not supplied here. It does not establish significance from mean differences alone. No test result changes alpha selection or model configuration.', '']
    (OUT / 'report.md').write_text('\n'.join(report), encoding='utf-8')
    print(f'Completed in {seconds:.1f} seconds', flush=True)


if __name__ == '__main__':
    main()
