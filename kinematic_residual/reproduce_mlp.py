"""One-command entry for the preserved training runner and documented finalizer.

The frozen training source has a post-fit storm-ID parsing defect. Preserve it
for provenance, accept only that exact known failure after all 24 selections
are sealed, and evaluate with the corrected explicit-year finalizer. No other
failure is accepted, and no fitting is repeated for a complete evidence set.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def check_sealed(out):
    lock = read(out / 'all_selections_frozen.json')
    assert lock['all_24_models_selected_before_2025_loading']
    assert len(lock['selections']) == 24
    for entry in lock['selections']:
        run = out / 'runs' / entry['name']
        assert read(run / 'status.json')['completed_epochs'] == 40
        assert hashlib.sha256((run / 'best.pt').read_bytes()).hexdigest() == entry['checkpoint_sha256']
    return lock['selections']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=HERE.parent / 'data')
    parser.add_argument('--output', type=Path, default=HERE.parent / 'results/mlp_residual')
    args = parser.parse_args()
    out = args.output.resolve()
    status_path = out / 'status.json'
    if status_path.exists() and read(status_path).get('status') == 'completed':
        entries = check_sealed(out)
        for entry in entries:
            for folder in ('runs', 'holdout2025'):
                run = out / folder / entry['name']
                assert read(run / 'status.json')['test_evaluated']
                assert (run / 'test_predictions.npz').is_file()
        print('Existing 24 sealed MLP fits and 48 forecast artifacts verified; no rerun.')
        return
    out.mkdir(parents=True, exist_ok=True)
    arguments = ['--data-root', str(args.data_root.resolve()), '--output', str(out)]
    log = out / 'reproduction_training.log'
    with log.open('wb') as stream:
        result = subprocess.run([sys.executable, str(HERE / 'mlp_residual_baseline.py'), *arguments],
                                stdout=stream, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        message = log.read_text(encoding='utf-8', errors='replace')
        if 'int(key[0][-4:])' not in message or 'ValueError: invalid literal for int() with base 10:' not in message:
            raise RuntimeError(f'Training failed for an unrecognized reason; inspect {log}')
        check_sealed(out)
        print('All 24 fits sealed. Applying the documented evaluation-only year-field correction.')
    subprocess.run([sys.executable, str(HERE / 'mlp_residual_finalize.py'), *arguments], check=True)
    assert read(status_path)['status'] == 'completed'


if __name__ == '__main__':
    main()
