"""Post-review 24h cohort expansion; original training statistics stay frozen."""
from pathlib import Path
import importlib.util
import json
import hashlib
from datetime import datetime, timezone
import numpy as np

HERE = Path(__file__).resolve().parent
REV = HERE.parents[1]
RAW = REV / 'data/hurdat2_frozen2024_plus2025.txt'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def dump(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def main():
    lock = HERE / 'design_lock.json'
    if lock.exists():
        raise FileExistsError('Preserve prior preparation; use a separate directory for changed rules.')
    weights = sorted((REV / 'results/controlled_matrix/runs').glob('lead4_*_rmw_*_seed*/best.pt'))
    weights = [p for p in weights if any(f'lead4_{head}_rmw_' in p.parent.name for head in ('direct','residual','adaptive')) and 'aux' not in p.parent.name]
    assert len(weights) == 18
    weights += sorted((REV / 'results/ridge').glob('lead4_rmw_*_model.npz'))
    assert len(weights) == 20
    dump(lock, {
        'declared_utc': datetime.now(timezone.utc).isoformat(),
        'analysis_status': 'Post-review sensitivity after original test and 2025 results were visible; not a new blind test.',
        'selection': 'All strict 12-history plus four-future standard six-hour timestamps; no interpolation; only evaluation population changes.',
        'profiles': {
            'raw_features': 'Frozen original raw-neighbor derived features computed before selecting standard records; exact preservation on original accepted windows; isolates population expansion with existing feature algorithm.',
            'synoptic_features': 'Same feature-building function applied after selecting standard records; recomputes motion features, including possible boundary changes in original windows. Training normalization and all weights remain frozen.'},
        'comparisons': 'CV and Ridge-on/off; A/B/D heads on/off, seeds42/123/2024. No favorable configuration selection. Expanded-alpha Ridge evaluated separately when available.',
        'subsets': ['original','added','combined','recorded_future_landfall'],
        'estimands': 'Window mean and equal-storm mean endpoint km; 10000 paired whole-storm resamples seed20260913; no multiple-testing adjustment; count distinct storms.',
        'landfall': 'Raw L records in (initialization,target], including asynoptic records; no claim of predicting landfall location/time.',
        'no_training_or_reselection': True, 'raw_sha256': sha(RAW),
        'preparation_sha256': sha(Path(__file__)),
        'frozen_model_sha256': {p.relative_to(REV).as_posix(): sha(p) for p in weights}})
    spec = importlib.util.spec_from_file_location('frozen_prepare', REV / 'data/prepare_frozen.py')
    lib = importlib.util.module_from_spec(spec); spec.loader.exec_module(lib)
    meta = json.loads((REV / 'data/controlled_12to1_lead4/metadata.json').read_text(encoding='utf-8'))
    stats = {k: np.asarray(v, dtype=np.float32) for k,v in meta['stats'].items()}
    df = lib.parse_hurdat2(str(RAW))
    df = df[df.storm_year.between(2022,2025)].copy().reset_index(drop=True)
    text_dummy = np.zeros((len(df),1),np.float32)  # text is not consumed in any frozen numerical model
    audit = json.loads((REV / 'audit/landfall_coverage_audit.json').read_text(encoding='utf-8'))
    manifests = {}
    original = {}
    for period, dirname in [('test2022_2024','controlled_12to1_lead4'),('extension2025','extension2025_lead4')]:
        for p in (REV/'data'/dirname/'test').glob('*.npy'):
            row=np.load(p,allow_pickle=True).item()
            original[(str(row['storm_id']),int(row['history_timestamp_hours'][-1]))]=row
    for key,p in audit['periods'].items():
        period='test2022_2024' if key=='2022-2024' else 'extension2025'
        manifests[period]=p['windows']
    arrays={}
    for sid,group in df.groupby('typhoon_id'):
        raw = lib.build_storm_arrays(group,text_dummy)
        syn=lib.build_storm_arrays(group[group.time.isin(lib.STANDARD_TIMES)],text_dummy)
        arrays[sid]=(raw,syn)
    checks=[]; rows_manifest=[]
    for period, windows in manifests.items():
        changed_original=0; max_change=0.; accepted_count=0
        for profile in ('raw_features','synoptic_features'):
            folder=HERE/'data'/profile/period
            folder.mkdir(parents=True,exist_ok=True)
            dump(folder/'metadata.json', {**meta,'test_years':period,'sample_policy':profile,
                 'frozen_training_statistics':True,'history_span_hours':66,
                 'scope':'Evaluation-only; original train/validation directories remain in frozen data root.'})
            for rec in windows:
                sid=rec['storm_id']; raw,syn=arrays[sid]
                init=int(np.datetime64(rec['initialization'],'h').astype(np.int64))
                ts=np.arange(init-66,init+25,6,dtype=np.int64)
                indices=np.asarray([np.flatnonzero(raw['timestamp_hours']==t)[0] for t in ts])
                syn_indices=np.asarray([np.flatnonzero(syn['timestamp_hours']==t)[0] for t in ts])
                assert np.all(np.diff(raw['timestamp_hours'][indices])==6)
                src=raw if profile=='raw_features' else syn
                ix=indices if profile=='raw_features' else syn_indices
                history=src['coords'][ix[:12]]; future=src['coords'][ix[-1:]]
                x=src['x_num'][ix[:12]]
                x=np.where(np.isfinite(x),x,stats['x_mean'])
                x=((x-stats['x_mean'])/stats['x_std']).astype(np.float32)
                yraw=lib.latlon_to_disp_km(future,history[-1])
                row={'storm_id':sid,'storm_year':int(sid[4:8]),'window':int(indices[0]),
                     'input_num':x,'history_latlon':history,'future_latlon':future,'origin':history[-1],
                     'target_disp_raw_km':yraw,'target_disp':((yraw-stats['target_mean'])/stats['target_std']).astype(np.float32),
                     'history_timestamp_hours':ts[:12],'future_timestamp_hours':ts[-1:],
                     'accepted_original':rec['accepted_original'],
                     'recorded_future_landfall':rec['L_forecast_open_left_closed_right']>0}
                if rec['accepted_original']:
                    old=original[(sid,init)]
                    for k in ('window','history_latlon','future_latlon','origin','target_disp_raw_km','target_disp'):
                        assert np.array_equal(old[k],row[k]),(sid,init,k)
                    if profile=='raw_features':
                        assert np.array_equal(old['input_num'],x),(sid,init,'input')
                        accepted_count+=1
                    else:
                        delta=np.abs(old['input_num']-x)
                        changed_original+=int(np.any(delta>0)); max_change=max(max_change,float(delta.max()))
                out=folder/'test'/f'{sid}_{int(indices[0]):05d}.npy'
                out.parent.mkdir(exist_ok=True);np.save(out,row)
                if profile=='raw_features':
                    rows_manifest.append({**rec,'period':period,'window':int(indices[0]),'initialization_epoch_hour':init})
            assert len(list((folder/'test').glob('*.npy')))==len(windows)
        checks.append({'period':period,'combined_windows':len(windows),'original_windows':accepted_count,
                       'added_windows':len(windows)-accepted_count,'raw_profile_original_inputs_exact':True,
                       'synoptic_profile_changed_original_inputs':changed_original,
                       'maximum_changed_original_standardized_feature':max_change})
    assert len(rows_manifest)==1092
    dump(HERE/'window_manifest.json',rows_manifest)
    dump(HERE/'preparation_checks.json',{'status':'passed','checks':checks,'raw_sha256':sha(RAW),
         'original_training_or_test_files_modified':False})
    print(json.dumps(checks))


if __name__=='__main__':main()
