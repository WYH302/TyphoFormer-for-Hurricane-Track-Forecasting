"""Prepare an explicitly retrospective, 24 h synoptic-first sensitivity.

Run with Python + NumPy + pandas + scikit-learn. Original artifacts are read only.
"""
from pathlib import Path
import importlib.util
import json
import hashlib
import zipfile
from types import SimpleNamespace
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REV = HERE.parents[1]
SPEC = importlib.util.spec_from_file_location('original_prepare', REV/'data/prepare_frozen.py')
LIB = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LIB)
SOURCE = REV/'data/hurdat2_frozen2024_plus2025.txt'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def validate(frame):
    if frame.duplicated(['typhoon_id', 'date', 'time']).any():
        raise ValueError('Duplicate or conflicting storm timestamp; no row-order resolution.')
    if not np.isfinite(frame[['latitude','longitude']].to_numpy()).all():
        raise ValueError('Nonfinite coordinates')
    return frame.sort_values(['typhoon_id','date','time']).copy()


def storm_arrays(frame):
    """Select synoptic records first; use actual elapsed hours for speed.

    accel6 is the saved historical name of a speed increment (km/h), not
    acceleration divided by time. Heading/turn channels remain dimensionless.
    """
    validate(frame)
    selected = frame[frame.time.isin(LIB.STANDARD_TIMES)].copy().reset_index(drop=True)
    arrays = LIB.build_storm_arrays(selected, np.zeros((len(selected),1),np.float32))
    if len(selected)>1:
        hours = np.diff(arrays['timestamp_hours'])
        if (hours <= 0).any():
            raise ValueError('Nonincreasing timestamp')
        dx = arrays['x_num'][:,LIB.FEATURE_NAMES.index('dx6')]
        dy = arrays['x_num'][:,LIB.FEATURE_NAMES.index('dy6')]
        speed = np.zeros(len(selected), np.float32)
        speed[1:] = np.hypot(dx[1:],dy[1:])/hours
        arrays['x_num'][:,LIB.FEATURE_NAMES.index('speed6')] = speed
        arrays['x_num'][:,LIB.FEATURE_NAMES.index('accel6')] = np.r_[0,np.diff(speed)]
    return arrays


def accepted_indices(arrays):
    ts = arrays['timestamp_hours']
    return [j for j in range(max(0,len(ts)-15)) if np.all(np.diff(ts[j:j+16])==6)]


def make_rows(frame, splits):
    rows=[]
    for sid,group in validate(frame).groupby('typhoon_id'):
        split=next((k for k,v in splits.items() if sid in v),None)
        if split is None:
            continue
        arr=storm_arrays(group)
        event_rows=group.loc[group.record_identifier=='L']
        event_dates=pd.to_datetime(event_rows.date.astype(str),format='%Y%m%d')
        event_dates=event_dates+pd.to_timedelta(event_rows.time//100,unit='h')+pd.to_timedelta(event_rows.time%100,unit='m')
        events=event_dates.astype('datetime64[ns]').astype('int64').to_numpy()/3.6e12
        for j in accepted_indices(arr):
            ts=arr['timestamp_hours'][j:j+16]
            history=arr['coords'][j:j+12]
            future=arr['coords'][j+15:j+16]
            raw=LIB.latlon_to_disp_km(future,history[-1])
            rows.append(dict(storm_id=sid,storm_year=int(sid[4:8]),split=split,
                window=int(ts[11]),window_id=f'{sid}:{int(ts[11])}:24h:synoptic_actual_dt_v1',
                input_raw=arr['x_num'][j:j+12].copy(),history_latlon=history,
                origin=history[-1],future_latlon=future,target_disp_raw_km=raw,
                history_timestamp_hours=ts[:12],future_timestamp_hours=ts[-1:],
                recorded_future_landfall=bool(np.any((events>ts[11])&(events<=ts[15])))))
    return rows


def fit_stats(rows):
    train=[r for r in rows if r['split']=='train']
    x=np.concatenate([r['input_raw'] for r in train])
    y=np.concatenate([r['target_disp_raw_km'] for r in train])
    with np.errstate(invalid='ignore'):
        mean=np.nanmean(x,axis=0);std=np.nanstd(x,axis=0)
    mean=np.where(np.isfinite(mean),mean,0).astype(np.float32)
    std=np.where(np.isfinite(std)&(std>1e-6),std,1).astype(np.float32)
    ys=y.std(0);ys=np.where(ys>1e-6,ys,1).astype(np.float32)
    return dict(x_mean=mean,x_std=std,target_mean=y.mean(0).astype(np.float32),target_std=ys)


def main():
    lock=HERE/'design_lock.json'
    if lock.exists():
        raise FileExistsError('Preparation exists; do not overwrite a locked sensitivity.')
    HERE.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(HERE/'before_editing_sources.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for journal in ('PCE','ESIN'):
            root=REV/f'submission_{journal}'
            for p in root.rglob('*'):
                if p.is_file() and p.suffix in ('.tex','.bib','.cls','.bst','.md','.txt','.json'):
                    archive.write(p,p.relative_to(REV))
    frame=validate(LIB.parse_hurdat2(str(SOURCE)))
    frame=frame[frame.storm_year.between(2004,2025)].copy()
    splits=LIB.storm_splits(frame,SimpleNamespace(train_years='2004-2021',test_years='2022-2024',split_seed=42,val_ratio=.1))
    splits['extension']=set(frame.loc[frame.storm_year==2025,'typhoon_id'])
    rows=make_rows(frame,splits)
    stats=fit_stats(rows)
    meta=json.loads((REV/'data/controlled_12to1_lead4/metadata.json').read_text(encoding='utf-8'))
    meta.update(source='Frozen Atlantic HURDAT2; synoptic-first actual-dt sensitivity',
        hurdat2_sha256=sha(SOURCE),history_hours=66,history_span_hours=66,
        feature_policy='Select standard times first; speed uses actual adjacent selected timestamp delta; accel6 is speed increment.',
        stats={k:v.tolist() for k,v in stats.items()},
        split_storm_ids={k:sorted(v) for k,v in splits.items()},
        post_test_visibility=True,normalization_fit_split='train')
    counts={k:sum(r['split']==k for r in rows) for k in splits}
    meta['splits']={k:dict(samples=counts[k],storms=len({r['storm_id'] for r in rows if r['split']==k})) for k in splits}
    for r in rows:
        x=r.pop('input_raw')
        r['input_num']=((np.where(np.isfinite(x),x,stats['x_mean'])-stats['x_mean'])/stats['x_std']).astype(np.float32)
        r['target_disp']=((r['target_disp_raw_km']-stats['target_mean'])/stats['target_std']).astype(np.float32)
        root=HERE/'data'/('extension' if r['split']=='extension' else 'main')
        folder=root/('test' if r['split']=='extension' else r['split'])
        folder.mkdir(parents=True,exist_ok=True)
        np.save(folder/f"{r['storm_id']}_{r['window']}.npy",r)
    for sub in ('main','extension'):
        dump(HERE/'data'/sub/'metadata.json',meta)
    dump(HERE/'window_manifest.json',[{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in r.items() if k not in ('input_num','target_disp','target_disp_raw_km','history_latlon','future_latlon','origin')} for r in rows])
    dump(lock,dict(status='prepared',source_sha256=sha(SOURCE),preparation_sha256=sha(__file__),
        scope='Post-review sensitivity after 2022-2025 results had been examined. Not a first blind evaluation.',
        designs=dict(neural_heads=['direct','residual'],seeds=[42,123,2024],rmw='off',lead_hours=24,
            epochs=40,encoder='original full-size',selection='minimum validation endpoint error',
            ridge_alphas=np.logspace(-4,10,29).tolist()),
        split_seed=42,counts=meta['splits'],bootstrap=dict(draws=10000,seed=20260913,unit='storm'),
        reason='Check corrected training and evaluation without split-dependent RMW reporting.'))
    print(json.dumps(meta['splits']))


if __name__=='__main__':
    main()
