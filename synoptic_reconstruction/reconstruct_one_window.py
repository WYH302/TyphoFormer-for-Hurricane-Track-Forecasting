"""Rebuild one saved test window and its forecast error; no fitting is performed."""
import argparse, json
import numpy as np
import pandas as pd
import prepare_revision as p

def main():
    cli=argparse.ArgumentParser()
    cli.add_argument('--storm', help='Storm ID; default: first saved test prediction')
    cli.add_argument('--initialization', type=int, help='Initialization as Unix epoch hour')
    args=cli.parse_args()
    predictions=np.load(p.HERE/'runs/residual_seed42/test_predictions.npz')
    ids=predictions['storm_id']; times=predictions['window']
    mask=np.ones(len(ids),bool)
    if args.storm: mask &= ids==args.storm
    if args.initialization is not None: mask &= times==args.initialization
    matched=np.flatnonzero(mask)
    if not len(matched): raise ValueError('No saved prediction matches the requested identifiers')
    ix=int(matched[0]); sid=str(ids[ix]); hour=int(times[ix])
    meta=json.loads((p.HERE/'data/main/metadata.json').read_text(encoding='utf-8'))
    expected_hash=meta['hurdat2_sha256']; assert p.sha(p.SOURCE)==expected_hash
    raw=p.LIB.parse_hurdat2(str(p.SOURCE)); group=raw[raw.typhoon_id==sid]
    rebuilt=next(r for r in p.make_rows(group,{'test':{sid}}) if r['window']==hour)
    saved=np.load(p.HERE/f'data/main/test/{sid}_{hour}.npy',allow_pickle=True).item()
    moments={k:np.asarray(v,np.float32) for k,v in meta['stats'].items()}
    values=rebuilt['input_raw']
    standardized=((np.where(np.isfinite(values),values,moments['x_mean'])-moments['x_mean'])/moments['x_std']).astype(np.float32)
    np.testing.assert_array_equal(standardized,saved['input_num'])
    np.testing.assert_array_equal(rebuilt['target_disp_raw_km'],saved['target_disp_raw_km'])
    np.testing.assert_array_equal(rebuilt['future_latlon'],predictions['future_latlon'][ix])
    assert rebuilt['recorded_future_landfall']==saved['recorded_future_landfall']
    keep=[i for i,n in enumerate(meta['feature_names']) if n!='rmax']
    assert standardized[:,keep].shape==(12,39)
    a=np.deg2rad(predictions['pred_latlon'][ix,0].astype(np.float64))
    b=np.deg2rad(rebuilt['future_latlon'][0].astype(np.float64))
    hav=np.sin((a[0]-b[0])/2)**2+np.cos(a[0])*np.cos(b[0])*np.sin((a[1]-b[1])/2)**2
    error=float(2*6371*np.arcsin(np.sqrt(np.clip(hav,0,1))))
    supplied=float(predictions['pred_error_km'][ix,0])
    assert abs(error-supplied)<0.01,(error,supplied)
    out=p.HERE/'generated';out.mkdir(exist_ok=True)
    table=pd.DataFrame(standardized[:,keep],columns=[meta['feature_names'][i] for i in keep])
    table.insert(0,'timestamp_UTC',[str(np.datetime64(int(t),'h')) for t in rebuilt['history_timestamp_hours']])
    table.to_csv(out/'one_window_standardized_features.csv',index=False)
    report=dict(status='passed',window_id=rebuilt['window_id'],source_sha256=expected_hash,
        policy=meta['feature_policy'],input_shape_RMW_off=[12,39],all_40_saved_channels_exact=True,
        history_timestamps_epoch_hours=rebuilt['history_timestamp_hours'].tolist(),
        target_timestamps_epoch_hours=rebuilt['future_timestamp_hours'].tolist(),
        recorded_future_landfall=rebuilt['recorded_future_landfall'],
        prediction_file='runs/residual_seed42/test_predictions.npz',prediction_index=ix,
        prediction_latlon_degrees=predictions['pred_latlon'][ix,0].tolist(),
        observed_target_latlon_degrees=rebuilt['future_latlon'][0].tolist(),
        recomputed_haversine_km=error,saved_float32_error_km=supplied,
        absolute_difference_km=abs(error-supplied),error_tolerance_km=0.01,
        features_file='generated/one_window_standardized_features.csv')
    p.dump(out/'one_window_reconstruction.json',report)
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
