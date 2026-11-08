"""Audit 1980--2003 coverage and write a traceable window-level manifest."""
import json
import numpy as np
import pandas as pd
import prepare_revision as p

def main():
    raw=p.LIB.parse_hurdat2(str(p.SOURCE));raw=raw[raw.storm_year.between(1980,2003)]
    manifest=[]
    for sid,group in raw.groupby('typhoon_id',sort=True):
        group=p.validate(group)
        ts=group._timestamp_hours.to_numpy(); times=group.time.to_numpy()
        original={int(ts[j+11]) for j in range(max(0,len(group)-15))
            if np.all(np.isin(times[j:j+16],list(p.LIB.STANDARD_TIMES))) and np.all(np.diff(ts[j:j+16])==6)}
        for row in p.make_rows(group,{'historical':{sid}}):
            manifest.append(dict(window_id=row['window_id'],storm_id=sid,
                initialization_epoch_hour=row['window'],target_epoch_hour=int(row['future_timestamp_hours'][0]),
                history_first_epoch_hour=int(row['history_timestamp_hours'][0]),
                accepted_original=row['window'] in original,recorded_future_landfall=row['recorded_future_landfall'],
                source_sha256=p.sha(p.SOURCE)))
    frame=pd.DataFrame(manifest)
    original=frame[frame.accepted_original]; event=frame[frame.recorded_future_landfall]
    results=dict(period='1980-2003',source_sha256=p.sha(p.SOURCE),raw_windows=len(original),
        raw_storms=int(original.storm_id.nunique()),synoptic_windows=len(frame),
        synoptic_storms=int(frame.storm_id.nunique()),future_L_windows=len(event),
        future_L_storms=int(event.storm_id.nunique()),manifest='generated/historical_window_manifest.csv',
        scope='Reuse in an earlier period of the same Atlantic HURDAT2 format; no predictor fitting.')
    assert (len(original),len(frame),len(event))==(3642,4624,395)
    out=p.HERE/'generated';out.mkdir(exist_ok=True)
    frame.to_csv(out/'historical_window_manifest.csv',index=False)
    p.dump(out/'historical_manifest_verification.json',results)
    print(json.dumps(results))

if __name__=='__main__':main()
