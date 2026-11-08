"""Correct unused cached event labels without changing any fitted input/output."""
import json
from types import SimpleNamespace
import numpy as np
import prepare_revision as p


def main():
    report=p.HERE/'annotation_correction.json'
    if report.exists():raise FileExistsError('Correction already recorded; preserve audit.')
    df=p.LIB.parse_hurdat2(str(p.SOURCE));df=df[df.storm_year.between(2004,2025)]
    splits=p.LIB.storm_splits(df,SimpleNamespace(train_years='2004-2021',test_years='2022-2024',split_seed=42,val_ratio=.1))
    splits['extension']=set(df.loc[df.storm_year==2025,'typhoon_id'])
    rows=p.make_rows(df,splits);changes=[]
    manifest=json.loads((p.HERE/'window_manifest.json').read_text(encoding='utf-8'))
    byid={r['window_id']:r for r in manifest}
    fields=['input_num','target_disp','target_disp_raw_km','history_latlon','origin','future_latlon']
    for r in rows:
        sub='extension' if r['split']=='extension' else 'main'
        split='test' if r['split']=='extension' else r['split']
        file=p.HERE/f'data/{sub}/{split}'/f"{r['storm_id']}_{r['window']}.npy"
        old=np.load(file,allow_pickle=True).item()
        if old['recorded_future_landfall']!=r['recorded_future_landfall']:
            before=p.sha(file);previous=old['recorded_future_landfall']
            consumed={k:old[k].copy() for k in fields}
            old['recorded_future_landfall']=r['recorded_future_landfall'];np.save(file,old)
            new=np.load(file,allow_pickle=True).item()
            for key in fields:np.testing.assert_array_equal(consumed[key],new[key])
            changes.append(dict(file=str(file.relative_to(p.HERE)),old_sha256=before,new_sha256=p.sha(file),old_event=previous,new_event=r['recorded_future_landfall']))
        byid[r['window_id']]['recorded_future_landfall']=r['recorded_future_landfall']
    p.dump(p.HERE/'window_manifest.json',manifest)
    p.dump(report,dict(changed_records=changes,model_consumed_fields_unchanged=True,
        explanation='After fitting, an unused cached event field was corrected to preserve minute precision. Frozen training configurations retain the original byte hashes; only recorded_future_landfall differs in the listed files. All numerical inputs, targets, weights, predictions and scores are unchanged. The original frozen-coverage analysis used its independently audited minute-precise labels and is unchanged.'))
    print('Corrected unused annotation records:',len(changes))


if __name__=='__main__':main()
