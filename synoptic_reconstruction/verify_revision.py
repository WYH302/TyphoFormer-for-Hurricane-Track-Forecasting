"""Verify added numeric inputs and reproduce paired comparisons from saved errors.

Does not fit models or use the original working checkout outside this archive.
"""
import json
import numpy as np
from types import SimpleNamespace
import prepare_revision as p
from summarize_revision import summarize


def main():
    frame=p.LIB.parse_hurdat2(str(p.SOURCE));frame=frame[frame.storm_year.between(2004,2025)]
    splits=p.LIB.storm_splits(frame,SimpleNamespace(train_years='2004-2021',test_years='2022-2024',split_seed=42,val_ratio=.1))
    splits['extension']=set(frame.loc[frame.storm_year==2025,'typhoon_id'])
    original=json.loads((p.REV/'data/controlled_12to1_lead4/metadata.json').read_text(encoding='utf-8'))
    for name in ('train','val','test'):
        assert set(original['split_storm_ids'][name])<=splits[name]
    assert not (splits['train']&splits['val'])
    assert not ((splits['train']|splits['val'])&(splits['test']|splits['extension']))
    rows=p.make_rows(frame,splits);stats=p.fit_stats(rows)
    event_differences=[]
    for r in rows:
        sub='extension' if r['split']=='extension' else 'main'
        split='test' if r['split']=='extension' else r['split']
        file=p.HERE/f'data/{sub}/{split}'/f"{r['storm_id']}_{r['window']}.npy"
        saved=np.load(file,allow_pickle=True).item()
        x=r['input_raw'];x=((np.where(np.isfinite(x),x,stats['x_mean'])-stats['x_mean'])/stats['x_std']).astype(np.float32)
        np.testing.assert_array_equal(saved['input_num'],x)
        for key in ('target_disp_raw_km','history_timestamp_hours','future_timestamp_hours','history_latlon','future_latlon'):
            np.testing.assert_array_equal(saved[key],r[key])
        if saved['recorded_future_landfall']!=r['recorded_future_landfall']:
            event_differences.append(r['window_id'])
    assert not event_differences,event_differences
    meta=json.loads((p.HERE/'data/main/metadata.json').read_text(encoding='utf-8'))
    for key,value in stats.items():np.testing.assert_array_equal(value,np.asarray(meta['stats'][key],np.float32))
    expected=json.loads((p.HERE/'sensitivity_results.json').read_text(encoding='utf-8'))
    summarize()
    actual=json.loads((p.HERE/'sensitivity_results.json').read_text(encoding='utf-8'))
    assert expected==actual
    results=dict(status='passed',numeric_windows_rebuilt=len(rows),all_numeric_inputs_and_targets_exact=True,
        original_storm_split_assignments_preserved=True,minute_precise_event_annotations_recovered=True,
        new_scores_reproduced_exactly=True,current_preparation_sha256=p.sha(p.HERE/'prepare_revision.py'),
        source_sha256=p.sha(p.SOURCE),score_script_sha256=p.sha(p.HERE/'summarize_revision.py'))
    p.dump(p.HERE/'revision_verification.json',results)
    print(json.dumps(results))


if __name__=='__main__':main()
