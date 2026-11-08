"""Executable behavioral contract, including a real fixed predictor check."""
import copy
import json
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import prepare_revision as p


class WindowContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw=p.LIB.parse_hurdat2(str(p.SOURCE))
        cls.group=next(g.copy() for _,g in raw.groupby('typhoon_id')
            if len(g)>40 and np.any(g.storm_year==2022))
        cls.group=cls.group[cls.group.time.isin(p.LIB.STANDARD_TIMES)].iloc[:24].copy()
        # Use a known complete synthetic timestamp sequence with actual archived values.
        dates=pd.date_range('2022-08-01',periods=len(cls.group),freq='6h')
        cls.group['date']=[int(t.strftime('%Y%m%d')) for t in dates]
        cls.group['time']=[int(t.strftime('%H%M')) for t in dates]
        cls.group['_timestamp_hours']=dates.as_unit('ns').astype('int64')//3600000000000
        cls.group['record_identifier']=''
        cls.sid=cls.group.typhoon_id.iloc[0]
        cls.splits={'train':{cls.sid},'test':set()}

    def test_asynoptic_insertion_invariance(self):
        base=p.storm_arrays(self.group)
        extra=self.group.iloc[[12]].copy()
        extra['time']=300;extra['_timestamp_hours']+=3
        changed=p.storm_arrays(pd.concat([self.group,extra]))
        np.testing.assert_array_equal(base['x_num'],changed['x_num'])
        self.assertEqual(p.accepted_indices(base),p.accepted_indices(changed))

    def test_required_record_missing(self):
        base=p.storm_arrays(self.group)
        missing=p.storm_arrays(self.group.drop(self.group.index[12]))
        self.assertIn(0,p.accepted_indices(base))
        self.assertEqual(p.accepted_indices(missing),[])

    def test_future_event_preserves_minutes(self):
        extra=self.group.iloc[[11]].copy()
        extra['time']+=30;extra['record_identifier']='L'
        before=p.make_rows(self.group,self.splits)[0]
        after=p.make_rows(pd.concat([self.group,extra]),self.splits)[0]
        self.assertFalse(before['recorded_future_landfall'])
        self.assertTrue(after['recorded_future_landfall'])
        np.testing.assert_array_equal(before['input_raw'],after['input_raw'])

    def test_future_information_isolation(self):
        before=p.make_rows(self.group,self.splits)[0]
        changed=self.group.copy()
        changed.iloc[12:,changed.columns.get_loc('latitude')]+=5
        changed.iloc[12:,changed.columns.get_loc('max_wind')]+=30
        changed.iloc[12:,changed.columns.get_loc('record_identifier')]='L'
        after=p.make_rows(changed,self.splits)[0]
        np.testing.assert_array_equal(before['input_raw'],after['input_raw'])
        self.assertNotEqual(before['recorded_future_landfall'],after['recorded_future_landfall'])
        sys.path.insert(0,str(p.REV/'experiments'))
        import torch
        from controlled_heads import ControlledNumericModel
        torch.manual_seed(17);torch.set_num_threads(2)
        model=ControlledNumericModel(39,12,[4],[0,0],[1,1],head='residual').eval()
        keep=[i for i,n in enumerate(p.LIB.FEATURE_NAMES) if n!='rmax']
        def predict(row):
            x=np.nan_to_num(row['input_raw'][:,keep],nan=0)
            with torch.no_grad():
                return model(torch.tensor(x[None]),torch.tensor(row['history_latlon'][None]))['pred'].numpy()
        np.testing.assert_array_equal(predict(before),predict(after))

    def test_train_only_normalization(self):
        rows=p.make_rows(self.group,self.splits)
        test=copy.deepcopy(rows[0]);test['split']='test'
        base=p.fit_stats(rows+[test])
        test['input_raw']+=100;test['target_disp_raw_km']+=1000
        after=p.fit_stats(rows+[test])
        for key in base:np.testing.assert_array_equal(base[key],after[key])

    def test_duplicate_or_conflicting_timestamp_rejected(self):
        extra=self.group.iloc[[0]].copy()
        for conflict in (False,True):
            if conflict:extra['latitude']+=1
            with self.assertRaises(ValueError):p.storm_arrays(pd.concat([self.group,extra]))

    def test_traceability_real_archive(self):
        manifest=json.loads((p.HERE/'window_manifest.json').read_text())
        raw=p.LIB.parse_hurdat2(str(p.SOURCE))
        item=next(r for r in manifest if r['split']=='test')
        arr=p.storm_arrays(raw[raw.typhoon_id==item['storm_id']])
        init=item['window'];indices=np.flatnonzero(arr['timestamp_hours']==init)
        self.assertEqual(len(indices),1)
        ix=int(indices[0])
        file=p.HERE/'data/main/test'/f"{item['storm_id']}_{init}.npy"
        saved=np.load(file,allow_pickle=True).item()
        np.testing.assert_array_equal(arr['coords'][ix-11:ix+1],saved['history_latlon'])
        self.assertEqual(item['window_id'],saved['window_id'])

    def test_speed_uses_actual_elapsed_hours(self):
        changed=self.group.drop(self.group.index[1])
        arr=p.storm_arrays(changed)
        d=arr['x_num'][1,[p.LIB.FEATURE_NAMES.index(n) for n in ('dx6','dy6')]]
        speed=arr['x_num'][1,p.LIB.FEATURE_NAMES.index('speed6')]
        self.assertAlmostEqual(float(speed),float(np.hypot(*d)/12),places=4)


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(WindowContract)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    p.dump(p.HERE/'contract_test_results.json',dict(tests_run=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),passed=result.wasSuccessful(),
        source_sha256=p.sha(p.SOURCE),test_script_sha256=p.sha(__file__),
        scope='Tests establish the stated implementation contract, not validity for all geoscience data.'))
    sys.exit(0 if result.wasSuccessful() else 1)
