"""Evaluate existing 24h checkpoints on expanded cohorts; never train/select."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import sys
import json
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
REV=HERE.parents[1]
sys.path.insert(0,str(REV/'experiments'))
from run_controlled import digest, evaluate, load_split
from controlled_heads import ControlledNumericModel


def main():
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    design=json.loads((HERE/'design_lock.json').read_text(encoding='utf-8'))
    runs=[REV/p for p in design['frozen_model_sha256'] if p.endswith('best.pt')]
    checks=[]
    for profile in ('raw_features','synoptic_features'):
        for period in ('test2022_2024','extension2025'):
            root=HERE/'data'/profile/period
            meta=json.loads((root/'metadata.json').read_text(encoding='utf-8'))
            rows=[np.load(p,allow_pickle=True).item() for p in sorted((root/'test').glob('*.npy'))]
            out=HERE/'results'/profile/period;out.mkdir(parents=True,exist_ok=True)
            for weight in runs:
                run=weight.parent;name=run.name
                cfg=json.loads((run/'frozen_config.json').read_text(encoding='utf-8'))
                for p,h in cfg['code_sha256'].items():
                    assert digest(REV/'experiments'/p)==h,p
                assert digest(weight)==design['frozen_model_sha256'][weight.relative_to(REV).as_posix()]
                original_meta=json.loads((REV/'data/controlled_12to1_lead4/metadata.json').read_text(encoding='utf-8'))
                for k in ('stats','feature_names','lead_steps','input_len'):assert meta[k]==original_meta[k]
                ix=[meta['feature_names'].index(f) for f in cfg['features']]
                data=load_split(root,'test',meta,ix)
                model=ControlledNumericModel(len(ix),meta['input_len'],meta['lead_steps'],meta['stats']['target_mean'],meta['stats']['target_std'],cfg['head'],cfg['d_model'],cfg['num_heads'],cfg['num_layers'],cfg['dropout'],cfg['cv_convention']).to('cuda')
                cp=torch.load(weight,map_location='cuda',weights_only=False)
                model.load_state_dict(cp['model'])
                summary,pred=evaluate(model,data,cfg['batch_size'],'cuda')
                dest=out/f'{name}.npz'
                assert not dest.exists(),dest
                np.savez_compressed(dest,**pred,storm_id=np.asarray([r['storm_id'] for r in rows]),
                     window=np.asarray([r['window'] for r in rows]),future_latlon=np.stack([r['future_latlon'] for r in rows]),
                     original=np.asarray([r['accepted_original'] for r in rows]),
                     landfall=np.asarray([r['recorded_future_landfall'] for r in rows]),
                     initialization=np.asarray([r['history_timestamp_hours'][-1] for r in rows]),
                     checkpoint_sha256=np.asarray(digest(weight)),lead_hours=np.asarray([24]))
                result={'profile':profile,'period':period,'run':name,'checkpoint_sha256':digest(weight),
                        'epoch':cp['epoch'],'summary':summary,'data_sha256':data['files'],'evaluation_sha256':digest(Path(__file__))}
                if profile=='raw_features':
                    oldfolder=run if period=='test2022_2024' else REV/'results/controlled_matrix/holdout2025'/name
                    old=np.load(oldfolder/'test_predictions.npz',allow_pickle=False)
                    oldmap={(str(s),int(w)):i for i,(s,w) in enumerate(zip(old['storm_id'],old['window']))}
                    ixnew=[i for i,r in enumerate(rows) if r['accepted_original']]
                    ixold=[oldmap[(rows[i]['storm_id'],int(rows[i]['window']))] for i in ixnew]
                    assert len(ixold)==len(oldmap)
                    assert np.array_equal(old['future_latlon'][ixold],np.stack([rows[i]['future_latlon'] for i in ixnew]))
                    maximum=float(np.max(np.abs(old['pred_error_km'][ixold]-pred['pred_error_km'][ixnew])))
                    assert maximum<0.01,(name,maximum)
                    result['original_forecasts_max_error_difference_km']=maximum
                (out/f'{name}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
                checks.append({k:v for k,v in result.items() if k!='data_sha256'})
                print(profile,period,name,flush=True)
                del model,cp;torch.cuda.empty_cache()
    (HERE/'neural_evaluation_checks.json').write_text(json.dumps({'status':'passed','evaluations':checks,'no_training_or_selection':True},indent=2),encoding='utf-8')


if __name__=='__main__':main()
