"""Bounded six-fit queue. Uses the unchanged checked training implementation."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json
import os
import subprocess
import sys
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
REV=HERE.parents[1]
EX=REV/'experiments'
sys.path.insert(0,str(EX))
from run_controlled import load_split, evaluate, digest
from controlled_heads import ControlledNumericModel


def fit(job):
    head,seed=job
    out=HERE/'runs'/f'{head}_seed{seed}'
    out.mkdir(parents=True,exist_ok=True)
    status=out/'status.json'
    if not status.exists() or json.loads(status.read_text())['status']!='completed':
        cmd=[sys.executable,str(EX/'run_controlled.py'),'--data-dir',str(HERE/'data/main'),
             '--output',str(out),'--head',head,'--rmw','off','--seed',str(seed),
             '--epochs','40','--device','cuda','--threads','2','--evaluate-test']
        if (out/'frozen_config.json').exists():cmd+=['--resume']
        with (out/'console.log').open('a',encoding='utf-8') as log:
            result=subprocess.run(cmd,stdout=log,stderr=log,timeout=2400,
                env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8'))
        if result.returncode:
            raise RuntimeError(f'{head}/{seed}: consult {out}/console.log')
    print(f'Completed {head} seed {seed}',flush=True)
    return out


def main():
    jobs=[(h,s) for h in ('direct','residual') for s in (42,123,2024)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        paths=list(pool.map(fit,jobs))
    # Extension evaluation starts only after every checkpoint was validation-selected.
    torch.set_num_threads(2)
    meta=json.loads((HERE/'data/extension/metadata.json').read_text())
    indices=[i for i,n in enumerate(meta['feature_names']) if n!='rmax']
    data=load_split(HERE/'data/extension','test',meta,indices)
    for out in paths:
        config=json.loads((out/'frozen_config.json').read_text())
        model=ControlledNumericModel(39,12,[4],meta['stats']['target_mean'],meta['stats']['target_std'],
            config['head'],256,4,3,.1,config['cv_convention']).to('cuda')
        state=torch.load(out/'best.pt',map_location='cuda',weights_only=False)
        model.load_state_dict(state['model'])
        summary,predictions=evaluate(model,data,32,'cuda')
        np.savez_compressed(out/'extension_predictions.npz',**predictions,
            storm_id=np.asarray([k[0] for k in data['keys']]),window=np.asarray([k[1] for k in data['keys']]),
            future_latlon=data['tensors']['future'].numpy(),history_latlon=data['tensors']['history'].numpy())
        (out/'extension.json').write_text(json.dumps(dict(summary=summary,checkpoint_sha256=digest(out/'best.pt')),indent=2))
    (HERE/'training_complete.json').write_text(json.dumps(dict(status='completed',fits=6,
        test_visibility='Post-review sensitivity, not a first blind test',runs=[str(p.relative_to(HERE)) for p in paths]),indent=2))
    print('Six fits and extension evaluations completed.',flush=True)


if __name__=='__main__':main()
