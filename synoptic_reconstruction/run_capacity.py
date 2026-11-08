"""Execute only the fixed nine post-review capacity fits and locked evaluation."""
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REV = HERE.parents[1]
EXPERIMENTS = REV / 'experiments'
sys.path.insert(0, str(EXPERIMENTS))
from process_scheduler import ProcessRegistry, schedule
from provenance import EVALUATION_FILES, source_hashes


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, value):
    tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2),encoding='utf-8')
    tmp.replace(p)


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    args=parser.parse_args()
    (HERE/'logs').mkdir(exist_ok=True)
    (HERE/'jobs').mkdir(exist_ok=True)
    jobs=[]
    for head in ('direct','residual','adaptive'):
        for seed in (42,123,2024):
            name=f'lead4_{head}_rmw_on_seed{seed}'
            command=[sys.executable,str(EXPERIMENTS/'run_controlled.py'), '--data-dir',str(REV/'data/controlled_12to1_lead4'),
                '--output',str(HERE/'runs'/name),'--head',head,'--rmw','on','--seed',str(seed),
                '--d-model','64','--num-heads','4','--num-layers','1','--epochs','40','--batch-size','32',
                '--lr','0.0001','--weight-decay','0.0001','--dropout','0.1','--patience','0',
                '--aux-direct','0','--aux-residual','0','--aux-gate','0','--device','cuda','--threads','4',
                '--cv-convention','legacy_previous_origin','--evaluate-test']
            jobs.append((name,command))
    fixed=dict(post_review_after_test_visibility=True,plan_sha256=sha(HERE/'POST_REVIEW_PLAN.md'),
        orchestrator_sha256=sha(Path(__file__)),code_sha256=source_hashes(EVALUATION_FILES+('process_scheduler.py',)),
        max_parallel=2,commands=dict(jobs),bootstrap_draws=10000,bootstrap_seed=20260913)
    plan_path=HERE/'plan_lock.json'
    if plan_path.exists():
        if not args.resume:
            raise RuntimeError('Plan already exists; use --resume for identical unfinished work')
        old=read(plan_path)
        assert old['fixed']==fixed, 'Frozen plan changed'
    else:
        assert not args.resume
        write(plan_path,dict(created_utc=now(),fixed=fixed))
    registry=ProcessRegistry()
    environment=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8')

    def launch(name, command):
        path=HERE/'runs'/name
        if (path/'status.json').exists() and read(path/'status.json').get('status')=='completed':
            assert read(path/'status.json')['test_evaluated']
            return
        if (path/'frozen_config.json').exists():
            assert args.resume
            command=command+['--resume']
        write(HERE/'jobs'/f'{name}.json',dict(status='running',started_utc=now(),command=command))
        with (HERE/'logs'/f'{name}.log').open('a',encoding='utf-8') as log:
            process=registry.start(command,stdout=log,stderr=log,env=environment)
            try:
                code=process.wait()
            finally:
                registry.unregister(process)
        write(HERE/'jobs'/f'{name}.json',dict(status='completed' if code==0 else 'failed',ended_utc=now(),returncode=code))
        if code:
            raise RuntimeError(f'{name} returned {code}; consult its log')
        print(f'Completed {name}',flush=True)

    result=schedule(jobs,launch,2,lambda state:write(HERE/'queue_status.json',dict(updated_utc=now(),**state)),registry)
    if result['failed']:
        raise RuntimeError('Capacity queue failed; no holdout evaluation')
    selected={}
    for name,_ in jobs:
        run=HERE/'runs'/name
        config=read(run/'frozen_config.json')
        status=read(run/'status.json')
        history=read(run/'training_history.json')
        assert len(history)==40 and config['d_model']==64 and config['num_layers']==1
        assert status['selected_epoch']==min(history,key=lambda x:x['validation_mean_km'])['epoch']
        selected[name]=dict(checkpoint_sha256=sha(run/'best.pt'),selected_epoch=status['selected_epoch'],
            config_sha256=sha(run/'frozen_config.json'),validation=read(run/'validation.json'))
    lock=HERE/'all_selections_frozen.json'
    if lock.exists():
        assert read(lock)['selected']==selected
    else:
        write(lock,dict(frozen_utc=now(),post_review_after_test_visibility=True,no_configuration_selection=True,selected=selected))
    write(HERE/'queue_status.json',dict(status='evaluating_2025',completed_fits=9,updated_utc=now()))
    for name,_ in jobs:
        out=HERE/'holdout2025'/name
        if (out/'status.json').exists() and read(out/'status.json').get('test_evaluated'):
            assert read(out/'evaluation_lock.json')['checkpoint_sha256']==selected[name]['checkpoint_sha256']
            continue
        command=[sys.executable,str(EXPERIMENTS/'evaluate_locked.py'),'--run',str(HERE/'runs'/name),
            '--data-dir',str(REV/'data/extension2025_lead4'),'--output',str(out),'--device','cuda']
        if out.exists():
            assert args.resume
            command+=['--resume']
        with (HERE/'logs'/f'{name}_2025.log').open('a',encoding='utf-8') as log:
            process=registry.start(command,stdout=log,stderr=log,env=environment)
            try:
                code=process.wait()
            finally:
                registry.unregister(process)
        if code:
            raise RuntimeError(f'2025 evaluation failed for {name}; do not retrain')
        print(f'Evaluated 2025 {name}',flush=True)
    assert source_hashes(EVALUATION_FILES+('process_scheduler.py',))==fixed['code_sha256']
    write(HERE/'queue_status.json',dict(status='completed',completed_fits=9,holdout2025_evaluated=True,completed_utc=now()))


if __name__=='__main__':
    main()
