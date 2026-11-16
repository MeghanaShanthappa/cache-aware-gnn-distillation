"""Sequential GPU study runner; logs every command and stops on any failure."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/leaderboard_cache'


def save(value):
    temporary=OUT/'status.json.pending'
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(OUT/'status.json')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seeds',nargs='+',type=int,default=[42,43,45])
    p.add_argument('--datasets',nargs='+',default=['ogbn-arxiv','ogbn-products'])
    args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy()
    env['LD_LIBRARY_PATH']=':'.join(str(x) for x in [
        ROOT/'leaderboard-upstream/gpu-packages/nvidia/cusparse/lib',
        ROOT/'leaderboard-upstream/gpu-packages/nvidia/cuda_runtime/lib',
        ROOT/'.venv-reproduction/lib/python3.10/site-packages/torch/lib'])
    status=dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        datasets=args.datasets,seeds=args.seeds,completed=[],state='running',
        arxiv_teacher_epochs_per_stage=2000,products_teacher_epochs=400,student_max_epochs=500)
    for seed in args.seeds:
        for dataset in args.datasets:
            for phase,script in [('teacher','leaderboard_teacher.py'),('distillation_cache','leaderboard_distill.py')]:
                command=[sys.executable,'-u',str(ROOT/script),'--dataset',dataset,'--seed',str(seed)]
                logfile=OUT/f'{dataset}-seed-{seed}-{phase}.log'
                status.update(current=dict(dataset=dataset,seed=seed,phase=phase,log=str(logfile)),
                    command=command)
                save(status)
                print(json.dumps(status['current']),flush=True)
                with logfile.open('a') as stream:
                    process=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
                if process.returncode:
                    status.update(state='failed',returncode=process.returncode);save(status)
                    raise SystemExit(process.returncode)
                status['completed'].append(dict(dataset=dataset,seed=seed,phase=phase))
                save(status)
                if phase=='distillation_cache':
                    subprocess.run([sys.executable,str(ROOT/'analyze_leaderboard.py')],cwd=ROOT,check=True)
    status.update(state='complete',finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    status.pop('current',None);save(status)


if __name__=='__main__':main()
