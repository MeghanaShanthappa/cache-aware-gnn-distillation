"""Finish verification and figures automatically when the GPU study completes."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/leaderboard_cache'


def record(**value):
    value['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    (OUT/'postprocess-status.json').write_text(json.dumps(value,indent=2)+'\n')


def main():
    record(state='waiting_for_training')
    deadline=time.monotonic()+10500
    while time.monotonic()<deadline:
        status=json.loads((OUT/'status.json').read_text())
        if status['state']=='failed':
            raise RuntimeError('Study worker failed; inspect status.json and the current log')
        if status['state']=='complete':break
        time.sleep(5)
    else:raise TimeoutError('Training is still running; postprocessing can be rerun')
    record(state='verifying_checkpoints')
    subprocess.run([sys.executable,str(ROOT/'verify_leaderboard.py')],cwd=ROOT,check=True)
    subprocess.run([sys.executable,str(ROOT/'analyze_leaderboard.py')],cwd=ROOT,check=True)
    os.environ['MPLCONFIGDIR']=str(ROOT/'.matplotlib-leaderboard')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=json.loads((OUT/'metrics.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for ax,name in zip(axes,['ogbn-arxiv','ogbn-products']):
        selected=[r for r in rows if r['dataset']==name]
        seeds=sorted({r['seed'] for r in selected})
        baseline=[r['accuracy']*100 for r in selected if r['policy']=='none']
        mean=lambda x:sum(x)/len(x)
        for policy,color in [('frequency','#4884b8'),('disagreement','#df9a30'),('benefit','#258268')]:
            xx=[0,1,5,10,20];yy=[mean(baseline)]
            for budget in [.01,.05,.1,.2]:
                yy.append(mean([r['accuracy']*100 for r in selected if r['policy']==policy and r['budget']==budget]))
            ax.plot(xx,yy,marker='o',label=policy.capitalize(),color=color,linewidth=2)
        teacher=[]
        for seed in seeds:
            info=json.loads((OUT/name/f'seed-{seed}'/'complete-giant.json').read_text())
            teacher.append(info['teacher_accuracy']*100)
        ax.axhline(mean(teacher),color='#777777',linestyle='--',linewidth=1.3,label='Local teacher')
        ax.set_title(name);ax.set_xlabel('Cache capacity (% of all nodes)')
        ax.set_ylabel('Standard test accuracy (%)');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Distilled MLP with cached teacher predictions',fontsize=13)
    fig.savefig(OUT/'cache-accuracy.png',dpi=200)
    plt.close(fig)
    report=OUT/'REPORT.md'
    text=report.read_text().replace('## Separate workload and serving measurements',
        '![Standard accuracy by cache capacity](cache-accuracy.png)\n\n## Separate workload and serving measurements')
    report.write_text(text)
    record(state='complete',report=str(report),evaluations=len(rows),
        checkpoint_verification=str(OUT/'checkpoint_verification.json'),
        quality_verification=str(OUT/'independent_verification.json'),figure=str(OUT/'cache-accuracy.png'))


if __name__=='__main__':
    try:main()
    except Exception as error:
        record(state='failed',error=str(error))
        raise
