"""Reconstruct all reported quality metrics independently from saved predictions."""
import csv
import json
from pathlib import Path
import statistics
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/leaderboard_cache'
MODELS={
 'ogbn-arxiv':dict(model='GIANT-XRT + AGDN + BoT + self-KD',rank=7,published=76.37),
 'ogbn-products':dict(model='GIANT-XRT + SAGN + SCR',rank=7,published=86.67)}


def main():
    rows=[]; teachers={}; checks=0
    for path in sorted(OUT.glob('ogbn-*/seed-*/metrics-giant.json')):
        folder=path.parent
        complete=folder/'complete-giant.json'
        if not complete.exists():continue
        name=folder.parent.name;seed=int(folder.name[5:])
        local=json.loads(path.read_text());metadata=json.loads(complete.read_text())
        labels=np.load(ROOT/'ogb-data'/name/'labels.npy')
        pred=np.load(folder/'predictions-giant.npz')
        workload=np.load(folder/'workload-giant.npz')
        caches=np.load(folder/'cache-indices-giant.npz')
        split=np.load(folder/'split.npz');test=split['test']
        assert not np.intersect1d(split['calibration'],split['model_valid']).size
        weights=workload['popularity'][test];weights/=weights.sum()
        shifted=.5*weights+.5*weights[::-1]
        for row in local:
            chosen=caches[f"{row['policy']}-{row['budget']}"]
            assert len(np.unique(chosen))==len(chosen)
            assert len(chosen)<=int(len(labels)*row['budget'])
            output=pred['student'].copy();output[chosen]=pred['teacher'][chosen]
            hit=np.zeros(len(labels),dtype=bool);hit[chosen]=True
            correct=output[test]==labels[test]
            expected=dict(accuracy=correct.mean(),weighted_accuracy=correct@weights,
                shifted_weighted_accuracy=correct@shifted,test_node_hit_rate=hit[test].mean(),
                request_hit_rate=hit[test]@weights)
            for key,value in expected.items():
                assert abs(row[key]-value)<1e-12,(path,key,row[key],value)
                checks+=1
        rows+=local;teachers[(name,seed)]=metadata['teacher_accuracy']*100
    OUT.mkdir(exist_ok=True)
    (OUT/'metrics.json').write_text(json.dumps(rows,indent=2)+'\n')
    if rows:
        with (OUT/'metrics.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    def fmt(values):
        if not values:return 'Not completed'
        return f'{statistics.mean(values):.2f}' if len(values)==1 else f'{statistics.mean(values):.2f} ± {statistics.pstdev(values):.2f}'
    lines=['# Top-ten OGB teachers with distillation and prediction caching','',
        'Selected from the [official OGB leaderboard](https://ogb.stanford.edu/docs/leader_nodeprop/) on 16 November 2026. Both teachers are ranked seventh. Published values are external reference scores; the other columns below are locally measured. This study uses ordinary distillation, without FAITH graph objectives.','',
        'The teacher follows the authors’ architecture and training recipe. Half of the official validation split is reserved exclusively for cache calibration, which differs from the published full-validation protocol. The feature-only student is a three-layer, width-256 MLP with batch normalization and dropout 0.1. It uses the same precomputed GIANT-XRT node embeddings as the teacher, but no neighbor aggregation or online teacher call. GIANT pretraining itself used graph structure and text; this is not the original raw-feature student experiment.','',
        'Teacher details: arxiv uses the released native-DGL AGDN training routine, 2,000 supervised epochs followed by 2,000 graph self-distillation epochs, RMSprop at 0.01. Products uses released SAGN models and supervised/SCR update functions, 400 epochs, 100 warmup epochs, Adam at 0.001, and the published confidence schedule. Product features use three row-mean propagation hops; training labels use nine. We save the best EMA state explicitly to avoid the released code applying final EMA parameters to an earlier best model.','',
        '## Primary result: standard test accuracy','',
        'Percent. Cache capacity is 10% of all graph nodes. All cache policies use the same frozen student and teacher. A gain is relative to that student without a cache. Multiple completed seeds are summarized as mean ± population standard deviation; single-seed values are preliminary.','',
        '| Dataset | Teacher | Published teacher | Local teacher | Distilled MLP | Frequency cache | Disagreement cache | Benefit cache | Benefit gain (pp) | Seeds |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,entry in MODELS.items():
        groups={p:[r for r in rows if r['dataset']==name and r['policy']==p and r['budget']==(0 if p=='none' else .1)] for p in ['none','frequency','disagreement','benefit']}
        seeds=sorted(r['seed'] for r in groups['none'])
        paired=[]
        for seed in seeds:
            base=next(r for r in groups['none'] if r['seed']==seed)
            cached=next(r for r in groups['benefit'] if r['seed']==seed)
            paired.append(100*(cached['accuracy']-base['accuracy']))
        cells=[name,entry['model'],f"{entry['published']:.2f}",fmt([teachers[(name,s)] for s in seeds])]
        cells += [fmt([100*r['accuracy'] for r in groups[p]]) for p in groups]
        cells += [fmt(paired),str(len(seeds))]
        lines.append('| '+' | '.join(cells)+' |')
    lines+=['','## Separate workload and serving measurements','',
        'Request-weighted accuracy uses synthetic Zipf(0.8) popularity, with 100,000 history requests and additive-one count smoothing. Frequency ties are broken pseudorandomly with fixed seed 7341, independently of the contiguous product split IDs; all strict frequency rankings are preserved. These values must not be compared with OGB leaderboard test accuracy. Cache selection uses history and model outputs; benefit regression uses only the reserved calibration labels. It may leave capacity unused when predicted benefit is nonpositive.','',
        '| Dataset | Uncached weighted accuracy | Benefit weighted accuracy | Teacher request hit rate | Uncached latency (ms) | Benefit latency (ms) |',
        '|---|---:|---:|---:|---:|---:|']
    for name in MODELS:
        base=[r for r in rows if r['dataset']==name and r['policy']=='none']
        cached=[r for r in rows if r['dataset']==name and r['policy']=='benefit' and r['budget']==.1]
        lines.append('| '+' | '.join([name,fmt([100*r['weighted_accuracy'] for r in base]),
            fmt([100*r['weighted_accuracy'] for r in cached]),fmt([100*r['request_hit_rate'] for r in cached]),
            fmt([r['latency_ms_median'] for r in base]),fmt([r['latency_ms_median'] for r in cached])])+' |')
    lines+=['','Latency is a GPU microbenchmark of 1,024 already-resident requests, including lookup, output assembly, and miss-only student inference. It excludes text encoding, host transfer, graph/teacher preprocessing, refresh, and network costs. It is measured with 10 warmups and 40 synchronized repetitions. A higher cache accuracy does not establish a latency improvement.','',
        '## Evidence and limitations','',
        f'{len(rows)} evaluation rows are complete. Independent reconstruction checked {checks} quality/hit-rate values from saved predictions, cache indices, and workloads. Budgets of 1%, 5%, 10%, and 20% are retained in the raw metrics. Test labels did not select checkpoints, cache nodes, or training settings.','',
        'Training: student cross entropy and teacher KL have weights 0.2 and 0.8, temperature 1; Adam learning rate 0.005; all-node minibatches of 65,536; maximum 500 epochs; model-validation selection every five epochs; minimum 50 epochs and patience 50 epochs. Each teacher/student seed is paired with a fixed synthetic workload.','',
        'This measures recovery of a distilled student’s accuracy through cached teacher predictions. It does not imply that caching improves the teacher itself or exceeds the leaderboard. Results are transductive and do not evaluate truly new nodes or a changing request stream.','',
        'Artifacts: `metrics.json`, `metrics.csv`, `independent_verification.json`, `status.json`, and per-seed teacher/student checkpoints, selected splits, prediction vectors, cache indices, workload history, calibration coefficients, training traces, and source hashes. Official implementations are under `leaderboard-upstream/AGDN` and `leaderboard-upstream/SCR`.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (OUT/'independent_verification.json').write_text(json.dumps(dict(rows=len(rows),
        quality_and_hit_values_verified=checks,budgets_verified=True,
        calibration_disjoint_from_model_selection=True,all_complete_rows_verified=True),indent=2)+'\n')
    print(json.dumps(dict(rows=len(rows),checked_values=checks)),flush=True)


if __name__=='__main__':main()
