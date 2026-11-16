"""Distill saved leaderboard teachers and evaluate matched static caches."""
import argparse
import csv
import json
import os
from pathlib import Path
import time

import leaderboard_common as C
from leaderboard_common import torch,np,F,ROOT,OUT,DATA


def log(**kw):print(json.dumps(kw),flush=True)


def train(args,target,x,teacher,split,labels):
    C.seed_all(args.seed)
    model=C.Student(x.shape[1],teacher.shape[1]).cuda()
    checkpoint=target/f'student-{args.student_input}.pt'
    if checkpoint.exists() and (target/f'student-trained-{args.student_input}.json').exists():
        model.load_state_dict(torch.load(checkpoint,map_location='cuda')['state'])
        return model
    opt=torch.optim.Adam(model.parameters(),lr=.005,weight_decay=0)
    train_mask=torch.zeros(len(x),device='cuda',dtype=torch.bool)
    train_mask[torch.as_tensor(split['train'],device='cuda')]=True
    valid=torch.as_tensor(split['model_valid'],device='cuda')
    y=torch.as_tensor(labels,device='cuda')
    tp=teacher.softmax(1)
    best,patience,trace=-1,0,[]
    begin=time.perf_counter()
    for epoch in range(1,args.student_epochs+1):
        model.train()
        permutation=torch.randperm(len(x),device='cuda')
        for batch in permutation.split(65536):
            z=model(x[batch]);mask=train_mask[batch]
            ce=F.cross_entropy(z[mask],y[batch][mask])
            kd=F.kl_div(z.log_softmax(1),tp[batch],reduction='batchmean')
            loss=.2*ce+.8*kd
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite distillation loss')
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        if epoch%5==0 or epoch==1:
            model.eval()
            with torch.no_grad():
                acc=(model(x[valid]).argmax(1)==y[valid]).float().mean().item()
            if acc>best:
                best,patience=acc,0
                torch.save(dict(state={k:v.cpu().clone() for k,v in model.state_dict().items()},
                    epoch=epoch,validation_accuracy=best,teacher_sha256=C.digest(target/'teacher.pt'),
                    inputs=args.student_input,parameters=sum(p.numel() for p in model.parameters())),checkpoint)
            else:patience+=5
            row=dict(epoch=epoch,validation_accuracy=acc,best_validation_accuracy=best,
                loss=loss.item(),seconds=time.perf_counter()-begin)
            trace.append(row);C.write(target/f'student-{args.student_input}-trace.json',trace)
            log(dataset=args.dataset,seed=args.seed,phase='distillation',**row)
            if epoch>=50 and patience>=50:break
    model.load_state_dict(torch.load(checkpoint,map_location='cuda')['state'])
    C.write(target/f'student-trained-{args.student_input}.json',dict(
        completed_epoch=epoch,selected_validation_accuracy=best,
        checkpoint_sha256=C.digest(checkpoint),launcher_sha256=C.digest(Path(__file__)),
        common_sha256=C.digest(ROOT/'leaderboard_common.py')))
    return model


@torch.no_grad()
def evaluate(args,target,model,x,teacher,z,split,labels):
    # Teachers may train concurrently, but serving latency needs a quiet GPU.
    waiting_logged=False
    while True:
        active=[]
        markers=list(OUT.glob('ogbn-*/seed-*/teacher-active.json'))
        markers+=list(OUT.glob('ogbn-*/seed-*/student-*-active.json'))
        for marker in markers:
            try:
                pid=json.loads(marker.read_text())['pid']
                os.kill(pid,0)
                active.append(pid)
            except (FileNotFoundError,ProcessLookupError):
                pass
        if not active:break
        if not waiting_logged:
            log(phase='waiting_for_exclusive_latency_measurement',teacher_pids=active)
            waiting_logged=True
        time.sleep(5)
    n=len(x);rng=np.random.default_rng(args.seed+500)
    pop=rng.permutation(np.arange(1,n+1)).astype(np.float64)**(-.8);pop/=pop.sum()
    history=rng.choice(n,100000,p=pop)
    counts=torch.as_tensor(np.bincount(history,minlength=n)+1,device='cuda',dtype=torch.float32)
    calibration=torch.as_tensor(split['calibration'],device='cuda')
    fit=C.fit_benefit(teacher[calibration],z[calibration],
        torch.as_tensor(labels[split['calibration']],device='cuda'))
    C.write(target/f'calibration-{args.student_input}.json',{k:v.cpu().tolist() for k,v in fit.items()})
    benefit=C.predict_benefit(teacher,z,fit)
    test=split['test'];testpop=pop[test]/pop[test].sum()
    shifted=.5*testpop+.5*testpop[::-1]
    query=torch.as_tensor(rng.choice(test,1024,p=testpop),device='cuda')
    student_pred=z.argmax(1).cpu().numpy()
    teacher_pred=teacher.argmax(1).cpu().numpy()
    np.savez_compressed(target/f'workload-{args.student_input}.npz',popularity=pop,
        history=history,query=query.cpu().numpy(),test=test)
    np.savez_compressed(target/f'predictions-{args.student_input}.npz',
        teacher=teacher_pred,student=student_pred)
    rows=[];chosen_all={}
    configs=[('none',0)]+[(p,b) for b in [.01,.05,.1,.2] for p in ['frequency','disagreement','benefit']]
    for policy,budget in configs:
        chosen=(torch.empty(0,dtype=torch.long,device='cuda') if policy=='none' else
            C.select(policy,teacher,z,counts,budget,benefit))
        chosen_np=chosen.cpu().numpy();mask=np.zeros(n,dtype=bool);mask[chosen_np]=True
        mixed=student_pred.copy();mixed[chosen_np]=teacher_pred[chosen_np]
        correct=(mixed[test]==labels[test])
        route=C.ServingCache(model,teacher,chosen)
        expected=z[query].clone();hit=mask[query.cpu().numpy()]
        expected[torch.as_tensor(hit,device='cuda')]=teacher[query[torch.as_tensor(hit,device='cuda')]]
        actual=route(query,x)
        torch.testing.assert_close(actual,expected,rtol=2e-4,atol=2e-4)
        assert np.array_equal(actual.argmax(1).cpu().numpy(),mixed[query.cpu().numpy()])
        for _ in range(10):route(query,x)
        torch.cuda.synchronize();elapsed=[]
        for _ in range(40):
            start=time.perf_counter();route(query,x);torch.cuda.synchronize()
            elapsed.append((time.perf_counter()-start)*1000)
        row=dict(dataset=args.dataset,seed=args.seed,student_input=args.student_input,
            policy=policy,budget=budget,cached_nodes=len(chosen),
            accuracy=float(correct.mean()),weighted_accuracy=float(correct@testpop),
            shifted_weighted_accuracy=float(correct@shifted),
            test_node_hit_rate=float(mask[test].mean()),request_hit_rate=float(mask[test]@testpop),
            latency_ms_median=float(np.median(elapsed)),latency_ms_p95=float(np.quantile(elapsed,.95)),
            cache_payload_bytes=len(chosen)*teacher.shape[1]*4,
            cache_index_bytes=0 if not len(chosen) else n*8)
        rows.append(row);chosen_all[f'{policy}-{budget}']=chosen_np
        del route
    np.savez_compressed(target/f'cache-indices-{args.student_input}.npz',**chosen_all)
    C.write(target/f'metrics-{args.student_input}.json',rows)
    checkpoint=torch.load(target/f'student-{args.student_input}.pt',map_location='cpu')
    teacher_correct=teacher_pred[test]==labels[test]
    C.write(target/f'complete-{args.student_input}.json',dict(rows=len(rows),
        student_checkpoint_sha256=C.digest(target/f'student-{args.student_input}.pt'),
        source_sha256=C.digest(Path(__file__)),common_sha256=C.digest(ROOT/'leaderboard_common.py'),
        teacher_accuracy=float(teacher_correct.mean()),teacher_weighted_accuracy=float(teacher_correct@testpop),
        student_parameters=checkpoint['parameters'],selected_epoch=checkpoint['epoch'],
        checkpoint_selection='model_valid only',cache_selection='history, teacher/student outputs, calibration labels only',
        protocol='Ordinary CE/KL distillation, lambda .2, temperature 1, three-layer width-256 BN MLP; no FAITH objectives',
        student_feature_note='Precomputed GIANT-XRT embeddings' if args.student_input=='giant' else 'Original OGB features',
        max_epochs=args.student_epochs,validation_every=5,patience_epochs=50,min_epochs=50,
        budgets=[.01,.05,.1,.2],route_matches_offline=True,request_distribution='Synthetic permuted Zipf(0.8)',
        teacher_cache='Frozen precomputed float32 logits; no online graph or teacher execution'))
    log(dataset=args.dataset,seed=args.seed,phase='complete',
        primary_results=[r for r in rows if r['budget'] in [0,.1]])


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True,choices=['ogbn-arxiv','ogbn-products'])
    p.add_argument('--seed',type=int,default=42);p.add_argument('--student-epochs',type=int,default=500)
    p.add_argument('--student-input',choices=['giant','raw'],default='giant')
    p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--train-only',action='store_true',help='Defer cache evaluation to the sequential study runner')
    args=p.parse_args();C.setup();target=args.output/args.dataset/f'seed-{args.seed}'
    if (target/f'complete-{args.student_input}.json').exists():return
    split=C.load_split(args.dataset)
    np.savez_compressed(target/'split.npz',**split)
    labels=np.load(ROOT/'ogb-data'/args.dataset/'labels.npy')
    teacher=torch.load(target/'teacher.pt',map_location='cpu')['logits'].cuda()
    features=(DATA/(args.dataset[5:]+'-giant.npy') if args.student_input=='giant'
        else ROOT/'ogb-data'/args.dataset/'features.npy')
    x=torch.as_tensor(np.load(features),device='cuda')
    active=target/f'student-{args.student_input}-active.json'
    C.write(active,dict(pid=os.getpid(),dataset=args.dataset,seed=args.seed))
    try:
        model=train(args,target,x,teacher,split,labels)
    finally:
        active.unlink(missing_ok=True)
    if args.train_only:return
    z=C.infer(model,x)
    evaluate(args,target,model,x,teacher,z,split,labels)


if __name__=='__main__':main()
