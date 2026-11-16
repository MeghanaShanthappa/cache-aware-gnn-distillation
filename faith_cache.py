"""Static cache and compute-matched joint refinement on saved full FAITH models."""
import argparse
import copy
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

import reproduce_cuda as runtime

torch, np, F = runtime.torch, runtime.np, runtime.F
support = runtime.support
ROOT = runtime.ROOT
BASE = ROOT / 'results/cuda_reproduction'
SEEDS = [42, 43, 45, 48, 52]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def incidence(similarity):
    """Sparse equivalent of the released dense-upper-triangle GDO builder."""
    upper = support.sp.triu(similarity, k=1).tocoo()
    keep = upper.data > 0
    row, col, values = upper.row[keep], upper.col[keep], upper.data[keep]
    order = np.lexsort((col, row))
    row, col, values = row[order], col[order], values[order]
    roots = np.sqrt(values)
    return support.sp.coo_matrix((np.stack([roots, -roots], 1).ravel(), (np.repeat(np.arange(len(row)), 2), np.stack([row, col], 1).ravel())), shape=(len(row), similarity.shape[0]))


def sparse_tensor(matrix, device):
    matrix = matrix.tocoo()
    indices = torch.as_tensor(np.stack([matrix.row, matrix.col]), dtype=torch.long, device=device)
    return torch.sparse_coo_tensor(indices, torch.as_tensor(matrix.data, dtype=torch.float32, device=device), matrix.shape).coalesce()


def prepare(name, device='cuda'):
    graph, labels, train, valid, test = support.released_data.load_data(name, seed=42)
    a = graph.adjacency_matrix()
    edges = a.indices()
    adjacency = support.sp.coo_matrix((a.val.numpy(), (edges[0].numpy(), edges[1].numpy())), shape=a.shape).tocsr()
    similarity = support.released_guide.calculate_similarity_matrix(adjacency, graph.ndata['feat'], metric='cosine')
    delta = incidence(similarity)
    lap = support.released_guide.laplacian(similarity)
    return dict(x=graph.ndata['feat'].to(device), y=labels.to(device), train=train.to(device), valid=valid.to(device), test=test.to(device), n=graph.num_nodes(), adjacency=sparse_tensor(adjacency, device), edges=edges.to(device), delta=sparse_tensor(delta, device), lap=sparse_tensor(lap, device))


def objective(logits, teacher, teacher_energy, data, config, weights):
    """Only nodewise CE and KL are reweighted; FAITH geometry/OSA stay intact."""
    train = data['train']
    ce = F.nll_loss(logits.log_softmax(1)[train], data['y'][train], reduction='none')
    ce = (ce * weights[train]).sum() / weights[train].sum()
    kd = F.kl_div((logits/config['tau']).log_softmax(1), (teacher/config['tau']).log_softmax(1), reduction='none', log_target=True).sum(1)
    kd = (kd * weights).sum() / weights.sum()
    energy = support.released_guide.compute_loss_energy(data['adjacency'], logits, teacher_energy, data['edges'], 500000)
    osa = torch.linalg.vector_norm(torch.sparse.mm(data['delta'], logits), dim=1).sum()
    return config['lamb']*ce + (1-config['lamb'])*kd + .02*energy + support.GAMMA[config['dataset']]*osa


@torch.no_grad()
def logits_of(model, x):
    model.eval()
    return model(None, x).detach()


def continue_student(initial, teacher, energy, data, config, counts, seed, steps, phases, budget=None):
    model = copy.deepcopy(initial)
    traces = []
    for phase in range(phases):
        support.released_utils.set_seed(seed + 10000 + phase)
        weights = torch.ones(data['n'], device=data['x'].device)
        selected = torch.empty(0, dtype=torch.long, device=data['x'].device)
        if budget is not None:
            z = logits_of(model, data['x'])
            selected = select_cache('disagreement', teacher, z, counts, max(1, int(data['n']*budget)), config['tau'])
            weights[selected] = .1
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        model.train()
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss = objective(model(None, data['x']), teacher, energy, data, config, weights)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite refinement loss')
            loss.backward()
            optimizer.step()
        traces.append(dict(phase=phase, steps=steps, cached_nodes=len(selected), last_loss=loss.item()))
    model.eval()
    return model, traces


@torch.no_grad()
def features(teacher, student):
    t, s = teacher.softmax(1), student.softmax(1)
    h = lambda p: -(p*p.clamp_min(1e-12).log()).sum(1)/math.log(p.shape[1])
    tt, ss = t.topk(2, dim=1).values, s.topk(2, dim=1).values
    kl = (t*(t.clamp_min(1e-12).log()-s.clamp_min(1e-12).log())).sum(1)
    return torch.stack([h(t), h(s), tt[:,0], ss[:,0], tt[:,0]-tt[:,1], ss[:,0]-ss[:,1], (t.argmax(1)==s.argmax(1)).float(), kl, h(t)-h(s)], 1)


@torch.no_grad()
def calibrate(teacher_validation, student_validation, validation_labels):
    """Receives validation-only arrays, never the test-label vector."""
    f = features(teacher_validation, student_validation).double()
    mean, scale = f.mean(0), f.std(0).clamp_min(1e-6)
    x = torch.cat([(f-mean)/scale, torch.ones((len(f),1), device=f.device, dtype=f.dtype)], 1)
    target = (teacher_validation.argmax(1)==validation_labels).double()-(student_validation.argmax(1)==validation_labels).double()
    penalty = torch.eye(x.shape[1], device=x.device, dtype=x.dtype)
    penalty[-1,-1] = 0
    coef = torch.linalg.solve(x.T@x+penalty, x.T@target)
    return dict(mean=mean, scale=scale, coef=coef)


@torch.no_grad()
def predict_benefit(teacher, student, fit):
    f = features(teacher, student).double()
    x = torch.cat([(f-fit['mean'])/fit['scale'], torch.ones((len(f),1), device=f.device, dtype=f.dtype)], 1)
    benefit = (x@fit['coef']).clamp(-1,1).float()
    benefit[teacher.argmax(1)==student.argmax(1)] = 0
    return benefit


@torch.no_grad()
def select_cache(policy, teacher, student, counts, k, tau, benefit=None):
    if policy == 'frequency':
        return counts.topk(k).indices
    if policy == 'disagreement':
        kl = F.kl_div((student/tau).log_softmax(1), (teacher/tau).log_softmax(1), reduction='none', log_target=True).sum(1).clamp_min(0)
        return (counts*kl).topk(k).indices
    if policy == 'benefit':
        scores = counts*benefit.clamp_min(0)
        selected = scores.topk(k).indices
        return selected[scores[selected]>0]
    raise ValueError(policy)


@torch.no_grad()
def unfairness(z, lap):
    return torch.trace(z.T@torch.sparse.mm(lap, z)).item()/1000


class ServingCache:
    def __init__(self, model, x, teacher, selected):
        self.model, self.x, self.classes = model, x, teacher.shape[1]
        self.size = len(selected)
        self.mapping = torch.full((len(x),), -1, dtype=torch.long, device=x.device)
        self.mapping[selected] = torch.arange(self.size, device=x.device)
        self.payload = teacher[selected].contiguous()

    def __call__(self, query):
        if self.size == 0:
            return self.model(None, self.x[query])
        if self.size == len(self.x):
            return self.payload[self.mapping[query]]
        slots = self.mapping[query]
        hits = slots>=0
        output = torch.empty((len(query), self.classes), device=self.x.device)
        output[hits] = self.payload[slots[hits]]
        output[~hits] = self.model(None, self.x[query[~hits]])
        return output


@torch.no_grad()
def timing(fn, query, repetitions=40):
    for _ in range(10):
        fn(query)
    torch.cuda.synchronize()
    elapsed = []
    for _ in range(repetitions):
        start = time.perf_counter()
        fn(query)
        torch.cuda.synchronize()
        elapsed.append((time.perf_counter()-start)*1000)
    return statistics.median(elapsed)


@torch.no_grad()
def evaluate(model, student, teacher, selected, data, popularity, query):
    mixed = student.clone()
    mixed[selected] = teacher[selected]
    serve = ServingCache(model, data['x'], teacher, selected)
    actual = serve(query)
    torch.testing.assert_close(actual, mixed[query], atol=1e-5, rtol=1e-4)
    agreement = (actual.argmax(1)==mixed[query].argmax(1)).float().mean().item()
    if agreement != 1.:
        raise RuntimeError('Serving path argmax differs from quality evaluation')
    test = data['test']
    correct = (mixed[test].argmax(1)==data['y'][test]).float()
    frequency = popularity[test]/popularity[test].sum()
    hit = torch.zeros(data['n'], device=data['x'].device)
    hit[selected] = 1
    return dict(accuracy=correct.mean().item(), weighted_accuracy=(correct*frequency).sum().item(), shifted_accuracy=(correct*(.5*frequency+.5*frequency.flip(0))).sum().item(), individual_unfairness=unfairness(mixed, data['lap']), probability_unfairness=unfairness(mixed.softmax(1),data['lap']), hit_rate=(hit[test]*frequency).sum().item(), query_batch_ms=timing(serve,query), cached_nodes=len(selected), payload_bytes=serve.payload.numel()*4, index_bytes=8*data['n'] if len(selected) else 0, model_bytes=sum(p.numel()*p.element_size() for p in model.parameters()), routing_argmax_agreement=agreement)


def run_seed(name, seed, data, args):
    source = BASE/name/f'seed{seed}'
    target = args.output/name/f'seed{seed}'
    target.mkdir(parents=True, exist_ok=True)
    config = json.loads((source/'config.json').read_text())
    manifest = dict(dataset=name, seed=seed, budgets=args.budgets, phases=args.phases, steps_per_phase=args.steps, cached_weight=.1, source_checkpoint_sha256=digest(source/'FAITH.pt'), teacher_checkpoint_sha256=digest(source/'teacher.pt'), source_config_sha256=digest(source/'config.json'), implementation_sha256=digest(Path(__file__)), model_source_sha256=digest(ROOT/'upstream/src_trans/models.py'), energy_source_sha256=digest(ROOT/'upstream/src_trans/utils_guide.py'), history_requests=100000, zipf_exponent=.8, calibration='all 500 validation labels, reused after checkpoint selection', version=1)
    if (target/'complete.json').exists():
        assert json.loads((target/'manifest.json').read_text())==manifest
        return json.loads((target/'metrics.json').read_text())
    support.write_json(target/'manifest.json',manifest)
    initial = support.released_models.Model(config, model_type='student').cuda()
    checkpoint = torch.load(source/'FAITH.pt',map_location='cuda')
    initial.load_state_dict(checkpoint['state'])
    initial.eval()
    base_logits = logits_of(initial,data['x'])
    torch.testing.assert_close(base_logits,checkpoint['logits'],atol=1e-5,rtol=1e-4)
    teacher_checkpoint = torch.load(source/'teacher.pt',map_location='cuda')
    teacher, energy = teacher_checkpoint['logits'],teacher_checkpoint['energy']
    source_metric = json.loads((source/'metrics.json').read_text())['methods']['FAITH']
    assert abs((base_logits[data['test']].argmax(1)==data['y'][data['test']]).float().mean().item()-source_metric['accuracy'])<1e-7
    assert abs(unfairness(base_logits,data['lap'])-source_metric['individual_unfairness'])<max(1e-5,abs(source_metric['individual_unfairness'])*1e-5)
    rng = np.random.default_rng(seed+500)
    pop = rng.permutation(np.arange(1,data['n']+1)).astype(float)**-.8
    pop /= pop.sum()
    history = rng.choice(data['n'],100000,p=pop)
    counts = torch.as_tensor(np.bincount(history,minlength=data['n'])+1,dtype=torch.float32,device='cuda')
    popularity = torch.as_tensor(pop,dtype=torch.float32,device='cuda')
    test = data['test'].cpu().numpy()
    query = torch.as_tensor(rng.choice(test,1024,p=pop[test]/pop[test].sum()),device='cuda')
    rows = []

    def add(model,z,selected,variant,policy,budget,training_budget=0,additional_steps=0):
        row = dict(dataset=name,seed=seed,variant=variant,policy=policy,budget=budget,training_cache_budget=training_budget,additional_steps=additional_steps)
        row.update(evaluate(model,z,teacher,selected,data,popularity,query))
        rows.append(row)

    def benefit_for(z,label):
        fit = calibrate(teacher[data['valid']],z[data['valid']],data['y'][data['valid']])
        support.write_json(target/f'{label}_calibration.json',{k:v.cpu().tolist() for k,v in fit.items()})
        return predict_benefit(teacher,z,fit)

    def usual(model,z,variant):
        benefit = benefit_for(z,variant)
        add(model,z,torch.empty(0,dtype=torch.long,device='cuda'),variant,'none',0,additional_steps=0 if variant=='frozen' else args.phases*args.steps)
        for b in args.budgets:
            k=max(1,int(data['n']*b))
            for policy in ['frequency','disagreement','benefit']:
                selected=select_cache(policy,teacher,z,counts,k,config['tau'],benefit)
                add(model,z,selected,variant,policy,b,additional_steps=0 if variant=='frozen' else args.phases*args.steps)

    start=time.perf_counter()
    usual(initial,base_logits,'frozen')
    print(json.dumps(dict(dataset=name,seed=seed,phase='frozen complete')),flush=True)
    control,trace=continue_student(initial,teacher,energy,data,config,counts,seed,args.steps,args.phases)
    control_z=logits_of(control,data['x'])
    torch.save(dict(state=control.state_dict(),logits=control_z.cpu()),target/'continued.pt')
    support.write_json(target/'continued_trace.json',trace)
    usual(control,control_z,'continued')
    for b in args.budgets:
        joint,trace=continue_student(initial,teacher,energy,data,config,counts,seed,args.steps,args.phases,b)
        z=logits_of(joint,data['x'])
        selected=select_cache('disagreement',teacher,z,counts,max(1,int(data['n']*b)),config['tau'])
        benefit=benefit_for(z,f'joint_{b}')
        torch.save(dict(state=joint.state_dict(),logits=z.cpu(),cache=selected.cpu()),target/f'joint_{b}.pt')
        support.write_json(target/f'joint_{b}_trace.json',trace)
        extra=args.phases*args.steps
        add(joint,z,torch.empty(0,dtype=torch.long,device='cuda'),'joint','none',0,b,extra)
        add(joint,z,selected,'joint','joint',b,b,extra)
        add(control,control_z,selected,'continued','joint_cache_control',b,b,extra)
        add(joint,z,select_cache('frequency',teacher,z,counts,len(selected),config['tau']),'joint','frequency',b,b,extra)
        add(joint,z,select_cache('benefit',teacher,z,counts,len(selected),config['tau'],benefit),'joint','benefit',b,b,extra)
    support.write_json(target/'metrics.json',rows)
    support.write_json(target/'complete.json',dict(rows=len(rows),seconds=time.perf_counter()-start,source_baseline_accuracy_and_unfairness_match=True))
    print(json.dumps(dict(dataset=name,seed=seed,phase='complete',rows=len(rows),seconds=round(time.perf_counter()-start,2))),flush=True)
    return rows


def write_report(rows,output):
    support.write_json(output/'metrics.json',rows)
    import csv
    with (output/'metrics.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    def table(variant,metric):
        lines=['| Dataset | No cache | Frequency | Disagreement | Predicted benefit |','|---|---:|---:|---:|---:|']
        for d in sorted({r['dataset'] for r in rows}):
            cells=[]
            for policy in ['none','frequency','disagreement','benefit']:
                vals=[100*r[metric] for r in rows if r['dataset']==d and r['variant']==variant and r['policy']==policy and r['budget']==(0 if policy=='none' else .1) and r['training_cache_budget']==0]
                cells.append(f'{statistics.mean(vals):.2f} ± {statistics.pstdev(vals):.2f}' if vals else '—')
            lines.append('| '+' | '.join([d]+cells)+' |')
        return '\n'.join(lines)
    text=['# Cache augmentation of the saved full FAITH students','','All five reproduction seeds are retained, including the low-accuracy CiteSeer seed. Teacher and frozen student checkpoints are unchanged. Evaluation uses the A100, static teacher-logit caches, four node-cache upper bounds, and synthetic Zipf requests. Results compare our local FAITH reproduction, not the paper’s unverified GLNN baseline.','','## Frozen FAITH at 10 percent cache','','Request-weighted accuracy in percent, mean ± population standard deviation across five training/workload seeds. No cache uses the same checkpoints and test popularity.','',table('frozen','weighted_accuracy'),'','Uniform test accuracy in percent:','',table('frozen','accuracy'),'','## Compute matched refinement comparison','','Independent continuation and joint refinement both add three phases of 50 full-graph Adam updates, with matched optimizer resets, initial weights, dropout seeds, and final-step evaluation. Joint selection uses frequency times KL; selected nodes receive 0.1 weights in CE/KL only. The complete FAITH energy and OSA losses remain unchanged. No continuation checkpoint is selected by test accuracy.','','| Dataset | Continued no cache | Continued frequency | Continued disagreement | Continued benefit | Joint cache | Joint benefit | Continued with joint cache |','|---|---:|---:|---:|---:|---:|---:|---:|']
    settings=[('continued','none',0,0),('continued','frequency',.1,0),('continued','disagreement',.1,0),('continued','benefit',.1,0),('joint','joint',.1,.1),('joint','benefit',.1,.1),('continued','joint_cache_control',.1,.1)]
    for d in sorted({r['dataset'] for r in rows}):
        cells=[]
        for variant,policy,b,tb in settings:
            vals=[100*r['weighted_accuracy'] for r in rows if r['dataset']==d and r['variant']==variant and r['policy']==policy and r['budget']==b and r['training_cache_budget']==tb]
            cells.append(f'{statistics.mean(vals):.2f} ± {statistics.pstdev(vals):.2f}' if vals else '—')
        text.append('| '+' | '.join([d]+cells)+' |')
    text += ['','## Protocol and limits','','Cache budgets are 1%, 5%, 10%, and 20% of all nodes. Float32 teacher logits use 4*k*C bytes and the dense int64 lookup uses 8*N bytes. The student parameters are reported separately. Predicted-benefit caches may leave slots empty if estimated benefit is nonpositive. Frequency uses 100,000 historical Zipf(0.8) requests with additive-one count smoothing. Cache selection uses all graph nodes, while accuracy and query timing use public test nodes.','', 'Benefit calibration is fixed ridge regression (penalty 1, unpenalized intercept), targeting teacher correctness minus student correctness on the 500 validation nodes. Features are prediction entropy, confidence, margins, agreement, and KL. These validation labels already selected the saved checkpoints; calibration is not an independently held-out experiment. Test labels never select a cache, checkpoint, or loss setting. No graph features or true nodewise test benefit enter cache scores.','', 'Reported paired accuracy uses the same synthetic popularity in every method for each seed. Uniform accuracy and a half-reversed popularity shift are also saved. Values are descriptive, with no significance claim. A strong improvement on the failed CiteSeer seed is recovery of an unstable base student, not proof of a general cache-training advantage.','', 'Individual unfairness uses the released graph-similarity Laplacian and full mixed logits, divided by 1,000. A second metric uses class probabilities to expose raw-logit scale effects. Hard teacher overrides do not automatically preserve FAITH’s fairness properties; quality and unfairness must be judged jointly.','', 'Timing uses 1,024 GPU-resident requests, 10 warmups, median of 40 synchronized calls. It includes lookup, output assembly, and miss-only student inference; it excludes host transfer, network, cache refresh, and offline teacher/cache construction. Query outputs and argmaxes are checked against quality evaluation. These are local microbenchmarks, not end-to-end serving measurements.','', 'The primary source-reproduction limitation remains: a verified CUDA message-sum adapter replaces unavailable native CUDA DGL. The serving MLP is the actual released student. Saved reproduction files remain untouched. The sourceFAITH pure node losses are reweighted only during our refinement; its graph energy and similarity objectives are kept intact.','', 'All budgets, latency, unfairness, actual memory, refinement traces, calibration coefficients, and source hashes are saved with the metrics.']
    metadata=json.loads((output/'environment.json').read_text())
    actual_schedule=f"Independent continuation and joint refinement both add {metadata['phases']} phases of {metadata['steps_per_phase']} full-graph Adam updates"
    document='\n'.join(text).replace('Independent continuation and joint refinement both add three phases of 50 full-graph Adam updates',actual_schedule).replace('The sourceFAITH pure node losses','The source FAITH node losses')
    (output/'REPORT.md').write_text(document+'\n')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--datasets',nargs='+',default=['cora','citeseer','pubmed'])
    parser.add_argument('--seeds',nargs='+',type=int,default=SEEDS)
    parser.add_argument('--steps',type=int,default=50)
    parser.add_argument('--phases',type=int,default=3)
    parser.add_argument('--budgets',nargs='+',type=float,default=[.01,.05,.1,.2])
    parser.add_argument('--output',type=Path,default=ROOT/'results/faith_cache')
    args=parser.parse_args()
    assert args.steps>0 and args.phases>0 and all(0<b<=1 for b in args.budgets)
    if not torch.cuda.is_available():raise RuntimeError('Host GPU access required')
    torch.set_num_threads(4)
    args.output.mkdir(parents=True,exist_ok=True)
    support.released_guide.F=F
    support.prepare_data()
    support.write_json(args.output/'environment.json',dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(),source_reproduction=str(BASE),source_reproduction_verified=json.loads((BASE/'independent_verification.json').read_text()),launcher_sha256=digest(Path(__file__)),steps_per_phase=args.steps,phases=args.phases,budgets=args.budgets))
    rows=[]
    for name in args.datasets:
        data=prepare(name)
        for seed in args.seeds:
            rows+=run_seed(name,seed,data,args)
            write_report(rows,args.output)
        del data
        torch.cuda.empty_cache()
    expected=len(args.datasets)*len(args.seeds)*(2+11*len(args.budgets))
    assert len(rows)==expected
    assert len({(r['dataset'],r['seed'],r['variant'],r['policy'],r['budget'],r['training_cache_budget']) for r in rows})==expected
    assert all(r['routing_argmax_agreement']==1 for r in rows)
    support.write_json(args.output/'verification.json',dict(rows=expected,dataset_seed_runs=len(args.datasets)*len(args.seeds),routing='passed',source_baseline='unchanged and matched',test_labels_used_for_selection=False,calibration_independence='validation reused after checkpoint selection; exploratory',additional_steps=args.steps*args.phases))
    print(f'Completed {expected} cache evaluations: {args.output/"REPORT.md"}',flush=True)


if __name__=='__main__':main()
