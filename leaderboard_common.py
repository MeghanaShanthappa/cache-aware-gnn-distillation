"""Shared, auditable protocol for top-ten OGB teacher/cache experiments."""
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.matplotlib-leaderboard'))
(ROOT/'.matplotlib-leaderboard').mkdir(exist_ok=True)
GPU_SITE = ROOT / 'leaderboard-upstream/dgl-conda/lib/python3.10/site-packages'
if GPU_SITE.exists():
    sys.path.insert(0, str(GPU_SITE))
import numpy as np
import torch
import torch.nn.functional as F

OUT = ROOT / 'results/leaderboard_cache'
DATA = ROOT / 'leaderboard-data'


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.pending')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as s:
        for b in iter(lambda: s.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def load_split(name):
    split = dict(np.load(ROOT / 'ogb-data' / name / 'splits.npz'))
    v = np.random.default_rng(1729).permutation(split['valid'])
    split['model_valid'], split['calibration'] = np.array_split(v, 2)
    for left, right in [('train', 'valid'), ('valid', 'test'), ('model_valid', 'calibration')]:
        assert not np.intersect1d(split[left], split[right]).size
    return split


def seed_all(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    torch.set_num_threads(4)
    # Full float32 arithmetic rather than TF32 for comparison to released code.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def metadata(name, seed, config):
    import subprocess
    upstream = 'AGDN' if name == 'ogbn-arxiv' else 'SCR'
    folder = ROOT / 'leaderboard-upstream' / upstream
    commit = subprocess.check_output(['git', '-C', str(folder), 'rev-parse', 'HEAD'], text=True).strip()
    sources = {str(p.relative_to(ROOT)): digest(p) for p in folder.rglob('*.py')}
    archive=OUT/'source_snapshots'
    archive.mkdir(parents=True,exist_ok=True)
    for script in ['leaderboard_teacher.py','leaderboard_common.py','leaderboard_distill.py']:
        source_path=ROOT/script
        (archive/(digest(source_path)+'-'+script)).write_bytes(source_path.read_bytes())
    return dict(dataset=name, seed=seed, config=config, torch=torch.__version__,
                gpu=torch.cuda.get_device_name(), upstream_commit=commit, source_sha256=sources,
                launcher_sha256=digest(ROOT / 'leaderboard_teacher.py'),
                common_sha256=digest(Path(__file__)),
                feature_sha256=digest(DATA / (name[5:] + '-giant.npy')),
                model_selection='First half of shuffled official validation (seed 1729)',
                cache_calibration='Other half of official validation; never used for checkpoint selection',
                test_selection=False)


class Student(torch.nn.Module):
    def __init__(self, inputs, classes):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(inputs, 256), torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(), torch.nn.Dropout(.1), torch.nn.Linear(256, 256),
            torch.nn.BatchNorm1d(256), torch.nn.ReLU(), torch.nn.Dropout(.1),
            torch.nn.Linear(256, classes))

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def infer(model, x, batch=65536):
    model.eval()
    return torch.cat([model(x[i:i+batch]) for i in range(0, len(x), batch)])


@torch.no_grad()
def benefit_features(t, s):
    tp, sp = t.softmax(1), s.softmax(1)
    entropy = lambda p: -(p * p.clamp_min(1e-12).log()).sum(1) / np.log(p.shape[1])
    tt, ss = tp.topk(2, 1).values, sp.topk(2, 1).values
    kl = (tp * (tp.clamp_min(1e-12).log() - sp.clamp_min(1e-12).log())).sum(1)
    return torch.stack([entropy(tp), entropy(sp), tt[:,0], ss[:,0], tt[:,0]-tt[:,1],
        ss[:,0]-ss[:,1], (tp.argmax(1)==sp.argmax(1)).float(), kl, entropy(tp)-entropy(sp)], 1)


@torch.no_grad()
def fit_benefit(t, s, labels):
    features = benefit_features(t, s).double()
    mean, scale = features.mean(0), features.std(0).clamp_min(1e-6)
    x = torch.cat([(features-mean)/scale, torch.ones((len(s),1), device=s.device)], 1)
    target = (t.argmax(1)==labels).double() - (s.argmax(1)==labels).double()
    penalty = torch.eye(x.shape[1], device=x.device, dtype=x.dtype)
    penalty[-1,-1] = 0
    return dict(mean=mean, scale=scale, coef=torch.linalg.solve(x.T@x+penalty, x.T@target))


@torch.no_grad()
def predict_benefit(t, s, fit):
    f = benefit_features(t, s).double()
    x = torch.cat([(f-fit['mean'])/fit['scale'], torch.ones((len(s),1), device=s.device)], 1)
    result = (x@fit['coef']).clamp(-1,1).float()
    result[t.argmax(1)==s.argmax(1)] = 0
    return result


@torch.no_grad()
def select(policy, t, s, counts, budget, benefit=None):
    k = int(len(t)*budget)
    if policy == 'frequency':
        # Most products have equal counts with only 100k history requests.
        # Break ties independently of node-ID ordering (products IDs are grouped
        # by train/validation/test), while preserving every strict count order.
        generator=torch.Generator(device=counts.device).manual_seed(7341)
        score=counts.double()+.25*torch.rand(counts.shape,generator=generator,
            device=counts.device,dtype=torch.float64)
    elif policy == 'disagreement':
        score = counts * F.kl_div(s.log_softmax(1), t.log_softmax(1),
            reduction='none', log_target=True).sum(1).clamp_min(0)
    elif policy == 'benefit':
        score = counts * benefit.clamp_min(0)
    else:
        raise ValueError(policy)
    chosen = score.topk(k).indices
    return chosen[score[chosen]>0] if policy == 'benefit' else chosen


class ServingCache:
    def __init__(self, model, teacher, chosen):
        self.model = model.eval()
        self.lookup = None
        self.values = None
        if len(chosen):
            self.lookup = torch.full((len(teacher),), -1, device=teacher.device, dtype=torch.long)
            self.lookup[chosen] = torch.arange(len(chosen), device=teacher.device)
            self.values = teacher[chosen].clone()

    @torch.no_grad()
    def __call__(self, ids, features):
        if self.lookup is None:
            return self.model(features[ids])
        slots = self.lookup[ids]
        hit = slots >= 0
        output = self.values.new_empty((len(ids), self.values.shape[1]))
        if hit.any():
            output[hit] = self.values[slots[hit]]
        if (~hit).any():
            output[~hit] = self.model(features[ids[~hit]])
        return output
