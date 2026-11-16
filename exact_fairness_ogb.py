"""Exact quadratic graph unfairness without constructing all-pairs similarity.

For symmetric binary A, R=D^(-1/2)A and S=RR^T are adjacency-row
cosines. trace(Z^T L_S Z) = sum_i (S1)_i ||Z_i||^2 - ||R^T Z||_F^2.
Unlike the nonquadratic OSA objective, this diagnostic admits sparse products.
"""
import json
import statistics
from pathlib import Path

import faith_ogb as O

torch, np, sp = O.torch, O.np, O.sp
ROOT, OUT = O.ROOT, O.ROOT / 'results/faith_ogb'


def factors(adjacency, degree):
    root = degree.rsqrt()
    neighbor_root = torch.sparse.mm(adjacency, root[:, None])
    similarity_degree = root * torch.sparse.mm(adjacency, neighbor_root)[:, 0]
    return root, similarity_degree


@torch.no_grad()
def unfairness(z, adjacency, root, similarity_degree):
    projection = torch.sparse.mm(adjacency, root[:, None] * z)
    return ((similarity_degree[:, None] * z.square()).sum() - projection.square().sum()).item() / 1000


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('Host GPU access required')
    torch.set_num_threads(4)
    rows = json.loads((OUT / 'metrics.json').read_text())
    assert len(rows) == 460
    result = []
    for dataset in ['ogbn-arxiv', 'ogbn-products']:
        a = sp.load_npz(ROOT / 'ogb-data' / dataset / 'adjacency.npz').astype(np.float64)
        a.data.fill(1)
        adjacency = O.csr_tensor(a, 'cuda')
        degree = torch.as_tensor(np.diff(a.indptr), dtype=torch.float64, device='cuda')
        root, similarity_degree = factors(adjacency, degree)
        del a
        for seed in O.SEEDS:
            folder = OUT / dataset / f'seed{seed}'
            teacher = torch.load(folder / 'teacher.pt', map_location='cpu')['logits'].to(device='cuda', dtype=torch.float64)
            local = []
            for label in ['FAITH', 'continued'] + [f'joint_{b}' for b in O.BUDGETS]:
                student = torch.load(folder / f'{label}.pt', map_location='cpu')['logits'].to(device='cuda', dtype=torch.float64)
                for row in rows:
                    if row['dataset'] != dataset or row['seed'] != seed:
                        continue
                    row_label = 'FAITH' if row['variant'] == 'frozen' else 'continued' if row['variant'] == 'continued' else f"joint_{row['training_cache_budget']}"
                    if row_label != label:
                        continue
                    selected = torch.as_tensor(np.load(folder / row['cache_file'])['selected'], device='cuda')
                    mixed = student.clone()
                    mixed[selected] = teacher[selected]
                    raw = unfairness(mixed, adjacency, root, similarity_degree)
                    # Match the original float32 softmax definition, then evaluate
                    # the quadratic form in float64 to reduce cancellation.
                    probability = unfairness(mixed.float().softmax(1).double(), adjacency, root, similarity_degree)
                    local.append({k: row[k] for k in ['dataset', 'seed', 'variant', 'policy', 'budget', 'training_cache_budget']} |
                                 dict(exact_individual_unfairness=raw, exact_probability_unfairness=probability,
                                      sampled_individual_unfairness=row['individual_unfairness'],
                                      sampled_probability_unfairness=row['probability_unfairness']))
                    del mixed
                del student
            assert len(local) == 46
            O.write(folder / 'exact_fairness.json', local)
            result.extend(local)
            del teacher
            print(f'Exact sparse fairness completed: {dataset} seed {seed}', flush=True)
        del adjacency, degree, root, similarity_degree
        torch.cuda.empty_cache()
    O.write(OUT / 'exact_fairness.json', result)
    O.write(OUT / 'exact_fairness_verification.json', dict(rows=len(result),
            method='Exact adjacency-row cosine Laplacian quadratic form via sparse factorization',
            accumulation='float64', probability_definition='float32 softmax followed by float64 graph quadratic form',
            metric_scale='divide by 1000, as in citation analysis',
            metrics_sha256=O.C.digest(OUT / 'metrics.json'), implementation_sha256=O.C.digest(Path(__file__)),
            training_unchanged=True, dense_small_graph_equivalence='unit tested'))
    lines = ['# Exact full-graph unfairness supplement', '',
             'The quadratic Laplacian metric admits exact sparse factorization even though the nonquadratic OSA training objective requires sampling. These values cover every graph node and cosine-similar pair. They supplement the sampled diagnostics in the original metrics without changing training, cache selection, accuracy or source artifacts. Float64 arithmetic reduces cancellation. Lower is better; values are divided by 1,000.', '',
             '| Dataset | Policy at 10% | Raw-logit unfairness | Probability unfairness |',
             '|---|---|---:|---:|']
    for dataset in ['ogbn-arxiv', 'ogbn-products']:
        for policy in ['none', 'frequency', 'disagreement', 'benefit']:
            group = [r for r in result if r['dataset'] == dataset and r['variant'] == 'frozen' and r['policy'] == policy and r['budget'] == (0 if policy == 'none' else .1)]
            lines.append(f"| {dataset} | {policy} | {statistics.mean(r['exact_individual_unfairness'] for r in group):.3f} | {statistics.mean(r['exact_probability_unfairness'] for r in group):.3f} |")
    (OUT / 'EXACT_FAIRNESS.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
