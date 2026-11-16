"""Check the repaired source training step against its literal GPU reference."""
from pathlib import Path
import copy
import json
import tempfile

import reproduce_hybrid as hybrid

torch, F, dgl = hybrid.torch, hybrid.F, hybrid.dgl
support = hybrid.support


def main():
    with tempfile.TemporaryDirectory(dir=hybrid.ROOT) as directory:
        source = hybrid.install(Path(directory))
        namespace = dict(support.released_training.__dict__)
        exec(compile(source, '<original-device-corrected-reference>', 'exec'), namespace)
        graph = dgl.add_self_loop(dgl.graph(([0, 1, 1, 2], [1, 0, 2, 1]), num_nodes=4))
        adjacency = graph.adjacency_matrix().to_dense().numpy()
        x = torch.randn(4, 3).cuda()
        y = torch.tensor([0, 1, 0, 1], device='cuda')
        teacher = torch.randn(4, 2, device='cuda')
        src, dst = graph.adjacency_matrix().indices().cuda()
        energy = torch.zeros(4, device='cuda').index_add(0, src, (teacher[src] - teacher[dst]).square().sum(1))
        delta = support.released_guide.GDO_optimized(support.released_guide.calculate_similarity_matrix(support.sp.csr_matrix(adjacency), x, metric='cosine'))
        param = dict(teacher='SAGE', student='MLP', num_layers=2, feat_dim=3, label_dim=2, hidden_dim=4, dropout_s=0., norm_type='none', lamb=.3, tau=1.1)
        model = support.released_models.Model(param, model_type='student').cuda()
        reference = copy.deepcopy(model)
        optimizer, optimizer_ref = [torch.optim.Adam(m.parameters(), lr=.01) for m in [model, reference]]
        support.released_training.beta, support.released_training.gamma = .02, 2e-5
        namespace['beta'], namespace['gamma'] = .02, 2e-5
        args = (graph.adjacency_matrix().indices(), x, y, teacher, torch.nn.NLLLoss(), torch.nn.KLDivLoss(reduction='batchmean', log_target=True))
        expected = namespace['train_mini_batch'](reference, *args, optimizer_ref, torch.arange(2), adjacency, delta, None, energy, param)
        support.STATIC.clear()
        actual = support.released_training.train_mini_batch(model, *args, optimizer, torch.arange(2), adjacency, delta, None, energy, param)
        hybrid.np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
        for a, b in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(a.grad, b.grad, rtol=1e-5, atol=1e-7)
            torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)
    result = dict(cuda_loss_equivalence=True, cuda_gradient_equivalence=True, cuda_optimizer_step_equivalence=True)
    (hybrid.ROOT / 'results/hybrid_reproduction/cuda_step_checks.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
