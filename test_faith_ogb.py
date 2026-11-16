import copy
import unittest

import faith_ogb as O
import exact_fairness_ogb as E

torch, np, sp = O.torch, O.np, O.sp


class LargeGraphTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        self.a = sp.csr_matrix(np.array([[1, 1, 1, 0], [1, 1, 0, 0],
                                         [1, 0, 1, 1], [0, 0, 1, 1]], dtype=np.float32))

    def test_csr_sage_matches_native_dgl_forward_and_gradients(self):
        dgl = O.C.runtime.dgl
        src, dst = self.a.nonzero()
        graph = dgl.graph((src, dst), num_nodes=4)
        config = dict(teacher='SAGE', student='MLP', num_layers=3, feat_dim=3,
                      label_dim=2, hidden_dim=7, dropout_t=0, norm_type='none')
        native = O.C.support.released_models.Model(config, model_type='teacher').eval()
        sparse = copy.deepcopy(native)
        x = torch.randn(4, 3, requires_grad=True)
        xx = x.detach().clone().requires_grad_()
        proxy = O.CSRGraph(O.csr_tensor(self.a, 'cpu'), torch.tensor(np.diff(self.a.indptr)))
        a, b = native(graph, x), sparse(proxy, xx)
        torch.testing.assert_close(a, b)
        a.square().sum().backward(); b.square().sum().backward()
        torch.testing.assert_close(x.grad, xx.grad)
        for p, q in zip(native.parameters(), sparse.parameters()):
            torch.testing.assert_close(p.grad, q.grad)

        # Constant first aggregation is an offline training optimization only.
        sparse.zero_grad()
        native.zero_grad()
        fixed = x.detach()
        proxy.constant_features = fixed
        proxy.constant_sum = torch.sparse.mm(proxy.adj, fixed)
        a, b = native(graph, fixed), sparse(proxy, fixed)
        torch.testing.assert_close(a, b)
        a.square().sum().backward(); b.square().sum().backward()
        for p, q in zip(native.parameters(), sparse.parameters()):
            torch.testing.assert_close(p.grad, q.grad)

    def test_energy_value_and_gradient_match_released_edge_loss(self):
        O.C.support.released_guide.F = O.F
        z = torch.randn(4, 3, requires_grad=True)
        zz = z.detach().clone().requires_grad_()
        teacher = torch.randn(4, 3)
        adjacency = O.C.sparse_tensor(self.a, 'cpu')
        src, dst = adjacency.indices()
        teacher_energy = torch.zeros(4).index_add(0, src, (teacher[src] - teacher[dst]).square().sum(1))
        degree = torch.tensor(np.diff(self.a.indptr), dtype=torch.float32)
        actual_energy = O.node_energy(teacher, adjacency, degree)
        torch.testing.assert_close(actual_energy, teacher_energy, atol=1e-5, rtol=1e-5)
        data = dict(adjacency=adjacency, degree=degree,
                    adj_teacher_energy=torch.sparse.mm(adjacency, teacher_energy[:, None]).squeeze(1),
                    teacher_edge_norm=(degree * teacher_energy.square()).sum().sqrt())
        original = O.C.support.released_guide.compute_loss_energy(adjacency, z, teacher_energy, adjacency.indices(), 2)
        adapted = O.energy_loss(zz, teacher_energy, data)
        torch.testing.assert_close(original, adapted, atol=1e-6, rtol=1e-5)
        original.backward(); adapted.backward()
        torch.testing.assert_close(z.grad, zz.grad, atol=1e-6, rtol=1e-5)

    def test_wedge_importance_expectation_equals_exact_osa_and_unfairness(self):
        # Enumerate every wedge rather than checking a noisy empirical mean.
        z = torch.randn(4, 3, requires_grad=True)
        degree = np.diff(self.a.indptr)
        total = sum(int(d) * (int(d) - 1) // 2 for d in degree)
        left, right, osa_weights, fairness_weights = [], [], [], []
        for center in range(4):
            neighbors = self.a.indices[self.a.indptr[center]:self.a.indptr[center+1]]
            for n, i in enumerate(neighbors):
                for j in neighbors[n+1:]:
                    common = self.a[i].multiply(self.a[j]).sum()
                    sim = common / np.sqrt(float(degree[i] * degree[j]))
                    left.append(i); right.append(j)
                    osa_weights.append(np.sqrt(sim) * total / common)
                    fairness_weights.append(sim * total / common)
        pairs = O.device_pairs(dict(left=np.array(left), right=np.array(right),
                                     osa_weight=np.array(osa_weights), fairness_weight=np.array(fairness_weights)), 'cpu')
        similarity = O.C.support.released_guide.calculate_similarity_matrix(self.a, None, metric='cosine')
        delta = O.C.sparse_tensor(O.C.incidence(similarity), 'cpu')
        exact = torch.linalg.vector_norm(torch.sparse.mm(delta, z), dim=1).sum()
        expected = O.sampled_osa(z, pairs)
        torch.testing.assert_close(exact, expected, atol=1e-6, rtol=1e-5)
        a = torch.autograd.grad(exact, z, retain_graph=True)[0]
        b = torch.autograd.grad(expected, z)[0]
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
        lap = O.C.sparse_tensor(O.C.support.released_guide.laplacian(similarity), 'cpu')
        fairness = torch.trace(z.T @ torch.sparse.mm(lap, z)).item() / 1000
        self.assertAlmostEqual(fairness, O.sampled_unfairness(z, pairs), places=6)

    def test_exact_sparse_fairness_matches_dense_laplacian(self):
        adjacency = O.csr_tensor(self.a.astype(np.float64), 'cpu')
        degree = torch.tensor(np.diff(self.a.indptr), dtype=torch.float64)
        root, sim_degree = E.factors(adjacency, degree)
        z = torch.randn(4, 3, dtype=torch.float64)
        r = torch.tensor(self.a.toarray(), dtype=torch.float64) * degree.rsqrt()[:, None]
        s = r @ r.T
        lap = torch.diag(s.sum(1)) - s
        exact = torch.trace(z.T @ lap @ z).item() / 1000
        self.assertAlmostEqual(exact, E.unfairness(z, adjacency, root, sim_degree), places=12)


if __name__ == '__main__':
    unittest.main(verbosity=2)
