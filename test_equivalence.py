"""Scientific correctness checks for the structural-loss adaptation."""
import ast
import copy
import unittest

import torch
from torch.nn import functional as F

import experiment as exp


class EquivalenceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        self.edges = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
        self.coo, self.adj, self.mean, self.degree = exp.make_graph(self.edges, 4, 'cpu')

    def test_released_energy_loss_values_and_gradients(self):
        source = (exp.UPSTREAM / 'src_trans/utils_guide.py').read_text()
        tree = ast.parse(source)
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'compute_loss_energy')
        namespace = {'torch': torch, 'F': F}
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<upstream-energy-function>', 'exec'), namespace)
        teacher = torch.randn(4, 3)
        z = torch.randn(4, 3, requires_grad=True)
        z_ref = z.detach().clone().requires_grad_()
        src, dst = self.coo.indices()
        reference_teacher = torch.zeros(4).index_add(0, src, (teacher[src] - teacher[dst]).square().sum(1))
        teacher_energy = exp.node_energy(teacher, self.adj, self.degree)
        torch.testing.assert_close(teacher_energy, reference_teacher, atol=2e-6, rtol=2e-6)
        actual = exp.structural_loss(z, self.adj, self.degree, teacher_energy, self.coo.indices())
        expected = namespace['compute_loss_energy'](self.coo, z_ref, reference_teacher, self.coo.indices(), 3)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
        actual.backward()
        expected.backward()
        torch.testing.assert_close(z.grad, z_ref.grad, atol=3e-6, rtol=3e-5)

    def test_teacher_aggregation_and_gradients(self):
        config = dict(hidden_dim=5, num_layers=2, dropout_t=0, dropout_s=0)
        model = exp.Network(3, 2, config, teacher=True).eval()
        reference = copy.deepcopy(model)
        x = torch.randn(4, 3)
        output = model(x, self.mean)
        # DGL gcn aggregator: (sum incoming + self)/(in_degree + 1).
        dense = self.coo.to_dense()
        h = x
        for i, layer in enumerate(reference.layers):
            h = layer((dense @ h + h) / (self.degree + 1)[:, None])
            if i == 0:
                h = F.relu(h)
        torch.testing.assert_close(output, h)
        output.square().sum().backward()
        h.square().sum().backward()
        for left, right in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(left.grad, right.grad)

    def test_benefit_fit_does_not_use_other_labels(self):
        teacher, student = torch.randn(20, 3), torch.randn(20, 3)
        labels = torch.randint(0, 3, (20,))
        calibration = torch.arange(20) < 8
        a, _ = exp.fit_benefit(teacher, student, labels, calibration)
        changed = labels.clone()
        changed[~calibration] = (changed[~calibration] + 1) % 3
        b, _ = exp.fit_benefit(teacher, student, changed, calibration)
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        self.assertTrue((a[teacher.argmax(1) == student.argmax(1)] == 0).all())


if __name__ == '__main__':
    unittest.main(verbosity=2)
