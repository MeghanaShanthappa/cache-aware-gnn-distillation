"""Audit the CUDA adapter against native DGL on the failed run's full graph."""
import copy
import json
import torch
import dgl
import reproduce_cuda as R


def main():
    torch.set_num_threads(4)
    folder = R.ROOT / 'results/cuda_reproduction/citeseer/seed45'
    graph = dgl.data.CiteseerGraphDataset(raw_dir=str(R.ROOT / 'dgl-reproduction-data'), verbose=False)[0]
    graph = dgl.add_self_loop(dgl.remove_self_loop(graph))
    config = json.loads((folder / 'config.json').read_text())
    saved = torch.load(folder / 'teacher.pt', map_location='cpu')
    cpu = R.support.released_models.Model(config, model_type='teacher').eval()
    cpu.load_state_dict(saved['state'])
    gpu = copy.deepcopy(cpu).cuda()
    expected = cpu(graph, graph.ndata['feat'])
    actual = gpu(R.SparseCUDAProxy(graph), graph.ndata['feat'].cuda())
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-4, atol=2e-5)
    mask = graph.ndata['train_mask']
    labels = graph.ndata['label']
    torch.nn.functional.cross_entropy(expected[mask], labels[mask]).backward()
    torch.nn.functional.cross_entropy(actual[mask.cuda()], labels[mask].cuda()).backward()
    maximum_gradient_error = 0.
    for a, b in zip(cpu.parameters(), gpu.parameters()):
        torch.testing.assert_close(b.grad.cpu(), a.grad, rtol=1e-3, atol=2e-6)
        maximum_gradient_error = max(maximum_gradient_error, (b.grad.cpu()-a.grad).abs().max().item())
    delta = actual.detach().cpu()-expected.detach()
    output = dict(dataset='citeseer', seed=45, full_graph_forward_close=True, full_graph_parameter_gradients_close=True, forward_max_abs_error=delta.abs().max().item(), forward_relative_l2_error=(delta.norm()/expected.detach().norm()).item(), maximum_parameter_gradient_abs_error=maximum_gradient_error, argmax_agreement=(actual.argmax(1).cpu()==expected.argmax(1)).float().mean().item())
    (R.ROOT / 'results/cuda_reproduction/real_graph_operator_check.json').write_text(json.dumps(output,indent=2))
    print(json.dumps(output,indent=2))


if __name__ == '__main__':
    main()
