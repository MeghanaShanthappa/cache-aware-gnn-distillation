import copy
import inspect
import unittest

import faith_cache as C

torch, np, F = C.torch, C.np, C.F


class SmallModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer=torch.nn.Linear(3,2)
        self.calls=0

    def forward(self, graph, features):
        self.calls+=1
        return self.layer(features)


class CacheTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(83)
        C.support.released_guide.F=F
        self.adj=C.support.sp.csr_matrix(np.array([[1,1,0,0],[1,1,1,0],[0,1,1,0],[0,0,0,1]],dtype=np.float32))
        self.sim=C.support.released_guide.calculate_similarity_matrix(self.adj,None,metric='cosine')

    def test_sparse_incidence_matches_released_dense_builder(self):
        expected=C.support.released_guide.GDO_optimized(self.sim)
        actual=C.incidence(self.sim)
        np.testing.assert_array_equal(actual.toarray(),expected.toarray())

    def test_uniform_objective_and_gradients_match_released_faith(self):
        delta=C.incidence(self.sim)
        adj=C.sparse_tensor(self.adj,'cpu')
        x=torch.randn(4,3)
        y=torch.tensor([0,1,0,1])
        teacher=torch.randn(4,2)
        src,dst=adj.indices()
        energy=torch.zeros(4).index_add(0,src,(teacher[src]-teacher[dst]).square().sum(1))
        data=dict(train=torch.tensor([0,1]),y=y,adjacency=adj,edges=adj.indices(),delta=C.sparse_tensor(delta,'cpu'))
        config=dict(dataset='cora',lamb=.3,tau=1.1)
        model=SmallModel()
        reference=copy.deepcopy(model)
        optimizer=torch.optim.Adam(model.parameters(),lr=.01)
        optimizer_ref=torch.optim.Adam(reference.parameters(),lr=.01)
        original=inspect.getsource(C.support.released_training.train_mini_batch).replace("device='cuda'",'device=logits.device')
        namespace=dict(C.support.released_training.__dict__)
        namespace.update(beta=.02,gamma=2e-5)
        exec(compile(original,'<released-reference>','exec'),namespace)
        parts=namespace['train_mini_batch'](reference,data['edges'],x,y,teacher,torch.nn.NLLLoss(),torch.nn.KLDivLoss(reduction='batchmean',log_target=True),optimizer_ref,data['train'],self.adj.toarray(),delta,None,energy,config)
        loss=C.objective(model(None,x),teacher,energy,data,config,torch.ones(4))
        self.assertAlmostEqual(loss.item(),sum(parts),places=6)
        loss.backward()
        optimizer.step()
        for a,b in zip(model.parameters(),reference.parameters()):
            torch.testing.assert_close(a.grad,b.grad,rtol=1e-5,atol=1e-7)
            torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-7)

    def test_full_cache_does_not_call_student(self):
        model=SmallModel()
        x=torch.randn(4,3)
        teacher=torch.randn(4,2)
        cache=C.ServingCache(model,x,teacher,torch.arange(4))
        query=torch.tensor([3,0,3,1])
        torch.testing.assert_close(cache(query),teacher[query],atol=0,rtol=0)
        self.assertEqual(model.calls,0)

    def test_partial_and_empty_cache_route_correctly(self):
        model=SmallModel().eval()
        x=torch.randn(4,3)
        teacher=torch.randn(4,2)
        baseline=model(None,x).detach()
        query=torch.tensor([3,0,3,1])
        for selected in [torch.tensor([0,3]),torch.empty(0,dtype=torch.long)]:
            mixed=baseline.clone();mixed[selected]=teacher[selected]
            cache=C.ServingCache(model,x,teacher,selected)
            torch.testing.assert_close(cache(query),mixed[query])

    def test_benefit_cache_obeys_cap_and_rejects_nonpositive_scores(self):
        teacher=torch.randn(4,2);student=torch.randn(4,2);counts=torch.ones(4)
        cache=C.select_cache('benefit',teacher,student,counts,2,1.,torch.tensor([.1,-.4,0.,.2]))
        self.assertEqual(set(cache.tolist()),{0,3})
        empty=C.select_cache('benefit',teacher,student,counts,2,1.,-torch.ones(4))
        self.assertEqual(len(empty),0)

    def test_agreeing_predictions_have_zero_estimated_benefit(self):
        teacher=torch.randn(20,3);student=teacher.clone();labels=torch.randint(0,3,(20,))
        fit=C.calibrate(teacher[:8],student[:8],labels[:8])
        result=C.predict_benefit(teacher,student,fit)
        self.assertTrue((result==0).all().item())


if __name__=='__main__':unittest.main(verbosity=2)
