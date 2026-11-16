"""Independent checks for cache correctness and leakage-sensitive plumbing."""
import unittest
import numpy as np
import torch
import torch.nn.functional as F
import leaderboard_common as C


class CacheChecks(unittest.TestCase):
    def test_budget_and_nonpositive_benefit(self):
        t=torch.randn(20,3);s=torch.randn(20,3)
        chosen=C.select('benefit',t,s,torch.ones(20),.1,torch.full((20,),-1.))
        self.assertEqual(len(chosen),0)
        chosen=C.select('frequency',t,s,torch.arange(20.),.1)
        self.assertEqual(set(chosen.tolist()),{18,19})

    def test_teacher_override_and_empty_cache(self):
        torch.manual_seed(11)
        model=C.Student(4,3).eval();x=torch.randn(12,4);t=torch.randn(12,3)
        chosen=torch.tensor([2,7]);ids=torch.tensor([7,4,7,2,11])
        route=C.ServingCache(model,t,chosen)
        expected=model(x[ids]);expected[[0,2]]=t[7];expected[3]=t[2]
        torch.testing.assert_close(route(ids,x),expected)
        empty=C.ServingCache(model,t,torch.empty(0,dtype=torch.long))
        self.assertIsNone(empty.lookup)
        torch.testing.assert_close(empty(ids,x),model(x[ids]))

    def test_frequency_ties_do_not_follow_contiguous_node_ids(self):
        t=torch.zeros(1000,3);s=t.clone();counts=torch.ones(1000)
        first=C.select('frequency',t,s,counts,.1)
        second=C.select('frequency',t,s,counts,.1)
        torch.testing.assert_close(first,second)
        self.assertTrue(20<(first<500).sum().item()<80)
        counts[999]=2
        self.assertIn(999,C.select('frequency',t,s,counts,.1).tolist())

    def test_benefit_agreeing_predictions_are_zero(self):
        torch.manual_seed(12)
        t=torch.randn(100,4);s=torch.randn(100,4);y=torch.randint(4,(100,))
        fit=C.fit_benefit(t[:50],s[:50],y[:50])
        predicted=C.predict_benefit(t,s,fit)
        self.assertTrue(torch.all(predicted[t.argmax(1)==s.argmax(1)]==0))
        self.assertTrue(torch.isfinite(predicted).all())

    def test_splits_and_product_node_order(self):
        for name in ['ogbn-arxiv','ogbn-products']:
            split=C.load_split(name)
            self.assertFalse(np.intersect1d(split['calibration'],split['model_valid']).size)
            np.testing.assert_array_equal(np.sort(np.concatenate([split['calibration'],split['model_valid']])),np.sort(split['valid']))
            if name=='ogbn-products':
                # Official SAGN assumes contiguous original product IDs.
                np.testing.assert_array_equal(np.concatenate([split['train'],split['valid'],split['test']]),np.arange(2449029))

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_sparse_preprocessing_matches_dense_and_training_labels_only(self):
        a=torch.tensor([[0.,1.,1.,0.],[1.,0.,0.,0.],[1.,0.,0.,1.],[0.,0.,1.,0.]])
        a=a/a.sum(1,keepdim=True)
        x=torch.randn(4,5)
        actual=x.cuda();matrix=a.cuda().to_sparse_csr()
        expected=x.clone()
        for _ in range(3):
            actual=torch.sparse.mm(matrix,actual);expected=a@expected
        torch.testing.assert_close(actual.cpu(),expected,rtol=1e-5,atol=1e-6)
        y=torch.zeros(4,2);y[0,1]=1
        self.assertEqual(y[1:].sum(),0)
        actual=y.cuda();expected=y
        for _ in range(9):
            actual=torch.sparse.mm(matrix,actual);expected=a@expected
        torch.testing.assert_close(actual.cpu(),expected,rtol=1e-5,atol=1e-6)


if __name__=='__main__':unittest.main()
