import unittest
import torch
from moirai_research.dgf_reference_math import hierarchical_advantages,complete_loss,mode_diversity,dynamic_terms
from moirai_research.crps_math import empirical_crps

class CoreMath(unittest.TestCase):
    def test_two_credit_axes_and_gradients(self):
        reward=torch.tensor([[[1.,2.,4.],[2.,5.,9.]]],dtype=torch.float64)
        sample,group=hierarchical_advantages(reward)
        self.assertTrue(torch.allclose(sample.mean(2),torch.zeros(1,2,dtype=torch.float64),atol=1e-12))
        self.assertTrue(torch.allclose(group.mean(1),torch.zeros(1,dtype=torch.float64),atol=1e-12))
        trajectory=torch.zeros_like(reward,requires_grad=True);direction=torch.zeros(1,2,dtype=torch.float64,requires_grad=True);diversity=torch.tensor(.1,dtype=torch.float64,requires_grad=True)
        loss,_=complete_loss(trajectory,direction,reward,diversity);loss.backward()
        self.assertGreater(float(trajectory.grad.norm()),0);self.assertGreater(float(direction.grad.norm()),0);self.assertLess(float(diversity.grad),0)
    def test_bounded_diversity_and_mask(self):
        means=torch.tensor([[[[0.],[1.],[2.],[3.]],[[.01],[1.02],[2.03],[3.04]]]],dtype=torch.float64,requires_grad=True)
        mask=torch.ones(1,4,1,dtype=torch.bool);scales=torch.ones(1,4,1,dtype=torch.float64)
        value,_=mode_diversity(means,mask,scales);self.assertGreater(float(value),0);self.assertLessEqual(float(value),1);value.backward();self.assertGreater(float(means.grad.norm()),0)
        truth=torch.arange(4,dtype=torch.float64).reshape(1,4,1);pred=truth[:,None,None].clone();mask[:,3]=False
        before=dynamic_terms(pred,truth,mask,torch.zeros(1,1),torch.ones(1,1,dtype=torch.bool));pred[...,3,:]=1e10
        after=dynamic_terms(pred,truth,mask,torch.zeros(1,1),torch.ones(1,1,dtype=torch.bool));self.assertTrue(torch.equal(before,after))
    def test_fair_crps_pair_correction(self):
        samples=torch.tensor([[0.],[1.],[2.]],dtype=torch.float64);truth=torch.tensor([1.],dtype=torch.float64)
        self.assertAlmostEqual(float(empirical_crps(samples,truth)),0.,places=12)
        self.assertAlmostEqual(float(empirical_crps(samples,truth,fair=False)),2/9,places=12)
if __name__=='__main__':unittest.main()
