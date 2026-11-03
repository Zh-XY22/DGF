"""Forecast objectives with component-labelled mixed discrete/continuous actions."""
import torch
from torch.distributions import Categorical,Normal,StudentT,LogNormal
from uni2ts.distribution.mixture import Mixture
from uni2ts.distribution.negative_binomial import NegativeBinomial

class NonfiniteForecastSample(FloatingPointError):
    """A sampled forecast is unusable; retain it without substituting values."""
    def __init__(self, forecast):
        super().__init__('Nonfinite sampled forecast; no dropping or clipping')
        self.forecast = forecast.detach()

class JointForecast:
    def __init__(self, mixture, loc, scale):
        assert isinstance(mixture,Mixture) and not mixture.has_rsample
        self.mixture,self.loc,self.scale=mixture,loc,scale
        assert not torch.as_tensor(loc).requires_grad and not torch.as_tensor(scale).requires_grad
        assert torch.isfinite(torch.as_tensor(scale)).all() and (torch.as_tensor(scale)>0).all()

    @classmethod
    def from_packed(cls,distr,batch):
        mask=batch['prediction_mask'][...,None]&batch['observed_mask']
        counts=mask.flatten(1).sum(1);assert (counts==counts[0]).all()
        b,n=len(mask),int(counts[0]);shape=distr.base_dist.batch_shape
        def pick(x):return torch.as_tensor(x,device=mask.device).expand(shape)[mask].reshape(b,n)
        components=[]
        for c in distr.base_dist.components:
            if isinstance(c,StudentT):params=dict(df=pick(c.df),loc=pick(c.loc),scale=pick(c.scale))
            elif isinstance(c,(Normal,LogNormal)):params=dict(loc=pick(c.loc),scale=pick(c.scale))
            elif isinstance(c,NegativeBinomial):params=dict(total_count=pick(c.total_count),logits=pick(c.logits))
            else:raise TypeError(type(c))
            components.append(type(c)(**params,validate_args=False))
        logits=distr.base_dist.weights.logits[mask].reshape(b,n,-1)
        joint=cls(Mixture(Categorical(logits=logits),components),pick(distr.loc),pick(distr.scale))
        target=batch['target'][mask].reshape(b,n)
        return joint,target

    @property
    def mean(self):return self.loc+self.scale*self.mixture.mean
    @property
    def variance(self):return self.scale.square()*self.mixture.variance
    def sample(self,g):
        with torch.no_grad():
            z=self.mixture.weights.sample((g,))
            all_y=torch.stack([c.sample((g,)) for c in self.mixture.components],-1)
            raw=all_y.gather(-1,z.unsqueeze(-1)).squeeze(-1)
            y=self.loc+self.scale*raw
            if not torch.isfinite(y).all():raise NonfiniteForecastSample(y)
        return dict(component=z,raw=raw,forecast=y)
    def log_prob(self,action):
        z,raw=action['component'],action['raw']
        assert not raw.requires_grad and not action['forecast'].requires_grad
        logits=self.mixture.weights.logits.expand(raw.shape+(len(self.mixture.components),))
        result=logits.gather(-1,z.unsqueeze(-1)).squeeze(-1)
        for i,c in enumerate(self.mixture.components):
            # Unselected branches receive a valid deterministic support value.
            fallback=torch.zeros_like(c.mean) if isinstance(c,NegativeBinomial) else c.mean.detach()
            safe=torch.where(z==i,raw,fallback)
            lp=c.log_prob(safe)
            if not isinstance(c,NegativeBinomial):lp=lp-self.scale.log()
            result=result+torch.where(z==i,lp,torch.zeros_like(lp))
        assert torch.isfinite(result).all()
        return result

def expected_mse(joint,target,g,objective):
    divisor=g if objective=='mean_prediction' else 1
    value=(joint.mean-target).square()+joint.variance/divisor
    assert torch.isfinite(value).all(), 'Non-finite predictive second moment'
    return value.mean()

def score_loss(joint,target,action,objective):
    """Coordinate score estimator; independent-coordinate terms are integrated out."""
    y=action['forecast'];g=y.shape[0];lp=joint.log_prob(action)
    if objective=='sample_mean_error':
        errors=(y-target).square()
        baseline=(errors.sum(0,keepdim=True)-errors)/(g-1) if g>1 else torch.zeros_like(errors)
        loss=((errors-baseline).detach()*lp).mean()
        measured=errors.mean()
    elif objective=='mean_prediction':
        others=y.sum(0,keepdim=True)-y
        error=(y.mean(0)-target).square()
        baseline=(others/g-target).square()
        # Sum over G scores: averaging this axis would incorrectly divide the gradient by G.
        loss=((error.unsqueeze(0)-baseline).detach()*lp).sum(0).mean()
        measured=error.mean()
    else:raise ValueError(objective)
    assert torch.isfinite(loss)
    return loss,measured

def stable_verl_kl(logprob, ref_logprob):
    """Same verl clipped low_var_kl, with bounded exponent before evaluation."""
    from verl.trainer.ppo.core_algos import kl_penalty
    # exp(d)-d-1 exceeds 10 at both endpoints and everywhere beyond them.
    # Bounding d there preserves the clipped value and its zero derivative.
    delta=(ref_logprob.double()-logprob.double()).clamp(-12.,4.)
    return kl_penalty(torch.zeros_like(delta),delta,'low_var_kl').to(logprob.dtype)
