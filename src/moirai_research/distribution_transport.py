"""Lawful location/scale transports of a frozen native reference distribution.

Moving atoms preclude ordinary likelihood-ratio PPO and a common-measure NLL
comparison. These controls use explicitly different, legitimate pathwise or
moment gradients. They are never silently routed through native PPO/KL.
"""
import torch
from torch import nn
from .actions import JointForecast
from .policy import Policy
from .io_utils import load_checkpoint


class TransportJoint(JointForecast):
    moving_atomic_support=True

    def __init__(self,reference,mode,mean=None,log_scale=None):
        self.reference=reference;self.mixture=reference.mixture;self.transport_mode=mode
        if mode=='P04_LOCATION':
            if mean is None or not torch.isfinite(mean).all():raise FloatingPointError('Invalid transported prediction mean')
            self.loc=reference.loc+(mean-reference.mean);self.scale=reference.scale;self._mean=mean
        elif mode=='P05_SCALE':
            factor=log_scale.exp()
            if not torch.isfinite(factor).all() or not (factor>0).all():raise FloatingPointError('Invalid positive calibration scale')
            self.loc=reference.loc+(1-factor)*(reference.mean-reference.loc)
            self.scale=factor*reference.scale;self._mean=reference.mean
        else:raise ValueError('Unknown distribution transport')

    @property
    def mean(self):return self._mean

    @property
    def variance(self):return self.scale.double().square()*self.mixture.variance.double()

    def log_prob(self,action):
        raise NotImplementedError('Moving atomic support requires the declared pathwise/moment objective; native PPO score is not legal')


class TransportPacked:
    def __init__(self,reference,current,mode,log_scale):
        self.reference=reference;self.current=current;self.mode=mode;self.log_scale=log_scale

    def forecast_joint(self,batch):
        reference,target=JointForecast.from_packed(self.reference,batch)
        mean=None
        if self.current is not None:
            current,truth=JointForecast.from_packed(self.current,batch)
            assert torch.equal(target,truth);mean=current.mean
        return TransportJoint(reference,self.mode,mean=mean,log_scale=self.log_scale),target

    def log_prob(self,value):
        raise NotImplementedError('Inherited packed native NLL is not a common-measure score for this moving-atom transport')


class TransportPolicy(nn.Module):
    def __init__(self,config):
        super().__init__();self.route='A0';self.k=1;self.routing='single';self.fixed_kappa=True
        self.transport_mode=config['transport_mode']
        if self.transport_mode not in ('P04_LOCATION','P05_SCALE'):raise ValueError('Unqualified transport')
        self.reference=Policy(config['weights'],'A0',revision=config.get('revision'),local_files_only=config.get('local_files_only',True))
        self.current=Policy(config['weights'],'A0',revision=config.get('revision'),local_files_only=config.get('local_files_only',True)) if self.transport_mode=='P04_LOCATION' else None
        if self.transport_mode=='P05_SCALE':self.log_scale=nn.Parameter(torch.zeros(()))
        state=load_checkpoint(config['checkpoint'],map_location='cpu',weights_only=False)
        if state.get('transport_mode'):
            if state['transport_mode']!=self.transport_mode:raise ValueError('Transport checkpoint type differs')
            self.load_state_dict(state['policy_state'],strict=True)
        else:
            if state.get('route')!='A0':raise ValueError('Transport controls require the qualified A0 reference')
            self.reference.load_state_dict(state['policy_state'],strict=True)
            if self.current is not None:self.current.load_state_dict(state['policy_state'],strict=True)
        self.reference.requires_grad_(False)

    @property
    def module(self):return (self.current if self.current is not None else self.reference).module

    def forward(self,batch):
        with torch.no_grad():ref,w,_,_=self.reference(batch)
        current=self.current(batch)[0] if self.current is not None else [None]
        if len(ref)!=1 or len(current)!=1:raise ValueError('Shared-mode transport is not qualified')
        return [TransportPacked(ref[0],current[0],self.transport_mode,getattr(self,'log_scale',None))],w,None,None
