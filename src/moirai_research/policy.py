"""Original Moirai families, eight trajectory modes, explicit router and freezes."""
from .io_utils import load_checkpoint
import copy
import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils._pytree import tree_map
from einops import rearrange
from uni2ts.common.torch_util import mask_fill,packed_attention_mask
from uni2ts.distribution._base import convert_to_container,tree_map_multi
from uni2ts.model.moirai import MoiraiModule
from .actions import JointForecast, NonfiniteForecastSample

FIELDS=['target','observed_mask','sample_id','time_id','variate_id','prediction_mask','patch_size']
WEIGHT_FLOOR=1e-4
KAPPA_FLOOR=1e-3
POLICY_VERSION='routes-v1.2-independent-w-kappa'
HEAD_PERTURBATION=1e-4

def raw_parameters(head,reprs,patch):
    return tree_map(lambda layer:rearrange(layer(reprs,patch),'... (dim out) -> ... out dim',out=head.out_size),convert_to_container(head.proj))

class LoRALinear(nn.Module):
    def __init__(self,base):
        super().__init__();self.base=base.requires_grad_(False)
        self.down=nn.Linear(base.in_features,16,bias=False)
        self.up=nn.Linear(16,base.out_features,bias=False)
        nn.init.zeros_(self.up.weight)
    def forward(self,x):return self.base(x)+2.*self.up(self.down(x))

class RawResidual(nn.Module):
    def __init__(self,module):
        super().__init__()
        self.down=nn.Linear(module.d_model,32,bias=False)
        self.up=module.distr_output.get_param_proj(32,module.patch_sizes)
        for p in self.up.parameters(): nn.init.zeros_(p)
    def forward(self,reprs,patch):return raw_parameters(self.up,self.down(reprs),patch)

class Policy(nn.Module):
    def __init__(self,weights,route,initial_checkpoint=None,revision=None,local_files_only=True):
        super().__init__();self.route=route;self.routing='single' if route=='A0' else 'softmax' if route.startswith('C-') else 'dirichlet'
        self.module=MoiraiModule.from_pretrained(weights,revision=revision,local_files_only=local_files_only)
        if initial_checkpoint:
            state=load_checkpoint(initial_checkpoint, map_location='cpu', weights_only=False)['state_dict']
            self.module.load_state_dict({k.removeprefix('module.'):v for k,v in state.items() if k.startswith('module.')},strict=True)
        self.fixed_kappa=True;self.k=1 if route=='A0' else 8
        if self.k==1:return
        frozen=route in ['D-LoRA','D-residual']
        if frozen:self.module.requires_grad_(False)
        if route=='D-LoRA':
            for name,child in list(self.module.named_modules()):
                if name.endswith(('q_proj','k_proj','v_proj')) and isinstance(child,nn.Linear):
                    parent,key=name.rsplit('.',1);self.module.get_submodule(parent)._modules[key]=LoRALinear(child)
        if route=='D-residual':
            self.residuals=nn.ModuleList([RawResidual(self.module) for _ in range(8)])
        else:
            # Full routes reuse the existing head as mode zero; LoRA uses eight trainable copies.
            heads=[copy.deepcopy(self.module.param_proj) for _ in range(8)]
            if not frozen:heads[0]=self.module.param_proj
            self.heads=nn.ModuleList(heads).requires_grad_(True)
            with torch.no_grad():
                for head in self.heads:
                    for parameter in head.parameters():
                        rms=parameter.float().square().mean().sqrt().clamp_min(1e-3)
                        parameter.add_(torch.randn_like(parameter)*rms*HEAD_PERTURBATION)
        self.router_hidden=nn.Sequential(nn.Linear(self.module.d_model,128),nn.Tanh())
        self.router_mean=nn.Linear(128,8)
        nn.init.normal_(self.router_mean.weight,std=1e-3);nn.init.zeros_(self.router_mean.bias)
        self.router_concentration=nn.Linear(128,1)
        nn.init.zeros_(self.router_concentration.weight)
        nn.init.constant_(self.router_concentration.bias,math.log(math.expm1(8-KAPPA_FLOOR)))
        self.router_concentration.requires_grad_(False)

    def configure_update(self,fixed_kappa=True,external=None):
        self.fixed_kappa=fixed_kappa
        if self.k==1:return
        if external:
            self.requires_grad_(False)
            if external!='uniform':
                self.router_hidden.requires_grad_(True);self.router_mean.requires_grad_(True)
            self.routing='dirichlet' if external=='dir' else 'softmax' if external=='softmax' else 'uniform'
        self.router_concentration.requires_grad_(self.routing=='dirichlet' and not fixed_kappa)

    def forward(self,batch):
        m=self.module
        if self.k==1:
            distr=m(**{key:batch[key] for key in FIELDS})
            return [distr],torch.ones((len(batch['target']),1),device=batch['target'].device),None,None
        target=batch['target'];observed=batch['observed_mask'];predmask=batch['prediction_mask']
        loc,scale=m.scaler(target,observed*~predmask.unsqueeze(-1),batch['sample_id'],batch['variate_id'])
        reprs=m.in_proj((target-loc)/scale,batch['patch_size'])
        reprs=mask_fill(reprs,predmask,m.mask_encoding.weight)
        reprs=m.encoder(reprs,packed_attention_mask(batch['sample_id']),time_id=batch['time_id'],var_id=batch['variate_id'])
        history_mask=(~predmask)&observed.any(-1)&(batch['sample_id']>0)
        pooled=(reprs*history_mask[...,None]).sum(-2)/history_mask.sum(-1,keepdim=True).clamp_min(1)
        hidden=self.router_hidden(pooled)
        soft=F.softmax(self.router_mean(hidden),dim=-1)
        kappa=torch.full_like(soft[...,:1],8.) if self.fixed_kappa else KAPPA_FLOOR+F.softplus(self.router_concentration(hidden))
        mean_route=WEIGHT_FLOOR+(1-8*WEIGHT_FLOOR)*soft
        alpha=kappa*mean_route
        weights=mean_route if self.routing=='dirichlet' else soft
        if self.routing=='uniform':weights=torch.ones_like(soft)/8
        distributions=[]
        if self.route=='D-residual':
            original=raw_parameters(m.param_proj,reprs,batch['patch_size'])
            for residual in self.residuals:
                addition=residual(reprs,batch['patch_size'])
                raw=tree_map_multi(lambda base,delta:base+delta,original,addition)
                bounded=tree_map_multi(lambda constraint,param:constraint(param),m.param_proj.domain_map,raw)
                distributions.append(m.distr_output.distribution(bounded,loc=loc,scale=scale))
        else:
            distributions=[m.distr_output.distribution(head(reprs,batch['patch_size']),loc=loc,scale=scale) for head in self.heads]
        return distributions,weights,alpha if self.routing=='dirichlet' else None,kappa if self.routing=='dirichlet' else None

def joints(output,batch):
    distributions,w,alpha,kappa=output
    experts=[];target=None
    for distribution in distributions:
        joint,truth=distribution.forecast_joint(batch) if hasattr(distribution,'forecast_joint') else JointForecast.from_packed(distribution,batch)
        if target is not None: assert torch.equal(truth,target)
        target=truth;experts.append(joint)
    return experts,target,w,alpha,kappa

def rollout(experts,w,alpha,j,g):
    """One pi per subgroup and one mode per whole candidate trajectory."""
    n,p=experts[0].mean.shape;k=len(experts)
    with torch.no_grad():
        pi=torch.distributions.Dirichlet(alpha).sample((j,)).transpose(0,1) if alpha is not None else w[:,None].expand(n,j,k)
        mode=torch.distributions.Categorical(probs=pi).sample((g,)).permute(1,2,0)
        forecast=experts[0].mean.new_empty((n,j,g,p));raw=torch.empty_like(forecast)
        component=torch.empty((n,j,g,p),device=w.device,dtype=torch.long)
        # Draw only as many samples per expert as the most occupied input needs.
        for index,expert in enumerate(experts):
            chosen=(mode==index).reshape(n,j*g)
            count=int(chosen.sum(-1).max())
            if not count:continue
            try:sampled=expert.sample(count)
            except NonfiniteForecastSample as error:
                error.expert_index=index
                raise
            for history in range(n):
                locations=chosen[history].nonzero().flatten();size=len(locations)
                for dest,key in [(forecast,'forecast'),(raw,'raw'),(component,'component')]:
                    dest.reshape(n,j*g,p)[history,locations]=sampled[key][:size,history]
    return dict(forecast=forecast.detach(),raw=raw.detach(),component=component.detach(),mode=mode.detach(),pi=pi.detach())

def expert_log_prob(experts,actions):
    n,j,g,p=actions['forecast'].shape
    # safe selected family support is enforced by JointForecast.log_prob.
    output=actions['forecast'].new_zeros((n,j,g,p))
    for index,expert in enumerate(experts):
        selected=actions['mode']==index
        if not selected.any():continue
        # Actions from another mode can violate this mode's discrete component support.
        own=dict(component=actions['component'].permute(1,2,0,3).reshape(j*g,n,p),
                 raw=actions['raw'].permute(1,2,0,3).reshape(j*g,n,p),
                 forecast=actions['forecast'].permute(1,2,0,3).reshape(j*g,n,p))
        lp=expert.log_prob(own).reshape(j,g,n,p).permute(2,0,1,3)
        output=output+torch.where(selected[...,None],lp,torch.zeros_like(lp))
    return output

def mixture_nll(output,batch):
    distributions,w,_,_=output
    valid=batch['prediction_mask'][...,None]&batch['observed_mask']
    counts=valid.flatten(1).sum(-1).clamp_min(1)
    scores=torch.stack([(d.log_prob(batch['target'])*valid).flatten(1).sum(-1) for d in distributions],-1)
    # A mode is fixed for the entire trajectory; average after joint logsumexp.
    return -(torch.logsumexp(w.log()+scores,dim=-1)/counts).mean()
