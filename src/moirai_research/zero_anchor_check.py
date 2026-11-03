"""Worker-only regression: a disabled NLL never enters the real PPO objective."""
import json
from unittest.mock import patch

import torch

from . import rl


def run_checks(device='cuda'):
    """Check loss/gradients without an optimizer or model/data execution."""
    parameter = torch.tensor(.02, dtype=torch.float64, device=device, requires_grad=True)
    sign = torch.tensor([1., -1.], dtype=torch.float64, device=device).reshape(1, 1, 2, 1)
    lp = parameter * sign.expand(1, 1, 2, 2)
    record = dict(actions={}, old=torch.zeros_like(lp), reference_lp=torch.zeros_like(lp),
        advantage=sign[..., None].expand(1, 1, 2, 2, 1), observed=torch.ones((1, 2, 1), dtype=torch.bool, device=device),
        weight=torch.ones((1, 2, 1), dtype=torch.float64, device=device), reward_mean=-1., reward_zero_std_fraction=0.)
    config = dict(protocol=dict(horizon=2, mode='S', variables=['OT']), arm='E03', nll_anchor_coefficient=0.)
    output = ([object()], torch.zeros((1, 2), dtype=torch.float64, device=device),
        torch.ones((1, 1), dtype=torch.float64, device=device), None, None)
    with patch.object(rl, 'joints', return_value=output), patch.object(rl, 'expert_log_prob', return_value=lp):
        for arm in ('E16', 'E17', 'E03'):
            local = dict(config, arm=arm, nll_anchor_coefficient=1. if arm != 'E03' else 0.)
            with patch.object(rl, 'nll', side_effect=RuntimeError('disabled NLL unexpectedly executed')) as unused:
                loss, diagnostic = rl.objective(lambda batch: None, {}, record, local)
                gradient = torch.autograd.grad(loss, parameter, retain_graph=True)[0]
                assert unused.call_count == 0
                assert torch.isfinite(loss) and torch.isfinite(gradient) and gradient != 0
                assert diagnostic['nll'] is None
                assert diagnostic['nll_status'] == 'NOT_COMPUTED_ZERO_COEFFICIENT'
                json.dumps(diagnostic, allow_nan=False)
        with patch.object(rl, 'nll', return_value=parameter.square()) as used:
            anchored, diagnostic = rl.objective(lambda batch: None, {}, record, dict(config, nll_anchor_coefficient=.7))
            anchored_gradient = torch.autograd.grad(anchored, parameter, retain_graph=True)[0]
            assert used.call_count == 1 and diagnostic['nll_status'] == 'COMPUTED_TRAINING_ANCHOR'
        with patch.object(rl, 'nll', side_effect=RuntimeError('disabled NLL unexpectedly executed')):
            plain, _ = rl.objective(lambda batch: None, {}, record, config)
            plain_gradient = torch.autograd.grad(plain, parameter)[0]
        torch.testing.assert_close(anchored - plain, .7 * parameter.square(), atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(anchored_gradient - plain_gradient, 1.4 * parameter, atol=1e-12, rtol=1e-12)
    return dict(status='PASS', zero_coefficient_NLL_calls=0, anchored_NLL_calls=1,
        zero_anchor_gradient_finite_nonzero=True, unchanged_active_anchor_loss_and_gradient=True,
        null_diagnostic_serializable=True, optimizer_steps=0, optimizer_trajectories=0,
        device=str(device))
