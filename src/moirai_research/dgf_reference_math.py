"""DGF tensor objectives; N histories, B groups, G trajectories, H times, D variables."""
import torch

EPS = 1e-6

def masked_mean(value, mask, dims):
    mask = mask.expand_as(value)
    return torch.where(mask, value, 0).sum(dims) / mask.sum(dims).clamp_min(1)

def relative(reward, axis):
    mean = reward.mean(axis, keepdim=True)
    std = reward.std(axis, correction=0, keepdim=True)
    return torch.where(std > 0, (reward - mean) / (std + EPS), torch.zeros_like(reward)).detach()

def hierarchical_advantages(reward):
    assert reward.ndim == 3, 'Reward must have N,B,G axes'
    return relative(reward, 2), relative(reward.mean(2), 1)

def spectrum(value, mask):
    amplitude = torch.fft.rfft(torch.where(mask, value, 0), dim=-2).abs()
    return amplitude / (amplitude.sum(-2, keepdim=True) + EPS), mask.all(-2)

def dynamic_terms(prediction, target, mask, history_last, history_valid):
    truth = target[:, None, None]
    valid = mask[:, None, None].expand_as(prediction)
    terms = []
    for order in (1, 2):
        window = valid[..., :prediction.shape[-2]-order, :].clone()
        for shift in range(1, order+1):
            window &= valid[..., shift:prediction.shape[-2]-order+shift, :]
        error = (prediction.diff(n=order, dim=-2) - truth.diff(n=order, dim=-2)).abs()
        terms.append(masked_mean(error, window, (-2, -1)))
    pred_fft, complete = spectrum(prediction, valid)
    true_fft, _ = spectrum(truth, mask[:, None, None])
    terms.append(masked_mean((pred_fft-true_fft).abs(), complete[..., None, :], (-2, -1)))
    boundary_valid = valid[..., 0, :] & history_valid[:, None, None]
    # Historical paper-anchor boundary convention; target-jump matching is a different recipe.
    terms.append(masked_mean((prediction[..., 0, :]-history_last[:, None, None]).abs(), boundary_valid, -1))
    return torch.stack(terms, -1)

def mode_diversity(means, mask, feature_scales, cap=1.0, rho=1.0):
    n, k, h, d = means.shape
    assert k > 1 and cap > 0 and feature_scales.shape == (n, 4, d)
    fft, complete = spectrum(means, mask[:, None].expand_as(means))
    features = (means, means.diff(dim=-2), means.diff(n=2, dim=-2), fft)
    masks = (mask, mask[:, :-1] & mask[:, 1:], mask[:, :-2] & mask[:, 1:-1] & mask[:, 2:], complete[:, 0, None, :])
    distance = means.new_zeros((n, k, k))
    for index, (feature, valid) in enumerate(zip(features, masks)):
        scale = feature_scales[:, index].clamp_min(EPS)[:, None, None, None, :]
        term = masked_mean((feature[:, :, None]-feature[:, None, :]).square()/scale, valid[:, None, None], (-2, -1))
        distance = distance + (1.0 if index == 0 else rho) * term
    pairs = ~torch.eye(k, device=means.device, dtype=torch.bool)
    pairwise = distance[:, pairs]
    return pairwise.clamp_max(cap).mean(), pairwise

def complete_loss(trajectory_log_prob, dirichlet_log_prob, reward, diversity, gamma=1.0, lambda_div=0.001):
    """Legacy balanced reductions: mean_NBG sample, mean_NB Dir, mean_N diversity.

    Relative to paper sums per history: gamma_paper=G*gamma and
    lambda_div_paper=B*G*lambda_div after dividing the full loss by B*G.
    """
    sample_adv, dir_adv = hierarchical_advantages(reward.detach())
    sample = (trajectory_log_prob * sample_adv).mean()
    direction = (dirichlet_log_prob * dir_adv).mean()
    loss = -sample - gamma * direction - lambda_div * diversity
    return loss, {'J_sample': sample, 'J_dir': direction, 'R_div': diversity}
