"""Predeclared descriptive contrasts and metric labels; no result selection."""

PAPER_TARGETS = {
    'ETTh1': dict(MSE=.364, MAE=.399),
    'ETTm1': dict(MSE=.280, MAE=.330),
}
PAPER_REFERENCE = dict(version='2606.10592v1', horizon=96, size='small',
    inference='MAP-mode analytic selected-mode mean', mse_table=9, mae_table=10,
    uri='https://arxiv.org/abs/2606.10592v1',
    limitation='Rounded paper means; exact checkpoint revision and all preprocessing/training details are not fully matched')

EXPLORATION_CONTRASTS = (
    ('E01', 'E00', 'A0_RL_vs_continue_NLL', 'Same preparation and additional input/update budget'),
    ('E03', 'E02', 'D_pre_RL_vs_continue_NLL', 'Same preparation and additional input/update budget'),
    ('E04', 'E03', 'weak_labels_vs_ordinary', 'Zero-confidence fallback can make this contrast unidentifiable'),
    ('E04', 'E05', 'weak_labels_vs_shuffled_labels', 'Same neighbors, confidence, distance and time strata'),
    ('E06', 'E03', 'point_distillation_vs_ordinary', 'Teacher preparation cost is additional'),
    ('E07', 'E03', 'FKL_vs_ordinary', 'Prediction-distribution distillation; not autoregressive OPD'),
    ('E08', 'E03', 'RKL_vs_ordinary', 'Prediction-distribution distillation with score-function gradient'),
    ('E07', 'E06', 'FKL_vs_point_distillation', 'Same teacher and preparation exposure, different objectives'),
    ('E08', 'E06', 'RKL_vs_point_distillation', 'Same teacher and preparation exposure, different objectives'),
    ('E07', 'E08', 'FKL_vs_RKL', 'KL direction changes both objective and Monte Carlo estimator'),
    ('E09', 'E03', 'hard_selection_vs_ordinary', 'Code-consistent AND plus declared coverage fallback'),
    ('E10', 'E03', 'soft_selection_vs_ordinary', 'Full joint true-label NLL remains present'),
    ('E09', 'E11', 'hard_selection_vs_matched_random_mask', 'Retention counts are matched by variable and horizon'),
    ('E10', 'E09', 'soft_vs_hard_selection', 'Nonzero floor changes the pointwise auxiliary weights'),
    ('E15', 'E03', 'RL_selection_vs_ordinary', 'Frozen input weights shared by all candidates'),
    ('E12', 'E03', 'block_global_credit_vs_position_R', 'Changes credit/normalization; not a pure reward-information effect'),
    ('E13', 'E12', 'matched_block_local_vs_global_credit', 'Identical block reward, baseline and common normalizer'),
    ('E14', 'E13', 'local_differences_added', 'Adds first/second difference information and dependency scores'),
    ('E01', 'E16', 'A0_NLL_anchor_added', 'Same preparation and RL except the true-label NLL anchor'),
    ('E03', 'E17', 'D_pre_NLL_anchor_added', 'Same preparation and RL except the true-label NLL anchor'),
)


def decision_metrics(protocol):
    """(reported name, actual stored field, interpretation of direction)."""
    keys = {
        'engineering': ('MSE', 'MAE', 'mean'),
        'MAP': ('MAP_MSE', 'MAP_MAE', 'MAP'),
        'mixture_analytic': ('mixture_analytic_MSE', 'mixture_analytic_MAE', 'mixture_analytic'),
    }
    if protocol not in keys:
        raise ValueError('Unregistered inference protocol')
    mse, mae, prefix = keys[protocol]
    result = [(name, source, 'lower') for name, source in [
        ('MSE', mse), ('MAE', mae), ('CRPS', 'CRPS'),
        ('marginal_NLL', 'marginal_NLL'), ('joint_NLL_per_position', 'joint_NLL_per_position'),
        ('raw_CRPS', 'raw_CRPS')]]
    result += [('coverage_80', 'coverage_80', 'target_0.8'),
               ('width_80', 'width_80', 'interpret_with_coverage')]
    result += [(name, prefix + '_' + name, 'lower') for name in [
        'diff1_L1', 'diff2_L1', 'fft_L1', 'boundary_change_L1', 'peak_MSE', 'turn_MSE']]
    if protocol == 'engineering':
        result += [('raw_MSE', 'raw_MSE', 'lower'), ('raw_MAE', 'raw_MAE', 'lower')]
    return tuple(result)
def generation_evidence(detail):
    """Receipt hardware cannot fill missing prediction-generation ownership."""
    generation=detail.get('readout_provenance',{}).get('test',{}).get('generation',{})
    return dict(prediction_gpu=generation.get('actual_gpu','UNKNOWN'),receipt_gpu=detail.get('actual_gpu','UNKNOWN'),
        model_precision=generation.get('model_precision','UNKNOWN'),metric_precision=generation.get('metric_precision','UNKNOWN'),
        tf32_matmul=generation.get('tf32_matmul','UNKNOWN'),tf32_cudnn=generation.get('tf32_cudnn','UNKNOWN'))

