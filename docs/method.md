# Method

For each history X, the shared Moirai backbone produces K=8 mode-conditioned legal forecast distributions. The router uses a128-wide Tanh hidden layer. The implementation parameterizes positive mean weights w by a floored softmax and a positive total concentration kappa by softplus plus a floor, then alpha=kappa*w. It does not use independent floor+softplus logits for each alpha component. Preparation fixes kappa at its initialized value; RL releases its input-dependent concentration head.

For each history and mini-group b, pi_b~Dirichlet(alpha(X)) is sampled once. Each of G trajectories chooses a single outer mode z_bg~Categorical(pi_b) for its complete future. Within each mode, the native distribution family labels are augmented actions. Future labels are used in rewards, not router inputs.

Accuracy reward is negative forecast MSE. Four dynamic penalties compare first differences, second differences, normalized rFFT amplitudes, and the absolute first-forecast/last-history jump. The boundary convention penalizes that jump itself rather than matching the true boundary increment. Temporal differences respect valid adjacent positions; masked padding does not create a new reward boundary. FFT includes DC and normalizes by total amplitude plus1e-6.

Combined reward is accuracy minus0.01 times the sum of the four penalties. Trajectory advantages standardize along G within each group; group advantages standardize group-average reward along B for the same history. Population standard deviation and epsilon1e-6 are used; constant rewards give legal zero advantage. Actions, rewards and advantages are detached for score-function updates.

The sample score uses the sum of valid joint augmented future log probabilities plus log Cat(z|pi). The group score is log Dir(pi|alpha), with its own group credit. Native family labels are distinguished from outer modes. No clipped importance-ratio surrogate is added by these examples.

Mode diversity uses squared distances between legal mode-mean level/difference/normalized-spectrum features, normalized by training-only feature scales, capped at1 and averaged across distinct mode pairs. It is an independent differentiable term, not a shared reward offset that cancels during centering.

The minimized objective is -J_sample -1*J_dir -0.001*R_div. Sample and group objectives use mean_NBG and mean_NB reductions; diversity uses mean_N. These balanced reductions and the weight-times-total concentration parameterization are explicit implementation conventions. Concentration and diversity participate in RL, while a legal zero gradient at symmetry or a cap is not automatically an error.

The complete pipeline starts from public pretrained weights, performs supervised training, initializes modes, performs ordinary NLL preparation and then optimizes this full objective. Prediction and fair-CRPS interfaces are described in evaluation.md. This package makes no claim that all paper implementation details or evaluation protocols are identical.
