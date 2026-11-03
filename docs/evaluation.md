# Evaluation

Both configurations use the checkpoint selected by validation MAP MSE. The CRPS configuration uses the complete DGF training objective; CRPS is an evaluation metric, not a training reward or checkpoint selector. Training and testing are separate commands; the runner freezes the actual validation choice before opening tests.

MAP uses expected Dirichlet weights alpha/sum(alpha), takes their maximum with first-index tie breaking, and returns that outer mode's legal distribution mean. One trajectory is used for both MSE and MAE. It is not the Dirichlet density mode, mixture mean, sample mean/median or a truth-selected mode. Native Student-t/Normal/NegativeBinomial/LogNormal families are distinct from the outer modes.

MSE and MAE average over effective forecast times, seven variables and all valid split windows. Standardized and inverse-transformed original-unit metrics are separate outputs; the external train-only scaler is distinct from model-internal normalization. Secondary 100-sample engineering and analytic mixture-mean readouts may be recorded, with their own names.

The formal fair empirical CRPS uses 1000 iid marginal samples:

L=(1/G) sum_i |x_i-y| - (1/[2G(G-1)]) sum_(i!=j) |x_i-x_j|.

The marginal evaluator independently samples an outer mode per candidate using expected weights. Training instead shares a Dirichlet draw within each mini-group; those correlated training samples are not passed to the iid fair correction. Seeds 1107001 and 1107003 are both evaluated, with the fixed test offset 104729 and global four-input block index. These are evaluation random seeds, not additional training repetitions.

The evaluation command records point predictions, effective configurations, MSE/MAE, fair and ordinary ensemble CRPS, NLL diagnostics, quantile scores and intervals under OUTPUT_DIR/evaluation. Probability samples are streamed, not saved as a huge full ensemble. Per-window probability statistics and point predictions retain enough indexing for metric inspection. Initial, selected and final checkpoints keep their identities.

A separate process reloads the selected checkpoint for evaluation. For an independent point-metric reduction, use the saved point predictions through the provided recomputation script. Scores are computed from the current user's run; no historical score file is bundled or read to manufacture output.

```bash
bash scripts/time_series_forecasting/ETTh1_CRPS.sh test
```

For H192/H336, a --smoke run uses two windows and one update per phase and is explicitly qualification-only; it does not produce a complete-window performance claim. Missing/invalid diagnostics are reported and do not silently switch the point readout.
