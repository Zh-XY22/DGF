# Configuration and stages

Two configurations are provided: `etth1_h336_mse.json` for point forecasting and `etth1_h192_crps.json` for probabilistic forecasting. Both use the same complete DGF method, fixed training budgets, seed 1117 and validation MAP MSE selection. No saved validation step is prescribed.

Each JSON declares the model and data versions, split/normalization/mask contract, every phase seed, optimizer groups, precision, budgets, validation, hierarchical objectives and evaluation randomness. Backend-fixed numerical conventions are checked; unsupported changes raise an error instead of silently being ignored. Computed scaler and diversity values are fitted from the user-provided training data and included in the effective stage configurations.

The phase order is:

1. A0 SFT: 10 full epochs, official packed NLL, LR1e-7, logical batch64, microbatch8. Its validation MAP MSE-selected weights initialize the modes.
2. Mode initialization: eight copied heads with seeded perturbations and an input router; initialization is weights-only, not optimizer continuation.
3. Ordinary preparation: one full epoch of legal trajectory-mixture NLL, LR1e-7, batch512, microbatch1. Total concentration is fixed. Its final state initializes RL.
4. Complete DGF RL: four full epochs, LR1.5e-6, batch512, microbatch1, B=8 groups and G=100 trajectories per group, one update per fresh rollout. Total concentration is trainable; all core objectives remain active.

AdamW resets for each stage: betas(0.9,0.98), epsilon1e-6, weight decay0.1 for declared linear/projection weights and0 elsewhere, global norm clipping1. There is no PPO clipping, reference KL or real-label NLL anchor in the complete RL objective. A final partial logical batch stays partial. Microbatch losses are weighted by their actual input count; a logical batch creates one optimizer update.

Validation includes step0, first-epoch sixteenths and subsequent-epoch quarters. Multiple thresholds can refer to the same optimizer state and reuse its evaluation. Training has a fixed budget without diagnostic-score early stopping. Users' validation values choose their checkpoints; the test set does not select anything.

With the environment variables from the README, phases can run independently:

```bash
bash scripts/time_series_forecasting/ETTh1_MSE.sh prepare
bash scripts/time_series_forecasting/ETTh1_MSE.sh sft
bash scripts/time_series_forecasting/ETTh1_MSE.sh initialize
bash scripts/time_series_forecasting/ETTh1_MSE.sh prepare-modes
bash scripts/time_series_forecasting/ETTh1_MSE.sh rl
bash scripts/time_series_forecasting/ETTh1_MSE.sh test
```

Replace MSE with CRPS for the second configuration. The one-command train executes these training phases in order. `dgf --help` lists commands. Export declared settings without importing model code:

```bash
python scripts/run.py export-config --config config/etth1_h336_mse.json --output-dir ../declared-settings
```

A real run also saves resolved_config.json, prepared protocol, phase JSONs, checkpoint selection records, training/validation curves and training_receipt.json. All numerical defaults used by these examples are explicit or are checked implementation constants.

## Recovery

At a safe update boundary, SIGTERM/SIGINT requests save a complete resume.pt with optimizer, constant-scheduler state, RNG, data cursor and selection history. Resume an identical run:

```bash
bash scripts/time_series_forecasting/ETTh1_MSE.sh train --resume
```

For a phase command, append --resume to that phase. Completed phases are checked and reused. Recovery requires the same full working directory, selected checkpoints, source, inputs and settings; changed bindings are rejected. Cross-machine path remapping is not supported: restore the original absolute working path. A weights-only copy is initialization, not complete-state recovery. Hardware failures before a safe save lose only progress since the last complete state.

Keep outputs outside the repository. For evaluation, leave CHECKPOINT unset to use your run's validation-selected state. An explicit CHECKPOINT is accepted only when it is byte-identical to that selected local state; it is not a way to choose a checkpoint using test labels.
