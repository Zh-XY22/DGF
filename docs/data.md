# Data and pretrained model

Each configuration pins the ETDataset revision and Moirai pretrained revision. `scripts/download_assets.py` downloads their public files to caller-selected directories. The model directory must contain config.json and the complete pretrained safetensors files. Model weights, datasets and credentials are not distributed with this source package.

ETTh1 is hourly multivariate forecasting with channels, in order: HUFL, HULL, MUFL, MULL, LUFL, LULL, OT. Training rows are [0,8640), validation [8640,11520), test [11520,14400). The remaining source rows are not added to these splits. Context is 5000, patch size 64, and the H192/H336 horizons are 192 and 336.

Every scored forecast label lies entirely inside its split. Historical context can precede the split boundary. Training starts must have a complete context and complete future; stride is one for each split. This gives 3305/2545/2545 train/validation/test windows at horizon336 and 3449/2689/2689 at horizon192.

An external scaler is fitted only on the training rows using sample standard deviation (ddof=1). Official Arrow builders use std+1e-10; explicit forecast arrays use std without that epsilon. The pretrained model also uses its native history-dependent instance scaling. These operations and their inverse transforms are distinct. Missing values are rejected by this complete-window protocol, observed zeros are retained, patch padding is masked and cropped to the requested horizon.

Diversity feature statistics for levels, first/second differences and normalized FFT amplitudes are fitted only on training data. No future test labels fit scalers, prototypes or thresholds. Prepared arrays, split indices, scaler values and the effective protocol are saved under OUTPUT_DIR/data.

Separate stages:

```bash
bash scripts/time_series_forecasting/ETTh1_MSE.sh prepare
```

Use the CRPS script for the other horizon. Do not mix prepared data or checkpoints between horizons.
