# DGF

Code for [Dirichlet-Guided Group Forecasting for Alleviating Over-smoothing in Time Series Forecasting](https://arxiv.org/abs/2606.10592).

DGF learns mode-conditioned forecast distributions with input-dependent Dirichlet concentrations. Training shares a sampled probability vector within each mini-group, selects one mode per future trajectory, and combines trajectory and group credit with dynamic consistency rewards and bounded mode diversity. See [docs/method.md](docs/method.md) for the implementation and objective conventions.

## Installation

Use Linux, Python 3.12 and an NVIDIA CUDA GPU:

```bash
git clone https://github.com/Zh-XY22/DGF.git
cd DGF
python -m venv ../dgf-venv
source ../dgf-venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation .
python -m pip check
dgf --help
```

See [ENVIRONMENT.md](ENVIRONMENT.md) for tested dependencies, hardware and precision. Installing Python packages does not install an NVIDIA driver.

## Data and Pretrained Model

Download the public ETTh1 CSV and pinned [Moirai-small](https://huggingface.co/Salesforce/moirai-1.0-R-small) snapshot outside this repository:

```bash
python scripts/download_assets.py --config config/etth1_h336_mse.json --data-dir ../dgf-data --model-dir ../dgf-model
export ETTH1_CSV="$(cd ../dgf-data && pwd)/ETTh1.csv"
export MOIRAI_MODEL_DIR="$(cd ../dgf-model && pwd)"
export CUDA_VISIBLE_DEVICES=0
```

The two configurations use the same pinned public data/model versions. If they are already downloaded, set those variables to the corresponding local paths. Details: [docs/data.md](docs/data.md).

## Training and Evaluation

The complete pipeline prepares data, trains single-mode supervised weights, initializes modes, runs ordinary preparation, and trains the complete DGF objective. Each configuration declares its stage budgets, validation selectors and parent transfers. Both configurations use the same method and select checkpoints by validation MAP MSE. Test evaluation remains a separate command.

### Point forecasting: ETTh1 H336

```bash
export OUTPUT_DIR="$(pwd)/../dgf-runs/etth1-h336-mse"
bash scripts/time_series_forecasting/ETTh1_MSE.sh train
bash scripts/time_series_forecasting/ETTh1_MSE.sh test
```

### Probabilistic forecasting: ETTh1 H192

```bash
export OUTPUT_DIR="$(pwd)/../dgf-runs/etth1-h192-crps"
bash scripts/time_series_forecasting/ETTh1_CRPS.sh train
bash scripts/time_series_forecasting/ETTh1_CRPS.sh test
```

The CRPS configuration uses the same complete DGF training objective and validation MAP MSE checkpoint selection. Its name identifies the evaluation use case; it does not replace the reward or select by validation CRPS. Both evaluation commands save point metrics, predictions and independent-marginal fair CRPS using 1000 samples. Users obtain metrics from their own runs.

Configurations: [H336 MSE](config/etth1_h336_mse.json) and [H192 CRPS](config/etth1_h192_crps.json). See [docs/configuration.md](docs/configuration.md) for explicit settings, phase commands and recovery, and [docs/evaluation.md](docs/evaluation.md) for metric definitions and output files.

## Citation

```bibtex
@misc{dgf2026,
  title={Dirichlet-Guided Group Forecasting for Alleviating Over-smoothing in Time Series Forecasting},
  year={2026},
  eprint={2606.10592},
  archivePrefix={arXiv},
  url={https://arxiv.org/abs/2606.10592}
}
```

## Acknowledgments

This implementation builds on [Moirai / uni2ts](https://github.com/SalesforceAIResearch/uni2ts). Upstream source and copyright notices are retained in [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Model weights and datasets retain their own licenses. Repository organization follows the staged-command style of [F-LLM](https://github.com/Zh-XY22/F-LLM); no F-LLM forecasting algorithm is used.
