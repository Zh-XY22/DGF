# Environment

The numerical implementation has been run on Linux with NVIDIA H100 80 GB HBM3, PyTorch 2.9.0 and CUDA runtime 12.4. The independent dependency installation environment used Python 3.12.3, PyTorch 2.9.0+cu128, CUDA 12.8, NumPy 1.26.4 and Lightning 2.4.0 without system-site packages. These are different tested environments, not a claim that every version combination has identical floating-point behavior.

The dependency pins are in requirements.txt. The CUDA-enabled Torch wheel and the system NVIDIA driver must be compatible. One process uses one visible GPU; distributed training is not required. Models and gradients use float32, CUDA matrix multiplication and cuDNN permit TF32, rewards and metric reductions use float64, and AMP is disabled.

For H192/H336 with the declared scientific microbatch of one, measured allocated memory was approximately 5.0–5.3 GiB, and allocator-reserved memory approximately 7.2–7.5 GiB on the tested GPU. These measurements do not establish the minimum device capacity: framework overhead, preparation, model copies and other software can change requirements. The environment actually used had 80 GB VRAM.

Data, pretrained snapshots, caches and outputs must be outside the source directory. The public algorithm and runner require no cluster scheduler, distributed filesystem, private credentials or server-specific mount.

Run the package checks on a compute node:

```bash
python -m unittest discover -s tests -v
```

A short three-phase check is explicitly marked qualification-only:

```bash
bash scripts/time_series_forecasting/ETTh1_MSE.sh train --smoke
bash scripts/time_series_forecasting/ETTh1_MSE.sh test --smoke
```

Use a separate output directory for qualification; it cannot be resumed as a full-budget run. This checks interfaces and updates, not full-budget training reproducibility.
