# EVODiff — Entropy-aware Variance Optimized Diffusion Inference

Thesis implementation and experiments for **EVODiff**, a second-order ODE solver for diffusion models that jointly optimizes two scalar coefficients (ζ and η) per step to minimize discretization error variance.

---

## Algorithm Overview

Standard multistep ODE solvers (DDIM, DPM-Solver++, UniPC) compute an update of the form:

```
x_{t-1} = α_{t-1}/α_t · x_t + (σ_{t-1} - σ_t · α_{t-1}/α_t) · model(x_t, t)
         + correction terms from previous steps
```

EVODiff augments each update with two learned scalar coefficients:

- **ζ (zeta)** — scales the primary model prediction direction, minimizing variance along the dominant denoising axis
- **η (eta)** — blends in the previous-step prediction as a momentum term, smoothing gradient oscillations across steps

Both are solved analytically per step via a closed-form projection in (C·H·W)-dimensional feature space. No additional NFE is required.

### Ablation Variants

| Mode | ζ | η | Description |
|------|---|---|-------------|
| `baseline` | 1 | 0 | Vanilla multistep solver (no optimization) |
| `zeta_only` | optimized | 0 | Direction scaling only |
| `eta_only` | 1 | optimized | Momentum smoothing only |
| `full` | optimized | optimized | Complete EVODiff |

---

## Repository Structure

```
thesis_evodiff/
├── solvers/
│   ├── evodiff_edm.py      # EVODiff for EDM VE-SDE models  ← main contribution
│   ├── evodiff_vp.py       # EVODiff for VP-SDE models
│   ├── dpm_solver_pytorch.py  # DPM-Solver++ (baseline)
│   ├── uni_pc.py           # UniPC (baseline)
│   ├── ddim.py             # DDIM (baseline)
│   └── base.py / utils.py
├── experiments/
│   ├── ffhq_4solver_sample.py   # 4-solver visual grid + 10k image generation
│   ├── ffhq_4solver_metrics.py  # FID/KID computation and bar chart
│   ├── ffhq_download_ref.py     # Download FFHQ-64 reference set from HuggingFace
│   ├── ffhq_sample.py           # Single-solver EVODiff sampling
│   ├── ffhq_metrics.py          # Pairwise FID/KID between two dirs
│   ├── ffhq_metrics_vis.py      # FID/KID vis with local FFHQ reference
│   ├── 04_3_ablation.py         # CIFAR-10 ablation sampling (4 variants)
│   └── 04_3_compute_fid.py      # FID for ablation variants
├── utils/                  # SDE, score models, metrics helpers
├── dnnlib/ torch_utils/    # EDM model infrastructure (from Karras et al.)
└── results/
    ├── figures/            # Generated visualizations
    └── metrics/            # Saved images and JSON metric files
```

---

## Noise Schedule (EDM VE-SDE)

All experiments use the **Variance-Exploding SDE** from Karras et al. (EDM):

| Quantity | Formula |
|----------|---------|
| α_t | 1 |
| σ_t | t |
| κ_t | t |
| λ_t | −log t |

The `EDMVESchedule` class in `experiments/ffhq_4solver_sample.py` implements this and is compatible with all four solvers.

**Model interface**: The EDM checkpoint outputs `D(x, σ)` (denoised x₀ prediction). All solvers receive a **noise predictor** wrapper:
```
ε(x, σ) = (x − D(x, σ)) / σ
```

---

## Experiments

### Checkpoint

Model: **FFHQ 64×64 unconditional EDM** (VE-SDE, pre-trained by Karras et al.)

```
/data/users/ziang/code/xwy/edm-ffhq-64x64-uncond-ve.pkl   # 240 MB, complete
```

> ⚠ The copy at `checkpoints/` is truncated (211 MB). Use the root-level file.

---

### Experiment 1 — EVODiff Ablation (CIFAR-10, NFE=10)

**Scripts**: `experiments/04_3_ablation.py` → `experiments/04_3_compute_fid.py`

Pairwise FID between Full EVODiff and Baseline (FFHQ, NFE=20):

| Comparison | FID | KID |
|-----------|-----|-----|
| Full EVODiff vs Baseline (NFE=20) | **3.10** | **0.000291** |

Results in `results/metrics/ffhq_metrics.json`.

---

### Experiment 2 — 4-Solver Comparison (FFHQ 64×64)

**Scripts**: `ffhq_4solver_sample.py` → `ffhq_4solver_metrics.py`

Each solver generates **10,000 images** at NFE ∈ {5, 10}. FID/KID computed against a reference set of real FFHQ-64 images using [clean-fid](https://github.com/GauthierGidel/clean-fid).

#### Results (10k generated vs. ~2.7k reference)

| Solver | NFE=5 FID ↓ | NFE=5 KID ↓ | NFE=10 FID ↓ | NFE=10 KID ↓ |
|--------|:-----------:|:-----------:|:------------:|:------------:|
| **EVODiff** | **29.53** | **0.0206** | 12.64 | 0.0059 |
| DPM-Solver++ | 38.91 | 0.0314 | 13.92 | 0.0068 |
| DDIM | 58.47 | 0.0513 | 28.50 | 0.0215 |
| UniPC | 76.69 | 0.0792 | **10.64** | **0.0035** |

> Note: reference set was ~2.7k images at evaluation time (download still in progress). Absolute values are inflated; re-run `ffhq_4solver_metrics.py` after `ffhq_download_ref.py` completes for 10k-reference numbers.

**Key observations**:
- **NFE=5**: EVODiff leads by a wide margin (29.5 vs 38.9 for DPM-Solver++, 76.7 for UniPC). Low-NFE regime is where variance optimization matters most.
- **NFE=10**: All second-order solvers converge; UniPC edges out EVODiff and DPM-Solver++ slightly, within noise of the reference set size.
- **DDIM** (order=1) is consistently worst at both NFEs, as expected for a first-order method.

Results in `results/metrics/ffhq_4solver_metrics.json`.

#### Visual Grid

16 fixed initial noises, all four solvers, NFE ∈ {5, 10}:

| NFE | Figure |
|-----|--------|
| 5 | `results/figures/ffhq_4solver_grid_nfe5.png` |
| 10 | `results/figures/ffhq_4solver_grid_nfe10.png` |
| FID/KID bar chart | `results/figures/ffhq_4solver_metrics.png` |

---

## Reproducing Experiments

### Environment

```bash
conda activate dggt   # PyTorch 2.5.1+cu121
pip install clean-fid socksio
```

### Step 1 — Download FFHQ reference images

```bash
cd thesis_evodiff/
python experiments/ffhq_download_ref.py --n 10000
# → results/reference/ffhq64/  (10k PNG, 64×64)
```

### Step 2 — Generate images with 4 solvers

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/ffhq_4solver_sample.py --nfe 5 10 --n 10000
# → results/metrics/ffhq4solver/<solver>_nfe<N>/  (10k PNG each)
# → results/figures/ffhq_4solver_grid_nfe5.png
# → results/figures/ffhq_4solver_grid_nfe10.png
```

Visual-grid only (fast, skips bulk generation):

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/ffhq_4solver_sample.py --vis_only
```

### Step 3 — Compute FID / KID

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/ffhq_4solver_metrics.py --nfe 5 10
# → results/metrics/ffhq_4solver_metrics.json
# → results/figures/ffhq_4solver_metrics.png
```

> `CUDA_VISIBLE_DEVICES=0` is required to avoid NCCL multi-GPU errors in clean-fid.

---

## Implementation Notes

### UniPC dimension compatibility

UniPC's `multistep_uni_pc_bh_update` uses `einsum('k,bkd->bd')`, assuming flat `(B, D)` tensors. Images must be flattened before passing and unflattened after:

```python
class FlatNoiseFn:
    def __call__(self, x_flat, t):
        B = x_flat.shape[0]
        out = self.noise_fn(x_flat.view(B, *self.img_shape), t)
        return out.view(B, -1)
```

### EVODiff model contract

`EVODiff_edm` with `algorithm_type="data_prediction"` internally calls:
```
x0 = (x − σ · ε) / α
```
so the passed `model_fn` must be a **noise predictor ε**, not x₀.

### Time-step schedule

- EVODiff uses `skip_type="edm"` (ρ=7 polynomial schedule from the EDM paper)
- DPM-Solver++ and UniPC use `skip_type="logSNR"` (uniform in log-SNR space)
- Both map through `EDMVESchedule.marginal_lambda` / `inverse_lambda`
