# NWRD wheat stripe rust segmentation benchmark

A seed-controlled, reproducible comparison of three segmentation architectures for wheat
stripe (yellow) rust on the **NWRD** dataset ([Anwar et al., *Sensors* 2023](https://doi.org/10.3390/s23156942)):

- **DeepLabV3+** with an ImageNet-pretrained ResNet-50 encoder
- **CANet-B4**: CANet ([Tang et al., 2021](https://doi.org/10.1016/j.imavis.2021.104309)) adapted with an ImageNet-pretrained EfficientNet-B4 encoder
- **UNet**: the NWRD paper's architecture, trained from scratch

Every model is trained and evaluated under one protocol: the same data, loss, schedule,
seeds and threshold rule. The test split is touched once.

## Main results

Test split: 10 held-out photographs (960 patches of 512×512, 8.0% rust pixels). Mean ± sample
standard deviation over seeds 42, 43 and 44. Threshold chosen on the validation split and
then frozen.

| Model | IoU | Dice / F1 | Precision | Recall | IoU @ 0.5 |
|---|---|---|---|---|---|
| **DeepLabV3+ R50** | **0.753 ± 0.020** | **0.859 ± 0.013** | **0.871 ± 0.043** | 0.848 ± 0.016 | **0.735 ± 0.022** |
| UNet (NWRD paper architecture) | 0.723 ± 0.030 | 0.839 ± 0.020 | 0.808 ± 0.045 | **0.874 ± 0.010** | 0.688 ± 0.043 |
| CANet-B4 | 0.718 ± 0.014 | 0.836 ± 0.009 | 0.831 ± 0.036 | 0.843 ± 0.024 | 0.693 ± 0.015 |

**Are the differences real?** These are paired differences on the same seeds, with a 95% CI
from a bootstrap that resamples whole test photographs:

| Comparison | ΔIoU | 95% CI (photographs) | Seed-level paired t-test p |
|---|---|---|---|
| DeepLabV3+ R50 − CANet-B4 | +0.034 | [+0.014, +0.059] | 0.17 |
| UNet − CANet-B4 | +0.005 | [−0.031, +0.036] | 0.71 |

- **DeepLabV3+ R50 vs CANet-B4:** DeepLabV3+ is better on every seed, and the gap is larger
  than test-set sampling noise. With only 3 seeds, the seed-level test is not significant.
- **CANet-B4 vs UNet:** statistically indistinguishable.
- **Per photograph** (IoU averaged over the 3 seeds): DeepLabV3+ is best on 5 of the 10 test
  photographs, UNet on 4 and CANet-B4 on 1. No model wins everywhere. These can be recomputed
  from `results/runs/<model>/seed<N>/test_counts.npz`.

<details>
<summary>Per-seed results</summary>

| Model | Seed | Val threshold | Best epoch | Test IoU | Dice | Precision | Recall | IoU @ 0.5 |
|---|---|---|---|---|---|---|---|---|
| DeepLabV3+ R50 | 42 | 0.70 | 22 | 0.7337 | 0.8464 | 0.8279 | 0.8657 | 0.7182 |
| DeepLabV3+ R50 | 43 | 0.77 | 9 | 0.7501 | 0.8572 | 0.8720 | 0.8430 | 0.7268 |
| DeepLabV3+ R50 | 44 | 0.87 | 21 | 0.7740 | 0.8726 | 0.9138 | 0.8350 | 0.7597 |
| UNet | 42 | 0.72 | 23 | 0.6978 | 0.8220 | 0.7725 | 0.8783 | 0.6565 |
| UNet | 43 | 0.70 | 30 | 0.7555 | 0.8607 | 0.8586 | 0.8629 | 0.7374 |
| UNet | 44 | 0.72 | 27 | 0.7154 | 0.8341 | 0.7913 | 0.8817 | 0.6707 |
| CANet-B4 | 42 | 0.79 | 16 | 0.7136 | 0.8329 | 0.8402 | 0.8257 | 0.6928 |
| CANet-B4 | 43 | 0.84 | 9 | 0.7340 | 0.8466 | 0.8609 | 0.8328 | 0.7074 |
| CANet-B4 | 44 | 0.73 | 8 | 0.7073 | 0.8285 | 0.7913 | 0.8695 | 0.6782 |

</details>

### Efficiency

Measured on a Tesla T4 with 512×512 input, torch 2.11. CPU latency uses one thread.

| Model | Params (M) | GFLOPs | GPU latency fp32 (ms) | GPU latency fp16 (ms) | Throughput fp16, bs 8 (img/s) | CPU latency (ms) | Train (min/epoch) |
|---|---|---|---|---|---|---|---|
| DeepLabV3+ R50 | 26.7 | 73.1 | **25.0** | **9.8** | **108.2** | **895** | **3.3** |
| CANet-B4 | 29.4 | **36.9** | 29.4 | 36.1 | 75.4 | 933 | 4.3 |
| UNet | 31.0 | 385.3 | 87.1 | 36.0 | 30.8 | 4760 | 6.8 |

CANet-B4 needs the fewest FLOPs. DeepLabV3+ is nevertheless faster in wall-clock time on both
GPU and CPU, so FLOPs alone do not predict latency here. Compared with UNet, CANet-B4 reaches
the same accuracy with about 10× fewer FLOPs.

## Protocol

| Item | Setting |
|---|---|
| Data | 512×512 patches from the 100 NWRD photographs ([`abdur548/nwrd-patched`](https://www.kaggle.com/datasets/abdur548/nwrd-patched) on Kaggle) |
| Split | By source photograph: 85 train (IDs 1–85), 5 validation (86–90), 10 test (91–100). No photograph appears in more than one split |
| Training patches | Every rust-containing patch, plus a fixed 20% sample of background-only patches (3,709 patches; same set for every model and seed) |
| Optimisation | 30 epochs, batch 8, AdamW (lr 5e-5, weight decay 1e-4), cosine annealing with warm restarts (T₀ 5, T_mult 2), fp16, gradient clipping 1.0 |
| Encoders | Pretrained encoders frozen for the first 3 epochs; UNet is trained from scratch |
| Loss | 0.7 × Dice + 0.3 × BCE, with `pos_weight` ≈ 3.65 from the training masks |
| Model selection | Epoch with the best validation IoU |
| Threshold | Chosen on validation (IoU-maximising, grid 0.05–0.95), frozen, applied once to test. Results at a fixed 0.5 are reported too |
| Metrics | Pooled over every test pixel (not averaged per batch or per patch) |
| Uncertainty | 3 seeds (42, 43, 44); paired bootstrap resampling whole test photographs; seed-level paired t-test |

## Comparison with the NWRD paper

The NWRD paper reports F1 0.557 for UNet with adaptive patching (Table 3). That number is
**not comparable** with the results above:
- **Patch size:** 128×128 patches, against 512×512 here.
- **Test unit and sampling:** both differ.
- **Encoders:** the paper does not use pretrained encoders.

The `unet` row above is the paper's architecture re-trained under this protocol, and it is the
like-for-like baseline.

## Limitations

- **Small test set.** 10 photographs: patches from one photograph are not independent, which is
  why the bootstrap resamples photographs rather than patches.
- **Few seeds.** With 3 seeds, the seed-level test has little power.
- **Run-to-run variation.** The same seed gives different results across sessions (GPU
  nondeterminism, torch 2.10 vs 2.11): about 0.005–0.04 IoU. That is similar in size to the
  gaps between models.
- **UNet may be under-trained.** Its best epochs were 23, 30 and 27 of 30, so it may still have
  been improving. A from-scratch UNet with lr 1e-3 (`unet_tuned`) diverged (NaN loss from
  epoch 21) and is not used as the baseline.
- **Encoder confound.** CANet-B4 and DeepLabV3+ use different encoders, so the gap mixes
  decoder and encoder effects. A same-encoder control (`deeplabv3plus_b4`) is implemented but
  not yet run.

## Repository layout

```
nwrd/                      training, evaluation, benchmark and comparison code
  config.py                the protocol (every setting above)
  models.py                CANet-B4, UNet, DeepLabV3+ definitions
  data.py                  split discovery, background subsampling
  train.py / evaluate.py   one (model, seed) run; resumable
  metrics.py               pooled pixel metrics
  benchmark.py             params, FLOPs, latency, throughput, memory
  compare.py               aggregates runs into comparison.md / comparison.json
  run.py                   train -> evaluate -> benchmark -> compare
notebooks/
  nwrd_experiments.ipynb   Colab / Kaggle notebook that runs everything
results/runs/
  comparison.md            final tables (generated)
  comparison.json          all summary numbers, bootstrap and t-tests
  benchmark.json           efficiency measurements
  <model>/seed<N>/         metrics.json, history.json, val threshold sweep, test pixel counts
  figures/                 training curves, qualitative results
  log.txt                  full training log
```

Model checkpoints are not included.

## Reproduce

**Colab or Kaggle (T4 GPU).** Open `notebooks/nwrd_experiments.ipynb`, attach the
`abdur548/nwrd-patched` dataset, and run all cells. Runs resume across sessions. Budget about
1.5–2.5 GPU-hours per model and seed.

**Locally, with a GPU:**

```bash
pip install -r requirements.txt
python -m nwrd.run --data-root /path/to/wheat_rust_patches --out runs
```

**Re-aggregate the published results** (no GPU needed):

```bash
python -m nwrd.run --stage compare --out results/runs
```

Single stages: `--stage train|eval|bench|figures|compare`.

## Citation

If you use this code or these results, please cite this repository (see `CITATION.cff`) and the
NWRD dataset:

```bibtex
@article{anwar2023nwrd,
  title   = {The NWRD Dataset: An Open-Source Annotated Segmentation Dataset of Diseased Wheat Crop},
  author  = {Anwar, Hirra and others},
  journal = {Sensors},
  volume  = {23},
  number  = {15},
  pages   = {6942},
  year    = {2023},
  doi     = {10.3390/s23156942}
}
```
