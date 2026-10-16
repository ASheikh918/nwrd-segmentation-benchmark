# CANet-B4 vs baselines: final comparison

Test split: 960 patches, 266 with rust, 8.0% rust pixels. Threshold chosen per run on val (IoU-maximising, grid 0.05–0.95), then frozen for test. Metrics pooled over all test pixels; mean ± sample std over seeds.

## Accuracy (test split, val-selected threshold)

| Model | Seeds | IoU | Dice | Precision | Recall | Specificity | Accuracy | IoU @0.5 | Threshold |
|---|---|---|---|---|---|---|---|---|---|
| **canet_b4** (ours) | 3 | 0.7183 ± 0.0140 | 0.8360 ± 0.0094 | 0.8308 ± 0.0358 | 0.8426 ± 0.0235 | 0.9848 ± 0.0043 | 0.9734 ± 0.0023 | 0.6928 ± 0.0146 | 0.79 |
| deeplabv3plus_r50 | 3 | 0.7526 ± 0.0203 | 0.8587 ± 0.0132 | 0.8712 ± 0.0430 | 0.8479 ± 0.0159 | 0.9888 ± 0.0044 | 0.9775 ± 0.0028 | 0.7349 ± 0.0219 | 0.78 |
| unet | 3 | 0.7229 ± 0.0296 | 0.8390 ± 0.0198 | 0.8075 ± 0.0453 | 0.8743 ± 0.0100 | 0.9815 ± 0.0054 | 0.9729 ± 0.0042 | 0.6882 ± 0.0432 | 0.71 |
| unet_tuned | 1 | 0.6298 | 0.7728 | 0.7107 | 0.8468 | 0.9698 | 0.9599 | 0.5929 | 0.67 |

## Against the NWRD paper (UNet + APF)

| Result | Precision | Recall | F1 / Dice | IoU |
|---|---|---|---|---|
| Paper, Table 3, full test set | 0.506 | 0.624 | 0.557 | – |
| Paper, Table 4, background-only patches removed | 0.593 | 0.552 | 0.564 | 0.438 |
| canet_b4, re-run: all test patches | 0.831 | 0.843 | 0.836 | 0.718 |
| canet_b4, re-run: rust-positive test patches (≈ Table 4) | 0.908 | 0.843 | 0.874 | 0.776 |
| deeplabv3plus_r50, re-run: all test patches | 0.871 | 0.848 | 0.859 | 0.753 |
| deeplabv3plus_r50, re-run: rust-positive test patches (≈ Table 4) | 0.920 | 0.848 | 0.882 | 0.789 |
| unet, re-run: all test patches | 0.808 | 0.874 | 0.839 | 0.723 |
| unet, re-run: rust-positive test patches (≈ Table 4) | 0.882 | 0.874 | 0.878 | 0.782 |
| unet_tuned, re-run: all test patches | 0.711 | 0.847 | 0.773 | 0.630 |
| unet_tuned, re-run: rust-positive test patches (≈ Table 4) | 0.835 | 0.847 | 0.841 | 0.725 |

Paper figures are as published, on a different test unit (full-image patches, 128px). The `unet` row is the paper's architecture re-trained under this protocol, so CANet vs `unet` is the like-for-like comparison.

## Is the difference real? (CANet-B4 minus baseline, test split)

| Baseline | Paired seeds | ΔIoU | 95% CI, resampling photographs | 95% CI, resampling patches | ΔDice | ΔDice 95% CI, photographs | seed-level t-test p |
|---|---|---|---|---|---|---|---|
| deeplabv3plus_r50 | 3 | -0.0343 | [-0.0590, -0.0138] (p 0.0000) | [-0.0562, -0.0128] (p 0.0032) | -0.0227 | [-0.0403, -0.0090] (p 0.0000) | 0.1695 |
| unet | 3 | -0.0046 | [-0.0357, +0.0309] (p 0.7384) | [-0.0267, +0.0168] (p 0.6748) | -0.0029 | [-0.0229, +0.0221] (p 0.7540) | 0.7127 |
| unet_tuned | 1 | +0.0838 | [+0.0026, +0.1526] (p 0.0364) | [+0.0498, +0.1185] (p 0.0000) | +0.0600 | [+0.0017, +0.1208] (p 0.0364) | n/a (1 seed) |

Two runs of the paper's UNet exist (shared lr and `unet_tuned`). The stronger on **validation** IoU, `unet`, is the paper-architecture baseline; choosing by test IoU would bias the comparison.

A CI that excludes 0 means the gap is larger than test-set sampling noise. Use the photograph interval: the test patches are cut from a few photographs, and patches of one photograph are not independent, so resampling patches understates the uncertainty. Neither bootstrap captures training-run variance; the seed-level t-test does, but has little power with 3 seeds.

## Efficiency (Tesla T4, 512x512 input, torch 2.11.0+cu128)

| Model | Params (M) | GFLOPs | latency_ms_bs1_fp32 | latency_ms_bs1_fp16 | throughput_img_s_bs8_fp16 | peak_mem_MB_bs8_fp16 | CPU latency bs1 (ms) | Train min/epoch |
|---|---|---|---|---|---|---|---|---|
| canet_b4 | 29.39 | 36.9 | 29.4 | 36.1 | 75.4 | 1167.7 | 933 | 4.3 |
| deeplabv3plus_r50 | 26.68 | 73.1 | 25.0 | 9.8 | 108.2 | 2243.0 | 895 | 3.3 |
| unet | 31.04 | 385.3 | 87.1 | 36.0 | 30.8 | 1873.6 | 4760 | 6.7 |
| unet_tuned | 31.04 | 385.3 | 82.8 | 34.3 | 29.7 | 1872.6 | 5130 | 6.3 |
| canet_b4_twopass | 29.39 | 52.5 | 52.1 | 58.1 | 40.6 | 707.6 | 1390 | – |

`canet_b4_twopass` is CANet as originally implemented (encoder run twice per forward). Outputs are bit-identical to `canet_b4`; only the cost differs.

## Models

- `canet_b4` — CANet, EfficientNet-B4 encoder (ours)
- `deeplabv3plus_r50` — DeepLabV3+, ResNet-50 encoder (baseline)
- `unet` — UNet, 4 encoder / 4 decoder blocks, from scratch (NWRD paper baseline)
- `unet_tuned` — UNet (paper architecture) with a from-scratch learning rate (config.unet_tuned_lr)
