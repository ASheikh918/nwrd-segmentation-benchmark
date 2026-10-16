"""Aggregates finished runs into the final comparison.

Outputs in <out_dir>/: comparison.md, comparison.json and figures/*.png.
"""
import glob
import json
import os

import numpy as np

from .models import MODELS

METRICS = ('IoU', 'Dice', 'Precision', 'Recall', 'Specificity', 'Accuracy')

# Published NWRD paper results (Anwar et al., Sensors 2023, doi:10.3390/s23156942), UNet + APF.
# Evaluated on 128px patches of 10 held-out full images; their IoU and F1 are not mutually
# consistent (likely per-image averaging), so treat them as approximate reference points.
PAPER = [
    ('Table 3, full test set', {'Precision': 0.506, 'Recall': 0.624, 'Dice': 0.557}),
    ('Table 4, background-only patches removed', {'Precision': 0.593, 'Recall': 0.552,
                                                  'Dice': 0.564, 'IoU': 0.438}),
]
OURS = 'canet_b4'


def load_runs(out_dir):
    runs = {}
    for path in sorted(glob.glob(os.path.join(out_dir, '*', 'seed*', 'metrics.json'))):
        run_dir = os.path.dirname(path)
        model, seed = os.path.basename(os.path.dirname(run_dir)), int(os.path.basename(run_dir)[4:])
        with open(path) as f:
            m = json.load(f)
        with open(os.path.join(run_dir, 'history.json')) as f:
            h = json.load(f)
        c = np.load(os.path.join(run_dir, 'test_counts.npz'))
        runs.setdefault(model, {})[seed] = {'dir': run_dir, 'metrics': m, 'history': h,
                                            'files': c['files'], 'counts': c['at_thr']}
    return runs


def _iou(c):
    tp, fp, fn = c[..., 0], c[..., 1], c[..., 2]
    return tp / np.maximum(tp + fp + fn, 1)


def _dice(c):
    tp, fp, fn = c[..., 0], c[..., 1], c[..., 2]
    return 2 * tp / np.maximum(2 * tp + fp + fn, 1)


def source_photo(filename):
    """NWRD patches are named <photo>_patch_<n>_x<col>_y<row>.png."""
    return str(filename).split('_patch_')[0]


def paired_bootstrap(ours, base, seeds, n_boot=5000, rng_seed=0, by_photo=False):
    """Paired bootstrap of the seed-averaged difference in pooled test IoU/Dice.

    Each resample draws test patches with replacement, or with by_photo whole source
    photographs with all their patches. Patches of one photograph are correlated (same plant,
    light and annotator), so by_photo is the honest interval; the patch-level one is narrower
    than the evidence supports. The same draws are used for both models and every seed."""
    for s in seeds:
        assert (ours[s]['files'] == base[s]['files']).all(), 'test file order differs'
    files = ours[seeds[0]]['files']
    n = len(files)
    rng = np.random.default_rng(rng_seed)
    if by_photo:
        photos, inv = np.unique([source_photo(f) for f in files], return_inverse=True)
        draws = rng.integers(0, len(photos), size=(n_boot, len(photos)))
        w = np.stack([np.bincount(d, minlength=len(photos)) for d in draws])[:, inv]
    else:
        w = rng.multinomial(n, np.full(n, 1 / n), size=n_boot)
    out = {}
    for name, fn in (('IoU', _iou), ('Dice', _dice)):
        d = np.mean([fn(w @ ours[s]['counts']) - fn(w @ base[s]['counts']) for s in seeds], 0)
        point = np.mean([fn(ours[s]['counts'].sum(0)) - fn(base[s]['counts'].sum(0)) for s in seeds])
        out[name] = {'delta': float(point), 'ci95': [float(np.percentile(d, 2.5)),
                                                     float(np.percentile(d, 97.5))],
                     'p_two_sided': float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))}
    return out


def seed_ttest(ours, base, seeds):
    if len(seeds) < 2:
        return None
    from scipy import stats
    a = [ours[s]['metrics']['test@thr']['IoU'] for s in seeds]
    b = [base[s]['metrics']['test@thr']['IoU'] for s in seeds]
    t, p = stats.ttest_rel(a, b)
    return {'t': float(t), 'p': float(p), 'n_seeds': len(seeds)}


def _mean_std(vals):
    vals = np.asarray(vals, float)
    return float(vals.mean()), float(vals.std(ddof=1)) if len(vals) > 1 else 0.0


def _fmt(m, s, n):
    return f'{m:.4f} ± {s:.4f}' if n > 1 else f'{m:.4f}'


def summarise(runs):
    summary = {}
    for model, seeds in runs.items():
        ms = [seeds[s]['metrics'] for s in sorted(seeds)]
        hs = [seeds[s]['history']['history'] for s in sorted(seeds)]
        row = {'seeds': sorted(seeds), 'n': len(ms)}
        for split in ('test@thr', 'test@0.5', 'val@thr', 'test_rust_patches@thr'):
            row[split] = {k: _mean_std([m[split][k] for m in ms]) for k in METRICS}
        row['threshold'] = _mean_std([m['threshold'] for m in ms])
        row['threshold_at_grid_edge'] = any(m['threshold_at_grid_edge'] for m in ms)
        row['best_epoch'] = [m['best_epoch'] for m in ms]
        row['train_min_per_epoch'] = _mean_std([np.mean(h['epoch_seconds']) / 60 for h in hs])
        summary[model] = row
    return summary


def write_markdown(path, summary, tests, bench, test_info):
    order = [m for m in (OURS,) if m in summary] + sorted(m for m in summary if m != OURS)
    L = ['# CANet-B4 vs baselines: final comparison', '',
         f'Test split: {test_info["test_images"]} patches, {test_info["test_rust_images"]} with rust, '
         f'{100 * test_info["test_rust_pixel_fraction"]:.1f}% rust pixels. Threshold chosen per run on '
         'val (IoU-maximising, grid 0.05–0.95), then frozen for test. Metrics pooled over all test '
         'pixels; mean ± sample std over seeds.', '',
         '## Accuracy (test split, val-selected threshold)', '',
         '| Model | Seeds | ' + ' | '.join(METRICS) + ' | IoU @0.5 | Threshold |',
         '|---|---|' + '---|' * (len(METRICS) + 2)]
    for m in order:
        r = summary[m]
        cells = [_fmt(*r['test@thr'][k], r['n']) for k in METRICS]
        thr = f'{r["threshold"][0]:.2f}' + (' ⚠ edge' if r['threshold_at_grid_edge'] else '')
        L.append(f'| {"**" + m + "** (ours)" if m == OURS else m} | {r["n"]} | ' + ' | '.join(cells)
                 + f' | {_fmt(*r["test@0.5"]["IoU"], r["n"])} | {thr} |')
    L += ['', '## Against the NWRD paper (UNet + APF)', '',
          '| Result | Precision | Recall | F1 / Dice | IoU |', '|---|---|---|---|---|']
    for name, p in PAPER:
        L.append(f'| Paper, {name} | ' + ' | '.join(f'{p[k]:.3f}' if k in p else '–'
                                                     for k in ('Precision', 'Recall', 'Dice', 'IoU')) + ' |')
    for m in order:
        r = summary[m]
        for split, label in (('test@thr', 'all test patches'),
                             ('test_rust_patches@thr', 'rust-positive test patches (≈ Table 4)')):
            L.append(f'| {m}, re-run: {label} | ' + ' | '.join(
                f'{r[split][k][0]:.3f}' for k in ('Precision', 'Recall', 'Dice', 'IoU')) + ' |')
    L += ['', 'Paper figures are as published, on a different test unit (full-image patches, '
          "128px). The `unet` row is the paper's architecture re-trained under this protocol, so "
          'CANet vs `unet` is the like-for-like comparison.']
    if tests:
        ci = lambda b: f'[{b["ci95"][0]:+.4f}, {b["ci95"][1]:+.4f}] (p {b["p_two_sided"]:.4f})'
        L += ['', '## Is the difference real? (CANet-B4 minus baseline, test split)', '',
              '| Baseline | Paired seeds | ΔIoU | 95% CI, resampling photographs | '
              '95% CI, resampling patches | ΔDice | ΔDice 95% CI, photographs | seed-level t-test p |',
              '|---|---|---|---|---|---|---|---|']
        for base, t in tests.items():
            bp, tt = t['bootstrap_photo'], t['seed_ttest']
            L.append(f'| {base} | {len(t["seeds"])} | {bp["IoU"]["delta"]:+.4f} | {ci(bp["IoU"])} | '
                     f'{ci(t["bootstrap"]["IoU"])} | {bp["Dice"]["delta"]:+.4f} | {ci(bp["Dice"])} | '
                     f'{"%.4f" % tt["p"] if tt else "n/a (1 seed)"} |')
        unets = [m for m in ('unet', 'unet_tuned') if m in summary]
        if len(unets) == 2:
            best = max(unets, key=lambda m: summary[m]['val@thr']['IoU'][0])
            L += ['', f'Two runs of the paper\'s UNet exist (shared lr and `unet_tuned`). The '
                  f'stronger on **validation** IoU, `{best}`, is the paper-architecture baseline; '
                  'choosing by test IoU would bias the comparison.']
        L += ['', 'A CI that excludes 0 means the gap is larger than test-set sampling noise. '
              'Use the photograph interval: the test patches are cut from a few photographs, and '
              'patches of one photograph are not independent, so resampling patches understates '
              'the uncertainty. Neither bootstrap captures training-run variance; the seed-level '
              't-test does, but has little power with 3 seeds.']
    if bench:
        env, rows = bench['env'], {r['model']: r for r in bench['results']}
        gpu_keys = [k for k in next(iter(rows.values())) if k.startswith('gpu_')]
        L += ['', f'## Efficiency ({env["device"]}, {next(iter(rows.values()))["input"]} input, '
              f'torch {env["torch"]})', '',
              '| Model | Params (M) | GFLOPs | ' + ' | '.join(k[4:] for k in gpu_keys)
              + ' | CPU latency bs1 (ms) | Train min/epoch |',
              '|---|---|---|' + '---|' * (len(gpu_keys) + 2)]
        for m in [x for x in dict.fromkeys(order + list(rows)) if x in rows]:
            r = rows[m]
            tr = summary.get(m, {}).get('train_min_per_epoch')
            L.append(f'| {m} | {r["params_M"]:.2f} | {r["GFLOPs"]:.1f} | '
                     + ' | '.join(f'{r[k]:.1f}' for k in gpu_keys)
                     + f' | {r["cpu_latency_ms_bs1_fp32"]:.0f} | {f"{tr[0]:.1f}" if tr else "–"} |')
        if 'canet_b4_twopass' in rows:
            L += ['', '`canet_b4_twopass` is CANet as originally implemented (encoder run twice per '
                  'forward). Outputs are bit-identical to `canet_b4`; only the cost differs.']
    L += ['', '## Models', ''] + [f'- `{m}` — {MODELS[m]}' for m in order]
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')


def plot_figures(runs, fig_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(fig_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for model, seeds in sorted(runs.items()):
        for key, ax in (('val_loss', axes[0]), ('val_iou', axes[1])):
            curves = [seeds[s]['history']['history'][key] for s in sorted(seeds)]
            n = min(map(len, curves))
            arr = np.array([c[:n] for c in curves])
            x = np.arange(1, n + 1)
            line, = ax.plot(x, arr.mean(0), label=model, lw=2 if model == OURS else 1.4)
            if len(curves) > 1:
                ax.fill_between(x, arr.min(0), arr.max(0), color=line.get_color(), alpha=0.15)
    axes[0].set_title('Validation loss'); axes[1].set_title('Validation IoU (pooled, threshold 0.5)')
    for ax in axes:
        ax.set_xlabel('Epoch'); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, 'training_curves.png'), dpi=150)
    plt.close(fig)


def compare(out_dir, log=print):
    runs = load_runs(out_dir)
    if not runs:
        log('[compare] no finished runs yet'); return None
    summary = summarise(runs)
    tests = {}
    if OURS in runs:
        for base in (m for m in runs if m != OURS):
            seeds = sorted(set(runs[OURS]) & set(runs[base]))
            if seeds:
                tests[base] = {'seeds': seeds,
                               'bootstrap': paired_bootstrap(runs[OURS], runs[base], seeds),
                               'bootstrap_photo': paired_bootstrap(runs[OURS], runs[base], seeds,
                                                                   by_photo=True),
                               'seed_ttest': seed_ttest(runs[OURS], runs[base], seeds)}
    bench_path = os.path.join(out_dir, 'benchmark.json')
    bench = json.load(open(bench_path)) if os.path.exists(bench_path) else None
    any_run = next(iter(next(iter(runs.values())).values()))['metrics']
    fig_dir = os.path.join(out_dir, 'figures')
    plot_figures(runs, fig_dir)
    write_markdown(os.path.join(out_dir, 'comparison.md'), summary, tests, bench, any_run)
    with open(os.path.join(out_dir, 'comparison.json'), 'w') as f:
        json.dump({'summary': summary, 'significance': tests, 'benchmark': bench}, f, indent=2)
    log(open(os.path.join(out_dir, 'comparison.md'), encoding='utf-8').read())
    return summary
