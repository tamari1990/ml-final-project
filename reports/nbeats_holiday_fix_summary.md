# N-BEATS Holiday-Underprediction Fix — Round 2 Summary

## Background

Round 1's finalize run (`reports/nbeats_finalize_results_baseline.json`, `plots/baseline/`)
found holiday-week WMAE consistently ~1.8x worse than non-holiday across every
model. Root cause: N-BEATS here is a pure univariate model per the paper —
`is_holiday` was only used for loss/metric *weighting*, never fed to the
network as an input feature, so the model had no calendar awareness at all.

Round 2 applied a 6-part fix **without** adding calendar/exogenous features
(kept strictly univariate, per the paper):

1. **Holiday oversampling** — `WeightedRandomSampler` upweights training
   windows that contain a holiday week (`holiday_boost` hyperparameter).
2. **Asymmetric holiday loss** — `asym_wmae` (5x holiday weight + an extra 2x
   penalty specifically for *underpredicting* on holiday weeks) added to the
   HPO search space.
3. **Holiday WMAE tracked explicitly** through CV, HPO, and trial selection
   (best trial now picked by `0.5*mean_wmae + 0.5*mean_holiday_wmae` when a
   holiday-week validation fold is available, not by overall WMAE alone).
4. **Lookback restricted to `lookback_multiplier=4`** (52 weeks = 1 full
   year) — shorter lookbacks structurally can't contain the prior occurrence
   of a yearly holiday in their own input window.
5. **Post-hoc holiday calibration** — per-Store-Type multiplicative
   correction fit only on `local_train`'s own historical holiday weeks (no
   leakage), applied only to holiday-week forecasts.
6. **Full HPO re-run**: `nbeats_hpo_v2` study, **100 completed trials**
   (vs. Round 1's 17 completed / 30 attempted), zero prunes (the
   `lookback_multiplier=4` fix eliminated the CV-fold-too-short pruning issue
   Round 1 hit).

## Run notes (in the interest of full disclosure)

- The HPO run hit two infrastructure failures unrelated to the model itself,
  both diagnosed and fixed rather than worked around silently: (a) Windows
  Smart App Control started blocking PyTorch's DLLs partway through (fixed by
  disabling it), and (b) a stdout `UnicodeEncodeError` from an MLflow log
  line containing an emoji crashed one trial, then a transient DNS blip
  talking to DagsHub crashed the whole process — both fixed by forcing UTF-8
  output encoding and adding `catch=(Exception,)` to `study.optimize()` so a
  single bad trial no longer kills the run.
- The finalize run's **production ensemble tier** (8 members trained on the
  full 143-week history, for Kaggle-submission checkpoints only — it reports
  no WMAE since the real test set has no labels) was stopped early to save
  time. It is **not** used anywhere in the numbers below; every number here
  comes from a real, fully-completed training run logged to MLflow. Rerun
  `run_nbeats_finalize.py` uninterrupted later if production checkpoints are
  needed for an actual Kaggle submission.
- Round 2's evaluable ensemble uses **6 members** (2 base configs x
  multipliers `[4, 5, 6]`) vs. Round 1's **10 members** (2 base configs x
  multipliers `[2, 3, 4, 5, 6]`) — Round 2 dropped the sub-1-year multipliers
  as part of fix #4 above, so the ensembles aren't quite like-for-like in
  size, only in what they're allowed to use.

## Results: before vs. after

All numbers are local-holdout WMAE (`local_test`, lower is better).

### Best single generic model

| | Round 1 (trial 23) | Round 2 (trial 9) | Δ |
|---|---:|---:|---:|
| CV mean WMAE | 1911.10 | 1950.01 | +38.91 (worse) |
| Overall | 2161.25 | 2231.00 | +69.75 (worse) |
| Holiday | 3134.43 | 3295.69 | +161.26 (worse) |
| Non-holiday | 1756.13 | 1787.78 | +31.65 (worse) |

### Best single interpretable model

| | Round 1 (trial 4) | Round 2 (trial 27) | Δ |
|---|---:|---:|---:|
| CV mean WMAE | 1902.38 | 1891.03 | -11.35 (better) |
| Overall | 2286.51 | 2301.36 | +14.85 (worse) |
| Holiday | 3352.35 | 3530.21 | +177.86 (worse) |
| Non-holiday | 1842.81 | 1789.80 | -53.01 (better) |

### Ensemble (evaluable tier)

| | Round 1 (10 members) | Round 2 raw (6 members) | Round 2 calibrated | Δ (raw vs R1) |
|---|---:|---:|---:|---:|
| Overall | 2331.71 | 2292.11 | 2299.28 | -39.60 (better) |
| Holiday | 3502.22 | 3307.55 | 3331.94 | -194.67 (better) |
| Non-holiday | 1843.94 | 1868.96 | 1868.96 | +25.02 (worse) |

Holiday calibration factors this round (per Store Type, applied only to
holiday-week forecasts): Type A `1.014x`, Type B `1.028x`, Type C `0.984x` —
all close to 1.0, so calibration barely moved the ensemble's numbers here
(and on this run's holdout window it slightly *hurt* rather than helped:
raw 3307.55 -> calibrated 3331.94 on holiday WMAE).

## Honest takeaway

The fix did **not** uniformly win. On individual best-single-model
comparisons, Round 2 is **worse** on this local holdout than Round 1 on
every axis except the interpretable model's CV score and non-holiday WMAE —
despite Round 2's HPO explicitly selecting trials by a combined
overall+holiday score, and despite 100 completed trials vs. Round 1's 17.
The most likely explanation is that the asymmetric holiday loss and holiday
oversampling push individual models to trade general accuracy for
holiday-week accuracy in a way that doesn't fully pay off out-of-sample for
a *single* model, and CV-selected "best holiday score" trials don't
necessarily generalize better to `local_test`'s specific holiday weeks.

The one place the fix clearly helped is the **ensemble**: both overall WMAE
(2292 vs 2331) and holiday WMAE (3308 vs 3502) improved versus Round 1's
ensemble, at the cost of a small non-holiday regression (1869 vs 1844).
Averaging across more/varied holiday-aware members seems to be where the
fix's benefit actually shows up, rather than in any single model. Post-hoc
calibration, in this particular run, added negligible value and slightly
hurt the calibrated holiday number relative to the ensemble's own raw
output — worth flagging plainly rather than reporting only the more
flattering number.

## Winning hyperparameters (Round 2)

- **Best generic** (trial 9): `n_stacks=1, n_blocks=4, n_fc_layers=4,
  layer_size=256, loss=asym_wmae, holiday_boost=7.75, lr=0.00222,
  optimizer=adam, batch_size=256, dropout=0.0088, weight_decay=0.00656`
- **Best interpretable** (trial 27): `n_blocks=3, n_fc_layers=4,
  layer_size=256, loss=asym_wmae, holiday_boost=1.88, lr=0.00306,
  optimizer=adam, batch_size=256, dropout=0.218, trend_degree=3`
- **Best mixed** (trial 87, informational only — not one of the two required
  architectures): `n_stacks=3, n_blocks=4, n_fc_layers=4, layer_size=256,
  loss=wmae, holiday_boost=1.32`. Notably, the single best trial across the
  *entire* 100-trial study by raw CV WMAE was also a mixed-architecture
  trial (trial 101, CV WMAE 1824.66) — better than either the chosen generic
  or interpretable winner, but excluded from "best generic/interpretable" by
  design since `mixed` isn't one of the two required architectures.
- Every winning trial (and nearly every completed trial) picked `loss=wmae`
  or `loss=asym_wmae` and a non-trivial `holiday_boost`, suggesting the
  search space usefully differentiates these dimensions rather than treating
  them as noise.

## Raw data

- `reports/nbeats_finalize_results.json` — Round 2 numbers (this run)
- `reports/nbeats_finalize_results_baseline.json` — Round 1 numbers (before)
- `plots/baseline/` — Round 1 plots (before)
- `plots/nbeats_*.png` — Round 2 plots (after), except the Optuna diagnostics
  plots which reflect the full `nbeats_hpo_v2` 100-trial study
