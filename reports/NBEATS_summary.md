# N-BEATS — Summary

> **Status: Round 2 (holiday-fix) FINAL.** Supersedes the Round 1 numbers
> previously in this file. Full before/after comparison, including the
> honest mixed result (ensemble improved, individual single models did not),
> is in `reports/nbeats_holiday_fix_summary.md` — read that alongside this
> file. Round 1's raw numbers are preserved in
> `reports/nbeats_finalize_results_baseline.json` and `plots/baseline/`.
>
> This round's HPO study (`nbeats_hpo_v2`) ran to **100/100 completed
> trials**, zero pruned. The finalize run's production ensemble tier (8
> members trained on full history, for Kaggle-submission checkpoints only —
> it reports no WMAE and isn't used in any number below) was stopped early
> to save time; every number below comes from a real, fully-completed
> training run logged to MLflow, reconstructed from those logged metrics
> after the run was cut short. Rerun `run_nbeats_finalize.py` uninterrupted
> later if production checkpoints are needed for an actual submission.

## Methodology recap

- Global, shared-weight N-BEATS (doubly-residual stacking, generic +
  interpretable [trend/Fourier-seasonality] configurations), trained across
  every Store-Dept series with sufficient history — no per-series models.
- Walk-forward CV **identical** to `model_experiment_LightGBM.ipynb`:
  `INITIAL_TRAIN_WEEKS=52`, `VAL_WEEKS=13`, `N_FOLDS=3`, expanding window,
  same fold dates/holidays (verified in `model_experiment_NBEATS.ipynb`
  Section 2-3).
- Horizon `H = VAL_WEEKS = 13`; lookback multiplier **fixed at 4x H** (52
  weeks = 1 full year) this round, not searched as a range — the CV folds
  are only 52/65/78 weeks, so 4x is the only multiplier that ever satisfies
  the >=2-valid-fold requirement (5x-7x always prune instantly regardless of
  any other hyperparameter), and shorter lookbacks (2x/3x) structurally
  can't contain the prior occurrence of a yearly holiday in their own input
  window — the reason for the holiday-fix work in the first place. See
  `reports/nbeats_holiday_fix_summary.md` for the full rationale.
- Loss: WMAE-weighted (holiday weeks x5) vs MAE vs SMAPE vs a new
  **asym_wmae** (adds a further 2x penalty specifically for underpredicting
  on holiday weeks), searched.
- **Holiday oversampling**: `WeightedRandomSampler` upweights training
  windows containing a holiday week, strength controlled by a searched
  `holiday_boost` hyperparameter.
- Trial selection: `0.5*mean_wmae + 0.5*mean_holiday_wmae` when a trial's CV
  folds include a usable holiday-week validation range, falling back to
  plain `mean_wmae` otherwise.
- MLflow experiment `NBEATS_Training` (DagsHub-hosted:
  https://dagshub.com/tgela23/ml-final-project.mlflow).

## Hyperparameter search

- Script: `run_nbeats_hpo.py`, study `nbeats_hpo_v2` (`nbeats_optuna.db`) —
  isolated from Round 1's `nbeats_hpo` study (different search space, not
  comparable, not used for selection).
- Search space: architecture (generic/interpretable/mixed), stack/block
  counts, FC depth (2-5 layers) and width (128/256/512), lookback fixed at
  4x H, learning rate (log-uniform 1e-4 - 1e-2), batch size (256/512/1024),
  optimizer (Adam/AdamW), weight decay, loss function
  (WMAE/MAE/SMAPE/asym_wmae), dropout, block weight sharing, holiday_boost.
- Trials: **100/100 completed, 0 pruned** (fixing the lookback at 4x
  eliminated the CV-fold-too-short pruning Round 1 hit on 10/30 trials).

### Best generic config

Trial 9, CV mean WMAE **1950.01** (Round 1: trial 23, 1911.10 — worse this
round on CV, see honest-takeaway note below).

```
architecture=generic, n_stacks=1, n_blocks=4, layer_size=256, n_fc_layers=4,
lookback_multiplier=4, loss=asym_wmae, batch_size=256, optimizer=adam,
learning_rate=0.002219, weight_decay=0.00656, dropout=0.0088,
share_weights=True, holiday_boost=7.75
```

### Best interpretable config

Trial 27, CV mean WMAE **1891.03** (Round 1: trial 4, 1902.38 — slightly
better this round).

```
architecture=interpretable, n_blocks=3, layer_size=256, n_fc_layers=4,
lookback_multiplier=4, loss=asym_wmae, batch_size=256, optimizer=adam,
learning_rate=0.003056, weight_decay=5.06e-06, dropout=0.218,
share_weights=False, trend_degree=3, holiday_boost=1.88
```

## Final WMAE (local_test holdout)

The `local_test` holdout is the same last-52-weeks-of-`train.csv` split as
the LightGBM notebook, evaluated via 4 non-overlapping 13-week rolling
blocks re-anchored on true observed history.

| Breakdown | Best Generic (single) | Best Interpretable (single) | Evaluable Ensemble (6 members, 4x-6x H) | Ensemble, calibrated |
|---|---|---|---|---|
| Overall | 2231.00 | 2301.36 | 2292.11 | 2299.28 |
| Holiday weeks | 3295.69 | 3530.21 | 3307.55 | 3331.94 |
| Non-holiday weeks | 1787.78 | 1789.80 | 1868.96 | 1868.96 |
| Store Type A | 2577.71 | 2656.51 | 2646.79 | 2654.49 |
| Store Type B | 2064.10 | 2139.95 | 2129.71 | 2138.60 |
| Store Type C | 947.80 | 946.46 | 928.69 | 925.60 |

Holiday calibration factors (per Store Type, applied only to holiday-week
forecasts, fit only on `local_train`'s own historical holidays — no
leakage): Type A `1.014x`, Type B `1.028x`, Type C `0.984x`.

**Honest takeaway (full detail in `reports/nbeats_holiday_fix_summary.md`):**
the fix did **not** uniformly win. Both single best-model configs are
*worse* on this local holdout than Round 1's on overall and holiday WMAE,
despite HPO explicitly selecting by a combined overall+holiday score across
100 completed trials (vs. Round 1's 17). The **ensemble** is where the fix
clearly helped: overall WMAE improved (2292 vs Round 1's 2332) and holiday
WMAE improved substantially (3308 vs 3502), at the cost of a small
non-holiday regression (1869 vs 1844). Post-hoc calibration added
negligible value this run and slightly hurt the calibrated holiday number
versus the ensemble's own raw output.

## Production model

**Not run this round** (stopped early to save time — see status note at the
top). Round 1's production ensemble (6 members, 2x-7x H, fit on full
143-week history) remains the last completed production artifact; rerun
`run_nbeats_finalize.py` uninterrupted to regenerate it with the Round 2
winning configs if an actual Kaggle submission is needed.

## MLflow runs

- Parent HPO run: `NBEATS_HPO_Search` (nested run per trial, named by trial
  number), study `nbeats_hpo_v2`.
- `NBEATS_Best_Generic`, `NBEATS_Best_Interpretable` — the single final fits
  reported above (run locally on CPU this round).
- `NBEATS_Ensemble_Evaluable` (nested `member_{arch}_{mult}x` per member).
- All at https://dagshub.com/tgela23/ml-final-project.mlflow — experiment
  `NBEATS_Training`.

## Plots

Round 2 plots under `plots/nbeats_*.png` (Round 1's preserved under
`plots/baseline/`), logged as MLflow artifacts on the runs above:
decomposition, backcast reconstruction, forecast-vs-actual (best/median/worst
series), error-by-horizon-step, residual diagnostics (time series/histogram/
ACF), WMAE distribution, WMAE by holiday/Store-Type, Optuna
importance/parallel-coordinate/optimization-history (reflecting the full
100-trial `nbeats_hpo_v2` study), train/val loss curves.
