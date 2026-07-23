# N-BEATS — Summary

> **Status: FINAL (search paused, not exhausted).** The Optuna search
> (`run_nbeats_hpo.py`) was paused at 30/80 requested trials to free up CPU
> for the finalize run below — it is safe to resume later and would only
> change the "best configs" if a later trial beats CV mean WMAE 1902.4. The
> numbers below are the real, final-fit results for the current best configs,
> produced by a hybrid run: the best-generic/best-interpretable single fits
> and the evaluable ensemble were run on a Colab T4 GPU (`nbeats_colab_finalize.py`,
> a GPU-enabled port of `run_nbeats_finalize.py`) after the local CPU run hit
> severe thread-contention slowdowns running alongside the HPO search.

## Methodology recap

- Global, shared-weight N-BEATS (doubly-residual stacking, generic +
  interpretable [trend/Fourier-seasonality] configurations), trained across
  every Store-Dept series with sufficient history — no per-series models.
- Walk-forward CV **identical** to `model_experiment_LightGBM.ipynb`:
  `INITIAL_TRAIN_WEEKS=52`, `VAL_WEEKS=13`, `N_FOLDS=3`, expanding window,
  same fold dates/holidays (verified in `model_experiment_NBEATS.ipynb`
  Section 2-3).
- Horizon `H = VAL_WEEKS = 13`; lookback multiplier searched 2x-7x `H`.
  Multiples 5x-7x don't have enough training windows within these CV folds
  (see the feasibility table in Section 4 of the notebook) and are
  correctly pruned by Optuna rather than scored on too little data — the
  full 2x-7x ensemble is instead built in two tiers (see below).
- Loss: WMAE-weighted (holiday weeks x5, `utils.metrics.wmae` semantics)
  vs MAE vs SMAPE, searched.
- MLflow experiment `NBEATS_Training` (DagsHub-hosted:
  https://dagshub.com/tgela23/ml-final-project.mlflow).

## Hyperparameter search

- Script: `run_nbeats_hpo.py`, study `nbeats_hpo` (`nbeats_optuna.db`).
- Search space: architecture (generic/interpretable/mixed), stack/block
  counts, FC depth (2-5 layers) and width (128/256/512), lookback multiplier
  (2-7x H), learning rate (log-uniform 1e-4 - 1e-2), batch size
  (256/512/1024), optimizer (Adam/AdamW), weight decay, loss function
  (WMAE/MAE/SMAPE), dropout, block weight sharing.
- Trials requested: 80. Trials attempted before pausing: **30** (17 completed,
  10 pruned as infeasible for their lookback multiplier/CV-fold combination,
  3 interrupted mid-trial by the pause). Resumable any time from the same
  `nbeats_optuna.db` study.

### Best generic config

Trial 23, CV mean WMAE **1911.10**.

```
architecture=generic, n_stacks=2, n_blocks=2, layer_size=256, n_fc_layers=4,
lookback_multiplier=4, loss=smape, batch_size=256, optimizer=adam,
learning_rate=0.000728, weight_decay=3.82e-06, dropout=0.191,
share_weights=False, trend_degree=3 (unused for generic)
```

### Best interpretable config

Trial 4, CV mean WMAE **1902.38** (best overall CV score of the search).

```
architecture=interpretable, n_blocks=2, layer_size=512, n_fc_layers=5,
lookback_multiplier=4, loss=mae, batch_size=1024, optimizer=adam,
learning_rate=0.001016, weight_decay=5.97e-06, dropout=0.188,
share_weights=True, trend_degree=4
```

## Final WMAE (local_test holdout)

The `local_test` holdout is the same last-52-weeks-of-`train.csv` split as
the LightGBM notebook, evaluated via 4 non-overlapping 13-week rolling
blocks re-anchored on true observed history (so the full 52-week holdout,
including Thanksgiving/Christmas/Super Bowl, is covered, not just the first
13 weeks).

| Breakdown | Best Generic (single) | Best Interpretable (single) | Evaluable Ensemble (10 members, 2x-6x H) |
|---|---|---|---|
| Overall | **2161.25** | 2286.51 | 2331.71 |
| Holiday weeks | 3134.43 | 3352.35 | 3502.22 |
| Non-holiday weeks | 1756.13 | 1842.81 | 1843.94 |
| Store Type A | 2495.68 | 2646.15 | 2675.32 |
| Store Type B | 2003.37 | 2113.33 | 2198.68 |
| Store Type C | 910.36 | 955.67 | 908.22 |

**Note:** the single best-generic model (trial 23) beat both the single
best-interpretable model and the full 10-member ensemble on this holdout —
ensembling did not help here. With only a 52-week/4-block holdout this isn't
strong evidence that ensembling is bad in general, just that it didn't pay
off on this particular holdout; worth keeping in mind rather than assuming
ensembles always win.

## Production model (2x-7x H ensemble, fit on full train.csv history)

6 members (multipliers 2x-7x H), each retrained from the best-generic config
on the **full 143-week `train.csv` history** (not just `local_train`) —
this is the only tier where the 7x H lookback has enough history to train
at all. This is the actual deliverable used for scoring Kaggle's real
`test.csv` (no labels available there, so no WMAE is reportable for it
directly; the evaluable-tier numbers above are the best available proxy for
expected performance). Trained model weights are logged as MLflow artifacts
on the nested `prod_member_{2..7}x` runs.

## MLflow runs

- Parent HPO run: `NBEATS_HPO_Search` (nested run per trial, named by trial number).
- `NBEATS_Best_Generic_Colab`, `NBEATS_Best_Interpretable_Colab` — the single
  final fits reported above (run on Colab GPU; a local CPU attempt of the
  same stages also exists under the non-`_Colab` run names, superseded by
  these).
- `NBEATS_Ensemble_Evaluable_Colab` (nested `member_{arch}_{mult}x_colab` per member).
- `NBEATS_Production_Final_Colab` (nested `prod_member_{mult}x_colab` per member,
  all using the best-generic config per `run_nbeats_finalize.py`'s convention
  of taking `base_configs[0]` for the production tier).
- All at https://dagshub.com/tgela23/ml-final-project.mlflow — experiment `NBEATS_Training`.

## Plots

All under `plots/nbeats_*.png`, logged as MLflow artifacts on the runs above:
decomposition, backcast reconstruction, forecast-vs-actual (best/median/worst
series), error-by-horizon-step, residual diagnostics (time series/histogram/
ACF), WMAE distribution, WMAE by holiday/Store-Type, Optuna
importance/parallel-coordinate/optimization-history, train/val loss curves.
