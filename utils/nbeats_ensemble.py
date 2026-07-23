"""Final-model training and ensembling for N-BEATS: fitting the winning
configs on all of local_train_raw, rolling-block evaluation across the full
52-week local_test_raw holdout, and prediction-averaging across lookback
multiples per the N-BEATS paper's ensembling recipe.

Two training tiers (see model_experiment_NBEATS.ipynb Section 6 for the
full rationale):
  * "evaluable" members: lookback multiples 2x-6x H, trained only on
    local_train_raw (91 weeks) so their WMAE against local_test_raw is a
    genuine, leakage-free holdout score. 7x H needs >=104 training weeks
    (lookback+horizon) and cannot be trained at all within the 91-week
    local_train_raw pool -- it is a "production-only" member (see below).
  * "production" members: same configs, trained on the FULL train.csv
    history (local_train_raw + local_test_raw, 143 weeks) once evaluation
    is done -- this is the deliverable used for real Kaggle test.csv
    inference and is the only tier where 7x H is trainable at all.
"""

import numpy as np

from utils.nbeats_data import build_eval_window, build_training_windows
from utils.nbeats_train import HORIZON, build_model, evaluate_on_window, forecast_series, train_fixed_epochs
from utils.metrics import wmae as wmae_ref


def train_final_member(config, sales, is_holiday, train_end_idx, n_epochs, seed=42, verbose=False, device='cpu'):
    """Train one ensemble member using EVERY window within [0, train_end_idx]
    (no validation slice reserved) for exactly `n_epochs`.

    `n_epochs` should come from the winning HPO trial's `mean_best_epoch`
    (mean best-epoch across CV folds) -- mirrors the LightGBM notebook's
    `best_n_estimators` convention (fit the final model for the CV-tuned
    number of rounds rather than re-running early stopping on a freshly
    reserved slice, which would just shrink the training pool for no
    benefit: this model's real held-out evaluation is the local_test rolling
    -block forecast in run_nbeats_finalize.py, never touched during
    training). Returns (model, history, lookback).
    """
    lookback = config['lookback_multiplier'] * HORIZON
    max_anchor_idx = train_end_idx - HORIZON
    X_tr, Y_tr, Yh_tr, _ = build_training_windows(
        sales, is_holiday, lookback, HORIZON, max_anchor_idx=max_anchor_idx)

    model = build_model(config, device=device)
    model, history = train_fixed_epochs(
        model, X_tr, Y_tr, Yh_tr, config, n_epochs, seed=seed, verbose=verbose)
    return model, history, lookback


def rolling_block_forecast(model, lookback, sales, is_holiday, first_anchor_idx, horizon, n_blocks):
    """Non-overlapping `n_blocks` forecasts of `horizon` weeks each,
    re-anchored on TRUE observed history each time (legitimate: at block k
    the model only ever sees actuals strictly before that block's own
    forecast window, exactly like re-forecasting weekly in production).

    Returns concatenated y_true, y_pred, is_holiday, series_idx (series_idx
    repeated per block, so grouping by it still identifies the series) and
    a `block_id`/`step_in_block` array for the per-horizon-step error plot.
    """
    all_true, all_pred, all_holiday, all_sidx, all_block, all_step = [], [], [], [], [], []
    for b in range(n_blocks):
        anchor = first_anchor_idx + b * horizon
        val_start = anchor + 1
        res = evaluate_on_window(model, sales, is_holiday, lookback, anchor, val_start, horizon)
        wmae_b, y_true, y_pred, holiday_grid, series_idx = res
        if y_true is None:
            continue
        all_true.append(y_true)
        all_pred.append(y_pred)
        all_holiday.append(holiday_grid)
        all_sidx.append(series_idx)
        all_block.append(np.full(len(series_idx), b))
        all_step.append(np.tile(np.arange(1, horizon + 1), (len(series_idx), 1)))

    return (np.concatenate(all_true), np.concatenate(all_pred), np.concatenate(all_holiday),
            np.concatenate(all_sidx), np.concatenate(all_block), np.concatenate(all_step))


def ensemble_rolling_forecast(members, sales, is_holiday, first_anchor_idx, horizon, n_blocks):
    """Same as rolling_block_forecast but averages predictions (in original
    Weekly_Sales units) across `members` -- list of dicts with 'model' and
    'lookback'. Restricts each block to the intersection of series every
    member can forecast (i.e. has a NaN-free lookback for), so the average
    is always over the same series."""
    all_true, all_pred, all_holiday, all_sidx, all_block, all_step = [], [], [], [], [], []
    for b in range(n_blocks):
        anchor = first_anchor_idx + b * horizon
        val_start = anchor + 1

        member_windows = []
        common_idx = None
        for m in members:
            X_eval, series_idx = build_eval_window(sales, m['lookback'], anchor)
            member_windows.append((X_eval, series_idx))
            common_idx = series_idx if common_idx is None else np.intersect1d(common_idx, series_idx)
        if common_idx is None or len(common_idx) == 0:
            continue

        y_true = sales[common_idx, val_start:val_start + horizon]
        valid_rows = ~np.isnan(y_true).any(axis=1)
        if valid_rows.sum() == 0:
            continue
        common_idx = common_idx[valid_rows]
        y_true = y_true[valid_rows]

        preds = []
        for m, (X_eval, series_idx) in zip(members, member_windows):
            pos = np.searchsorted(series_idx, common_idx)
            X_sub = X_eval[pos]
            preds.append(forecast_series(m['model'], X_sub))
        y_pred = np.mean(preds, axis=0)

        holiday_grid = np.broadcast_to(is_holiday[val_start:val_start + horizon], y_true.shape)
        all_true.append(y_true)
        all_pred.append(y_pred)
        all_holiday.append(holiday_grid)
        all_sidx.append(common_idx)
        all_block.append(np.full(len(common_idx), b))
        all_step.append(np.tile(np.arange(1, horizon + 1), (len(common_idx), 1)))

    return (np.concatenate(all_true), np.concatenate(all_pred), np.concatenate(all_holiday),
            np.concatenate(all_sidx), np.concatenate(all_block), np.concatenate(all_step))


def compute_wmae(y_true, y_pred, is_holiday):
    return wmae_ref(y_true.flatten(), y_pred.flatten(), is_holiday.flatten())
