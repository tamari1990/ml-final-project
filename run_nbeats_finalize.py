"""Finalize N-BEATS: pull the best generic/interpretable configs from the
Optuna study (run_nbeats_hpo.py), retrain them properly, build both
ensemble tiers, regenerate every required plot with the real winning
models, log everything to MLflow (DagsHub), and write the markdown summary
deliverable (reports/NBEATS_summary.md).

Works with however many trials have completed so far -- safe to run before
the full search finishes (best-so-far), and safe to re-run later once more
trials land (it always re-reads the study fresh).

Usage:
    .venv/Scripts/python run_nbeats_finalize.py
"""

import json
import os

import dagshub
import mlflow
import numpy as np
import optuna
import pandas as pd
import torch

from utils import nbeats_plots as P
from utils.nbeats_data import build_eval_window, build_panel
from utils.nbeats_ensemble import (
    ensemble_rolling_forecast, rolling_block_forecast, train_final_member,
)
from utils.nbeats_train import HORIZON, forecast_series, local_split_idx, make_cv_folds, run_cv_for_config

DATA_DIR = 'data/raw/walmart-recruiting-store-sales-forecasting/'
STUDY_NAME = 'nbeats_hpo'
STORAGE = 'sqlite:///nbeats_optuna.db'
PLOTS_DIR = 'plots'
REPORTS_DIR = 'reports'
MODELS_DIR = 'models'

# Restricted to >=4 (>= 52 weeks = 1 full year, since HORIZON=13): shorter
# lookbacks structurally can't contain the prior occurrence of a yearly
# holiday in their own window, so they're guessing blind on holiday weeks
# and just dilute the ensemble average (see the holiday-underprediction
# pattern in reports/nbeats_finalize_results.json / nbeats_wmae_breakdown.png).
EVALUABLE_MULTIPLIERS = [4, 5, 6]
PRODUCTION_MULTIPLIERS = [4, 5, 6, 7]


def trial_config(trial):
    p = trial.params
    return dict(
        architecture=p['architecture'],
        n_stacks=p.get('n_stacks', 2),
        n_blocks=p['n_blocks'],
        layer_size=p['layer_size'],
        n_fc_layers=p['n_fc_layers'],
        lookback_multiplier=p['lookback_multiplier'],
        loss=p['loss'],
        batch_size=p['batch_size'],
        optimizer=p['optimizer'],
        learning_rate=p['learning_rate'],
        weight_decay=p['weight_decay'],
        dropout=p['dropout'],
        share_weights=p['share_weights'],
        trend_degree=p.get('trend_degree', 3),
        holiday_boost=p.get('holiday_boost', 0.0),
    )


def best_trial_for_architectures(study, archs):
    """Combined-score selection: 0.5*mean_wmae + 0.5*mean_holiday_wmae when a
    trial has a usable mean_holiday_wmae (not every trial's CV folds contain a
    holiday week in their val range), falling back to pure mean_wmae
    otherwise -- so a config that wins overall but is quietly terrible on
    holidays doesn't get picked as "best" over one with a real, if slightly
    worse-overall, holiday score."""
    complete = [t for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE and t.params.get('architecture') in archs]
    if not complete:
        return None

    def combined_score(t):
        holiday_wmae = t.user_attrs.get('mean_holiday_wmae')
        if holiday_wmae is None:
            return t.value
        return 0.5 * t.value + 0.5 * holiday_wmae

    return min(complete, key=combined_score)


def per_series_wmae(y_true, y_pred, holiday_grid, series_idx):
    abs_err = np.abs(y_true - y_pred)
    weights = np.where(holiday_grid, 5, 1)
    out = {}
    for s in np.unique(series_idx):
        mask = series_idx == s
        w = weights[mask]
        out[s] = float((abs_err[mask] * w).sum() / w.sum())
    return out


def breakdown_wmae(y_true, y_pred, holiday_grid, series_idx, store_types):
    from utils.nbeats_ensemble import compute_wmae
    holiday_mask = holiday_grid.astype(bool)
    overall = compute_wmae(y_true, y_pred, holiday_grid)
    holiday_wmae = compute_wmae(y_true[holiday_mask], y_pred[holiday_mask], holiday_grid[holiday_mask]) \
        if holiday_mask.any() else float('nan')
    non_holiday_wmae = compute_wmae(y_true[~holiday_mask], y_pred[~holiday_mask], holiday_grid[~holiday_mask])
    type_wmae = {}
    for t in ['A', 'B', 'C']:
        type_series = set(np.where(store_types == t)[0])
        mask = np.array([s in type_series for s in series_idx])
        if mask.sum() == 0:
            continue
        type_wmae[t] = compute_wmae(y_true[mask], y_pred[mask], holiday_grid[mask])
    return overall, holiday_wmae, non_holiday_wmae, type_wmae


def compute_holiday_calibration(members, sales, is_holiday, store_types, local_train_end_idx):
    """Multiplicative holiday correction per Store Type: correction =
    mean(actual)/mean(predicted) over holiday weeks the ensemble members can
    all see *within local_train only* (no lookahead into local_test). This is
    a post-hoc bias correction applied to holiday-week forecasts at inference
    -- calendar info (`is_holiday`) is already used for evaluation/loss
    weighting elsewhere in this pipeline, so using it here doesn't add any
    new input to the network; the model itself stays purely univariate.
    Returns {'A': factor, 'B': factor, 'C': factor} (1.0 = no correction,
    e.g. if a type had no usable holiday occurrences)."""
    max_lookback = max(m['lookback'] for m in members)
    holiday_idxs = [i for i in range(max_lookback, local_train_end_idx + 1) if is_holiday[i]]

    sums_actual = {t: 0.0 for t in ['A', 'B', 'C']}
    sums_pred = {t: 0.0 for t in ['A', 'B', 'C']}
    counts = {t: 0 for t in ['A', 'B', 'C']}

    for h in holiday_idxs:
        anchor = h - 1
        member_windows = []
        common_idx = None
        for m in members:
            X_eval, series_idx = build_eval_window(sales, m['lookback'], anchor)
            member_windows.append((X_eval, series_idx))
            common_idx = series_idx if common_idx is None else np.intersect1d(common_idx, series_idx)
        if common_idx is None or len(common_idx) == 0:
            continue

        y_true_h = sales[common_idx, h]
        valid = ~np.isnan(y_true_h)
        if valid.sum() == 0:
            continue
        common_idx = common_idx[valid]
        y_true_h = y_true_h[valid]

        preds = []
        for m, (X_eval, series_idx) in zip(members, member_windows):
            pos = np.searchsorted(series_idx, common_idx)
            member_forecast = forecast_series(m['model'], X_eval[pos])
            preds.append(member_forecast[:, 0])  # position 0 of the forecast = this holiday week
        y_pred_h = np.mean(preds, axis=0)

        for t in ['A', 'B', 'C']:
            type_series = set(np.where(store_types == t)[0])
            tmask = np.array([s in type_series for s in common_idx])
            if tmask.sum() == 0:
                continue
            sums_actual[t] += float(y_true_h[tmask].sum())
            sums_pred[t] += float(y_pred_h[tmask].sum())
            counts[t] += int(tmask.sum())

    correction = {}
    for t in ['A', 'B', 'C']:
        correction[t] = sums_actual[t] / sums_pred[t] if counts[t] > 0 and sums_pred[t] > 0 else 1.0
    return correction


def apply_holiday_calibration(y_pred, holiday_grid, series_idx, store_types, correction):
    """Multiply holiday-week forecasts by their Store Type's correction
    factor; non-holiday weeks pass through unchanged."""
    y_pred_calibrated = y_pred.copy()
    holiday_mask = holiday_grid.astype(bool)
    for t, factor in correction.items():
        type_series = set(np.where(store_types == t)[0])
        type_mask = np.array([s in type_series for s in series_idx])
        combined_mask = holiday_mask & type_mask[:, None]
        y_pred_calibrated[combined_mask] = y_pred_calibrated[combined_mask] * factor
    return y_pred_calibrated


def loss_curve_and_epochs(cfg, sales, is_holiday, cv_folds, max_epochs=60, patience=8):
    """Re-runs CV for `cfg` on just the largest (last) fold to get a genuine
    train-loss/val-WMAE curve (Section 8 plot 9) and a best-epoch count to
    fit the real final model for (via train_fixed_epochs) -- cheap (one
    fold) and mirrors the LightGBM notebook's best_n_estimators convention
    instead of reserving a redundant validation slice from the final fit."""
    result = run_cv_for_config(cfg, sales, is_holiday, cv_folds[-1:], max_epochs=max_epochs,
                                patience=patience, min_folds=1)
    if result is None:
        return {'train_loss': [], 'val_wmae': []}, max_epochs // 2
    fd = result['fold_details'][0]
    return fd['history'], max(1, int(round(fd['best_epoch'])))


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    dagshub.init(repo_owner='tgela23', repo_name='ml-final-project', mlflow=True)
    mlflow.set_experiment('NBEATS_Training')

    train = pd.read_csv(DATA_DIR + 'train.csv', parse_dates=['Date'])
    stores = pd.read_csv(DATA_DIR + 'stores.csv')
    sales, store_ids, dept_ids, store_types, all_dates, is_holiday = build_panel(train, stores)
    n_dates = sales.shape[1]
    local_train_end_idx = local_split_idx(n_dates)
    cv_folds = make_cv_folds(n_dates)

    study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE)
    n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    print(f'Study has {len(study.trials)} trials, {n_complete} completed.')
    if n_complete == 0:
        print('No completed trials yet -- run run_nbeats_hpo.py first (or longer).')
        return

    # 'generic' and 'interpretable' are the two configs the task asks to train
    # independently; 'mixed' is an extra searched variant, not counted as
    # either, so it never masquerades as "the" generic or interpretable winner.
    best_generic_trial = best_trial_for_architectures(study, ['generic'])
    best_interp_trial = best_trial_for_architectures(study, ['interpretable'])
    best_mixed_trial = best_trial_for_architectures(study, ['mixed'])

    results = {}

    # ---- Best generic ----
    if best_generic_trial is not None:
        cfg = trial_config(best_generic_trial)
        print(f'\nBest generic/mixed config (trial {best_generic_trial.number}, '
              f'CV mean_wmae={best_generic_trial.value:.2f}): {cfg}')
        with mlflow.start_run(run_name='NBEATS_Best_Generic'):
            mlflow.log_params(cfg)
            mlflow.log_metric('cv_mean_wmae', best_generic_trial.value)

            curve_hist, n_epochs = loss_curve_and_epochs(cfg, sales, is_holiday, cv_folds)
            mlflow.log_metric('final_fit_n_epochs', n_epochs)
            for ep, (tl, vw) in enumerate(zip(curve_hist['train_loss'], curve_hist['val_wmae']), start=1):
                mlflow.log_metric('train_loss', tl, step=ep)
                mlflow.log_metric('val_wmae', vw, step=ep)

            model_g, hist_g, lb_g = train_final_member(
                cfg, sales, is_holiday, local_train_end_idx, n_epochs)

            y_true, y_pred, hgrid, sidx, block, step = rolling_block_forecast(
                model_g, lb_g, sales, is_holiday, local_train_end_idx, HORIZON, n_blocks=4)
            overall, hol, nonhol, by_type = breakdown_wmae(y_true, y_pred, hgrid, sidx, store_types)
            mlflow.log_metric('local_test_wmae_overall', overall)
            mlflow.log_metric('local_test_wmae_holiday', hol)
            mlflow.log_metric('local_test_wmae_non_holiday', nonhol)
            for t, v in by_type.items():
                mlflow.log_metric(f'local_test_wmae_type_{t}', v)

            P.plot_loss_curves(curve_hist, 'Best generic — train/val curves', f'{PLOTS_DIR}/nbeats_loss_curves_generic.png')
            mlflow.log_artifact(f'{PLOTS_DIR}/nbeats_loss_curves_generic.png')
            torch.save(model_g.state_dict(), f'{MODELS_DIR}/nbeats_best_generic.pt')
            mlflow.log_artifact(f'{MODELS_DIR}/nbeats_best_generic.pt')

            print(f'  local_test WMAE: overall={overall:.2f} holiday={hol:.2f} non_holiday={nonhol:.2f} by_type={by_type}')
            results['best_generic'] = dict(
                trial=best_generic_trial.number, config=cfg, cv_mean_wmae=best_generic_trial.value,
                local_test_wmae_overall=overall, local_test_wmae_holiday=hol,
                local_test_wmae_non_holiday=nonhol, local_test_wmae_by_type=by_type,
            )

    # ---- Best interpretable ----
    if best_interp_trial is not None:
        cfg = trial_config(best_interp_trial)
        print(f'\nBest interpretable config (trial {best_interp_trial.number}, '
              f'CV mean_wmae={best_interp_trial.value:.2f}): {cfg}')
        with mlflow.start_run(run_name='NBEATS_Best_Interpretable'):
            mlflow.log_params(cfg)
            mlflow.log_metric('cv_mean_wmae', best_interp_trial.value)

            curve_hist_i, n_epochs_i = loss_curve_and_epochs(cfg, sales, is_holiday, cv_folds)
            mlflow.log_metric('final_fit_n_epochs', n_epochs_i)
            for ep, (tl, vw) in enumerate(zip(curve_hist_i['train_loss'], curve_hist_i['val_wmae']), start=1):
                mlflow.log_metric('train_loss', tl, step=ep)
                mlflow.log_metric('val_wmae', vw, step=ep)

            model_i, hist_i, lb_i = train_final_member(
                cfg, sales, is_holiday, local_train_end_idx, n_epochs_i)

            y_true, y_pred, hgrid, sidx, block, step = rolling_block_forecast(
                model_i, lb_i, sales, is_holiday, local_train_end_idx, HORIZON, n_blocks=4)
            overall, hol, nonhol, by_type = breakdown_wmae(y_true, y_pred, hgrid, sidx, store_types)
            mlflow.log_metric('local_test_wmae_overall', overall)
            mlflow.log_metric('local_test_wmae_holiday', hol)
            mlflow.log_metric('local_test_wmae_non_holiday', nonhol)
            for t, v in by_type.items():
                mlflow.log_metric(f'local_test_wmae_type_{t}', v)

            P.plot_loss_curves(curve_hist_i, 'Best interpretable — train/val curves', f'{PLOTS_DIR}/nbeats_loss_curves_interpretable.png')
            mlflow.log_artifact(f'{PLOTS_DIR}/nbeats_loss_curves_interpretable.png')

            def series_label(s):
                return f'Store {store_ids[s]} Dept {dept_ids[s]}'

            Xe, series_idx_i = build_eval_window(sales, lb_i, local_train_end_idx)
            P.plot_decomposition(model_i, Xe[0], all_dates, series_label(series_idx_i[0]),
                                  f'{PLOTS_DIR}/nbeats_decomposition.png')
            P.plot_backcast_reconstruction(model_i, Xe[0], series_label(series_idx_i[0]),
                                            f'{PLOTS_DIR}/nbeats_backcast_reconstruction.png')
            mlflow.log_artifact(f'{PLOTS_DIR}/nbeats_decomposition.png')
            mlflow.log_artifact(f'{PLOTS_DIR}/nbeats_backcast_reconstruction.png')

            torch.save(model_i.state_dict(), f'{MODELS_DIR}/nbeats_best_interpretable.pt')
            mlflow.log_artifact(f'{MODELS_DIR}/nbeats_best_interpretable.pt')

            print(f'  local_test WMAE: overall={overall:.2f} holiday={hol:.2f} non_holiday={nonhol:.2f} by_type={by_type}')
            results['best_interpretable'] = dict(
                trial=best_interp_trial.number, config=cfg, cv_mean_wmae=best_interp_trial.value,
                local_test_wmae_overall=overall, local_test_wmae_holiday=hol,
                local_test_wmae_non_holiday=nonhol, local_test_wmae_by_type=by_type,
            )

    if best_mixed_trial is not None:
        print(f'\n(Informational) Best mixed-architecture trial {best_mixed_trial.number}: '
              f'CV mean_wmae={best_mixed_trial.value:.2f}, params={trial_config(best_mixed_trial)} '
              f'-- logged as a searched variant, not one of the two required configs.')
        results['best_mixed_informational'] = dict(
            trial=best_mixed_trial.number, config=trial_config(best_mixed_trial),
            cv_mean_wmae=best_mixed_trial.value,
        )

    # ---- Ensemble (evaluable tier: 2x-6x H, fit on local_train, scored on local_test) ----
    base_configs = [c for c in [
        trial_config(best_generic_trial) if best_generic_trial else None,
        trial_config(best_interp_trial) if best_interp_trial else None,
    ] if c is not None]

    if base_configs:
        print(f'\nTraining evaluable ensemble: multipliers {EVALUABLE_MULTIPLIERS} x {len(base_configs)} base config(s)')
        with mlflow.start_run(run_name='NBEATS_Ensemble_Evaluable'):
            mlflow.log_param('multipliers', EVALUABLE_MULTIPLIERS)
            mlflow.log_param('n_base_configs', len(base_configs))
            members = []
            for base_cfg in base_configs:
                # One epoch-count estimate per base architecture, reused across
                # its lookback-multiple variants (a fresh single-fold CV re-run
                # per multiplier would be far more compute for a marginal gain).
                _, base_n_epochs = loss_curve_and_epochs(base_cfg, sales, is_holiday, cv_folds)
                for mult in EVALUABLE_MULTIPLIERS:
                    cfg = dict(base_cfg, lookback_multiplier=mult)
                    with mlflow.start_run(run_name=f'member_{base_cfg["architecture"]}_{mult}x', nested=True):
                        mlflow.log_params(cfg)
                        mlflow.log_metric('n_epochs', base_n_epochs)
                        model_m, hist_m, lb_m = train_final_member(
                            cfg, sales, is_holiday, local_train_end_idx, base_n_epochs)
                        print(f'  member {base_cfg["architecture"]} x{mult}: trained {len(hist_m["train_loss"])} epochs, '
                              f'final train_loss={hist_m["train_loss"][-1]:.2f}')
                        members.append({'model': model_m, 'lookback': lb_m, 'config': cfg})

            y_true, y_pred, hgrid, sidx, block, step = ensemble_rolling_forecast(
                members, sales, is_holiday, local_train_end_idx, HORIZON, n_blocks=4)
            overall, hol, nonhol, by_type = breakdown_wmae(y_true, y_pred, hgrid, sidx, store_types)
            mlflow.log_metric('local_test_wmae_overall', overall)
            mlflow.log_metric('local_test_wmae_holiday', hol)
            mlflow.log_metric('local_test_wmae_non_holiday', nonhol)
            for t, v in by_type.items():
                mlflow.log_metric(f'local_test_wmae_type_{t}', v)
            print(f'  ENSEMBLE local_test WMAE: overall={overall:.2f} holiday={hol:.2f} non_holiday={nonhol:.2f} by_type={by_type}')

            # ---- Post-hoc holiday calibration (bias correction, not a model input) ----
            # Derived only from local_train's own historical holiday occurrences,
            # applied only to holiday-week forecasts -- report both the raw and
            # calibrated holiday WMAE so it's clear how much of the gap this closes.
            holiday_correction = compute_holiday_calibration(members, sales, is_holiday, store_types, local_train_end_idx)
            y_pred_cal = apply_holiday_calibration(y_pred, hgrid, sidx, store_types, holiday_correction)
            overall_cal, hol_cal, nonhol_cal, by_type_cal = breakdown_wmae(y_true, y_pred_cal, hgrid, sidx, store_types)
            mlflow.log_params({f'holiday_correction_type_{t}': v for t, v in holiday_correction.items()})
            mlflow.log_metric('local_test_wmae_overall_calibrated', overall_cal)
            mlflow.log_metric('local_test_wmae_holiday_calibrated', hol_cal)
            mlflow.log_metric('local_test_wmae_non_holiday_calibrated', nonhol_cal)
            for t, v in by_type_cal.items():
                mlflow.log_metric(f'local_test_wmae_type_{t}_calibrated', v)
            print(f'  holiday correction factors: {holiday_correction}')
            print(f'  ENSEMBLE local_test WMAE (calibrated): overall={overall_cal:.2f} holiday={hol_cal:.2f} '
                  f'non_holiday={nonhol_cal:.2f} by_type={by_type_cal}')

            # Full plot suite from the ensemble's rolling forecast (the final reportable numbers).
            abs_err = np.abs(y_true - y_pred)
            wmae_by_series = per_series_wmae(y_true, y_pred, hgrid, sidx)
            series_wmae_arr = np.array(list(wmae_by_series.values()))
            series_keys_arr = np.array(list(wmae_by_series.keys()))
            order = np.argsort(series_wmae_arr)
            best_s, med_s, worst_s = series_keys_arr[order[0]], series_keys_arr[order[len(order) // 2]], series_keys_arr[order[-1]]

            def series_label(s):
                return f'Store {store_ids[s]} Dept {dept_ids[s]}'

            examples = []
            for label, s in [('Best', best_s), ('Median', med_s), ('Worst', worst_s)]:
                mask = sidx == s
                examples.append((f'{label}: {series_label(s)}', y_true[mask].flatten(), y_pred[mask].flatten(), wmae_by_series[s]))
            P.plot_forecast_vs_actual(examples, f'{PLOTS_DIR}/nbeats_forecast_vs_actual.png')
            P.plot_error_by_horizon(step, abs_err, f'{PLOTS_DIR}/nbeats_error_by_horizon.png')
            P.plot_wmae_distribution(series_wmae_arr, f'{PLOTS_DIR}/nbeats_wmae_distribution.png')
            P.plot_wmae_breakdown(hol, nonhol, by_type, f'{PLOTS_DIR}/nbeats_wmae_breakdown.png')

            row_dates = np.array([
                all_dates[local_train_end_idx + 1 + b * HORIZON: local_train_end_idx + 1 + b * HORIZON + HORIZON]
                for b in block
            ])
            resid = y_true - y_pred
            mean_resid_by_date = pd.Series(resid.flatten(), index=row_dates.flatten()).groupby(level=0).mean().sort_index()
            P.plot_residual_diagnostics(mean_resid_by_date.values, mean_resid_by_date.index, f'{PLOTS_DIR}/nbeats_residuals')

            for fname in ['nbeats_forecast_vs_actual.png', 'nbeats_error_by_horizon.png',
                          'nbeats_wmae_distribution.png', 'nbeats_wmae_breakdown.png',
                          'nbeats_residuals_timeseries.png', 'nbeats_residuals_hist.png', 'nbeats_residuals_acf.png']:
                mlflow.log_artifact(f'{PLOTS_DIR}/{fname}')

            results['ensemble_evaluable'] = dict(
                multipliers=EVALUABLE_MULTIPLIERS, n_members=len(members),
                local_test_wmae_overall=overall, local_test_wmae_holiday=hol,
                local_test_wmae_non_holiday=nonhol, local_test_wmae_by_type=by_type,
                holiday_calibration=dict(
                    correction_factors=holiday_correction,
                    local_test_wmae_overall=overall_cal, local_test_wmae_holiday=hol_cal,
                    local_test_wmae_non_holiday=nonhol_cal, local_test_wmae_by_type=by_type_cal,
                ),
            )

        # ---- Production tier: 2x-7x H, fit on FULL train.csv history (143 weeks) ----
        print(f'\nTraining production ensemble: multipliers {PRODUCTION_MULTIPLIERS} (full history, no held-out eval)')
        with mlflow.start_run(run_name='NBEATS_Production_Final'):
            mlflow.log_param('multipliers', PRODUCTION_MULTIPLIERS)
            prod_cfg = base_configs[0]
            _, prod_n_epochs = loss_curve_and_epochs(prod_cfg, sales, is_holiday, cv_folds)
            mlflow.log_metric('n_epochs', prod_n_epochs)
            for mult in PRODUCTION_MULTIPLIERS:
                cfg = dict(prod_cfg, lookback_multiplier=mult)
                with mlflow.start_run(run_name=f'prod_member_{mult}x', nested=True):
                    mlflow.log_params(cfg)
                    # Trained on the FULL train.csv history (local_train + local_test,
                    # 143 weeks) -- this is the only tier where 7x H has enough
                    # training weeks (>=104) to be trainable at all (see
                    # utils/nbeats_ensemble.py docstring). No WMAE is reportable
                    # here since Kaggle's real test.csv has no labels; the
                    # evaluable-tier ensemble above is the reported proxy.
                    model_p, hist_p, lb_p = train_final_member(
                        cfg, sales, is_holiday, n_dates - 1, prod_n_epochs)
                    torch.save(model_p.state_dict(), f'{MODELS_DIR}/nbeats_production_{mult}x.pt')
                    mlflow.log_artifact(f'{MODELS_DIR}/nbeats_production_{mult}x.pt')
                    print(f'  production member x{mult}: trained {len(hist_p["train_loss"])} epochs, '
                          f'final train_loss={hist_p["train_loss"][-1]:.2f}')
            results['production_ensemble'] = dict(multipliers=PRODUCTION_MULTIPLIERS, base_config=prod_cfg)

    # ---- Optuna diagnostics ----
    try:
        P.plot_optuna_diagnostics(study, f'{PLOTS_DIR}/nbeats_optuna')
        print('saved optuna diagnostics plots')
    except Exception as e:
        print(f'optuna diagnostics plots skipped: {e}')

    with open(f'{REPORTS_DIR}/nbeats_finalize_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print('\nDone. Results written to reports/nbeats_finalize_results.json')


if __name__ == '__main__':
    main()
