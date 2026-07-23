"""Optuna hyperparameter search for N-BEATS (overnight-run entry point).

Scores every trial with the exact walk-forward CV harness from
model_experiment_NBEATS.ipynb / utils.nbeats_train (same 3-fold, 52/13-week
boundaries as model_experiment_LightGBM.ipynb). Each trial is logged as a
nested MLflow run under experiment 'NBEATS_Training' (DagsHub-hosted),
parent run 'NBEATS_HPO_Search'. The Optuna study is persisted to a local
sqlite file so the notebook can reload it afterward for the hyperparameter-
importance / parallel-coordinate / optimization-history plots (Step 4/8).

Usage:
    .venv/Scripts/python run_nbeats_hpo.py --n-trials 60 --max-epochs 40
"""

import argparse
import time

import dagshub
import mlflow
import numpy as np
import optuna
import pandas as pd
from optuna.integration.mlflow import MLflowCallback

from utils.nbeats_data import build_panel
from utils.nbeats_train import HORIZON, make_cv_folds, run_cv_for_config

DATA_DIR = 'data/raw/walmart-recruiting-store-sales-forecasting/'


def build_objective(sales, is_holiday, cv_folds, max_epochs, patience):
    def objective(trial):
        architecture = trial.suggest_categorical('architecture', ['generic', 'interpretable', 'mixed'])
        n_stacks = 2 if architecture == 'interpretable' else trial.suggest_int('n_stacks', 1, 3)
        n_blocks = trial.suggest_int('n_blocks', 1, 4)
        n_fc_layers = trial.suggest_int('n_fc_layers', 2, 5)
        layer_size = trial.suggest_categorical('layer_size', [128, 256, 512])
        # lookback_multiplier restricted to >=4 (>= 52 weeks = 1 full year, since
        # HORIZON=13): members with a shorter lookback structurally can't contain
        # the prior occurrence of a yearly holiday in their own window, so they're
        # guessing blind on holiday weeks and just dilute the ensemble/HPO signal.
        lookback_multiplier = trial.suggest_int('lookback_multiplier', 4, 7)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [256, 512, 1024])
        optimizer = trial.suggest_categorical('optimizer', ['adam', 'adamw'])
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
        loss = trial.suggest_categorical('loss', ['wmae', 'mae', 'smape', 'asym_wmae'])
        dropout = trial.suggest_float('dropout', 0.0, 0.3)
        share_weights = trial.suggest_categorical('share_weights', [True, False])
        trend_degree = trial.suggest_int('trend_degree', 2, 4) if architecture in ('interpretable', 'mixed') else 3
        # Oversampling boost for training windows containing a holiday week (see
        # utils.nbeats_train._make_loader) -- 0 reproduces the original uniform
        # shuffle, so HPO can also discover that no boost is best.
        holiday_boost = trial.suggest_float('holiday_boost', 0.0, 8.0)

        config = dict(
            architecture=architecture, n_stacks=n_stacks, n_blocks=n_blocks,
            n_fc_layers=n_fc_layers, layer_size=layer_size,
            lookback_multiplier=lookback_multiplier, learning_rate=learning_rate,
            batch_size=batch_size, optimizer=optimizer, weight_decay=weight_decay,
            loss=loss, dropout=dropout, share_weights=share_weights, trend_degree=trend_degree,
            holiday_boost=holiday_boost,
        )

        t0 = time.time()
        result = run_cv_for_config(config, sales, is_holiday, cv_folds,
                                    max_epochs=max_epochs, patience=patience, seed=42)
        elapsed = time.time() - t0

        if result is None:
            raise optuna.TrialPruned('insufficient valid folds for this lookback_multiplier')

        mlflow.log_params(config)
        mlflow.log_metric('n_folds_used', result['n_folds_used'])
        mlflow.log_metric('std_wmae', result['std_wmae'])
        mlflow.log_metric('mean_best_epoch', result['mean_best_epoch'])
        mlflow.log_metric('elapsed_sec', elapsed)
        # Stored as a user_attr (not just an MLflow metric) so run_nbeats_finalize.py
        # can read it straight back off the winning trial to size the final fit's
        # epoch count -- mirrors the LightGBM notebook's best_n_estimators
        # (mean best_iteration across CV folds) convention.
        trial.set_user_attr('mean_best_epoch', result['mean_best_epoch'])
        # Holiday-only CV WMAE, tracked alongside (not instead of) the primary
        # mean_wmae objective -- lets run_nbeats_finalize.py select on a combined
        # score instead of a config that wins overall but is quietly terrible on
        # holidays. NaN (as a user_attr) doesn't round-trip through sqlite/JSON
        # cleanly, so store None when no fold had a holiday week in its val range.
        mean_holiday_wmae = result['mean_holiday_wmae']
        if not np.isnan(mean_holiday_wmae):
            mlflow.log_metric('mean_holiday_wmae', mean_holiday_wmae)
            trial.set_user_attr('mean_holiday_wmae', mean_holiday_wmae)
        else:
            trial.set_user_attr('mean_holiday_wmae', None)
        for i, fd in enumerate(result['fold_details'], start=1):
            mlflow.log_metric(f'fold{i}_wmae', fd['wmae'])
            mlflow.log_metric(f'fold{i}_n_train_windows', fd['n_train_windows'])
            if not np.isnan(fd['holiday_wmae']):
                mlflow.log_metric(f'fold{i}_holiday_wmae', fd['holiday_wmae'])

        print(f"trial {trial.number}: mean_wmae={result['mean_wmae']:.2f} "
              f"mean_holiday_wmae={mean_holiday_wmae:.2f} "
              f"(folds_used={result['n_folds_used']}, {elapsed:.0f}s) {config}")
        return result['mean_wmae']
    return objective


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-trials', type=int, default=60)
    parser.add_argument('--max-epochs', type=int, default=40)
    parser.add_argument('--patience', type=int, default=6)
    parser.add_argument('--study-name', default='nbeats_hpo')
    parser.add_argument('--storage', default='sqlite:///nbeats_optuna.db')
    args = parser.parse_args()

    dagshub.init(repo_owner='tgela23', repo_name='ml-final-project', mlflow=True)
    mlflow.set_experiment('NBEATS_Training')

    train = pd.read_csv(DATA_DIR + 'train.csv', parse_dates=['Date'])
    stores = pd.read_csv(DATA_DIR + 'stores.csv')
    sales, store_ids, dept_ids, store_types, all_dates, is_holiday = build_panel(train, stores)
    cv_folds = make_cv_folds(sales.shape[1])

    print(f'Panel: {sales.shape[0]} series x {sales.shape[1]} weeks. CV folds: {cv_folds}')

    mlflc = MLflowCallback(
        tracking_uri=mlflow.get_tracking_uri(),
        metric_name='mean_wmae',
        create_experiment=False,
        mlflow_kwargs={'nested': True},
    )

    objective = mlflc.track_in_mlflow()(
        build_objective(sales, is_holiday, cv_folds, args.max_epochs, args.patience)
    )

    with mlflow.start_run(run_name='NBEATS_HPO_Search'):
        mlflow.log_param('n_trials_requested', args.n_trials)
        mlflow.log_param('max_epochs', args.max_epochs)
        mlflow.log_param('patience', args.patience)
        for i, (te, vs, ve) in enumerate(cv_folds, start=1):
            mlflow.log_param(f'fold{i}_train_len', te + 1)
            mlflow.log_param(f'fold{i}_val_range', f'[{vs},{ve}]')

        study = optuna.create_study(
            direction='minimize', study_name=args.study_name,
            storage=args.storage, load_if_exists=True,
        )
        study.optimize(objective, n_trials=args.n_trials, callbacks=[mlflc])

        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        mlflow.log_metric('n_trials_completed', len(completed))
        if completed:
            mlflow.log_metric('best_mean_wmae', study.best_value)
            for k, v in study.best_params.items():
                mlflow.log_param(f'best_{k}', v)

    print('\nDone.')
    print('Best trial:', study.best_trial.number if completed else 'none completed')
    if completed:
        print('Best value:', study.best_value)
        print('Best params:', study.best_params)


if __name__ == '__main__':
    main()
