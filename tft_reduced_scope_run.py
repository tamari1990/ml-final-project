"""TFT — reduced-scope local run (NOT the canonical tuned result).

Standalone script, not a notebook, run under real time pressure (project
due the same day) after two full attempts at model_experiment_TFT.ipynb's
proper 4-combo grid search failed to complete: local CPU execution measured
~16 min/epoch at the CV-split scale (confirmed non-viable for any real
grid), and a parallel Colab GPU run died twice to idle-session timeouts.

What this script does instead, disclosed honestly rather than hidden:
- No hyperparameter tuning, no CV. Reuses the baseline hyperparameters
  already validated on an earlier real Colab GPU run (hidden_size=16,
  attention_head_size=1, learning_rate=0.03, mean val WMAE 4052.72 across
  3 CV splits, per-split epochs [14, 9, 9]).
- N_EPOCHS_FINAL=15 is set from that same baseline run's own convergence
  behavior (median ~9-14 epochs), not guessed blind.
- One direct fit on local_train_raw (91 weeks, LOOKBACK=52,
  HORIZON_FINAL=52 via TFT's variable-length encoder -- unlike PatchTST,
  TFT doesn't need a shorter LOOKBACK_FINAL workaround here), then a
  holdout evaluation against local_test_raw -- same local-test-holdout
  WMAE convention every other notebook in this project reports.

Real, measured result from the actual run (see README/presentation for
current numbers): local-test holdout WMAE 3996.37, ~43.6 min training on
CPU. Plot saved to plots/tft_reduced_scope_actual_vs_predicted.png.
Logged to DagsHub MLflow as a distinctly-named 'TFT_Reduced_Scope_FinalFit'
run under the TFT_Training experiment, kept separate from the properly
tuned run so it's never confused with one.

Run from the repo root: python tft_reduced_scope_run.py
"""
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import lightning.pytorch as pl

if not getattr(torch.load, '_is_trusted_wrapper', False):
    _original_torch_load = torch.load
    def _torch_load_trusted(*args, **kwargs):
        kwargs['weights_only'] = False
        return _original_torch_load(*args, **kwargs)
    _torch_load_trusted._is_trusted_wrapper = True
    torch.load = _torch_load_trusted

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

from utils.tft import build_tft_panel, make_training_dataset
from utils.metrics import wmae

pd.set_option('display.max_columns', 50)
SEED = 42
pl.seed_everything(SEED)
ACCELERATOR = 'gpu' if torch.cuda.is_available() else 'cpu'
print('accelerator:', ACCELERATOR)

DATA_DIR = 'data/raw/walmart-recruiting-store-sales-forecasting/'
train = pd.read_csv(DATA_DIR + 'train.csv', parse_dates=['Date'])
features = pd.read_csv(DATA_DIR + 'features.csv', parse_dates=['Date'])
stores = pd.read_csv(DATA_DIR + 'stores.csv')
train = train.sort_values(['Store', 'Dept', 'Date']).reset_index(drop=True)

unique_dates = np.sort(train['Date'].unique())
cutoff_date = unique_dates[-52]
local_train_raw = train[train['Date'] < cutoff_date].copy()
local_test_raw = train[train['Date'] >= cutoff_date].copy()
print(f'local_train_raw: {local_train_raw.shape}, local_test_raw: {local_test_raw.shape}')

LOOKBACK = 52
HORIZON_FINAL = 52

PARAMS = dict(hidden_size=16, attention_head_size=1)
LEARNING_RATE = 0.03
N_EPOCHS_FINAL = 15
BASELINE_CV_WMAE = 4052.72  # real, from the earlier completed Colab run -- reduced-scope run has no CV number of its own

print(f'\nReduced-scope run: params={PARAMS}, learning_rate={LEARNING_RATE}, '
      f'N_EPOCHS_FINAL={N_EPOCHS_FINAL} (no CV/tuning -- time-boxed)')

eval_panel = build_tft_panel(local_train_raw, features, stores)
eval_training_ds = make_training_dataset(eval_panel, LOOKBACK, HORIZON_FINAL)
eval_train_dl = eval_training_ds.to_dataloader(train=True, batch_size=128, num_workers=0)
print(f'{len(eval_training_ds)} training windows')

holdout_model = TemporalFusionTransformer.from_dataset(
    eval_training_ds, learning_rate=LEARNING_RATE, **PARAMS, dropout=0.1,
    hidden_continuous_size=max(PARAMS['hidden_size'] // 2, 4), loss=QuantileLoss(), optimizer='adam',
)
holdout_trainer = pl.Trainer(
    max_epochs=N_EPOCHS_FINAL, accelerator=ACCELERATOR, enable_progress_bar=False, enable_model_summary=False,
    logger=False, enable_checkpointing=False,
)

t0 = time.time()
holdout_trainer.fit(holdout_model, train_dataloaders=eval_train_dl)
print(f'\nTraining took {time.time()-t0:.1f}s ({N_EPOCHS_FINAL} epochs)')

holdout_panel = build_tft_panel(pd.concat([local_train_raw, local_test_raw], ignore_index=True), features, stores)
holdout_pred_ds = TimeSeriesDataSet.from_dataset(eval_training_ds, holdout_panel, predict=True, stop_randomization=True)

holdout_raw = holdout_model.predict(holdout_pred_ds, mode='raw', return_index=True, return_x=True, batch_size=256, num_workers=0)
holdout_median = np.clip(holdout_model.predict(holdout_pred_ds, mode='prediction', batch_size=256, num_workers=0).cpu().numpy(), 0, None)
holdout_index = holdout_raw.index

pred_rows = []
for row_i in range(holdout_median.shape[0]):
    store, dept = str(int(holdout_index.iloc[row_i]['Store'])), str(int(holdout_index.iloc[row_i]['Dept']))
    start_idx = int(holdout_index.iloc[row_i]['time_idx'])
    sub = holdout_panel[(holdout_panel['Store'].astype(str) == store) & (holdout_panel['Dept'].astype(str) == dept) &
                         (holdout_panel['time_idx'] >= start_idx) & (holdout_panel['time_idx'] < start_idx + HORIZON_FINAL)]
    sub = sub.sort_values('time_idx')
    for step, (_, r) in enumerate(sub.iterrows()):
        pred_rows.append((int(store), int(dept), r['Date'], r['Weekly_Sales'], holdout_median[row_i, step], bool(r['IsHoliday'])))

pred_df = pd.DataFrame(pred_rows, columns=['Store', 'Dept', 'Date', 'Actual', 'Predicted', 'IsHoliday'])
pred_df['Residual'] = pred_df['Actual'] - pred_df['Predicted']

holdout_wmae = wmae(pred_df['Actual'], pred_df['Predicted'], pred_df['IsHoliday'])
print(f'\n=== RESULT ===')
print(f'Local-test holdout WMAE (reduced-scope: no tuning, fixed baseline params, {N_EPOCHS_FINAL} epochs): {holdout_wmae:.2f}')
print(f'(baseline CV WMAE from earlier real Colab run, for reference: {BASELINE_CV_WMAE:.2f})')

# Plot: actual vs predicted for 3 sample series, same combos as every other notebook
sample_combos = [(1, 1), (1, 72), (20, 1)]
fig, axes = plt.subplots(len(sample_combos), 1, figsize=(11, 9), sharex=True)
for ax, (store, dept) in zip(axes, sample_combos):
    sub = pred_df[(pred_df['Store'] == store) & (pred_df['Dept'] == dept)].sort_values('Date')
    ax.plot(sub['Date'], sub['Actual'], label='Actual', marker='o', markersize=3)
    ax.plot(sub['Date'], sub['Predicted'], label='Predicted', marker='x', markersize=3)
    ax.set_title(f'Store {store}, Dept {dept}')
    ax.legend()
axes[-1].set_xlabel('Date')
fig.suptitle('TFT (reduced scope, no tuning) -- actual vs. predicted, local-test holdout')
plt.tight_layout()
plt.savefig('plots/tft_reduced_scope_actual_vs_predicted.png', dpi=150, bbox_inches='tight')
print('\nplot saved to plots/tft_reduced_scope_actual_vs_predicted.png')

# Log to MLflow/DagsHub, honestly labeled as reduced-scope so it's not
# confused with a properly tuned run in the tracking history.
import dagshub
dagshub.init(repo_owner='tgela23', repo_name='ml-final-project', mlflow=True)
import mlflow
mlflow.set_experiment('TFT_Training')
with mlflow.start_run(run_name='TFT_Reduced_Scope_FinalFit'):
    mlflow.log_param('reduced_scope', True)
    mlflow.log_param('reason', 'time constraint -- no CV/grid search, baseline params reused directly')
    mlflow.log_params(PARAMS)
    mlflow.log_param('learning_rate', LEARNING_RATE)
    mlflow.log_param('n_epochs_final', N_EPOCHS_FINAL)
    mlflow.log_metric('baseline_cv_wmae_reference', BASELINE_CV_WMAE)
    mlflow.log_metric('local_test_holdout_wmae', holdout_wmae)
print('MLflow run logged: TFT_Reduced_Scope_FinalFit')
print('\nDONE')
