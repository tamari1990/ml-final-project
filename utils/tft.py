"""Temporal Fusion Transformer (Lim et al., 2021) for the Walmart Store Sales
Forecasting task, built on `pytorch_forecasting` rather than hand-rolled —
unlike `utils/dlinear.py`, TFT has enough moving parts (variable selection
networks, gated residual networks, an LSTM encoder/decoder, interpretable
multi-head attention, quantile output heads) that reimplementing it
correctly without a GPU to iterate against would be high-risk for low
payoff. This module wraps `pytorch_forecasting`'s `TimeSeriesDataSet` and
`TemporalFusionTransformer` with the data-prep and raw-input-inference
conventions this project already established for the other three notebooks.

Unlike DLinear (deliberately minimal: own history + IsHoliday only), this
notebook uses `features.csv`/`stores.csv` too — TFT's variable-selection
network exists specifically to learn which covariates matter and downweight
the rest, so feeding it only 2 inputs would waste the architecture.

Requires torch, pytorch_forecasting, lightning. Not a dependency of
utils/feature_engineering.py, utils/metrics.py, or utils/dlinear.py, and not
imported by any other notebook — only model_experiment_TFT.ipynb (run on
Colab with GPU) uses this module.
"""

import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer

MARKDOWN_COLS = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']

STATIC_CATEGORICALS = ['Store', 'Dept', 'Type']
STATIC_REALS = ['Size']
TIME_VARYING_KNOWN_REALS = (
    ['time_idx', 'IsHoliday', 'Month', 'WeekOfYear', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
    + MARKDOWN_COLS
)
TIME_VARYING_UNKNOWN_REALS = ['Weekly_Sales']


def build_tft_panel(df, features, stores):
    """Merge df (Store/Dept/Date[/Weekly_Sales/IsHoliday]) with features.csv/
    stores.csv, reindex every series onto the full weekly calendar spanning
    df's own date range (same gap policy as utils.dlinear.
    build_full_calendar_panel: missing weeks get Weekly_Sales=0, "no sales
    recorded" rather than NaN — a department not existing yet reads the same
    as a department selling nothing, a documented simplification), and add
    the calendar/index columns TimeSeriesDataSet needs: `time_idx` (a
    sequential per-series integer — required, not the raw Date), `Month`/
    `WeekOfYear`, and `sample_weight` (5x on holiday weeks, matching every
    other notebook's WMAE-aligned weighting).

    features.csv covers 2010-02-05 -> 2013-07-26 (past even Kaggle's
    test.csv), so Temperature/Fuel_Price/CPI/Unemployment/MarkDowns are
    legitimately "known" covariates for any date in this project, not just
    observed history — same treatment the tree notebooks gave them.
    MarkDowns are ~50-64% NaN (the promotion program didn't start until
    2011-11-11) and filled with 0 here, same "not running" interpretation
    used across this project rather than imputing a value.
    """
    full_dates = pd.date_range(df['Date'].min(), df['Date'].max(), freq='7D')
    pairs = df[['Store', 'Dept']].drop_duplicates()
    grid = pairs.merge(pd.DataFrame({'Date': full_dates}), how='cross')
    cols = ['Store', 'Dept', 'Date', 'Weekly_Sales'] + (['IsHoliday'] if 'IsHoliday' in df.columns else [])
    full = grid.merge(df[cols], on=['Store', 'Dept', 'Date'], how='left')
    full['Weekly_Sales'] = full['Weekly_Sales'].fillna(0.0)
    if 'IsHoliday' in full.columns:
        full['IsHoliday'] = full['IsHoliday'].where(full['IsHoliday'].notna(), False).astype(bool)
    else:
        full['IsHoliday'] = False

    full = full.merge(stores, on='Store', how='left')
    full = full.merge(features.drop(columns=['IsHoliday']), on=['Store', 'Date'], how='left')
    full[MARKDOWN_COLS] = full[MARKDOWN_COLS].fillna(0.0)
    for col in ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment']:
        full[col] = full.groupby('Store')[col].transform(lambda s: s.ffill().bfill())

    full['IsHoliday'] = full['IsHoliday'].astype(float)
    full['Month'] = full['Date'].dt.month.astype(float)
    full['WeekOfYear'] = full['Date'].dt.isocalendar().week.astype(float)
    full['sample_weight'] = np.where(full['IsHoliday'] > 0.5, 5.0, 1.0)

    full = full.sort_values(['Store', 'Dept', 'Date']).reset_index(drop=True)
    full['time_idx'] = full.groupby(['Store', 'Dept']).cumcount()

    full['Store'] = full['Store'].astype(str).astype('category')
    full['Dept'] = full['Dept'].astype(str).astype('category')
    full['Type'] = full['Type'].astype('category')
    full['Size'] = full['Size'].astype(float)

    return full


def make_training_dataset(panel, max_encoder_length, max_prediction_length, min_prediction_idx=None):
    """TimeSeriesDataSet over panel's rows with time_idx <=
    (panel's max time_idx - max_prediction_length) — i.e. every window whose
    encoder+decoder both fit inside panel, the same "training portion" idea
    as utils.dlinear.SeriesWindowDataset, just expressed in
    pytorch_forecasting's own idiom instead of hand-built sliding windows.
    """
    train_cutoff = panel['time_idx'].max() - max_prediction_length
    return TimeSeriesDataSet(
        panel[panel.time_idx <= train_cutoff],
        time_idx='time_idx',
        target='Weekly_Sales',
        group_ids=['Store', 'Dept'],
        min_encoder_length=max_encoder_length // 2,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        min_prediction_idx=min_prediction_idx,
        static_categoricals=STATIC_CATEGORICALS,
        static_reals=STATIC_REALS,
        time_varying_known_reals=TIME_VARYING_KNOWN_REALS,
        time_varying_unknown_reals=TIME_VARYING_UNKNOWN_REALS,
        target_normalizer=GroupNormalizer(groups=['Store', 'Dept']),
        weight='sample_weight',
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=False,  # panel is already gap-free by construction
    )


class TFTForecastPipeline:
    """Raw-input inference wrapper, the TFT analogue of
    DLinearForecastPipeline/the tree notebooks' sklearn Pipelines: fit()
    stores the fitted TimeSeriesDataSet's encoding (via
    TimeSeriesDataSet.from_dataset) and the full merged/reindexed history;
    predict() takes bare Store/Dept/Date/IsHoliday rows (e.g. test.csv
    exactly as-is), merges in features.csv/stores.csv/calendar columns the
    same way build_tft_panel does, appends them after the stored history,
    and returns median (0.5 quantile) Weekly_Sales predictions — no manual
    feature work by the caller.
    """

    def __init__(self, model, max_encoder_length, max_prediction_length, features, stores, device='cpu'):
        self.model = model
        self.max_encoder_length = max_encoder_length
        self.max_prediction_length = max_prediction_length
        self.features = features
        self.stores = stores
        self.device = device

    def fit(self, train_df):
        self._raw_train_df = train_df[['Store', 'Dept', 'Date', 'Weekly_Sales', 'IsHoliday']].copy()
        self._raw_train_df['Store'] = self._raw_train_df['Store'].astype(int)
        self._raw_train_df['Dept'] = self._raw_train_df['Dept'].astype(int)
        self._raw_train_df['Date'] = pd.to_datetime(self._raw_train_df['Date'])
        self._series_ = self._raw_train_df[['Store', 'Dept']].drop_duplicates()
        self.last_date_ = self._raw_train_df['Date'].max()

        self.history_panel_ = build_tft_panel(self._raw_train_df, self.features, self.stores)
        self.training_dataset_ = make_training_dataset(
            self.history_panel_, self.max_encoder_length, self.max_prediction_length,
        )
        return self

    def predict(self, raw_df):
        """raw_df: bare Store/Dept/Date/IsHoliday rows (e.g. test.csv as-is).
        Series absent from the fitted history are returned as NaN (documented
        limitation, matching DLinearForecastPipeline's contract) rather than
        silently dropped or zero-filled."""
        req = raw_df[['Store', 'Dept', 'Date', 'IsHoliday']].copy()
        req['Store'] = req['Store'].astype(int)
        req['Dept'] = req['Dept'].astype(int)
        req['Date'] = pd.to_datetime(req['Date'])

        future_dates = pd.date_range(
            self.last_date_ + pd.Timedelta(weeks=1), periods=self.max_prediction_length, freq='7D',
        )
        future_grid = self._series_.merge(pd.DataFrame({'Date': future_dates}), how='cross')
        future_grid = future_grid.merge(req, on=['Store', 'Dept', 'Date'], how='left')
        future_grid['IsHoliday'] = future_grid['IsHoliday'].fillna(False)
        future_grid['Weekly_Sales'] = np.nan

        combined_raw = pd.concat(
            [self._raw_train_df, future_grid[['Store', 'Dept', 'Date', 'Weekly_Sales', 'IsHoliday']]],
            ignore_index=True,
        )
        future_panel = build_tft_panel(combined_raw, self.features, self.stores)

        pred_dataset = TimeSeriesDataSet.from_dataset(
            self.training_dataset_, future_panel, predict=True, stop_randomization=True,
        )
        result = self.model.predict(
            pred_dataset, mode='prediction', return_index=True, batch_size=256, num_workers=0,
        )
        preds = np.clip(result.output.cpu().numpy(), 0, None)  # (n_series, max_prediction_length)
        index = result.index  # one row per series, in the same order as preds' first dimension

        pred_rows = []
        for row_i in range(preds.shape[0]):
            store = int(index.iloc[row_i]['Store'])
            dept = int(index.iloc[row_i]['Dept'])
            for step in range(preds.shape[1]):
                d = self.last_date_ + pd.Timedelta(weeks=step + 1)
                pred_rows.append((store, dept, d, float(preds[row_i, step])))
        pred_df = pd.DataFrame(pred_rows, columns=['Store', 'Dept', 'Date', 'Weekly_Sales'])

        merged = req[['Store', 'Dept', 'Date']].merge(pred_df, on=['Store', 'Dept', 'Date'], how='left')
        return merged['Weekly_Sales'].to_numpy()
