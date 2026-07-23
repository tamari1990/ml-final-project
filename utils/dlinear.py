"""DLinear (decomposition + linear) global forecasting model, shared across
every (Store, Dept) series, for the Walmart Store Sales Forecasting task.

Adapted from "Are Transformers Effective for Time Series Forecasting?"
(Zeng et al., 2022): a trend/seasonal moving-average decomposition followed
by one linear layer per component. Departures from the paper, both aimed at
this dataset specifically:

- Shared weights across all series (paper's "individual=False" mode), with
  each (Store, Dept) series treated as a batch sample rather than a channel.
  A single linear map generalizes across all ~3300 series instead of fitting
  one model per series or per store.
- The horizon's future `IsHoliday` flag (known in advance — it's a calendar
  fact, not something to forecast) is concatenated onto both linear layers'
  input. Thanksgiving/Christmas weeks are the hardest to predict and the
  most heavily weighted (5x) in WMAE, so giving the model direct, explicit
  access to "is this output week a holiday" — rather than hoping it infers
  yearly seasonality purely from `lag52`-equivalent history — is the single
  biggest lever available without abandoning DLinear's linear simplicity.

Requires torch. Not a dependency of utils/feature_engineering.py or
utils/metrics.py, and not imported by the LightGBM/XGBoost notebooks —
only model_experiment_DLinear.ipynb (run on Colab, where torch + GPU are
available) uses this module.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class MovingAvg(nn.Module):
    """Centered moving average along the sequence dimension, edge-padded by
    repeating the first/last timestep so the output has the same length as
    the input (odd or even kernel_size both work)."""

    def __init__(self, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):  # x: (batch, seq_len, channels)
        pad_left = (self.kernel_size - 1) // 2
        pad_right = self.kernel_size - 1 - pad_left
        front = x[:, 0:1, :].repeat(1, pad_left, 1)
        end = x[:, -1:, :].repeat(1, pad_right, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)


class SeriesDecomp(nn.Module):
    """Splits a series into trend (moving average) and seasonal (residual)."""

    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


class DLinear(nn.Module):
    """Shared-weight DLinear with an auxiliary future-holiday input.

    forward(target_hist, aux):
        target_hist: (batch, lookback, 1) — normalized sales history.
        aux: (batch, horizon) or None — future IsHoliday flags (0/1) for the
             block being predicted, known in advance.
    Returns (batch, horizon) — normalized sales forecast for the block.
    """

    def __init__(self, lookback, horizon, kernel_size=25, n_aux_channels=0):
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.decomp = SeriesDecomp(kernel_size)
        in_features = lookback + n_aux_channels
        self.Linear_Seasonal = nn.Linear(in_features, horizon)
        self.Linear_Trend = nn.Linear(in_features, horizon)

    def forward(self, target_hist, aux=None):
        seasonal, trend = self.decomp(target_hist)
        seasonal = seasonal.squeeze(-1)
        trend = trend.squeeze(-1)
        if aux is not None:
            seasonal = torch.cat([seasonal, aux], dim=1)
            trend = torch.cat([trend, aux], dim=1)
        return self.Linear_Seasonal(seasonal) + self.Linear_Trend(trend)


def build_full_calendar_panel(df):
    """Reindex every (Store, Dept) series onto the full weekly calendar
    spanning df's own date range, inserting explicit rows for any missing
    week. Missing Weekly_Sales/IsHoliday are filled with 0/False.

    Unlike utils.feature_engineering's gap-handling (which only needs a
    gap-free calendar to make row-positional lag/rolling correct, and
    leaves genuine gaps as NaN for lag features to skip), DLinear needs a
    literal numeric value at every timestep to form fixed-length windows —
    so gaps are filled with 0 (interpreted as "no sales that week"), not
    left as NaN. This is a real modeling simplification: a gap more often
    means "this department didn't exist yet at this store" than "sold
    nothing" — documented here rather than silently assumed.
    """
    full_dates = pd.date_range(df['Date'].min(), df['Date'].max(), freq='7D')
    pairs = df[['Store', 'Dept']].drop_duplicates()
    grid = pairs.merge(pd.DataFrame({'Date': full_dates}), how='cross')
    cols = ['Store', 'Dept', 'Date', 'Weekly_Sales'] + (['IsHoliday'] if 'IsHoliday' in df.columns else [])
    full = grid.merge(df[cols], on=['Store', 'Dept', 'Date'], how='left')
    full['Weekly_Sales'] = full['Weekly_Sales'].fillna(0.0)
    if 'IsHoliday' in full.columns:
        # .where(), not .fillna() — .fillna() on this object-dtype column (a mix
        # of real True/False and NaN from the left-merge) hits pandas' deprecated
        # silent-downcast-after-fillna path and raises a FutureWarning; .where()
        # doesn't share that codepath.
        full['IsHoliday'] = full['IsHoliday'].where(full['IsHoliday'].notna(), False).astype(bool)
    else:
        full['IsHoliday'] = False
    return full.sort_values(['Store', 'Dept', 'Date']).reset_index(drop=True)


def compute_series_stats(panel):
    """Per-(Store, Dept) mean/std of Weekly_Sales, for z-score normalization.

    Must be computed from a training-only panel (never one that includes the
    held-out evaluation weeks) to avoid leakage. std is floored at 1.0 for
    constant/all-zero series so normalization never divides by ~0.
    """
    stats = {}
    for key, g in panel.groupby(['Store', 'Dept']):
        sales = g.sort_values('Date')['Weekly_Sales'].to_numpy(dtype=np.float32)
        mean, std = float(sales.mean()), float(sales.std())
        stats[key] = (mean, std if std > 1e-6 else 1.0)
    return stats


def series_arrays_from_panel(panel):
    """{(Store, Dept): (sales_array, holiday_array, dates_array)}, time-ordered."""
    arrays = {}
    for key, g in panel.groupby(['Store', 'Dept']):
        g = g.sort_values('Date')
        arrays[key] = (
            g['Weekly_Sales'].to_numpy(dtype=np.float32),
            g['IsHoliday'].to_numpy(dtype=np.float32),
            g['Date'].to_numpy(),
        )
    return arrays


class SeriesWindowDataset(Dataset):
    """Sliding (lookback, horizon) windows across every series in series_arrays.

    Every series contributes floor(n_weeks - lookback - horizon + 1, 0)
    windows — including sparse/short series (which, after
    build_full_calendar_panel's zero-filling, just contribute mostly-zero
    windows rather than being dropped). No series is excluded and no
    top-N/sampling is applied: every (Store, Dept) pair in the input panel
    is represented.
    """

    def __init__(self, series_arrays, series_stats, lookback, horizon, holiday_weight=5.0):
        self.lookback = lookback
        self.horizon = horizon
        self.samples = []
        for key, (sales, holiday, _dates) in series_arrays.items():
            mean, std = series_stats[key]
            sales_norm = (sales - mean) / std
            n = len(sales_norm)
            for start in range(0, n - lookback - horizon + 1):
                hist = sales_norm[start:start + lookback]
                fut_holiday = holiday[start + lookback: start + lookback + horizon]
                target = sales_norm[start + lookback: start + lookback + horizon]
                self.samples.append((hist, fut_holiday, target))
        self.holiday_weight = holiday_weight

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        hist, fut_holiday, target = self.samples[idx]
        return (
            torch.from_numpy(hist.copy()).float().unsqueeze(-1),
            torch.from_numpy(fut_holiday.copy()).float(),
            torch.from_numpy(target.copy()).float(),
        )


def weighted_mae_loss(pred, target, is_holiday, holiday_weight=5.0):
    """Training loss on normalized values, weighted the same way as the
    competition's WMAE metric (holiday timesteps count 5x) — so gradient
    descent is optimizing toward the actual evaluation metric, not plain MAE."""
    weights = torch.where(is_holiday > 0.5, holiday_weight, 1.0)
    return (weights * (pred - target).abs()).sum() / weights.sum()


def recursive_forecast(model, hist_norm, future_holiday, lookback, horizon, n_blocks, device='cpu'):
    """Chain n_blocks direct-horizon predictions to cover n_blocks * horizon
    future weeks, feeding each block's own predictions back in as history
    for the next block (autoregressive at the block level — the model
    itself is still a single direct-multistep DLinear per block, never
    given ground truth from the evaluation period).

    hist_norm: 1D array, length >= lookback (normalized). future_holiday:
    1D array, length horizon * n_blocks (raw 0/1, known in advance).
    Returns a 1D np.array of length horizon * n_blocks, normalized scale.
    """
    model.eval()
    buffer = list(hist_norm[-lookback:])
    preds = []
    with torch.no_grad():
        for b in range(n_blocks):
            hist_t = torch.tensor(buffer[-lookback:], dtype=torch.float32, device=device).view(1, lookback, 1)
            aux_t = torch.tensor(
                future_holiday[b * horizon:(b + 1) * horizon], dtype=torch.float32, device=device
            ).view(1, horizon)
            block_pred = model(hist_t, aux_t).squeeze(0).cpu().numpy()
            preds.extend(block_pred.tolist())
            buffer.extend(block_pred.tolist())
    return np.array(preds[: horizon * n_blocks])


class DLinearForecastPipeline:
    """Raw-input inference wrapper, the DLinear analogue of the
    sklearn Pipelines the LightGBM/XGBoost notebooks save: fit() stores
    per-series history and normalization stats; predict() takes bare
    Store/Dept/Date/IsHoliday rows (e.g. test.csv exactly as-is) and returns
    Weekly_Sales predictions, handling the recursive block rollout and
    denormalization internally so the caller does no manual feature work.

    Series present in the prediction request but absent from the fitted
    history get NaN predictions (documented limitation, not silently
    zero-filled) — this DLinear was trained on every series that exists in
    the training data, but the real test.csv is not guaranteed to only
    contain those.
    """

    def __init__(self, model, lookback, horizon, device='cpu'):
        self.model = model
        self.lookback = lookback
        self.horizon = horizon
        self.device = device

    def fit(self, train_df):
        panel = build_full_calendar_panel(train_df)
        self.stats_ = compute_series_stats(panel)
        self.history_ = series_arrays_from_panel(panel)
        self.last_date_ = panel['Date'].max()
        return self

    def predict(self, raw_df):
        holiday_lookup = raw_df.set_index(['Store', 'Dept', 'Date'])['IsHoliday']
        pred_rows = []
        for key, g in raw_df.groupby(['Store', 'Dept']):
            req_dates = pd.to_datetime(g['Date'].sort_values().unique())
            if key not in self.history_ or req_dates.max() <= self.last_date_:
                for d in req_dates:
                    pred_rows.append((key[0], key[1], d, np.nan))
                continue

            mean, std = self.stats_[key]
            sales, _holiday, _dates = self.history_[key]
            n_weeks_needed = int(round((req_dates.max() - self.last_date_) / pd.Timedelta(weeks=1)))
            n_blocks = int(np.ceil(n_weeks_needed / self.horizon))
            future_calendar = pd.date_range(
                self.last_date_ + pd.Timedelta(weeks=1), periods=n_blocks * self.horizon, freq='7D'
            )
            future_holiday = np.array(
                [bool(holiday_lookup.get((key[0], key[1], d), False)) for d in future_calendar],
                dtype=np.float32,
            )

            hist_norm = (sales[-self.lookback:] - mean) / std
            preds_norm = recursive_forecast(
                self.model, hist_norm, future_holiday, self.lookback, self.horizon, n_blocks, self.device
            )
            preds_raw = np.clip(preds_norm * std + mean, 0, None)
            for d, p in zip(future_calendar, preds_raw):
                if d in req_dates:
                    pred_rows.append((key[0], key[1], d, float(p)))

        pred_df = pd.DataFrame(pred_rows, columns=['Store', 'Dept', 'Date', 'Weekly_Sales'])
        merged = raw_df[['Store', 'Dept', 'Date']].merge(pred_df, on=['Store', 'Dept', 'Date'], how='left')
        return merged['Weekly_Sales'].to_numpy()
