"""PatchTST ("A Time Series is Worth 64 Words: Long-Term Forecasting with
Transformers", Nie et al. 2023) for the Walmart Store Sales Forecasting
task — hand-rolled like utils/dlinear.py, not library-based like
utils/tft.py. The architecture is simple enough to validate correctly by
hand (patching + a plain Transformer encoder + a linear head, no LSTM, no
gating/variable-selection networks, no static-covariate machinery), unlike
TFT.

Departures from the original paper, both carried over from this project's
DLinear notebook for the same reasons:

- Shared weights across every series (channel-independent, same
  "individual=False"-style philosophy as DLinear/TFT) rather than one model
  per series.
- The horizon's future IsHoliday flag + week-of-year sin/cos are
  concatenated into the flattened patch representation before the final
  linear head. The original paper is purely univariate with no covariate
  mechanism at all — but this project's DLinear notebook found a genuine,
  visible demand spike (Easter) with zero calendar signal when the model
  had no access to anything beyond raw history, and adding a future-known
  covariate channel fixed a real chunk of that gap. No reason to repeat the
  same blind spot here.

What IS kept faithful to the paper (unlike the two changes above): RevIN-
style instance normalization — each window is normalized by *its own*
lookback mean/std (not a fixed per-series baseline like
utils.dlinear.compute_series_stats), and the model denormalizes its own
output using those same instance stats. This is a real, validated part of
why PatchTST handles distribution shift well and is kept as designed,
not swapped out for this project's per-series convention.

Reuses utils.dlinear's generic (not DLinear-specific) data-prep helpers —
build_full_calendar_panel, series_arrays_from_panel, week_of_year_sin_cos,
build_aux_features — rather than duplicating them.

Requires torch. Not imported by any other notebook — only
model_experiment_PatchTST.ipynb uses this module.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from utils.dlinear import (
    build_full_calendar_panel, series_arrays_from_panel, build_aux_features,
)

__all__ = [
    'build_full_calendar_panel', 'series_arrays_from_panel', 'build_aux_features',
    'PatchTST', 'PatchTSTWindowDataset', 'weighted_mae_loss', 'recursive_forecast',
    'PatchTSTForecastPipeline',
]


class PatchTST(nn.Module):
    """RevIN instance-normalized, channel-independent PatchTST.

    forward(hist, aux):
        hist: (batch, lookback) — RAW (un-normalized) sales history. RevIN
              stats are computed from this tensor per-sample inside forward,
              not supplied by the caller.
        aux: (batch, horizon * 3) or None — future IsHoliday + week-sin +
             week-cos for the block being predicted (see
             utils.dlinear.build_aux_features for the fixed concatenation
             order), concatenated onto the flattened patch representation.
    Returns (batch, horizon) — RAW-scale sales forecast (denormalized
    internally via the same per-sample RevIN stats used going in).
    """

    def __init__(self, lookback, horizon, patch_len=13, stride=13, d_model=32,
                 n_heads=4, n_layers=2, d_ff=64, dropout=0.1, n_aux_channels=0):
        super().__init__()
        assert (lookback - patch_len) % stride == 0, \
            'lookback must be reachable by an integer number of strided patches'
        self.lookback = lookback
        self.horizon = horizon
        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = (lookback - patch_len) // stride + 1

        self.patch_embed = nn.Linear(patch_len, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        nn.init.normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout,
            activation='gelu', batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.dropout = nn.Dropout(dropout)
        head_in = self.num_patches * d_model + n_aux_channels
        self.head = nn.Linear(head_in, horizon)

    def _patch(self, x):
        # x: (batch, lookback) -> (batch, num_patches, patch_len)
        return x.unfold(dimension=1, size=self.patch_len, step=self.stride)

    def forward(self, hist, aux=None):
        mean = hist.mean(dim=1, keepdim=True)
        std = hist.std(dim=1, keepdim=True).clamp_min(1e-5)
        hist_norm = (hist - mean) / std

        patches = self._patch(hist_norm)  # (batch, num_patches, patch_len)
        x = self.patch_embed(patches) + self.pos_embed  # (batch, num_patches, d_model)
        x = self.encoder(x)
        x = self.dropout(x)
        x = x.reshape(x.size(0), -1)  # (batch, num_patches * d_model)
        if aux is not None:
            x = torch.cat([x, aux], dim=1)
        out_norm = self.head(x)  # (batch, horizon)

        return out_norm * std + mean


class PatchTSTWindowDataset(Dataset):
    """Sliding (lookback, horizon) windows across every series in
    series_arrays, RAW (un-normalized) — RevIN normalization happens inside
    PatchTST.forward, not here, since it needs to be computed per-window
    from that window's own lookback slice. Every series contributes windows
    (mostly-zero for sparse/short ones after build_full_calendar_panel's
    zero-filling) — no series is excluded and no top-N/sampling is applied,
    same convention as utils.dlinear.SeriesWindowDataset.
    """

    def __init__(self, series_arrays, lookback, horizon):
        self.lookback = lookback
        self.horizon = horizon
        self.samples = []
        for key, (sales, holiday, dates) in series_arrays.items():
            n = len(sales)
            for start in range(0, n - lookback - horizon + 1):
                hist = sales[start:start + lookback]
                fut_holiday = holiday[start + lookback: start + lookback + horizon]
                fut_dates = dates[start + lookback: start + lookback + horizon]
                aux = build_aux_features(fut_holiday, fut_dates)
                target = sales[start + lookback: start + lookback + horizon]
                self.samples.append((hist, aux, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        hist, aux, target = self.samples[idx]
        return (
            torch.from_numpy(hist.copy()).float(),
            torch.from_numpy(aux.copy()).float(),
            torch.from_numpy(target.copy()).float(),
        )


def weighted_mae_loss(pred, target, is_holiday, holiday_weight=5.0):
    """Same WMAE-aligned weighted training loss as utils.dlinear — holiday
    timesteps count 5x, matching the competition metric directly."""
    weights = torch.where(is_holiday > 0.5, holiday_weight, 1.0)
    return (weights * (pred - target).abs()).sum() / weights.sum()


def recursive_forecast(model, hist_raw, future_holiday, future_dates, lookback, horizon, n_blocks, device='cpu'):
    """Chain n_blocks direct-horizon predictions to cover n_blocks * horizon
    future weeks, feeding each block's own predictions back in as history —
    same block-level-autoregressive pattern as utils.dlinear.
    recursive_forecast. hist_raw is RAW scale (no pre-normalization needed;
    PatchTST.forward computes its own RevIN stats from whatever raw window
    it's given, including a buffer partially filled with earlier blocks'
    own predictions).
    """
    model.eval()
    buffer = list(hist_raw[-lookback:])
    preds = []
    with torch.no_grad():
        for b in range(n_blocks):
            hist_t = torch.tensor(buffer[-lookback:], dtype=torch.float32, device=device).view(1, lookback)
            block_holiday = future_holiday[b * horizon:(b + 1) * horizon]
            block_dates = future_dates[b * horizon:(b + 1) * horizon]
            aux = build_aux_features(block_holiday, block_dates)
            aux_t = torch.tensor(aux, dtype=torch.float32, device=device).view(1, -1)
            block_pred = model(hist_t, aux_t).squeeze(0).cpu().numpy()
            preds.extend(block_pred.tolist())
            buffer.extend(block_pred.tolist())
    return np.array(preds[: horizon * n_blocks])


class PatchTSTForecastPipeline:
    """Raw-input inference wrapper, same contract as
    DLinearForecastPipeline: fit() stores every series' full history;
    predict() takes bare Store/Dept/Date/IsHoliday rows and returns
    Weekly_Sales predictions, handling the (possibly recursive) block
    rollout internally. No per-series stats needed here (unlike
    DLinearForecastPipeline) since RevIN computes its own normalization
    per-window inside the model.

    Series present in the prediction request but absent from the fitted
    history get NaN predictions (documented limitation, not silently
    zero-filled), same contract as DLinearForecastPipeline/
    TFTForecastPipeline.
    """

    def __init__(self, model, lookback, horizon, device='cpu'):
        self.model = model
        self.lookback = lookback
        self.horizon = horizon
        self.device = device

    def fit(self, train_df):
        panel = build_full_calendar_panel(train_df)
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

            sales, _holiday, _dates = self.history_[key]
            n_weeks_needed = int(round((req_dates.max() - self.last_date_) / pd.Timedelta(weeks=1)))
            n_blocks = int(np.ceil(n_weeks_needed / self.horizon))
            future_calendar = pd.date_range(
                self.last_date_ + pd.Timedelta(weeks=1), periods=n_blocks * self.horizon, freq='7D',
            )
            future_holiday = np.array(
                [bool(holiday_lookup.get((key[0], key[1], d), False)) for d in future_calendar],
                dtype=np.float32,
            )

            hist_raw = sales[-self.lookback:]
            preds_raw = recursive_forecast(
                self.model, hist_raw, future_holiday, future_calendar.to_numpy(),
                self.lookback, self.horizon, n_blocks, self.device,
            )
            preds_raw = np.clip(preds_raw, 0, None)
            for d, p in zip(future_calendar, preds_raw):
                if d in req_dates:
                    pred_rows.append((key[0], key[1], d, float(p)))

        pred_df = pd.DataFrame(pred_rows, columns=['Store', 'Dept', 'Date', 'Weekly_Sales'])
        merged = raw_df[['Store', 'Dept', 'Date']].merge(pred_df, on=['Store', 'Dept', 'Date'], how='left')
        return merged['Weekly_Sales'].to_numpy()
