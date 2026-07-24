"""TimesFM 2.5 (Google Research's pretrained time-series foundation model)
for the Walmart Store Sales Forecasting task — used **zero-shot**, not
trained from scratch like every other notebook in this project. There are
no learnable weights being fit to our data at all: `google/timesfm-2.5-
200m-pytorch` was pretrained once on a large, diverse time-series corpus,
and this module just feeds it each series' own history as context and asks
for a forecast.

Departs from DLinear/TFT/PatchTST in a way that's structural, not a design
choice: there's no training loop, so the "training-window budget" problem
that constrained all three other deep-learning notebooks (LOOKBACK +
HORIZON having to fit inside local_train_raw's 91 weeks) doesn't apply
here — nothing is being trained, so there's no windows-vs-target-length
trade-off to make. Horizon is freely choosable at inference time (validated
directly: a 52-week-compiled model accepts a 52-week horizon request
without complaint, and forecast() happily produces longer horizons too via
_compiled_decode's internal chunking).

Context length is NOT freely choosable below a minimum, though — see
PATCH_LEN / min_context_length() below. This was found empirically (not
in any TimesFM doc) after CV eval silently returned WMAE=nan for short
context lengths.

No covariates (IsHoliday etc.) are used, despite TimesFM technically
supporting them via `forecast_with_covariates()` — that path requires the
`timesfm[xreg]` extra (pulls in `jax`, a second deep learning framework
alongside torch, purely for a linear covariate-regression step) and
changes the plain `forecast()` output shape in ways that need separate
handling. More importantly, avoiding covariates is true to what TimesFM is
actually being tested for here: whether a pretrained foundation model's
zero-shot forecast is competitive *without* any of the manual feature
engineering every other notebook in this project needed.

Reuses utils.dlinear's generic (not DLinear-specific) data-prep helpers —
build_full_calendar_panel, series_arrays_from_panel — rather than
duplicating them.

Requires torch, timesfm. Not imported by any other notebook — only
model_experiment_TimesFM.ipynb uses this module.
"""

import numpy as np
import pandas as pd

from utils.dlinear import build_full_calendar_panel, series_arrays_from_panel

__all__ = [
    'build_full_calendar_panel', 'series_arrays_from_panel',
    'load_timesfm', 'min_context_length', 'forecast_series',
    'TimesFMForecastPipeline', 'CHECKPOINT', 'PATCH_LEN',
]

CHECKPOINT = 'google/timesfm-2.5-200m-pytorch'

PATCH_LEN = 32  # TimesFM 2.5's internal input patch size (model.model.p) —
                # fixed by the pretrained checkpoint's architecture, not a
                # choice made in this project.


def load_timesfm(max_context, max_horizon):
    """Downloads (once; cached by huggingface_hub afterward) and compiles
    the pretrained checkpoint. Compiled once with generous max_context/
    max_horizon covering every use in this notebook (CV eval's short
    horizon and the direct 52-week final horizon both fit under one
    compile) — actual per-call context/horizon can be shorter or, for
    horizon, even longer than these compiled maximums; they're just the
    caps the compiled decode path is sized for, not hard limits.
    """
    from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch
    from timesfm.configs import ForecastConfig

    model = TimesFM_2p5_200M_torch.from_pretrained(CHECKPOINT)
    model.compile(ForecastConfig(
        max_context=max_context,
        max_horizon=max_horizon,
        normalize_inputs=True,
        use_continuous_quantile_head=True,
        infer_is_positive=True,
        fix_quantile_crossing=True,
    ))
    return model


def min_context_length(model):
    """Shortest raw context length that's safe to pass to forecast_series
    for this compiled model.

    compile() silently rounds max_context up to a multiple of PATCH_LEN
    (e.g. compile(max_context=52) actually reserves 64 internally — verified
    via model.forecast_config.max_context after compiling). forecast() then
    left-pads any shorter real context with zeros, masked as "missing", up
    to that effective max_context. If the padding alone fills the entire
    leading patch (PATCH_LEN values are ALL padding), that patch's per-patch
    normalization degenerates and the model emits an all-NaN forecast for
    the whole batch item — silently, no error raised.

    Empirically verified against the real pretrained checkpoint (not
    documented anywhere in the timesfm package): with max_context=52
    (-> effective 64), context_length<=32 -> all-NaN, context_length>=33 ->
    valid output. 33 == effective_max_context - PATCH_LEN + 1 exactly, i.e.
    the shortest context that still leaves >=1 real value in the leading
    patch.
    """
    return model.forecast_config.max_context - PATCH_LEN + 1


def forecast_series(model, contexts, horizon, batch_size=256):
    """contexts: list of 1D np.arrays (raw, un-normalized history — RevIN-
    style instance normalization happens inside TimesFM itself, same
    handling as utils.patchtst never needing a separate normalization
    step). Returns (point, quantiles) concatenated across all batches:
    point (n_series, horizon), quantiles (n_series, horizon, 10).

    Raises ValueError rather than silently returning NaN if any context is
    shorter than min_context_length(model) — see that function's docstring.
    """
    min_ctx = min_context_length(model)
    shortest = min((len(c) for c in contexts), default=None)
    if shortest is not None and shortest < min_ctx:
        raise ValueError(
            f'context length {shortest} is below the safe minimum of {min_ctx} '
            f'for this compiled model (max_context={model.forecast_config.max_context}) '
            f'-- TimesFM silently returns all-NaN forecasts below this length '
            f'(see utils.timesfm_model.min_context_length docstring), it does '
            f'not raise its own error.'
        )
    points, quantiles_list = [], []
    for start in range(0, len(contexts), batch_size):
        batch = [np.asarray(c, dtype=np.float32) for c in contexts[start:start + batch_size]]
        point, quantiles = model.forecast(horizon=horizon, inputs=batch)
        points.append(np.asarray(point))
        quantiles_list.append(np.asarray(quantiles))
    return np.concatenate(points, axis=0), np.concatenate(quantiles_list, axis=0)


class TimesFMForecastPipeline:
    """Raw-input inference wrapper, same contract as
    DLinearForecastPipeline/PatchTSTForecastPipeline: fit() stores every
    series' full history (no training happens — "fit" here just means
    "remember the data to use as context later"); predict() takes bare
    Store/Dept/Date/IsHoliday rows and returns Weekly_Sales predictions.

    Series present in the prediction request but absent from the fitted
    history get NaN predictions (documented limitation, not silently
    zero-filled), same contract as the other three pipelines.
    """

    def __init__(self, model, context_length, horizon):
        min_ctx = min_context_length(model)
        if context_length < min_ctx:
            raise ValueError(
                f'context_length={context_length} is below the safe minimum of '
                f'{min_ctx} for this compiled model -- see '
                f'utils.timesfm_model.min_context_length docstring.'
            )
        self.model = model
        self.context_length = context_length
        self.horizon = horizon

    def fit(self, train_df):
        panel = build_full_calendar_panel(train_df)
        self.history_ = series_arrays_from_panel(panel)
        self.last_date_ = panel['Date'].max()
        return self

    def predict(self, raw_df):
        req = raw_df[['Store', 'Dept', 'Date']].copy()
        req['Date'] = pd.to_datetime(req['Date'])

        keys, contexts, n_blocks_list = [], [], []
        future_calendars = {}
        for key, g in req.groupby(['Store', 'Dept']):
            req_dates = g['Date'].sort_values().unique()
            if key not in self.history_ or req_dates.max() <= self.last_date_:
                continue
            sales, _holiday, _dates = self.history_[key]
            n_weeks_needed = int(round((req_dates.max() - self.last_date_) / pd.Timedelta(weeks=1)))
            n_blocks = int(np.ceil(n_weeks_needed / self.horizon))
            future_calendar = pd.date_range(
                self.last_date_ + pd.Timedelta(weeks=1), periods=n_blocks * self.horizon, freq='7D',
            )
            keys.append(key)
            contexts.append(sales[-self.context_length:])
            n_blocks_list.append(n_blocks)
            future_calendars[key] = future_calendar

        pred_rows = []
        if keys:
            # group by n_blocks so every call in a batch requests the same horizon
            unique_blocks = sorted(set(n_blocks_list))
            for nb in unique_blocks:
                idx = [i for i, n in enumerate(n_blocks_list) if n == nb]
                batch_contexts = [contexts[i] for i in idx]
                point, _ = forecast_series(self.model, batch_contexts, horizon=nb * self.horizon)
                point = np.clip(point, 0, None)
                for row_i, i in enumerate(idx):
                    key = keys[i]
                    for step, d in enumerate(future_calendars[key]):
                        pred_rows.append((key[0], key[1], d, float(point[row_i, step])))

        pred_df = pd.DataFrame(pred_rows, columns=['Store', 'Dept', 'Date', 'Weekly_Sales'])
        # explicit float cast: an empty pred_rows (e.g. every requested series
        # unseen) makes pandas infer object dtype for Weekly_Sales, which
        # turns the left-merge's fill value into None instead of a proper
        # float NaN, breaking np.isnan() downstream.
        pred_df['Weekly_Sales'] = pred_df['Weekly_Sales'].astype(float)
        merged = req.merge(pred_df, on=['Store', 'Dept', 'Date'], how='left')
        return merged['Weekly_Sales'].to_numpy()
