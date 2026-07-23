"""Global panel construction and windowing for N-BEATS.

N-BEATS is trained as one shared-weight model across every Store-Dept series
(no per-series fitting), reading fixed-length (lookback, horizon) windows of
raw Weekly_Sales only -- per the original paper it is a pure univariate
model, so Store/Dept/holiday information is used only for window bookkeeping
(which series a window belongs to, its Store Type, holiday flags on the
forecast steps for WMAE weighting/evaluation), never as a network input.

Reuses utils.feature_engineering.HOLIDAY_DATES so the definition of a
"holiday week" is identical to the LightGBM pipeline and utils.metrics.wmae.
"""

import numpy as np
import pandas as pd

from utils.feature_engineering import HOLIDAY_DATES

EPS = 1.0  # floor for mean-scale normalization (Weekly_Sales is in dollars)


def build_panel(train_df, stores_df):
    """Build a gap-free global panel: one row per (Store, Dept) per calendar
    week over the full observed date range, Weekly_Sales NaN where missing.

    Returns:
        sales: (n_series, n_dates) float32 array, NaN = missing week
        store_ids, dept_ids: (n_series,) arrays identifying each row
        store_types: (n_series,) array of 'A'/'B'/'C'
        all_dates: (n_dates,) DatetimeIndex, weekly frequency
        is_holiday: (n_dates,) bool array, True on a HOLIDAY_DATES week
    """
    all_dates = pd.date_range(train_df['Date'].min(), train_df['Date'].max(), freq='7D')
    pivot = train_df.pivot_table(index=['Store', 'Dept'], columns='Date',
                                  values='Weekly_Sales', aggfunc='mean')
    pivot = pivot.reindex(columns=all_dates)

    store_ids = pivot.index.get_level_values('Store').to_numpy()
    dept_ids = pivot.index.get_level_values('Dept').to_numpy()
    type_lookup = stores_df.set_index('Store')['Type']
    store_types = type_lookup.loc[store_ids].to_numpy()

    sales = pivot.to_numpy(dtype=np.float32)

    holiday_dates = set()
    for dates in HOLIDAY_DATES.values():
        holiday_dates.update(pd.to_datetime(dates))
    is_holiday = np.array([d in holiday_dates for d in all_dates], dtype=bool)

    return sales, store_ids, dept_ids, store_types, all_dates, is_holiday


def build_training_windows(sales, is_holiday, lookback, horizon, max_anchor_idx, stride=1):
    """Every valid (lookback -> horizon) window, for every series, with
    anchors from `lookback - 1` up to `max_anchor_idx` inclusive (so the
    window's forecast never reaches past index max_anchor_idx + horizon).

    A window is kept only if neither its lookback nor its horizon slice
    contains a NaN (gappy series are simply under-represented, never
    imputed). No down-sampling of series or of valid anchors (stride=1
    unless the caller explicitly widens it) -- vectorized per-anchor across
    all series at once (via sliding_window_view) rather than a
    series*anchor Python double loop, since this gets called once per
    fold/lookback combination during HPO.

    Returns X (N, lookback), Y (N, horizon), Y_holiday (N, horizon) float32/
    bool arrays and `series_idx` (N,) mapping each window back to its row in
    `sales` (i.e. to store_ids[series_idx], dept_ids[series_idx]).
    """
    n_series, n_dates = sales.shape
    max_anchor_idx = min(max_anchor_idx, n_dates - 1 - horizon)
    if max_anchor_idx < lookback - 1:
        return (np.empty((0, lookback), dtype=np.float32),
                np.empty((0, horizon), dtype=np.float32),
                np.empty((0, horizon), dtype=bool),
                np.empty((0,), dtype=np.int64))

    swv = np.lib.stride_tricks.sliding_window_view
    x_windows = swv(sales, lookback, axis=1)          # (n_series, n_dates-lookback+1, lookback)
    y_windows = swv(sales, horizon, axis=1)            # (n_series, n_dates-horizon+1, horizon)
    x_valid = ~np.isnan(x_windows).any(axis=2)         # (n_series, n_dates-lookback+1)
    y_valid = ~np.isnan(y_windows).any(axis=2)         # (n_series, n_dates-horizon+1)
    yh_windows = swv(is_holiday, horizon)               # (n_dates-horizon+1, horizon)

    xs, ys, yhs, sidx = [], [], [], []
    for anchor in range(lookback - 1, max_anchor_idx + 1, stride):
        a_x = anchor - lookback + 1
        b_y = anchor + 1
        mask = x_valid[:, a_x] & y_valid[:, b_y]
        if not mask.any():
            continue
        n_hit = int(mask.sum())
        xs.append(x_windows[mask, a_x, :])
        ys.append(y_windows[mask, b_y, :])
        yhs.append(np.broadcast_to(yh_windows[b_y], (n_hit, horizon)))
        sidx.append(np.nonzero(mask)[0])

    if not xs:
        return (np.empty((0, lookback), dtype=np.float32),
                np.empty((0, horizon), dtype=np.float32),
                np.empty((0, horizon), dtype=bool),
                np.empty((0,), dtype=np.int64))

    return (np.concatenate(xs).astype(np.float32),
            np.concatenate(ys).astype(np.float32),
            np.concatenate(yhs),
            np.concatenate(sidx).astype(np.int64))


def build_eval_window(sales, lookback, anchor_idx):
    """One window per series ending exactly at `anchor_idx` (the last date
    of that fold's/split's training range) -- the model's forecast input.

    Series without a full, NaN-free lookback ending at anchor_idx are
    dropped (cannot be forecast from insufficient/gappy history).
    Returns X (N, lookback) and `series_idx` (N,) mapping back into `sales`.
    """
    n_series = sales.shape[0]
    lb_slice = slice(anchor_idx - lookback + 1, anchor_idx + 1)
    valid = ~np.isnan(sales[:, lb_slice])
    keep = valid.all(axis=1)
    series_idx = np.nonzero(keep)[0]
    X = sales[series_idx, lb_slice].astype(np.float32)
    return X, series_idx


def scale_windows(X, eps=EPS):
    """Per-window mean-abs scale (N,1); network trains on X/scale, and
    predictions are multiplied back by scale before computing any loss/WMAE
    so the metric stays in real Weekly_Sales dollar units."""
    scale = np.mean(np.abs(X), axis=1, keepdims=True)
    scale = np.where(scale < eps, eps, scale)
    return scale.astype(np.float32)
