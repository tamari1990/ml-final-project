"""Shared feature engineering for the Walmart Store Sales Forecasting models.

Reused across model notebooks (LightGBM, XGBoost, ...) so the feature
definitions stay identical between architectures.
"""

import numpy as np
import pandas as pd

CATEGORICAL_COLS = ['Store', 'Dept', 'Type']

# Official Kaggle "Super Bowl / Labor Day / Thanksgiving / Christmas" dates,
# 2010-2013 (covers both train.csv and test.csv). The Christmas week is
# labeled by its last day (e.g. 2010-12-31), not Dec 25.
HOLIDAY_DATES = {
    'SuperBowl': ['2010-02-12', '2011-02-11', '2012-02-10', '2013-02-08'],
    'LaborDay': ['2010-09-10', '2011-09-09', '2012-09-07', '2013-09-06'],
    'Thanksgiving': ['2010-11-26', '2011-11-25', '2012-11-23', '2013-11-29'],
    'Christmas': ['2010-12-31', '2011-12-30', '2012-12-28', '2013-12-27'],
}


def merge_raw(df, features, stores):
    """Merge a train/test frame with stores.csv and features.csv.

    features.csv carries its own IsHoliday column, identical to train/test's,
    so it's dropped here to avoid a duplicate/suffix collision.
    """
    return (
        df.merge(stores, on='Store', how='left')
          .merge(features.drop(columns=['IsHoliday']), on=['Store', 'Date'], how='left')
    )


def add_calendar_features(df):
    """Year/Month/WeekOfYear/DayOfYear plus a boolean flag per named holiday."""
    df = df.copy()
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
    df['DayOfYear'] = df['Date'].dt.dayofyear

    for name, dates in HOLIDAY_DATES.items():
        df[f'Is{name}'] = df['Date'].isin(pd.to_datetime(dates))
    return df


def _reindex_to_full_calendar(df):
    """Insert explicit NaN rows for any missing week in each Store-Dept series.

    About 18% of (Store, Dept) series in this dataset have gaps in their
    weekly history (a dept not operating some weeks). groupby().shift(n) /
    .rolling(w) both operate by row position, not by calendar time — on a
    gappy series that silently makes "lag13" mean "13 rows back" (which can
    be *more* than 13 calendar weeks) instead of "13 weeks back". Reindexing
    onto the full weekly calendar first makes every series gap-free, so a
    row-positional shift/rolling becomes calendar-correct by construction.
    """
    all_dates = pd.date_range(df['Date'].min(), df['Date'].max(), freq='7D')
    pairs = df[['Store', 'Dept']].drop_duplicates()
    grid = pairs.merge(pd.DataFrame({'Date': all_dates}), how='cross')
    return grid.merge(df, on=['Store', 'Dept', 'Date'], how='left')


def add_lag_features(df, lags=(13, 52)):
    """lag13 / lag52 of Weekly_Sales per Store-Dept series, time-ordered.

    Requires a gap-free calendar per Store-Dept (see _reindex_to_full_calendar)
    so that a shift of `lag` rows is truly a shift of `lag` calendar weeks.
    Must be called on the full time-ordered series (train history + whatever
    window is being featurized) — never per-fold. NaN where history isn't
    available yet; never filled.
    """
    df = df.sort_values(['Store', 'Dept', 'Date']).copy()
    grouped = df.groupby(['Store', 'Dept'])['Weekly_Sales']
    for lag in lags:
        df[f'lag{lag}'] = grouped.shift(lag)
    return df


def add_rolling_features(df, windows=(4, 8)):
    """Rolling mean/std of Weekly_Sales per Store-Dept, shifted by 1 first.

    Requires a gap-free calendar per Store-Dept (see _reindex_to_full_calendar)
    so window w truly spans w calendar weeks. The shift(1) happens before the
    rolling window so window w's row t aggregates weeks [t-w, t-1] — the
    current row's own Weekly_Sales is never part of its own rolling stat.
    """
    df = df.sort_values(['Store', 'Dept', 'Date']).copy()
    shifted = df.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(1)
    df['_shifted_sales'] = shifted
    grouped = df.groupby(['Store', 'Dept'])['_shifted_sales']
    for w in windows:
        df[f'roll_mean_{w}'] = grouped.rolling(w).mean().reset_index(level=[0, 1], drop=True)
        df[f'roll_std_{w}'] = grouped.rolling(w).std().reset_index(level=[0, 1], drop=True)
    df = df.drop(columns=['_shifted_sales'])
    return df


def encode_categoricals(df, categorical_cols=CATEGORICAL_COLS):
    """Cast Store/Dept/Type to pandas 'category' dtype for LightGBM's native categorical support."""
    df = df.copy()
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
    return df


def build_features(df, features, stores, history_df=None, is_train=True):
    """Full feature pipeline: merge -> calendar -> lag -> rolling -> categorical encode.

    df: raw rows to featurize (e.g. a train slice, a local test slice, or
        Kaggle's test.csv).
    features, stores: features.csv / stores.csv, unmodified.
    history_df: raw rows (with Weekly_Sales) strictly *before* df's own
        start, used only to supply lag/rolling context so df's earliest
        rows aren't starved of history that legitimately exists. Required
        whenever df doesn't already contain its own series start — e.g.
        featurizing Kaggle's test.csv needs history_df=train (test has no
        Weekly_Sales of its own); featurizing a walk-forward CV validation
        fold needs history_df=that fold's train range. Leave it as None
        when df already spans the full series from its start (e.g.
        featurizing the whole of train.csv in one call).

        IMPORTANT for leakage-safety: history_df must stop strictly before
        df starts. Never pass a history_df that includes anything from
        df's own date range or later — otherwise lag/rolling values for the
        end of history_df would leak into rows that are supposed to be
        held out.
    is_train: if False, df is assumed to have no Weekly_Sales column (e.g.
        Kaggle test.csv) — a NaN Weekly_Sales column is added so lag/rolling
        can run, then the engineered rows for df are returned without it.
    """
    if history_df is not None:
        overlap = history_df['Date'].max()
        df_start = df['Date'].min()
        if overlap >= df_start:
            raise ValueError(
                f'history_df must end strictly before df starts to avoid leakage '
                f'(history_df max Date={overlap.date()}, df min Date={df_start.date()})'
            )

    work = df[['Store', 'Dept', 'Date'] + (['Weekly_Sales'] if 'Weekly_Sales' in df.columns else [])].copy()
    work['_is_own'] = True
    if not is_train and 'Weekly_Sales' not in work.columns:
        work['Weekly_Sales'] = np.nan

    if history_df is not None:
        hist = history_df[['Store', 'Dept', 'Date'] + (['Weekly_Sales'] if 'Weekly_Sales' in history_df.columns else [])].copy()
        hist['_is_own'] = False
        if 'Weekly_Sales' not in hist.columns:
            hist['Weekly_Sales'] = np.nan
        combined = pd.concat([hist, work], ignore_index=True)
    else:
        combined = work

    # Lag/rolling need a gap-free weekly calendar per Store-Dept to be
    # calendar-correct; only Store/Dept/Date/Weekly_Sales are needed for that,
    # so features.csv/stores.csv are joined afterward, only onto real rows.
    combined = _reindex_to_full_calendar(combined)
    combined = add_lag_features(combined)
    combined = add_rolling_features(combined)

    out = combined[combined['_is_own'] == True].drop(columns=['_is_own'])  # noqa: E712

    # IsHoliday is the only column train.csv/test.csv carry beyond
    # Store/Dept/Date/Weekly_Sales; restore it (it was dropped when `work`
    # was pared down to just the columns the calendar-reindex step needs).
    out = out.merge(df[['Store', 'Dept', 'Date', 'IsHoliday']], on=['Store', 'Dept', 'Date'], how='left')

    out = merge_raw(out, features, stores)
    out = add_calendar_features(out)
    out = encode_categoricals(out)

    if not is_train:
        out = out.drop(columns=['Weekly_Sales'])

    return out.sort_values(['Store', 'Dept', 'Date']).reset_index(drop=True)
