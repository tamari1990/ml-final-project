"""Shared feature engineering for the Walmart Store Sales Forecasting models.

Reused across model notebooks (LightGBM, XGBoost, ...) so the feature
definitions stay identical between architectures.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

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

    features.csv has zero CPI/Unemployment coverage for its last ~13 weeks
    (2013-05-03 -> 2013-07-26, confirmed directly against the raw file) -- a
    real gap in the competition data itself, not specific to any one caller.
    Filled here (forward-fill per store on features.csv's own full timeline,
    before the merge) rather than on the merged/output frame, because a
    caller processing a narrow date slice at a time (e.g. one week of a
    recursive multi-step forecast) would otherwise have nothing adjacent
    within its own call to fill from -- features.csv itself always has full
    continuity regardless of how narrow df is. train.csv (and everything
    derived from it, including every local CV/holdout evaluation in this
    project) ends 2012-10-26, so this window is never exercised there --
    only Kaggle's real test.csv reaches it. bfill is a safety net for a
    series starting mid-gap (not the case here, but not a given in general).
    """
    features = features.sort_values(['Store', 'Date']).copy()
    features[['CPI', 'Unemployment']] = (
        features.groupby('Store')[['CPI', 'Unemployment']].transform(lambda s: s.ffill().bfill())
    )
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


def recursive_predict(test_df, initial_history_df, features, stores, feature_selector, model, verbose=False):
    """Genuinely blind multi-step-ahead prediction for a fitted lag/rolling-
    feature model (XGBoost/LightGBM style), one calendar week at a time.

    Calling build_features(test_df, ..., is_train=False) on a *whole*
    multi-week test set in one shot is a real bug, not a safe shortcut:
    with no Weekly_Sales at all for test_df, any lag/rolling window shorter
    than the full test span (lag13, roll_mean_4/8, roll_std_4/8 here) ends
    up referencing *other still-unknown test rows* once the window moves
    past the first few weeks, cascading into NaN for most of the test
    period. This is exactly what happened to the real Kaggle submission
    here: roll_mean_4/8 went NaN from the 2nd test week onward, lag13 from
    the 13th -- for 4 of 5 lag/rolling features, over most of the 39-week
    test window -- and the internal holdout evaluation never caught it
    because it evaluates on local_test_raw with is_train=True (real
    ground-truth Weekly_Sales throughout, since local_test_raw is a slice
    of train.csv, not genuinely-unseen data) -- a fundamentally easier task
    that never exercises the all-NaN-future scenario Kaggle's real test.csv
    actually is. Root-caused directly: internal holdout WMAE was 1639.12,
    actual Kaggle score was ~8260-8495 (public/private) -- confirmed via a
    genuinely-blind local re-evaluation using *this* function instead
    (1580.80, in line with the original number, not ~5x worse) that the gap
    was entirely this bug, not a real generalization problem.

    Fixes it by predicting one week at a time and feeding each week's own
    predictions back into a running history before featurizing the next
    week, so short lag/rolling windows get real numbers (predictions
    standing in for the still-unknown truth) instead of NaN.

    test_df: raw Store/Dept/Date[/IsHoliday] rows to predict, e.g. test.csv.
    initial_history_df: real history (Store/Dept/Date/Weekly_Sales) strictly
        before test_df's first date, e.g. all of train.csv.
    feature_selector, model: a fitted pipeline's own 'feature_selection' and
        'model' steps (reuse them directly -- don't refit).
    """
    running_history = initial_history_df[['Store', 'Dept', 'Date', 'Weekly_Sales']].copy()
    test_df = test_df.copy()
    test_df['Date'] = pd.to_datetime(test_df['Date'])
    unique_dates = np.sort(test_df['Date'].unique())

    all_preds = []
    baseline_nan_rate = None
    for i, d in enumerate(unique_dates):
        week_rows = test_df[test_df['Date'] == d]
        feat_week = build_features(week_rows, features, stores, history_df=running_history, is_train=False)

        X_week = feature_selector.transform(feat_week)
        # Some NaN in lag/rolling columns is expected and legitimate here --
        # ~18% of series have real calendar gaps (a dept not operating some
        # weeks), and _reindex_to_full_calendar fills those with genuine
        # NaN, not 0 (see its docstring). XGBoost (tree_method='hist')
        # handles this natively via a learned default split direction, same
        # as it was trained to. What this loop exists to prevent is NaN
        # from *missing recursive history* -- i.e. a lag/rolling window
        # landing on a week that's still unknown because a prior iteration
        # failed to feed its predictions back. That failure mode shows up
        # as a NaN rate that grows over the course of the loop (more weeks
        # forecast = more missing history), not a roughly-constant baseline
        # rate present from week 1 onward (what genuine calendar gaps look
        # like, since they don't depend on how far into the forecast we
        # are) -- so only raise if the rate climbs well past week 1's own.
        n_nan = int(X_week.isna().sum().sum())
        nan_rate = n_nan / X_week.size
        if baseline_nan_rate is None:
            baseline_nan_rate = nan_rate
        elif nan_rate > baseline_nan_rate + 0.05:
            nan_cols = X_week.columns[X_week.isna().any()].tolist()
            raise ValueError(
                f'week {d}: NaN rate {nan_rate:.4f} is well above week-1 baseline '
                f'{baseline_nan_rate:.4f} in columns {nan_cols} -- recursive history feed is likely broken'
            )

        preds_week = np.clip(model.predict(X_week), 0, None)

        pred_rows = feat_week[['Store', 'Dept', 'Date']].copy()
        pred_rows['Weekly_Sales'] = preds_week
        all_preds.append(pred_rows)

        running_history = pd.concat([running_history, pred_rows], ignore_index=True)

        if verbose:
            print(f'  week {i + 1}/{len(unique_dates)} ({d.date() if hasattr(d, "date") else d}): '
                  f'{len(week_rows)} rows, nan_rate={nan_rate:.4f}, pred mean={preds_week.mean():.1f}, max={preds_week.max():.1f}')

    return pd.concat(all_preds, ignore_index=True)


class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    """Wraps build_features (merge + calendar + lag + rolling) as a Pipeline step.

    Bakes in features.csv/stores.csv at construction time and, once fit, the
    full raw training history — so a fitted pipeline's transform()/predict()
    can be called on bare Store/Dept/Date/IsHoliday rows (e.g. test.csv,
    unmerged, un-featurized) with no manual preprocessing by the caller.

    Whether a call is a training-time pass (has its own Weekly_Sales, safe to
    treat as self-contained series history) or a genuine future-prediction
    pass (no Weekly_Sales, needs the stored history for lag/rolling context)
    is inferred from whether 'Weekly_Sales' is present in X — this mirrors
    build_features' own is_train/history_df contract.

    WARNING: calling predict()/transform() on a *whole* multi-week
    future-prediction batch in one shot silently produces NaN lag/rolling
    features for most of that batch — see recursive_predict()'s docstring
    for exactly why and the real incident this caused. Safe uses: (a)
    training-time calls (X has its own Weekly_Sales), or (b) a
    future-prediction batch no longer than the shortest lag/rolling window
    (4 weeks, here). For anything longer — e.g. a full Kaggle test.csv —
    use recursive_predict() instead of calling this (or Pipeline.predict())
    directly.
    """

    def __init__(self, features, stores):
        self.features = features
        self.stores = stores

    def fit(self, X, y=None):
        self.history_df_ = X.copy()
        return self

    def transform(self, X):
        is_train = 'Weekly_Sales' in X.columns
        history_df = None if is_train else self.history_df_
        return build_features(X, self.features, self.stores, history_df=history_df, is_train=is_train)


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Subsets to the feature columns chosen in feature selection, casting bool columns to int for LightGBM."""

    def __init__(self, feature_names):
        self.feature_names = feature_names

    def fit(self, X, y=None):
        self.selected_features_ = list(self.feature_names)
        return self

    def transform(self, X):
        X = X[self.selected_features_].copy()
        for c in X.select_dtypes(include='bool').columns:
            X[c] = X[c].astype(int)
        return X
