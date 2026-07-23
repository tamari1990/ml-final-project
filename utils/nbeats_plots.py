"""Plotting functions for the N-BEATS experiment notebook.

Static PNGs saved under `plots/` and logged as MLflow artifacts (mirrors the
LightGBM notebook's convention: keep artifacts small, log PNGs not raw
dataframes). Styling matches the repo's existing seaborn/matplotlib
convention (see eda_exploration.ipynb / model_experiment_LightGBM.ipynb):
whitegrid theme, 'deep' categorical palette used in a fixed order (never
re-cycled per filter), a single hue for magnitude/sequential plots, no
dual-axis charts.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf

sns.set_theme(style='whitegrid', palette='deep')
plt.rcParams['figure.figsize'] = (12, 5)
plt.rcParams['font.size'] = 11

PALETTE = sns.color_palette('deep')
ACTUAL_COLOR = PALETTE[0]
FORECAST_COLOR = PALETTE[1]
TREND_COLOR = PALETTE[2]
SEASON_COLOR = PALETTE[3]
STORE_TYPE_COLORS = {'A': PALETTE[0], 'B': PALETTE[1], 'C': PALETTE[2]}


def _savefig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_decomposition(model, X_series, dates, series_label, path):
    """Trend + seasonality components vs actual, for one interpretable-arch
    series (Section 8, plot 1 — N-BEATS' signature interpretability output)."""
    import torch
    from utils.nbeats_data import scale_windows

    device = next(model.parameters()).device
    scale = scale_windows(X_series[None, :])
    x = torch.tensor(X_series[None, :] / scale, dtype=torch.float32, device=device)
    with torch.no_grad():
        forecast, backcast_resid, stack_forecasts = model(x, return_decomposition=True)
    keys = list(stack_forecasts.keys())
    trend_key = next((k for k in keys if 'trend' in k), None)
    season_key = next((k for k in keys if 'seasonality' in k), None)

    fig, ax = plt.subplots(figsize=(12, 5))
    horizon = forecast.shape[1]
    lookback = X_series.shape[0]
    t_hist = np.arange(-lookback, 0)
    t_fut = np.arange(0, horizon)

    ax.plot(t_hist, X_series, color=ACTUAL_COLOR, label='Actual (lookback window)')
    ax.plot(t_fut, forecast[0].cpu().numpy() * scale[0], color=FORECAST_COLOR, label='Total forecast')
    if trend_key is not None:
        ax.plot(t_fut, stack_forecasts[trend_key][0].cpu().numpy() * scale[0], color=TREND_COLOR,
                linestyle='--', label='Trend component')
    if season_key is not None:
        ax.plot(t_fut, stack_forecasts[season_key][0].cpu().numpy() * scale[0], color=SEASON_COLOR,
                linestyle='--', label='Seasonality component')
    ax.axvline(0, color='gray', linewidth=1, linestyle=':')
    ax.set_title(f'N-BEATS interpretable decomposition — {series_label}')
    ax.set_xlabel('Weeks relative to forecast origin')
    ax.set_ylabel('Weekly_Sales')
    ax.legend()
    _savefig(fig, path)


def plot_backcast_reconstruction(model, X_series, series_label, path):
    """Backcast reconstruction vs the actual lookback window (Section 8, plot 2)."""
    import torch
    from utils.nbeats_data import scale_windows

    device = next(model.parameters()).device
    scale = scale_windows(X_series[None, :])
    x = torch.tensor(X_series[None, :] / scale, dtype=torch.float32, device=device)
    with torch.no_grad():
        _, backcast_residual = model(x)
    reconstructed = X_series - backcast_residual[0].cpu().numpy() * scale[0]

    fig, ax = plt.subplots(figsize=(12, 5))
    t = np.arange(len(X_series))
    ax.plot(t, X_series, color=ACTUAL_COLOR, label='Actual input window')
    ax.plot(t, reconstructed, color=FORECAST_COLOR, linestyle='--', label='Model backcast reconstruction')
    ax.set_title(f'Backcast reconstruction — {series_label}')
    ax.set_xlabel('Week (within lookback window)')
    ax.set_ylabel('Weekly_Sales')
    ax.legend()
    _savefig(fig, path)


def plot_forecast_vs_actual(examples, path):
    """Forecast vs actual for best/median/worst-WMAE series (Section 8, plot 3).
    `examples`: list of (label, y_true_1d, y_pred_1d, wmae) tuples."""
    fig, axes = plt.subplots(len(examples), 1, figsize=(12, 4 * len(examples)), sharex=True)
    if len(examples) == 1:
        axes = [axes]
    for ax, (label, y_true, y_pred, series_wmae) in zip(axes, examples):
        t = np.arange(1, len(y_true) + 1)
        ax.plot(t, y_true, color=ACTUAL_COLOR, marker='o', label='Actual')
        ax.plot(t, y_pred, color=FORECAST_COLOR, marker='o', label='Forecast')
        ax.set_title(f'{label} (WMAE={series_wmae:.0f})')
        ax.set_ylabel('Weekly_Sales')
        ax.legend()
    axes[-1].set_xlabel('Forecast step (h)')
    _savefig(fig, path)


def plot_error_by_horizon(step, abs_error, path):
    """Mean absolute error by forecast horizon step h=1..H (Section 8, plot 4)."""
    df_step = np.asarray(step).flatten()
    df_err = np.asarray(abs_error).flatten()
    steps = np.unique(df_step)
    means = [df_err[df_step == s].mean() for s in steps]
    stds = [df_err[df_step == s].std() for s in steps]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(steps, means, yerr=stds, color=ACTUAL_COLOR, capsize=3)
    ax.set_title('Mean absolute error by forecast horizon step')
    ax.set_xlabel('Horizon step (h)')
    ax.set_ylabel('Mean |error|')
    ax.set_xticks(steps)
    _savefig(fig, path)


def plot_residual_diagnostics(residuals, dates, path_prefix):
    """Residuals over time, histogram, and ACF (Section 8, plot 5) — three
    separate figures saved as f'{path_prefix}_timeseries.png' /
    '_hist.png' / '_acf.png'."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(dates, residuals, color=ACTUAL_COLOR, linewidth=1)
    ax.axhline(0, color='gray', linewidth=1)
    ax.set_title('Residuals (actual - forecast) over time')
    ax.set_ylabel('Residual')
    _savefig(fig, f'{path_prefix}_timeseries.png')

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(residuals, bins=60, color=ACTUAL_COLOR, ax=ax)
    ax.axvline(0, color='gray', linewidth=1)
    ax.set_title('Residual distribution')
    ax.set_xlabel('Residual')
    _savefig(fig, f'{path_prefix}_hist.png')

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_acf(residuals, ax=ax, lags=min(20, len(residuals) // 2 - 1))
    ax.set_title('ACF of residuals')
    _savefig(fig, f'{path_prefix}_acf.png')


def plot_wmae_distribution(series_wmae, path):
    """Histogram of per-series WMAE across all Store-Dept combos, flagging
    worst performers (Section 8, plot 6)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(series_wmae, bins=60, color=ACTUAL_COLOR, ax=ax)
    p90 = np.percentile(series_wmae, 90)
    ax.axvline(p90, color=PALETTE[3], linestyle='--', label=f'90th pct = {p90:.0f}')
    ax.set_title('Per-series WMAE distribution')
    ax.set_xlabel('WMAE')
    ax.legend()
    _savefig(fig, path)


def plot_wmae_breakdown(holiday_wmae, non_holiday_wmae, type_wmae, path):
    """WMAE by holiday vs non-holiday and by Store Type A/B/C, side by side
    (Section 8, plot 7). `type_wmae`: dict {'A': v, 'B': v, 'C': v}."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(['Non-holiday', 'Holiday'], [non_holiday_wmae, holiday_wmae],
                color=[ACTUAL_COLOR, PALETTE[3]])
    axes[0].set_title('WMAE: holiday vs non-holiday weeks')
    axes[0].set_ylabel('WMAE')

    types = ['A', 'B', 'C']
    axes[1].bar(types, [type_wmae.get(t, np.nan) for t in types],
                color=[STORE_TYPE_COLORS[t] for t in types])
    axes[1].set_title('WMAE by Store Type')
    axes[1].set_ylabel('WMAE')
    _savefig(fig, path)


def plot_optuna_diagnostics(study, prefix):
    """Hyperparameter-importance, parallel-coordinate, optimization-history
    (Section 8, plot 8) via optuna's matplotlib backend, saved as
    f'{prefix}_importance.png' / '_parallel.png' / '_history.png'."""
    import optuna.visualization.matplotlib as opt_mpl

    ax = opt_mpl.plot_param_importances(study)
    ax.figure.tight_layout()
    ax.figure.savefig(f'{prefix}_importance.png', dpi=120, bbox_inches='tight')
    plt.close(ax.figure)

    ax = opt_mpl.plot_parallel_coordinate(study)
    ax.figure.tight_layout()
    ax.figure.savefig(f'{prefix}_parallel.png', dpi=120, bbox_inches='tight')
    plt.close(ax.figure)

    ax = opt_mpl.plot_optimization_history(study)
    ax.figure.tight_layout()
    ax.figure.savefig(f'{prefix}_history.png', dpi=120, bbox_inches='tight')
    plt.close(ax.figure)


def plot_loss_curves(history, title, path):
    """Train loss / val WMAE per epoch for a final chosen configuration
    (Section 8, plot 9). `history`: {'train_loss': [...], 'val_wmae': [...]}."""
    fig, ax = plt.subplots(figsize=(9, 5))
    epochs = np.arange(1, len(history['train_loss']) + 1)
    ax.plot(epochs, history['train_loss'], color=ACTUAL_COLOR, label='Train loss')
    ax.plot(epochs, history['val_wmae'], color=FORECAST_COLOR, label='Val WMAE')
    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss / WMAE')
    ax.legend()
    _savefig(fig, path)
