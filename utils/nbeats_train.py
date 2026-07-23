"""Training / evaluation harness for N-BEATS, shared between
model_experiment_NBEATS.ipynb and run_nbeats_hpo.py (the overnight Optuna
search script), the same way utils/feature_engineering.py is shared by every
model notebook.

CV folds are index-based (see `make_cv_folds`) but mirror
model_experiment_LightGBM.ipynb's walk-forward CV *exactly*: same
INITIAL_TRAIN_WEEKS=52 / VAL_WEEKS=13 / N_FOLDS=3 boundaries, same
Thanksgiving/Christmas-reserved-for-final-holdout property, over the same
143-week date range -- verified to produce identical (train_len, val_len)
per fold as the LightGBM notebook's date-based version.
"""

import numpy as np
import torch

from utils.nbeats_data import build_training_windows, build_eval_window, scale_windows
from utils.nbeats_model import NBeatsNet
from utils.nbeats_losses import LOSS_FUNCTIONS
from utils.metrics import wmae as wmae_ref

LOCAL_TEST_WEEKS = 52
INITIAL_TRAIN_WEEKS = 52
VAL_WEEKS = 13
N_FOLDS = 3
HORIZON = VAL_WEEKS

MIN_WINDOWS_PER_FOLD = 30  # below this, a (fold, lookback) combo is too data-starved to score fairly


def make_cv_folds(n_dates, initial_train_weeks=INITIAL_TRAIN_WEEKS, val_weeks=VAL_WEEKS, n_folds=N_FOLDS):
    """Expanding-window (train_end_idx, val_start_idx, val_end_idx) index
    triples, 0-based & inclusive, walking forward -- index analogue of
    LightGBM's make_walk_forward_folds, restricted to stay inside the local
    (non-held-out) range."""
    folds = []
    for i in range(n_folds):
        train_end_idx = initial_train_weeks + i * val_weeks - 1
        val_start_idx = train_end_idx + 1
        val_end_idx = val_start_idx + val_weeks - 1
        folds.append((train_end_idx, val_start_idx, val_end_idx))
    return folds


def local_split_idx(n_dates, local_test_weeks=LOCAL_TEST_WEEKS):
    """local_train covers [0, local_train_end_idx], local_test covers
    [local_train_end_idx+1, n_dates-1] -- index analogue of LightGBM's
    last-52-weeks holdout."""
    local_train_end_idx = n_dates - local_test_weeks - 1
    return local_train_end_idx


def max_windows_for_lookback(train_len, lookback, horizon=HORIZON):
    """How many anchor positions a training range of `train_len` weeks can
    offer for a given lookback (upper bound; per-series NaN gaps reduce this
    further). <=0 means the config is infeasible for this range."""
    return train_len - lookback - horizon + 1


def build_model(config, device='cpu'):
    """Instantiate an NBeatsNet from a flat hyperparameter dict, placed on
    `device` (defaults to 'cpu' so every existing call site is unaffected;
    pass device='cuda' to train on GPU -- train_model/train_fixed_epochs/
    forecast_series auto-detect the model's device from its own parameters,
    so no other call site needs to change)."""
    arch = config['architecture']
    if arch == 'generic':
        stack_types = ['generic'] * config['n_stacks']
    elif arch == 'interpretable':
        stack_types = ['trend', 'seasonality']
    elif arch == 'mixed':
        stack_types = ['trend', 'seasonality'] + ['generic'] * max(0, config['n_stacks'] - 2)
    else:
        raise ValueError(arch)

    lookback = config['lookback_multiplier'] * HORIZON
    model = NBeatsNet(
        backcast_size=lookback,
        forecast_size=HORIZON,
        stack_types=stack_types,
        n_blocks_per_stack=config['n_blocks'],
        layer_size=config['layer_size'],
        n_layers=config['n_fc_layers'],
        degree=config.get('trend_degree', 3),
        harmonics=config.get('seasonality_harmonics'),
        dropout=config.get('dropout', 0.0),
        share_weights=config.get('share_weights', False),
    )
    return model.to(device)


def _make_loader(X, Y, Yh, batch_size, generator=None, holiday_boost=0.0):
    """`holiday_boost=0` (default) is the original uniform-shuffle loader.
    `holiday_boost>0` switches to a WeightedRandomSampler that upweights any
    training window whose horizon (Yh) contains at least one holiday week --
    holiday weeks are rare in the window pool, so the network otherwise gets
    little gradient signal from them. Window weight = 1 + (n_holiday_weeks_in_
    window * holiday_boost); no new input feature, only sampling frequency
    changes, so the model is still purely univariate."""
    scale = scale_windows(X)
    Xn = torch.tensor(X / scale, dtype=torch.float32)
    Yn = torch.tensor(Y / scale, dtype=torch.float32)
    Yh_t = torch.tensor(Yh, dtype=torch.float32)
    scale_t = torch.tensor(scale, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(Xn, Yn, Yh_t, scale_t)
    if holiday_boost > 0:
        n_holiday_weeks = Yh.astype(np.float64).sum(axis=1)
        weights = torch.tensor(1.0 + n_holiday_weeks * holiday_boost, dtype=torch.double)
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=weights, num_samples=len(weights), replacement=True, generator=generator)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)


def train_model(model, X_tr, Y_tr, Yh_tr, config, eval_fn=None, max_epochs=60,
                patience=8, seed=42, verbose=False):
    """Train `model` on windowed arrays with early stopping on `eval_fn`
    (called each epoch, must return a scalar val WMAE -- lower is better).

    Returns (best_state_dict, history) where history has per-epoch
    train_loss / val_wmae lists (for the loss-curve plot).
    """
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    loader = _make_loader(X_tr, Y_tr, Yh_tr, config['batch_size'], generator=gen,
                           holiday_boost=config.get('holiday_boost', 0.0))
    loss_fn = LOSS_FUNCTIONS[config['loss']]

    if config['optimizer'] == 'adamw':
        opt = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],
                                 weight_decay=config.get('weight_decay', 0.0))
    else:
        opt = torch.optim.Adam(model.parameters(), lr=config['learning_rate'],
                                weight_decay=config.get('weight_decay', 0.0))

    best_val = np.inf
    best_state = None
    best_epoch = 0
    epochs_since_improve = 0
    history = {'train_loss': [], 'val_wmae': []}

    device = next(model.parameters()).device
    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_losses = []
        for xb, yb, yhb, sb in loader:
            xb, yb, yhb, sb = xb.to(device), yb.to(device), yhb.to(device), sb.to(device)
            forecast, _ = model(xb)
            pred_raw = forecast * sb
            y_raw = yb * sb
            loss = loss_fn(pred_raw, y_raw, yhb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_losses.append(loss.item())
        train_loss = float(np.mean(epoch_losses))
        history['train_loss'].append(train_loss)

        val_wmae = eval_fn(model) if eval_fn is not None else train_loss
        history['val_wmae'].append(val_wmae)

        if verbose:
            print(f'  epoch {epoch:3d}: train_loss={train_loss:.2f} val_wmae={val_wmae:.2f}')

        if val_wmae < best_val - 1e-6:
            best_val = val_wmae
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_val, best_epoch


def train_fixed_epochs(model, X_tr, Y_tr, Yh_tr, config, n_epochs, seed=42, verbose=False):
    """Train for exactly `n_epochs` with no early stopping and no held-out
    validation slice -- used for final-member fits where `n_epochs` was
    already chosen from CV (mean best-epoch across folds, mirroring the
    LightGBM notebook's `best_n_estimators` = mean best_iteration across
    folds). Lets the final fit use its *entire* training pool for gradient
    updates instead of reserving a redundant validation block."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    loader = _make_loader(X_tr, Y_tr, Yh_tr, config['batch_size'], generator=gen,
                           holiday_boost=config.get('holiday_boost', 0.0))
    loss_fn = LOSS_FUNCTIONS[config['loss']]

    if config['optimizer'] == 'adamw':
        opt = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],
                                 weight_decay=config.get('weight_decay', 0.0))
    else:
        opt = torch.optim.Adam(model.parameters(), lr=config['learning_rate'],
                                weight_decay=config.get('weight_decay', 0.0))

    history = {'train_loss': []}
    device = next(model.parameters()).device
    for epoch in range(1, max(1, n_epochs) + 1):
        model.train()
        epoch_losses = []
        for xb, yb, yhb, sb in loader:
            xb, yb, yhb, sb = xb.to(device), yb.to(device), yhb.to(device), sb.to(device)
            forecast, _ = model(xb)
            pred_raw = forecast * sb
            y_raw = yb * sb
            loss = loss_fn(pred_raw, y_raw, yhb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_losses.append(loss.item())
        train_loss = float(np.mean(epoch_losses))
        history['train_loss'].append(train_loss)
        if verbose:
            print(f'  epoch {epoch:3d}: train_loss={train_loss:.2f}')
    return model, history


@torch.no_grad()
def forecast_series(model, X_eval):
    model.eval()
    device = next(model.parameters()).device
    scale = scale_windows(X_eval)
    Xn = torch.tensor(X_eval / scale, dtype=torch.float32, device=device)
    forecast, _ = model(Xn)
    return forecast.cpu().numpy() * scale


def evaluate_on_window(model, sales, is_holiday, lookback, anchor_idx, val_start_idx, horizon=HORIZON):
    """Forecast every eligible series from `anchor_idx` and score WMAE
    against the true values in [val_start_idx, val_start_idx+horizon-1].
    Returns (wmae, y_true, y_pred, is_holiday_grid, series_idx)."""
    X_eval, series_idx = build_eval_window(sales, lookback, anchor_idx)
    if len(series_idx) == 0:
        return None, None, None, None, series_idx
    y_true = sales[series_idx, val_start_idx:val_start_idx + horizon]
    valid_rows = ~np.isnan(y_true).any(axis=1)
    if valid_rows.sum() == 0:
        return None, None, None, None, series_idx[valid_rows]
    X_eval = X_eval[valid_rows]
    y_true = y_true[valid_rows]
    series_idx = series_idx[valid_rows]
    y_pred = forecast_series(model, X_eval)
    holiday_grid = np.broadcast_to(is_holiday[val_start_idx:val_start_idx + horizon], y_true.shape)
    fold_wmae = wmae_ref(y_true.flatten(), y_pred.flatten(), holiday_grid.flatten())
    return fold_wmae, y_true, y_pred, holiday_grid, series_idx


def run_cv_for_config(config, sales, is_holiday, cv_folds, max_epochs=60, patience=8, seed=42,
                       verbose=False, min_folds=2, device='cpu'):
    """Walk-forward CV score for one hyperparameter config: trains one model
    per fold (train windows built only from that fold's own train range) and
    evaluates on that fold's held-out val window, exactly mirroring the
    LightGBM CV harness's per-fold fit/predict/WMAE loop.

    Folds whose training range can't produce >= MIN_WINDOWS_PER_FOLD windows
    for this config's lookback are skipped (too data-starved to trust);
    the trial is invalid (returns None) if fewer than `min_folds` qualify
    (default 2, for real HPO scoring; pass 1 when deliberately scoring a
    single fold, e.g. a diagnostic re-run for a loss-curve plot).
    """
    lookback = config['lookback_multiplier'] * HORIZON
    fold_wmaes = []
    fold_details = []

    for train_end_idx, val_start_idx, val_end_idx in cv_folds:
        train_len = train_end_idx + 1
        if max_windows_for_lookback(train_len, lookback) < 1:
            continue

        max_anchor_idx = train_end_idx - HORIZON
        X_tr, Y_tr, Yh_tr, sidx_tr = build_training_windows(
            sales, is_holiday, lookback, HORIZON, max_anchor_idx=max_anchor_idx)
        if len(X_tr) < MIN_WINDOWS_PER_FOLD:
            continue

        model = build_model(config, device=device)

        def eval_fn(m, ti=train_end_idx, vi=val_start_idx):
            res = evaluate_on_window(m, sales, is_holiday, lookback, ti, vi, HORIZON)
            return res[0] if res[0] is not None else np.inf

        model, history, best_val, best_epoch = train_model(
            model, X_tr, Y_tr, Yh_tr, config, eval_fn=eval_fn,
            max_epochs=max_epochs, patience=patience, seed=seed, verbose=verbose)

        # Holiday-only WMAE at this fold's best (already-loaded) state -- re-uses
        # the same val window `evaluate_on_window` already scores overall, just
        # sliced to holiday weeks, so a config that wins on mean_wmae but is
        # quietly terrible on holidays doesn't look identical to one that isn't.
        res = evaluate_on_window(model, sales, is_holiday, lookback, train_end_idx, val_start_idx, HORIZON)
        _, fold_y_true, fold_y_pred, fold_holiday_grid, _ = res
        fold_holiday_wmae = np.nan
        if fold_y_true is not None:
            holiday_mask = fold_holiday_grid.astype(bool)
            if holiday_mask.any():
                fold_holiday_wmae = wmae_ref(fold_y_true[holiday_mask].flatten(),
                                              fold_y_pred[holiday_mask].flatten(),
                                              fold_holiday_grid[holiday_mask].flatten())

        fold_wmaes.append(best_val)
        fold_details.append({
            'train_end_idx': train_end_idx, 'val_start_idx': val_start_idx,
            'n_train_windows': len(X_tr), 'wmae': best_val, 'best_epoch': best_epoch, 'history': history,
            'holiday_wmae': fold_holiday_wmae,
        })

    if len(fold_wmaes) < min_folds:
        return None

    holiday_wmaes = [fd['holiday_wmae'] for fd in fold_details if not np.isnan(fd['holiday_wmae'])]

    return {
        'mean_wmae': float(np.mean(fold_wmaes)),
        'std_wmae': float(np.std(fold_wmaes)),
        'n_folds_used': len(fold_wmaes),
        'mean_best_epoch': float(np.mean([fd['best_epoch'] for fd in fold_details])),
        'mean_holiday_wmae': float(np.mean(holiday_wmaes)) if holiday_wmaes else float('nan'),
        'fold_details': fold_details,
    }
