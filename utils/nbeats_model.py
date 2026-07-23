"""N-BEATS architecture (Oreshkin et al., 2020), from scratch in PyTorch.

Shared, importable module (mirrors utils/feature_engineering.py's role for
LightGBM) reused by model_experiment_NBEATS.ipynb. Implements the paper's
doubly-residual stacking, both the generic (fully learned basis) and
interpretable (polynomial trend + Fourier seasonality basis) configurations,
and a simple prediction-averaging ensemble across trained members.
"""

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Basis functions
# ---------------------------------------------------------------------------

class GenericBasis(nn.Module):
    """Fully learned basis: the block's own FC output IS the backcast/forecast.

    theta_b has size `backcast_size`, theta_f has size `forecast_size` — no
    fixed basis matrix, matching the paper's "generic" architecture where the
    interpretation of the basis is left entirely to the learned weights.
    """

    def __init__(self, backcast_size, forecast_size):
        super().__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size

    @property
    def theta_backcast_size(self):
        return self.backcast_size

    @property
    def theta_forecast_size(self):
        return self.forecast_size

    def forward(self, theta_b, theta_f):
        return theta_b, theta_f


class TrendBasis(nn.Module):
    """Polynomial basis of given degree, shared coefficients drive both
    backcast and forecast (captures slow-moving trend)."""

    def __init__(self, degree, backcast_size, forecast_size):
        super().__init__()
        self.degree = degree
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size
        n = degree + 1
        t_b = np.arange(backcast_size, dtype=np.float32) / backcast_size
        t_f = np.arange(forecast_size, dtype=np.float32) / forecast_size
        backcast_basis = np.stack([t_b ** i for i in range(n)], axis=0)   # (n, backcast_size)
        forecast_basis = np.stack([t_f ** i for i in range(n)], axis=0)   # (n, forecast_size)
        self.register_buffer('backcast_basis', torch.tensor(backcast_basis, dtype=torch.float32))
        self.register_buffer('forecast_basis', torch.tensor(forecast_basis, dtype=torch.float32))

    @property
    def theta_backcast_size(self):
        return self.degree + 1

    @property
    def theta_forecast_size(self):
        return self.degree + 1

    def forward(self, theta_b, theta_f):
        backcast = theta_b @ self.backcast_basis
        forecast = theta_f @ self.forecast_basis
        return backcast, forecast


class SeasonalityBasis(nn.Module):
    """Fourier basis (cos/sin harmonics), captures periodic seasonality."""

    def __init__(self, harmonics, backcast_size, forecast_size):
        super().__init__()
        self.harmonics = harmonics
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size

        t_b = np.arange(backcast_size, dtype=np.float32) / backcast_size
        t_f = np.arange(forecast_size, dtype=np.float32) / forecast_size

        def build(t):
            cos = np.stack([np.cos(2 * np.pi * i * t) for i in range(harmonics)], axis=0)
            sin = np.stack([np.sin(2 * np.pi * i * t) for i in range(1, harmonics + 1)], axis=0)
            return np.concatenate([cos, sin], axis=0)  # (2*harmonics, size)

        backcast_basis = build(t_b)
        forecast_basis = build(t_f)
        self.register_buffer('backcast_basis', torch.tensor(backcast_basis, dtype=torch.float32))
        self.register_buffer('forecast_basis', torch.tensor(forecast_basis, dtype=torch.float32))

    @property
    def theta_backcast_size(self):
        return 2 * self.harmonics

    @property
    def theta_forecast_size(self):
        return 2 * self.harmonics

    def forward(self, theta_b, theta_f):
        backcast = theta_b @ self.backcast_basis
        forecast = theta_f @ self.forecast_basis
        return backcast, forecast


# ---------------------------------------------------------------------------
# Block / Stack / Net
# ---------------------------------------------------------------------------

class NBeatsBlock(nn.Module):
    """Standard N-BEATS block: `n_layers` FC+ReLU layers -> basis-expansion
    coefficients (theta_b, theta_f) -> basis layer -> (backcast, forecast)."""

    def __init__(self, input_size, layer_size, n_layers, basis: nn.Module, dropout=0.0):
        super().__init__()
        layers = []
        prev = input_size
        for _ in range(n_layers):
            layers.append(nn.Linear(prev, layer_size))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = layer_size
        self.fc_stack = nn.Sequential(*layers)
        self.theta_b_fc = nn.Linear(layer_size, basis.theta_backcast_size)
        self.theta_f_fc = nn.Linear(layer_size, basis.theta_forecast_size)
        self.basis = basis

    def forward(self, x):
        h = self.fc_stack(x)
        theta_b = self.theta_b_fc(h)
        theta_f = self.theta_f_fc(h)
        return self.basis(theta_b, theta_f)


def _make_basis(stack_type, backcast_size, forecast_size, degree=3, harmonics=None):
    if stack_type == 'generic':
        return GenericBasis(backcast_size, forecast_size)
    elif stack_type == 'trend':
        return TrendBasis(degree, backcast_size, forecast_size)
    elif stack_type == 'seasonality':
        h = harmonics or max(1, forecast_size // 2 - 1)
        return SeasonalityBasis(h, backcast_size, forecast_size)
    raise ValueError(f'unknown stack_type {stack_type!r}')


class NBeatsStack(nn.Module):
    """A sequence of blocks of one basis type. `share_weights=True` reuses a
    single block instance at every position (paper's weight-sharing option);
    otherwise each position gets its own independently-trained block."""

    def __init__(self, stack_type, n_blocks, backcast_size, forecast_size,
                 layer_size, n_layers, degree=3, harmonics=None, dropout=0.0,
                 share_weights=False):
        super().__init__()
        self.stack_type = stack_type
        if share_weights:
            block = NBeatsBlock(
                backcast_size, layer_size, n_layers,
                _make_basis(stack_type, backcast_size, forecast_size, degree, harmonics),
                dropout,
            )
            self.blocks = nn.ModuleList([block] * n_blocks)
        else:
            self.blocks = nn.ModuleList([
                NBeatsBlock(
                    backcast_size, layer_size, n_layers,
                    _make_basis(stack_type, backcast_size, forecast_size, degree, harmonics),
                    dropout,
                )
                for _ in range(n_blocks)
            ])

    def forward(self, residuals):
        """Doubly-residual pass through this stack's blocks.

        Returns (residuals_after_stack, stack_forecast_sum) so the caller can
        keep chaining stacks and separately track this stack's total
        contribution to the forecast (needed for trend/seasonality
        decomposition plots).
        """
        stack_forecast = 0.0
        for block in self.blocks:
            backcast, forecast = block(residuals)
            residuals = residuals - backcast
            stack_forecast = stack_forecast + forecast
        return residuals, stack_forecast


class NBeatsNet(nn.Module):
    """Full N-BEATS network: ordered list of stacks, doubly-residual across
    the whole stack sequence (a stack's output residual feeds the next
    stack, exactly like blocks within a stack).

    `stack_types`: list of 'generic' | 'trend' | 'seasonality', e.g.
      - generic architecture:        ['generic', 'generic']
      - interpretable architecture:  ['trend', 'seasonality']
      - mixed:                       ['trend', 'seasonality', 'generic']
    """

    def __init__(self, backcast_size, forecast_size, stack_types=('generic', 'generic'),
                 n_blocks_per_stack=3, layer_size=256, n_layers=4, degree=3,
                 harmonics=None, dropout=0.0, share_weights=False):
        super().__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size
        self.stack_types = list(stack_types)
        self.stacks = nn.ModuleList([
            NBeatsStack(
                stack_type, n_blocks_per_stack, backcast_size, forecast_size,
                layer_size, n_layers, degree=degree, harmonics=harmonics,
                dropout=dropout, share_weights=share_weights,
            )
            for stack_type in stack_types
        ])

    def forward(self, x, return_decomposition=False):
        residuals = x
        forecast = torch.zeros(x.shape[0], self.forecast_size, device=x.device, dtype=x.dtype)
        stack_forecasts = {}
        for i, stack in enumerate(self.stacks):
            residuals, stack_forecast = stack(residuals)
            forecast = forecast + stack_forecast
            key = f'{i}_{stack.stack_type}'
            stack_forecasts[key] = stack_forecasts.get(key, 0.0) + stack_forecast
        if return_decomposition:
            return forecast, residuals, stack_forecasts
        return forecast, residuals


# ---------------------------------------------------------------------------
# Ensembling
# ---------------------------------------------------------------------------

class NBeatsEnsemble:
    """Averages forecasts (in original Weekly_Sales units) across trained
    NBeatsNet members, per the paper's ensembling recipe: members differ by
    lookback-window multiple of the horizon and/or training loss."""

    def __init__(self, members):
        """members: list of dicts with keys 'model' (NBeatsNet, eval mode),
        'lookback_multiple', 'loss_name', 'scaler_fn' (callable(window)->scale)."""
        self.members = members

    def predict(self, series_by_member):
        """series_by_member: dict lookback_multiple -> (x_tensor, scale_tensor)
        already windowed/scaled per member's own lookback length.
        Returns the mean forecast across members, in original units."""
        preds = []
        with torch.no_grad():
            for m in self.members:
                x, scale = series_by_member[m['lookback_multiple']]
                forecast, _ = m['model'](x)
                preds.append((forecast * scale).cpu().numpy())
        return np.mean(preds, axis=0)
