"""Torch training losses for N-BEATS.

`wmae_loss` mirrors utils.metrics.wmae exactly (holiday weeks weighted 5x) so
the training objective matches the competition metric; `mae_loss` and
`smape_loss` are provided for comparison, per the Optuna search space.
`asym_wmae_loss` additionally penalizes UNDERprediction on holiday weeks --
added to address a documented pattern where every architecture badly
underpredicts holiday-week sales spikes (see reports/nbeats_finalize_results.json:
local_test_wmae_holiday ~1.8x worse than non-holiday), without adding any new
network input (still a pure univariate model -- this only reshapes the loss).
"""

import torch


def wmae_loss(y_pred, y_true, is_holiday):
    weights = torch.where(is_holiday.bool(), 5.0, 1.0)
    return torch.sum(weights * torch.abs(y_pred - y_true)) / torch.sum(weights)


def mae_loss(y_pred, y_true, is_holiday=None):
    return torch.mean(torch.abs(y_pred - y_true))


def smape_loss(y_pred, y_true, is_holiday=None, eps=1e-8):
    numerator = 2.0 * torch.abs(y_pred - y_true)
    denominator = torch.abs(y_pred) + torch.abs(y_true) + eps
    return torch.mean(numerator / denominator)


def asym_wmae_loss(y_pred, y_true, is_holiday, under_penalty=2.0):
    """Same 5x holiday weighting as `wmae_loss`, plus an extra
    `under_penalty` multiplier specifically on holiday-week terms where the
    model underpredicts (y_pred < y_true) -- directly targets the
    underprediction-on-spikes pattern rather than just weighting holiday
    weeks symmetrically."""
    holiday_mask = is_holiday.bool()
    weights = torch.where(holiday_mask, 5.0, 1.0)
    underpredicted = holiday_mask & (y_pred < y_true)
    weights = weights * torch.where(underpredicted, under_penalty, 1.0)
    return torch.sum(weights * torch.abs(y_pred - y_true)) / torch.sum(weights)


LOSS_FUNCTIONS = {
    'wmae': wmae_loss,
    'mae': mae_loss,
    'smape': smape_loss,
    'asym_wmae': asym_wmae_loss,
}
