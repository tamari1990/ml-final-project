"""Competition metric: Weighted Mean Absolute Error (holiday weeks weighted 5x)."""

import numpy as np


def wmae(y_true, y_pred, is_holiday):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    weights = np.where(np.asarray(is_holiday, dtype=bool), 5, 1)
    return np.average(np.abs(y_true - y_pred), weights=weights)
