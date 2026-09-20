"""Past-only window construction and the train/validation/test split.

This is the only place the model inputs are built. Two guarantees live here and
are covered by `test_features.py`:

1. The normalization statistics are fitted on training values only, so nothing
   after `TRAIN_END` can influence the scale of any split.
2. Every candidate model is scored on identical target timestamps, because
   eligibility requires a complete `MAX_HISTORY`-step history regardless of how
   many of those steps the model itself consumes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from protocol import MAX_HISTORY, STEP_MS, TEST_END, TEST_START, TRAIN_END

# One split's arrays. Keys are documented in `windows` below.
Split = dict[str, np.ndarray]
Scaler = dict[str, float | int]


def windows(frame: pd.DataFrame, area: int, lookback: int) -> tuple[dict[str, Split], Scaler]:
    """Build past-only windows on the common eligibility mask for one area.

    Args:
        frame: Observations indexed by epoch milliseconds on a complete, ordered
            10-minute grid, with one column per area named by its square ID.
        area: Square ID to model. Selects the column `str(area)`.
        lookback: Steps of history the model consumes, between 1 and MAX_HISTORY.

    Returns:
        A ``(data, scaler)`` pair. ``data`` maps each of ``train``/``validation``/
        ``test`` to arrays:

        - ``x``: normalized history, shape ``(n, lookback, 1)``, float32.
        - ``delta``: normalized correction to persistence, the regression target.
        - ``y``: the raw observed target value.
        - ``last``: the observation one step before the target (persistence baseline).
        - ``seasonal``: the observation one day before the target (seasonal baseline).
        - ``timestamps``: the target's epoch milliseconds.

        ``scaler`` records the training mean and standard deviation actually used,
        plus the boundary they were fitted up to, so a saved run can be replayed.

    Raises:
        ValueError: If the lookback is out of range, the time grid has a gap, the
            training scale is degenerate, or any split ends up empty.
    """
    if not 1 <= lookback <= MAX_HISTORY:
        raise ValueError('lookback must be between 1 and 144')
    timestamps = frame.index.to_numpy(dtype=np.int64)
    if not np.all(np.diff(timestamps) == STEP_MS):
        raise ValueError('Input requires a complete ordered 10-minute time grid')
    values = frame[str(area)].to_numpy(dtype=np.float64)

    # Fitted on training rows only; everything downstream reuses these two numbers.
    train_values = values[timestamps < TRAIN_END]
    mean = float(np.nanmean(train_values))
    std = float(np.nanstd(train_values))
    if not np.isfinite(std) or std <= 0:
        raise ValueError('Training scale must be finite and positive')

    # A target is eligible only when the full MAX_HISTORY window plus the target
    # itself is observed. Shorter lookbacks slice this same mask, so a model with
    # less history cannot win by being scored on easier, more complete stretches.
    all_windows = np.lib.stride_tricks.sliding_window_view(values, MAX_HISTORY + 1)
    valid = np.isfinite(all_windows).all(axis=1)
    target_times = timestamps[MAX_HISTORY:][valid]
    history = all_windows[valid, :-1][:, -lookback:]
    target = all_windows[valid, -1]

    x = ((history - mean) / std).astype(np.float32)[..., None]
    delta = ((target - history[:, -1]) / std).astype(np.float32)
    masks = {
        'train': target_times < TRAIN_END,
        'validation': (target_times >= TRAIN_END) & (target_times < TEST_START),
        'test': (target_times >= TEST_START) & (target_times < TEST_END),
    }
    data = {
        key: {
            'x': x[mask],
            'delta': delta[mask],
            'y': target[mask],
            'last': history[mask, -1],
            'seasonal': all_windows[valid, 0][mask],
            'timestamps': target_times[mask],
        }
        for key, mask in masks.items()
    }
    if any(len(d['y']) == 0 for d in data.values()):
        raise ValueError('Empty split after missing-window exclusions')
    scaler = {'mean': mean, 'std': std, 'fit_end_exclusive': TRAIN_END, 'common_history': MAX_HISTORY}
    return data, scaler


def invert(normalized: np.ndarray, scaler: Scaler) -> np.ndarray:
    """Map normalized history back to activity units; the exact inverse of `windows`."""
    return np.asarray(normalized) * scaler['std'] + scaler['mean']


def to_forecast(last: np.ndarray, correction: np.ndarray, scaler: Scaler) -> np.ndarray:
    """Turn a normalized correction into a non-negative activity-unit forecast.

    Activity counts cannot be negative, so the correction is applied to persistence
    and then clipped at zero. This is the only place that arithmetic is written.
    """
    return np.maximum(0, np.asarray(last) + scaler['std'] * np.asarray(correction))
