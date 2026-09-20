"""Replay one saved one-step forecast using only observations before its target.

This is the demonstration entry point: it loads weights that were written during
training and rebuilds a single forecast from history alone. It refuses to guess.
If any of the required preceding observations is missing, off-grid or duplicated,
it raises rather than silently forecasting from a shorter or misaligned window.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from features import to_forecast
from protocol import MODELS, STEP_MS

ROOT = Path(__file__).resolve().parent

# pandas Timestamp.value is nanoseconds; the series grid is milliseconds.
NANOSECONDS_PER_MS = 1_000_000


def predict(run: str | Path, kind: str, timestamp: str) -> float:
    """Rebuild the forecast one saved model would make for `timestamp`.

    Args:
        run: A run directory written by `experiments.run`.
        kind: Model name within that run.
        timestamp: Target instant, with an explicit timezone offset. It must land
            exactly on the 10-minute grid.

    Returns:
        The forecast in activity units.

    Raises:
        ValueError: If the timestamp is naive, off-grid, or the required history
            is incomplete, non-finite or not present exactly once.
    """
    run = Path(run)
    summary = json.loads((run / 'summary.json').read_text())
    entry = next(r for r in summary['runs'] if r['model'] == kind)
    scaler = entry['scaler']
    lookback = entry['config']['lookback']
    area = entry['area']

    frame = pd.read_csv(ROOT / 'results/selected_series.csv', index_col='timestamp_ms')
    target = pd.Timestamp(timestamp)
    if target.tz is None:
        raise ValueError('Include an explicit timezone offset in the target timestamp')
    if target.value % (STEP_MS * NANOSECONDS_PER_MS):
        raise ValueError('Target must align to the 10-minute grid')
    milliseconds = target.value // NANOSECONDS_PER_MS

    # Strictly earlier than the target, and every intervening step present exactly
    # once: a duplicate row must not be allowed to paper over a missing interval.
    selected = frame.loc[(frame.index >= milliseconds - lookback * STEP_MS)
                         & (frame.index < milliseconds), str(area)]
    expected = np.arange(milliseconds - lookback * STEP_MS, milliseconds, STEP_MS, dtype=np.int64)
    if not np.array_equal(selected.index.to_numpy(), expected):
        raise ValueError('Required history must contain each preceding 10-minute timestamp exactly once')
    history = selected.to_numpy()
    if len(history) != lookback or not np.isfinite(history).all():
        raise ValueError('Required observed history is incomplete')

    x = ((history - scaler['mean']) / scaler['std']).astype(np.float32)[None, :, None]
    if kind == 'RidgeAR':
        with np.load(run / 'RidgeAR.npz') as model:
            correction = float(x[0, :, 0] @ model['coef'] + model['intercept'])
    else:
        from models import tensorflow
        tf = tensorflow()
        model = tf.keras.models.load_model(run / f'{kind}.keras', compile=False)
        correction = float(model(x, training=False).numpy().reshape(-1)[0])
    return float(to_forecast(history[-1], correction, scaler))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--model', choices=list(MODELS), required=True)
    parser.add_argument('--timestamp', required=True)
    args = parser.parse_args()
    print(json.dumps({'model': args.model, 'target': args.timestamp,
                      'prediction': predict(args.run, args.model, args.timestamp)}))
