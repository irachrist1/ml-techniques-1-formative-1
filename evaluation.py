"""Small, dependency-free evaluation primitives for one-step traffic forecasts."""

import math
from datetime import datetime
from zoneinfo import ZoneInfo


def score(actual, predicted):
    """MAPE excludes zero targets; report their count instead of hiding them."""
    actual, predicted = list(actual), list(predicted)
    if not actual or len(actual) != len(predicted):
        raise ValueError('Expected equal, nonempty actual and predicted sequences')
    if not all(math.isfinite(v) for v in actual + predicted):
        raise ValueError('Missing/nonfinite values must be handled explicitly')
    errors = [abs(a - p) for a, p in zip(actual, predicted)]
    percentages = [e / abs(a) for a, e in zip(actual, errors) if a != 0]
    return {
        'n': len(actual),
        'mae': sum(errors) / len(errors),
        'rmse': math.sqrt(sum(e * e for e in errors) / len(errors)),
        'mape_percent_nonzero': 100 * sum(percentages) / len(percentages) if percentages else None,
        'mape_excluded_zero_targets': len(actual) - len(percentages),
    }


def rolling_examples(timestamps, values, lookback, start, end, step_ms=600_000):
    """Yield history and next target within [start,end), never future inputs.

    Bounds and timestamps are Unix milliseconds. Histories may cross a split
    boundary, but targets cannot. Missing values or time gaps omit the window.
    This implements rolling one-step evaluation using already observed history,
    not recursive forecasting of the whole test week from a single origin.
    """
    if len(timestamps) != len(values) or lookback < 1 or start >= end or step_ms <= 0:
        raise ValueError('Invalid series, window, or interval')
    if any(b <= a for a, b in zip(timestamps, timestamps[1:])):
        raise ValueError('Timestamps must be unique and increasing')
    for i in range(lookback, len(values)):
        if not start <= timestamps[i] < end:
            continue
        times = timestamps[i-lookback:i+1]
        window = values[i-lookback:i+1]
        if any(b-a != step_ms for a, b in zip(times, times[1:])):
            continue
        if any(v is None or not math.isfinite(v) for v in window):
            continue
        yield timestamps[i], list(window[:-1]), window[-1]


def milan_midnight_ms(date):
    """Explicit local-calendar boundaries; raw epoch timestamps remain UTC."""
    return int(datetime.fromisoformat(date).replace(tzinfo=ZoneInfo('Europe/Rome')).timestamp() * 1000)
