"""Small, dependency-free evaluation primitives for one-step traffic forecasts."""

import math
from datetime import datetime
from zoneinfo import ZoneInfo


def score(actual, predicted) -> dict:
    """MAPE excludes zero targets; report their count instead of hiding them."""
    actual, predicted = list(actual), list(predicted)
    if not actual or len(actual) != len(predicted):
        raise ValueError('Expected equal, nonempty actual and predicted sequences')
    if not all(math.isfinite(v) for v in actual + predicted):
        raise ValueError('Missing/nonfinite values must be handled explicitly')
    errors = [abs(a - p) for a, p in zip(actual, predicted, strict=True)]
    percentages = [e / abs(a) for a, e in zip(actual, errors, strict=True) if a != 0]
    return {
        'n': len(actual),
        'mae': sum(errors) / len(errors),
        'rmse': math.sqrt(sum(e * e for e in errors) / len(errors)),
        'mape_percent_nonzero': 100 * sum(percentages) / len(percentages) if percentages else None,
        'mape_excluded_zero_targets': len(actual) - len(percentages),
    }


def milan_midnight_ms(date: str) -> int:
    """Explicit local-calendar boundaries; raw epoch timestamps remain UTC."""
    return int(datetime.fromisoformat(date).replace(tzinfo=ZoneInfo('Europe/Rome')).timestamp() * 1000)
