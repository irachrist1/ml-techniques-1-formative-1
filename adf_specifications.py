"""Record the ADF unit-root test under more than one lag specification.

The report argues that the ADF result on this series is specification sensitive:
a short lag ceiling cannot span the 144-step daily cycle, and the p-value moves
by many orders of magnitude when the ceiling is widened to admit one. That claim
needs both numbers on record, not just the one the study settled on.

This runs from `results/selected_series.csv`, which is committed, so the
comparison can be rebuilt without the 20 GB of raw daily files. The widest
specification is asserted against the value already stored in `eda_summary.json`,
so this script cannot quietly disagree with the analysis it is annotating.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from analyze import ADF_MAX_LAG, longest_run
from protocol import INTERVALS_PER_DAY, TRAIN_END

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'

#: 30 was the original ceiling; it is below one daily cycle (144 steps), which is
#: exactly why it is kept here as the contrasting specification.
MAX_LAGS = [30, ADF_MAX_LAG]


def specification(segment: np.ndarray, maxlag: int) -> dict:
    """Augmented Dickey-Fuller with a constant, AIC choosing lags up to `maxlag`."""
    statistic, pvalue, lags, nobs, critical, _ = adfuller(segment, maxlag=maxlag, autolag='AIC')
    return {
        'max_lag_allowed': maxlag,
        'spans_a_full_daily_cycle': maxlag >= INTERVALS_PER_DAY,
        'lags_selected': int(lags),
        'observations_used': int(nobs),
        'statistic': float(statistic),
        'pvalue': float(pvalue),
        'critical_values': critical,
    }


def main() -> None:
    eda = json.loads((RESULTS / 'eda_summary.json').read_text())
    frame = pd.read_csv(RESULTS / 'selected_series.csv', index_col='timestamp_ms')
    train = frame.loc[frame.index < TRAIN_END, str(eda['top3'][0])]
    segment = longest_run(train.to_numpy())
    assert len(segment) == eda['adf']['segment_n'], 'Segment differs from the recorded analysis'

    results = [specification(segment, maxlag) for maxlag in MAX_LAGS]
    widest = next(r for r in results if r['max_lag_allowed'] == ADF_MAX_LAG)
    np.testing.assert_allclose(widest['pvalue'], eda['adf']['pvalue'], rtol=1e-9)
    assert widest['lags_selected'] == eda['adf']['lags']

    report = {
        'area': eda['top3'][0],
        'segment_n': len(segment),
        'scope': 'Longest fully observed training-period run for the busiest area.',
        'specifications': results,
        'note': ('Both specifications reject the unit-root null under their own assumptions. '
                 'Neither proves strict stationarity or removes seasonal structure.'),
    }
    (RESULTS / 'adf_specifications.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
