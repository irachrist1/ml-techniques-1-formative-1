"""Full-period ranking and training-period characterization from verified daily arrays."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import STL

from evaluation import milan_midnight_ms

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
FIGURES = RESULTS / 'figures'
DATES = pd.date_range('2013-11-01', '2013-12-31', freq='D').strftime('%Y-%m-%d').tolist()
# One full day is 144 ten-minute steps, so an ADF lag ceiling below that cannot
# span a daily cycle. Let AIC choose inside a window that can.
ADF_MAX_LAG = 168
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.dpi': 150})


def longest_run(values):
    valid = np.isfinite(values)
    edges = np.diff(np.r_[False, valid, False].astype(int))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    if not len(starts):
        raise ValueError('No observed segment')
    i = np.argmax(ends - starts)
    return values[starts[i]:ends[i]]


def save(fig, name):
    fig.savefig(FIGURES / name, bbox_inches='tight')
    plt.close(fig)


def load_days():
    """Accumulate full-period totals and observation counts one day at a time."""
    totals = np.zeros(10000)
    observed = np.zeros(10000, dtype=int)
    audits = []
    for date in DATES:
        path = ROOT / 'data/processed' / f'{date}.npz'
        if not path.exists():
            raise FileNotFoundError(f'Missing required day: {date}')
        with np.load(path) as day:
            expected = milan_midnight_ms(date) + np.arange(144) * 600000
            if day['internet'].shape != (144, 10000) or not np.array_equal(day['timestamps'], expected):
                raise ValueError(f'Invalid time grid {date}')
            values = day['internet']
            totals += np.nansum(values, axis=0)
            observed += np.isfinite(values).sum(axis=0)
        audits.append(json.loads((ROOT / 'data/processed' / f'{date}-audit.json').read_text()))
    return totals, observed, audits


def coverage_robustness(rank, top_n=3):
    """Sensitivity to imputing each area's observed mean into its missing bins.

    This is an assumption-based scenario, not an upper bound: unobserved
    activity could be larger or smaller than the observed mean.
    """
    partial = rank[rank.coverage < 1].copy()
    cutoff = float(rank.total_internet_activity.iloc[top_n - 1])
    if partial.empty:
        return {'areas_below_full_coverage': 0, 'top3_cutoff_total': cutoff,
                'largest_rescaled_partial_total': 0.0, 'top3_changes_under_observed_mean_imputation': False}
    rescaled = partial.total_internet_activity / partial.coverage
    return {
        'areas_below_full_coverage': int(len(partial)),
        'minimum_coverage': float(rank.coverage.min()),
        'top3_cutoff_total': cutoff,
        'largest_rescaled_partial_total': float(rescaled.max()),
        'assumption': 'Missing bins have the same mean as observed bins within each area',
        'is_upper_bound': False,
        'top3_changes_under_observed_mean_imputation': bool(rescaled.max() > cutoff),
    }


def weekly_profile(series, local_index):
    """Mean weekday and weekend level, for comparing areas on evidence."""
    weekend = local_index.dayofweek >= 5
    values = series.to_numpy()
    weekday_mean = float(np.nanmean(values[~weekend]))
    weekend_mean = float(np.nanmean(values[weekend]))
    return weekday_mean, weekend_mean


def main():
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    totals, observed, audits = load_days()

    rank = pd.DataFrame({'square': np.arange(1, 10001), 'total_internet_activity': totals,
                         'observed_intervals': observed, 'coverage': observed / 8784})
    rank = rank.sort_values(['total_internet_activity', 'square'], ascending=[False, True])
    rank.to_csv(RESULTS / 'area_ranking.csv', index=False)

    top3 = rank.square.head(3).astype(int).tolist()
    areas = list(dict.fromkeys(top3 + [4159, 4556]))

    pieces = []
    for date in DATES:
        with np.load(ROOT / 'data/processed' / f'{date}.npz') as day:
            pieces.append(pd.DataFrame(day['internet'][:, np.array(areas) - 1],
                                       columns=[str(a) for a in areas], index=day['timestamps']))
    frame = pd.concat(pieces)
    frame.index.name = 'timestamp_ms'
    frame.to_csv(RESULTS / 'selected_series.csv')

    local = pd.to_datetime(frame.index, unit='ms', utc=True).tz_convert('Europe/Rome')
    train = frame.iloc[np.asarray(frame.index) < milan_midnight_ms('2013-12-09')]
    local_train = local[:len(train)]
    top = train[str(top3[0])]

    acf = [top.corr(top.shift(lag)) for lag in range(1009)]
    lags = [1, 6, 36, 144, 288, 1008]
    # Rough 95% band for "no correlation" under white noise, for reading the plot only.
    acf_band = float(1.96 / np.sqrt(len(top)))

    segment = longest_run(top.to_numpy())
    adf = adfuller(segment, maxlag=ADF_MAX_LAG, autolag='AIC')
    diff_adf = adfuller(np.diff(segment), maxlag=ADF_MAX_LAG, autolag='AIC')

    # Interpolation is used for descriptive STL on training dates only, never for model targets.
    filled = top.interpolate(limit_direction='both').to_numpy()
    decomposition = STL(filled, period=144, robust=True).fit()
    denom = np.var(decomposition.resid + decomposition.seasonal)
    strength = max(0, 1 - np.var(decomposition.resid) / denom) if denom else 0

    stats = []
    for area in areas:
        x = frame[str(area)]
        t = train[str(area)]
        values = t.to_numpy()
        finite = values[np.isfinite(values)]
        weekday_mean, weekend_mean = weekly_profile(t, local_train)
        stats.append({
            'square': area, 'full_total': float(x.sum()), 'full_missing': int(x.isna().sum()),
            'train_mean': float(t.mean()), 'train_std': float(t.std()),
            'train_cv': float(t.std() / t.mean()), 'train_median': float(t.median()),
            'train_p99': float(np.quantile(finite, .99)), 'train_max': float(t.max()),
            'train_zero_count': int((t == 0).sum()),
            'train_acf_1': float(t.corr(t.shift(1))), 'train_acf_144': float(t.corr(t.shift(144))),
            'train_weekday_mean': weekday_mean, 'train_weekend_mean': weekend_mean,
            'weekend_over_weekday': weekend_mean / weekday_mean,
        })
    pd.DataFrame(stats).to_csv(RESULTS / 'area_characteristics.csv', index=False)
    pd.DataFrame(audits).drop(columns=['scope', 'limitations'], errors='ignore').to_csv(
        RESULTS / 'daily_audits.csv', index=False)

    summary = {
        'top3': top3, 'areas': areas, 'days': 61, 'intervals_per_area': 8784,
        'raw_bytes': sum(a['input_bytes'] for a in audits),
        'raw_rows': sum(a['raw_rows'] for a in audits),
        'missing_square_time_bins': int(8784 * 10000 - observed.sum()),
        'full_activity_sum': float(totals.sum()),
        'top3_share': float(rank.total_internet_activity.head(3).sum() / totals.sum()),
        'top1_percent_share': float(rank.total_internet_activity.head(100).sum() / totals.sum()),
        'coverage_min': float(rank.coverage.min()),
        'coverage_median': float(rank.coverage.median()),
        'coverage_robustness': coverage_robustness(rank),
        'training_end_exclusive': '2013-12-09 Europe/Rome',
        'top_area_acf': dict(zip(map(str, lags), map(lambda i: float(acf[i]), lags))),
        'acf_white_noise_band_95': acf_band,
        'adf': {'segment_n': len(segment), 'statistic': float(adf[0]), 'pvalue': float(adf[1]),
                'lags': int(adf[2]), 'max_lag_allowed': ADF_MAX_LAG, 'critical_values': adf[4]},
        'difference_adf': {'statistic': float(diff_adf[0]), 'pvalue': float(diff_adf[1]),
                           'lags': int(diff_adf[2]), 'max_lag_allowed': ADF_MAX_LAG},
        'stl_daily_strength': float(strength), 'stl_interpolated_points': int(top.isna().sum()),
        'area_stats': stats,
    }
    (RESULTS / 'eda_summary.json').write_text(json.dumps(summary, indent=2) + '\n')

    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.hist(totals / 1e6, bins=70, color='#14645a')
    ax.set(xlabel='Total Internet activity (millions of activity units)',
           ylabel='Number of geographical areas',
           title='Milan: total Internet activity across 10,000 areas, Nov-Dec 2013')
    save(fig, 'traffic_distribution.png')

    fig, axs = plt.subplots(len(areas), 1, figsize=(10, 10), sharex=True)
    for ax, area in zip(axs, areas):
        ax.plot(local[:2016], frame[str(area)].iloc[:2016], lw=.7, color='#14645a')
        ax.set_ylabel(f'Area {area}\nActivity')
        ax.grid(alpha=.15)
    axs[0].set_title('First two weeks: the top three areas plus 4159 and 4556')
    axs[-1].set_xlabel('Date (Europe/Rome), 1-14 November 2013')
    fig.autofmt_xdate()
    save(fig, 'first_two_weeks.png')

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.5))
    axs[0].plot(np.arange(1009) / 144, acf, color='#14645a')
    for sign in (1, -1):
        axs[0].axhline(sign * acf_band, color='#a45b32', lw=.8, ls='--')
    axs[0].axhline(0, color='#999999', lw=.6)
    axs[0].set(xlabel='Lag (days)', ylabel='Pairwise autocorrelation',
               title=f'Area {top3[0]}: training-period dependence')
    axs[0].text(.02, .04, f'dashed: +/-1.96/sqrt(n) = {acf_band:.3f}', transform=axs[0].transAxes, fontsize=7,
                color='#a45b32')
    profile = pd.DataFrame({'activity': top.to_numpy(), 'slot': local_train.hour * 6 + local_train.minute // 10,
                            'weekend': local_train.dayofweek >= 5})
    for label, weekend, color in [('Weekdays', False, '#14645a'), ('Weekends', True, '#a45b32')]:
        g = profile[profile.weekend == weekend].groupby('slot').activity.mean()
        axs[1].plot(g.index / 6, g, label=label, color=color)
    axs[1].set(xlabel='Local hour', ylabel='Mean Internet activity', title='Mean daily profile, training dates')
    axs[1].legend()
    fig.tight_layout()
    save(fig, 'temporal_analysis.png')

    fig, axs = plt.subplots(4, 1, figsize=(9, 7), sharex=True)
    for ax, data, label in zip(axs, [filled, decomposition.trend, decomposition.seasonal, decomposition.resid],
                               ['Observed', 'Trend', 'Daily seasonal', 'Residual']):
        ax.plot(local_train, data, lw=.6, color='#14645a')
        ax.set_ylabel(label)
    axs[0].set_title(f'Area {top3[0]}: robust STL, period = 144 intervals (training only)')
    axs[-1].set_xlabel('Date (Europe/Rome); missing training values interpolated for this plot only')
    fig.autofmt_xdate()
    save(fig, 'stl_training.png')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
