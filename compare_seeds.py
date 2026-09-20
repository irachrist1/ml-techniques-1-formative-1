"""Ask whether each area's model ranking survives seed-to-seed variation.

`summarize_results.py` reports a winner per area. That winner is chosen on a
single reference seed, while the neural models move by several RMSE units when
only the seed changes. This script asks the obvious follow-up question: is the
gap between the best and second-best model larger than the spread the seed alone
produces?

It reads only `results/test_metrics_all_seeds.csv`, which is written from saved
predictions. Nothing here retrains, and nothing here changes a reported metric;
it is an additional diagnostic computed from results that already exist.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from protocol import REFERENCE_SEED

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'


def separation(metrics: pd.DataFrame) -> pd.DataFrame:
    """Compare each area's best-versus-second gap against its seed spread.

    Two deliberately conservative criteria are reported side by side:

    - ``ranges_overlap``: do the observed [min, max] RMSE ranges of the top two
      models overlap across seeds? With three seeds this is a crude but entirely
      assumption-free check.
    - ``gap_exceeds_combined_sd``: is the gap in means larger than the sum of the
      two sample standard deviations? Three seeds is far too few to justify a
      confidence interval, so this is a rule of thumb, not a significance test.
    """
    rows = []
    for area, group in metrics.groupby('area'):
        stats = group.groupby('model').rmse.agg(['mean', 'std', 'min', 'max'])
        ordered = stats.sort_values('mean')
        best, second = ordered.index[0], ordered.index[1]
        gap = float(ordered['mean'].iloc[1] - ordered['mean'].iloc[0])
        reference = group[group.seed == REFERENCE_SEED]
        rows.append({
            'area': area,
            'best_by_seed_mean': best,
            'second_by_seed_mean': second,
            'best_mean_rmse': float(ordered['mean'].iloc[0]),
            'second_mean_rmse': float(ordered['mean'].iloc[1]),
            'gap_rmse': gap,
            'best_seed_sd': float(ordered['std'].iloc[0]),
            'second_seed_sd': float(ordered['std'].iloc[1]),
            'largest_seed_sd_in_area': float(stats['std'].max()),
            'ranges_overlap': bool(ordered['min'].iloc[1] <= ordered['max'].iloc[0]),
            'gap_exceeds_combined_sd': bool(gap > ordered['std'].iloc[0] + ordered['std'].iloc[1]),
            'reference_seed_best': reference.loc[reference.rmse.idxmin(), 'model'],
            'reference_seed_agrees_with_mean': bool(reference.loc[reference.rmse.idxmin(), 'model'] == best),
        })
    return pd.DataFrame(rows)


def main() -> None:
    metrics = pd.read_csv(RESULTS / 'test_metrics_all_seeds.csv')
    table = separation(metrics)
    table.to_csv(RESULTS / 'seed_separation.csv', index=False)
    summary = {
        'areas': int(len(table)),
        'areas_where_ranges_overlap': int(table.ranges_overlap.sum()),
        'areas_where_gap_exceeds_combined_sd': int(table.gap_exceeds_combined_sd.sum()),
        'areas_where_reference_seed_disagrees_with_seed_mean':
            int((~table.reference_seed_agrees_with_mean).sum()),
        'largest_seed_sd_observed': float(table.largest_seed_sd_in_area.max()),
        'smallest_gap_rmse': float(table.gap_rmse.min()),
        'interpretation': ('Where the top two models\' seed ranges overlap, that area\'s '
                           'ranking is not separated by this evidence and should be read as '
                           'one week at one seed, not as a general ordering.'),
        'caveat': 'Three seeds measure optimization variability only, not forecast uncertainty.',
    }
    (RESULTS / 'seed_separation.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(table.to_string(index=False))
    print()
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
