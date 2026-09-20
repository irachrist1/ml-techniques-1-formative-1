"""Repeated, isolated-process full-day memory comparison with recorded provenance."""
import argparse
import json
import platform
import resource
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation import milan_midnight_ms
from prepare_day import NAMES, aggregate_chunk, file_md5, read_projected

ROOT = Path(__file__).resolve().parent

# Recorded verbatim in the output so the numbers are never read without their caveats.
SCOPE = ('Complete November 1 file; separate process per sample; alternating order; filesystem cache '
         'and background load uncontrolled. Peak before output serialization; wall time excludes '
         'imports, checksum and serialization. Parser transient allocations and sum/count arrays are '
         'included in peak RSS.')


def worker(mode: str, path: Path, directory: Path) -> None:
    start = time.perf_counter()
    sums = np.zeros((144, 10000))
    counts = np.zeros((144, 10000), dtype=np.uint32)
    start_ms = milan_midnight_ms('2013-11-01')
    if mode == 'baseline':
        frame = pd.read_csv(path, sep='\t', header=None, names=NAMES)
        dataframe_bytes = int(frame.memory_usage(deep=True).sum())
        grouped = frame.groupby(['timestamp', 'square']).internet.agg(['sum', 'count'])
        slots = (grouped.index.get_level_values('timestamp').to_numpy() - start_ms) // 600000
        squares = grouped.index.get_level_values('square').to_numpy() - 1
        sums[slots, squares] = grouped['sum'].to_numpy()
        counts[slots, squares] = grouped['count'].to_numpy()
    else:
        dataframe_bytes = 0
        for frame in read_projected(path, chunksize=100000):
            dataframe_bytes = max(dataframe_bytes, int(frame.memory_usage(deep=True).sum()))
            aggregate_chunk(frame, sums, counts, start_ms)
    sums[counts == 0] = np.nan
    seconds = time.perf_counter() - start
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == 'Darwin' else 1024)
    np.savez(directory / f'{mode}.npz', internet=sums, observed_rows=counts)
    print(json.dumps({'mode': mode, 'wall_seconds': seconds, 'peak_rss_bytes': int(peak),
                      'max_dataframe_bytes': dataframe_bytes}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=['baseline', 'chunked'])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/memory_benchmark_recheck.json')
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    directory = ROOT / 'data/benchmark'
    directory.mkdir(parents=True, exist_ok=True)
    path = ROOT / 'data/raw/sms-call-internet-mi-2013-11-01.txt'
    if args.worker:
        return worker(args.worker, path, directory)
    samples = []
    differences = []
    for repeat in range(args.repeats):
        # Alternate order; this does not control or eliminate filesystem caching.
        order = ['baseline', 'chunked'] if repeat % 2 == 0 else ['chunked', 'baseline']
        for mode in order:
            process = subprocess.run([sys.executable, __file__, '--worker', mode],
                                     capture_output=True, text=True, check=True)
            samples.append({'repeat': repeat + 1, **json.loads(process.stdout)})
        with np.load(directory / 'baseline.npz') as a, np.load(directory / 'chunked.npz') as b:
            np.testing.assert_array_equal(np.isnan(a['internet']), np.isnan(b['internet']))
            np.testing.assert_array_equal(a['observed_rows'], b['observed_rows'])
            np.testing.assert_allclose(a['internet'], b['internet'], rtol=1e-12, atol=1e-9, equal_nan=True)
            differences.append(float(np.nanmax(abs(a['internet'] - b['internet']))))
    runs = []
    for mode in ['baseline', 'chunked']:
        group = [s for s in samples if s['mode'] == mode]
        row = {'mode': mode}
        for key in ['wall_seconds', 'peak_rss_bytes', 'max_dataframe_bytes']:
            values = [s[key] for s in group]
            row[key] = float(np.median(values))
            row[key + '_min'] = min(values)
            row[key + '_max'] = max(values)
        runs.append(row)
    report = {
        'recorded_utc': datetime.now(UTC).isoformat(), 'input': path.name,
        'input_bytes': path.stat().st_size, 'input_md5': file_md5(path),
        'platform': platform.platform(), 'machine': platform.machine(), 'python': platform.python_version(),
        'numpy': np.__version__, 'pandas': pd.__version__, 'repeats_per_method': args.repeats,
        'scope': SCOPE,
        'runs': runs, 'samples': samples,
        'peak_rss_reduction_percent': 100 * (1 - runs[1]['peak_rss_bytes'] / runs[0]['peak_rss_bytes']),
        'same_missing_mask': True, 'same_observation_counts': True,
        'max_abs_aggregation_difference': max(differences),
    }
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
