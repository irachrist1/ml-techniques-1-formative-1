"""Validate and aggregate one publisher daily file without loading it all.

Country-code records are summed into a (time, square) matrix. Missing Internet
fields do not become measurements of zero. Output covers all 10,000 squares.
"""
import argparse
import hashlib
import json
import platform
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation import milan_midnight_ms

NAMES = ['square', 'timestamp', 'country', 'sms_in', 'sms_out', 'call_in', 'call_out', 'internet']
DTYPES = {'square': 'int32', 'timestamp': 'int64', 'internet': 'float64'}
STEP = 600_000


def read_projected(path, **kwargs):
    return pd.read_csv(path, sep='\t', header=None, names=NAMES,
                       usecols=['square', 'timestamp', 'internet'], dtype=DTYPES, **kwargs)


def aggregate_chunk(chunk, sums, counts, start_ms):
    """Add bounded input to accumulators, validating before any mutation."""
    offsets = chunk.timestamp.to_numpy() - start_ms
    squares = chunk.square.to_numpy()
    values = chunk.internet.to_numpy()
    if ((offsets < 0) | (offsets >= len(sums)*STEP) | (offsets % STEP != 0)).any():
        raise ValueError('Timestamp outside local day or off the 10-minute grid')
    if ((squares < 1) | (squares > 10000)).any():
        raise ValueError('Square ID outside 1..10000')
    if np.isinf(values).any() or (values < 0).any():
        raise ValueError('Invalid Internet activity')
    valid = ~np.isnan(values)
    slots = offsets[valid] // STEP
    ids = squares[valid] - 1
    np.add.at(sums, (slots, ids), values[valid])
    np.add.at(counts, (slots, ids), 1)


def prepare(path, date, output, expected_md5, chunksize=100_000):
    if chunksize < 1:
        raise ValueError('chunksize must be positive')
    started = time.perf_counter()
    digest = hashlib.md5()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    if digest.hexdigest() != expected_md5:
        raise ValueError('Publisher checksum does not match; do not process this file')

    start = milan_midnight_ms(date)
    next_date = (pd.Timestamp(date) + pd.Timedelta(days=1)).date().isoformat()
    intervals = (milan_midnight_ms(next_date)-start)//STEP
    sums = np.zeros((intervals, 10000), dtype=np.float64)
    counts = np.zeros_like(sums, dtype=np.uint32)

    # Compare the same bounded rows. This measures DataFrame storage, not total RSS.
    baseline = pd.read_csv(path, sep='\t', header=None, names=NAMES, nrows=chunksize)
    optimized = read_projected(path, nrows=chunksize)
    sample_rows = len(baseline)
    before = int(baseline.memory_usage(deep=True).sum())
    after = int(optimized.memory_usage(deep=True).sum())
    pd.testing.assert_frame_equal(baseline[['square','timestamp','internet']], optimized, check_dtype=False)
    del baseline, optimized

    rows = missing_fields = chunks = 0
    largest_chunk = 0
    for chunk in read_projected(path, chunksize=chunksize):
        aggregate_chunk(chunk, sums, counts, start)
        rows += len(chunk)
        missing_fields += int(chunk.internet.isna().sum())
        chunks += 1
        largest_chunk = max(largest_chunk, int(chunk.memory_usage(deep=True).sum()))
    missing_bins = counts == 0
    sums[missing_bins] = np.nan
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / f'{date}.npz', internet=sums, observed_rows=counts,
                        timestamps=start+np.arange(intervals)*STEP, square_ids=np.arange(1,10001))
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report = {
        'date': date, 'scope': 'Daily aggregation audit',
        'input_name': path.name, 'input_bytes': path.stat().st_size,
        'md5': digest.hexdigest(), 'raw_rows': rows, 'chunks': chunks,
        'timezone': 'Europe/Rome', 'intervals': intervals, 'squares': 10000,
        'missing_internet_fields': missing_fields,
        'missing_square_time_bins': int(missing_bins.sum()),
        'observed_square_time_bins': int((~missing_bins).sum()),
        'total_internet_activity': float(np.nansum(sums)),
        'sample_rows_for_memory_comparison': sample_rows,
        'sample_dataframe_bytes_before': before, 'sample_dataframe_bytes_after': after,
        'sample_dataframe_reduction_percent': 100*(1-after/before),
        'memory_comparison': 'Same rows: all 8 inferred columns versus 3 projected typed columns; not process RSS',
        'largest_projected_chunk_bytes': largest_chunk,
        'accumulator_bytes': sums.nbytes+counts.nbytes,
        'whole_process_peak_rss_bytes': int(peak_rss if platform.system()=='Darwin' else peak_rss*1024),
        'wall_seconds_including_checksum_and_sample_benchmark': time.perf_counter()-started,
        'platform': platform.platform(), 'machine': platform.machine(),
        'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
        'limitations': ['Checksums validate transfer integrity, not source measurement accuracy', 'An observed aggregate can have partial country-code coverage',
                         'Process peak includes imports and both sample frames; not an isolated before/after RSS benchmark'],
    }
    (output / f'{date}-audit.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('file', type=Path)
    parser.add_argument('--date', required=True)
    parser.add_argument('--md5', required=True)
    parser.add_argument('--output', type=Path, default=Path('data/processed'))
    parser.add_argument('--chunksize', type=int, default=100_000)
    args = parser.parse_args()
    print(json.dumps(prepare(args.file, args.date, args.output, args.md5, args.chunksize), indent=2))
