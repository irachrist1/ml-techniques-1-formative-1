"""Compare regenerated results against the committed ones, numerically.

`reproduce.py` rebuilds every results table from the saved predictions. On the
machine the study was run on, those files come back byte-identical. On a
different CPU architecture they do not: summation of the same float64 values can
land one unit in the last place apart, which shows up as
`75.96750464580491` against `75.9675046458049`.

That is float noise, not a regression, so this compares values with a tight
relative tolerance instead of comparing bytes. A tolerance of 1e-9 is roughly
five orders of magnitude tighter than anything that could matter to a reported
result, and about five orders looser than last-place noise.

Usage:  python check_regenerated.py
Exits non-zero, naming the file and the value, if anything moved.
"""
from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

#: Relative tolerance for every regenerated number.
RTOL = 1e-9

#: verify_submission.py records the measured difference between each replayed
#: forecast and the stored one. Those differences are themselves float noise and
#: are expected to change between machines, so the file is checked by its own
#: guarantees rather than value by value.
REPLAY_REPORT = 'results/verification.json'
REPLAY_CEILING = 1e-4  # the atol verify_submission.py already enforces per forecast


def committed(path: str) -> bytes:
    """The version of `path` in HEAD."""
    return subprocess.run(['git', 'show', f'HEAD:{path}'], cwd=ROOT,
                          capture_output=True, check=True).stdout


def compare_frames(path: str, old: pd.DataFrame, new: pd.DataFrame) -> list[str]:
    problems = []
    if list(old.columns) != list(new.columns):
        return [f'{path}: columns changed']
    if len(old) != len(new):
        return [f'{path}: row count {len(old)} -> {len(new)}']
    for column in old.columns:
        a, b = old[column], new[column]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            close = np.isclose(a, b, rtol=RTOL, atol=0, equal_nan=True)
            for i in np.flatnonzero(~close)[:3]:
                problems.append(f'{path}: {column}[{i}] {a.iloc[i]!r} -> {b.iloc[i]!r}')
        elif not a.equals(b):
            problems.append(f'{path}: column {column} differs')
    return problems


def compare_values(path: str, old, new, trail: str = '') -> list[str]:
    """Recursively compare JSON values, allowing float noise."""
    if isinstance(old, dict) and isinstance(new, dict):
        if old.keys() != new.keys():
            return [f'{path}{trail}: keys changed']
        return [p for k in old for p in compare_values(path, old[k], new[k], f'{trail}.{k}')]
    if isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            return [f'{path}{trail}: length {len(old)} -> {len(new)}']
        return [p for i, (a, b) in enumerate(zip(old, new, strict=True))
                for p in compare_values(path, a, b, f'{trail}[{i}]')]
    if isinstance(old, float) or isinstance(new, float):
        if not np.isclose(old, new, rtol=RTOL, atol=0, equal_nan=True):
            return [f'{path}{trail}: {old!r} -> {new!r}']
        return []
    return [] if old == new else [f'{path}{trail}: {old!r} -> {new!r}']


def main() -> None:
    tracked = subprocess.run(['git', 'ls-files', 'results/*.csv', 'results/*.json'],
                             cwd=ROOT, capture_output=True, text=True, check=True)
    problems, checked = [], 0
    for path in tracked.stdout.split():
        if path == REPLAY_REPORT:
            continue
        current = (ROOT / path).read_bytes()
        reference = committed(path)
        if current == reference:
            checked += 1
            continue
        if path.endswith('.csv'):
            problems += compare_frames(path, pd.read_csv(io.BytesIO(reference)),
                                       pd.read_csv(io.BytesIO(current)))
        else:
            problems += compare_values(path, json.loads(reference), json.loads(current))
        checked += 1

    report = json.loads((ROOT / REPLAY_REPORT).read_text())
    if report['status'] != 'passed':
        problems.append(f'{REPLAY_REPORT}: status is {report["status"]!r}')
    if report['largest_replay_difference'] > REPLAY_CEILING:
        problems.append(f'{REPLAY_REPORT}: largest replay difference '
                        f'{report["largest_replay_difference"]:.3g} exceeds {REPLAY_CEILING:g}')

    if problems:
        print(f'{len(problems)} regenerated value(s) moved beyond rtol={RTOL:g}:')
        for problem in problems:
            print(f'  {problem}')
        sys.exit(1)
    print(f'{checked} regenerated results files match the committed ones within rtol={RTOL:g}; '
          f'{REPLAY_REPORT} passed with largest replay difference '
          f'{report["largest_replay_difference"]:.3g}')


if __name__ == '__main__':
    main()
