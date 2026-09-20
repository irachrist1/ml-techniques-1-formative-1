"""Recover incomplete daily downloads using bounded HTTP range requests.

Optional companion to `download_data.py`, for the case where a transfer keeps
failing part-way through a 340 MB file. It fetches fixed-size byte ranges in
parallel, concatenates them onto whatever prefix already downloaded, and only
accepts the result once the assembled file matches the publisher's MD5.

Do not run this at the same time as the ordinary downloader: both write into
`data/raw/`, and concurrent writes to the same partial file would corrupt it.
"""
from __future__ import annotations

import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from download_data import ROOT, fetch_one, manifest
from prepare_day import READ_BLOCK, file_md5

#: Range size per request. Small enough to retry cheaply, large enough that the
#: per-request signed-URL round trip is not the dominant cost.
RANGE_BYTES = 8 * 1024 * 1024
ATTEMPTS = 3
WORKERS = 6


def signed_url(file_id: int, email: str) -> str:
    """One signed URL. Each range request gets a fresh one; they expire quickly."""
    response = subprocess.run(
        ['curl', '-fsS', '--max-time', '40', '-H', 'Content-Type: application/json', '-X', 'POST',
         f'https://dataverse.harvard.edu/api/access/datafile/{file_id}?signed=true',
         '--data-binary', '@-'],
        input=json.dumps({'guestbookResponse': {'email': email, 'answers': []}}),
        text=True, capture_output=True, check=True)
    return json.loads(response.stdout)['data']['signedUrl']


def range_part(job: tuple[dict[str, Any], str, int, int, Path]) -> Path:
    """Fetch one byte range, retrying until the piece is exactly the right size.

    Size is the only per-piece check available; the publisher publishes a
    checksum for the whole file, not for ranges. The assembled file is checksummed
    before anything is accepted, so a corrupt-but-correctly-sized piece is still
    caught, just later.

    Raises:
        RuntimeError: If the range could not be fetched at the expected size.
    """
    entry, email, start, end, path = job
    if path.exists() and path.stat().st_size == end - start + 1:
        return path
    for _ in range(ATTEMPTS):
        url = signed_url(entry['id'], email)
        result = subprocess.run(
            ['curl', '-fsSL', '--connect-timeout', '30', '--max-time', '300',
             '--range', f'{start}-{end}', '-K', '-', '-o', str(path)],
            input='url = ' + json.dumps(url) + '\n', text=True, capture_output=True)
        if result.returncode == 0 and path.stat().st_size == end - start + 1:
            print(f"Range verified by size: {entry['filename']} {start}-{end}", flush=True)
            return path
    raise RuntimeError(f"Failed range {entry['filename']} {start}-{end}")


def plan_ranges(missing: list[dict[str, Any]], email: str, partdir: Path):
    """Work out which ranges each missing file still needs.

    Raises:
        ValueError: If a partial file is larger than the published size, which
            means it is not a prefix of the real file and must not be reused.
    """
    jobs, plans = [], []
    for entry in missing:
        partial = ROOT / 'data/raw' / (entry['filename'] + '.part')
        prefix = partial.stat().st_size if partial.exists() else 0
        if prefix > entry['filesize']:
            raise ValueError('Oversized partial file')
        parts = []
        for start in range(prefix, entry['filesize'], RANGE_BYTES):
            end = min(start + RANGE_BYTES, entry['filesize']) - 1
            piece = partdir / f"{entry['id']}_{start}_{end}.part"
            parts.append(piece)
            jobs.append((entry, email, start, end, piece))
        plans.append((entry, partial, prefix, parts))
    return jobs, plans


def main() -> None:
    email = os.environ['DATAVERSE_EMAIL']
    missing = [f for f in manifest() if not (ROOT / 'data/raw' / f['filename']).exists()]
    partdir = ROOT / 'data/ranges'
    partdir.mkdir(parents=True, exist_ok=True)
    jobs, plans = plan_ranges(missing, email, partdir)

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for future in as_completed([pool.submit(range_part, job) for job in jobs]):
            future.result()

    for entry, partial, prefix, parts in plans:
        assembled = ROOT / 'data/raw' / (entry['filename'] + '.assembled')
        with assembled.open('wb') as out:
            for piece in ([partial] if prefix else []) + parts:
                with piece.open('rb') as source:
                    for block in iter(lambda s=source: s.read(READ_BLOCK), b''):
                        out.write(block)
        if assembled.stat().st_size != entry['filesize'] or file_md5(assembled) != entry['md5']:
            raise ValueError('Assembled file checksum mismatch')
        assembled.replace(ROOT / 'data/raw' / entry['filename'])
        print(fetch_one(entry, email), flush=True)
    print('Recovery complete', flush=True)


if __name__ == '__main__':
    main()
