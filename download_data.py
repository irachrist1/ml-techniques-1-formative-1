"""Resume checksum-verified downloads and daily processing from Dataverse.

Supply the downloader's own email through DATAVERSE_EMAIL after accepting the
dataset's ODbL terms. Signed URLs are transient and are never logged or written
to disk: they are handed to curl over stdin so they do not appear in `ps` output.

A file is only trusted once its MD5 matches the publisher's manifest, and an
existing raw file with the wrong checksum is treated as an error rather than
silently re-downloaded over.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from prepare_day import file_md5

ROOT = Path(__file__).resolve().parent

#: The 61 November and December daily files; January 1 is outside the study period.
FILE_PREFIXES = ('sms-call-internet-mi-2013-11-', 'sms-call-internet-mi-2013-12-')
EXPECTED_FILES = 61
ATTEMPTS = 3


def signed_url_request(file_id: int, email: str) -> subprocess.CompletedProcess:
    """Ask Dataverse for a signed URL, answering the publisher's guestbook."""
    url = f'https://dataverse.harvard.edu/api/access/datafile/{file_id}?signed=true'
    return subprocess.run(
        ['curl', '-fsS', '--max-time', '40', '-H', 'Content-Type: application/json',
         '-X', 'POST', url, '--data-binary', '@-'],
        input=json.dumps({'guestbookResponse': {'email': email, 'answers': []}}),
        capture_output=True, text=True)


def fetch_one(entry: dict[str, Any], email: str) -> dict[str, Any]:
    """Download one daily file if needed, verify it, then aggregate it.

    Raises:
        ValueError: If a raw file already on disk fails its checksum.
        RuntimeError: If the download or the aggregation step does not succeed.
    """
    name = entry['filename']
    date = name.removeprefix('sms-call-internet-mi-').removesuffix('.txt')
    raw = ROOT / 'data/raw' / name
    raw.parent.mkdir(parents=True, exist_ok=True)
    output = ROOT / 'data/processed'
    audit = output / f'{date}-audit.json'

    if raw.exists() and file_md5(raw) != entry['md5']:
        raise ValueError(f'Existing raw file has incorrect checksum: {name}')

    if not raw.exists():
        partial = raw.with_suffix('.txt.part')
        for attempt in range(ATTEMPTS):
            response = signed_url_request(entry['id'], email)
            if response.returncode:
                if attempt == ATTEMPTS - 1:
                    raise RuntimeError(f'Guestbook endpoint failed for {name}')
                continue
            signed = json.loads(response.stdout)['data']['signedUrl']
            # Pass the transient URL through stdin rather than exposing it in ps output.
            config = 'url = ' + json.dumps(signed) + '\n'
            result = subprocess.run(
                ['curl', '-fLsS', '--retry', '2', '--connect-timeout', '30',
                 '--max-time', '1800', '-C', '-', '-K', '-', '-o', str(partial)],
                input=config, capture_output=True, text=True)
            if result.returncode == 0 and file_md5(partial) == entry['md5']:
                partial.replace(raw)
                break
            if attempt == ATTEMPTS - 1:
                raise RuntimeError(f'Download/checksum failed for {name} (curl {result.returncode})')

    # Re-aggregate when the audit is missing, describes a different file, or its
    # output array is gone. Otherwise the day is already done.
    current = json.loads(audit.read_text()) if audit.exists() else {}
    if current.get('md5') != entry['md5'] or not (output / f'{date}.npz').exists():
        result = subprocess.run(
            [sys.executable, str(ROOT / 'prepare_day.py'), str(raw), '--date', date,
             '--md5', entry['md5'], '--output', str(output)], capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f'Processing failed for {name}: {result.stderr[-600:]}')
    return {'date': date, 'filename': name, 'bytes': raw.stat().st_size,
            'md5': entry['md5'], 'status': 'verified_processed'}


def manifest() -> list[dict[str, Any]]:
    """The 61 expected daily files from the saved Dataverse metadata snapshot."""
    metadata = json.loads((ROOT / 'sources/dataset-metadata.json').read_text())
    files = [item['dataFile'] for item in metadata['data']['latestVersion']['files']
             if item['dataFile']['filename'].startswith(FILE_PREFIXES)]
    names = [item['filename'] for item in files]
    if len(files) != EXPECTED_FILES or len(set(names)) != EXPECTED_FILES:
        raise ValueError('Expected 61 distinct November-December daily files')
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    email = os.environ.get('DATAVERSE_EMAIL')
    if not email:
        parser.error('Set DATAVERSE_EMAIL to your own email after reviewing the dataset terms')

    files = manifest()
    progress = ROOT / 'data/download-progress.json'
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch_one, item, email) for item in files]
        for future in as_completed(futures):
            try:
                row = future.result()
                rows.append(row)
                print(f"{len(rows)}/{EXPECTED_FILES} verified and processed: {row['date']}", flush=True)
            except Exception as exc:
                errors.append(str(exc))
                print(str(exc), flush=True)
            # Write progress atomically so an interrupted run leaves a readable file.
            temp = progress.with_suffix('.tmp')
            temp.write_text(json.dumps({'complete': sorted(rows, key=lambda x: x['date']),
                                        'errors': errors,
                                        'elapsed_seconds': time.perf_counter() - started}, indent=2))
            temp.replace(progress)
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
