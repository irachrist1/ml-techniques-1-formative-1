"""Resume checksum-verified downloads and daily processing from Dataverse.

Supply the downloader's own email through DATAVERSE_EMAIL after accepting the
dataset's ODbL terms. Signed URLs are transient and never logged or persisted.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def md5(path):
    h = hashlib.md5()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def fetch_one(entry, email):
    name = entry['filename']
    date = name.removeprefix('sms-call-internet-mi-').removesuffix('.txt')
    raw = ROOT/'data/raw'/name
    raw.parent.mkdir(parents=True, exist_ok=True)
    output = ROOT/'data/processed'
    audit = output/f'{date}-audit.json'
    if raw.exists() and md5(raw) != entry['md5']:
        raise ValueError(f'Existing raw file has incorrect checksum: {name}')
    if not raw.exists():
        partial = raw.with_suffix('.txt.part')
        url = f"https://dataverse.harvard.edu/api/access/datafile/{entry['id']}?signed=true"
        for attempt in range(3):
            response = subprocess.run(['curl','-fsS','--max-time','40','-H','Content-Type: application/json',
                '-X','POST',url,'--data-binary','@-'], input=json.dumps({'guestbookResponse':{'email':email,'answers':[]}}),
                capture_output=True, text=True)
            if response.returncode:
                if attempt == 2: raise RuntimeError(f'Guestbook endpoint failed for {name}')
                continue
            signed = json.loads(response.stdout)['data']['signedUrl']
            # Pass transient URL through stdin, rather than exposing it in ps output.
            config = 'url = '+json.dumps(signed)+'\n'
            result = subprocess.run(['curl','-fLsS','--retry','2','--connect-timeout','30',
                '--max-time','1800','-C','-','-K','-','-o',str(partial)], input=config,
                capture_output=True, text=True)
            if result.returncode == 0 and md5(partial) == entry['md5']:
                partial.replace(raw)
                break
            if attempt == 2: raise RuntimeError(f'Download/checksum failed for {name} (curl {result.returncode})')
    current = json.loads(audit.read_text()) if audit.exists() else {}
    if current.get('md5') != entry['md5'] or not (output/f'{date}.npz').exists():
        result = subprocess.run([sys.executable,str(ROOT/'prepare_day.py'),str(raw),'--date',date,
            '--md5',entry['md5'],'--output',str(output)], capture_output=True, text=True)
        if result.returncode: raise RuntimeError(f'Processing failed for {name}: {result.stderr[-600:]}')
    return {'date':date,'filename':name,'bytes':raw.stat().st_size,'md5':entry['md5'],'status':'verified_processed'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=4)
    args = parser.parse_args()
    email = os.environ.get('DATAVERSE_EMAIL')
    if not email: parser.error('Set DATAVERSE_EMAIL to your own email after reviewing the dataset terms')
    metadata = json.loads((ROOT/'sources/dataset-metadata.json').read_text())
    files = [item['dataFile'] for item in metadata['data']['latestVersion']['files']
             if item['dataFile']['filename'].startswith(('sms-call-internet-mi-2013-11-','sms-call-internet-mi-2013-12-'))]
    names = [item['filename'] for item in files]
    if len(files) != 61 or len(set(names)) != 61: raise ValueError('Expected 61 distinct November–December daily files')
    progress = ROOT/'data/download-progress.json'
    rows, errors = [], []
    started=time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(fetch_one,item,email):item for item in files}
        for future in as_completed(futures):
            try:
                row=future.result();rows.append(row)
                print(f"{len(rows)}/61 verified and processed: {row['date']}",flush=True)
            except Exception as exc:
                errors.append(str(exc));print(str(exc),flush=True)
            temp=progress.with_suffix('.tmp')
            temp.write_text(json.dumps({'complete':sorted(rows,key=lambda x:x['date']),'errors':errors,
                'elapsed_seconds':time.perf_counter()-started},indent=2));temp.replace(progress)
    if errors: raise SystemExit(1)


if __name__=='__main__':main()
