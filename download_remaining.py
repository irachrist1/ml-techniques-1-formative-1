"""Recover incomplete daily downloads using bounded HTTP range requests."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import json,os,subprocess,sys
from pathlib import Path
from download_data import ROOT,md5,fetch_one


def signed_url(file_id,email):
    r=subprocess.run(['curl','-fsS','--max-time','40','-H','Content-Type: application/json','-X','POST',
        f'https://dataverse.harvard.edu/api/access/datafile/{file_id}?signed=true','--data-binary','@-'],
        input=json.dumps({'guestbookResponse':{'email':email,'answers':[]}}),text=True,capture_output=True,check=True)
    return json.loads(r.stdout)['data']['signedUrl']


def range_part(job):
    entry,email,start,end,path=job
    if path.exists() and path.stat().st_size==end-start+1:return path
    for attempt in range(3):
        url=signed_url(entry['id'],email)
        r=subprocess.run(['curl','-fsSL','--connect-timeout','30','--max-time','300','--range',f'{start}-{end}',
            '-K','-','-o',str(path)],input='url = '+json.dumps(url)+'\n',text=True,capture_output=True)
        if r.returncode==0 and path.stat().st_size==end-start+1:
            print(f"Range verified by size: {entry['filename']} {start}-{end}",flush=True);return path
    raise RuntimeError(f"Failed range {entry['filename']} {start}-{end}")


def main():
    email=os.environ['DATAVERSE_EMAIL']
    files=[x['dataFile'] for x in json.loads((ROOT/'sources/dataset-metadata.json').read_text())['data']['latestVersion']['files']
        if x['dataFile']['filename'].startswith(('sms-call-internet-mi-2013-11-','sms-call-internet-mi-2013-12-'))]
    missing=[f for f in files if not (ROOT/'data/raw'/f['filename']).exists()]
    jobs=[];plans=[];partdir=ROOT/'data/ranges';partdir.mkdir(exist_ok=True)
    for f in missing:
        partial=ROOT/'data/raw'/(f['filename']+'.part')
        prefix=partial.stat().st_size if partial.exists() else 0
        if prefix>f['filesize']:raise ValueError('Oversized partial file')
        parts=[]
        for start in range(prefix,f['filesize'],8*1024*1024):
            end=min(start+8*1024*1024,f['filesize'])-1
            p=partdir/f"{f['id']}_{start}_{end}.part";parts.append(p);jobs.append((f,email,start,end,p))
        plans.append((f,partial,prefix,parts))
    with ThreadPoolExecutor(max_workers=6) as pool:
        for future in as_completed([pool.submit(range_part,j) for j in jobs]):future.result()
    for f,partial,prefix,parts in plans:
        assembled=ROOT/'data/raw'/(f['filename']+'.assembled')
        with assembled.open('wb') as out:
            for p in ([partial] if prefix else [])+parts:
                with p.open('rb') as source:
                    for block in iter(lambda:source.read(1024*1024),b''):out.write(block)
        if assembled.stat().st_size!=f['filesize'] or md5(assembled)!=f['md5']:raise ValueError('Assembled file checksum mismatch')
        assembled.replace(ROOT/'data/raw'/f['filename'])
        print(fetch_one(f,email),flush=True)
    print('Recovery complete',flush=True)


if __name__=='__main__':main()
