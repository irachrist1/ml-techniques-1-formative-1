"""Compare full-day baseline and chunked aggregation in separate processes."""
import argparse
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import numpy as np
import pandas as pd
from evaluation import milan_midnight_ms
from prepare_day import NAMES, read_projected, aggregate_chunk
ROOT=Path(__file__).resolve().parent


def worker(mode,path,directory):
    start=time.perf_counter();sums=np.zeros((144,10000));counts=np.zeros((144,10000),dtype=np.uint32)
    start_ms=milan_midnight_ms('2013-11-01')
    if mode=='baseline':
        frame=pd.read_csv(path,sep='\t',header=None,names=NAMES)
        dataframe_bytes=int(frame.memory_usage(deep=True).sum())
        grouped=frame.groupby(['timestamp','square']).internet.agg(['sum','count'])
        slots=(grouped.index.get_level_values('timestamp').to_numpy()-start_ms)//600000
        squares=grouped.index.get_level_values('square').to_numpy()-1
        sums[slots,squares]=grouped['sum'].to_numpy();counts[slots,squares]=grouped['count'].to_numpy()
    else:
        dataframe_bytes=0
        for frame in read_projected(path,chunksize=100000):
            dataframe_bytes=max(dataframe_bytes,int(frame.memory_usage(deep=True).sum()))
            aggregate_chunk(frame,sums,counts,start_ms)
    sums[counts==0]=np.nan
    seconds=time.perf_counter()-start
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system()=='Darwin' else 1024)
    np.savez(directory/f'{mode}.npz',internet=sums,observed_rows=counts)
    print(json.dumps({'mode':mode,'wall_seconds':seconds,'peak_rss_bytes':int(peak),'max_dataframe_bytes':dataframe_bytes}))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--worker',choices=['baseline','chunked']);args=parser.parse_args()
    directory=ROOT/'data/benchmark';directory.mkdir(parents=True,exist_ok=True)
    path=ROOT/'data/raw/sms-call-internet-mi-2013-11-01.txt'
    if args.worker:return worker(args.worker,path,directory)
    runs=[]
    for mode in ['baseline','chunked']:
        result=subprocess.run([sys.executable,__file__,'--worker',mode],capture_output=True,text=True,check=True)
        runs.append(json.loads(result.stdout))
    with np.load(directory/'baseline.npz') as a,np.load(directory/'chunked.npz') as b:
        same_missing=np.array_equal(np.isnan(a['internet']),np.isnan(b['internet']))
        same_counts=np.array_equal(a['observed_rows'],b['observed_rows'])
        close=np.allclose(a['internet'],b['internet'],rtol=1e-12,atol=1e-9,equal_nan=True)
        error=float(np.nanmax(np.abs(a['internet']-b['internet'])))
    if not (same_missing and same_counts and close):raise ValueError('Aggregation outputs disagree')
    report={'input':path.name,'scope':'Identical full daily file, isolated Python processes; one run per method; filesystem cache uncontrolled',
        'runs':runs,'peak_rss_reduction_percent':100*(1-runs[1]['peak_rss_bytes']/runs[0]['peak_rss_bytes']),
        'same_missing_mask':same_missing,'same_observation_counts':same_counts,'max_abs_aggregation_difference':error}
    (ROOT/'results/memory_benchmark.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
