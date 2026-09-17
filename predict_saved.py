"""Replay one saved one-step forecast using only observations before its target."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent


def predict(run,kind,timestamp):
    run=Path(run);summary=json.loads((run/'summary.json').read_text())
    entry=next(r for r in summary['runs'] if r['model']==kind)
    scaler=entry['scaler'];lookback=entry['config']['lookback'];area=entry['area']
    frame=pd.read_csv(ROOT/'results/selected_series.csv',index_col='timestamp_ms')
    target=pd.Timestamp(timestamp)
    if target.tz is None:raise ValueError('Include an explicit timezone offset in the target timestamp')
    milliseconds=target.value//1_000_000
    history=frame.loc[(frame.index>=milliseconds-lookback*600000)&(frame.index<milliseconds),str(area)].to_numpy()
    if len(history)!=lookback or not np.isfinite(history).all():raise ValueError('Required observed history is incomplete')
    x=((history-scaler['mean'])/scaler['std']).astype(np.float32)[None,:,None]
    if kind=='RidgeAR':
        with np.load(run/'RidgeAR.npz') as model:correction=float(x[0,:,0]@model['coef']+model['intercept'])
    else:
        from experiments import tensorflow
        tf=tensorflow();model=tf.keras.models.load_model(run/f'{kind}.keras',compile=False)
        correction=float(model(x,training=False).numpy().reshape(-1)[0])
    return max(0,float(history[-1]+scaler['std']*correction))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',required=True)
    parser.add_argument('--model',choices=['RidgeAR','LSTM','CausalCNN'],required=True)
    parser.add_argument('--timestamp',required=True);args=parser.parse_args()
    print(json.dumps({'model':args.model,'target':args.timestamp,'prediction':predict(args.run,args.model,args.timestamp)}))
