"""Train from past data, tune on Dec 9–15, and evaluate a frozen study on Dec 16–22.

All models learn a correction to persistence. No missing target is imputed.
Every architecture uses the same eligible timestamps (a complete 144-step history).
"""
import argparse
import hashlib
import json
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
os.environ.setdefault('OMP_NUM_THREADS','4')
from pathlib import Path
import platform
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from evaluation import milan_midnight_ms, score

ROOT=Path(__file__).resolve().parent
MAX_HISTORY=144
TRAIN_END=milan_midnight_ms('2013-12-09')
TEST_START=milan_midnight_ms('2013-12-16')
TEST_END=milan_midnight_ms('2013-12-23')


def windows(frame,area,lookback):
    if not 1<=lookback<=MAX_HISTORY:raise ValueError('lookback must be between 1 and 144')
    timestamps=frame.index.to_numpy(dtype=np.int64)
    if not np.all(np.diff(timestamps)==600000):raise ValueError('Input requires a complete ordered 10-minute time grid')
    values=frame[str(area)].to_numpy(dtype=np.float64)
    train_values=values[timestamps<TRAIN_END]
    mean=float(np.nanmean(train_values));std=float(np.nanstd(train_values))
    if not np.isfinite(std) or std<=0:raise ValueError('Training scale must be finite and positive')
    all_windows=np.lib.stride_tricks.sliding_window_view(values,MAX_HISTORY+1)
    valid=np.isfinite(all_windows).all(axis=1)
    target_times=timestamps[MAX_HISTORY:][valid]
    history=all_windows[valid,:-1][:,-lookback:]
    target=all_windows[valid,-1]
    x=((history-mean)/std).astype(np.float32)[...,None]
    delta=((target-history[:,-1])/std).astype(np.float32)
    masks={'train':target_times<TRAIN_END,'validation':(target_times>=TRAIN_END)&(target_times<TEST_START),
        'test':(target_times>=TEST_START)&(target_times<TEST_END)}
    data={key:{'x':x[mask],'delta':delta[mask],'y':target[mask],'last':history[mask,-1],
               'seasonal':all_windows[valid,0][mask],'timestamps':target_times[mask]} for key,mask in masks.items()}
    if any(len(d['y'])==0 for d in data.values()):raise ValueError('Empty split after missing-window exclusions')
    return data,{'mean':mean,'std':std,'fit_end_exclusive':TRAIN_END,'common_history':MAX_HISTORY}


def tensorflow():
    import tensorflow as tf
    try:
        tf.config.threading.set_intra_op_parallelism_threads(4)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except RuntimeError:pass
    tf.config.experimental.enable_op_determinism()
    return tf


def neural_model(kind,config):
    tf=tensorflow();layers=tf.keras.layers
    inputs=layers.Input((config['lookback'],1),name='normalized_history')
    if kind=='LSTM':
        z=layers.LSTM(config['width'],name='gated_memory')(inputs)
    elif kind=='CausalCNN':
        z=inputs
        for dilation in config['dilations']:
            z=layers.Conv1D(config['width'],3,padding='causal',dilation_rate=dilation,activation='relu',name=f'causal_d{dilation}')(z)
        z=layers.Cropping1D((config['lookback']-1,0))(z)
        z=layers.Flatten()(z)
    else:raise ValueError(kind)
    output=layers.Dense(1,kernel_initializer='zeros',bias_initializer='zeros',name='persistence_correction')(z)
    model=tf.keras.Model(inputs,output,name=kind)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config['learning_rate'],clipnorm=1.0),loss='mse')
    return model


def dataset(tf,d,seed=None,batch=128):
    ds=tf.data.Dataset.from_tensor_slices((d['x'],d['delta'][:,None]))
    if seed is not None:ds=ds.shuffle(len(d['y']),seed=seed,reshuffle_each_iteration=True)
    options=tf.data.Options();options.threading.private_threadpool_size=1;options.experimental_deterministic=True
    return ds.batch(batch).with_options(options).prefetch(1)


def fit_one(kind,config,data,seed):
    train=data['train'];validation=data['validation']
    history={}
    if kind=='RidgeAR':
        model=Ridge(alpha=config['alpha'],solver='svd')
        start=time.perf_counter();model.fit(train['x'][:,:,0],train['delta']);training_seconds=time.perf_counter()-start
        predict=lambda x:model.predict(x[:,:,0])
        parameters=int(model.coef_.size+1)
    else:
        tf=tensorflow();tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(seed)
        start=time.perf_counter();model=neural_model(kind,config)
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=config.get('patience',4),min_delta=1e-5,restore_best_weights=True),tf.keras.callbacks.TerminateOnNaN()]
        result=model.fit(dataset(tf,train,seed),validation_data=dataset(tf,validation),epochs=config['epochs'],callbacks=callbacks,verbose=0,shuffle=False)
        training_seconds=time.perf_counter()-start
        history={k:[float(v) for v in vals] for k,vals in result.history.items()}
        if not all(np.isfinite(history['loss'])):raise ValueError('Nonfinite training loss')
        @tf.function(reduce_retracing=True)
        def forward(x):return model(x,training=False)
        predict=lambda x:forward(tf.convert_to_tensor(x)).numpy().reshape(-1)
        parameters=int(model.count_params())
    return model,predict,training_seconds,parameters,history


def run(config_path,area,seed,phase):
    configs=json.loads(Path(config_path).read_text())
    name=configs['id'];out=ROOT/'results/runs'/f'{phase}_{name}_area{area}_seed{seed}'
    if (out/'summary.json').exists():raise FileExistsError(f'Run already exists: {out}; use another ID rather than overwriting evidence')
    out.mkdir(parents=True,exist_ok=True)
    frame=pd.read_csv(ROOT/'results/selected_series.csv',index_col='timestamp_ms')
    summaries=[]
    for kind,config in configs['models'].items():
        data,scaler=windows(frame,area,config['lookback'])
        model,predict,seconds,parameters,history=fit_one(kind,config,data,seed)
        model_summary={'model':kind,'area':area,'seed':seed,'phase':phase,'experiment':name,'config':config,
            'rationale':configs['rationale'],'scaler':scaler,'train_n':len(data['train']['y']),
            'training_seconds':seconds,'parameter_count':parameters,'epochs_run':len(history.get('loss',[])),
            'best_epoch':int(np.argmin(history['val_loss'])+1) if history else None}
        for split in (['validation'] if phase=='tune' else ['validation','test']):
            d=data[split];correction=predict(d['x']);prediction=np.maximum(0,d['last']+scaler['std']*correction)
            model_summary[split]=score(d['y'],prediction)
            table=pd.DataFrame({'timestamp_ms':d['timestamps'],'actual':d['y'],'prediction':prediction,'persistence':d['last'],'seasonal_daily':d['seasonal']})
            table.to_csv(out/f'{kind}_{split}_predictions.csv',index=False)
        timing_data=data['validation' if phase=='tune' else 'test']
        predict(timing_data['x'])
        times=[]
        for _ in range(5):
            start=time.perf_counter();predict(timing_data['x']);times.append(time.perf_counter()-start)
        point=[]
        predict(timing_data['x'][:1])
        for i in range(min(30,len(timing_data['y']))):
            start=time.perf_counter();predict(timing_data['x'][i:i+1]);point.append(time.perf_counter()-start)
        model_summary['inference']={'batch_n':len(timing_data['y']),'warm_batch_seconds_median':float(np.median(times)),
            'warm_batch_seconds_runs':times,'single_seconds_median':float(np.median(point)),'single_seconds_p95':float(np.quantile(point,.95))}
        (out/f'{kind}_history.json').write_text(json.dumps(history,indent=2))
        (out/f'{kind}_scaler.json').write_text(json.dumps(scaler,indent=2))
        if kind=='RidgeAR':np.savez(out/f'{kind}.npz',coef=model.coef_,intercept=model.intercept_)
        else:model.save(out/f'{kind}.keras')
        summaries.append(model_summary)
        (out/'partial_summary.json').write_text(json.dumps(summaries,indent=2))
        print(json.dumps({'area':area,'model':kind,'seed':seed,'phase':phase,'val_rmse':model_summary['validation']['rmse'],'seconds':seconds}),flush=True)
    (out/'summary.json').write_text(json.dumps({'config':configs,'data_sha256':hashlib.sha256((ROOT/'results/selected_series.csv').read_bytes()).hexdigest(),'platform':platform.platform(),'machine':platform.machine(),'runs':summaries},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--area',type=int,required=True)
    parser.add_argument('--seed',type=int,default=42);parser.add_argument('--phase',choices=['tune','final'],default='tune')
    args=parser.parse_args();run(args.config,args.area,args.seed,args.phase)
