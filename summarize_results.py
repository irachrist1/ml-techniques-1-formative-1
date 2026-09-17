"""Build comparisons and failure diagnostics from frozen, saved test predictions."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from evaluation import score, milan_midnight_ms

ROOT=Path(__file__).resolve().parent
RESULTS=ROOT/'results';FIGURES=RESULTS/'figures'
COLORS={'RidgeAR':'#14645a','LSTM':'#b05f28','CausalCNN':'#53679a'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})


def main():
    selected=json.loads((ROOT/'configs/final.json').read_text())
    eda=json.loads((RESULTS/'eda_summary.json').read_text());areas=eda['areas'];name=selected['id']
    rows=[];predictions={};reference_times={};timing=[];baseline_rows=[]
    for area in areas:
        for seed in [42,43,44]:
            folder=RESULTS/'runs'/f'final_{name}_area{area}_seed{seed}'
            summary=json.loads((folder/'summary.json').read_text())
            for r in summary['runs']:
                kind=r['model'];p=pd.read_csv(folder/f'{kind}_test_predictions.csv')
                if p.timestamp_ms.duplicated().any():raise ValueError('Duplicate predictions')
                if not p.timestamp_ms.between(milan_midnight_ms('2013-12-16'),milan_midnight_ms('2013-12-23')-1).all():raise ValueError('Wrong test dates')
                previous=reference_times.setdefault(area,p.timestamp_ms.to_numpy())
                np.testing.assert_array_equal(previous,p.timestamp_ms.to_numpy())
                metrics=score(p.actual,p.prediction)
                if not np.isclose(metrics['rmse'],r['test']['rmse']):raise ValueError('Saved metrics and predictions disagree')
                rows.append({'area':area,'model':kind,'seed':seed,**metrics})
                timing.append({'area':area,'model':kind,'seed':seed,'parameters':r['parameter_count'],
                    'epochs':r['epochs_run'],'best_epoch':r['best_epoch'],'train_seconds':r['training_seconds'],
                    'batch_inference_seconds':r['inference']['warm_batch_seconds_median'],
                    'single_inference_ms':1000*r['inference']['single_seconds_median']})
                if seed==42:
                    predictions[(area,kind)]=p
                    local=pd.to_datetime(p.timestamp_ms,unit='ms',utc=True).dt.tz_convert('Europe/Rome')
                    fig,ax=plt.subplots(figsize=(9,2.7));ax.plot(local,p.actual,label='Observed',color='#252525',lw=.85);ax.plot(local,p.prediction,label=kind,color=COLORS[kind],lw=.7,alpha=.85)
                    ax.set(title=f'Area {area}: {kind}, rolling one-step forecasts (seed 42)',xlabel='16–22 December 2013, Europe/Rome',ylabel='Internet activity units');ax.legend(ncol=2,frameon=False);fig.autofmt_xdate();fig.savefig(FIGURES/f'forecast_{area}_{kind}.png',bbox_inches='tight');plt.close(fig)
            if seed==42:
                for kind,col in [('Persistence','persistence'),('Daily seasonal','seasonal_daily')]:baseline_rows.append({'area':area,'model':kind,'seed':42,**score(p.actual,p[col])})
    metrics=pd.DataFrame(rows);baselines=pd.DataFrame(baseline_rows);times=pd.DataFrame(timing)
    metrics.to_csv(RESULTS/'test_metrics_all_seeds.csv',index=False);baselines.to_csv(RESULTS/'baseline_metrics.csv',index=False);times.to_csv(RESULTS/'timing.csv',index=False)
    reference=pd.concat([metrics[metrics.seed==42],baselines],ignore_index=True);reference.to_csv(RESULTS/'reference_metrics.csv',index=False)
    aggregate=metrics.groupby(['area','model']).agg(rmse_mean=('rmse','mean'),rmse_std=('rmse','std'),rmse_min=('rmse','min'),rmse_max=('rmse','max'),mae_mean=('mae','mean'),mape_mean=('mape_percent_nonzero','mean')).reset_index()
    aggregate.to_csv(RESULTS/'seed_summary.csv',index=False)
    # Select the failure event by largest reference-model absolute error / training std.
    failure=None
    for area in areas:
        scale=next(r['train_std'] for r in eda['area_stats'] if r['square']==area)
        for kind in COLORS:
            p=predictions[(area,kind)];error=np.abs(p.actual-p.prediction).to_numpy();i=int(np.argmax(error));severity=float(error[i]/scale)
            if failure is None or severity>failure['severity_train_std']:
                failure={'area':area,'model':kind,'index':i,'timestamp_ms':int(p.timestamp_ms.iloc[i]),'actual':float(p.actual.iloc[i]),'prediction':float(p.prediction.iloc[i]),'abs_error':float(error[i]),'severity_train_std':severity}
    area=failure['area'];center=failure['timestamp_ms'];fig,ax=plt.subplots(figsize=(9,3.5))
    p=predictions[(area,'RidgeAR')];mask=p.timestamp_ms.between(center-18*600000,center+18*600000)
    local=pd.to_datetime(p.timestamp_ms[mask],unit='ms',utc=True).dt.tz_convert('Europe/Rome');ax.plot(local,p.actual[mask],color='#252525',lw=1.6,label='Observed')
    for kind in COLORS:ax.plot(local,predictions[(area,kind)].prediction[mask],lw=1,label=kind,color=COLORS[kind])
    ax.set(title=f'Largest standardized forecast miss: area {area}',xlabel='Local date and time (Europe/Rome), ±3 hours around event',ylabel='Internet activity units');ax.legend(ncol=4,fontsize=8,frameon=False);fig.autofmt_xdate();fig.savefig(FIGURES/'failure_case.png',bbox_inches='tight');plt.close(fig)
    # Actual-traffic deciles are descriptive test diagnostics, never used to tune models.
    diagnostics=[]
    for area in areas:
        for kind in COLORS:
            p=predictions[(area,kind)].copy();p['bin']=pd.qcut(p.actual,10,duplicates='drop');p['error']=p.prediction-p.actual
            for label,g in p.groupby('bin',observed=True):diagnostics.append({'area':area,'model':kind,'actual_lower':label.left,'actual_upper':label.right,'n':len(g),'bias':g.error.mean(),'mae':g.error.abs().mean()})
    pd.DataFrame(diagnostics).to_csv(RESULTS/'error_by_actual_decile.csv',index=False)
    winners=[]
    for area in areas:
        a=reference[reference.area==area];best=a.loc[a.rmse.idxmin()]
        learned=a[a.model.isin(COLORS)];b=learned.loc[learned.rmse.idxmin()]
        persist=float(a.loc[a.model=='Persistence','rmse'].iloc[0])
        winners.append({'area':area,'best_including_baselines':best['model'],'best_learned':b['model'],
            'best_learned_rmse':float(b.rmse),'persistence_rmse':persist,'improvement_over_persistence_percent':100*(1-float(b.rmse)/persist),'test_n':int(b['n'])})
    tuning=[]
    for path in sorted((RESULTS/'runs').glob('tune_*/summary.json'), key=lambda p: next((i for i,n in enumerate(['initial','daily_history','capacity']) if p.parent.name.startswith('tune_'+n+'_')),99)):
        for r in json.loads(path.read_text())['runs']:tuning.append({'experiment':r['experiment'],'model':r['model'],'area':r['area'],'seed':r['seed'],'val_rmse':r['validation']['rmse'],'val_mae':r['validation']['mae'],'train_seconds':r['training_seconds'],'best_epoch':r['best_epoch'],'config':json.dumps(r['config']),'rationale':r['rationale']})
    pd.DataFrame(tuning).to_csv(RESULTS/'tuning_log.csv',index=False)
    # Learning curves for the reference fit on the highest-ranked area.
    fig,axs=plt.subplots(1,2,figsize=(9,3))
    for ax,kind in zip(axs,['LSTM','CausalCNN']):
        history=json.loads((RESULTS/'runs'/f'final_{name}_area{areas[0]}_seed42'/f'{kind}_history.json').read_text())
        ax.plot(np.arange(1,len(history['loss'])+1),history['loss'],label='Training');ax.plot(np.arange(1,len(history['val_loss'])+1),history['val_loss'],label='Validation');ax.set(title=f'{kind}: area {areas[0]}, seed 42',xlabel='Epoch',ylabel='MSE of standardized correction');ax.legend(frameon=False)
    fig.tight_layout();fig.savefig(FIGURES/'learning_curves.png',bbox_inches='tight');plt.close(fig)
    summary={'winners':winners,'failure':failure,'seed_count':3,'reference_seed':42,
        'timing_by_model':times.groupby('model')[['train_seconds','batch_inference_seconds','single_inference_ms']].mean().reset_index().to_dict('records')}
    (RESULTS/'study_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
