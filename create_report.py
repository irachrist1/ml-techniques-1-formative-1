"""Create an evidence-based research report; render_report.py builds its PDF."""
import json
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parent
R=ROOT/'results'


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])


def main():
    eda=json.loads((R/'eda_summary.json').read_text());study=json.loads((R/'study_summary.json').read_text());memory=json.loads((R/'memory_benchmark.json').read_text());config=json.loads((ROOT/'configs/final.json').read_text())
    ranking=pd.read_csv(R/'area_ranking.csv');metrics=pd.read_csv(R/'reference_metrics.csv');tuning=pd.read_csv(R/'tuning_log.csv');seeds=pd.read_csv(R/'seed_summary.csv');times=pd.read_csv(R/'timing.csv')
    top=eda['top3'];areas=eda['areas'];winner=study['winners'][0];failure=study['failure'];pages=[]
    repo_file=ROOT/'output/repository.json'
    repo=json.loads(repo_file.read_text())['url']
    pages.append(f'''# One-step mobile-network traffic forecasting in Milan

**Christian Tonny | ML Techniques I | Formative Assignment 1**

## 1. Introduction

Mobile-network operators need estimates of near-future demand to inform resource allocation. This study asks how three different sequential models compare for one-step-ahead Internet-activity forecasting, and whether their performance changes across geographical areas. The target is the next 10-minute activity value, not a directly measured bandwidth or byte count.

The study processes the complete November–December 2013 Milan collection, identifies the highest-activity areas, and compares regularized linear autoregression, gated recurrent memory and causal convolution. Persistence and daily-seasonal forecasts establish whether added model complexity is useful. The December 16–22 evaluation period is held out from fitting and hyperparameter decisions.

## 2. Related work and model motivation

Barlacchi et al. [1] describe the source data and document daily, weekly and spatial variation. These observations motivate checking temporal dependence and comparing areas, rather than assuming one shared traffic pattern. We retain the assignment's square IDs without assigning unverified land-use labels.

Azari et al. [2] compare LSTM and ARIMA on cellular packet traffic and show that the usefulness of model complexity depends on traffic regime, feature availability and training data. This motivates an LSTM and a simple statistical comparator. Our RidgeAR is a regularized autoregression, not a replication of their ARIMA; the sampling interval and target also differ.

Bai et al. [3] motivate causal dilated convolutions as a sequence-modeling alternative to recurrence. Our compact CausalCNN tests this mechanism without reproducing their full residual TCN. Their benchmark rankings are not assumed to hold for Milan. Zhang and Patras [4] exploit spatial information and longer horizons; their work motivates extensions beyond our single-area, one-step setting rather than a direct numerical comparison.

**Headline evidence:** on the highest-activity area, {top[0]}, the best learned reference model is {winner['best_learned']} (RMSE {winner['best_learned_rmse']:.2f}); persistence RMSE is {winner['persistence_rmse']:.2f}. Conclusions below include all selected areas and repeated seeds.''')
    b,o=memory['runs']
    pages.append(f'''## 3. Dataset, preparation and memory management

The publisher's daily files contain square ID, interval timestamp, country code, SMS-in/out, call-in/out and Internet activity. The 61 required files occupy {eda['raw_bytes']/1e9:.2f} GB and contain {eda['raw_rows']:,} rows. January 1, available in the collection, is excluded. Each raw file was checked against its published MD5 checksum. Timestamps were validated against a 10-minute grid with Europe/Rome calendar boundaries.

Internet activity is summed across country-code records for each square and interval. Empty Internet fields are not treated as observed zeros. An aggregate is missing only when no finite Internet value is present for that square and interval. The resulting grid has 8,784 time intervals per area; {eda['missing_square_time_bins']:,} of the 87,840,000 square-time bins are missing. Partial country-code coverage cannot be diagnosed as complete traffic from this dataset alone.

Chunked CSV loading projects only square, timestamp and Internet columns, uses int32 square IDs and retains float64 traffic. Fixed daily sum/count arrays require 17.28 MB. Processing one day at a time avoids holding the 20.48 GB raw collection in RAM. Checksums and date validation prevent accidental duplicate-day ingestion; they do not prove the original observations are error-free.

{table(['Full-day method','Peak process RAM (MB)','Wall time (s)','Largest DataFrame (MB)'],[[b['mode'],f"{b['peak_rss_bytes']/1e6:.1f}",f"{b['wall_seconds']:.2f}",f"{b['max_dataframe_bytes']/1e6:.1f}"],[o['mode'],f"{o['peak_rss_bytes']/1e6:.1f}",f"{o['wall_seconds']:.2f}",f"{o['max_dataframe_bytes']/1e6:.1f}"]])}

On the same complete November 1 file, separate processes show a {memory['peak_rss_reduction_percent']:.1f}% reduction in peak RAM. Missing masks and observation counts match exactly; the largest aggregate difference is {memory['max_abs_aggregation_difference']:.2g}, consistent with floating-point summation order. Timing is a single local run per method with uncontrolled filesystem cache, not a universal speedup claim. The raw files remain on disk; this trades disk space and repeated I/O for bounded memory.

![Figure 1. Total observed Internet activity by area over November–December. Missing values are excluded from totals; coverage is separately audited.](../results/figures/traffic_distribution.png)

The distribution is right-skewed: median total activity is {ranking.total_internet_activity.median():,.0f}, whereas the 99th percentile is {ranking.total_internet_activity.quantile(.99):,.0f} and the maximum is {ranking.total_internet_activity.max():,.0f}. A small upper tail carries much higher total activity than a typical area. The top three squares are **{', '.join(map(str,top))}**. They account for {eda['top3_share']*100:.2f}% of observed activity; the top 1% of areas account for {eda['top1_percent_share']*100:.2f}%. Ranking is by full-period totals as requested, and therefore uses evaluation-period activity for area selection, not for model fitting. This limits claims of prospective area selection.''')
    pages.append(f'''## 4. Exploratory and temporal analysis

![Figure 2. The required first-two-week series for the three highest-activity areas and squares 4159 and 4556. Axes retain each area's own activity scale; compare both level and shape.](../results/figures/first_two_weeks.png)

![Figure 3. Training-only autocorrelation and weekday/weekend daily profiles for area {top[0]}. Correlations use available paired observations without compressing time gaps.](../results/figures/temporal_analysis.png)

The series differ beyond their scale. Areas 5259 and 4159 show pronounced weekday plateaus and lower weekend activity, while area 5161 has stronger weekend peaks. Area 5059 has sustained daytime plateaus with short fluctuations; 4556 is noisier relative to its lower mean and contains isolated spikes. These differences are consistent with different activity schedules, but do not identify the land use or establish the cause of individual spikes.

Area {top[0]} has lag-one autocorrelation {eda['top_area_acf']['1']:.3f}, daily-lag correlation {eda['top_area_acf']['144']:.3f} and weekly-lag correlation {eda['top_area_acf']['1008']:.3f}. These distinguish immediate persistence from repeated daily/weekly patterns and motivate explicitly testing both short and full-day histories. Weekday/weekend differences in Figure 3 are descriptive; no location-specific causal explanation is inferred.''')
    chars=[]
    for a in eda['area_stats']:chars.append([a['square'],f"{a['train_mean']:.1f}",f"{a['train_cv']:.2f}",f"{a['train_acf_144']:.3f}",a['full_missing']])
    architecture=[]
    for kind,c in config['models'].items():
        detail=f"alpha={c['alpha']}" if kind=='RidgeAR' else (f"{c['width']} gated units" if kind=='LSTM' else f"{c['width']} filters; dilations {','.join(map(str,c['dilations']))}")
        if kind != 'RidgeAR': detail += f"; Adam lr={c['learning_rate']}; max epochs={c['epochs']}; patience={c['patience']}"
        architecture.append([kind,c['lookback'],detail])
    pages.append(f'''### 4.1 Statistical characterization

{table(['Area','Training mean','Training CV','Daily-lag corr.','Missing bins (full period)'],chars)}

CV is standard deviation divided by mean. Differences in level, variability and daily dependence justify comparing areas separately. A stronger daily profile does not itself imply that yesterday's value will beat the immediately previous value for a 10-minute horizon.

On the highest area's longest fully observed training segment ({eda['adf']['segment_n']:,} points), the ADF test gives statistic {eda['adf']['statistic']:.3f}, p={eda['adf']['pvalue']:.3g}, using {eda['adf']['lags']} selected lags. After first differencing, p={eda['difference_adf']['pvalue']:.3g}. The raw-series result rejects the unit-root null at 5% under the test's assumptions. This does not establish strict stationarity or rule out the clearly visible seasonality and changing traffic levels. A robust STL decomposition with period 144 yields daily seasonal strength {eda['stl_daily_strength']:.3f}; {eda['stl_interpolated_points']} missing training points were interpolated for this descriptive decomposition only. The complete STL figure is included in the repository.

## 5. Methodology

For target x(t+1), each input contains only x(t-L+1), ..., x(t). All candidates use identical eligible target timestamps, requiring a fully observed 144-step history even when L is shorter. Missing histories and targets are omitted and counted; model targets are never imputed. This conservative common mask avoids scoring different models on easier or harder subsets.

{table(['Split','Local dates','Purpose'],[['Training','Nov 1–Dec 8','Weights and normalization'],['Validation','Dec 9–15','Tuning and early stopping'],['Test','Dec 16–22','Frozen evaluation']])}

Each area's mean and standard deviation are fitted only on training values. Models learn the standardized correction (x(t+1)-x(t))/sigma. The forecast is x(t) plus sigma times the predicted correction, clipped at zero. This gives all models a persistence starting point. RidgeAR minimizes squared correction error plus an L2 coefficient penalty; it is linear in the lagged observations. LSTM consumes an L-by-1 sequence, uses tanh state activation and sigmoid gates, and passes the final hidden state to a scalar linear head; no dropout is applied. RidgeAR flattens the same normalized history into L lag features with a fitted intercept. CausalCNN stacks kernel-three causal ReLU convolutions and reads the final time position through a linear head. Its receptive field is 1 + 2 × sum(dilations).

{table(['Selected model','Lookback (steps)','Architecture / penalty'],architecture)}

The neural output head starts at zero. Adam uses gradient-norm clipping at 1, batch size 128 and deterministic seeded training-data shuffling. Validation MSE controls early stopping (minimum improvement 0.00001) and restores the best weights. Each area contributes 5,328 training targets and 1,008 validation targets after the common history requirement. Weights are fitted independently per area. Predictions are rolling one-step forecasts using newly observed history during the test week, not seven-day recursive forecasts.''')
    tuning_rows=[]
    for _,r in tuning.iterrows():tuning_rows.append([r.experiment,r.model,f'{r.val_rmse:.2f}',f'{r.train_seconds:.1f}'])
    pages.append(f'''## 6. Iterative experimentation

Hyperparameters were developed on the highest-activity area's validation period. Each new candidate was specified after reviewing the preceding validation results. The exact parameters, losses, predictions and rationale are saved under `configs/` and `results/runs/`; `tuning_log.csv` provides the consolidated record. Test metrics were not used to select the model settings.

{table(['Experiment','Model','Validation RMSE','Training seconds'],tuning_rows)}

**Final selection rationale:** {config['rationale']}

The selected configuration for each model is shared across all five areas, while weights and normalization are fitted per area. This tests transfer of hyperparameter choices; it is not exhaustive area-specific tuning. Final neural fits use seeds 42, 43 and 44. Seed 42 is the predesignated plot/reference run, not the best seed selected after testing. RidgeAR is deterministic; repeating its fit provides timing observations rather than independent model randomness. Repeated seeds describe optimization sensitivity, not confidence over future weeks.

![Figure 4. Training and validation loss for the final reference fits on the highest-activity area. Early stopping uses validation loss; test data never controls epochs.](../results/figures/learning_curves.png)

The project records the full training histories and best epochs. Model selection prioritizes validation RMSE because large misses matter for demand estimation; MAE and percentage error provide complementary evidence. All reported errors are inverse-transformed to original activity units.''')
    metric_tables=[]
    for area in top:
        rows=[[r.model,f'{r.mae:.2f}',f'{r.rmse:.2f}',f'{r.mape_percent_nonzero:.2f}'] for _,r in metrics[metrics.area==area].iterrows()]
        metric_tables.append(f'### Area {area}\n\n'+table(['Model (seed 42)','MAE','RMSE','MAPE (%)'],rows))
    metric_text='\n\n'.join(metric_tables)
    timing_rows=[]
    for r in study['timing_by_model']:timing_rows.append([r['model'],f"{r['train_seconds']:.2f}",f"{r['batch_inference_seconds']*1000:.2f}",f"{r['single_inference_ms']:.3f}"])
    pages.append(f'''## 7. Results and comparison

{metric_text}

MAE is the mean absolute error; RMSE is the square root of mean squared error; MAPE is 100 times the mean of |prediction - actual| / |actual| over nonzero actual values. MAPE excludes zero actual targets and records the excluded count in the CSV. It is undefined if every target is zero; no arbitrary epsilon is inserted. MAE and RMSE use every eligible target. Each test week nominally has 1,008 points; the per-area scored counts are {', '.join(str(w['area'])+': '+str(w['test_n']) for w in study['winners'])}.

{table(['Model','Training (s)','Full test batch (ms)','Single forecast (ms)'],timing_rows)}

Timing is averaged over five areas and three final fits per model on Apple M2 Pro, 16 GB RAM, CPU execution, four TensorFlow intra-operation threads. Neural training includes construction/compilation and validation; Ridge timing covers fitting. Warm inference is measured synchronously: median of five full-batch calls and median of 30 single-sample calls per fit, then averaged. Inference timing covers the model forward pass; window construction, scaling and correction reconstruction are excluded. These are local measurements with background system activity, not production throughput guarantees.''')
    for figure_number,area in enumerate(top, start=5):
        pages.append(f'''## Forecasts for area {area}

These are the three required model-specific comparisons for this area. Figures use the same observed targets and reference seed. Predictions use actual observations available up to the preceding interval.

![Figure {figure_number}a. RidgeAR, area {area}.](../results/figures/forecast_{area}_RidgeAR.png)

![Figure {figure_number}b. LSTM, area {area}.](../results/figures/forecast_{area}_LSTM.png)

![Figure {figure_number}c. CausalCNN, area {area}.](../results/figures/forecast_{area}_CausalCNN.png)''')
    winner_text=' '.join(f"Area {w['area']}: {w['best_learned']} is the best learned reference model; its RMSE is {abs(w['improvement_over_persistence_percent']):.1f}% {'below' if w['improvement_over_persistence_percent']>=0 else 'above'} persistence." for w in study['winners'])
    reference_area=metrics[metrics.area==top[0]].set_index('model')
    seed_area=seeds[seeds.area==top[0]].set_index('model')
    sensitivity=' '.join(f"{kind}: mean RMSE {seed_area.loc[kind,'rmse_mean']:.2f} (SD {seed_area.loc[kind,'rmse_std']:.2f})." for kind in ['RidgeAR','LSTM','CausalCNN'])
    traffic=pd.read_csv(R/'selected_series.csv',index_col='timestamp_ms')[str(failure['area'])]
    preceding=float(traffic.loc[failure['timestamp_ms']-600000]);following=float(traffic.loc[failure['timestamp_ms']+600000])
    failure_context=f"Observed activity changes from {preceding:.2f} in the preceding interval to {failure['actual']:.2f} at the miss, then to {following:.2f} in the next interval."
    speed=times.groupby('model').train_seconds.mean()
    cost_context=f"Average neural training takes {speed['LSTM']/speed['RidgeAR']:.0f} times as long for LSTM and {speed['CausalCNN']/speed['RidgeAR']:.0f} times as long for CausalCNN as RidgeAR on this machine. Linear lag weights therefore offer a low-cost baseline; LSTM gates and nonlinear convolution add flexibility, but require a measurable accuracy benefit."
    second=top[1];first_stats=next(a for a in eda['area_stats'] if a['square']==top[0]);second_stats=next(a for a in eda['area_stats'] if a['square']==second)
    second_scores=metrics[metrics.area==second].set_index('model')
    area_context=f"Area {second} has lower training variability (CV {second_stats['train_cv']:.2f} versus {first_stats['train_cv']:.2f}) and stronger daily dependence ({second_stats['train_acf_144']:.3f} versus {first_stats['train_acf_144']:.3f}) than area {top[0]}. Its RidgeAR RMSE is {second_scores.loc['RidgeAR','rmse']:.2f}, versus {second_scores.loc['LSTM','rmse']:.2f} for LSTM and {second_scores.loc['CausalCNN','rmse']:.2f} for CausalCNN. A regularized linear response is competitive on this more regular profile. These observations support evaluating model suitability by traffic regime, consistent with the caution motivated by [2]; five selected areas cannot establish a causal relationship between variability and architecture rankings."
    seed_rows=[]
    for _,r in seeds.iterrows():seed_rows.append([r.area,r.model,f'{r.rmse_mean:.2f}',f'{r.rmse_std:.2f}'])
    pages.append(f'''## 8. Discussion and failure analysis

{winner_text}

For area {top[0]}, repeated-seed results are: {sensitivity} The seed variation qualifies small differences between neural models. {cost_context} Strong adjacent-interval dependence can leave little improvement for a more expressive model at a one-step horizon. A lower error on one area is not evidence of universal superiority. Daily-seasonal persistence performs much worse than immediate persistence in all three primary areas, showing that strong periodicity alone does not make yesterday's level an adequate next-interval forecast. {area_context}

### 8.1 Failure case

![Figure 8. A local view of the largest seed-42 model error after normalizing by each area's training standard deviation. This case is selected for diagnosis after freezing the models.](../results/figures/failure_case.png)

The selected failure is area {failure['area']} at {pd.to_datetime(failure['timestamp_ms'],unit='ms',utc=True).tz_convert('Europe/Rome').strftime('%Y-%m-%d %H:%M %Z')}. {failure['model']} predicts {failure['prediction']:.2f} for an observed value of {failure['actual']:.2f}, an absolute error of {failure['abs_error']:.2f} ({failure['severity_train_std']:.2f} training standard deviations). {failure_context} Abrupt changes challenge all three models because they must infer the next interval from past activity alone; the surrounding curves show their response lag. Event context and neighboring-area measurements are unavailable to these univariate models, so the specific real-world cause cannot be established. The repository also reports error by actual-traffic decile to inspect systematic peak underprediction.

---PAGE---

## 9. Conclusion and future work

On the busiest area, {winner['best_learned']} gives the lowest seed-42 learned-model RMSE ({winner['best_learned_rmse']:.2f}), compared with persistence at {winner['persistence_rmse']:.2f}. The experiment also shows how rankings vary across areas and seeds. Complexity is justified only when it improves the relevant errors enough to offset training and execution costs; persistence remains a necessary benchmark. The model rankings and exact improvement percentages are reported above rather than assumed from previous literature.

Limitations include one held-out week, a small set of areas selected using full-period activity, shared hyperparameters tuned on one area, conservative missing-window exclusions, and limited random-seed repetitions. Follow-up work should use rolling-origin evaluation over more weeks, separate prospective area selection from evaluation, tune within computationally matched budgets, assess missingness explicitly, and add spatial/event covariates for abrupt changes. These extensions require new experiments; their benefits are not claimed here.''')
    pages.append(f'''## Appendix A. Sensitivity across seeds

{table(['Area','Model','RMSE mean (3 seeds)','RMSE sample SD'],seed_rows)}

Standard deviations summarize optimization variation over seeds 42, 43 and 44. They are not confidence intervals and do not account for the choice of test week or area. Supplementary model-specific plots for squares 4159 and 4556 are included under `results/figures/`, alongside all nine primary-area plots. All per-timestamp predictions are retained for independent checking.

## References

[1] G. Barlacchi et al., “A multi-source dataset of urban life in the city of Milan and the Province of Trentino,” Scientific Data, vol. 2, art. 150055, 2015, doi: 10.1038/sdata.2015.55. https://doi.org/10.1038/sdata.2015.55.

[2] A. Azari, P. Papapetrou, S. Denic, and G. Peters, “Cellular Traffic Prediction and Classification: a comparative evaluation of LSTM and ARIMA,” arXiv:1906.00939, 2019. https://arxiv.org/abs/1906.00939.

[3] S. Bai, J. Z. Kolter, and V. Koltun, “An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling,” arXiv:1803.01271, 2018. https://arxiv.org/abs/1803.01271.

[4] C. Zhang and P. Patras, “Long-Term Mobile Traffic Forecasting Using Deep Spatio-Temporal Neural Networks,” arXiv:1712.08083, 2017. https://arxiv.org/abs/1712.08083.

[5] Telecom Italia, “Telecommunications - SMS, Call, Internet - MI,” Harvard Dataverse, 2015. https://doi.org/10.7910/DVN/EGZHFV. Data under ODbL 1.0; [from BigDataChallenge contest](http://www.telecomitalia.com/tit/en/bigdatachallenge.html).

[6] Source code and reproducibility materials: [GitHub repository]({repo}).

## AI assistance disclosure

OpenAI Codex provided substantial assistance with code, experimental design, model implementation, testing, analysis, figures and report drafting. All reported numerical results were computed from the published dataset; synthetic data were used only for unit-test fixtures.''')
    output=ROOT/'output';output.mkdir(exist_ok=True)
    target=output/'report.md'
    if target.exists():raise FileExistsError('Do not overwrite an existing report; archive it explicitly first')
    text='\n\n---PAGE---\n\n'.join(pages).replace('\n\n---PAGE---\n\n## Appendix A.', '\n\n## Appendix A.')
    target.write_text(text+'\n')
    print(target)


if __name__=='__main__':main()
