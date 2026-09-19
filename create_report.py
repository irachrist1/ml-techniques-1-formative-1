"""Create an evidence-based research report; render_report.py builds its PDF."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
R = ROOT / 'results'

VIDEO_PLACEHOLDER = 'PASTE_VIDEO_LINK_HERE'


def table(headers, rows):
    return '\n'.join(
        ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
        + ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def local_time(ms, fmt='%Y-%m-%d %H:%M %Z'):
    return pd.to_datetime(ms, unit='ms', utc=True).tz_convert('Europe/Rome').strftime(fmt)


def main():
    eda = json.loads((R / 'eda_summary.json').read_text())
    study = json.loads((R / 'study_summary.json').read_text())
    memory = json.loads((R / 'memory_benchmark.json').read_text())
    config = json.loads((ROOT / 'configs/final.json').read_text())
    ranking = pd.read_csv(R / 'area_ranking.csv')
    metrics = pd.read_csv(R / 'reference_metrics.csv')
    tuning = pd.read_csv(R / 'tuning_log.csv')
    seeds = pd.read_csv(R / 'seed_summary.csv')
    times = pd.read_csv(R / 'timing.csv')
    val_test = pd.read_csv(R / 'validation_vs_test_ranking.csv')
    peak = pd.read_csv(R / 'peak_bias_summary.csv')

    top = eda['top3']
    areas = eda['areas']
    winner = study['winners'][0]
    failure = study['failure']
    budget = study['epoch_budget_audit']
    agreement = study['ranking_agreement']
    underprediction = study['peak_underprediction']
    params = study['parameters_by_model']
    robust = eda['coverage_robustness']
    stats = {s['square']: s for s in eda['area_stats']}
    repo = json.loads((ROOT / 'output/repository.json').read_text())['url']
    pages = []

    pages.append(f'''# One-step mobile-network traffic forecasting in Milan

**Christian Tonny | ML Techniques I | Formative Assignment 1**

**Demo video:** {VIDEO_PLACEHOLDER}

**Code:** {repo}

## 1. Introduction

Network operators have to decide where to put capacity before demand arrives, so a forecast of the next few minutes of traffic has a practical use. This report asks a narrow version of that question: given only an area's own recent history, how well can three different sequential models predict its Internet activity ten minutes ahead, and does the answer change from one area to another?

The value being predicted is the dataset's Internet activity count for the next 10-minute interval. It is not bandwidth and not a byte count, so the error figures below are in activity units and cannot be read as megabytes.

I process the whole November-December 2013 Milan collection, rank all 10,000 areas by total activity, and then compare a regularized linear autoregression, an LSTM and a causal dilated CNN on the busiest areas. Persistence and a daily-seasonal forecast are included so that any gain from the more complex models has something to beat. December 16-22 is held back and is never used to pick a model or a hyperparameter.

## 2. Related work and model motivation

Barlacchi et al. [1] describe this dataset and document its daily, weekly and spatial variation. That is the reason I check temporal dependence per area instead of assuming one shared pattern, and the reason I do not attach land-use labels to square IDs. The paper's well-known Bocconi and Navigli examples are squares 4259 and 4456, not the 4159 and 4556 this assignment asks for, so no interpretation transfers.

Azari et al. [2] compare LSTM and ARIMA on cellular packet traffic and find that whether complexity pays depends on the traffic regime, the features available and how much training data there is. That is what pushed me to keep a simple statistical comparator alongside the neural models rather than assuming the neural models win. My RidgeAR is not their ARIMA: it has no moving-average error terms and no differencing operator, it is a penalized linear fit on lagged values.

Bai et al. [3] argue that causal dilated convolutions can replace recurrence for sequence modeling. My CausalCNN uses that mechanism but not their full residual TCN, so their benchmark rankings should not be expected to carry over. Zhang and Patras [4] use spatial structure and longer horizons; that work sets out what my single-area, one-step setup is missing rather than giving me a number to compare against.

**Headline result:** on area {top[0]}, the busiest area, the best learned model at the reference seed is {winner['best_learned']} with RMSE {winner['best_learned_rmse']:.2f}, against persistence at {winner['persistence_rmse']:.2f}. Section 8 shows why I do not treat that as a general ranking.''')

    b, o = memory['runs']
    pages.append(f'''## 3. Dataset, preparation and memory management

Each daily file from the publisher holds square ID, interval timestamp, country code, SMS in/out, call in/out and Internet activity. The 61 files I need total {eda['raw_bytes'] / 1e9:.2f} GB and {eda['raw_rows']:,} rows. January 1 exists in the collection but falls outside the period, so it is excluded. Every file is checked against the publisher's MD5 before anything reads it, and timestamps are validated against a 10-minute grid using Europe/Rome calendar boundaries.

Internet activity is summed over the country-code rows for each square and interval. An empty Internet field is not an observed zero: a bin counts as missing only when no finite value exists for it at all. That leaves 8,784 intervals per area, with {eda['missing_square_time_bins']:,} of the 87,840,000 square-time bins missing. One caveat I cannot resolve from this data alone is that a bin can be observed but only partially covered across country codes, and there is no way to tell that apart from complete coverage.

The loader reads each file in chunks, keeps only square, timestamp and Internet, stores square IDs as int32 and leaves activity as float64. The accumulators are fixed-size sum and count arrays of 144 by 10,000, which is 17.28 MB regardless of how big the file is. Working one day at a time is what keeps the {eda['raw_bytes'] / 1e9:.2f} GB collection off the heap. Checksums and date validation also stop a day being ingested twice, though they say nothing about whether the original measurements are right.

{table(['Full-day method', 'Peak process RAM (MB)', 'Wall time (s)', 'Largest DataFrame (MB)'],
       [[b['mode'], f"{b['peak_rss_bytes'] / 1e6:.1f}", f"{b['wall_seconds']:.2f}", f"{b['max_dataframe_bytes'] / 1e6:.1f}"],
        [o['mode'], f"{o['peak_rss_bytes'] / 1e6:.1f}", f"{o['wall_seconds']:.2f}", f"{o['max_dataframe_bytes'] / 1e6:.1f}"]])}

Both methods run in their own process on the same complete November 1 file, and peak RSS drops by {memory['peak_rss_reduction_percent']:.1f}%. The comparison is only worth anything if the outputs match, so the benchmark asserts that: the missing masks are identical, the observation counts are identical, and the largest difference between the two aggregates is {memory['max_abs_aggregation_difference']:.2g}, which is summation order. The wall times are a single run each with an uncontrolled filesystem cache, so I would not quote the speedup. What this does trade away is disk and repeated I/O, since the raw files stay where they are.

![Figure 1. Total observed Internet activity by area over November-December. Totals exclude missing values, and coverage is audited separately.](../results/figures/traffic_distribution.png)

The distribution has a long right tail. Median total activity is {ranking.total_internet_activity.median():,.0f}, the 99th percentile is {ranking.total_internet_activity.quantile(.99):,.0f}, and the maximum is {ranking.total_internet_activity.max():,.0f}. The top three squares are **{', '.join(map(str, top))}**. They hold {eda['top3_share'] * 100:.2f}% of observed activity and the top 1% of areas hold {eda['top1_percent_share'] * 100:.2f}%.

Because totals only count observed bins, a badly covered area is understated and could in principle be ranked too low. {robust['areas_below_full_coverage']} areas have less than full coverage and the worst sits at {robust['minimum_coverage'] * 100:.1f}%. If I scale every one of them up by 1/coverage, which is the most generous correction available, the largest becomes {robust['largest_rescaled_partial_total']:,.0f}, still well below the {robust['top3_cutoff_total']:,.0f} needed to reach third place. So missingness did not change who is in the top three. Ranking on full-period totals does mean the selection used the evaluation week's activity, which is a selection dependence worth naming even though it never touches model fitting.''')

    a_top = stats[top[0]]
    pages.append(f'''## 4. Exploratory and temporal analysis

![Figure 2. First two weeks for the three busiest areas plus squares 4159 and 4556. Each panel keeps its own scale, so compare shape as well as level.](../results/figures/first_two_weeks.png)

![Figure 3. Training-period autocorrelation and weekday/weekend daily profiles for area {top[0]}. The dashed lines are the +/-1.96/sqrt(n) white-noise band.](../results/figures/temporal_analysis.png)

The five series differ in more than scale. Areas {top[2]} and 4159 sit low through the first weekend and then jump to a weekday plateau, and they drop again on 9-10 November. Area {top[0]} runs the other way, with its highest peaks at the weekend. Area {top[1]} holds a broad daytime plateau with short bursts, and 4556 is the noisiest relative to its own mean and carries isolated one-interval spikes.

The weekday/weekend split is measurable, not just visible. Over the training dates, area {top[0]} averages {a_top['train_weekend_mean']:.0f} at weekends against {a_top['train_weekday_mean']:.0f} on weekdays, a ratio of {a_top['weekend_over_weekday']:.2f}, while area {top[2]} is at {stats[top[2]]['weekend_over_weekday']:.2f} and 4159 at {stats[4159]['weekend_over_weekday']:.2f}. One calendar detail matters for reading the left edge of Figure 2: 1 November is All Saints' Day, a public holiday in Italy, and it fell on a Friday in 2013. So the first low stretch is a three-day holiday weekend rather than an ordinary one, and the 9-10 November weekend is the cleaner comparison. These are activity schedules; I cannot tell from this data what land use produces them or what caused any individual spike.

For area {top[0]}, correlation with the previous interval is {eda['top_area_acf']['1']:.3f}, with the same time yesterday {eda['top_area_acf']['144']:.3f}, and with the same time last week {eda['top_area_acf']['1008']:.3f}. All three are far above the {eda['acf_white_noise_band_95']:.3f} white-noise band, so none of them is in doubt. The interesting part is the gap between them: the 10-minute lag is much stronger than the daily lag, which is a direct argument for giving the models recent history rather than relying on the daily shape. Correlation at a six-hour lag is {eda['top_area_acf']['36']:.3f}, effectively nothing, which is the half-day trough of the daily cycle.''')

    characteristics = [[s, f"{stats[s]['train_mean']:.1f}", f"{stats[s]['train_cv']:.2f}",
                        f"{stats[s]['train_acf_1']:.3f}", f"{stats[s]['train_acf_144']:.3f}",
                        f"{stats[s]['weekend_over_weekday']:.2f}", stats[s]['full_missing']] for s in areas]
    pages.append(f'''### 4.1 Statistical characterization

{table(['Area', 'Training mean', 'Training CV', 'Lag-1 corr.', 'Daily-lag corr.', 'Weekend/weekday', 'Missing bins'], characteristics)}

CV is standard deviation over mean. The areas differ in level, in variability and in how much of their structure is daily rather than immediate, which is why I fit and score them separately. Note that a strong daily correlation does not mean yesterday's value is a good 10-minute forecast, and Section 7 shows exactly that.

On the longest fully observed training stretch for area {top[0]} ({eda['adf']['segment_n']:,} points) the ADF statistic is {eda['adf']['statistic']:.3f} with p = {eda['adf']['pvalue']:.2e}, using {eda['adf']['lags']} lags chosen by AIC from a ceiling of {eda['adf']['max_lag_allowed']}. I set that ceiling above 144 on purpose, because a lag window shorter than one day cannot span the seasonal cycle the test is being asked to look through. After first differencing, p = {eda['difference_adf']['pvalue']:.0e}. The unit-root null is rejected, but that is all it is: the series plainly has a daily cycle and shifting levels, and ADF does not contradict either. A robust STL with period 144 puts the daily seasonal strength at {eda['stl_daily_strength']:.3f}, and {eda['stl_interpolated_points']} training points needed interpolation for that decomposition, which is descriptive only and never feeds a model.

![Figure 4. Robust STL of the training period for area {top[0]}. The daily component dominates and the trend moves slowly, which is the pattern the models have to exploit.](../results/figures/stl_training.png)''')

    selected_rows = [['RidgeAR', config['models']['RidgeAR']['lookback'],
                      f"alpha={config['models']['RidgeAR']['alpha']}; {params.get('RidgeAR', 0)} parameters"],
                     ['LSTM', config['models']['LSTM']['lookback'],
                      f"{config['models']['LSTM']['width']} gated units; Adam lr={config['models']['LSTM']['learning_rate']}; "
                      f"max epochs={config['models']['LSTM']['epochs']}; patience={config['models']['LSTM']['patience']}; "
                      f"{params.get('LSTM', 0)} parameters"],
                     ['CausalCNN', config['models']['CausalCNN']['lookback'],
                      f"{config['models']['CausalCNN']['width']} filters; dilations "
                      f"{','.join(map(str, config['models']['CausalCNN']['dilations']))}; Adam lr={config['models']['CausalCNN']['learning_rate']}; "
                      f"max epochs={config['models']['CausalCNN']['epochs']}; patience={config['models']['CausalCNN']['patience']}; "
                      f"{params.get('CausalCNN', 0)} parameters"]]
    pages.append(f'''## 5. Methodology

To predict x(t+1) a model sees only x(t-L+1) through x(t). Every candidate is required to have a fully observed 144-step history before a timestamp counts as eligible, even when its own lookback is shorter. That is deliberate: without it a short-lookback model would qualify on more timestamps and could win by being scored on an easier subset. Windows with a missing value or a time gap are dropped and counted, and no target is ever imputed.

{table(['Split', 'Local dates', 'Purpose'],
       [['Training', 'Nov 1 - Dec 8', 'Weights and normalization'],
        ['Validation', 'Dec 9 - 15', 'Tuning and early stopping'],
        ['Test', 'Dec 16 - 22', 'Frozen evaluation']])}

Each area's mean and standard deviation come from its training values only. What the models actually learn is the standardized change, (x(t+1) - x(t))/sigma, and the forecast is x(t) plus sigma times the predicted change, clipped at zero. Every model therefore starts from persistence and only has to learn the correction. This is worth flagging early because it shapes the failure mode in Section 8.3: if the next interval is an isolated spike, the correction that minimizes squared error over the training set is close to zero, so the miss is built into the parameterization rather than being a tuning problem.

RidgeAR flattens the normalized history into L lag features and minimizes squared error plus an L2 penalty, with a fitted intercept. The LSTM reads the L-by-1 sequence with tanh state activation and sigmoid gates and passes its final hidden state to a linear scalar head, with no dropout. CausalCNN stacks kernel-3 causal ReLU convolutions with increasing dilation and reads the last time position through a linear head.

{table(['Selected model', 'Lookback (steps)', 'Architecture / penalty'], selected_rows)}

The CNN's receptive field is 1 + 2 x sum(dilations) = {1 + 2 * sum(config['models']['CausalCNN']['dilations'])} steps, which is more than the {config['models']['CausalCNN']['lookback']}-step input it is given. The last dilation stages therefore reach past the start of the window and add parameters without adding history, and trimming them would be the first thing I would try next.

The neural output heads are initialized at zero. Adam uses gradient-norm clipping at 1, batch size 128, and seeded shuffling of the training windows. Shuffling here is safe because each window is built causally before shuffling, so reordering independent examples cannot move information backwards in time. Validation MSE drives early stopping, which restores the best weights. Each area gives 5,328 training targets and 1,008 validation targets after the common-history rule. Weights and normalization are fitted per area. The test-week predictions are rolling one-step forecasts that use observations already available up to the previous interval; they are not a seven-day recursive forecast from a single origin.''')

    experiment_rows = [[r.experiment, r.model, f'{r.val_rmse:.2f}', int(r.epochs_run), f'{r.train_seconds:.1f}']
                       for r in tuning.itertuples()]
    pages.append(f'''## 6. Iterative experimentation

All tuning happened on area {top[0]}'s validation week. Each round was specified after reading the previous round's results, and the reasoning for every change is stored in the `configs/*.json` file that produced it, with the consolidated record in `results/tuning_log.csv`. Test data was never scored during tuning.

{table(['Experiment', 'Model', 'Validation RMSE', 'Epochs run', 'Training seconds'], experiment_rows)}

Round 1 used six hours of history. Round 2 extended it to a full day, which improved all three models, so I kept it. For the CNN I also had to extend the dilations to cover the longer window, so that round changes two things at once for that model and is not a clean single-variable comparison. Round 3 tested capacity: raising RidgeAR's alpha from 1 to 10 helped slightly, doubling the LSTM to 32 units made it worse so I kept 16, and doubling the CNN to 16 filters helped slightly so I kept that.

Round 4 came out of an audit rather than a hunch. Checking `epochs_run` against the epoch cap in the completed fits showed the cap, not the patience rule, was ending most neural training. On area {top[0]} the models stop on their own inside 20 epochs, which is exactly why three rounds of tuning on that area never exposed the problem. Raising the cap to {config['models']['LSTM']['epochs']} leaves area {top[0]}'s validation RMSE unchanged and lets the other areas train until early stopping actually fires. After the change, {budget['stopped_early']} of {budget['neural_fits']} neural fits stop early and the longest runs {budget['max_epochs_run']} epochs. I am treating this as fixing an arbitrary constraint rather than as tuning, since a budget that cuts training off before the stopping rule triggers is not a considered choice.

One thing the table shows that I should not gloss over: RidgeAR has the lowest validation RMSE in every single round. Section 8 returns to this, because it does not match what happens on the test week.

The selected configuration is shared across all five areas while weights and normalization are fitted per area, so this tests whether hyperparameters transfer, not what each area's best settings would be. Final neural fits use seeds 42, 43 and 44, with 42 fixed in advance as the plotting and reference run rather than picked afterwards. RidgeAR is deterministic, so repeating it only varies the timing.

![Figure 5. Training and validation loss for the reference fits on area {top[0]}. Early stopping watches validation loss; test data never influences the number of epochs.](../results/figures/learning_curves.png)

Model selection leans on validation RMSE because large misses are what matter for capacity decisions, with MAE and MAPE as secondary evidence. All errors are converted back to activity units before being reported.''')

    results_pages = []
    for area in top:
        area_rows = []
        for model in ['RidgeAR', 'LSTM', 'CausalCNN', 'Persistence', 'Daily seasonal']:
            row = metrics[(metrics.area == area) & (metrics.model == model)].iloc[0]
            area_rows.append([model, f'{row.mae:.2f}', f'{row.rmse:.2f}', f'{row.mape_percent_nonzero:.2f}'])
        results_pages.append(f'''### Area {area}

{table([f'Model (seed {study["reference_seed"]})', 'MAE', 'RMSE', 'MAPE (%)'], area_rows)}''')
    timing_rows = [[t.model, f'{t.train_seconds:.2f}', f'{1000 * t.batch_inference_seconds:.2f}',
                    f'{t.single_inference_ms:.3f}']
                   for t in pd.DataFrame(study['timing_by_model']).itertuples()]
    scored = ', '.join(f'{w["area"]}: {w["test_n"]}' for w in study['winners'])
    pages.append('## 7. Results and comparison\n\n' + '\n\n'.join(results_pages) + f'''

MAE is the mean absolute error, RMSE is the square root of mean squared error, and MAPE is 100 times the mean of |prediction - actual| / |actual| over nonzero targets. MAPE excludes zero targets and the excluded count is written to the CSV; no epsilon is substituted. MAE and RMSE use every eligible target. The test week has 1,008 intervals and all of them are scored for every area ({scored}).

{table(['Model', 'Training (s)', 'Full test batch (ms)', 'Single forecast (ms)'], timing_rows)}

Timing is averaged over five areas and three fits per model on an Apple M2 Pro with 16 GB RAM, CPU only, four TensorFlow intra-op threads. The two scopes are not identical and should not be compared as if they were: neural training time includes building and compiling the graph and running validation each epoch, while the Ridge figure is the solve alone. Inference is measured warm and synchronously, as the median of five full-batch calls and of 30 single-sample calls per fit. It covers the forward pass only, so window construction, scaling and reconstructing the forecast are excluded for all three models.''')

    for i, area in enumerate(top):
        letter = chr(ord('a'))
        figure_number = 6 + i
        pages.append(f'''## Forecasts for area {area}

These are the three required model comparisons for this area. All three use the same observed targets and the same reference seed, and each prediction uses only observations available up to the previous interval.

![Figure {figure_number}{letter}. RidgeAR, area {area}.](../results/figures/forecast_{area}_RidgeAR.png)

![Figure {figure_number}{chr(ord("a") + 1)}. LSTM, area {area}.](../results/figures/forecast_{area}_LSTM.png)

![Figure {figure_number}{chr(ord("a") + 2)}. CausalCNN, area {area}.](../results/figures/forecast_{area}_CausalCNN.png)''')

    improvements = ' '.join(
        f"Area {w['area']}: {w['best_learned']}, {w['improvement_over_persistence_percent']:.1f}% below persistence."
        for w in study['winners'])
    top_seed = seeds[seeds.area == top[0]]
    seed_text = ' '.join(f"{r.model}: mean RMSE {r.rmse_mean:.2f} (SD {r.rmse_std:.2f})." for r in top_seed.itertuples())
    ratio = {t['model']: t['train_seconds'] for t in study['timing_by_model']}
    vt_rows = [[int(r.area), r.validation_best, r.test_best,
                f'{r.val_RidgeAR:.2f} / {r.test_RidgeAR:.2f}',
                f'{r.val_LSTM:.2f} / {r.test_LSTM:.2f}',
                f'{r.val_CausalCNN:.2f} / {r.test_CausalCNN:.2f}'] for r in val_test.itertuples()]
    peak_top = peak[peak.area == top[0]]
    peak_rows = [[r.model, f'{r.lowest_decile_bias:+.1f}', f'{r.highest_decile_bias:+.1f}',
                  f'{r.highest_decile_mae:.1f}'] for r in peak_top.itertuples()]
    beats = []
    for w in study['winners']:
        a = metrics[(metrics.area == w['area']) & (metrics.model.isin(['RidgeAR', 'LSTM', 'CausalCNN']))]
        persist = float(metrics[(metrics.area == w['area']) & (metrics.model == 'Persistence')].rmse.iloc[0])
        beats.extend([(w['area'], r.model, float(r.rmse), persist) for r in a.itertuples() if r.rmse >= persist])
    if beats:
        misses = '; '.join(f'area {a} {m} at {r:.2f} against persistence {q:.2f}' for a, m, r, q in beats)
        beat_text = (f'{15 - len(beats)} of the 15 area-model fits beat persistence at the reference seed. '
                     f'The exception is {misses}, which is a tie rather than a loss but is worth naming.')
    else:
        beat_text = 'All 15 area-model fits beat persistence at the reference seed.'
    pages.append(f'''## 8. Discussion

{beat_text} {improvements}

Repeated seeds on area {top[0]} give: {seed_text} That spread is the right context for the differences between the two neural models, which are smaller than it. Average neural training costs {ratio['LSTM'] / ratio['RidgeAR']:.0f} times as long for the LSTM and {ratio['CausalCNN'] / ratio['RidgeAR']:.0f} times as long for the CNN as RidgeAR on this machine, against {params.get('LSTM', 0)} and {params.get('CausalCNN', 0)} parameters versus {params.get('RidgeAR', 0)}. Given how strong the 10-minute dependence is, there is not much room left above persistence for a more expressive model to claim at this horizon.

The daily-seasonal baseline is much worse than plain persistence everywhere, which is the clearest result in the study. Area {top[0]} has a daily-lag correlation of {stats[top[0]]['train_acf_144']:.3f} and yesterday's value is still a bad forecast for ten minutes ahead. Strong periodicity tells you about shape, not about level at short range.

### 8.1 Validation and test do not agree

{table(['Area', 'Validation best', 'Test best', 'RidgeAR val/test', 'LSTM val/test', 'CausalCNN val/test'], vt_rows)}

The two weeks disagree about the winner in {agreement['disagree']} of {agreement['areas']} areas. This is the result I trust least in the whole study and the one I would chase first. On area {top[0]} RidgeAR is clearly ahead on the validation week and clearly behind on the test week, and the gap in each direction is larger than the seed spread, so it is not optimization noise. With one tuning week and one test week there is no way to tell from this experiment whether the test week or the validation week is the unusual one. It does mean the headline in Section 2 should be read as "what happened on 16-22 December", not as a ranking of the three architectures.

### 8.2 All three models shrink toward the recent level

{table([f'Model (area {top[0]})', 'Lowest-decile bias', 'Highest-decile bias', 'Highest-decile MAE'], peak_rows)}

Splitting the test errors by actual traffic decile shows the same shape almost everywhere: {underprediction['negative_highest_decile_bias']} of {underprediction['combinations']} area-model combinations under-forecast the busiest decile, {underprediction['positive_lowest_decile_bias']} of {underprediction['combinations']} over-forecast the quietest, and the worst case is {underprediction['worst_highest_decile_bias']:.1f}. So the models mostly shrink toward the recent level, compressing both ends of the range. That follows from what they are trained to do. The target is a correction to persistence, the correction is fitted to minimize squared error across mostly ordinary intervals, and the result is a conservative estimate that lags any sharp move in either direction. It is also why the peaks, which are the intervals a capacity decision would actually care about, are the ones the models handle worst.

### 8.3 Failure case

![Figure 9. The largest seed-{study['reference_seed']} miss after scaling each area's error by its training standard deviation. Chosen after the models were frozen.](../results/figures/failure_case.png)

The worst standardized miss is area {failure['area']} at {local_time(failure['timestamp_ms'])}. {failure['model']} predicts {failure['prediction']:.2f} against an observed {failure['actual']:.2f}, an error of {failure['abs_error']:.2f}, or {failure['severity_train_std']:.2f} training standard deviations. Activity goes {failure['previous_actual']:.2f} then {failure['actual']:.2f} then {failure['next_step']['actual']:.2f}: a single interval spike with nothing in the preceding history that points to it.

Two things are worth taking from this. First, the miss is structural rather than a tuning failure, for the reason given in Section 5 — with a persistence-correction target the loss-minimizing response to an unpredictable one-interval jump is to predict roughly no change. Second, look at the step after the spike. All three models over-predict it ({', '.join(f"{k} {failure['next_step'][k]:.0f}" for k in ['RidgeAR', 'LSTM', 'CausalCNN'])} against an actual of {failure['next_step']['actual']:.0f}), because the spike is now in their input and the persistence anchor carries it forward. So a single outlier costs two errors, not one. A univariate model has no event context and no neighbouring-area signal, so I cannot say what caused the spike, and with one interval involved I cannot rule out a measurement artifact either.''')

    seed_rows = [[int(r.area), r.model, f'{r.rmse_mean:.2f}', f'{r.rmse_std:.2f}'] for r in seeds.itertuples()]
    pages.append(f'''## 9. Conclusion and future work

On the busiest area, {winner['best_learned']} gives the lowest reference-seed RMSE at {winner['best_learned_rmse']:.2f} against persistence at {winner['persistence_rmse']:.2f}. {14 - len(beats) + 1 if beats else 15} of the 15 area-model fits beat persistence, the one exception being a tie. The daily-seasonal baseline is far behind everywhere, which says the useful signal at a 10-minute horizon is recent level rather than daily shape.

I am less confident about the ranking between the three models than about that. Validation and test disagree on {agreement['disagree']} of {agreement['areas']} areas, the differences between the two neural models are inside the seed spread, and RidgeAR wins on validation in every tuning round while losing on test for the busiest area. What the evidence does support is that a 145-parameter linear model is a serious competitor here, and that anything more expensive needs to show a gain that survives more than one week before it is worth the training cost.

The main limitations are one held-out week, five areas chosen using full-period totals, hyperparameters tuned on a single area and single seed, and three seeds that measure optimization variance rather than uncertainty about future weeks. The epoch cap in the first three rounds is a concrete example of how a constraint set on one area can quietly distort the others, and I only caught it by auditing `epochs_run` against the budget. Next steps I would actually run: rolling-origin evaluation across several weeks so the ranking question can be settled, area selection made prospectively from a period before the evaluation window, matched compute budgets per model, trimming the CNN dilations to the input length, and adding neighbouring-area inputs to see whether spikes like the one in Section 8.3 are predictable from space even when they are not predictable from time.

## Appendix A. Sensitivity across seeds

{table(['Area', 'Model', 'RMSE mean (3 seeds)', 'RMSE sample SD'], seed_rows)}

These standard deviations cover seeds {', '.join(map(str, [42, 43, 44]))} and describe how much the optimizer moves, nothing else. They are not confidence intervals and they say nothing about a different test week or a different area. Model-specific plots for squares 4159 and 4556 are in `results/figures/` alongside the nine primary-area plots, and every per-timestamp prediction is kept so the numbers can be rechecked independently.

## References

[1] G. Barlacchi et al., "A multi-source dataset of urban life in the city of Milan and the Province of Trentino," Scientific Data, vol. 2, art. 150055, 2015, doi: 10.1038/sdata.2015.55. https://doi.org/10.1038/sdata.2015.55.

[2] A. Azari, P. Papapetrou, S. Denic, and G. Peters, "Cellular Traffic Prediction and Classification: a comparative evaluation of LSTM and ARIMA," arXiv:1906.00939, 2019. https://arxiv.org/abs/1906.00939.

[3] S. Bai, J. Z. Kolter, and V. Koltun, "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling," arXiv:1803.01271, 2018. https://arxiv.org/abs/1803.01271.

[4] C. Zhang and P. Patras, "Long-Term Mobile Traffic Forecasting Using Deep Spatio-Temporal Neural Networks," arXiv:1712.08083, 2017. https://arxiv.org/abs/1712.08083.

[5] Telecom Italia, "Telecommunications - SMS, Call, Internet - MI," Harvard Dataverse, 2015. https://doi.org/10.7910/DVN/EGZHFV. Data under ODbL 1.0; [from BigDataChallenge contest](http://www.telecomitalia.com/tit/en/bigdatachallenge.html).

[6] Source code and reproducibility materials: [GitHub repository]({repo}).

[7] Demo video: {VIDEO_PLACEHOLDER}''')

    output = ROOT / 'output'
    output.mkdir(exist_ok=True)
    target = output / 'report.md'
    if target.exists():
        raise FileExistsError('Do not overwrite an existing report; archive it explicitly first')
    text = '\n\n---PAGE---\n\n'.join(pages)
    for seam in ['## Appendix A.', '### 4.1 Statistical characterization']:
        text = text.replace('\n\n---PAGE---\n\n' + seam, '\n\n' + seam)
    target.write_text(text + '\n')
    print(target)


if __name__ == '__main__':
    main()
