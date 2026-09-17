# One-step mobile-network traffic forecasting in Milan

**Christian Tonny | ML Techniques I | Formative Assignment 1**

## 1. Introduction

Mobile-network operators need estimates of near-future demand to inform resource allocation. This study asks how three different sequential models compare for one-step-ahead Internet-activity forecasting, and whether their performance changes across geographical areas. The target is the next 10-minute activity value, not a directly measured bandwidth or byte count.

The study processes the complete November–December 2013 Milan collection, identifies the highest-activity areas, and compares regularized linear autoregression, gated recurrent memory and causal convolution. Persistence and daily-seasonal forecasts establish whether added model complexity is useful. The December 16–22 evaluation period is held out from fitting and hyperparameter decisions.

## 2. Related work and model motivation

Barlacchi et al. [1] describe the source data and document daily, weekly and spatial variation. These observations motivate checking temporal dependence and comparing areas, rather than assuming one shared traffic pattern. We retain the assignment's square IDs without assigning unverified land-use labels.

Azari et al. [2] compare LSTM and ARIMA on cellular packet traffic and show that the usefulness of model complexity depends on traffic regime, feature availability and training data. This motivates an LSTM and a simple statistical comparator. Our RidgeAR is a regularized autoregression, not a replication of their ARIMA; the sampling interval and target also differ.

Bai et al. [3] motivate causal dilated convolutions as a sequence-modeling alternative to recurrence. Our compact CausalCNN tests this mechanism without reproducing their full residual TCN. Their benchmark rankings are not assumed to hold for Milan. Zhang and Patras [4] exploit spatial information and longer horizons; their work motivates extensions beyond our single-area, one-step setting rather than a direct numerical comparison.

**Headline evidence:** on the highest-activity area, 5161, the best learned reference model is CausalCNN (RMSE 118.16); persistence RMSE is 134.88. Conclusions below include all selected areas and repeated seeds.

---PAGE---

## 3. Dataset, preparation and memory management

The publisher's daily files contain square ID, interval timestamp, country code, SMS-in/out, call-in/out and Internet activity. The 61 required files occupy 20.48 GB and contain 314,966,126 rows. January 1, available in the collection, is excluded. Each raw file was checked against its published MD5 checksum. Timestamps were validated against a 10-minute grid with Europe/Rome calendar boundaries.

Internet activity is summed across country-code records for each square and interval. Empty Internet fields are not treated as observed zeros. An aggregate is missing only when no finite Internet value is present for that square and interval. The resulting grid has 8,784 time intervals per area; 145,354 of the 87,840,000 square-time bins are missing. Partial country-code coverage cannot be diagnosed as complete traffic from this dataset alone.

Chunked CSV loading projects only square, timestamp and Internet columns, uses int32 square IDs and retains float64 traffic. Fixed daily sum/count arrays require 17.28 MB. Processing one day at a time avoids holding the 20.48 GB raw collection in RAM. Checksums and date validation prevent accidental duplicate-day ingestion; they do not prove the original observations are error-free.

| Full-day method | Peak process RAM (MB) | Wall time (s) | Largest DataFrame (MB) |
| --- | --- | --- | --- |
| baseline | 917.8 | 2.23 | 309.9 |
| chunked | 124.6 | 1.60 | 2.0 |

On the same complete November 1 file, separate processes show a 86.4% reduction in peak RAM. Missing masks and observation counts match exactly; the largest aggregate difference is 1.8e-12, consistent with floating-point summation order. Timing is a single local run per method with uncontrolled filesystem cache, not a universal speedup claim. The raw files remain on disk; this trades disk space and repeated I/O for bounded memory.

![Figure 1. Total observed Internet activity by area over November–December. Missing values are excluded from totals; coverage is separately audited.](../results/figures/traffic_distribution.png)

The distribution is right-skewed: median total activity is 274,706, whereas the 99th percentile is 4,666,202 and the maximum is 12,682,673. A small upper tail carries much higher total activity than a typical area. The top three squares are **5161, 5059, 5259**. They account for 0.62% of observed activity; the top 1% of areas account for 11.09%. Ranking is by full-period totals as requested, and therefore uses evaluation-period activity for area selection, not for model fitting. This limits claims of prospective area selection.

---PAGE---

## 4. Exploratory and temporal analysis

![Figure 2. The required first-two-week series for the three highest-activity areas and squares 4159 and 4556. Axes retain each area's own activity scale; compare both level and shape.](../results/figures/first_two_weeks.png)

![Figure 3. Training-only autocorrelation and weekday/weekend daily profiles for area 5161. Correlations use available paired observations without compressing time gaps.](../results/figures/temporal_analysis.png)

The series differ beyond their scale. Areas 5259 and 4159 show pronounced weekday plateaus and lower weekend activity, while area 5161 has stronger weekend peaks. Area 5059 has sustained daytime plateaus with short fluctuations; 4556 is noisier relative to its lower mean and contains isolated spikes. These differences are consistent with different activity schedules, but do not identify the land use or establish the cause of individual spikes.

Area 5161 has lag-one autocorrelation 0.982, daily-lag correlation 0.894 and weekly-lag correlation 0.952. These distinguish immediate persistence from repeated daily/weekly patterns and motivate explicitly testing both short and full-day histories. Weekday/weekend differences in Figure 3 are descriptive; no location-specific causal explanation is inferred.

---PAGE---

### 4.1 Statistical characterization

| Area | Training mean | Training CV | Daily-lag corr. | Missing bins (full period) |
| --- | --- | --- | --- | --- |
| 5161 | 1511.4 | 0.92 | 0.894 | 0 |
| 5059 | 1342.6 | 0.71 | 0.912 | 0 |
| 5259 | 1314.1 | 0.85 | 0.669 | 0 |
| 4159 | 317.2 | 0.59 | 0.743 | 0 |
| 4556 | 582.4 | 0.43 | 0.769 | 0 |

CV is standard deviation divided by mean. Differences in level, variability and daily dependence justify comparing areas separately. A stronger daily profile does not itself imply that yesterday's value will beat the immediately previous value for a 10-minute horizon.

On the highest area's longest fully observed training segment (5,472 points), the ADF test gives statistic -14.934, p=1.35e-27, using 30 selected lags. After first differencing, p=1e-20. The raw-series result rejects the unit-root null at 5% under the test's assumptions. This does not establish strict stationarity or rule out the clearly visible seasonality and changing traffic levels. A robust STL decomposition with period 144 yields daily seasonal strength 0.870; 0 missing training points were interpolated for this descriptive decomposition only. The complete STL figure is included in the repository.

## 5. Methodology

For target x(t+1), each input contains only x(t-L+1), ..., x(t). All candidates use identical eligible target timestamps, requiring a fully observed 144-step history even when L is shorter. Missing histories and targets are omitted and counted; model targets are never imputed. This conservative common mask avoids scoring different models on easier or harder subsets.

| Split | Local dates | Purpose |
| --- | --- | --- |
| Training | Nov 1–Dec 8 | Weights and normalization |
| Validation | Dec 9–15 | Tuning and early stopping |
| Test | Dec 16–22 | Frozen evaluation |

Each area's mean and standard deviation are fitted only on training values. Models learn the standardized correction (x(t+1)-x(t))/sigma. The forecast is x(t) plus sigma times the predicted correction, clipped at zero. This gives all models a persistence starting point. RidgeAR minimizes squared correction error plus an L2 coefficient penalty; it is linear in the lagged observations. LSTM consumes an L-by-1 sequence, uses tanh state activation and sigmoid gates, and passes the final hidden state to a scalar linear head; no dropout is applied. RidgeAR flattens the same normalized history into L lag features with a fitted intercept. CausalCNN stacks kernel-three causal ReLU convolutions and reads the final time position through a linear head. Its receptive field is 1 + 2 × sum(dilations).

| Selected model | Lookback (steps) | Architecture / penalty |
| --- | --- | --- |
| RidgeAR | 144 | alpha=10 |
| LSTM | 144 | 16 gated units; Adam lr=0.001; max epochs=20; patience=4 |
| CausalCNN | 144 | 16 filters; dilations 1,2,4,8,16,32,64; Adam lr=0.001; max epochs=20; patience=4 |

The neural output head starts at zero. Adam uses gradient-norm clipping at 1, batch size 128 and deterministic seeded training-data shuffling. Validation MSE controls early stopping (minimum improvement 0.00001) and restores the best weights. Each area contributes 5,328 training targets and 1,008 validation targets after the common history requirement. Weights are fitted independently per area. Predictions are rolling one-step forecasts using newly observed history during the test week, not seven-day recursive forecasts.

---PAGE---

## 6. Iterative experimentation

Hyperparameters were developed on the highest-activity area's validation period. Each new candidate was specified after reviewing the preceding validation results. The exact parameters, losses, predictions and rationale are saved under `configs/` and `results/runs/`; `tuning_log.csv` provides the consolidated record. Test metrics were not used to select the model settings.

| Experiment | Model | Validation RMSE | Training seconds |
| --- | --- | --- | --- |
| initial | RidgeAR | 159.07 | 0.0 |
| initial | LSTM | 163.30 | 4.4 |
| initial | CausalCNN | 161.02 | 5.5 |
| daily_history | RidgeAR | 153.13 | 0.0 |
| daily_history | LSTM | 160.26 | 20.4 |
| daily_history | CausalCNN | 160.28 | 12.9 |
| capacity | RidgeAR | 152.39 | 0.3 |
| capacity | LSTM | 162.04 | 18.5 |
| capacity | CausalCNN | 160.09 | 27.2 |

**Final selection rationale:** Full-day history improved all three validation RMSEs over six-hour history. Raising RidgeAR alpha from 1 to 10 improved RMSE from 153.13 to 152.39. Doubling LSTM width worsened RMSE from 160.26 to 162.04, so retain 16 units. Doubling CNN width improved RMSE slightly from 160.28 to 160.09, so select 16 filters. Each model uses its minimum-validation-RMSE configuration across three sequential rounds; these settings were frozen before computing test results. The small CNN validation gain should not be interpreted as statistically established superiority.

The selected configuration for each model is shared across all five areas, while weights and normalization are fitted per area. This tests transfer of hyperparameter choices; it is not exhaustive area-specific tuning. Final neural fits use seeds 42, 43 and 44. Seed 42 is the predesignated plot/reference run, not the best seed selected after testing. RidgeAR is deterministic; repeating its fit provides timing observations rather than independent model randomness. Repeated seeds describe optimization sensitivity, not confidence over future weeks.

![Figure 4. Training and validation loss for the final reference fits on the highest-activity area. Early stopping uses validation loss; test data never controls epochs.](../results/figures/learning_curves.png)

The project records the full training histories and best epochs. Model selection prioritizes validation RMSE because large misses matter for demand estimation; MAE and percentage error provide complementary evidence. All reported errors are inverse-transformed to original activity units.

---PAGE---

## 7. Results and comparison

### Area 5161

| Model (seed 42) | MAE | RMSE | MAPE (%) |
| --- | --- | --- | --- |
| RidgeAR | 85.08 | 124.18 | 9.80 |
| LSTM | 81.13 | 118.42 | 9.09 |
| CausalCNN | 79.96 | 118.16 | 8.36 |
| Persistence | 92.80 | 134.88 | 9.19 |
| Daily seasonal | 338.59 | 619.04 | 25.94 |

### Area 5059

| Model (seed 42) | MAE | RMSE | MAPE (%) |
| --- | --- | --- | --- |
| RidgeAR | 71.79 | 99.66 | 7.91 |
| LSTM | 75.46 | 106.22 | 7.78 |
| CausalCNN | 73.96 | 103.49 | 7.61 |
| Persistence | 81.52 | 114.38 | 7.96 |
| Daily seasonal | 171.74 | 245.87 | 18.02 |

### Area 5259

| Model (seed 42) | MAE | RMSE | MAPE (%) |
| --- | --- | --- | --- |
| RidgeAR | 65.86 | 93.17 | 7.74 |
| LSTM | 69.64 | 98.76 | 8.07 |
| CausalCNN | 69.62 | 99.65 | 8.47 |
| Persistence | 75.97 | 109.58 | 8.11 |
| Daily seasonal | 470.32 | 861.62 | 71.62 |

MAE is the mean absolute error; RMSE is the square root of mean squared error; MAPE is 100 times the mean of |prediction - actual| / |actual| over nonzero actual values. MAPE excludes zero actual targets and records the excluded count in the CSV. It is undefined if every target is zero; no arbitrary epsilon is inserted. MAE and RMSE use every eligible target. Each test week nominally has 1,008 points; the per-area scored counts are 5161: 1008, 5059: 1008, 5259: 1008, 4159: 1008, 4556: 1008.

| Model | Training (s) | Full test batch (ms) | Single forecast (ms) |
| --- | --- | --- | --- |
| CausalCNN | 47.26 | 76.13 | 0.965 |
| LSTM | 27.46 | 80.28 | 4.035 |
| RidgeAR | 0.10 | 0.20 | 0.063 |

Timing is averaged over five areas and three final fits per model on Apple M2 Pro, 16 GB RAM, CPU execution, four TensorFlow intra-operation threads. Neural training includes construction/compilation and validation; Ridge timing covers fitting. Warm inference is measured synchronously: median of five full-batch calls and median of 30 single-sample calls per fit, then averaged. Inference timing covers the model forward pass; window construction, scaling and correction reconstruction are excluded. These are local measurements with background system activity, not production throughput guarantees.

---PAGE---

## Forecasts for area 5161

These are the three required model-specific comparisons for this area. Figures use the same observed targets and reference seed. Predictions use actual observations available up to the preceding interval.

![Figure 5a. RidgeAR, area 5161.](../results/figures/forecast_5161_RidgeAR.png)

![Figure 5b. LSTM, area 5161.](../results/figures/forecast_5161_LSTM.png)

![Figure 5c. CausalCNN, area 5161.](../results/figures/forecast_5161_CausalCNN.png)

---PAGE---

## Forecasts for area 5059

These are the three required model-specific comparisons for this area. Figures use the same observed targets and reference seed. Predictions use actual observations available up to the preceding interval.

![Figure 6a. RidgeAR, area 5059.](../results/figures/forecast_5059_RidgeAR.png)

![Figure 6b. LSTM, area 5059.](../results/figures/forecast_5059_LSTM.png)

![Figure 6c. CausalCNN, area 5059.](../results/figures/forecast_5059_CausalCNN.png)

---PAGE---

## Forecasts for area 5259

These are the three required model-specific comparisons for this area. Figures use the same observed targets and reference seed. Predictions use actual observations available up to the preceding interval.

![Figure 7a. RidgeAR, area 5259.](../results/figures/forecast_5259_RidgeAR.png)

![Figure 7b. LSTM, area 5259.](../results/figures/forecast_5259_LSTM.png)

![Figure 7c. CausalCNN, area 5259.](../results/figures/forecast_5259_CausalCNN.png)

---PAGE---

## 8. Discussion and failure analysis

Area 5161: CausalCNN is the best learned reference model; its RMSE is 12.4% below persistence. Area 5059: RidgeAR is the best learned reference model; its RMSE is 12.9% below persistence. Area 5259: RidgeAR is the best learned reference model; its RMSE is 15.0% below persistence. Area 4159: RidgeAR is the best learned reference model; its RMSE is 7.7% below persistence. Area 4556: RidgeAR is the best learned reference model; its RMSE is 10.2% below persistence.

For area 5161, repeated-seed results are: RidgeAR: mean RMSE 124.18 (SD 0.00). LSTM: mean RMSE 120.12 (SD 1.93). CausalCNN: mean RMSE 118.49 (SD 2.91). The seed variation qualifies small differences between neural models. Average neural training takes 267 times as long for LSTM and 460 times as long for CausalCNN as RidgeAR on this machine. Linear lag weights therefore offer a low-cost baseline; LSTM gates and nonlinear convolution add flexibility, but require a measurable accuracy benefit. Strong adjacent-interval dependence can leave little improvement for a more expressive model at a one-step horizon. A lower error on one area is not evidence of universal superiority. Daily-seasonal persistence performs much worse than immediate persistence in all three primary areas, showing that strong periodicity alone does not make yesterday's level an adequate next-interval forecast. Area 5059 has lower training variability (CV 0.71 versus 0.92) and stronger daily dependence (0.912 versus 0.894) than area 5161. Its RidgeAR RMSE is 99.66, versus 106.22 for LSTM and 103.49 for CausalCNN. A regularized linear response is competitive on this more regular profile. These observations support evaluating model suitability by traffic regime, consistent with the caution motivated by [2]; five selected areas cannot establish a causal relationship between variability and architecture rankings.

### 8.1 Failure case

![Figure 8. A local view of the largest seed-42 model error after normalizing by each area's training standard deviation. This case is selected for diagnosis after freezing the models.](../results/figures/failure_case.png)

The selected failure is area 4556 at 2013-12-17 00:40 CET. CausalCNN predicts 405.11 for an observed value of 717.77, an absolute error of 312.66 (1.24 training standard deviations). Observed activity changes from 414.87 in the preceding interval to 717.77 at the miss, then to 348.86 in the next interval. Abrupt changes challenge all three models because they must infer the next interval from past activity alone; the surrounding curves show their response lag. Event context and neighboring-area measurements are unavailable to these univariate models, so the specific real-world cause cannot be established. The repository also reports error by actual-traffic decile to inspect systematic peak underprediction.

---PAGE---

## 9. Conclusion and future work

On the busiest area, CausalCNN gives the lowest seed-42 learned-model RMSE (118.16), compared with persistence at 134.88. The experiment also shows how rankings vary across areas and seeds. Complexity is justified only when it improves the relevant errors enough to offset training and execution costs; persistence remains a necessary benchmark. The model rankings and exact improvement percentages are reported above rather than assumed from previous literature.

Limitations include one held-out week, a small set of areas selected using full-period activity, shared hyperparameters tuned on one area, conservative missing-window exclusions, and limited random-seed repetitions. Follow-up work should use rolling-origin evaluation over more weeks, separate prospective area selection from evaluation, tune within computationally matched budgets, assess missingness explicitly, and add spatial/event covariates for abrupt changes. These extensions require new experiments; their benefits are not claimed here.

## Appendix A. Sensitivity across seeds

| Area | Model | RMSE mean (3 seeds) | RMSE sample SD |
| --- | --- | --- | --- |
| 4159 | CausalCNN | 20.15 | 0.78 |
| 4159 | LSTM | 20.99 | 1.01 |
| 4159 | RidgeAR | 19.88 | 0.00 |
| 4556 | CausalCNN | 37.65 | 1.15 |
| 4556 | LSTM | 41.15 | 2.20 |
| 4556 | RidgeAR | 35.57 | 0.00 |
| 5059 | CausalCNN | 103.34 | 3.10 |
| 5059 | LSTM | 101.23 | 4.53 |
| 5059 | RidgeAR | 99.66 | 0.00 |
| 5161 | CausalCNN | 118.49 | 2.91 |
| 5161 | LSTM | 120.12 | 1.93 |
| 5161 | RidgeAR | 124.18 | 0.00 |
| 5259 | CausalCNN | 99.20 | 0.58 |
| 5259 | LSTM | 98.35 | 0.37 |
| 5259 | RidgeAR | 93.17 | 0.00 |

Standard deviations summarize optimization variation over seeds 42, 43 and 44. They are not confidence intervals and do not account for the choice of test week or area. Supplementary model-specific plots for squares 4159 and 4556 are included under `results/figures/`, alongside all nine primary-area plots. All per-timestamp predictions are retained for independent checking.

## References

[1] G. Barlacchi et al., “A multi-source dataset of urban life in the city of Milan and the Province of Trentino,” Scientific Data, vol. 2, art. 150055, 2015, doi: 10.1038/sdata.2015.55. https://doi.org/10.1038/sdata.2015.55.

[2] A. Azari, P. Papapetrou, S. Denic, and G. Peters, “Cellular Traffic Prediction and Classification: a comparative evaluation of LSTM and ARIMA,” arXiv:1906.00939, 2019. https://arxiv.org/abs/1906.00939.

[3] S. Bai, J. Z. Kolter, and V. Koltun, “An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling,” arXiv:1803.01271, 2018. https://arxiv.org/abs/1803.01271.

[4] C. Zhang and P. Patras, “Long-Term Mobile Traffic Forecasting Using Deep Spatio-Temporal Neural Networks,” arXiv:1712.08083, 2017. https://arxiv.org/abs/1712.08083.

[5] Telecom Italia, “Telecommunications - SMS, Call, Internet - MI,” Harvard Dataverse, 2015. https://doi.org/10.7910/DVN/EGZHFV. Data under ODbL 1.0; [from BigDataChallenge contest](http://www.telecomitalia.com/tit/en/bigdatachallenge.html).

[6] Source code and reproducibility materials: [GitHub repository](https://github.com/irachrist1/ml-techniques-1-formative-1).

## AI assistance disclosure

OpenAI Codex provided substantial assistance with code, experimental design, model implementation, testing, analysis, figures and report drafting. All reported numerical results were computed from the published dataset; synthetic data were used only for unit-test fixtures.
