# One-step mobile-network traffic forecasting in Milan

**Gentil Christian Tonny Iradukunda | ML Techniques I | Formative Assignment 1**

**Video:** [https://youtu.be/6sHUk8ux-5g](https://youtu.be/6sHUk8ux-5g)  ·  **Code:** https://github.com/irachrist1/ml-techniques-1-formative-1

## 1. Introduction

The question here is how three fairly different sequential models compare when you ask them for the next ten minutes of Internet activity, and whether the answer changes depending on which part of Milan you look at.

One thing to get out of the way first. The quantity I forecast is the activity count the dataset ships with, not bandwidth and not bytes. Every error figure below is in activity units, so none of them should be read as megabytes or as a capacity number.

I process the whole November-December 2013 collection to rank all 10,000 areas, then compare RidgeAR, LSTM and CausalCNN against two baselines, immediate persistence and persistence at a one-day lag. The evaluation week is December 16-22.

## 2. Related work and model motivation

Barlacchi et al. [1] set out how these records are aggregated and scaled, and how much they vary by day and by week. That is most of the reason I look at each area's temporal structure on its own instead of assuming the areas behave alike.

A note on labels. I claim no land use for squares 4159 or 4556. The Bocconi and Navigli examples usually quoted from that paper are squares 4259 and 4456, which are different squares, so neither label transfers to the ones this assignment specifies.

Azari et al. [2] put an LSTM against ARIMA on cellular packet traffic and find that whether the extra complexity pays depends on the features, the amount of training data and the traffic conditions. That, plus the very strong lag-one dependence in Section 4, is why I wanted a cheap linear model in the comparison rather than three neural networks. RidgeAR is that model. The L2 penalty keeps the correlated lag coefficients from blowing up, and the cost is that it cannot learn a nonlinear interaction at all. It is also not ARIMA. It predicts a correction to persistence and has no moving-average error terms.

The LSTM is there to test one thing, whether gated nonlinear processing of a day's observations actually beats a set of linear lag weights. Its cell and hidden state evolve inside each window and reset between windows, so no state carries across. That flexibility is not free in training or inference time, and with only a day of input the model never sees a full weekly cycle.

The third model comes from Bai et al. [3], who argue for causal dilated convolutions as an alternative to recurrence. My CausalCNN builds nonlinear local features across the input window, but it is a stripped-down version of what they describe, with no residual blocks and no weight normalization. I spell out its receptive field and padding in Section 5, because that detail decides how much history the model can actually reach. I am not assuming their benchmark rankings carry over to this data. Zhang and Patras [4] were useful for a different reason. They use spatial information and much longer horizons, so their accuracy numbers are not something I can compare against, but they did show me where I would take this next.

For what it is worth up front, on the busiest area at my reference seed the CausalCNN does come out ahead, with RMSE 118.16 against persistence at 134.88. Sections 7 and 8 take most of that back.

---PAGE---

## 3. Dataset, preparation and memory management

61 daily files, 314,966,126 rows, 20.48 GB on disk. I drop January 1. Both the download and the daily preparation step verify the publisher's MD5 values before anything else runs. Timestamps sit on Europe/Rome boundaries and a ten-minute grid. Square IDs are validated before I narrow them to int32, and activity stays float64.

Within each square and interval I sum the Internet values over the country codes. Empty fields are dropped, observed zeros are kept because a zero is a measurement rather than an absence, and a bin counts as missing only if it has no observed Internet value at all. That gives 145,354 missing bins out of 87,840,000.

There is a hole in that definition I cannot close with this data. A bin might have rows for some country codes and be missing others, which from the outside is indistinguishable from full coverage. So 145,354 is a floor on the coverage problem, not a measurement of it.

The strategy is column projection plus chunked reading into fixed daily sum and count arrays. Those arrays come to 17.28 MB and their size does not grow with how much of the file I have read, so the full collection never has to be resident. They are not the whole memory story, though. Parser objects, validation and transient allocations all land in the process footprint too, which is why I measured peak RSS rather than adding up array sizes. The raw files stay on disk. I am buying bounded memory with storage and I/O.

**Table 1. Repeated November 1 benchmark; decimal MB, medians over 3 isolated processes per method.**

| Method | Peak RSS | RSS range | Wall seconds | Largest frame |
| --- | --- | --- | --- | --- |
| baseline | 1087.6 | 773.5-1125.4 | 2.32 | 309.9 |
| chunked | 384.7 | 347.1-391.1 | 3.78 | 2.0 |

Median peak reduction is 64.6%. I checked the two paths agree before reading anything into that. Every pair matches on missing masks and observation counts, with a maximum aggregate difference of 1.8e-12, which is floating-point noise. I alternate which method runs first, though I cannot control cache state or whatever else the laptop is doing.

The chunked path is the slower of the two here, and most of that gap is not the chunking. It is the per-chunk square-ID validation. Reading the same file with the IDs narrowed straight to int32 takes about 1.8 s, against 3.7 s with validation. Before I added that check the chunked path was marginally faster than loading the whole day. I kept it anyway, because narrowing first lets an out-of-range ID quietly wrap into a valid one, and a silently wrong square ID is a worse failure than a slow script.

Two scoping notes. Wall time excludes imports, checksums and writing the output, while peak memory includes imports and everything up to the peak. One more thing belongs on the record. An earlier single run of this benchmark gave 86.4% reduction. It did not reproduce once I ran it with repeats, so it stays in `memory_benchmark.json` for provenance but I do not rely on it.

![Figure 1. Total observed activity across 10,000 areas over November-December.](../results/figures/traffic_distribution.png)

Heavily right-skewed. Median 274,706, 99th percentile 4,666,202, maximum 12,682,673, so the busiest area carries roughly forty-six times the middle of the distribution. The top three are **5161, 5059, 5259**, holding 0.62% of all observed activity between them.

It is worth checking whether an area with patchy coverage could really belong in that top three. If I fill each area's missing bins with its own observed mean, the largest adjusted partial-area total comes to 891,902, well short of the 10,436,792 that third place requires. That is reassuring, but it is a what-if resting on an assumption about the missing bins rather than a bound on what could be hiding in them.

One more thing I would rather flag than bury. I selected these areas on totals across the whole two months, and that includes the week I evaluate on. The brief asks for exactly that, so I have done what was asked, but my choice of area is not independent of my test week.

---PAGE---

## 4. Exploratory and temporal analysis

![Figure 2. November 1-14 for the top three areas and squares 4159 and 4556; panels retain their own scales.](../results/figures/first_two_weeks.png)

![Figure 3. Pairwise autocorrelation and weekday/weekend profiles for area 5161, computed on training data only. The dashed lines are the white-noise reference statsmodels draws by default (reference 9); they are not confidence intervals for a seasonal series like this one.](../results/figures/temporal_analysis.png)

The five series do not look alike. 5259 and 4159 both sit on weekday plateaus and fall away at the weekend. 5161 does the opposite, peaking hardest on weekends. 5059 has broad flat daytime stretches. 4556 is the odd one out, with isolated spikes and the weakest lag-one correlation of the five, and yet the lowest coefficient of variation.

Before reading anything into the left edge of Figure 2, check the calendar. 1 November 2013 was All Saints' Day, a public holiday in Italy, and it fell on a Friday. That makes the low opening stretch a three-day holiday weekend rather than a normal one, so 9-10 November is the fairer comparison.

So these areas run on different schedules. That does not tell me what they are used for, and I am not going to guess from a time series.

For the two additional analyses I chose temporal dependence and weekday/weekend profiles, both on area 5161.

Lag-one correlation is 0.982, which is probably the most consequential number in this report. It is why persistence is such a hard baseline and why every model here is fed recent history. Daily correlation is 0.894 and weekly is 0.952, so there is real structure further back, and that is what pushed me to test longer lookbacks and to put calendar features on the future work list. Six hours out the correlation is essentially nothing (-0.017), and the trough sits at twelve hours (-0.753), which is just the daily cycle showing up as anti-correlation half a day apart.

These are descriptive numbers. They tell me which inputs are worth giving a model, not which architecture will win.

---PAGE---

### 4.1 Statistical characterization

**Table 2. Training-period characteristics; all five selected full-period series have zero missing bins.**

| Area | Mean | CV | Lag 1 | Lag 144 | Weekend/weekday |
| --- | --- | --- | --- | --- | --- |
| 5161 | 1511.4 | 0.92 | 0.982 | 0.894 | 1.36 |
| 5059 | 1342.6 | 0.71 | 0.972 | 0.912 | 0.81 |
| 5259 | 1314.1 | 0.85 | 0.989 | 0.669 | 0.42 |
| 4159 | 317.2 | 0.59 | 0.969 | 0.743 | 0.54 |
| 4556 | 582.4 | 0.43 | 0.943 | 0.769 | 1.14 |

CV here is the sample standard deviation over the mean. The last column is the interesting one. Area 5161 averages 1847 on weekends against 1357 on weekdays, while 5259 runs at a ratio of 0.42, which is close to the inverse. Two areas in the same city pulling in opposite directions, which is the main reason I evaluate each area separately rather than pooling them.

Two things are worth keeping apart here. A strongly repeating daily shape does not imply that yesterday's value at the same time is a good ten-minute-ahead forecast. Tables 6 to 8 show it is a genuinely bad one.

On the 5,472 complete training values the ADF test [8] gives a statistic of -4.103 with p=0.000957. The regression includes a constant, and AIC picks 152 lags out of a maximum of 168.

That maximum is worth explaining, because I changed it partway through. My first run searched up to 30 lags and returned p=1.35e-27. Widening the search so the daily cycle could enter moved p to 9.57e-04. That is the same data and the same test, with roughly twenty orders of magnitude between the two p-values. I do not think it makes the first run invalid, and I am not claiming ADF must always span a seasonal cycle. What it does show is how much the answer depends on the specification, which is worth more than either p-value on its own.

Both specifications reject the unit-root null under their own assumptions. Neither proves stationarity in any strict sense, and neither makes the seasonal structure go away. I have not run residual diagnostics on the ADF regression, which is a gap. The first-differenced series gives p=3.3e-15, and that on its own is not a reason to difference the forecasting target.

![Figure 4. Robust STL (see reference 10) on training dates, period 144. The daily seasonal component, changing trend and residual spikes describe different sources of variation.](../results/figures/stl_training.png)

Daily seasonal strength comes out at 0.870, using max(0, 1 - Var(residual)/Var(residual + seasonal)). That is a ratio within the seasonal-plus-residual part of the decomposition only. It is not the share of total observed variance explained and should not be read that way. No training values needed interpolating, and nothing from the STL feeds into any model.

I recomputed both the ADF and STL output to check the two were consistent. They are. A series can have a strong daily cycle and still reject a unit root, because seasonality and a stochastic trend are different things.

---PAGE---

## 5. Methodology

Every target x(t+1) is predicted from x(t-L+1) through x(t) and nothing else. No future value enters anywhere.

A window is eligible when the full 144-step history is observed and the target is finite. I apply that same rule to every candidate, including the earlier L=36 experiments where only 36 of those steps are consumed, so no two models are ever scored on different subsets of the week. Windows containing a missing value are dropped, and if the timestamp grid is incomplete or out of order the loader raises rather than carrying on quietly. Each area ends up with 5,328 training targets and 1,008 each for validation and test.

**Table 3. Target boundaries in Europe/Rome; end dates are inclusive.**

| Split | Dates | Use |
| --- | --- | --- |
| Training | Nov 1-Dec 8 | Weights and scaling |
| Validation | Dec 9-15 | Settings and stopping |
| Evaluation | Dec 16-22 | Reported rolling one-step errors |

The input representation is the same for all three models. Each one receives the history standardized with that area's own training mean mu and population standard deviation sigma, shaped (L,1). RidgeAR flattens it into L lag features. All three learn the same target, delta = (x(t+1)-x(t))/sigma, and I reconstruct as max(0, x(t) + sigma*predicted_delta), clipped at zero because activity cannot be negative.

There are three places leakage could enter, so I will take them in turn. Scaling statistics come only from training observations and no target is imputed, so that one is clean. Shuffling the training windows looks alarming but is not a problem, because the windows are fully formed before the shuffle happens. The third is worth stating plainly. When I forecast a target late in the test week, the history feeding it does contain earlier test values, which have by then been observed. That is what rolling one-step-ahead forecasting means, and it is not a seven-day recursive forecast.

**Table 4. Final model structures.**

| Model | Lookback | Structure | Parameters |
| --- | --- | --- | --- |
| RidgeAR | 144 | L2 alpha=10; fitted intercept | 145 |
| LSTM | 144 | 16 gated units; final hidden state | 1169 |
| CausalCNN | 144 | 16 filters; kernel 3; dilations 1,2,4,8,16,32,64 | 4785 |

Ridge minimizes squared correction error plus an L2 penalty on the coefficients, so its regularization pulls it toward predicting no correction, and therefore toward persistence. The LSTM is sigmoid gates and tanh cell updates with a linear output head, no dropout, no state carried between windows. The CausalCNN is causal ReLU convolutions with a scalar head read off the final position.

The CNN's receptive field is worth a note of its own. With those dilations the theoretical field is 255 steps, but only 144 observations exist, so much of that reach is padding. That does not make the deepest dilation free to remove. Dropping dilation 64 cuts the field to 127, which is shorter than the input, so the model loses the oldest 17 real observations. Removing it is an ablation rather than a tidy-up.

The neural output heads initialize at zero, so an untrained model starts out predicting exactly persistence. Adam at learning rate 0.001, gradient-norm clipping at 1, batch size 128. Early stopping watches standardized validation MSE with patience 4 and min_delta=1e-5, under an 80-epoch cap. The saved metadata records the restored epoch separately from the raw minimum-loss epoch, which matters in Section 6. Weights and scaling are fitted per area; nothing is shared.

It is fair to ask whether predicting a correction pushes the models toward predicting no change. It should not. Up to a scale factor, minimizing squared error in delta is the same objective as minimizing squared forecast error after reconstruction, before the clip. Whether they nonetheless under-react is a separate and empirical question, and Section 8 suggests they do. That could come from the inputs, the capacity, the regularization or the optimization, and I have not run the comparison that would separate those.

---PAGE---

## 6. Iterative experimentation and hyperparameter tuning

**Table 5. Area 5161, seed 42 validation experiments. These are measured runs, not equal compute budgets.**

| Experiment | Model | Val RMSE | Epochs | Train seconds |
| --- | --- | --- | --- | --- |
| initial | RidgeAR | 159.07 | 0 | 0.021 |
| initial | LSTM | 163.30 | 13 | 4.105 |
| initial | CausalCNN | 161.02 | 20 | 4.514 |
| daily_history | RidgeAR | 153.13 | 0 | 0.051 |
| daily_history | LSTM | 160.26 | 19 | 16.879 |
| daily_history | CausalCNN | 160.28 | 16 | 10.198 |
| capacity | RidgeAR | 152.39 | 0 | 0.041 |
| capacity | LSTM | 162.04 | 9 | 13.023 |
| capacity | CausalCNN | 160.09 | 11 | 9.126 |
| budget | RidgeAR | 152.39 | 0 | 0.040 |
| budget | LSTM | 160.26 | 19 | 17.367 |
| budget | CausalCNN | 160.09 | 11 | 9.251 |

Round 2 is the one that mattered. Extending the history from six hours to a full day improved all three models. I have to qualify the CNN's share of that, because I changed its dilation depth in the same round and cannot separate the two causes.

Round 3 kept Ridge at alpha 10 and the LSTM at width 16. Width 16 for the CNN moved validation RMSE from 160.28 to 160.09, a difference of 0.12%. That settled the configuration only because I had committed in advance to taking the lowest validation RMSE, and it is nowhere near large enough to claim the wider CNN is genuinely better. Every candidate here ran on a single tuning seed, which is a real weakness in the tuning design.

Round 4 needs a chronology, because it happened after I already had test results.

The original study capped training at 20 epochs. Going back through the stopping histories, 22 of 30 neural fits had run the full 20. One also triggered patience on the same epoch, so 21 stopped purely on the cap, and ten had their lowest recorded validation loss at the very last epoch. Those ten are the concerning ones, because a model still improving when you switch it off has not had a fair run. The cap did not bind for area 5161 at seed 42, the configuration I quote most often, but it did bind for that area's CNN at seeds 43 and 44.

So I raised the cap to 80 across the board and reran. That is a sensitivity fix on the training budget, not a claim that finite budgets are wrong in principle. A cap that nothing reaches is fine.

After the rerun all 30 fits stop through patience and the longest runs 55 epochs, comfortably inside the new cap. No test target has ever entered a training or early-stopping loss. But I should not pretend the evaluation week is untouched. I looked at test results, changed a training setting, and looked again. That is a reused holdout and deserves to be read as one. The original numbers are preserved in commit a5e5111 and `revision_provenance.json` records the order things happened in.

I keep seeds 42, 43 and 44, with 42 as the reference. One caveat on that. I cannot show from the commits exactly when I settled on 42, so treat it as a convention I am declaring rather than one I can prove was fixed before I saw results.

![Figure 5. Training and validation correction MSE, area 5161, seed 42. The two curves cover different calendar periods, so there is no reason for them to sit at the same level.](../results/figures/learning_curves.png)

Validation loss sitting below training loss in Figure 5 looks wrong until you remember that these are two different weeks of traffic, and that the training loss is averaged over updates while the weights are still moving. It is not evidence of leakage. It is also not proof that anything converged.

---PAGE---

## 7. Results and timing

**Table 6. Area 5161; seed 42, 1,008 eligible targets.**

| Model (seed 42) | MAE | RMSE | MAPE (%) |
| --- | --- | --- | --- |
| RidgeAR | 85.08 | 124.18 | 9.80 |
| LSTM | 81.13 | 118.42 | 9.09 |
| CausalCNN | 79.96 | 118.16 | 8.36 |
| Persistence | 92.80 | 134.88 | 9.19 |
| Daily seasonal | 338.59 | 619.04 | 25.94 |

**Table 7. Area 5059; seed 42, 1,008 eligible targets.**

| Model (seed 42) | MAE | RMSE | MAPE (%) |
| --- | --- | --- | --- |
| RidgeAR | 71.79 | 99.66 | 7.91 |
| LSTM | 75.46 | 106.22 | 7.78 |
| CausalCNN | 73.96 | 103.49 | 7.61 |
| Persistence | 81.52 | 114.38 | 7.96 |
| Daily seasonal | 171.74 | 245.87 | 18.02 |

**Table 8. Area 5259; seed 42, 1,008 eligible targets.**

| Model (seed 42) | MAE | RMSE | MAPE (%) |
| --- | --- | --- | --- |
| RidgeAR | 65.86 | 93.17 | 7.74 |
| LSTM | 66.84 | 95.02 | 7.82 |
| CausalCNN | 66.64 | 93.61 | 8.21 |
| Persistence | 75.97 | 109.58 | 8.11 |
| Daily seasonal | 470.32 | 861.62 | 71.62 |

MAE and RMSE use every eligible target. MAPE is 100 times the mean absolute relative error over nonzero targets, and I record how many get excluded. If a target set were all zeros, MAPE is undefined and my code returns that rather than slipping an epsilon into the denominator to manufacture a number. In practice none of this bites. All five series have the full 1,008 scored targets and not one zero. Errors are in activity units, after inverting the standardization.

**Table 9. Mean timing across five areas and three fits per model.**

| Model | Training (s) | Test batch (ms) | Single (ms) |
| --- | --- | --- | --- |
| CausalCNN | 17.090 | 30.70 | 0.298 |
| LSTM | 23.126 | 34.59 | 2.067 |
| RidgeAR | 0.046 | 0.10 | 0.047 |

Measured on an Apple M2 Pro, 16 GB RAM, CPU only, four TensorFlow intra-op threads. The training column is not measuring the same thing for all three models, and the ratio should not be read as though it were. Neural training includes graph construction, compilation and per-epoch validation, whereas Ridge is the fit call and nothing else. Inference is warm and synchronous, taken as the median of five full-batch and 30 single-sample calls per fit and then averaged across fits, and it excludes window construction, scaling and reconstruction. This is a laptop with other things running on it, so the numbers are the right order of magnitude rather than latency anyone could rely on.

---PAGE---

## Forecasts for area 5161

Observed and predicted activity are on identical timestamps, at reference seed 42, using history that had already been observed at the moment of prediction.

![Figure 6a. RidgeAR, area 5161: December 16-22 rolling one-step forecasts.](../results/figures/forecast_5161_RidgeAR.png)

![Figure 6b. LSTM, area 5161: December 16-22 rolling one-step forecasts.](../results/figures/forecast_5161_LSTM.png)

![Figure 6c. CausalCNN, area 5161: December 16-22 rolling one-step forecasts.](../results/figures/forecast_5161_CausalCNN.png)

---PAGE---

## Forecasts for area 5059

Same construction as the three plots above, at seed 42, on matched timestamps and already-observed history.

![Figure 7a. RidgeAR, area 5059: December 16-22 rolling one-step forecasts.](../results/figures/forecast_5059_RidgeAR.png)

![Figure 7b. LSTM, area 5059: December 16-22 rolling one-step forecasts.](../results/figures/forecast_5059_LSTM.png)

![Figure 7c. CausalCNN, area 5059: December 16-22 rolling one-step forecasts.](../results/figures/forecast_5059_CausalCNN.png)

---PAGE---

## Forecasts for area 5259

Again at seed 42, on matched timestamps and already-observed history.

![Figure 8a. RidgeAR, area 5259: December 16-22 rolling one-step forecasts.](../results/figures/forecast_5259_RidgeAR.png)

![Figure 8b. LSTM, area 5259: December 16-22 rolling one-step forecasts.](../results/figures/forecast_5259_LSTM.png)

![Figure 8c. CausalCNN, area 5259: December 16-22 rolling one-step forecasts.](../results/figures/forecast_5259_CausalCNN.png)

---PAGE---

## 8. Comparative discussion

Fourteen of the fifteen learned fits at the reference seed beat persistence on RMSE. The exception is the LSTM on area 4556, at 39.76 against 39.62. That is a 0.35% loss, which is a rounding error on one week rather than a difference I have any right to call real.

Daily-seasonal persistence loses in all five areas, badly. I take that as confirmation that at a ten-minute horizon the thing to beat is the previous value, full stop. I would not take it as evidence that seasonal information is useless. Nobody here was handed the daily shape *and* recent history at once, and a model with both might well use it. I have not tested that, and it is on the future work list.

**Table 10. Validation/test winners at seed 42 only.**

| Area | Validation best | Test best |
| --- | --- | --- |
| 5161 | RidgeAR | CausalCNN |
| 5059 | CausalCNN | RidgeAR |
| 5259 | CausalCNN | RidgeAR |
| 4159 | LSTM | RidgeAR |
| 4556 | CausalCNN | RidgeAR |

Table 10 is the uncomfortable one. At seed 42 the validation winner and the test winner disagree in every one of the five areas. At seeds 43 and 44 it drops to two out of five.

The spread across seeds explains most of that. On area 5161 the CNN's validation RMSE across the three seeds is 160.09, 152.02 and 148.03, while Ridge sits unmoved at 152.39 because it has no random initialization. The seed alone is enough to flip which model wins validation. So Table 10 is not purely a story about December 9-15 being a different week from December 16-22. Three seeds tell me something about optimization variance, and nothing at all about variance across weeks I never tested.

**Table 11. Evaluation RMSE mean (sample SD), three seeds.**

| Area | RidgeAR | LSTM | CausalCNN |
| --- | --- | --- | --- |
| 5161 | 124.18 (0.00) | 120.12 (1.93) | 118.49 (2.91) |
| 5059 | 99.66 (0.00) | 101.82 (4.43) | 103.68 (2.58) |
| 5259 | 93.17 (0.00) | 96.05 (1.67) | 91.18 (2.28) |
| 4159 | 19.88 (0.00) | 20.72 (1.26) | 20.15 (0.78) |
| 4556 | 35.57 (0.00) | 42.08 (2.01) | 37.65 (1.15) |

Area 5259 makes the point from the other direction. Ridge wins at seed 42, but the CNN has the lower mean across three seeds, so the choice of seed decides the answer.

On 5059 the linear model does well, and that area also has the lower training CV and stronger daily dependence, which is the kind of correspondence the brief asks me to look for. I will note it and leave it there. Five areas cannot establish that traffic characteristics cause an architecture ranking, and 5059 is the area where that story happens to be tidiest.

**Table 12. Area 5161 signed error by realized-target decile; 101 observations per extreme decile.**

| Model | Low-decile bias | High-decile bias | High-decile MAE |
| --- | --- | --- | --- |
| RidgeAR | +16.4 | -80.8 | 164.4 |
| LSTM | +13.3 | -50.3 | 151.8 |
| CausalCNN | +4.3 | -36.2 | 152.1 |

Across all areas, 14 of 15 high-decile biases are negative and 13 of 15 low-decile biases positive. The models undershoot the peaks and overshoot the troughs, all three of them.

I want to be careful about what that proves. The deciles are defined by what actually happened, so even a perfect forecaster would show this pattern at the extremes. If a target landed in the top decile partly by chance, the unpredictable part of it can only sit on one side of the forecast. So this is a good description of where the errors concentrate, but it is not proof that the correction target causes the under-reaction. The test would be a model that predicts the value directly, to see whether the bias survives.

---PAGE---

### 8.1 Failure case and limits of the explanation

![Figure 9. The biggest single error at seed 42, scaled by each area's training standard deviation so areas of different size compare fairly. I picked this case after the models were finished.](../results/figures/failure_case.png)

The worst single miss at seed 42 is area 4556 at 2013-12-17 00:40 CET. The CausalCNN predicts 405.11 where the actual is 717.77, an absolute error of 312.66, or 1.24 training standard deviations for that area.

It is a one-interval spike. Activity goes 414.87, then 717.77, then straight back to 348.86. All three models miss the rise and then all three overshoot the fall, predicting 554.76 (RidgeAR), 635.22 (LSTM) and 512.68 (CausalCNN) against an actual of 348.86.

The mechanism I would propose is straightforward. The spike enters both the next input window and the persistence anchor the correction is applied to, so all three models drag it forward by one step. The lag in the plot is consistent with that.

Three caveats, which matter more than the explanation. A model could in principle learn to cancel that anchor effect, so this is not a hard limit of any of these architectures and I am not claiming more tuning would fail to help. This is also, by construction, the single worst error I could find, so it is the opposite of a typical case. And I have no idea what caused the spike. With one interval involved I cannot rule out a measurement artifact.

If I were carrying on, the next experiment would be direct-target against correction-target under matched validation and matched compute, then checking abrupt changes like this across more than a single week. Neighbouring areas presumably carry some signal about a spike like this, but I have not shown it. An asymmetric loss would be the obvious move if underprediction were the expensive error in practice, though it trades accuracy against calibration and I would want to measure both.

## 9. Conclusion and future work

On the busiest area, at my reference seed, the CNN takes RMSE from persistence's 134.88 down to 118.16. That is a real improvement and I am not going to talk it down. RidgeAR stays close behind on a fraction of the training time, at 0.046 seconds against 17 and 23.

But the ranking moves when I change the area, and when I change the seed, and I have only the one week. So I will not name a best architecture here, because the evidence does not carry it. What I will defend is narrower. At a ten-minute horizon a 145-parameter linear model is a serious competitor to both neural architectures, and the burden sits with the expensive models to show a gain that survives more than one week and more than one seed.

The limitations, in one place rather than scattered. One evaluation week, reused after I changed the epoch budget. Areas chosen on totals that include that same week. Hyperparameters tuned almost entirely on one area at one seed. Partial coverage in some aggregation bins that I have no way to detect. And training budgets that are not matched across models, which makes Table 9 a description of what I ran rather than a fair compute comparison.

The next steps follow off that list. Pick the areas from an earlier period so selection is independent of the test week. Evaluate over several weeks with a rolling origin. Give each model the same compute budget. Test direct-target against correction-target properly. And given the 0.894 daily and 0.952 weekly correlations, try calendar features. That is a hypothesis drawn from the autocorrelation rather than something I have evidence for yet.

---PAGE---

## AI assistance disclosure

I used AI assistance throughout this project specifically during debugging of the code, designing the experiments, writing the tests, running the analysis and alignment and drafting some parts of this report. Every number here comes from the real dataset and can be recomputed from the saved predictions in the repository. The only invented values anywhere in the project are the small fixtures in the unit tests.

I am responsible for what I am submitting. That includes being able to explain and defend the data handling, the memory decisions, the model choices, the experiments, and the conclusions I have drawn from them.

## References

[1] G. Barlacchi et al., "A multi-source dataset of urban life in the city of Milan and the Province of Trentino," Scientific Data, vol. 2, art. 150055, 2015, doi: 10.1038/sdata.2015.55. https://doi.org/10.1038/sdata.2015.55.

[2] A. Azari, P. Papapetrou, S. Denic, and G. Peters, "Cellular Traffic Prediction and Classification: a comparative evaluation of LSTM and ARIMA," arXiv:1906.00939, 2019. https://arxiv.org/abs/1906.00939.

[3] S. Bai, J. Z. Kolter, and V. Koltun, "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling," arXiv:1803.01271, 2018. https://arxiv.org/abs/1803.01271.

[4] C. Zhang and P. Patras, "Long-Term Mobile Traffic Forecasting Using Deep Spatio-Temporal Neural Networks," arXiv:1712.08083, 2017. https://arxiv.org/abs/1712.08083.

[5] Telecom Italia, "Telecommunications - SMS, Call, Internet - MI," Harvard Dataverse, 2015, doi: 10.7910/DVN/EGZHFV. https://doi.org/10.7910/DVN/EGZHFV. ODbL 1.0; attribution and terms are retained in DATA_LICENSE.md.

[6] C. Tonny, "ML Techniques I formative assignment: source code and reproducibility materials," GitHub. [Repository](https://github.com/irachrist1/ml-techniques-1-formative-1).

[7] Individual project video: [https://youtu.be/6sHUk8ux-5g](https://youtu.be/6sHUk8ux-5g)

[8] statsmodels developers, "adfuller: Augmented Dickey-Fuller unit root test," statsmodels documentation. https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html.

[9] statsmodels developers, "acf: Autocorrelation function," statsmodels documentation. https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.acf.html.

[10] R. B. Cleveland, W. S. Cleveland, J. E. McRae, and I. Terpenning, "STL: A Seasonal-Trend Decomposition Procedure Based on Loess," Journal of Official Statistics, vol. 6, no. 1, pp. 3-73, 1990. Method implemented by statsmodels STL: https://www.statsmodels.org/stable/generated/statsmodels.tsa.seasonal.STL.html.
