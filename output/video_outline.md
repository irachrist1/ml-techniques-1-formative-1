# Individual video plan: 7-10 minutes

Timings for presenting the study and demonstrating the code.

## 0:00-0:45 - Question and why it matters

One-step (10-minute) Internet-activity forecasting across Milan areas. Say clearly that the
target is an activity count, not GB. Say why a short-horizon forecast is useful for capacity.

## 0:45-2:00 - Real data and memory

Show `prepare_day.py`, the dataset DOI in `DATA_LICENSE.md` and `results/memory_benchmark.json`.
Cover: summing country-code rows, why an empty Internet field is not an observed zero, and why
chunked loading keeps peak memory bounded. Stress that the benchmark compares two processes on
the same complete day and asserts the outputs match before reporting the reduction.

## 2:00-3:15 - What the data shows

Show `traffic_distribution.png`, `first_two_weeks.png` and `temporal_analysis.png`. The full-period
top three are [5161, 5059, 5259]. Compare their scale and shape against 4159 and 4556. Mention that
1 November is an Italian public holiday, so the first low stretch is a three-day weekend. Describe
one pattern you can actually see, and keep it separate from any guess at its cause.

## 3:15-4:45 - Three model mechanisms

Open `experiments.py`. Walk through training-only scaling, the past-only window, the common
144-step eligibility rule and the learned correction to persistence. Explain RidgeAR's lag
coefficients, the LSTM gates and the CNN's causal receptive field
(255 steps against a
144-step input). Show `configs/final.json` and justify one
change from `results/tuning_log.csv`. Explain why 16-22 December never picked a hyperparameter, and
why rolling one-step prediction is allowed to use the already observed part of the test week.

## 4:45-6:15 - Results and live demonstration

Show `results/reference_metrics.csv` and a forecast plot. On area 5161 the best learned model at
seed 42 is CausalCNN, RMSE 118.16,
against persistence at 134.88. Explain MAE versus RMSE and the zero-target
rule for MAPE. Then run:

```sh
python -m unittest -v
python predict_saved.py --run results/runs/final_selected_area5161_seed42 --model RidgeAR --timestamp '2013-12-16T12:00:00+01:00'
```

This rebuilds one forecast from saved parameters and past observations only. Nothing retrains on
camera. If the script reports an incomplete history, pick another fully observed timestamp.

## 6:15-7:45 - The two results I am least comfortable with

Show `results/validation_vs_test_ranking.csv`: validation and test disagree about the winner in
5 of 5 areas. Then `results/peak_bias_summary.csv`: all
15 area-model combinations under-forecast the busiest
decile and over-forecast the quietest. Tie both to the persistence-correction target.

Show `failure_case.png`: area 4556, error 313
(1.24 training SD), and point out that the step *after* the spike is
also wrong because the spike is now in the input.

Mention the epoch-cap audit: 0 of 30 neural fits now stop
early rather than hitting the budget, after round four raised the cap.

## 7:45-9:00 - Conclusion and next experiment

The defensible claim, the limitation of one test week and five areas, and the next experiment
(rolling-origin evaluation across several weeks). Close on the repository and how to reproduce.

## Check you can answer these before recording

- Why does country-code aggregation matter, and what can it still hide?
- How is a missing activity value different from an observed zero?
- Does selecting areas on full-period totals affect what you can claim?
- Where exactly are the normalization statistics fitted?
- Why is shuffling training windows not leakage here?
- How does predicting the next interval differ from predicting a whole week?
- Why is the CNN receptive field larger than the input, and does that matter?
- Why can persistence or RidgeAR beat a neural model at this horizon?
- What do three seeds measure, and what do they not measure?
- Which result justified round four of tuning?
- Why do validation and test disagree, and what would settle it?
