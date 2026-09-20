# Appendix experiments

Run after the report was submitted. They score on the **validation week only**
(December 9-15); the test week is not read. They live in their own configuration
files and run directories, so they cannot reach `results/tuning_log.csv` or any
number in the report. No configuration used in the study was changed, and no
reported metric is affected.

Everything below is one area (5161) at one seed (42), which is
the same narrow basis the study's own tuning used. Nothing here establishes a
test-week result.

Full table: `appendix_results.csv`. Rebuild with `python appendix_experiments.py`.

## A. Learning rate and batch size were never tuned

The study fixed the learning rate at 0.001 and the batch size at 128 for every
round and never varied either. This sweeps learning rate over
0.0003, 0.001, 0.003 and batch size over
64, 128, 256, with everything else exactly
`configs/final.json`.

| Model | Learning rate | Batch | Validation RMSE | Epochs |
| --- | --- | --- | --- | --- |
| CausalCNN | 0.003 | 128 | 150.08 | 16 |
| CausalCNN | 0.003 | 256 | 150.48 | 18 |
| CausalCNN | 0.001 | 256 | 151.24 | 31 |
| CausalCNN | 0.003 | 64 | 157.24 | 8 |
| CausalCNN | 0.0003 | 64 | 159.63 | 23 |
| CausalCNN | 0.0003 | 256 | 159.87 | 32 |
| CausalCNN | 0.001 | 128 | 160.09 | 11 |
| CausalCNN | 0.0003 | 128 | 160.13 | 18 |
| CausalCNN | 0.001 | 64 | 160.73 | 6 |
| LSTM | 0.003 | 64 | 151.28 | 13 |
| LSTM | 0.003 | 128 | 151.88 | 22 |
| LSTM | 0.001 | 128 | 160.26 | 19 |
| LSTM | 0.001 | 64 | 160.48 | 13 |
| LSTM | 0.003 | 256 | 162.42 | 11 |
| LSTM | 0.001 | 256 | 163.49 | 9 |
| LSTM | 0.0003 | 64 | 163.61 | 9 |
| LSTM | 0.0003 | 128 | 163.97 | 9 |
| LSTM | 0.0003 | 256 | 164.05 | 15 |

- **LSTM**: best at learning rate 0.003, batch 64, validation RMSE 151.28 against 160.26 at the study's settings, an improvement of 5.60%.
- **CausalCNN**: best at learning rate 0.003, batch 128, validation RMSE 150.08 against 160.09 at the study's settings, an improvement of 6.25%.

This matters for how the report's comparison should be read. At the study's
settings, the best neural validation RMSE on this area is 160.09 and
RidgeAR's is 152.39, so the linear model wins. With the learning rate
tuned, the best neural validation RMSE is 150.08, which is
below RidgeAR.

The honest reading is that the study's neural models were undertrained on a
hyperparameter it never examined, so its results do not separate "a linear model
is sufficient at this horizon" from "the neural models were not tuned". Settling
that needs a retrained study on an untouched test week, which this is not.

## B. Round three changed three things at once

`configs/capacity.json` changed the Ridge penalty and both neural widths in one
round, so the round could not attribute its outcome. Each change is re-run here
on its own against the round-two baseline.

| Model | Change from round 2 | Round 2 RMSE | After the change | Difference |
| --- | --- | --- | --- | --- |
| RidgeAR | alpha 1.0 to 10 | 153.13 | 152.39 | -0.74 |
| LSTM | width 16 to 32 | 160.26 | 162.04 | +1.78 |
| CausalCNN | width 8 to 16 | 160.28 | 160.09 | -0.19 |

The three factors turn out to be separable, and each of the study's round-three
decisions is supported: the stronger Ridge penalty helps, the wider CNN helps
slightly, and the wider LSTM hurts and was correctly not adopted. The study
reached the right conclusions; it just could not demonstrate them at the time.

As a side check, the unchanged models in each run reproduce their round-two
validation RMSE exactly, across separate process invocations. That is the
determinism machinery working.
