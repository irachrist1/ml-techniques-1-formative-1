"""Build the submission guide and an implementation-specific video walkthrough."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    eda = json.loads((ROOT / 'results/eda_summary.json').read_text())
    study = json.loads((ROOT / 'results/study_summary.json').read_text())
    config = json.loads((ROOT / 'configs/final.json').read_text())
    area = eda['top3'][0]
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    winner = study['winners'][0]
    agreement = study['ranking_agreement']
    budget = study['epoch_budget_audit']
    failure = study['failure']

    video = f'''# Individual video plan: 7-10 minutes

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
top three are {eda['top3']}. Compare their scale and shape against 4159 and 4556. Mention that
1 November is an Italian public holiday, so the first low stretch is a three-day weekend. Describe
one pattern you can actually see, and keep it separate from any guess at its cause.

## 3:15-4:45 - Three model mechanisms

Open `experiments.py`. Walk through training-only scaling, the past-only window, the common
144-step eligibility rule and the learned correction to persistence. Explain RidgeAR's lag
coefficients, the LSTM gates and the CNN's causal receptive field
({1 + 2 * sum(config['models']['CausalCNN']['dilations'])} steps against a
{config['models']['CausalCNN']['lookback']}-step input). Show `configs/final.json` and justify one
change from `results/tuning_log.csv`. Explain why 16-22 December never picked a hyperparameter, and
why rolling one-step prediction is allowed to use the already observed part of the test week.

## 4:45-6:15 - Results and live demonstration

Show `results/reference_metrics.csv` and a forecast plot. On area {area} the best learned model at
seed {study['reference_seed']} is {winner['best_learned']}, RMSE {winner['best_learned_rmse']:.2f},
against persistence at {winner['persistence_rmse']:.2f}. Explain MAE versus RMSE and the zero-target
rule for MAPE. Then run:

```sh
python -m unittest -v
python predict_saved.py --run results/runs/final_{config['id']}_area{area}_seed{study['reference_seed']} --model RidgeAR --timestamp '2013-12-16T12:00:00+01:00'
```

This rebuilds one forecast from saved parameters and past observations only. Nothing retrains on
camera. If the script reports an incomplete history, pick another fully observed timestamp.

## 6:15-7:45 - The two results I am least comfortable with

Show `results/validation_vs_test_ranking.csv`: validation and test disagree about the winner in
{agreement['disagree']} of {agreement['areas']} areas. Then `results/peak_bias_summary.csv`: all
{study['peak_underprediction']['combinations']} area-model combinations under-forecast the busiest
decile and over-forecast the quietest. Tie both to the persistence-correction target.

Show `failure_case.png`: area {failure['area']}, error {failure['abs_error']:.0f}
({failure['severity_train_std']:.2f} training SD), and point out that the step *after* the spike is
also wrong because the spike is now in the input.

Mention the epoch-cap audit: {budget['hit_budget']} of {budget['neural_fits']} neural fits now stop
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
'''
    (out / 'video_outline.md').write_text(video)

    checklist = f'''# Before submitting

Verification state is recorded in `results/verification.json`.

- [ ] Record the 7-10 minute individual video using `video_outline.md`.
- [ ] Paste the video URL over `PASTE_VIDEO_LINK_HERE` in `output/report.md` (title block and
      reference [7]), then rebuild the PDF with `python render_report.py`.
- [ ] Confirm the GitHub repository is reachable by the grader. A private repository needs an
      explicit access arrangement.
- [ ] Open the rebuilt PDF and check every figure rendered and no placeholder text remains.
- [ ] Submit the PDF through Canvas by September 20, 2026, 23:59 Kigali time.
- [ ] Open the repository and video links from a logged-out browser session.
'''
    (out / 'submission_checklist.md').write_text(checklist)
    print(out / 'video_outline.md')
    print(out / 'submission_checklist.md')


if __name__ == '__main__':
    main()
