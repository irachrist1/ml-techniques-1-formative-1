"""Build the submission guide and an implementation-specific video walkthrough."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent


def main():
    eda=json.loads((ROOT/'results/eda_summary.json').read_text());study=json.loads((ROOT/'results/study_summary.json').read_text());config=json.loads((ROOT/'configs/final.json').read_text())
    area=eda['top3'][0];out=ROOT/'output';out.mkdir(exist_ok=True)
    winner=study['winners'][0]
    video=f'''# Individual video: approximately 8–9 minutes

Use the following timings to present the study and demonstrate the code.

## 0:00–0:45 — Question and practical relevance

Introduce one-step (10-minute) Internet-activity forecasting across Milan areas. Explain the distinction between activity units and data volume in GB. State why forecasting could inform network capacity allocation.

## 0:45–2:00 — Real data and memory

Show `prepare_day.py`, the published dataset DOI in `DATA_LICENSE.md`, and `results/memory_benchmark.json`. Explain summing country-code records, why an empty activity field is not the same as observed zero, why all timestamps are preserved, and why loading one day in chunks reduces memory. The separate-process comparison is on the same complete day; it is not a theoretical extrapolation from a tiny sample.

## 2:00–3:15 — What the data reveals

Show `traffic_distribution.png`, `first_two_weeks.png` and `temporal_analysis.png`. The full-period top three are {eda['top3']}. Compare their scale and temporal variation with squares 4159 and 4556. Explain one pattern you can actually see and distinguish that observation from a possible real-world explanation. Do not invent neighborhood identities.

## 3:15–4:45 — Three model mechanisms and honest evaluation

Open `experiments.py`. Walk through training-only scaling, the past-only input window and the learned correction to persistence. Explain the linear coefficients of RidgeAR, gates in LSTM and the causal receptive field of CausalCNN. Show `configs/final.json` and justify a specific change using `results/tuning_log.csv`. Explain why December 16–22 never selected hyperparameters and why rolling one-step prediction can use the already observed part of the test week.

## 4:45–6:15 — Results and demonstration

Show `results/reference_metrics.csv` and a forecast plot. On area {area}, the best learned reference model is {winner['best_learned']}, RMSE {winner['best_learned_rmse']:.2f}; persistence RMSE is {winner['persistence_rmse']:.2f}. Compare against another area rather than declaring a universal winner. Explain MAE versus RMSE and the zero-target rule for MAPE. Show timing and seed variability.

Run these commands from the project directory with the environment activated:

```sh
python -m unittest -v
python predict_saved.py --run results/runs/final_{config['id']}_area{area}_seed42 --model RidgeAR --timestamp '2013-12-16T12:00:00+01:00'
```

The demonstration reconstructs one forecast from saved parameters and past observations. It does not retrain during the video. Use another fully observed timestamp if the script explicitly reports an incomplete history.

## 6:15–7:30 — Failure case and trade-off

Show `failure_case.png`. Identify the visible miss, describe how the predictions respond around it, and explain what a univariate model cannot know. Compare that limitation with the computational cost of a more complex model. Do not assign a real-world event cause without evidence.

## 7:30–8:30 — Conclusion and next experiment

State the result you find most defensible, a limitation of using one test week and selected areas, and the next experiment you would run. Explain how AI assisted the project and what you checked yourself. End by showing the repository and its reproduction instructions.

## Understanding check before recording

- Why does country-code aggregation matter?
- How are missing activity and an observed zero different?
- Can full-period area selection affect generalization claims?
- Where are normalization statistics fitted?
- How is predicting the next interval different from predicting an entire future week?
- How does the convolutional receptive field cover the lookback?
- Why could persistence or RidgeAR beat a neural model?
- What does random-seed variation fail to measure?
- Which result justifies the next hyperparameter change?
- What would you change for an unseen area or a longer horizon?
'''
    (out/'video_outline.md').write_text(video)
    (out/'submission_checklist.md').write_text('''# Before submitting

Validation results are recorded in `results/verification.json`.

- Review the report against the saved figures and metrics. Revise the interpretation and conclusions in your own words; retain an accurate AI disclosure.
- Resolve any questions using the code walkthrough and explain at least one modeling decision and failure case without reading generated prose.
- Record the required 7–10 minute individual video using `video_outline.md`.
- Add the real accessible video URL to the report references. 
- Ensure the GitHub repository is accessible to the grader. A private repository needs an appropriate access arrangement.
- Rebuild the PDF with `python render_report.py` after editing `output/report.md`; inspect the rendered pages.
- Submit the PDF through Canvas by September 20, 2026, 23:59 Kigali time. Confirm the repository and video links work for the intended audience.

''')
    print('Video outline and submission checklist created')


if __name__=='__main__':main()
