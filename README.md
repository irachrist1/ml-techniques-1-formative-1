# Milan mobile-network traffic forecasting

ML Techniques I · Formative Assignment 1 · Christian Tonny

A reproducible comparison of RidgeAR, LSTM and a causal dilated CNN for rolling one-step Internet-activity forecasting. The full November–December 2013 dataset is processed; December 16–22 is the held-out test week. The study includes all five areas analyzed in this study: top three (5161, 5059, 5259), plus 4159 and 4556.

## Submission materials

- `output/report.md`: editable research report.
- `output/formative1_report.docx`: Word version with editable text and tables.
- `output/pdf/formative1_report.pdf`: research report PDF.
- `output/video_outline.md`: timed 8–9 minute presentation and implementation walkthrough.
- `output/submission_checklist.md`: remaining student actions and link checks.
- `results/verification.json`: saved-model replay and metric verification.

## Results at a glance

Seed 42 is the predetermined reference run; seeds 43 and 44 test optimization sensitivity. These are activity-unit errors, not GB or bandwidth measurements.

- Area 5161: best learned reference model **CausalCNN**, RMSE 118.16; persistence 134.88.
- Area 5059: best learned reference model **RidgeAR**, RMSE 99.66; persistence 114.38.
- Area 5259: best learned reference model **RidgeAR**, RMSE 93.17; persistence 109.58.
- Area 4159: best learned reference model **RidgeAR**, RMSE 19.88; persistence 21.54.
- Area 4556: best learned reference model **RidgeAR**, RMSE 35.57; persistence 39.62.

Complete tables: `results/reference_metrics.csv`, `results/seed_summary.csv`, `results/timing.csv`. Each model-specific forecast plot is under `results/figures/`; all test predictions and trained models are retained under `results/runs/`. Baselines are persistence and daily seasonal persistence.

## Setup

Use **Python 3.12**. This project was run on an Apple M2 Pro with 16 GB RAM, using CPU execution. No API key or paid cloud service is needed for training.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m unittest -v
```

`requirements.txt` lists direct dependencies; `requirements-lock.txt` records the complete executed environment. Operating-system differences can affect package availability and floating-point/timing results.

## Inspect the delivered study without downloading 20 GB

The selected five-area series, model files, plots and predictions are included. After installing dependencies:

```sh
python summarize_results.py
python verify_submission.py
python predict_saved.py --run results/runs/final_selected_area5161_seed42 --model RidgeAR --timestamp '2013-12-16T12:00:00+01:00'
```

The replay command loads saved parameters and uses only observations preceding the target. It refuses incomplete histories. The verification script reloads every final model and checks one interior forecast, recomputes metrics, checks split dates, verifies data fingerprints and enforces identical scored timestamps across models.

## Reproduce from original data

Source: [Telecom Italia / Harvard Dataverse](https://doi.org/10.7910/DVN/EGZHFV). The required 61 daily files total **20.48 GB**; January 1 is excluded. Allow additional disk space for processed arrays and incomplete downloads. Read the publisher's ODbL terms and `DATA_LICENSE.md` before downloading or distributing derivatives. The metadata snapshot lists the publisher's checksums.

Provide your own email for the publisher's required guestbook response. Do not commit the email or transient signed URLs.

```sh
read -r DATAVERSE_EMAIL
export DATAVERSE_EMAIL
python download_data.py --workers 4
python analyze.py
python memory_benchmark.py
```

The downloader resumes partial transfers, checks MD5 before processing and refuses to trust an existing raw file with a wrong checksum. `download_remaining.py` is an optional bounded-range recovery tool; do not run it concurrently with the ordinary downloader. The latter recovery tool retains temporary pieces, which can be removed after the full files have passed checksums.

`analyze.py` requires all 61 daily outputs, validates the timestamp grid, ranks every area over the full period and rebuilds the five selected series. It generates distribution, temporal-analysis, first-two-week and STL figures. Missing country-level Internet fields are omitted from sums; aggregate bins with no observed values stay missing.

## Reproduce training

`configs/initial.json` and the subsequent candidate configurations record the manual validation experiments. `configs/final.json` freezes the per-model settings before test evaluation. `results/tuning_log.csv` documents the observed validation results and the rationale for changes.

The delivered `results/runs/` contains completed runs. To retrain without overwriting those records, first move that directory to a separate backup location. Then run:

```sh
python run_final.py
```

This fits all three models independently on five areas for seeds 42, 43 and 44, then regenerates comparisons and verifies saved models. Existing completed runs are skipped. A partially written run should be moved aside before retrying; completed experiment IDs are protected against accidental overwrite.

To rerun a validation candidate in a clean results directory:

```sh
python experiments.py --config configs/initial.json --area 5161 --seed 42 --phase tune
```

Training: November 1–December 8. Validation: December 9–15. Test: December 16–22, all local Europe/Rome dates. Normalization is fitted on training values only. All candidates require the same complete 144-step history for eligibility, including those that consume fewer lags. No missing target is imputed. The shared hyperparameters are tuned on the highest-activity area; weights and scaling are fitted separately per area. Test observations enter only as already observed history for later one-step predictions, not for fitting or tuning.

## Edit the report

Edit `output/report.md` and rebuild with:

```sh
python render_report.py
```

The renderer rebuilds the PDF. `create_report.py` generates the report from results and deliberately refuses to overwrite an existing editable report. `prepare_submission.py` regenerates the video outline and checklist. Figures are computed from data; they are not illustrative synthetic curves.

## Word editing

Open `output/formative1_report.docx` in Word and edit it directly. Export the edited document to PDF for submission. Word edits do not automatically update `output/report.md`; running the Markdown renderer later will rebuild from the Markdown, not incorporate Word edits.

To regenerate a Word copy from the Markdown, use Node.js and `docx` 9.6.1:

```sh
npm install --no-save --package-lock=false docx@9.6.1
node create_word.cjs
```

This replaces the Word file, so preserve any Word-only edits first. Figures are embedded images; text and tables remain editable.

## Code map

- `prepare_day.py`: chunked aggregation and daily audit.
- `analyze.py`: full-period ranking and training-period EDA.
- `experiments.py`: past-only windows, training, timing and prediction records.
- `evaluation.py`: metrics and chronological helper functions.
- `summarize_results.py`: comparisons, seed sensitivity and failure analysis.
- `memory_benchmark.py`: isolated-process baseline versus chunked memory comparison.
- `predict_saved.py`, `verify_submission.py`: model replay and output checks.

## Scope and limitations

This is a single-week, single-horizon empirical study. Area selection uses full-period totals because the assignment requests it; this is a selection dependence and limits prospective generalization claims. Models use only the selected area's history. Shared tuning does not optimize every area separately. Missing-window exclusion can bias the evaluation toward fully observed periods. Seed variation is not a confidence interval over future weeks. Local training and inference measurements include system noise and are not production service benchmarks.

Dataset attribution and ODbL obligations: `DATA_LICENSE.md`. Research basis and differences from prior studies: `RESEARCH_NOTES.md`. Artificial values occur only in unit-test fixtures, never in reported assignment results.
