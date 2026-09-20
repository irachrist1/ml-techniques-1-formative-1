"""Refresh project instructions after all verified experiment outputs exist."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

eda = json.loads((ROOT / 'results/eda_summary.json').read_text())
study = json.loads((ROOT / 'results/study_summary.json').read_text())
config = json.loads((ROOT / 'configs/final.json').read_text())
repository = json.loads((ROOT / 'output/repository.json').read_text())
separation = json.loads((ROOT / 'results/seed_separation.json').read_text())
verification = json.loads((ROOT / 'results/verification.json').read_text())

rows = '\n'.join(
    f"- Area {w['area']}: best learned reference model **{w['best_learned']}**, "
    f"RMSE {w['best_learned_rmse']:.2f}; persistence {w['persistence_rmse']:.2f}."
    for w in study['winners'])
top = eda['top3'][0]

text = f'''# Milan mobile-network traffic forecasting

ML Techniques I · Formative Assignment 1 · Christian Tonny

**Video:** {repository['video']}

A reproducible comparison of RidgeAR, LSTM and a causal dilated CNN for rolling one-step Internet-activity forecasting. The full November–December 2013 dataset is processed; December 16–22 is the held-out test week. The study covers five areas: the top three by full-period activity ({', '.join(map(str, eda['top3']))}), plus 4159 and 4556.

## Quick start

Everything needed to rebuild the study's tables, figures and model checks is committed. No download is required.

```sh
make setup        # Python 3.12 venv + the exact recorded environment
make reproduce    # tests, then rebuild every table and figure, then revalidate all 45 saved models
```

`make reproduce` runs `python reproduce.py`, which is the single entry point. It takes about 20 seconds and ends by reloading every saved model and replaying {verification['replayed_forecasts']} forecasts. `make help` lists the other targets; `python reproduce.py --stages all` runs the whole chain including the 20 GB download.

## Results at a glance

Seed 42 is the reference run; seeds 43 and 44 test optimization sensitivity. These are activity-unit errors, not GB or bandwidth measurements.

{rows}

Read those rankings with `results/seed_separation.csv` beside them. In {separation['areas_where_ranges_overlap']} of {separation['areas']} areas the top two models' RMSE ranges overlap across seeds, and only {separation['areas_where_gap_exceeds_combined_sd']} area has a gap larger than the combined seed standard deviation. Validation and test also disagree about the winner in {study['ranking_agreement']['disagree']} of {study['ranking_agreement']['areas']} areas. One week at one seed does not establish an ordering of these three architectures.

Complete tables: `results/reference_metrics.csv`, `results/validation_metrics_all_seeds.csv`, `results/validation_vs_test_ranking.csv`, `results/seed_separation.csv`, `results/peak_bias_summary.csv`, `results/seed_summary.csv`, `results/timing.csv`. Forecast plots are under `results/figures/`; all test predictions and trained models are retained under `results/runs/`. Baselines are persistence and daily-seasonal persistence.

## Setup

Use **Python 3.12**. The study was run on an Apple M2 Pro with 16 GB RAM on CPU. No API key or paid service is needed.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m unittest -v
```

`requirements.txt` lists direct dependencies; `requirements-lock.txt` records the complete executed environment. Operating-system differences can affect package availability and floating-point or timing results.

## Inspect the delivered study without downloading 20 GB

The selected five-area series, model files, plots and predictions are all committed:

```sh
python reproduce.py
python predict_saved.py --run results/runs/final_{config['id']}_area{top}_seed42 --model RidgeAR --timestamp '2013-12-16T12:00:00+01:00'
```

The replay command loads saved parameters and uses only observations preceding the target. It refuses incomplete, off-grid or duplicated histories. `verify_submission.py` reloads every final model, checks the first, middle and last test forecasts, recomputes metrics, checks split dates, and verifies data fingerprints, training scalers, stopping metadata and the exact eligible validation and test timestamps.

## Tests

```sh
python -m unittest -v
```

{verification['complete_final_models']} models are covered by the verification script; the unit tests cover the things that would silently corrupt results if they broke: window and lag construction, the half-open split boundaries, fit-on-training-only normalization, inverse-transform round trips, chunk-boundary aggregation, the metric definitions, and the saved-model replay guards. `test_report_provenance.py` pins each numeric claim in the report to the artifact that produces it, so a rerun that moved a result would fail rather than leave the report disagreeing with its own tables.

## Reproduce from original data

Source: [Telecom Italia / Harvard Dataverse](https://doi.org/10.7910/DVN/EGZHFV). The required 61 daily files total **20.48 GB**; January 1 is excluded. Allow extra disk for processed arrays and partial downloads. Read the publisher's ODbL terms and `DATA_LICENSE.md` before downloading or redistributing derivatives. The metadata snapshot lists the publisher's checksums.

Provide your own email for the publisher's required guestbook response. Do not commit the email or transient signed URLs.

```sh
read -r DATAVERSE_EMAIL
export DATAVERSE_EMAIL
make data
```

The downloader resumes partial transfers, checks MD5 before processing and refuses to trust an existing raw file with a wrong checksum. `download_remaining.py` is an optional bounded-range recovery tool; do not run it at the same time as the ordinary downloader. `analyze.py` requires all 61 daily outputs, validates the timestamp grid, ranks every area over the full period and rebuilds the five selected series.

## Reproduce training

`configs/initial.json`, `configs/daily_history.json`, `configs/capacity.json` and `configs/budget.json` record the four validation rounds in order, each carrying the reasoning for its change in its own `rationale` field. `configs/final.json` records the frozen settings. `results/tuning_log.csv` is generated from the run summaries, not transcribed.

The epoch-budget revision in round four followed an audit made after the original test results were available. That is a reevaluation on the same week, not a fresh untouched holdout, and `results/revision_provenance.json` keeps the chronology.

```sh
make tune     # the four validation rounds, on area {top}, seed 42
make train    # the frozen configuration on five areas and three seeds
```

Completed runs are skipped rather than overwritten, because a run directory is evidence. To retrain from nothing, move `results/runs/` aside first.

Training: November 1–December 8. Validation: December 9–15. Test: December 16–22, all local Europe/Rome dates. Normalization is fitted on training values only. All candidates require the same complete 144-step history for eligibility, including those that consume fewer lags. No missing target is imputed. Shared hyperparameters were tuned on the highest-activity area; weights and scaling are fitted separately per area. Test observations enter only as already observed history for later one-step predictions, never for fitting or tuning.

## Appendix experiments

`results/appendix/` holds experiments run after the report was submitted, to test how far the tuning conclusions hold. They use their own configuration files and run directories; no configuration used in the study was changed and no reported metric is affected. `results/appendix/README.md` states what each one asked and what it found.

## Edit the report

`output/report.md` is the editable source and is hand-edited, so `create_report.py` refuses to overwrite it without `--replace`. Rebuild the PDF with:

```sh
make report
```

`output/pdf/` is generated and not committed. `output/formative1_report.docx` is the Word version; regenerate it from the Markdown with Node.js and `docx` 9.6.1:

```sh
npm install --no-save --package-lock=false docx@9.6.1
node create_word.cjs
```

That replaces the Word file, so preserve any Word-only edits first.

## Code map

Pipeline:

- `protocol.py`: the fixed experimental protocol — grid, split boundaries, seeds, model registry.
- `prepare_day.py`: chunked aggregation of one raw daily file, plus the daily audit.
- `analyze.py`: full-period ranking and training-period exploratory analysis.
- `features.py`: past-only windows, the common eligibility mask, splits and scaling.
- `models.py`: the three model definitions and the deterministic input pipeline.
- `experiments.py`: training orchestration; writes one self-describing run directory.
- `evaluation.py`: metrics and chronological helpers.
- `summarize_results.py`: comparisons, seed sensitivity and failure diagnostics.
- `compare_seeds.py`: whether each area's ranking survives seed variation.
- `adf_specifications.py`: the ADF test under both lag specifications.
- `predict_saved.py`, `verify_submission.py`: model replay and output checks.
- `reproduce.py`: the single entry point that runs the stages in order.

Not needed to reproduce the study: `create_report.py`, `render_report.py`, `prepare_submission.py`, `write_readme.py`, `memory_benchmark.py`, `download_data.py`, `download_remaining.py`.

## Scope and limitations

This is a single-week, single-horizon, single-origin study of five areas. Area selection uses full-period totals because the assignment requests it, which is a selection dependence and limits prospective generalization. Models see only their own area's history and no calendar features. Shared tuning does not optimize each area separately, and the learning rate and batch size were held fixed throughout the study's tuning rounds. Missing-window exclusion can bias evaluation toward fully observed periods. Three seeds measure optimization variability, not uncertainty about future weeks, and `results/seed_separation.csv` shows most per-area rankings are not separated by that variability. Local training and inference timings include system noise and are not production benchmarks.

AI assistance is recorded in `AI_CONTRIBUTIONS.md`, and the report carries the disclosure. Dataset attribution and ODbL obligations: `DATA_LICENSE.md`. Research basis and differences from prior studies: `RESEARCH_NOTES.md`. Artificial values occur only in unit-test fixtures, never in reported results.
'''

(ROOT / 'README.md').write_text(text)
print(ROOT / 'README.md')
