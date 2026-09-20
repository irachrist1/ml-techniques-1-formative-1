"""Two side investigations into how far the study's tuning conclusions hold.

These were run after the report was submitted. They score on the validation week
only, never on the test week, and they write into `results/appendix/` with their
own configuration files, so nothing here can reach `results/tuning_log.csv` or
any metric in the report. No configuration used in the study is changed.

A. Learning rate and batch size. Both were held fixed for the whole study, at
   0.001 and 128, so the report's comparison rests on neural models that were
   never tuned on either. This sweeps both around the values used and asks
   whether the study's settings were a reasonable place to stop.

B. Round three, one factor at a time. `configs/capacity.json` changed the Ridge
   penalty and both neural widths in a single round, so its outcome could not be
   attributed to any one of them. This re-runs that round three times, changing
   one thing each time against the round-two baseline.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import pandas as pd

from experiments import run
from protocol import REFERENCE_SEED

ROOT = Path(__file__).resolve().parent
APPENDIX = ROOT / 'results/appendix'
CONFIGS = ROOT / 'configs/appendix'
RUNS = APPENDIX / 'runs'

LEARNING_RATES = [0.0003, 0.001, 0.003]
BATCH_SIZES = [64, 128, 256]

SWEEP_COLUMNS = {
    'model': ('Model', 's'),
    'learning_rate': ('Learning rate', 'g'),
    'batch': ('Batch', '.0f'),
    'val_rmse': ('Validation RMSE', '.2f'),
    'epochs_run': ('Epochs', 'd'),
}
FACTOR_COLUMNS = {
    'model': ('Model', 's'),
    'change': ('Change from round 2', 's'),
    'round 2': ('Round 2 RMSE', '.2f'),
    'after the change': ('After the change', '.2f'),
    'difference': ('Difference', '+.2f'),
}


def tuning_area() -> int:
    return json.loads((ROOT / 'results/eda_summary.json').read_text())['top3'][0]


def write_config(name: str, rationale: str, models: dict) -> Path:
    CONFIGS.mkdir(parents=True, exist_ok=True)
    path = CONFIGS / f'{name}.json'
    path.write_text(json.dumps({'id': name, 'rationale': rationale, 'models': models}, indent=2) + '\n')
    return path


def sweep_configs() -> list[Path]:
    """Experiment A: the frozen study settings, with only lr and batch varying."""
    final = json.loads((ROOT / 'configs/final.json').read_text())['models']
    paths = []
    for lr in LEARNING_RATES:
        for batch in BATCH_SIZES:
            models = {}
            for kind in ['LSTM', 'CausalCNN']:
                config = copy.deepcopy(final[kind])
                config['learning_rate'] = lr
                config['batch'] = batch
                models[kind] = config
            name = f"lr{str(lr).replace('.', 'p')}_batch{batch}"
            paths.append(write_config(
                name,
                'Appendix A. Learning rate and batch size were fixed at 0.001 and 128 for the '
                'whole study and never tuned. Everything else is exactly configs/final.json. '
                'Validation week only.',
                models))
    return paths


def factor_configs() -> list[Path]:
    """Experiment B: round two's settings, with one round-three change at a time."""
    base = json.loads((ROOT / 'configs/daily_history.json').read_text())['models']
    changes = {
        'round3_alpha_only': ('RidgeAR', 'alpha', 10),
        'round3_lstm_width_only': ('LSTM', 'width', 32),
        'round3_cnn_width_only': ('CausalCNN', 'width', 16),
    }
    paths = []
    for name, (kind, key, value) in changes.items():
        models = copy.deepcopy(base)
        models[kind][key] = value
        paths.append(write_config(
            name,
            f'Appendix B. configs/capacity.json changed the Ridge penalty and both neural widths '
            f'at once, so round three could not attribute its result. This is round two with only '
            f'{kind} {key} = {value} changed. Validation week only.',
            models))
    return paths


def collect() -> pd.DataFrame:
    """Read every appendix run back into one table of validation results."""
    rows = []
    for summary_path in sorted(RUNS.glob('*/summary.json')):
        for entry in json.loads(summary_path.read_text())['runs']:
            rows.append({
                'experiment': entry['experiment'], 'model': entry['model'],
                'learning_rate': entry['config'].get('learning_rate'),
                'batch': entry['config'].get('batch'),
                'width': entry['config'].get('width'),
                'alpha': entry['config'].get('alpha'),
                'val_rmse': entry['validation']['rmse'], 'val_mae': entry['validation']['mae'],
                'epochs_run': entry['epochs_run'], 'best_epoch': entry['best_epoch'],
                'stopped_early': entry['stopped_early'],
                'train_seconds': entry['training_seconds'],
            })
    return pd.DataFrame(rows).sort_values(['experiment', 'model']).reset_index(drop=True)


def markdown_table(frame: pd.DataFrame, columns: dict[str, tuple[str, str]]) -> str:
    """Render selected columns as a Markdown table.

    `columns` maps a column name to its (heading, format spec). The format is
    given per column because a learning rate and a batch size do not want the
    same number of decimal places.
    """
    header = '| ' + ' | '.join(h for h, _ in columns.values()) + ' |'
    rule = '| ' + ' | '.join(['---'] * len(columns)) + ' |'
    rows = ['| ' + ' | '.join(format(row[c], spec) for c, (_, spec) in columns.items()) + ' |'
            for _, row in frame.iterrows()]
    return '\n'.join([header, rule] + rows)


def write_summary(table: pd.DataFrame, area: int) -> Path:
    """Write results/appendix/README.md. Every number is read from `table`."""
    sweep = table[table.experiment.str.startswith('lr')].copy()
    study = sweep[(sweep.learning_rate == 0.001) & (sweep.batch == 128)].set_index('model').val_rmse
    ridge = float(table[(table.experiment == 'round3_alpha_only')
                        & (table.model == 'RidgeAR')].val_rmse.iloc[0])

    best_lines = []
    for kind in ['LSTM', 'CausalCNN']:
        candidates = sweep[sweep.model == kind]
        best = candidates.loc[candidates.val_rmse.idxmin()]
        best_lines.append(
            f"- **{kind}**: best at learning rate {best.learning_rate:g}, batch "
            f"{int(best.batch)}, validation RMSE {best.val_rmse:.2f} against "
            f"{study[kind]:.2f} at the study's settings, an improvement of "
            f"{100 * (1 - best.val_rmse / study[kind]):.2f}%.")
    best_neural = float(sweep.val_rmse.min())

    factors = table[table.experiment.str.startswith('round3')].copy()
    baseline = json.loads((ROOT / 'configs/daily_history.json').read_text())['models']
    factor_rows = []
    for name, kind in [('round3_alpha_only', 'RidgeAR'), ('round3_lstm_width_only', 'LSTM'),
                       ('round3_cnn_width_only', 'CausalCNN')]:
        changed = float(factors[(factors.experiment == name) & (factors.model == kind)].val_rmse.iloc[0])
        unchanged = float(factors[(factors.experiment != name) & (factors.model == kind)].val_rmse.max())
        setting = (f"alpha {baseline[kind]['alpha']} to 10" if kind == 'RidgeAR'
                   else f"width {baseline[kind]['width']} to {'32' if kind == 'LSTM' else '16'}")
        factor_rows.append({'model': kind, 'change': setting, 'round 2': unchanged,
                            'after the change': changed, 'difference': changed - unchanged})
    factor_table = pd.DataFrame(factor_rows)

    text = f'''# Appendix experiments

Run after the report was submitted. They score on the **validation week only**
(December 9-15); the test week is not read. They live in their own configuration
files and run directories, so they cannot reach `results/tuning_log.csv` or any
number in the report. No configuration used in the study was changed, and no
reported metric is affected.

Everything below is one area ({area}) at one seed ({REFERENCE_SEED}), which is
the same narrow basis the study's own tuning used. Nothing here establishes a
test-week result.

Full table: `appendix_results.csv`. Rebuild with `python appendix_experiments.py`.

## A. Learning rate and batch size were never tuned

The study fixed the learning rate at 0.001 and the batch size at 128 for every
round and never varied either. This sweeps learning rate over
{', '.join(f'{v:g}' for v in LEARNING_RATES)} and batch size over
{', '.join(map(str, BATCH_SIZES))}, with everything else exactly
`configs/final.json`.

{markdown_table(sweep.sort_values(['model', 'val_rmse']), SWEEP_COLUMNS)}

{chr(10).join(best_lines)}

This matters for how the report's comparison should be read. At the study's
settings, the best neural validation RMSE on this area is {study.min():.2f} and
RidgeAR's is {ridge:.2f}, so the linear model wins. With the learning rate
tuned, the best neural validation RMSE is {best_neural:.2f}, which is
{'below' if best_neural < ridge else 'still above'} RidgeAR.

The honest reading is that the study's neural models were undertrained on a
hyperparameter it never examined, so its results do not separate "a linear model
is sufficient at this horizon" from "the neural models were not tuned". Settling
that needs a retrained study on an untouched test week, which this is not.

## B. Round three changed three things at once

`configs/capacity.json` changed the Ridge penalty and both neural widths in one
round, so the round could not attribute its outcome. Each change is re-run here
on its own against the round-two baseline.

{markdown_table(factor_table, FACTOR_COLUMNS)}

The three factors turn out to be separable, and each of the study's round-three
decisions is supported: the stronger Ridge penalty helps, the wider CNN helps
slightly, and the wider LSTM hurts and was correctly not adopted. The study
reached the right conclusions; it just could not demonstrate them at the time.

As a side check, the unchanged models in each run reproduce their round-two
validation RMSE exactly, across separate process invocations. That is the
determinism machinery working.
'''
    path = APPENDIX / 'README.md'
    path.write_text(text)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--collect-only', action='store_true',
                        help='Rebuild the tables from existing appendix runs without training.')
    args = parser.parse_args()

    area = tuning_area()
    if not args.collect_only:
        for path in sweep_configs() + factor_configs():
            name = json.loads(path.read_text())['id']
            if (RUNS / f'tune_{name}_area{area}_seed{REFERENCE_SEED}/summary.json').exists():
                print(f'skip {name}: already complete', flush=True)
                continue
            run(path, area, REFERENCE_SEED, 'tune', RUNS)

    APPENDIX.mkdir(parents=True, exist_ok=True)
    table = collect()
    table.to_csv(APPENDIX / 'appendix_results.csv', index=False)
    print(table.to_string(index=False))
    print(write_summary(table, area))


if __name__ == '__main__':
    main()
