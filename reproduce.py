"""Single entry point that reproduces the study end to end.

Stages, in dependency order:

  data       Download the 61 raw daily files, aggregate them, rank all 10,000
             areas and rebuild the selected series. Needs DATAVERSE_EMAIL and
             about 21 GB of disk. Everything downstream is committed, so this
             stage is only needed to rebuild the study from the publisher's data.
  tune       Replay the four validation rounds in order on the tuning area.
  train      Fit the frozen final configuration on every area and seed.
  summarize  Rebuild every results table and figure from saved predictions.
  verify     Reload every saved model, replay forecasts and recheck metrics.
  appendix   Post-submission side investigations, written to results/appendix/.
             Validation week only; cannot affect any number in the report.

Default (`python reproduce.py`) runs test + summarize + verify: it regenerates
every table and figure in the report and revalidates all 45 saved models without
downloading anything. `python reproduce.py --stages all` runs the whole chain.

Completed runs are skipped rather than overwritten, because a run directory is
evidence. To retrain from nothing, move `results/runs/` aside first.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from protocol import REFERENCE_SEED, SEEDS

ROOT = Path(__file__).resolve().parent

#: The four tuning rounds, in the order they were actually run. Each config file
#: carries the reasoning for its change in its own `rationale` field.
TUNING_ROUNDS = ['initial', 'daily_history', 'capacity', 'budget']

STAGES = ['test', 'data', 'tune', 'train', 'summarize', 'verify', 'appendix']
DEFAULT_STAGES = ['test', 'summarize', 'verify']


def step(*command: str) -> None:
    """Run one command from the repository root, echoing it, and stop on failure."""
    print(f'\n$ {" ".join(command)}', flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def tuning_area() -> int:
    """The single area the shared hyperparameters were tuned on (the busiest one)."""
    return json.loads((ROOT / 'results/eda_summary.json').read_text())['top3'][0]


def stage_test() -> None:
    step(sys.executable, '-m', 'unittest', '-v')


def stage_data() -> None:
    step(sys.executable, 'download_data.py', '--workers', '4')
    step(sys.executable, 'analyze.py')
    step(sys.executable, 'memory_benchmark.py')


def stage_tune() -> None:
    area = tuning_area()
    for name in TUNING_ROUNDS:
        if (ROOT / 'results/runs' / f'tune_{name}_area{area}_seed{REFERENCE_SEED}/summary.json').exists():
            print(f'skip tune/{name}: already complete', flush=True)
            continue
        step(sys.executable, 'experiments.py', '--config', f'configs/{name}.json',
             '--area', str(area), '--seed', str(REFERENCE_SEED), '--phase', 'tune')


def stage_train() -> None:
    config = json.loads((ROOT / 'configs/final.json').read_text())
    areas = json.loads((ROOT / 'results/eda_summary.json').read_text())['areas']
    for area in areas:
        for seed in SEEDS:
            if (ROOT / 'results/runs' / f"final_{config['id']}_area{area}_seed{seed}/summary.json").exists():
                print(f'skip final/area{area}/seed{seed}: already complete', flush=True)
                continue
            step(sys.executable, 'experiments.py', '--config', 'configs/final.json',
                 '--area', str(area), '--seed', str(seed), '--phase', 'final')


def stage_summarize() -> None:
    step(sys.executable, 'summarize_results.py')
    step(sys.executable, 'compare_seeds.py')


def stage_verify() -> None:
    step(sys.executable, 'verify_submission.py')
    step(sys.executable, 'check_regenerated.py')


def stage_appendix() -> None:
    step(sys.executable, 'appendix_experiments.py')


RUNNERS = {'test': stage_test, 'data': stage_data, 'tune': stage_tune,
           'train': stage_train, 'summarize': stage_summarize, 'verify': stage_verify,
           'appendix': stage_appendix}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stages', nargs='+', default=DEFAULT_STAGES,
                        choices=STAGES + ['all'],
                        help=f'Stages to run (default: {" ".join(DEFAULT_STAGES)}).')
    args = parser.parse_args()
    stages = STAGES if 'all' in args.stages else [s for s in STAGES if s in args.stages]

    started = time.perf_counter()
    for name in stages:
        print(f'\n{"=" * 70}\n== {name}\n{"=" * 70}', flush=True)
        RUNNERS[name]()
    print(f'\nCompleted stages {stages} in {time.perf_counter() - started:.1f}s', flush=True)


if __name__ == '__main__':
    main()
