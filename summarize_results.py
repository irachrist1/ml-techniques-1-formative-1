"""Build comparisons and failure diagnostics from frozen, saved test predictions."""
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from evaluation import score
from protocol import COLORS, REFERENCE_SEED, SEEDS, TEST_END, TEST_START

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
FIGURES = RESULTS / 'figures'
TIMING_COLUMNS = ['train_seconds', 'batch_inference_seconds', 'single_inference_ms']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.dpi': 150})


def load_predictions(folder: Path, kind: str, split: str) -> pd.DataFrame:
    p = pd.read_csv(folder / f'{kind}_{split}_predictions.csv')
    if p.timestamp_ms.duplicated().any():
        raise ValueError(f'Duplicate predictions in {folder.name}/{kind}_{split}')
    return p


def forecast_plot(area: int, kind: str, p: pd.DataFrame) -> None:
    local = pd.to_datetime(p.timestamp_ms, unit='ms', utc=True).dt.tz_convert('Europe/Rome')
    fig, ax = plt.subplots(figsize=(9, 2.7))
    ax.plot(local, p.actual, label='Observed', color='#252525', lw=.85)
    ax.plot(local, p.prediction, label=kind, color=COLORS[kind], lw=.7, alpha=.85)
    ax.set(title=f'Area {area}: {kind}, rolling one-step forecasts (seed {REFERENCE_SEED})',
           xlabel='16-22 December 2013, Europe/Rome', ylabel='Internet activity units')
    ax.legend(ncol=2, frameon=False)
    fig.autofmt_xdate()
    fig.savefig(FIGURES / f'forecast_{area}_{kind}.png', bbox_inches='tight')
    plt.close(fig)


def failure_neighbors(predictions: dict, area: int, index: int) -> tuple:
    reference = predictions[(area, 'RidgeAR')]
    previous = float(reference.actual.iloc[index - 1]) if index > 0 else None
    following = None
    if index + 1 < len(reference):
        following = {'actual': float(reference.actual.iloc[index + 1]),
                     **{kind: float(predictions[(area, kind)].prediction.iloc[index + 1]) for kind in COLORS}}
    return previous, following


def main() -> None:
    selected = json.loads((ROOT / 'configs/final.json').read_text())
    eda = json.loads((RESULTS / 'eda_summary.json').read_text())
    areas = eda['areas']
    name = selected['id']

    rows, validation_rows, baseline_rows, timing = [], [], [], []
    predictions, reference_times = {}, {}

    for area in areas:
        for seed in SEEDS:
            folder = RESULTS / 'runs' / f'final_{name}_area{area}_seed{seed}'
            summary = json.loads((folder / 'summary.json').read_text())
            for r in summary['runs']:
                kind = r['model']
                p = load_predictions(folder, kind, 'test')
                if not p.timestamp_ms.between(TEST_START, TEST_END - 1).all():
                    raise ValueError('Wrong test dates')
                previous = reference_times.setdefault(area, p.timestamp_ms.to_numpy())
                np.testing.assert_array_equal(previous, p.timestamp_ms.to_numpy())

                metrics = score(p.actual, p.prediction)
                if not np.isclose(metrics['rmse'], r['test']['rmse']):
                    raise ValueError('Saved metrics and predictions disagree')
                rows.append({'area': area, 'model': kind, 'seed': seed, **metrics})

                v = load_predictions(folder, kind, 'validation')
                validation_rows.append({'area': area, 'model': kind, 'seed': seed, **score(v.actual, v.prediction)})

                timing.append({
                    'area': area, 'model': kind, 'seed': seed, 'parameters': r['parameter_count'],
                    'epochs': r['epochs_run'], 'epoch_budget': r.get('epoch_budget'),
                    'stopped_early': r.get('stopped_early'), 'best_epoch': r['best_epoch'],
                    'train_seconds': r['training_seconds'],
                    'batch_inference_seconds': r['inference']['warm_batch_seconds_median'],
                    'single_inference_ms': 1000 * r['inference']['single_seconds_median'],
                })
                if seed == REFERENCE_SEED:
                    predictions[(area, kind)] = p
                    forecast_plot(area, kind, p)

            if seed == REFERENCE_SEED:
                # Baselines live in every model's file identically; read them from one
                # named model rather than whichever happened to be last in the loop.
                base = predictions[(area, 'RidgeAR')]
                for label, column in [('Persistence', 'persistence'), ('Daily seasonal', 'seasonal_daily')]:
                    baseline_rows.append({'area': area, 'model': label, 'seed': REFERENCE_SEED,
                                          **score(base.actual, base[column])})

    metrics = pd.DataFrame(rows)
    validation = pd.DataFrame(validation_rows)
    baselines = pd.DataFrame(baseline_rows)
    times = pd.DataFrame(timing)
    metrics.to_csv(RESULTS / 'test_metrics_all_seeds.csv', index=False)
    validation.to_csv(RESULTS / 'validation_metrics_all_seeds.csv', index=False)
    baselines.to_csv(RESULTS / 'baseline_metrics.csv', index=False)
    times.to_csv(RESULTS / 'timing.csv', index=False)

    reference = pd.concat([metrics[metrics.seed == REFERENCE_SEED], baselines], ignore_index=True)
    reference.to_csv(RESULTS / 'reference_metrics.csv', index=False)

    aggregate = metrics.groupby(['area', 'model']).agg(
        rmse_mean=('rmse', 'mean'), rmse_std=('rmse', 'std'), rmse_min=('rmse', 'min'),
        rmse_max=('rmse', 'max'), mae_mean=('mae', 'mean'), mape_mean=('mape_percent_nonzero', 'mean')
    ).reset_index()
    aggregate.to_csv(RESULTS / 'seed_summary.csv', index=False)

    # Does the model that wins the tuning week also win the held-out week?
    ranking_rows = []
    for area in areas:
        v = validation[(validation.seed == REFERENCE_SEED) & (validation.area == area)]
        t = metrics[(metrics.seed == REFERENCE_SEED) & (metrics.area == area)]
        v_best = v.loc[v.rmse.idxmin(), 'model']
        t_best = t.loc[t.rmse.idxmin(), 'model']
        ranking_rows.append({'area': area, 'validation_best': v_best, 'test_best': t_best,
                             'agree': v_best == t_best,
                             **{f'val_{m}': float(v[v.model == m].rmse.iloc[0]) for m in COLORS},
                             **{f'test_{m}': float(t[t.model == m].rmse.iloc[0]) for m in COLORS}})
    ranking = pd.DataFrame(ranking_rows)
    ranking.to_csv(RESULTS / 'validation_vs_test_ranking.csv', index=False)

    seed_rankings = []
    for seed in SEEDS:
        for area in areas:
            v = validation[(validation.seed == seed) & (validation.area == area)]
            t = metrics[(metrics.seed == seed) & (metrics.area == area)]
            vb, tb = v.loc[v.rmse.idxmin(), 'model'], t.loc[t.rmse.idxmin(), 'model']
            seed_rankings.append({'seed': seed, 'area': area, 'validation_best': vb,
                                  'test_best': tb, 'agree': vb == tb})
    pd.DataFrame(seed_rankings).to_csv(RESULTS / 'validation_vs_test_by_seed.csv', index=False)

    # Select the failure event by largest reference-model absolute error / training std.
    failure = None
    for area in areas:
        scale = next(r['train_std'] for r in eda['area_stats'] if r['square'] == area)
        for kind in COLORS:
            p = predictions[(area, kind)]
            error = np.abs(p.actual - p.prediction).to_numpy()
            i = int(np.argmax(error))
            severity = float(error[i] / scale)
            if failure is None or severity > failure['severity_train_std']:
                failure = {'area': area, 'model': kind, 'index': i,
                           'timestamp_ms': int(p.timestamp_ms.iloc[i]), 'actual': float(p.actual.iloc[i]),
                           'prediction': float(p.prediction.iloc[i]), 'abs_error': float(error[i]),
                           'severity_train_std': severity}
    fa, fi = failure['area'], failure['index']
    failure['previous_actual'], failure['next_step'] = failure_neighbors(predictions, fa, fi)

    center = failure['timestamp_ms']
    fig, ax = plt.subplots(figsize=(9, 3.5))
    p = predictions[(fa, 'RidgeAR')]
    mask = p.timestamp_ms.between(center - 18 * 600000, center + 18 * 600000)
    local = pd.to_datetime(p.timestamp_ms[mask], unit='ms', utc=True).dt.tz_convert('Europe/Rome')
    ax.plot(local, p.actual[mask], color='#252525', lw=1.6, label='Observed')
    for kind in COLORS:
        ax.plot(local, predictions[(fa, kind)].prediction[mask], lw=1, label=kind, color=COLORS[kind])
    ax.set(title=f'Largest standardized forecast miss: area {fa}',
           xlabel='Local date and time (Europe/Rome), +/-3 hours around event',
           ylabel='Internet activity units')
    ax.legend(ncol=4, fontsize=8, frameon=False)
    fig.autofmt_xdate()
    fig.savefig(FIGURES / 'failure_case.png', bbox_inches='tight')
    plt.close(fig)

    # Actual-traffic deciles are descriptive test diagnostics, never used to tune models.
    diagnostics, peak_rows = [], []
    for area in areas:
        for kind in COLORS:
            p = predictions[(area, kind)].copy()
            p['bin'] = pd.qcut(p.actual, 10, duplicates='drop')
            p['error'] = p.prediction - p.actual
            grouped = list(p.groupby('bin', observed=True))
            for label, g in grouped:
                diagnostics.append({'area': area, 'model': kind, 'actual_lower': label.left,
                                    'actual_upper': label.right, 'n': len(g),
                                    'bias': g.error.mean(), 'mae': g.error.abs().mean()})
            low, high = grouped[0][1], grouped[-1][1]
            peak_rows.append({'area': area, 'model': kind,
                              'lowest_decile_bias': float(low.error.mean()),
                              'highest_decile_bias': float(high.error.mean()),
                              'highest_decile_mae': float(high.error.abs().mean()),
                              'overall_bias': float(p.error.mean())})
    pd.DataFrame(diagnostics).to_csv(RESULTS / 'error_by_actual_decile.csv', index=False)
    peak = pd.DataFrame(peak_rows)
    peak.to_csv(RESULTS / 'peak_bias_summary.csv', index=False)

    winners = []
    for area in areas:
        a = reference[reference.area == area]
        best = a.loc[a.rmse.idxmin()]
        learned = a[a.model.isin(COLORS)]
        b = learned.loc[learned.rmse.idxmin()]
        persist = float(a.loc[a.model == 'Persistence', 'rmse'].iloc[0])
        winners.append({'area': area, 'best_including_baselines': best['model'], 'best_learned': b['model'],
                        'best_learned_rmse': float(b.rmse), 'persistence_rmse': persist,
                        'improvement_over_persistence_percent': 100 * (1 - float(b.rmse) / persist),
                        'test_n': int(b['n'])})

    order = ['initial', 'daily_history', 'capacity', 'budget']
    tuning = []
    for path in sorted((RESULTS / 'runs').glob('tune_*/summary.json'),
                       key=lambda p: next((i for i, n in enumerate(order)
                                           if p.parent.name.startswith('tune_' + n + '_')), 99)):
        for r in json.loads(path.read_text())['runs']:
            tuning.append({'experiment': r['experiment'], 'model': r['model'], 'area': r['area'],
                           'seed': r['seed'], 'val_rmse': r['validation']['rmse'],
                           'val_mae': r['validation']['mae'], 'train_seconds': r['training_seconds'],
                           'epochs_run': r['epochs_run'], 'best_epoch': r['best_epoch'],
                           'config': json.dumps(r['config']), 'rationale': r['rationale']})
    pd.DataFrame(tuning).to_csv(RESULTS / 'tuning_log.csv', index=False)

    # Learning curves for the reference fit on the highest-ranked area.
    fig, axs = plt.subplots(1, 2, figsize=(9, 3))
    for ax, kind in zip(axs, ['LSTM', 'CausalCNN'], strict=True):
        history = json.loads((RESULTS / 'runs' / f'final_{name}_area{areas[0]}_seed{REFERENCE_SEED}'
                              / f'{kind}_history.json').read_text())
        ax.plot(np.arange(1, len(history['loss']) + 1), history['loss'], label='Training')
        ax.plot(np.arange(1, len(history['val_loss']) + 1), history['val_loss'], label='Validation')
        ax.set(title=f'{kind}: area {areas[0]}, seed {REFERENCE_SEED}', xlabel='Epoch',
               ylabel='MSE of standardized correction')
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / 'learning_curves.png', bbox_inches='tight')
    plt.close(fig)

    neural = times[times.model != 'RidgeAR']
    summary = {
        'winners': winners, 'failure': failure, 'seed_count': len(SEEDS), 'reference_seed': REFERENCE_SEED,
        'ranking_agreement': {'areas': len(ranking), 'agree': int(ranking.agree.sum()),
                              'disagree': int((~ranking.agree).sum())},
        'epoch_budget_audit': {'neural_fits': int(len(neural)),
                               'stopped_early': int(neural.stopped_early.sum()),
                               'hit_budget': int((~neural.stopped_early.astype(bool)).sum()),
                               'max_epochs_run': int(neural.epochs.max()),
                               'budget': int(neural.epoch_budget.max())},
        'peak_underprediction': {
            'combinations': int(len(peak)),
            'negative_highest_decile_bias': int((peak.highest_decile_bias < 0).sum()),
            'positive_lowest_decile_bias': int((peak.lowest_decile_bias > 0).sum()),
            'worst_highest_decile_bias': float(peak.highest_decile_bias.min()),
        },
        'timing_by_model': times.groupby('model')[TIMING_COLUMNS].mean().reset_index().to_dict('records'),
        'parameters_by_model': times.groupby('model').parameters.max().astype(int).to_dict(),
    }
    (RESULTS / 'study_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
