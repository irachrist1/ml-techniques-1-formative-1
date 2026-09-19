"""Check complete study artifacts, leakage boundaries, metrics and saved-model replay."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from predict_saved import predict
from evaluation import score, milan_midnight_ms

ROOT = Path(__file__).resolve().parent
SEEDS = [42, 43, 44]
MODELS = ['RidgeAR', 'LSTM', 'CausalCNN']


def main():
    eda = json.loads((ROOT / 'results/eda_summary.json').read_text())
    config = json.loads((ROOT / 'configs/final.json').read_text())
    frame = pd.read_csv(ROOT / 'results/selected_series.csv', index_col='timestamp_ms')
    assert len(frame) == 8784 and np.all(np.diff(frame.index) == 600000)
    assert set(frame.columns) == set(map(str, eda['areas']))
    digest = hashlib.sha256((ROOT / 'results/selected_series.csv').read_bytes()).hexdigest()

    checks = 0
    replay = []
    capped = []
    for area in eda['areas']:
        reference = None
        for seed in SEEDS:
            folder = ROOT / 'results/runs' / f"final_{config['id']}_area{area}_seed{seed}"
            summary = json.loads((folder / 'summary.json').read_text())
            assert summary['data_sha256'] == digest
            assert summary['config'] == config
            for entry in summary['runs']:
                kind = entry['model']
                p = pd.read_csv(folder / f'{kind}_test_predictions.csv')
                assert p.timestamp_ms.is_unique
                assert (p.timestamp_ms >= milan_midnight_ms('2013-12-16')).all()
                assert (p.timestamp_ms < milan_midnight_ms('2013-12-23')).all()
                assert np.isfinite(p[['actual', 'prediction']]).all().all()
                np.testing.assert_allclose(p.actual, frame.loc[p.timestamp_ms, str(area)], rtol=1e-12)

                # Persistence and the daily-seasonal baseline must be the observations
                # one step and one day before each target, not anything re-derived.
                index = {t: i for i, t in enumerate(frame.index.to_numpy())}
                for j in (0, len(p) // 2, len(p) - 1):
                    i = index[int(p.timestamp_ms.iloc[j])]
                    np.testing.assert_allclose(p.persistence.iloc[j], frame[str(area)].iloc[i - 1], rtol=1e-12)
                    np.testing.assert_allclose(p.seasonal_daily.iloc[j], frame[str(area)].iloc[i - 144], rtol=1e-12)

                if reference is None:
                    reference = p.timestamp_ms.to_numpy()
                else:
                    np.testing.assert_array_equal(reference, p.timestamp_ms)

                m = score(p.actual, p.prediction)
                np.testing.assert_allclose(
                    [m[k] for k in ['mae', 'rmse', 'mape_percent_nonzero']],
                    [entry['test'][k] for k in ['mae', 'rmse', 'mape_percent_nonzero']], rtol=1e-10)
                assert m['n'] == entry['test']['n']
                assert m['mape_excluded_zero_targets'] == entry['test']['mape_excluded_zero_targets']
                assert entry['scaler']['fit_end_exclusive'] == milan_midnight_ms('2013-12-09')

                # Early stopping, not the epoch cap, should be ending neural training.
                if entry.get('epoch_budget') and entry['epochs_run']:
                    if not entry.get('stopped_early'):
                        capped.append(f"{area}/{kind}/seed{seed}")

                # Replay the first, middle and last target from every saved model, so
                # boundary positions are covered and not only a comfortable interior one.
                for position in (0, len(p) // 2, len(p) - 1):
                    timestamp = pd.to_datetime(p.timestamp_ms.iloc[position], unit='ms', utc=True).isoformat()
                    value = predict(folder, kind, timestamp)
                    stored = float(p.prediction.iloc[position])
                    np.testing.assert_allclose(value, stored, rtol=3e-5, atol=1e-4)
                    replay.append({'area': area, 'model': kind, 'seed': seed, 'position': position,
                                   'absolute_replay_difference': abs(value - stored)})
                checks += 1

    assert checks == len(eda['areas']) * len(SEEDS) * len(MODELS)
    assert len(list((ROOT / 'results/figures').glob('forecast_*.png'))) >= len(eda['areas']) * len(MODELS)
    (ROOT / 'results/verification.json').write_text(json.dumps({
        'complete_final_models': checks,
        'replayed_forecasts': len(replay),
        'largest_replay_difference': max(r['absolute_replay_difference'] for r in replay),
        'neural_fits_ending_at_the_epoch_cap': capped,
        'saved_model_replays': replay,
        'selected_series_sha256': digest,
        'status': 'passed',
    }, indent=2))
    print(f'Passed: {checks} saved models, {len(replay)} replayed forecasts, matching metrics and common timestamps')
    if capped:
        print(f'Note: {len(capped)} neural fits still ended at the epoch cap: {capped}')


if __name__ == '__main__':
    main()
