"""Train one configuration on one area and seed, and record everything it did.

Training data ends 2013-12-09. Tuning rounds score on 2013-12-09 to 12-15 and
never touch the test week. The final phase additionally scores 2013-12-16 to
12-22, which is only ever read after the configuration has been frozen.

Each run writes its weights, per-split predictions, training history, scaler and
a `summary.json` into its own directory, so every number in the report can be
recomputed from disk without retraining.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

# Set before TensorFlow is imported anywhere, or the settings are ignored.
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('OMP_NUM_THREADS', '4')

import platform
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from evaluation import score
from features import to_forecast, windows
from models import make_dataset, neural_model, receptive_field, tensorflow
from protocol import DEFAULT_BATCH

ROOT = Path(__file__).resolve().parent

Predictor = Callable[[np.ndarray], np.ndarray]


def fit_one(kind: str, config: dict[str, Any], data: dict[str, dict[str, np.ndarray]],
            seed: int) -> tuple[Any, Predictor, float, int, dict[str, list[float]], Any]:
    """Fit one model and return it together with everything worth recording.

    Returns:
        ``(model, predict, training_seconds, parameter_count, history, stopping)``.
        ``history`` is empty and ``stopping`` is None for RidgeAR, which has no
        epochs. ``predict`` maps a batch of normalized histories to normalized
        corrections, whatever the underlying library is.

    Raises:
        ValueError: If neural training produced a nonfinite loss.
    """
    train, validation = data['train'], data['validation']
    history: dict[str, list[float]] = {}
    stopping = None
    if kind == 'RidgeAR':
        model = Ridge(alpha=config['alpha'], solver='svd')
        start = time.perf_counter()
        model.fit(train['x'][:, :, 0], train['delta'])
        training_seconds = time.perf_counter() - start
        predict = lambda x: model.predict(x[:, :, 0])
        parameters = int(model.coef_.size + 1)
    else:
        tf = tensorflow()
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)
        batch = config.get('batch', DEFAULT_BATCH)
        start = time.perf_counter()
        model = neural_model(kind, config)
        stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=config.get('patience', 4),
                                                    min_delta=1e-5, restore_best_weights=True)
        callbacks = [stopping, tf.keras.callbacks.TerminateOnNaN()]
        result = model.fit(make_dataset(tf, train, seed, batch),
                           validation_data=make_dataset(tf, validation, batch=batch),
                           epochs=config['epochs'], callbacks=callbacks, verbose=0, shuffle=False)
        training_seconds = time.perf_counter() - start
        history = {k: [float(v) for v in vals] for k, vals in result.history.items()}
        if not all(np.isfinite(history['loss'])):
            raise ValueError('Nonfinite training loss')

        @tf.function(reduce_retracing=True)
        def forward(x):
            return model(x, training=False)

        predict = lambda x: forward(tf.convert_to_tensor(x)).numpy().reshape(-1)
        parameters = int(model.count_params())
    return model, predict, training_seconds, parameters, history, stopping


def measure_inference(predict: Predictor, split: dict[str, np.ndarray]) -> dict[str, Any]:
    """Time warm batch and single-example inference, after discarding a warm-up call.

    The first call compiles and traces, so it is run and thrown away before any
    measurement. These are laptop timings, not a production service benchmark.
    """
    predict(split['x'])
    times = []
    for _ in range(5):
        start = time.perf_counter()
        predict(split['x'])
        times.append(time.perf_counter() - start)
    point = []
    predict(split['x'][:1])
    for i in range(min(30, len(split['y']))):
        start = time.perf_counter()
        predict(split['x'][i:i + 1])
        point.append(time.perf_counter() - start)
    return {
        'batch_n': len(split['y']),
        'warm_batch_seconds_median': float(np.median(times)),
        'warm_batch_seconds_runs': times,
        'single_seconds_median': float(np.median(point)),
        'single_seconds_p95': float(np.quantile(point, .95)),
    }


def run(config_path: str | Path, area: int, seed: int, phase: str) -> Path:
    """Fit every model in one configuration file and write the run directory.

    Args:
        config_path: A file from `configs/`, carrying an `id`, a `rationale` and
            one entry per model.
        area: Square ID to fit.
        seed: Optimization seed.
        phase: ``'tune'`` scores validation only; ``'final'`` also scores the test
            week. Tuning rounds must never read the test split.

    Returns:
        The directory the run was written to.

    Raises:
        FileExistsError: If a completed run with this identity already exists.
            Experiment records are evidence, so they are never overwritten.
    """
    configs = json.loads(Path(config_path).read_text())
    name = configs['id']
    out = ROOT / 'results/runs' / f'{phase}_{name}_area{area}_seed{seed}'
    if (out / 'summary.json').exists():
        raise FileExistsError(f'Run already exists: {out}; use another ID rather than overwriting evidence')
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(ROOT / 'results/selected_series.csv', index_col='timestamp_ms')
    summaries = []
    for kind, config in configs['models'].items():
        data, scaler = windows(frame, area, config['lookback'])
        model, predict, seconds, parameters, history, stopping = fit_one(kind, config, data, seed)

        # EarlyStopping only accepts an epoch as "best" when it beats the previous
        # best by more than min_delta, so argmin(val_loss) is not always the epoch
        # whose weights were restored. Record what the callback actually restored.
        restored_epoch = int(stopping.best_epoch) + 1 if stopping is not None else None
        lowest_val_epoch = int(np.argmin(history['val_loss'])) + 1 if history else None

        model_summary: dict[str, Any] = {
            'model': kind, 'area': area, 'seed': seed, 'phase': phase, 'experiment': name,
            'config': config, 'rationale': configs['rationale'], 'scaler': scaler,
            'train_n': len(data['train']['y']), 'training_seconds': seconds,
            'parameter_count': parameters, 'epochs_run': len(history.get('loss', [])),
            'epoch_budget': config.get('epochs'),
            'stopped_early': (len(history['loss']) < config['epochs']) if history else None,
            'best_epoch': restored_epoch,
            'lowest_val_loss_epoch': lowest_val_epoch,
        }
        if kind == 'CausalCNN':
            model_summary['receptive_field'] = receptive_field(config['dilations'])
        for split in (['validation'] if phase == 'tune' else ['validation', 'test']):
            d = data[split]
            prediction = to_forecast(d['last'], predict(d['x']), scaler)
            model_summary[split] = score(d['y'], prediction)
            table = pd.DataFrame({'timestamp_ms': d['timestamps'], 'actual': d['y'],
                                  'prediction': prediction, 'persistence': d['last'],
                                  'seasonal_daily': d['seasonal']})
            table.to_csv(out / f'{kind}_{split}_predictions.csv', index=False)

        model_summary['inference'] = measure_inference(
            predict, data['validation' if phase == 'tune' else 'test'])
        (out / f'{kind}_history.json').write_text(json.dumps(history, indent=2))
        (out / f'{kind}_scaler.json').write_text(json.dumps(scaler, indent=2))
        if kind == 'RidgeAR':
            np.savez(out / f'{kind}.npz', coef=model.coef_, intercept=model.intercept_)
        else:
            model.save(out / f'{kind}.keras')
        summaries.append(model_summary)
        (out / 'partial_summary.json').write_text(json.dumps(summaries, indent=2))
        print(json.dumps({'area': area, 'model': kind, 'seed': seed, 'phase': phase,
                          'val_rmse': model_summary['validation']['rmse'], 'seconds': seconds,
                          'epochs': model_summary['epochs_run']}), flush=True)
    (out / 'summary.json').write_text(json.dumps({
        'config': configs,
        'data_sha256': hashlib.sha256((ROOT / 'results/selected_series.csv').read_bytes()).hexdigest(),
        'platform': platform.platform(), 'machine': platform.machine(), 'runs': summaries}, indent=2))
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--area', type=int, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--phase', choices=['tune', 'final'], default='tune')
    args = parser.parse_args()
    run(args.config, args.area, args.seed, args.phase)
