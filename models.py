"""Model definitions for the three compared forecasters.

Every model predicts a *correction to persistence* rather than the level itself.
The output layer is initialized to zero, so an untrained network starts out
reproducing the persistence baseline exactly and has to earn any deviation from
it. That makes "did the model learn anything" a question the training curve can
answer, and it keeps the three architectures directly comparable.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from protocol import DEFAULT_BATCH


def tensorflow() -> Any:
    """Import TensorFlow with deterministic, bounded-thread settings applied once.

    Thread limits are set before the first op runs; a RuntimeError means the
    runtime is already initialized, in which case the existing settings stand.
    """
    import tensorflow as tf
    try:
        tf.config.threading.set_intra_op_parallelism_threads(4)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except RuntimeError:
        pass
    tf.config.experimental.enable_op_determinism()
    return tf


def receptive_field(dilations: list[int], kernel: int = 3) -> int:
    """Steps of history the stacked causal convolutions can actually reach."""
    return 1 + (kernel - 1) * sum(dilations)


def neural_model(kind: str, config: dict[str, Any]) -> Any:
    """Build and compile the LSTM or causal CNN described by `config`.

    Args:
        kind: Either ``'LSTM'`` or ``'CausalCNN'``.
        config: Requires ``lookback``, ``width`` and ``learning_rate``; the CNN
            also requires ``dilations``.

    Raises:
        ValueError: If `kind` is not a known neural architecture.
    """
    tf = tensorflow()
    layers = tf.keras.layers
    inputs = layers.Input((config['lookback'], 1), name='normalized_history')
    if kind == 'LSTM':
        z = layers.LSTM(config['width'], name='gated_memory')(inputs)
    elif kind == 'CausalCNN':
        z = inputs
        for dilation in config['dilations']:
            z = layers.Conv1D(config['width'], 3, padding='causal', dilation_rate=dilation,
                              activation='relu', name=f'causal_d{dilation}')(z)
        # Causal padding keeps the full sequence; only the last position has seen
        # the whole window, so everything earlier is discarded before the head.
        z = layers.Cropping1D((config['lookback'] - 1, 0))(z)
        z = layers.Flatten()(z)
    else:
        raise ValueError(kind)
    output = layers.Dense(1, kernel_initializer='zeros', bias_initializer='zeros',
                          name='persistence_correction')(z)
    model = tf.keras.Model(inputs, output, name=kind)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config['learning_rate'], clipnorm=1.0),
                  loss='mse')
    return model


def make_dataset(tf: Any, split: dict[str, np.ndarray], seed: int | None = None,
                 batch: int = DEFAULT_BATCH) -> Any:
    """Wrap one split as a deterministic tf.data pipeline.

    Shuffling is applied only when a seed is given, which is how training is
    distinguished from validation. Shuffling training *windows* is not leakage:
    each window already contains only observations preceding its own target, so
    the order they are visited in cannot move information backwards in time.
    """
    ds = tf.data.Dataset.from_tensor_slices((split['x'], split['delta'][:, None]))
    if seed is not None:
        ds = ds.shuffle(len(split['y']), seed=seed, reshuffle_each_iteration=True)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    options.experimental_deterministic = True
    return ds.batch(batch).with_options(options).prefetch(1)
