"""Fixed experimental protocol: grid, split boundaries, seeds and the model registry.

Every module that needs a split date, a seed list or a model name imports it from
here, so the protocol is stated once and cannot drift between the training code,
the summary tables and the report generator.

`verify_submission.py` deliberately does *not* import the boundaries from this
module. It re-derives them from date strings so that a wrong constant here would
still be caught by an independent check rather than agreed with.
"""
from __future__ import annotations

from evaluation import milan_midnight_ms

#: Milliseconds between consecutive observations (the publisher's 10-minute grid).
STEP_MS: int = 600_000

#: Intervals in one day, and the common history every candidate model must have
#: available before a target is eligible for training or scoring.
INTERVALS_PER_DAY: int = 144
MAX_HISTORY: int = INTERVALS_PER_DAY

# Chronological, half-open boundaries in epoch milliseconds, anchored on Europe/Rome
# local midnight rather than UTC. Training is everything before TRAIN_END.
DATA_START: int = milan_midnight_ms('2013-11-01')
TRAIN_END: int = milan_midnight_ms('2013-12-09')
TEST_START: int = milan_midnight_ms('2013-12-16')
TEST_END: int = milan_midnight_ms('2013-12-23')
DATA_END: int = milan_midnight_ms('2014-01-01')

#: Optimization seeds for the final study. Seed 42 is the reference run; the other
#: two measure sensitivity to initialization and shuffling, not forecast uncertainty.
SEEDS: tuple[int, ...] = (42, 43, 44)
REFERENCE_SEED: int = 42

#: Model registry. The order is the order used in every table and figure legend.
MODELS: tuple[str, ...] = ('RidgeAR', 'LSTM', 'CausalCNN')

#: Plot colors, keyed by model name so a table and its figure cannot disagree.
COLORS: dict[str, str] = {'RidgeAR': '#14645a', 'LSTM': '#b05f28', 'CausalCNN': '#53679a'}

#: Default minibatch size when a configuration does not state one. Recorded here so
#: that runs made before `batch` became a tunable field keep their original behavior.
DEFAULT_BATCH: int = 128
