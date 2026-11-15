"""Fixed distribution-guided targets and stratified retained anchor references."""

from dataclasses import dataclass

import numpy as np

from src.config import DEFAULT_SEED, UnlearningConfig
from src.models.vqc import validate_targets


def make_forget_target(calibration_probabilities, forget_class: int = 2,
                       beta: float = 1.0, method: str = "guided") -> np.ndarray:
    if forget_class not in (0, 1, 2) or beta <= 0:
        raise ValueError("Choose a valid forget class and beta > 0.")
    if method == "uniform":
        target = np.ones(3)
    elif method == "guided":
        probabilities = validate_targets(calibration_probabilities, len(calibration_probabilities))
        target = probabilities.mean(axis=0)**beta
    else:
        raise ValueError("method must be guided or uniform.")
    target[forget_class] = 0.0
    if target.sum() <= 0:
        raise ValueError("No retained-class probability mass is available for the target.")
    return target / target.sum()


@dataclass(frozen=True)
class UnlearningReferences:
    forget_indices: np.ndarray
    anchor_indices: np.ndarray
    calibration_indices: np.ndarray
    anchor_probabilities: np.ndarray
    calibration_probabilities: np.ndarray
    original_weights: np.ndarray


def prepare_references(model, features, labels, forgotten, original_weights,
                       config: UnlearningConfig, seed: int = DEFAULT_SEED) -> UnlearningReferences:
    """Use training records only; predictions are cached and never updated."""
    features, labels = np.asarray(features), np.asarray(labels)
    forgotten = np.asarray(forgotten, dtype=bool)
    if forgotten.shape != (len(features),) or labels.shape != forgotten.shape:
        raise ValueError("The forget mask, labels, and training features must align.")
    forget_indices, retained_indices = np.flatnonzero(forgotten), np.flatnonzero(~forgotten)
    if not len(forget_indices) or not len(retained_indices):
        raise ValueError("Unlearning requires nonempty forget and retain training sets.")
    random = np.random.default_rng(seed)
    calibration_count = min(config.calibration_size or len(forget_indices), len(forget_indices))
    calibration = random.choice(forget_indices, calibration_count, replace=False)
    anchors = []
    for label in np.unique(labels[retained_indices]):
        candidates = retained_indices[labels[retained_indices] == label]
        count = max(1, int(np.ceil(config.anchor_fraction * len(candidates))))
        anchors.extend(random.choice(candidates, count, replace=False))
    anchors = np.sort(np.asarray(anchors, dtype=int))
    return UnlearningReferences(
        forget_indices, anchors, np.sort(calibration),
        model.predict_proba(features[anchors], original_weights).copy(),
        model.predict_proba(features[np.sort(calibration)], original_weights).copy(),
        np.asarray(original_weights, float).copy(),
    )
