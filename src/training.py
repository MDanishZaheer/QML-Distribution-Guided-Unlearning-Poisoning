"""Readable NumPy Adam training with cosine learning rates and saved best weights."""

from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import numpy as np
import pandas as pd

from src.config import DEFAULT_SEED, TrainingConfig
from src.models.vqc import VQC, one_hot


@dataclass
class TrainingResult:
    weights: np.ndarray
    last_weights: np.ndarray
    history: pd.DataFrame
    best_step: int
    selection_metric: str
    best_score: float


def cosine_learning_rate(step: int, config: TrainingConfig) -> float:
    progress = (step - 1) / max(1, config.steps - 1)
    return config.minimum_learning_rate + 0.5 * (
        config.learning_rate - config.minimum_learning_rate
    ) * (1 + np.cos(np.pi * progress))


def optimize(initial_weights, objective: Callable, evaluate: Callable,
             config: TrainingConfig, selection_metric: str,
             callback: Callable | None = None) -> TrainingResult:
    """objective(weights) returns a mini-batch loss/gradient; evaluate uses no test data.

    One step is one mini-batch update, NOT one epoch. The best checkpoint includes
    the initial weights. Callbacks receive a dictionary at each evaluation point.
    """
    weights = np.asarray(initial_weights, dtype=float).copy()
    first_moment = np.zeros_like(weights)
    second_moment = np.zeros_like(weights)
    best_weights, best_step, best_score = weights.copy(), 0, np.inf
    history, stale_checks = [], 0
    started = perf_counter()

    def record(step, learning_rate, gradient_norm):
        nonlocal best_weights, best_step, best_score, stale_checks
        values = dict(evaluate(weights))
        score = float(values[selection_metric])
        if not np.isfinite(score):
            raise FloatingPointError("Non-finite checkpoint selection loss.")
        if score < best_score:
            best_weights, best_step, best_score = weights.copy(), step, score
            stale_checks = 0
        else:
            stale_checks += 1
        row = {"step": step, "learning_rate": learning_rate,
               "gradient_norm": gradient_norm, "elapsed_seconds": perf_counter() - started,
               **values}
        history.append(row)
        if callback is not None:
            callback(dict(row))

    record(0, 0.0, 0.0)
    for step in range(1, config.steps + 1):
        loss, gradient = objective(weights)
        gradient = np.asarray(gradient, dtype=float)
        if gradient.shape != weights.shape or not np.isfinite(gradient).all() or not np.isfinite(loss):
            raise FloatingPointError("Invalid mini-batch loss or gradient.")
        norm = float(np.linalg.norm(gradient))
        if config.gradient_clip is not None and norm > config.gradient_clip:
            gradient *= config.gradient_clip / norm
        first_moment = 0.9 * first_moment + 0.1 * gradient
        second_moment = 0.999 * second_moment + 0.001 * gradient**2
        corrected_first = first_moment / (1 - 0.9**step)
        corrected_second = second_moment / (1 - 0.999**step)
        learning_rate = cosine_learning_rate(step, config)
        weights -= learning_rate * corrected_first / (np.sqrt(corrected_second) + 1e-8)
        if step % config.evaluate_every == 0 or step == config.steps:
            record(step, learning_rate, norm)
            if config.patience is not None and stale_checks >= config.patience:
                break
    return TrainingResult(best_weights, weights.copy(), pd.DataFrame(history),
                          best_step, selection_metric, best_score)


def fit_vqc(model: VQC, train_features, train_labels, validation_features,
            validation_labels, config: TrainingConfig | None = None, seed: int = DEFAULT_SEED,
            initial_weights=None, callback=None) -> TrainingResult:
    """Supervised training; select checkpoints using validation cross-entropy only."""
    config = config or TrainingConfig()
    features = np.asarray(train_features, float)
    targets = one_hot(np.asarray(train_labels))
    validation_targets = one_hot(np.asarray(validation_labels))
    if len(features) != len(targets) or not len(features) or not len(validation_targets):
        raise ValueError("Nonempty aligned training and validation arrays are required.")
    random = np.random.default_rng(seed)
    if initial_weights is None:
        initial_weights = np.random.default_rng(seed).normal(
            0.0, config.initialization_std, model.n_parameters)

    def objective(weights):
        indices = random.choice(len(features), min(config.batch_size, len(features)), replace=False)
        return model.loss_and_gradient(features[indices], targets[indices], weights)

    def evaluate(weights):
        return {"train_loss": model.loss(features, targets, weights),
                "validation_loss": model.loss(validation_features, validation_targets, weights)}

    return optimize(initial_weights, objective, evaluate, config, "validation_loss", callback)
