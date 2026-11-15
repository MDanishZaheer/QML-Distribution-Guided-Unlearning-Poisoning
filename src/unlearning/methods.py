"""Malik-style class unlearning: guided/uniform targets, anchors, and weight penalty."""

import numpy as np

from src.config import DEFAULT_SEED, UnlearningConfig
from src.training import TrainingResult, optimize
from src.unlearning.targets import UnlearningReferences, make_forget_target


def unlearning_loss_and_gradient(model, forget_features, anchor_features,
                                 target, anchor_probabilities, weights,
                                 original_weights, alpha: float, regularization: float):
    """Positive minimization equivalent of the paper's maximization objective."""
    forget_targets = np.broadcast_to(target, (len(forget_features), 3))
    loss, gradient = model.loss_and_gradient(forget_features, forget_targets, weights)
    if alpha:
        anchor_loss, anchor_gradient = model.loss_and_gradient(anchor_features, anchor_probabilities, weights)
        loss += alpha * anchor_loss
        gradient += alpha * anchor_gradient
    difference = np.asarray(weights) - original_weights
    loss += regularization * np.dot(difference, difference)
    gradient += 2 * regularization * difference
    return float(loss), gradient


def unlearn(model, training_features, references: UnlearningReferences,
            config: UnlearningConfig | None = None, method: str = "guided",
            seed: int = DEFAULT_SEED, callback=None) -> tuple[TrainingResult, np.ndarray]:
    """Start each method at the SAME original weights and use fixed teacher targets.

    Select by the full training unlearning objective, never by test metrics.
    """
    config = config or UnlearningConfig()
    features = np.asarray(training_features)
    target = make_forget_target(references.calibration_probabilities,
                                config.forget_class, config.beta, method)
    forget_features = features[references.forget_indices]
    anchor_features = features[references.anchor_indices]
    random = np.random.default_rng(seed)
    model.executor.reset_seed(seed)

    def objective(weights):
        forget_batch = random.choice(len(forget_features),
                                     min(config.training.batch_size, len(forget_features)), replace=False)
        anchor_batch = random.choice(len(anchor_features),
                                     min(config.training.batch_size, len(anchor_features)), replace=False)
        return unlearning_loss_and_gradient(
            model, forget_features[forget_batch], anchor_features[anchor_batch], target,
            references.anchor_probabilities[anchor_batch], weights, references.original_weights,
            config.alpha, config.regularization,
        )

    def evaluate(weights):
        forget_loss = model.loss(forget_features, np.broadcast_to(target, (len(forget_features), 3)), weights)
        anchor_loss = model.loss(anchor_features, references.anchor_probabilities, weights)
        distance = float(np.sum((weights - references.original_weights)**2))
        return {"forget_loss": forget_loss, "anchor_loss": anchor_loss,
                "parameter_distance_squared": distance,
                "objective": forget_loss + config.alpha * anchor_loss + config.regularization * distance}

    result = optimize(references.original_weights, objective, evaluate, config.training, "objective", callback)
    return result, target
