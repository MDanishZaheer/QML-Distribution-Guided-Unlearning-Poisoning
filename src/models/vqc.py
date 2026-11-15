"""Three-class VQC, with parameter shifts applied BEFORE classical cross-entropy."""

import numpy as np
from scipy.special import logsumexp, softmax

from src.config import DEFAULT_SEED, BackendConfig, CircuitConfig
from src.models.backends import CircuitExecutor
from src.models.circuits import build_malik_circuit


def one_hot(labels, n_classes: int = 3) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Labels must be a one-dimensional integer array.")
    if np.any(labels < 0) or np.any(labels >= n_classes):
        raise ValueError("Labels fall outside the classifier output space.")
    return np.eye(n_classes)[labels]


def validate_targets(targets, n_samples: int) -> np.ndarray:
    targets = np.asarray(targets, dtype=float)
    if targets.shape != (n_samples, 3) or n_samples == 0:
        raise ValueError("Targets must have shape (n_samples, 3) with at least one sample.")
    if not np.isfinite(targets).all() or (targets < 0).any():
        raise ValueError("Targets must contain finite, nonnegative probabilities.")
    if not np.allclose(targets.sum(axis=1), 1.0):
        raise ValueError("Each target distribution must sum to one.")
    return targets


class VQC:
    def __init__(self, circuit_config: CircuitConfig | None = None,
                 backend_config: BackendConfig | None = None, seed: int = DEFAULT_SEED):
        self.circuit_config = circuit_config or CircuitConfig()
        self.backend_config = backend_config or BackendConfig()
        self.circuit = build_malik_circuit(self.circuit_config)
        self.n_parameters = self.circuit_config.n_parameters
        self.executor = CircuitExecutor(self.circuit, self.backend_config, seed)

    def _validate_inputs(self, features, weights):
        features, weights = np.asarray(features, float), np.asarray(weights, float)
        if features.ndim != 2 or features.shape[1] != 4:
            raise ValueError("features must have shape (n, 4).")
        if weights.shape != (self.n_parameters,):
            raise ValueError(f"Expected {self.n_parameters} weights.")
        if not np.isfinite(features).all() or not np.isfinite(weights).all():
            raise ValueError("Features and weights must be finite.")
        return features, weights

    def expectations(self, features, weights) -> np.ndarray:
        features, weights = self._validate_inputs(features, weights)
        return self.executor.run(features, np.broadcast_to(weights, (len(features), len(weights))))

    def predict_proba(self, features, weights) -> np.ndarray:
        return softmax(self.circuit_config.logit_scale * self.expectations(features, weights), axis=1)

    def predict(self, features, weights) -> np.ndarray:
        return self.predict_proba(features, weights).argmax(axis=1)

    def loss(self, features, targets, weights) -> float:
        targets = validate_targets(targets, len(features))
        logits = self.circuit_config.logit_scale * self.expectations(features, weights)
        return float(-np.mean(np.sum(targets * (logits - logsumexp(logits, axis=1, keepdims=True)), axis=1)))

    def expectations_and_jacobian(self, features, weights):
        """Return z[n,3] and dz/dtheta[n,3,p], with bounded host-side batching.

        Every theta occurs in exactly one RY/RZ gate, so +/- pi/2 shifts apply.
        We deliberately retain the paper's nominal terminal RZ parameters.
        """
        features, weights = self._validate_inputs(features, weights)
        n_parameters = self.n_parameters
        n_variants = 1 + 2 * n_parameters
        samples_per_batch = max(1, 4096 // n_variants)
        values, jacobians = [], []
        shifted = np.tile(weights, (n_variants, 1))
        for index in range(n_parameters):
            shifted[1 + 2 * index, index] += np.pi / 2
            shifted[2 + 2 * index, index] -= np.pi / 2
        for start in range(0, len(features), samples_per_batch):
            batch = features[start:start + samples_per_batch]
            result = self.executor.run(np.tile(batch, (n_variants, 1)),
                                       np.repeat(shifted, len(batch), axis=0))
            result = result.reshape(n_variants, len(batch), 3)
            values.append(result[0])
            derivative = (result[1::2] - result[2::2]) / 2
            jacobians.append(derivative.transpose(1, 2, 0))
        if not len(features):
            return np.empty((0, 3)), np.empty((0, 3, n_parameters))
        return np.concatenate(values), np.concatenate(jacobians)

    def loss_and_gradient(self, features, targets, weights):
        """Chain rule: dCE/dtheta = sum_k (p_k - target_k) * dz_k/dtheta."""
        targets = validate_targets(targets, len(features))
        expectations, jacobian = self.expectations_and_jacobian(features, weights)
        scale = self.circuit_config.logit_scale
        logits = scale * expectations
        probabilities = softmax(logits, axis=1)
        log_probabilities = logits - logsumexp(logits, axis=1, keepdims=True)
        loss = -np.mean(np.sum(targets * log_probabilities, axis=1))
        gradient = scale * np.einsum("nk,nkp->p", probabilities - targets, jacobian) / len(features)
        return float(loss), gradient

    def metadata(self):
        result = self.executor.metadata()
        result["nominal_trainable_parameters"] = self.n_parameters
        result["logit_scale"] = self.circuit_config.logit_scale
        return result
