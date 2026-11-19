"""Independent numerical checks of Malik Eqs. 6-8; synthetic data only.

The NumPy simulator below does not call the project's circuit builder, backend,
softmax, loss or differentiation code. It follows the six-qubit gate sequence
in the supplied paper, providing a separate oracle for the production code.
"""

import unittest

import numpy as np

from src.config import DEFAULT_SEED, BackendConfig, CircuitConfig
from src.models.vqc import VQC
from src.unlearning.methods import unlearning_loss_and_gradient
from src.unlearning.targets import make_forget_target


AUDIT_DIAGNOSTICS = {}


def paper_expectations(features, weights):
    """Direct state amplitudes, little-endian qubits; one weight row per input."""
    features = np.asarray(features, float)
    weights = np.broadcast_to(weights, (len(features), 72))
    indices = np.arange(64)
    state = np.zeros((len(features), 64), complex)
    state[:, 0] = 1

    def ry(qubit, angles):
        zero = indices[(indices & (1 << qubit)) == 0]
        one = zero | (1 << qubit)
        cosine, sine = np.cos(angles / 2)[:, None], np.sin(angles / 2)[:, None]
        first, second = state[:, zero].copy(), state[:, one].copy()
        state[:, zero] = cosine * first - sine * second
        state[:, one] = sine * first + cosine * second

    def rz(qubit, angles):
        signs = 1 - 2 * ((indices >> qubit) & 1)
        state[:] *= np.exp(-0.5j * angles[:, None] * signs[None, :])

    def cx(control, target):
        permutation = indices ^ (((indices >> control) & 1) << target)
        state[:] = state[:, permutation]

    parameter = 0
    for _ in range(2):
        for qubit in range(4):
            ry(qubit, features[:, qubit])
        for control, target in ((0, 1), (1, 2), (2, 3)):
            cx(control, target)
            rz(target, features[:, control] * features[:, target])
            cx(control, target)
        for _ in range(3):
            for qubit in range(6):
                ry(qubit, weights[:, parameter])
                rz(qubit, weights[:, parameter + 1])
                parameter += 2
            for control in range(6):
                cx(control, (control + 1) % 6)
    signs = np.stack([1 - 2 * ((indices >> q) & 1) for q in (3, 4, 5)], axis=1)
    return np.abs(state)**2 @ signs


def paper_probabilities(features, weights):
    logits = paper_expectations(features, weights)
    exponentials = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def paper_loss_parts(probabilities, forget, anchors, target, teacher, weights,
                     original, alpha=1.0, regularization=0.01):
    forget_loss = -np.mean(np.sum(target * np.log(probabilities[forget]), axis=1))
    anchor_loss = -np.mean(np.sum(teacher * np.log(probabilities[anchors]), axis=1))
    distance = float(np.sum((weights - original)**2))
    return {"forget_loss": float(forget_loss), "anchor_loss": float(anchor_loss),
            "parameter_distance_squared": distance,
            "objective": float(forget_loss + alpha * anchor_loss + regularization * distance)}


class GuidedPaperAuditTests(unittest.TestCase):
    def test_full_paper_circuit_matches_independent_amplitude_calculation(self):
        random = np.random.default_rng(DEFAULT_SEED)
        features, weights = random.uniform(0, np.pi, (7, 4)), random.normal(0, 0.8, 72)
        model = VQC(CircuitConfig(), BackendConfig(shots=None, method="statevector",
                                                  device="CPU", max_parallel_threads=1))
        expected = paper_expectations(features, weights)
        actual = model.expectations(features, weights)
        error = float(np.max(np.abs(expected - actual)))
        AUDIT_DIAGNOSTICS["max_independent_expectation_error"] = error
        np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=0)

    def test_target_averages_before_exponentiation_and_normalization(self):
        probabilities = np.array([[0.05, 0.35, 0.60], [0.70, 0.20, 0.10], [0.20, 0.10, 0.70]])
        for forgotten in range(3):
            for beta in (0.25, 0.5, 1.0, 2.0):
                means = probabilities.sum(axis=0) / 3
                expected = np.array([0 if k == forgotten else means[k]**beta for k in range(3)])
                expected /= expected.sum()
                np.testing.assert_allclose(make_forget_target(probabilities, forgotten, beta), expected,
                                           atol=1e-15, rtol=0)
                self.assertEqual(make_forget_target(probabilities, forgotten, beta)[forgotten], 0)

    def test_all_72_unlearning_derivatives_match_independent_finite_differences(self):
        random = np.random.default_rng(DEFAULT_SEED)
        features = random.uniform(0, np.pi, (7, 4))
        original = random.normal(0, 0.3, 72)
        weights = original + random.normal(0, 0.15, 72)
        forget, anchors = np.array([0, 1, 2]), np.array([3, 4, 5, 6])
        original_probabilities = paper_probabilities(features, original)
        teacher = original_probabilities[anchors]
        target = original_probabilities[forget].mean(axis=0)
        target[2] = 0
        target /= target.sum()
        model = VQC(CircuitConfig(), BackendConfig(shots=None, method="statevector",
                                                  device="CPU", max_parallel_threads=1))
        for alpha, regularization in ((1.0, 0.01), (0.8, 0.2)):
            loss, gradient = unlearning_loss_and_gradient(model, features[forget], features[anchors],
                target, teacher, weights, original, alpha, regularization)
            def objective(theta):
                probabilities = paper_probabilities(features, theta)
                return paper_loss_parts(probabilities, forget, anchors, target, teacher,
                                        theta, original, alpha, regularization)["objective"]
            numerical = []
            for coordinate in range(72):
                plus, minus = weights.copy(), weights.copy()
                plus[coordinate] += 1e-5
                minus[coordinate] -= 1e-5
                numerical.append((objective(plus) - objective(minus)) / 2e-5)
            error = float(np.max(np.abs(gradient - numerical)))
            AUDIT_DIAGNOSTICS[f"max_full_gradient_error_alpha{alpha}_lambda{regularization}"] = error
            self.assertAlmostEqual(loss, objective(weights), places=12)
            np.testing.assert_allclose(gradient, numerical, atol=2e-8, rtol=0)


if __name__ == "__main__":
    unittest.main()
