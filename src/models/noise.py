"""Gate channels and joint readout probabilities for fixed six-qubit models.

Density-matrix evolution integrates gate noise exactly. Readout confusion and
optional shot sampling are applied to the joint distribution of q3/q4/q5.
No model weights, features or training procedures are changed.
"""

from dataclasses import asdict, dataclass

import numpy as np
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, pauli_error
from scipy.special import softmax

from src.config import DEFAULT_SEED, BackendConfig
from src.models.backends import CircuitExecutor
from src.models.circuits import READOUT_QUBITS, build_malik_circuit


FAMILIES = ("depolarizing", "bit_flip", "phase_flip", "readout", "combined")
DEFAULT_LEVELS = (0.0, 0.001, 0.005, 0.01, 0.02)


@dataclass(frozen=True)
class NoiseSetting:
    family: str = "none"
    level: float = 0.0

    def __post_init__(self):
        if self.family not in ("none", *FAMILIES) or not np.isfinite(self.level) or not 0 <= self.level <= 0.5:
            raise ValueError("Choose a registered noise family and a level between 0 and 0.5.")
        if self.family == "none" and self.level != 0:
            raise ValueError("The noise-free control requires level zero.")

    @property
    def gate_channel(self):
        if self.level == 0 or self.family in ("none", "readout"):
            return "none"
        return "depolarizing" if self.family == "combined" else self.family

    @property
    def readout_error(self):
        return self.level if self.family in ("readout", "combined") else 0.0

    def gate_setting(self):
        return NoiseSetting(self.gate_channel, self.level if self.gate_channel != "none" else 0.0)

    def describe(self):
        gate = self.gate_channel
        return {**asdict(self), "gate_channel": gate,
                "one_qubit_strength": self.level if gate != "none" else 0.0,
                "two_qubit_strength": 2 * self.level if gate != "none" else 0.0,
                "readout_flip_probability": self.readout_error,
                "two_qubit_rule": ("joint two-qubit depolarizing channel" if gate == "depolarizing" else
                                     "independent channel on each CX operand" if gate != "none" else "none")}


def noise_settings(levels=DEFAULT_LEVELS, families=FAMILIES):
    levels, families = tuple(levels), tuple(families)
    if (not levels or 0.0 not in levels or len(set(levels)) != len(levels)
            or not families or len(set(families)) != len(families) or not set(families).issubset(FAMILIES)):
        raise ValueError("Include one zero level and unique registered families/levels.")
    for level in levels:
        NoiseSetting("depolarizing", level)
    return [NoiseSetting()] + [NoiseSetting(family, level) for family in families
                              for level in sorted(levels) if level > 0]


def build_gate_noise(setting):
    """RY/RZ errors follow each rotation; CX errors follow each entangling gate.

For depolarizing, p and 2p are Aer channel parameters, not probabilities of
nonidentity Pauli errors. For X/Z channels, they are Pauli-flip probabilities.
The 2p rule follows the project's earlier sweep; it is a declared synthetic
stress test, not calibration from a particular IBM device.
"""
    setting = setting.gate_setting()
    model = NoiseModel()
    if setting.family == "none":
        return model
    p = setting.level
    if setting.family == "depolarizing":
        one, two = depolarizing_error(p, 1), depolarizing_error(2 * p, 2)
    else:
        pauli = "X" if setting.family == "bit_flip" else "Z"
        one = pauli_error([(pauli, p), ("I", 1 - p)])
        operand = pauli_error([(pauli, 2 * p), ("I", 1 - 2 * p)])
        two = operand.tensor(operand)
    model.add_all_qubit_quantum_error(one, ["ry", "rz"])
    model.add_all_qubit_quantum_error(two, ["cx"])
    return model


@dataclass(frozen=True)
class NoiseExecution:
    device: str = "GPU"
    circuit_batch_size: int = 512
    threads: int = 4
    memory_mb: int = 16384

    def __post_init__(self):
        if self.device not in ("CPU", "GPU") or min(self.circuit_batch_size, self.threads, self.memory_mb) < 1:
            raise ValueError("Choose CPU/GPU and positive execution limits.")


class DensityReadoutExecutor(CircuitExecutor):
    """Use the unchanged parameter-binding path with density-matrix probabilities."""

    def __init__(self, circuit_config, setting, execution, seed=DEFAULT_SEED):
        self.setting, self.execution = setting.gate_setting(), execution
        config = BackendConfig(shots=None, method="density_matrix", device=execution.device,
                               allow_cpu_fallback=False, circuit_batch_size=execution.circuit_batch_size,
                               max_parallel_threads=execution.threads, max_memory_mb=execution.memory_mb,
                               optimization_level=0)
        super().__init__(build_malik_circuit(circuit_config), config, seed)

    def _initialize_aer(self):
        self.program.save_probabilities(list(READOUT_QUBITS), label="readout_probabilities")
        super()._initialize_aer()

    def _make_aer(self, device):
        return AerSimulator(method="density_matrix", device=device, precision="double",
                            basis_gates=["ry", "rz", "cx"], noise_model=build_gate_noise(self.setting),
                            max_memory_mb=self.execution.memory_mb,
                            max_parallel_threads=self.execution.threads, max_parallel_experiments=0)

    def _run_aer_batch(self, columns, seed):
        result = self.backend.run(self.compiled, parameter_binds=[columns], shots=1,
                                  seed_simulator=seed).result()
        if not result.success:
            raise RuntimeError(f"Noisy density-matrix evaluation failed: {result.status}")
        return validate_joint([result.data(i)["readout_probabilities"] for i in range(len(result.results))])


def validate_joint(values):
    values = np.asarray(values, float)
    if (values.ndim != 2 or values.shape[1] != 8 or not np.isfinite(values).all()
            or np.any(values < -1e-12) or not np.allclose(values.sum(axis=1), 1, atol=1e-10, rtol=0)):
        raise ValueError("Expected normalized joint probabilities for three readout qubits.")
    values = np.maximum(values, 0)
    return values / values.sum(axis=1, keepdims=True)


def readout_distribution(joint, probability):
    """Independent symmetric classical flips; bit 0 corresponds to physical q3."""
    joint = validate_joint(joint)
    if not 0 <= probability <= 0.5:
        raise ValueError("Readout flip probability must be between zero and 0.5.")
    if probability == 0:
        return joint.copy()
    flip = np.array([[1 - probability, probability], [probability, 1 - probability]])
    transition = np.kron(np.kron(flip, flip), flip)
    return validate_joint(joint @ transition)


def class_probabilities(joint, *, readout_error=0.0, shots=None, seed=DEFAULT_SEED, logit_scale=1.0):
    """Convert exact or shot-sampled joint readouts to the existing softmax head.

Sampling the eight outcomes together preserves correlations between Z3/Z4/Z5.
Finite-shot and exact results share the same density evolution, avoiding a
second circuit simulation. This models measurement shots, not training seeds.
"""
    distribution = readout_distribution(joint, readout_error)
    if shots is not None:
        if not isinstance(shots, int) or isinstance(shots, bool) or shots < 1:
            raise ValueError("shots must be a positive integer or None.")
        random = np.random.default_rng(seed)
        distribution = np.stack([random.multinomial(shots, row) for row in distribution]) / shots
    signs = np.array([[1 - 2 * ((outcome >> bit) & 1) for bit in range(3)] for outcome in range(8)])
    expectations = distribution @ signs
    return softmax(logit_scale * expectations, axis=1), expectations
