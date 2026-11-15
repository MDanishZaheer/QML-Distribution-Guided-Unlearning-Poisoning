"""Batched Aer execution and an optional, explicitly selected IBM SamplerV2 path."""

from dataclasses import asdict
from dataclasses import replace
from time import perf_counter
import warnings

import numpy as np
from qiskit import ClassicalRegister, transpile
from qiskit.quantum_info import Pauli
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

from src.config import DEFAULT_SEED, BackendConfig
from src.models.circuits import READOUT_QUBITS, circuit_resources, parameter_groups


def build_noise_model(config: BackendConfig) -> NoiseModel:
    """Independent depolarizing gate errors and symmetric readout flips."""
    noise = NoiseModel()
    if config.one_qubit_error:
        error = depolarizing_error(config.one_qubit_error, 1)
        noise.add_all_qubit_quantum_error(error, ["ry", "rz"])
    if config.two_qubit_error:
        noise.add_all_qubit_quantum_error(depolarizing_error(config.two_qubit_error, 2), ["cx"])
    if config.readout_error:
        p = config.readout_error
        noise.add_all_qubit_readout_error(ReadoutError([[1 - p, p], [p, 1 - p]]))
    return noise


def counts_to_expectations(counts: dict) -> np.ndarray:
    """Classical bit 0 measures q3, bit 1 q4, bit 2 q5 (little-endian)."""
    total = sum(counts.values())
    if total <= 0:
        raise ValueError("Empty measurement counts.")
    expectations = np.zeros(3)
    for bitstring, count in counts.items():
        text = str(bitstring).replace(" ", "")
        value = int(text, 16) if text.startswith("0x") else int(text, 2)
        expectations += count * np.array([1 - 2 * ((value >> bit) & 1) for bit in range(3)])
    return expectations / total


class CircuitExecutor:
    """Compile once, bind batches, and expose honest execution/resource metadata."""

    def __init__(self, circuit, config: BackendConfig, seed: int = DEFAULT_SEED):
        self.config = config
        self.seed = seed
        self.random = np.random.default_rng(seed)
        self.input_parameters, self.weight_parameters = parameter_groups(circuit)
        self.original_resources = circuit_resources(circuit)
        self.stats = {"jobs": 0, "circuit_evaluations": 0, "shots": 0}
        self.job_ids = []
        self.fallback_reason = None
        self.available_devices = []
        self.program = circuit.copy()
        if config.shots is None:
            for index, qubit in enumerate(READOUT_QUBITS):
                self.program.save_expectation_value(Pauli("Z"), [qubit], label=f"z{index}")
        else:
            self.program.add_register(ClassicalRegister(3, "readout"))
            self.program.measure(READOUT_QUBITS, self.program.cregs[-1])

        if config.use_ibm_hardware:
            self._initialize_ibm()
        else:
            self._initialize_aer()

    def _initialize_aer(self):
        self.available_devices = list(AerSimulator().available_devices())
        requested = self.config.device
        use_gpu = requested in ("auto", "GPU") and "GPU" in self.available_devices
        if requested == "GPU" and not use_gpu:
            if not self.config.allow_cpu_fallback:
                raise RuntimeError("The installed Aer build does not expose a GPU device.")
            self.fallback_reason = "GPU requested but unavailable in this Aer build."
            warnings.warn(self.fallback_reason + " Using CPU.", stacklevel=3)
        self.actual_device = "GPU" if use_gpu else "CPU"
        self.actual_method = self.config.method
        if self.actual_method == "automatic":
            self.actual_method = "density_matrix" if any(self.config.noise_rates) else "statevector"
        self.backend = self._make_aer(self.actual_device)
        self.compiled = transpile(
            self.program, self.backend,
            optimization_level=self.config.optimization_level, seed_transpiler=self.seed,
        )

    def _make_aer(self, device):
        return AerSimulator(
            method=self.actual_method, device=device,
            basis_gates=["ry", "rz", "cx"],
            noise_model=build_noise_model(self.config),
            max_memory_mb=self.config.max_memory_mb,
            max_parallel_threads=self.config.max_parallel_threads,
            max_parallel_experiments=1,
        )

    def _initialize_ibm(self):
        # Optional import: local runs require neither this package nor an account.
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
        except ImportError as error:
            raise ImportError("Install qiskit-ibm-runtime only when enabling IBM execution.") from error
        service = QiskitRuntimeService()  # Uses the user's configured account, never stores tokens here.
        self.backend = service.backend(self.config.ibm_backend_name)
        self.sampler = SamplerV2(mode=self.backend)
        self.actual_device = self.backend.name
        self.actual_method = "IBM hardware"
        self.compiled = transpile(self.program, self.backend,
                                  optimization_level=self.config.optimization_level,
                                  seed_transpiler=self.seed)

    def reset_seed(self, seed: int):
        """Pair simulated sampling streams across comparison methods."""
        self.random = np.random.default_rng(seed)

    def _parameter_columns(self, features, weights):
        columns = {p: features[:, i].tolist() for i, p in enumerate(self.input_parameters)}
        columns.update({p: weights[:, i].tolist() for i, p in enumerate(self.weight_parameters)})
        return {p: columns[p] for p in self.compiled.parameters}

    def _run_aer_batch(self, columns, seed):
        result = self.backend.run(
            self.compiled, parameter_binds=[columns],
            shots=self.config.shots if self.config.shots is not None else 1,
            seed_simulator=seed,
        ).result()
        if not result.success:
            raise RuntimeError(f"Aer execution failed: {result.status}")
        if self.config.shots is None:
            return np.array([[result.data(i)[f"z{k}"] for k in range(3)]
                             for i in range(len(result.results))], dtype=float)
        return np.array([counts_to_expectations(result.get_counts(i))
                         for i in range(len(result.results))])

    def _run_ibm_batch(self, columns):
        values = np.column_stack([columns[p] for p in self.compiled.parameters])
        job = self.sampler.run([(self.compiled, values)], shots=self.config.shots)
        self.job_ids.append(job.job_id())
        samples = job.result()[0].data.readout
        return np.array([counts_to_expectations(samples.get_counts(i)) for i in range(len(values))])

    def run(self, features: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Evaluate aligned rows: one four-feature input and one weight vector each."""
        features, weights = np.asarray(features, float), np.asarray(weights, float)
        if features.ndim != 2 or features.shape[1] != 4:
            raise ValueError("features must have shape (n, 4).")
        if weights.shape != (len(features), len(self.weight_parameters)):
            raise ValueError("weights must have shape (n, number_of_weights).")
        if not np.isfinite(features).all() or not np.isfinite(weights).all():
            raise ValueError("Circuit inputs must be finite.")
        if len(features) == 0:
            return np.empty((0, 3))
        output = []
        for start in range(0, len(features), self.config.circuit_batch_size):
            end = min(start + self.config.circuit_batch_size, len(features))
            columns = self._parameter_columns(features[start:end], weights[start:end])
            if self.config.use_ibm_hardware:
                values = self._run_ibm_batch(columns)
            else:
                seed = int(self.random.integers(0, 2**31 - 1))
                try:
                    values = self._run_aer_batch(columns, seed)
                except Exception as error:
                    if self.actual_device != "GPU" or not self.config.allow_cpu_fallback:
                        raise
                    self.fallback_reason = f"GPU execution failed: {type(error).__name__}: {error}"
                    warnings.warn(self.fallback_reason + "; retrying on CPU.", stacklevel=2)
                    self.actual_device = "CPU"
                    self.backend = self._make_aer("CPU")
                    values = self._run_aer_batch(columns, seed)
            output.append(values)
            self.stats["jobs"] += 1
            self.stats["circuit_evaluations"] += end - start
            self.stats["shots"] += (end - start) * (self.config.shots or 0)
        return np.vstack(output)

    def metadata(self) -> dict:
        return {"requested": asdict(self.config), "actual_device": self.actual_device,
                "actual_method": self.actual_method, "available_aer_devices": self.available_devices,
                "expectation_mode": "exact" if self.config.shots is None else "sampled",
                "fallback_reason": self.fallback_reason, "original_circuit": self.original_resources,
                "compiled_circuit": circuit_resources(self.compiled),
                "resources": dict(self.stats), "ibm_job_ids": list(self.job_ids)}


def benchmark_aer_devices(circuit, features, weights, config: BackendConfig | None = None,
                          repeats: int = 3, seed: int = DEFAULT_SEED) -> list[dict]:
    """Compare available local devices on an actual workload; never submit IBM jobs.

    Exclude compilation and one warm-up from timings. Use the result to choose
    device='CPU' or 'GPU' explicitly before an expensive study. Small circuits
    need measured timings; GPU availability alone is not a speedup measurement.
    """
    config = config or BackendConfig()
    if config.use_ibm_hardware or repeats < 1:
        raise ValueError("Device benchmarks require local Aer and repeats >= 1.")
    features = np.asarray(features, float)
    weights = np.asarray(weights, float)
    if weights.ndim == 1:
        weights = np.broadcast_to(weights, (len(features), len(weights)))
    rows = []
    for device in AerSimulator().available_devices():
        if device not in ("CPU", "GPU"):
            continue
        executor = CircuitExecutor(circuit, replace(config, device=device, allow_cpu_fallback=False), seed)
        executor.run(features, weights)
        times = []
        for _ in range(repeats):
            started = perf_counter()
            executor.run(features, weights)
            times.append(perf_counter() - started)
        rows.append({"device": device, "method": executor.actual_method,
                     "median_seconds": float(np.median(times)), "repeats": repeats,
                     "circuit_evaluations_per_repeat": len(features), "shots": config.shots})
    return rows
