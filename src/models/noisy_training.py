"""Independent optimizers sharing exact noisy density-matrix circuit batches."""

from dataclasses import asdict, dataclass, replace

from qiskit_aer import AerSimulator

from src.config import DEFAULT_SEED, BackendConfig
from src.models.backends import CircuitExecutor
from src.models.noise import NoiseSetting, build_gate_noise
from src.models.parallel_aer import AerBatchService


@dataclass(frozen=True)
class NoisyBackendIdentity:
    runtime: BackendConfig
    training_noise: NoiseSetting
    expectation_mode: str = "exact density matrix with exact readout confusion"


class NoisyExpectationExecutor(CircuitExecutor):
    def __init__(self, circuit, config, execution, seed, setting):
        self.execution, self.setting = execution, setting
        runtime = replace(config, shots=None, method="density_matrix", device=execution.device,
                          allow_cpu_fallback=False, use_ibm_hardware=False, ibm_backend_name=None,
                          one_qubit_error=0.0, two_qubit_error=0.0, readout_error=0.0,
                          circuit_batch_size=execution.circuit_batch_size, max_memory_mb=execution.memory_mb,
                          max_parallel_threads=execution.threads, optimization_level=0)
        super().__init__(circuit, runtime, seed)

    def _make_aer(self, device):
        return AerSimulator(method="density_matrix", device=device, precision="double",
                            basis_gates=["ry", "rz", "cx"], noise_model=build_gate_noise(self.setting),
                            max_memory_mb=self.execution.memory_mb,
                            max_parallel_threads=self.execution.threads, max_parallel_experiments=0,
                            runtime_parameter_bind_enable=self.execution.runtime_binding,
                            batched_shots_gpu=self.execution.gpu_parameter_batching)

    def run(self, features, weights):
        # For symmetric independent readout flips, each Z mean is multiplied by
        # (1-2r). Applying this inside the executor also scales every shifted
        # expectation, so the inherited VQC chain-rule gradient remains correct.
        return super().run(features, weights) * (1 - 2 * self.setting.readout_error)

    def metadata(self):
        return {**super().metadata(), "training_noise": self.setting.describe(),
                "readout_evaluation": "exact independent symmetric classical confusion"}


class NoisyBatchService(AerBatchService):
    def __init__(self, circuit_config, backend_config, execution, setting=NoiseSetting("combined", 0.01),
                 seed=DEFAULT_SEED):
        self.setting = setting
        super().__init__(circuit_config, backend_config, execution, seed,
                         executor_factory=lambda circuit, config, candidate, stream:
                             NoisyExpectationExecutor(circuit, config, candidate, stream, setting))
        self.backend_config = NoisyBackendIdentity(self.executor.config, setting)

    def metadata(self):
        return {**super().metadata(), "training_noise": asdict(self.setting)}
