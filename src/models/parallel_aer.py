"""Batch independent exact-expectation requests through one Aer instance.

Each optimizer keeps its own weights, RNG and checkpoint. A single service
combines their circuit requests, avoiding one CUDA context per optimizer.
"""

from concurrent.futures import Future
from dataclasses import asdict, dataclass, replace
from queue import Empty, Queue
from threading import Event, Thread
from time import monotonic, process_time
import os
import subprocess

import numpy as np
from qiskit_aer import AerSimulator

from src.config import DEFAULT_SEED
from src.models.backends import CircuitExecutor
from src.models.circuits import build_malik_circuit
from src.models.vqc import VQC


@dataclass(frozen=True)
class AerExecution:
    device: str = "GPU"
    circuit_batch_size: int = 2048
    workers: int = 2
    threads: int = 2
    memory_mb: int = 16384  # Limit, not a request to allocate this much VRAM.
    runtime_binding: bool = False
    gpu_parameter_batching: bool = False
    gather_ms: float = 2.0

    def __post_init__(self):
        if self.device not in ("CPU", "GPU"):
            raise ValueError("Choose an explicit Aer CPU/GPU device.")
        if min(self.circuit_batch_size, self.workers, self.threads, self.memory_mb) < 1:
            raise ValueError("Execution limits must be positive.")
        if self.gather_ms < 0:
            raise ValueError("Gather time must be nonnegative.")
        if self.gpu_parameter_batching and (self.device != "GPU" or not self.runtime_binding):
            raise ValueError("GPU parameter batching requires GPU and runtime binding.")


class _TunedExecutor(CircuitExecutor):
    def __init__(self, circuit, config, execution, seed):
        if config.shots is not None or any(config.noise_rates) or config.use_ibm_hardware:
            raise ValueError("The ablation service requires noiseless exact expectations.")
        self.execution = execution
        runtime = replace(config, device=execution.device, method="statevector",
                          allow_cpu_fallback=False, circuit_batch_size=execution.circuit_batch_size,
                          max_memory_mb=execution.memory_mb, max_parallel_threads=execution.threads)
        super().__init__(circuit, runtime, seed)

    def _make_aer(self, device):
        return AerSimulator(method="statevector", device=device, precision="double",
                            basis_gates=["ry", "rz", "cx"],
                            max_parallel_threads=self.execution.threads,
                            max_parallel_experiments=0, max_memory_mb=self.execution.memory_mb,
                            runtime_parameter_bind_enable=self.execution.runtime_binding,
                            batched_shots_gpu=self.execution.gpu_parameter_batching)


class AerBatchService:
    """One background Aer owner; concurrent callers receive their exact slices."""

    def __init__(self, circuit_config, backend_config, execution, seed=DEFAULT_SEED, *, executor_factory=None):
        self.circuit_config, self.backend_config = circuit_config, backend_config
        self.execution = execution
        factory = executor_factory or _TunedExecutor
        self.executor = factory(build_malik_circuit(circuit_config), backend_config, execution, seed)
        self.queue = Queue(maxsize=max(4, execution.workers * 2))
        self.closed = False
        self.batches, self.requests = 0, 0
        self.thread = Thread(target=self._serve, name="ablation-aer", daemon=True)
        self.thread.start()

    def submit(self, features, weights):
        if self.closed:
            raise RuntimeError("Aer service is closed.")
        future = Future()
        self.queue.put((np.asarray(features, float).copy(), np.asarray(weights, float).copy(), future))
        return future.result()

    def _serve(self):
        stop = False
        while not stop:
            first = self.queue.get()
            if first is None:
                return
            requests = [first]
            rows = len(first[0])
            deadline = monotonic() + self.execution.gather_ms / 1000
            while rows < self.execution.circuit_batch_size:
                remaining = deadline - monotonic()
                if remaining <= 0:
                    break
                try:
                    item = self.queue.get(timeout=remaining)
                except Empty:
                    break
                if item is None:
                    stop = True
                    break
                requests.append(item)
                rows += len(item[0])
            try:
                features = np.concatenate([request[0] for request in requests])
                weights = np.concatenate([request[1] for request in requests])
                result = self.executor.run(features, weights)
                offset = 0
                for request_features, _, future in requests:
                    future.set_result(result[offset:offset + len(request_features)].copy())
                    offset += len(request_features)
                self.batches += 1
                self.requests += len(requests)
            except BaseException as error:
                for _, _, future in requests:
                    future.set_exception(error)

    def model(self, seed=DEFAULT_SEED):
        model = VQC.__new__(VQC)
        model.circuit_config, model.backend_config = self.circuit_config, self.backend_config
        model.circuit = build_malik_circuit(self.circuit_config)
        model.n_parameters = self.circuit_config.n_parameters
        model.executor = _ClientExecutor(self, seed)
        return model

    def metadata(self):
        return {"execution": asdict(self.execution), "merged_batches": self.batches,
                "logical_requests": self.requests,
                "physical_aer_resources": dict(self.executor.stats)}

    def close(self):
        if not self.closed:
            self.closed = True
            self.queue.put(None)
            self.thread.join()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class _ClientExecutor:
    """Per-fit random streams and logical resource counters, compatible with Adam resume."""

    def __init__(self, service, seed):
        self.service = service
        self.random = np.random.default_rng(seed)
        self.stats = {"jobs": 0, "circuit_evaluations": 0, "shots": 0}
        self.job_ids = []
        self.actual_device = service.executor.actual_device

    def reset_seed(self, seed):
        self.random = np.random.default_rng(seed)

    def run(self, features, weights):
        if not len(features):
            return np.empty((0, 3))
        self.random.integers(0, 2**31 - 1)  # Stream is local; exact expectations do not sample shots.
        result = self.service.submit(features, weights)
        self.stats["jobs"] += 1
        self.stats["circuit_evaluations"] += len(features)
        return result

    def metadata(self):
        result = self.service.executor.metadata()
        result["resources"] = dict(self.stats)
        result["resource_scope"] = "per-model logical requests; physical jobs recorded at study level"
        result["ablation_execution"] = asdict(self.service.execution)
        return result


def gpu_sample():
    """nvidia-smi data for the allocated device, without new dependencies."""
    columns = ("utilization.gpu", "memory.used", "memory.total", "memory.free", "power.draw")
    try:
        completed = subprocess.run(["nvidia-smi", "--query-gpu=" + ",".join(columns),
                                    "--format=csv,noheader,nounits"], capture_output=True,
                                   text=True, timeout=5, check=False)
        if completed.returncode or not completed.stdout.strip():
            return {}
        values = completed.stdout.splitlines()[0].split(",")
        return {column: float(value.strip()) for column, value in zip(columns, values)
                if value.strip().replace(".", "", 1).isdigit()}
    except (OSError, subprocess.TimeoutExpired):
        return {}


class GpuMonitor:
    def __init__(self, interval=2.0):
        self.interval, self.rows = interval, []
        self.stop = Event()

    def _sample(self):
        start = monotonic()
        previous_wall, previous_cpu = start, process_time()
        while not self.stop.is_set():
            sample = gpu_sample()
            now, cpu = monotonic(), process_time()
            cpu_percent = 100 * (cpu - previous_cpu) / max(now - previous_wall, 1e-9)
            self.rows.append({"elapsed_seconds": now - start, **sample,
                              "process_cpu_core_percent": cpu_percent,
                              "allocated_cpu_capacity_percent": cpu_percent / (os.cpu_count() or 1)})
            previous_wall, previous_cpu = now, cpu
            self.stop.wait(self.interval)

    def __enter__(self):
        self.thread = Thread(target=self._sample, name="ablation-gpu-monitor", daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()


def require_gpu():
    if "GPU" not in AerSimulator().available_devices():
        raise RuntimeError("runtime Aer does not expose GPU execution. Select a GPU runtime and install "
                           "the compatible GPU-enabled Aer package, then restart the runtime. "
                           "No ablation or supervised training has started.")
