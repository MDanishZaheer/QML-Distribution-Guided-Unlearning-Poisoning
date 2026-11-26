"""Batch independent fold optimizers through one persistent PyTorch GPU owner.

Execution identity is distinct from Aer. This changes the derivative engine,
not the circuit, channel, loss, minibatches, NumPy Adam, or checkpoint rules.
"""

from collections import defaultdict
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
from threading import Lock, Thread
from time import monotonic
from unittest.mock import patch

import numpy as np

from src.config import DEFAULT_SEED
from src.models.circuits import build_malik_circuit
from src.models.noise import NoiseSetting
from src.models.noisy_training import NoisyBackendIdentity
from src.models.parallel_aer import _ClientExecutor
from src.models.vqc import VQC, validate_targets
from tools.benchmark_torch_noise import schedule


ENGINE = "torch_exact_density_autograd_v1"


@dataclass(frozen=True)
class TorchExecution:
    device: str = "GPU"
    workers: int = 20
    max_models: int = 20
    prediction_rows: int = 128
    gather_ms: float = 5.0
    runtime_fingerprint: str = ""
    engine: str = ENGINE
    oom_policy: str = "split independent models, preserve logical update"

    def __post_init__(self):
        if self.engine != ENGINE or self.device not in ("CPU", "GPU"):
            raise ValueError("Unsupported Torch execution identity.")
        if min(self.workers, self.max_models, self.prediction_rows) < 1 or self.gather_ms < 0:
            raise ValueError("Execution capacities must be positive.")
        if not self.runtime_fingerprint:
            raise ValueError("Check and pin the actual Torch runtime before fitting.")


class TorchProcess:
    def __init__(self, python=None, *, device="GPU"):
        self.device = "cuda:0" if device == "GPU" else "cpu"
        script = Path(__file__).with_name("torch_noisy_worker.py")
        environment = dict(os.environ, PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1",
                           CUBLAS_WORKSPACE_CONFIG=":4096:8")
        self.process = subprocess.Popen([str(python or sys.executable), "-u", "-B", str(script)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None, text=True, encoding="utf-8", env=environment)
        self.lock = Lock()
        try:
            self.runtime = self.call({"kind": "hello", "device": self.device})
        except BaseException:
            self.close()
            raise
        self.fingerprint = hashlib.sha256(json.dumps(self.runtime, sort_keys=True).encode()).hexdigest()

    def call(self, request):
        with self.lock:
            if self.process.poll() is not None:
                raise RuntimeError("Torch CUDA process stopped; committed optimizer checkpoints are preserved.")
            self.process.stdin.write(json.dumps(request, allow_nan=False) + "\n")
            self.process.stdin.flush()
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError("Torch CUDA process exited without a result. No optimizer update was committed.")
            reply = json.loads(line)
            if not reply["ok"]:
                raise RuntimeError(reply["error"] + "\n" + reply["traceback"])
            return reply["result"]

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.write('{"kind":"close"}\n')
                self.process.stdin.flush()
                self.process.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                self.process.terminate()
                self.process.wait(timeout=10)
        self.process.stdin.close()
        self.process.stdout.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


@dataclass(frozen=True)
class TorchBackendIdentity(NoisyBackendIdentity):
    engine: str = ENGINE
    runtime_fingerprint: str = ""


class _ExecutorInfo:
    def __init__(self, config, execution, setting, process):
        self.config = replace(config, shots=None, method="density_matrix", device=execution.device,
            allow_cpu_fallback=False, use_ibm_hardware=False, ibm_backend_name=None,
            one_qubit_error=0., two_qubit_error=0., readout_error=0., optimization_level=0,
            circuit_batch_size=execution.prediction_rows, max_parallel_threads=2)
        self.actual_device = "torch:" + process.device
        self.stats = {"jobs": 0, "circuit_evaluations": 0, "shots": 0}
        self.execution, self.setting, self.process = execution, setting, process

    def metadata(self):
        return {"actual_device": self.actual_device, "backend_config": asdict(self.config),
                "engine": ENGINE, "torch_runtime": self.process.runtime,
                "training_noise": self.setting.describe(), "resources": dict(self.stats),
                "gradient": "exact channel automatic differentiation, double precision"}


class TorchVQC(VQC):
    def loss_and_gradient(self, features, targets, weights):
        features, weights = self._validate_inputs(features, weights)
        targets = validate_targets(targets, len(features))
        result = self.executor.service.submit("gradient", features, weights, targets)
        self.executor.random.integers(0, 2**31 - 1)
        self.executor.stats["jobs"] += 1
        self.executor.stats["circuit_evaluations"] += len(features)
        return float(result[0]), np.asarray(result[1], dtype=float)


class TorchBatchService:
    def __init__(self, circuit_config, backend_config, execution, setting=NoiseSetting("combined", .01),
                 seed=DEFAULT_SEED, *, process=None):
        self.execution, self.setting, self.circuit_config = execution, setting, circuit_config
        if setting.gate_channel not in ("none", "depolarizing"):
            raise ValueError("The validated Torch backend supports depolarizing and readout channels only.")
        self.process = process or TorchProcess(device=execution.device)
        self.own_process = process is None
        if self.process.fingerprint != execution.runtime_fingerprint:
            if self.own_process:
                self.process.close()
            raise ValueError("Torch/CUDA/device/helper versions changed. Do not resume this optimizer with a different runtime.")
        self.executor = _ExecutorInfo(backend_config, execution, setting, self.process)
        self.backend_config = TorchBackendIdentity(self.executor.config, setting,
            runtime_fingerprint=self.process.fingerprint)
        self.model_spec = {"operations": schedule(circuit_config), "n_parameters": circuit_config.n_parameters,
            "device": self.process.device, "logit_scale": circuit_config.logit_scale,
            "one_qubit_strength": setting.level if setting.gate_channel != "none" else 0.,
            "two_qubit_strength": 2 * setting.level if setting.gate_channel != "none" else 0.,
            "readout_flip": setting.readout_error}
        self.queue = Queue(maxsize=max(4, execution.workers * 2))
        self.closed, self.batches, self.requests = False, 0, 0
        self.batch_sizes, self.gradient_batch_sizes, self.memory = [], [], {}
        self.thread = Thread(target=self._serve, name="torch-fold-batching", daemon=True)
        self.thread.start()

    def submit(self, kind, features=None, weights=None, targets=None):
        # _ClientExecutor calls submit(features, aligned_weights).
        if not isinstance(kind, str):
            kind, features, weights = "predict", kind, features
        if self.closed or not self.thread.is_alive():
            raise RuntimeError("Torch batch service is closed.")
        future = Future()
        self.queue.put((kind, np.asarray(features, float).copy(), np.asarray(weights, float).copy(),
                        None if targets is None else np.asarray(targets, float).copy(), future))
        return future.result()

    def _perform(self, group):
        kind = group[0][0]
        if kind == "gradient":
            features = np.stack([r[1] for r in group])
            weights = np.stack([r[2] for r in group])
            targets = np.stack([r[3] for r in group])
            chunks = [(features, weights, targets)]
        else:
            features = np.concatenate([r[1] for r in group])
            weights = np.concatenate([r[2] for r in group])
            chunks = [(features[i:i+self.execution.prediction_rows], weights[i:i+self.execution.prediction_rows], None)
                      for i in range(0, len(features), self.execution.prediction_rows)]
        results = []
        for x, w, y in chunks:
            request = {"kind": kind, "model": self.model_spec, "features": x.tolist(), "weights": w.tolist()}
            if y is not None:
                request["targets"] = y.tolist()
            result = self.process.call(request)
            self.memory = {key: max(self.memory.get(key, 0), value) for key, value in result["memory"].items()}
            self.executor.stats["jobs"] += result["physical_calls"]
            self.executor.stats["circuit_evaluations"] += int(np.prod(x.shape[:-1]))
            results.append(result)
        self.batches += 1
        self.requests += len(group)
        self.batch_sizes.append(len(group))
        if kind == "gradient":
            self.gradient_batch_sizes.append(results[0]["memory"]["max_successful_gradient_models"])
            loss, gradient = results[0]["loss"], results[0]["gradient"]
            for i, row in enumerate(group):
                row[4].set_result((loss[i], np.asarray(gradient[i])))
        else:
            predictions = np.concatenate([np.asarray(r["expectations"]) for r in results])
            offset = 0
            for row in group:
                row[4].set_result(predictions[offset:offset + len(row[1])].copy())
                offset += len(row[1])

    def _serve(self):
        stop = False
        while not stop:
            first = self.queue.get()
            if first is None:
                return
            requests = [first]
            deadline = monotonic() + self.execution.gather_ms / 1000
            while len(requests) < self.execution.max_models:
                try:
                    remaining = deadline - monotonic()
                    if remaining <= 0:
                        break
                    row = self.queue.get(timeout=remaining)
                except Empty:
                    break
                if row is None:
                    stop = True
                    break
                requests.append(row)
            groups = defaultdict(list)
            for row in requests:
                groups[(row[0], len(row[1]) if row[0] == "gradient" else 0)].append(row)
            for group in groups.values():
                try:
                    self._perform(group)
                except BaseException as error:
                    for row in group:
                        if not row[4].done():
                            row[4].set_exception(error)

    def model(self, seed=DEFAULT_SEED):
        model = TorchVQC.__new__(TorchVQC)
        model.circuit_config, model.backend_config = self.circuit_config, self.backend_config
        model.circuit = build_malik_circuit(self.circuit_config)
        model.n_parameters = self.circuit_config.n_parameters
        model.executor = _ClientExecutor(self, seed)
        return model

    def metadata(self):
        return {"engine": ENGINE, "execution": asdict(self.execution), "torch_runtime": self.process.runtime,
                "training_noise": asdict(self.setting), "merged_batches": self.batches,
                "logical_requests": self.requests, "largest_concurrent_model_batch": max(self.batch_sizes, default=0),
                "largest_concurrent_gradient_batch": max(self.gradient_batch_sizes, default=0),
                "physical_torch_resources": dict(self.executor.stats), "cuda_memory": self.memory}

    def close(self):
        if not self.closed:
            self.closed = True
            self.queue.put(None)
            self.thread.join()
            if self.own_process:
                self.process.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


@contextmanager
def torch_training_backend(process):
    """Scope replacement to the separate noisy runner; never alter saved science source."""
    import src.noisy_training as module
    class SharedTorchService(TorchBatchService):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs, process=process)
    with patch.object(module, "NoisyBatchService", SharedTorchService):
        yield
