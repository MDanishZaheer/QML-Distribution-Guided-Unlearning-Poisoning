"""One persistent CUDA owner for independent noisy models (JSON over stdio).

Only this process imports torch. NumPy optimizers and durable checkpoints stay
in the orchestration process. The validated complex128 channel implementation
is shared with the synthetic probe, rather than reimplemented here.
"""

import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import traceback

import numpy as np
import torch

from torch_density_probe import TorchDensityProbe


def provenance(device):
    cuda = device.startswith("cuda")
    if cuda and not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is unavailable. Select a GPU runtime; CPU fallback is disabled.")
    return {"engine": "torch_exact_density_autograd_v1", "torch": torch.__version__,
            "cuda": torch.version.cuda, "numpy": np.__version__, "python": platform.python_version(),
            "device": device, "gpu": torch.cuda.get_device_name() if cuda else None,
            "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory if cuda else 0,
            "dtype": "complex128/float64", "deterministic_algorithms": True,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "source": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                       for name in ("torch_noisy_worker.py", "torch_density_probe.py", "torch_noisy_service.py",
                                    "torch_noisy_workflow.py", "cpu_folds.py")}}


def main():
    models, splits = {}, 0
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            kind = request["kind"]
            if kind == "hello":
                device = request["device"]
                result = provenance(device)
            elif kind == "close":
                return
            else:
                key = json.dumps(request["model"], sort_keys=True)
                if key not in models:
                    models[key] = TorchDensityProbe(**request["model"])
                model = models[key]
                x = np.asarray(request["features"], dtype=float)
                w = np.asarray(request["weights"], dtype=float)
                y = np.asarray(request.get("targets", []), dtype=float)
                if device.startswith("cuda"):
                    torch.cuda.reset_peak_memory_stats()
                successful_models, successful_calls = 0, 0

                def evaluate(a, b, c):
                    # Catch outside the tensor operation so the failed graph's
                    # traceback is released before retrying smaller model groups.
                    oom = False
                    nonlocal successful_models, successful_calls
                    try:
                        if kind == "gradient":
                            loss, gradient = model.loss_and_gradient(a, c, b)
                            successful_models = max(successful_models, len(a))
                            successful_calls += 1
                            return {"loss": np.atleast_1d(loss), "gradient": gradient}
                        result = model.predict(a[:, None, :], b)["expectations"][:, 0]
                        successful_calls += 1
                        return {"expectations": result}
                    except torch.cuda.OutOfMemoryError:
                        oom = True
                    if oom:
                        nonlocal splits
                        if len(a) == 1:
                            raise RuntimeError("A single-model request exceeds CUDA memory; checkpoint state is unchanged.")
                        splits += 1
                        gc.collect()
                        torch.cuda.empty_cache()
                        middle = len(a) // 2
                        left = evaluate(a[:middle], b[:middle], c[:middle])
                        right = evaluate(a[middle:], b[middle:], c[middle:])
                        return {k: np.concatenate((left[k], right[k]), axis=0) for k in left}

                result = {k: v.tolist() for k, v in evaluate(x, w, y).items()}
                result["physical_calls"] = successful_calls
                result["memory"] = {"oom_splits": splits, "max_successful_gradient_models": successful_models}
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                    result["memory"].update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved())
            print(json.dumps({"ok": True, "result": result}, allow_nan=False), flush=True)
        except Exception as error:
            print(json.dumps({"ok": False, "error": str(error), "traceback": traceback.format_exc()}), flush=True)


if __name__ == "__main__":
    main()
