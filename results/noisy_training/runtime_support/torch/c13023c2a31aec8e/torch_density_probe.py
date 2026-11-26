"""Isolated double-precision noisy VQC prototype; not a training backend.

The operation schedule is exported from the actual Qiskit circuit by the probe.
This module deliberately imports no project scientific code or Qiskit, allowing
an existing PyTorch environment to be used as a separate validation worker.
No optimizer checkpoint is opened or converted here.
"""

from pathlib import Path
from time import perf_counter
import argparse
import hashlib
import json
import platform

import numpy as np
import torch


class TorchDensityProbe:
    def __init__(self, operations, n_parameters=72, *, device="cpu", logit_scale=1.0,
                 one_qubit_strength=.01, two_qubit_strength=.02, readout_flip=.01):
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("Choose an explicit CPU or CUDA device.")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable; no CPU fallback was used.")
        self.operations = operations
        self.n_parameters, self.logit_scale = n_parameters, logit_scale
        self.one, self.two, self.readout = one_qubit_strength, two_qubit_strength, readout_flip
        if min(self.one, self.two, self.readout) < 0 or max(self.one, self.two, self.readout) > 1:
            raise ValueError("This prototype supports nonnegative channel strengths at most one.")
        for op in operations:
            if op["gate"] not in ("ry", "rz", "cx"):
                raise ValueError("Unsupported gate in the exported circuit schedule.")
        self.real, self.complex = torch.float64, torch.complex128
        basis = torch.arange(64, device=self.device)
        self.sign = [(1 - 2 * ((basis >> q) & 1)).to(self.real) for q in range(6)]
        self.flip = [basis ^ (1 << q) for q in range(6)]
        self.same = [(z[:, None] == z[None, :]).to(self.real) * .5 for z in self.sign]
        self.cx = {(q, r): basis ^ (((basis >> q) & 1) << r)
                   for q in range(6) for r in range(6) if q != r}
        self.readout_signs = torch.stack(self.sign[3:6], dim=1)

    def _inputs(self, features, weights, *, gradient=False):
        x, w = np.asarray(features, dtype=float), np.asarray(weights, dtype=float)
        single = x.ndim == 2 and w.ndim == 1
        if single:
            x, w = x[None], w[None]
        if (x.ndim != 3 or x.shape[-1] != 4 or w.ndim != 2
                or w.shape != (len(x), self.n_parameters) or not x.shape[1]
                or not np.isfinite(x).all() or not np.isfinite(w).all()):
            raise ValueError("Expected aligned (models,batch,4) and (models,parameters) finite arrays.")
        return (torch.as_tensor(x, dtype=self.real, device=self.device),
                torch.tensor(w, dtype=self.real, device=self.device, requires_grad=gradient), single)

    def _replace_with_mixed_subsystem(self, rho, qubits):
        # Trace the selected subsystem and tensor its maximally mixed state.
        # For two qubits this is I_4/4 tensor Tr_pair(rho), not two independent
        # strength-p errors. Mixing that replacement once gives Aer's JOINT
        # two-qubit depolarizing channel E(rho)=(1-p)rho+p I_4/4 tensor Tr_pair(rho).
        for q in qubits:
            flipped = rho.index_select(1, self.flip[q]).index_select(2, self.flip[q])
            rho = (rho + flipped) * self.same[q]
        return rho

    def _depolarize(self, rho, qubits, strength):
        if strength == 0:
            return rho
        return (1 - strength) * rho + strength * self._replace_with_mixed_subsystem(rho, qubits)

    def _evolve(self, x, weights):
        models, batch = x.shape[:2]
        rho = torch.zeros((models * batch, 64, 64), dtype=self.complex, device=self.device)
        rho[:, 0, 0] = 1.0
        for op in self.operations:
            qubits, gate = op["qubits"], op["gate"]
            q = qubits[0]
            if gate == "cx":
                perm = self.cx[tuple(qubits)]
                rho = rho.index_select(1, perm).index_select(2, perm)
                rho = self._depolarize(rho, qubits, self.two)
                continue
            angle = op["angle"]
            if angle["kind"] == "weight":
                values = weights[:, angle["index"]].repeat_interleave(batch)
            elif angle["kind"] == "input":
                values = x[:, :, angle["indices"][0]].reshape(-1)
                for index in angle["indices"][1:]:
                    values = values * x[:, :, index].reshape(-1)
            else:
                raise ValueError("Unsupported rotation expression.")
            if gate == "ry":
                c = torch.cos(values / 2)[:, None, None]
                s = torch.sin(values / 2)[:, None, None]
                rho = c * rho - s * self.sign[q][None, :, None] * rho.index_select(1, self.flip[q])
                rho = c * rho - s * self.sign[q][None, None, :] * rho.index_select(2, self.flip[q])
            else:
                phase = torch.exp(-.5j * values[:, None] * self.sign[q][None, :])
                rho = phase[:, :, None] * rho * phase.conj()[:, None, :]
            rho = self._depolarize(rho, qubits, self.one)
        diagonal = rho.diagonal(dim1=-2, dim2=-1).real
        z = (diagonal @ self.readout_signs) * (1 - 2 * self.readout)
        return z.reshape(models, batch, 3), rho.reshape(models, batch, 64, 64)

    def _cross_entropy(self, x, weights, targets):
        target = np.asarray(targets, dtype=float)
        if target.ndim == 2 and len(x) == 1:
            target = target[None]
        if (target.shape != (*x.shape[:2], 3) or not np.isfinite(target).all()
                or (target < 0).any() or not np.allclose(target.sum(axis=-1), 1, atol=1e-12, rtol=0)):
            raise ValueError("Targets must be aligned probability distributions.")
        target = torch.as_tensor(target, dtype=self.real, device=self.device)
        z, _ = self._evolve(x, weights)
        return -(target * torch.log_softmax(self.logit_scale * z, dim=-1)).sum(dim=-1).mean(dim=-1)

    def predict(self, features, weights, *, include_density=False):
        x, w, single = self._inputs(features, weights)
        with torch.no_grad():
            z, rho = self._evolve(x, w)
            result = {"expectations": z.cpu().numpy(),
                      "probabilities": torch.softmax(self.logit_scale * z, dim=-1).cpu().numpy()}
            if include_density:
                result["density"] = rho.cpu().numpy()
        return {key: value[0] if single else value for key, value in result.items()}

    def loss_and_gradient(self, features, targets, weights):
        x, w, single = self._inputs(features, weights, gradient=True)
        losses = self._cross_entropy(x, w, targets)
        # SUM model losses: averaging here would incorrectly divide each
        # independent optimizer's gradient by the number of concurrent models.
        gradient, = torch.autograd.grad(losses.sum(), w)
        loss, grad = losses.detach().cpu().numpy(), gradient.detach().cpu().numpy()
        return (float(loss[0]), grad[0]) if single else (loss, grad)

    def unlearning_loss_and_gradient(self, features, targets, weights, *, anchor_features,
                                    anchor_targets, original_weights, alpha=1., regularization=.01):
        x, w, single = self._inputs(features, weights, gradient=True)
        anchor = np.asarray(anchor_features, dtype=float)
        if single:
            anchor = anchor[None]
        if anchor.ndim != 3 or anchor.shape[0] != len(x) or anchor.shape[-1] != 4:
            raise ValueError("Anchor features must align with the independent models.")
        original = np.asarray(original_weights, dtype=float)
        if single:
            original = original[None]
        if original.shape != tuple(w.shape) or alpha < 0 or regularization < 0:
            raise ValueError("Original weights or loss coefficients differ.")
        a = torch.as_tensor(anchor, dtype=self.real, device=self.device)
        origin = torch.as_tensor(original, dtype=self.real, device=self.device)
        losses = self._cross_entropy(x, w, targets)
        if alpha:
            losses = losses + alpha * self._cross_entropy(a, w, anchor_targets)
        losses = losses + regularization * ((w - origin)**2).sum(dim=-1)
        gradient, = torch.autograd.grad(losses.sum(), w)
        loss, grad = losses.detach().cpu().numpy(), gradient.detach().cpu().numpy()
        return (float(loss[0]), grad[0]) if single else (loss, grad)


def synchronize(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def worker(manifest_file, output, *, device="cpu", threads=2):
    """Run serialized synthetic cases and timed calls, never real model fits."""
    torch.set_num_threads(threads)
    manifest_file, output = Path(manifest_file).resolve(), Path(output).resolve()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("synthetic_only") is not True:
        raise ValueError("This isolated probe accepts an explicitly synthetic manifest only.")
    output.mkdir(parents=True, exist_ok=True)
    report = {"prototype": True, "research_results": False, "device": device,
              "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
              "python_version": platform.python_version(), "numpy_version": np.__version__,
              "platform": platform.platform(), "cpu_threads": threads, "dtype": "complex128/float64",
              "gpu": torch.cuda.get_device_name() if device.startswith("cuda") else None,
              "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "cases": []}
    for case in manifest["cases"]:
        print(f"Torch {device}: {case['name']} ...", flush=True)
        inputs = np.load(manifest_file.parent / case["file"], allow_pickle=False)
        arrays = {key: inputs[key] for key in inputs.files}
        model = TorchDensityProbe(manifest["operations"], manifest["n_parameters"], device=device,
                                  logit_scale=case.get("logit_scale", 1.), **case["noise"])
        def call():
            if case.get("objective", "supervised") == "unlearning":
                return model.unlearning_loss_and_gradient(arrays["features"], arrays["targets"], arrays["weights"],
                    anchor_features=arrays["anchor_features"], anchor_targets=arrays["anchor_targets"],
                    original_weights=arrays["original_weights"], alpha=case.get("alpha", 1.),
                    regularization=case.get("regularization", .01))
            return model.loss_and_gradient(arrays["features"], arrays["targets"], arrays["weights"])
        loss, gradient = call()  # Warmup excluded from the timed measurements.
        prediction = model.predict(arrays["features"], arrays["weights"], include_density=case.get("check_density", False))
        if case.get("trajectory_steps", 0):
            # The production NumPy Adam implementation is the independent
            # reference in the Aer process. Use the same explicit minibatches,
            # bias corrections, epsilon and cosine schedule in this worker.
            weights = arrays["weights"].copy()
            first, second = np.zeros_like(weights), np.zeros_like(weights)
            states = [(weights.copy(), first.copy(), second.copy())]
            steps = case["trajectory_steps"]
            for step, indices in enumerate(arrays["batch_indices"], 1):
                if case.get("objective") == "unlearning":
                    anchors = arrays["anchor_indices"][step - 1]
                    _, grad = model.unlearning_loss_and_gradient(arrays["features"][indices], arrays["targets"][indices], weights,
                        anchor_features=arrays["anchor_features"][anchors], anchor_targets=arrays["anchor_targets"][anchors],
                        original_weights=arrays["original_weights"], alpha=case.get("alpha", 1.),
                        regularization=case.get("regularization", .01))
                else:
                    _, grad = model.loss_and_gradient(arrays["features"][indices], arrays["targets"][indices], weights)
                first = .9 * first + .1 * grad
                second = .999 * second + .001 * grad**2
                rate = .001 + .5 * (.02 - .001) * (1 + np.cos(np.pi * (step - 1) / max(1, steps - 1)))
                weights -= rate * (first / (1 - .9**step)) / (np.sqrt(second / (1 - .999**step)) + 1e-8)
                states.append((weights.copy(), first.copy(), second.copy()))
            for index, key in enumerate(("weights", "first_moment", "second_moment")):
                prediction["trajectory_" + key] = np.asarray([state[index] for state in states])
        np.savez_compressed(output / (case["name"] + ".npz"), loss=loss, gradient=gradient, **prediction)
        seconds = []
        if case.get("repeats", 0):
            if device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()
            for _ in range(case["repeats"]):
                synchronize(device)
                start = perf_counter()
                call()  # Includes input transfer, backward and returned CPU arrays.
                synchronize(device)
                seconds.append(perf_counter() - start)
        record = {"name": case["name"], "objective": case.get("objective", "supervised"),
                  "models": arrays["weights"].shape[0] if arrays["weights"].ndim == 2 else 1,
                  "batch": arrays["features"].shape[-2], "seconds": seconds,
                  "median_seconds": float(np.median(seconds)) if seconds else None}
        if device.startswith("cuda"):
            record.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved())
        report["cases"].append(record)
        print(f"Torch {device}: {case['name']} finished; median={record['median_seconds']}", flush=True)
        (output / "worker_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("output")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    worker(args.manifest, args.output, device=args.device, threads=args.threads)
