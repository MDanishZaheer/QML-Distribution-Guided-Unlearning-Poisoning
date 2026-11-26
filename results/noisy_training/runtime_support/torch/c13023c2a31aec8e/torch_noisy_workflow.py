"""Single-GPU, all-fold noisy study, with separate verified recovery storage."""

from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import pandas as pd

from src import artifacts, noisy_training
from src.config import DEFAULT_SEED, CircuitConfig, BackendConfig
from src.models.noise import NoiseSetting
from src.models.noisy_training import NoisyBatchService
from src.models.parallel_aer import AerExecution
from tools.cpu_folds import HANDOFF, cpu_completed_models, _validate_import
from tools.noisy_drive import DriveRecovery, STAGES, _read_verified, publish
from tools.noisy_workflow import locate_index
from tools.parallel_folds import COMMON, _check_protocol, _copy_files, _inputs, _local, _protocol, _verified_json
from tools.torch_noisy_service import TorchExecution, TorchBatchService, ENGINE


def _source_stores(drive, folds, source):
    if source == "none":
        return []
    if source == "auto":
        source = ("parallel_cpu" if (drive / "parallel_cpu/plan.json").exists() else
                  "parallel" if (drive / "parallel/plan.json").exists() else "serial")
    if source == "serial":
        return [drive]
    if source not in ("parallel", "parallel_cpu"):
        raise ValueError("Import source must be auto, serial, parallel, parallel_cpu, or none.")
    plan = _verified_json(drive / source / "plan.json")
    if not plan["ready"] or not set(folds).issubset(plan["folds"]):
        raise ValueError("Previous fold handoff is incomplete; select the correct source explicitly.")
    return [drive / source / f"fold_{f}" for f in folds]


def prepare_torch_handoff(review, drive_root, *, folds=(0, 1, 2, 3, 4),
                          setting=NoiseSetting("combined", .01), source="auto",
                          restart_unfinished=False, emit=print):
    """Read old stores only; import fully verified models, preserve partials there.

    A new engine cannot continue Aer Adam state. Restart permission is explicit
    in the notebook. Repeated setup never resets the new Torch recovery store.
    """
    root, drive = Path(review.root).resolve(), Path(drive_root).resolve()
    target = drive / "torch"
    problems, historical, bindings = _inputs(review, folds)
    base = {"engine": ENGINE, "folds": list(folds), "code_fingerprint": artifacts.source_fingerprint(root),
            "original_source": review.source_fingerprint, "training_noise": asdict(setting), "seed": DEFAULT_SEED,
            "historical_inputs_fingerprint": artifacts.fingerprint({k: m.identity(root) for k, m in historical.items()})}
    plan_file = target / "plan.json"
    if plan_file.exists():
        plan = _verified_json(plan_file)
        _check_protocol(plan, base)
        emit("Existing PyTorch handoff retained; its completed models and optimizer progress were not reset.")
        return plan
    sources = _source_stores(drive, folds, source)
    inventory, unfinished, teachers, provenance = {}, [], {}, []
    recovery = DriveRecovery(root, target, emit=emit)
    if any((target / name).exists() for name in ("recovery/state.zip", "recovery/state.zip.previous")):
        # A setup interrupted after its snapshot can be completed idempotently.
        recovery.restore()
    with TemporaryDirectory(prefix="qml-torch-handoff-") as temporary:
        for i, store in enumerate(sources):
            source_root = Path(temporary) / str(i)
            source_root.mkdir()
            previous = DriveRecovery(source_root, store, emit=emit)
            previous.restore(read_only=True)
            index_path = locate_index(source_root)
            index = artifacts.load_json(index_path) if index_path else None
            manifests = [p for stage in STAGES for p in (source_root / "results/runs" / stage).glob("*/manifest.json")]
            if manifests and index is None:
                raise ValueError("Old checkpoints have no study index/provenance. Preserve them and recover the full recovery folder first.")
            if index:
                source_folds = sorted({row["fold"] for row in index["protocol"]["bindings"]})
                sp, sh, sb = _inputs(review, source_folds)
                common = {k: index["protocol"][k] for k in COMMON}
                _check_protocol(common, {k: base[k] for k in ("code_fingerprint", "training_noise", "seed")})
                _check_protocol(index["protocol"], _protocol(review, sp, sh, sb, common))
                provenance.append({"store": str(store), "index_sha256": artifacts._file_digest(index_path),
                                   "actual_protocol": index["protocol"]})
                for teacher in (index_path.parent / "teachers").glob("*.json"):
                    content = teacher.read_bytes()
                    if teacher.name in teachers and teachers[teacher.name] != content:
                        raise ValueError("Conflicting saved teachers across source stores.")
                    teachers[teacher.name] = content
            for file in manifests:
                manifest = artifacts.load_json(file)
                identity = manifest["identity"]
                if identity["config"]["data"]["fold"] not in folds:
                    continue
                if not (store.parent.name == "parallel_cpu" and manifest["status"] == "complete"
                        and _validate_import(source_root, artifacts.load_run(file.parent), index)):
                    _check_protocol(identity, common)
                relative = file.parent.relative_to(source_root).as_posix()
                if manifest["status"] != "complete":
                    checkpoint = file.parent / "checkpoints/model.resume.json"
                    unfinished.append({"store": str(store), "run": relative,
                        "step": artifacts.load_json(checkpoint)["step"] if checkpoint.exists() else 0,
                        "manifest_sha256": artifacts._file_digest(file),
                        "checkpoint_sha256": artifacts._file_digest(checkpoint) if checkpoint.exists() else None})
                    continue
                artifacts.load_run(file.parent)
                entry = {"manifest_sha256": artifacts._file_digest(file), "identity": identity}
                if relative in inventory:
                    if inventory[relative]["manifest_sha256"] != entry["manifest_sha256"]:
                        raise ValueError("Conflicting completed model attempts in source stores.")
                    continue
                record = previous.catalog["runs"].get(relative)
                if record is None:
                    raise ValueError("Completed model has no verified recovery archive.")
                content = _read_verified(_local(store, record["archive"]))
                if hashlib.sha256(content).hexdigest() != record["sha256"]:
                    raise ValueError("Completed archive checksum changed.")
                _copy_files(file.parent, root / relative)
                publish(_local(target, record["archive"]), content)
                recovery.catalog["runs"][relative] = record
                inventory[relative] = {**entry, "archive_sha256": record["sha256"]}
        if unfinished and not restart_unfinished:
            raise ValueError("Partial Aer fits cannot resume in PyTorch. Explicitly enable RESTART_UNFINISHED_AER; originals remain untouched.")
        for name, content in teachers.items():
            path = root / HANDOFF / "teachers" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        audit = {**base, "imported_completed": inventory, "preserved_unfinished": unfinished,
                 "source_provenance": provenance,
                 "policy": "Completed artifacts retain actual source/device. No partial Aer optimizer state is converted."}
        artifacts.save_json(root / HANDOFF / "torch.json", audit)
        recovery.snapshot()
    plan = {**base, "ready": True, "imported_completed": len(inventory),
            "preserved_partial_aer": len(unfinished), "source_stores": [str(p) for p in sources]}
    publish(plan_file, json.dumps(plan, indent=2).encode())
    emit(f"PyTorch recovery ready: {len(inventory)} immutable completed imports; {len(unfinished)} old partials preserved at source.")
    return plan


@contextmanager
def torch_completed_models(root):
    with cpu_completed_models(root, audit_name="torch.json", usage_name="torch_usage.json"):
        yield


def prepare_torch_execution(root, process, *, counts=(5, 10, 20), device="GPU"):
    """Synthetic throughput selection and Aer agreement; reuse the exact pinned choice.

    This uses synthetic arrays only, never accuracy or validation/test data to
    choose a parallelism setting. Saved choices are strict on Torch/CUDA/device
    and helper hashes, and are backed up in ordinary recovery metadata.
    """
    root = Path(root).resolve()
    counts = tuple(dict.fromkeys(counts))
    if not counts or any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in counts):
        raise ValueError("Choose positive integer worker candidates.")
    directory = root / "results/noisy_training/torch_performance"
    pinned = directory / "selection.json"
    current = {"code_fingerprint": artifacts.source_fingerprint(root),
               "environment": noisy_training.execution_environment()}
    if pinned.exists():
        record = artifacts.load_json(pinned)
        _check_protocol(record, current)
        execution = TorchExecution(**record["selected"])
        if execution.runtime_fingerprint != process.fingerprint or execution.device != device:
            raise ValueError("Pinned Torch runtime changed. Restore matching packages/device/helpers to resume; no fits were started.")
        return execution, pd.DataFrame(record["candidates"]), directory
    if locate_index(root):
        raise ValueError("A Torch study exists without its execution selection. Recover its full metadata first.")
    rng = np.random.default_rng(DEFAULT_SEED)
    circuit, backend, noise = CircuitConfig(), BackendConfig(), NoiseSetting("combined", .01)
    x, w = rng.uniform(0, np.pi, (2, 4)), rng.normal(0, .1, 72)
    y = np.eye(3)[[0, 2]]
    with NoisyBatchService(circuit, backend, AerExecution(device="CPU", workers=1, threads=1), noise) as aer:
        reference_loss, reference_gradient = aer.model().loss_and_gradient(x, y, w)
        reference_probability = aer.model().predict_proba(x, w)
    rows = []
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    for count in counts:
        candidate = TorchExecution(device=device, workers=count, max_models=count, runtime_fingerprint=process.fingerprint)
        with TorchBatchService(circuit, backend, candidate, noise, process=process) as service:
            loss, gradient = service.model().loss_and_gradient(x, y, w)
            np.testing.assert_allclose(loss, reference_loss, atol=1e-10, rtol=0)
            np.testing.assert_allclose(gradient, reference_gradient, atol=1e-9, rtol=0)
            np.testing.assert_allclose(service.model().predict_proba(x, w), reference_probability, atol=1e-10, rtol=0)
            features = rng.uniform(0, np.pi, (count, 10, 4))
            weights = rng.normal(0, .1, (count, 72))
            targets = np.eye(3)[rng.integers(0, 3, (count, 10))]
            # Compare each batched model against its own direct prediction and
            # gradient. Independent model losses must not be averaged together.
            expected = [service.model().loss_and_gradient(features[i], targets[i], weights[i]) for i in range(count)]
            samples = []
            for _ in range(2):
                barrier = Barrier(count)
                def request(i):
                    barrier.wait(timeout=30)
                    return service.model().loss_and_gradient(features[i], targets[i], weights[i])
                started = perf_counter()
                with ThreadPoolExecutor(max_workers=count) as pool:
                    actual = list(pool.map(request, range(count)))
                samples.append(perf_counter() - started)
                for a, e in zip(actual, expected):
                    np.testing.assert_allclose(a[0], e[0], atol=1e-10, rtol=0)
                    np.testing.assert_allclose(a[1], e[1], atol=1e-9, rtol=0)
            row = {"models": count, "gradient_evaluations_per_second": count / float(np.median(samples)),
                   "seconds": float(np.median(samples)), "largest_concurrent_model_batch": max(service.batch_sizes),
                   "largest_concurrent_gradient_batch": max(service.gradient_batch_sizes),
                   "max_gradient_error": float(np.max(np.abs(gradient - reference_gradient))), **service.memory}
            if count > 1 and row["largest_concurrent_gradient_batch"] < 2:
                raise RuntimeError("Concurrent gradients were not batched. Training has not started.")
            rows.append(row)
            print(f"Synthetic Torch check: {count} workers, {row['gradient_evaluations_per_second']:.2f} gradients/s, "
                  f"actual merged batch {row['largest_concurrent_model_batch']}", flush=True)
    # Every candidate spans all five folds. Leave unused GPU memory rather than
    # increase an execution setting that reduced measured throughput.
    best = max(rows, key=lambda r: r["gradient_evaluations_per_second"])
    execution = TorchExecution(device=device, workers=best["models"], max_models=best["models"], runtime_fingerprint=process.fingerprint)
    record = {**current, "synthetic_only": True, "research_results": False, "selected": asdict(execution),
              "runtime": process.runtime, "candidates": rows}
    artifacts.save_json(pinned, record)
    rows_frame = pd.DataFrame(rows)
    rows_frame.to_csv(directory / "benchmark.csv", index=False)
    support = root / "results/noisy_training/runtime_support/torch" / process.fingerprint[:16]
    for name in ("torch_noisy_worker.py", "torch_noisy_service.py", "torch_density_probe.py", "torch_noisy_workflow.py", "cpu_folds.py"):
        _copy_files(Path(__file__).with_name(name), support / name)
    artifacts.save_json(support / "provenance.json", {"runtime": process.runtime, "execution": asdict(execution)})
    return execution, rows_frame, directory
