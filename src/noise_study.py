"""Resumable all-fold inference under synthetic noise, using saved weights only."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile
import hashlib
import os

import numpy as np
import pandas as pd

from src.ablations import execution_environment
from src.artifacts import (fingerprint, load_arrays, load_json, save_arrays, save_json,
                           save_source_snapshot, source_fingerprint)
from src.config import DEFAULT_SEED, CircuitConfig
from src.execution_resources import memory_budget_mb, benchmark_policy
from src.evaluation.saved_results import protect_review
from src.models.noise import (DEFAULT_LEVELS, FAMILIES, DensityReadoutExecutor, NoiseExecution,
                              NoiseSetting, class_probabilities, noise_settings, validate_joint)
from src.models.parallel_aer import GpuMonitor, gpu_sample, require_gpu


@dataclass
class NoiseInput:
    run: dict
    feature_run: dict
    circuit: CircuitConfig
    features: np.ndarray
    labels: np.ndarray
    sample_ids: np.ndarray

    @property
    def key(self):
        return self.run["path"].name

    def identity(self, root):
        return {"run": self.run["path"].relative_to(root).as_posix(),
                "artifacts": self.run["manifest"]["artifact_sha256"],
                "feature_run": self.feature_run["path"].relative_to(root).as_posix(),
                "circuit": asdict(self.circuit),
                "numeric_inputs": fingerprint({"features": self.features, "labels": self.labels,
                                               "sample_ids": self.sample_ids,
                                               "weights": self.run["weights"]["model"]})}


def load_noise_inputs(review, folds=(0, 1, 2, 3, 4), conditions="all"):
    """Each model uses its own frozen PCA features; unlearned models use their parent.

The review loader already verifies all test IDs, split fingerprints and original
source. Fitted sklearn objects are never unpickled or refitted here.
"""
    folds = tuple(folds)
    if not folds or len(set(folds)) != len(folds) or not set(folds).issubset(review.split_paths):
        raise ValueError("Select unique folds from the saved experiment.")
    if conditions not in ("all", "clean_and_label_flip_50pct"):
        raise ValueError("Unknown noise condition selection.")
    models, bindings, features = {}, [], {}
    for condition in review.conditions.values():
        config = condition["config"]
        fold, poison = config["data"]["fold"], config["poison"]
        if fold not in folds:
            continue
        if conditions != "all" and not (poison["kind"] == "none" or
                    (poison["kind"] == "label_flip" and poison["rate"] == 0.5)):
            continue
        if config["seed"] != DEFAULT_SEED or config["backend"]["shots"] is not None:
            raise ValueError("Expected the completed seed-6304 exact-expectation study.")
        case = "clean" if poison["kind"] == "none" else "poisoned"
        name = "clean" if case == "clean" else f"{poison['kind']}_{poison['rate'] * 100:g}pct"
        original = condition["models"][case]
        reference = condition["models"][f"retrained_{case}"]
        selected = {"original": original, "retrained": reference,
                    **{method: condition["unlearned"][f"{case}_{method}"] for method in ("guided", "uniform")}}
        for role, run in selected.items():
            feature_run = original if role in ("guided", "uniform") else run
            key = run["path"].name
            if key not in models:
                if feature_run["path"] not in features:
                    features[feature_run["path"]] = load_arrays(feature_run["path"] / "features.npz")["test"]
                predictions = run["predictions"]
                np.testing.assert_array_equal(predictions["test_sample_ids"], reference["predictions"]["test_sample_ids"])
                np.testing.assert_array_equal(predictions["test_labels"], reference["predictions"]["test_labels"])
                models[key] = NoiseInput(run, feature_run, CircuitConfig(**run["config"]["circuit"]),
                                         features[feature_run["path"]], predictions["test_labels"],
                                         predictions["test_sample_ids"])
                if len(models[key].features) != len(models[key].labels):
                    raise ValueError("Saved test features and labels are not aligned.")
            elif models[key].feature_run["path"] != feature_run["path"]:
                raise ValueError("One checkpoint unexpectedly refers to different preprocessing.")
            bindings.append({"condition": name, "attack": poison["kind"], "poison_rate": poison["rate"],
                             "fold": fold, "role": role, "run_id": key,
                             "original_id": original["path"].name, "reference_id": reference["path"].name,
                             "condition_manifest": condition["manifest_path"].relative_to(review.root).as_posix(),
                             "realized_training_fraction": condition["poisoning"]["actual_fraction_all_training"],
                             "realized_eligible_fraction": condition["poisoning"]["actual_fraction_eligible"]})
    frame = pd.DataFrame(bindings).sort_values(["fold", "attack", "poison_rate", "role"]).reset_index(drop=True)
    if frame.empty or frame.duplicated(["condition", "fold", "role"]).any():
        raise ValueError("Missing or duplicate noise study model roles.")
    return models, frame


def noise_plan(models, bindings, levels=DEFAULT_LEVELS, families=FAMILIES, shots=(None, 2048)):
    settings = noise_settings(levels, families)
    shots = tuple(shots)
    if (not shots or len(set(shots)) != len(shots) or None not in shots or
            any(value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 1) for value in shots)):
        raise ValueError("Include exact mode (None) and unique optional positive shot budgets.")
    return pd.DataFrame([{"noise_family": s.family, "noise_level": s.level,
                          "modes": ["exact" if n is None else f"shots_{n}" for n in shots],
                          "condition_fold_pairs": len(bindings[["condition", "fold"]].drop_duplicates()),
                          "unique_saved_models": len(models), **s.describe()} for s in settings])


def benchmark_noise(model, root, *, require_gpu_device=True, candidates=None):
    """Benchmark batched density inference separately from statevector optimization."""
    if require_gpu_device:
        require_gpu()
    root = Path(root).resolve()
    automatic = candidates is None
    if candidates is None:
        sample = gpu_sample()
        memory = memory_budget_mb(sample)
        devices = ["CPU", "GPU"] if require_gpu_device else ["CPU"]
        candidates = [NoiseExecution(device, batch, min(4, os.cpu_count() or 1), memory)
                      for device in devices for batch in (128, 512, 2048)]
    environment, source = execution_environment(), source_fingerprint(root)
    identity = {"version": 1, "source": source, "environment": environment,
                "input": model.identity(root), "candidates": benchmark_policy(candidates, automatic=automatic)}
    destination = root / "results/noise_studies/performance" / fingerprint(identity)[:16]
    file = destination / "benchmark.json"
    if file.exists():
        cached = load_json(file)
        return NoiseExecution(**cached["selected"]), pd.DataFrame(cached["candidates"]), destination
    # Repeat saved rows to test device throughput with enough batched circuits.
    x = np.tile(model.features, (max(1, 256 // len(model.features)), 1))[:256]
    w = np.broadcast_to(model.run["weights"]["model"], (len(x), model.circuit.n_parameters))
    setting = NoiseSetting("depolarizing", 0.01)
    rows = []
    with protect_review():
        reference = DensityReadoutExecutor(model.circuit, setting, NoiseExecution("CPU", 128, 1)).run(x[:3], w[:3])
        for candidate in candidates:
            row = {**asdict(candidate), "status": "failed"}
            try:
                executor = DensityReadoutExecutor(model.circuit, setting, candidate)
                np.testing.assert_allclose(executor.run(x[:3], w[:3]), reference, atol=1e-10, rtol=0)
                start = perf_counter()
                result = executor.run(x, w)
                elapsed = perf_counter() - start
                validate_joint(result)
                row.update(status="passed", seconds=elapsed, circuits_per_second=len(x) / elapsed)
            except Exception as error:
                row["error"] = f"{type(error).__name__}: {error}"
            rows.append(row)
            print(f"Noise benchmark {candidate.device}, batch={candidate.circuit_batch_size}: {row['status']}", flush=True)
    passed = [row for row in rows if row["status"] == "passed"]
    if not passed:
        raise RuntimeError("No density-matrix execution candidate passed; no noise study started.")
    best = max(passed, key=lambda row: row["circuits_per_second"])
    selected = NoiseExecution(**{key: best[key] for key in NoiseExecution.__dataclass_fields__})
    save_json(file, {"selected": asdict(selected), "candidates": rows, "identity": identity,
                     "scope": "Inference performance diagnostic, not research results."})
    pd.DataFrame(rows).to_csv(destination / "candidates.csv", index=False)
    return selected, pd.DataFrame(rows), destination


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _local(root, relative):
    file = (root / relative).resolve()
    if not file.is_relative_to(root):
        raise ValueError("Noise artifact path leaves the project root.")
    return file


def _load_density(directory, identity=None):
    metadata = load_json(directory / "complete.json")
    if identity is not None and fingerprint(metadata["identity"]) != fingerprint(identity):
        raise ValueError("Noise cache identity changed.")
    file = directory / "joint.npz"
    if _digest(file) != metadata["sha256"]:
        raise ValueError("Cached noise probabilities changed or are incomplete.")
    return load_arrays(file), metadata


def run_noise_study(review, models, bindings, execution, *, levels=DEFAULT_LEVELS, families=FAMILIES,
                    shots=(None, 2048), allow_execution=False, force_evaluate=False, progress=None,
                    performance_directory=None):
    """Only fixed-weight inference is allowed; completed per-model caches resume.

Requests for different folds/models with the same circuit/noise are combined
in one parameter-binding batch. Separate weights/features remain aligned.
"""
    if not allow_execution:
        raise RuntimeError("Enable RUN_NOISE_STUDY to execute fixed-weight noise evaluation.")
    root = review.root
    noise_plan(models, bindings, levels, families, shots)
    settings = noise_settings(levels, families)
    gates = sorted({fingerprint(s.gate_setting()): s.gate_setting() for s in settings}.values(),
                   key=lambda s: (s.family != "none", s.family, s.level))
    environment, source = execution_environment(), source_fingerprint(root)
    protocol = {"version": 1, "original_report": review.report_path.relative_to(root).as_posix(),
                "original_source": review.source_fingerprint, "current_source": source,
                "environment": environment, "execution": asdict(execution), "seed": DEFAULT_SEED,
                "settings": [asdict(s) for s in settings], "shots": list(shots),
                "bindings": bindings.to_dict("records"), "inputs": {key: value.identity(root) for key, value in models.items()},
                "training_updates": 0, "preprocessing_refitted": False,
                "inference_only": True, "compilation": "original ry/rz/cx; optimization_level=0",
                "sampling": "seed 6304 per model/profile; joint multinomial; not independent training seeds"}
    study_id = fingerprint(protocol)[:16] + (f"_{uuid4().hex[:8]}" if force_evaluate else "")
    destination = root / "results/noise_studies" / study_id
    index_file = destination / "study.json"
    previous = load_json(index_file) if index_file.exists() else None
    history = list(previous.get("previous_executions", [])) if previous else []
    if previous:
        history.append({key: previous.get(key) for key in
                        ("executed_utc", "complete", "elapsed_seconds", "density_evaluated", "density_loaded", "resources")})
    index = {"protocol": protocol, "complete": False, "density_records": [], "prediction_records": [],
             "executed_utc": datetime.now(timezone.utc).isoformat(), "previous_executions": history,
             "performance_directory": None if performance_directory is None else
                 Path(performance_directory).relative_to(root).as_posix()}
    started, planned = perf_counter(), len(models) * len(gates)
    last_index_save, last_progress = started, started - 10
    if not (destination / "source.zip").exists():
        destination.mkdir(parents=True, exist_ok=True)
        save_source_snapshot(destination / "source.zip")
    save_json(index_file, index)
    densities = {}
    resources = []
    with protect_review(), GpuMonitor() as monitor:
        try:
            circuits = {fingerprint(model.circuit): model.circuit for model in models.values()}
            for setting in gates:
                for circuit_key, circuit in circuits.items():
                    selected = [model for model in models.values() if fingerprint(model.circuit) == circuit_key]
                    pending = []
                    for model in selected:
                        identity = {"source": source, "environment": environment, "execution": asdict(execution),
                                    "input": model.identity(root), "gate_noise": asdict(setting), "seed": DEFAULT_SEED}
                        directory = root / "results/noise_studies/cache" / fingerprint(identity)[:24]
                        if force_evaluate:
                            directory = directory.with_name(directory.name + f"_{study_id}")
                        if (directory / "complete.json").exists():
                            arrays, _ = _load_density(directory, identity)
                            np.testing.assert_array_equal(arrays["sample_ids"], model.sample_ids)
                            np.testing.assert_array_equal(arrays["labels"], model.labels)
                            densities[(model.key, fingerprint(setting))] = arrays["joint"]
                            index["density_records"].append({"run_id": model.key, "gate_noise": asdict(setting),
                                "cache": directory.relative_to(root).as_posix(), "status": "loaded"})
                        else:
                            pending.append((model, directory, identity))
                    executor = DensityReadoutExecutor(circuit, setting, execution) if pending else None
                    while pending:
                        # Merge complete model requests, then let Aer split its bounded circuit batches.
                        group, count = [], 0
                        while pending and (not group or count + len(pending[0][0].features) <= execution.circuit_batch_size):
                            item = pending.pop(0)
                            group.append(item)
                            count += len(item[0].features)
                        x = np.concatenate([item[0].features for item in group])
                        w = np.concatenate([np.broadcast_to(item[0].run["weights"]["model"],
                                            (len(item[0].features), circuit.n_parameters)) for item in group])
                        joint, offset = executor.run(x, w), 0
                        for model, directory, identity in group:
                            values = joint[offset:offset + len(model.features)]
                            offset += len(model.features)
                            if setting.family == "none":
                                actual, _ = class_probabilities(values, logit_scale=circuit.logit_scale)
                                np.testing.assert_allclose(actual, model.run["predictions"]["model__test"], atol=1e-9, rtol=0,
                                                           err_msg="Zero-noise predictions differ from the saved model.")
                            save_arrays(directory / "joint.npz", joint=values, labels=model.labels, sample_ids=model.sample_ids)
                            save_json(directory / "complete.json", {"identity": identity, "sha256": _digest(directory / "joint.npz")})
                            densities[(model.key, fingerprint(setting))] = values
                            index["density_records"].append({"run_id": model.key, "gate_noise": asdict(setting),
                                "cache": directory.relative_to(root).as_posix(), "status": "evaluated"})
                        # Each model cache is already durable. Throttle the large storage index,
                        # which is reconstructed from those caches after an abrupt runtime loss.
                        now = perf_counter()
                        if now - last_index_save >= 10:
                            save_json(index_file, index)
                            last_index_save = now
                        if progress and (now - last_progress >= 10 or len(index["density_records"]) == planned):
                            progress({"completed": len(index["density_records"]), "planned": planned,
                                      "gate_channel": setting.family, "level": setting.level,
                                      "elapsed_seconds": perf_counter() - started})
                            last_progress = now
                    if executor is not None:
                        resources.append({"gate_noise": asdict(setting), **executor.metadata()})
            # A cached zero-noise control is also checked before any derived predictions are accepted.
            for model in models.values():
                actual, _ = class_probabilities(densities[(model.key, fingerprint(NoiseSetting()))],
                                               logit_scale=model.circuit.logit_scale)
                np.testing.assert_allclose(actual, model.run["predictions"]["model__test"], atol=1e-9, rtol=0)
            for setting in settings:
                for model in models.values():
                    joint = densities[(model.key, fingerprint(setting.gate_setting()))]
                    for budget in shots:
                        mode = "exact" if budget is None else f"shots_{budget}"
                        probabilities, expectations = class_probabilities(joint, readout_error=setting.readout_error,
                              shots=budget, seed=DEFAULT_SEED, logit_scale=model.circuit.logit_scale)
                        key = fingerprint({"run_id": model.key, "setting": setting, "shots": budget})[:24]
                        file = destination / "predictions" / f"{key}.npz"
                        save_arrays(file, probabilities=probabilities, expectations=expectations,
                                    labels=model.labels, sample_ids=model.sample_ids)
                        index["prediction_records"].append({"run_id": model.key, **setting.describe(),
                                    "mode": mode, "shots": budget, "file": file.relative_to(root).as_posix(),
                                    "sha256": _digest(file)})
                save_json(index_file, index)
            index["complete"] = True
        finally:
            index.update(elapsed_seconds=perf_counter() - started, resources=resources,
                         density_evaluated=sum(r["status"] == "evaluated" for r in index["density_records"]),
                         density_loaded=sum(r["status"] == "loaded" for r in index["density_records"]))
            save_json(index_file, index)
            if monitor.rows:
                pd.DataFrame(monitor.rows).to_csv(destination / "gpu_telemetry.csv", index=False)
    return index_file


def load_noise_predictions(index_file, root):
    """Portable numeric reload with checksums; never run a simulator on a cache miss."""
    root = Path(root).resolve()
    index_file = _local(root, index_file)
    index = load_json(index_file)
    predictions = {}
    for row in index["prediction_records"]:
        key = (row["run_id"], row["family"], row["level"], row["mode"])
        if key in predictions:
            raise ValueError("Duplicate model/noise/mode predictions.")
        path = _local(root, row["file"])
        if _digest(path) != row["sha256"]:
            raise ValueError("Saved noise predictions are missing or changed.")
        arrays = load_arrays(path)
        if arrays["probabilities"].shape != (len(arrays["labels"]), 3):
            raise ValueError("Noise predictions and labels differ in length.")
        predictions[key] = (row, arrays)
    return index, predictions


def noise_download_bundle(index_file, root):
    """Incremental new noise artifacts; preserve their actual runtime provenance."""
    root, index_file = Path(root).resolve(), Path(index_file).resolve()
    index, _ = load_noise_predictions(index_file, root)
    destination = index_file.parent / "noise_download.zip"
    directories = [index_file.parent]
    directories.extend(_local(root, row["cache"]) for row in index["density_records"])
    if index.get("performance_directory"):
        directories.append(_local(root, index["performance_directory"]))
    files = {p for directory in directories for p in directory.rglob("*")
             if p.is_file() and p != destination and p.suffix != ".tmp"}
    files.update(root / name for name in ("src/noise_study.py", "src/models/noise.py",
                 "src/evaluation/noise_analysis.py", "main_project.ipynb", "README.md"))
    files.update((root / "src").rglob("*.py"))
    with ZipFile(destination, "w", ZIP_DEFLATED) as archive:
        for file in sorted(files):
            _local(root, file)
            archive.write(file, file.relative_to(root).as_posix())
    return destination
