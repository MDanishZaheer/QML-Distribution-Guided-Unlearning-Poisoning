"""Separate noisy supervised/retraining/unlearning fits with full Adam continuation."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Lock
from time import perf_counter
from unittest.mock import patch
from zipfile import ZIP_DEFLATED, ZipFile
import os
import shutil

import numpy as np
import pandas as pd

from src import artifacts, training
from src.ablations import execution_environment
from src.artifacts import (fingerprint, finish_run, load_arrays, load_json, load_run, save_arrays,
                           save_json, save_source_snapshot, save_training_result, source_fingerprint)
from src.checkpoints import CheckpointSession
from src.checkpoint_storage import CheckpointMirror, checkpoint_status
from src.config import DEFAULT_SEED, ExperimentConfig
from src.data.feature_randomization import experiment_config
from src.evaluation.metrics import classification_metrics, compare_to_retraining, forgetting_changes
from src.execution_resources import (memory_budget_mb, parallel_worker_counts, benchmark_policy,
                                     aer_cpu_threads, execution_refinements)
from src.models.noise import NoiseSetting
from src.models.noisy_training import NoisyBackendIdentity, NoisyBatchService
from src.models.parallel_aer import AerExecution, GpuMonitor, gpu_sample, require_gpu
from src.noise_study import load_noise_inputs
from src.unlearning import methods
from src.unlearning.targets import UnlearningReferences


@dataclass
class NoisyProblem:
    name: str
    fold: int
    config: ExperimentConfig
    original: dict
    reference: dict
    arrays: dict
    reference_arrays: dict
    membership: UnlearningReferences

    @property
    def key(self):
        return f"fold_{self.fold}_{self.name}"


def load_training_problems(review, folds=(0, 1, 2, 3, 4), conditions="all"):
    historical, bindings = load_noise_inputs(review, folds, conditions)
    lookup = {row["manifest_path"].relative_to(review.root).as_posix(): row for row in review.conditions.values()}
    problems, loaded = [], {}
    for (name, fold), group in bindings.groupby(["condition", "fold"], sort=True):
        condition = lookup[group.iloc[0].condition_manifest]
        config = experiment_config(condition["config"])
        if config.seed != DEFAULT_SEED:
            raise ValueError("Noisy training keeps the experiment seed at 6304.")
        case = "clean" if config.poison.kind == "none" else "poisoned"
        original, reference = condition["models"][case], condition["models"][f"retrained_{case}"]
        for run in (original, reference):
            if run["path"] not in loaded:
                loaded[run["path"]] = load_arrays(run["path"] / "features.npz")
        arrays, retained = loaded[original["path"]], loaded[reference["path"]]
        guided = condition["unlearned"][f"{case}_guided"]
        saved = load_json(guided["path"] / "training_references.json")["references"]
        refs = UnlearningReferences(**{key: np.asarray(value) for key, value in saved.items()})
        forgotten = np.flatnonzero(arrays["train_labels"] == config.unlearning.forget_class)
        np.testing.assert_array_equal(refs.forget_indices, forgotten)
        np.testing.assert_array_equal(retained["train_ids"], arrays["train_ids"][~np.isin(np.arange(len(arrays["train"])), forgotten)])
        if np.intersect1d(refs.anchor_indices, forgotten).size or not np.isin(refs.calibration_indices, forgotten).all():
            raise ValueError("Cached anchor/calibration membership is not training-only.")
        problems.append(NoisyProblem(name, int(fold), config, original, reference, arrays, retained, refs))
    return problems, historical, bindings


def training_plan(problems):
    supervised = {}
    for p in problems:
        for role, run in (("original", p.original), ("retrained", p.reference)):
            supervised[run["path"].name] = {"stage": "noisy_supervised", "fold": p.fold, "role": role,
                 "historical_problem": run["path"].name, "updates": run["config"]["training"]["steps"]}
    return pd.DataFrame([*supervised.values(), *[{"stage": "noisy_unlearning", "fold": p.fold,
                "role": method, "historical_problem": p.key, "updates": p.config.unlearning.training.steps}
                for p in problems for method in ("guided", "uniform")]])


def _forbid(*args, **kwargs):
    raise RuntimeError("Noisy training permits only its separate noisy-model stages; original fitting/PCA/downloads/cleanup stay blocked.")


@contextmanager
def noisy_training_execution(root, review_guard=None):
    """Allow fitting only noisy-service models and only the two new checkpoint roots."""
    from src.evaluation.saved_results import protect_review
    import src.checkpoints as checkpoints
    root = Path(root).resolve()
    allowed = {root / "results/runs/noisy_supervised", root / "results/runs/noisy_unlearning"}
    if review_guard is not None:
        review_guard.close()
    try:
        with ExitStack() as stack:
            fit, unlearn, predict = training.fit_vqc, methods.unlearn, __import__("src.models.vqc", fromlist=["VQC"]).VQC.predict_proba
            def check_model(function):
                def checked(model, *args, **kwargs):
                    if not isinstance(getattr(model, "backend_config", None), NoisyBackendIdentity):
                        return _forbid()
                    return function(model, *args, **kwargs)
                return checked
            stack.enter_context(patch("src.training.fit_vqc", check_model(fit)))
            stack.enter_context(patch("src.unlearning.methods.unlearn", check_model(unlearn)))
            stack.enter_context(patch("src.models.vqc.VQC.predict_proba", check_model(predict)))
            for target in ("src.experiments.run_clean_vqc", "src.experiments.run_classical_cv",
                           "src.experiments.fit_vqc", "src.experiments.unlearn", "src.experiment_stages.fit_vqc",
                           "src.experiment_stages.unlearn", "src.experiment_stages.train_condition",
                           "src.models.classical.fit_classical_baselines", "src.experiments.fit_classical_baselines",
                           "src.experiment_stages.fit_classical_baselines", "src.data.preprocessing.ImagePreprocessor.fit",
                           "src.data.dataset.download_mnist", "src.checkpoints.retire_pilot_checkpoints"):
                stack.enter_context(patch(target, _forbid))
            original_open, create = CheckpointSession.open, artifacts.create_run
            def restricted_open(session, directory, *args, **kwargs):
                if Path(directory).resolve() not in allowed:
                    return _forbid()
                return original_open(session, directory, *args, **kwargs)
            def restricted_create(directory, *args, **kwargs):
                if Path(directory).resolve() not in allowed:
                    return _forbid()
                return create(directory, *args, **kwargs)
            stack.enter_context(patch.object(CheckpointSession, "open", restricted_open))
            stack.enter_context(patch.object(artifacts, "create_run", restricted_create))
            stack.enter_context(patch.object(checkpoints, "create_run", restricted_create))
            yield
    finally:
        if review_guard is not None:
            review_guard.enter_context(protect_review())


def benchmark_noisy_training(problem, setting, root, *, candidates=None, require_gpu_device=True,
                             minimum_workers=5, probe_steps=2, worker_counts=None):
    if setting.family == "none" or setting.level <= 0:
        raise ValueError("Select a positive training noise profile.")
    if require_gpu_device:
        require_gpu()
    automatic = candidates is None
    if candidates is None:
        memory = memory_budget_mb(gpu_sample())
        candidates = [AerExecution(circuit_batch_size=batch, workers=n, threads=aer_cpu_threads(),
                      memory_mb=memory, runtime_binding=binding, gpu_parameter_batching=binding)
                      for batch, binding in ((2048, False), (8192, False), (32768, True))
                      for n in (worker_counts or parallel_worker_counts())]
    initial_candidates = len(candidates)
    identity = {"source": source_fingerprint(root), "environment": execution_environment(), "noise": setting,
                "original_problem": problem.original["path"].name, "candidates": benchmark_policy(candidates, automatic=automatic),
                "probe_steps": probe_steps, "minimum_workers": minimum_workers,
                "refine_cpu_and_gather": automatic}
    directory = Path(root) / "results/noisy_training/performance" / fingerprint(identity)[:16]
    if (directory / "benchmark.json").exists():
        saved = load_json(directory / "benchmark.json")
        return AerExecution(**saved["selected"]), pd.DataFrame(saved["candidates"]), directory
    arrays = problem.arrays
    features = arrays["train"][:2]
    targets = np.eye(3)[arrays["train_labels"][:2]]
    weights = problem.original["weights"]["model"]
    reference_execution = AerExecution(device="CPU", workers=1, threads=1, circuit_batch_size=256)
    with NoisyBatchService(problem.config.circuit, problem.config.backend, reference_execution, setting) as cpu:
        expected_loss, expected_gradient = cpu.model().loss_and_gradient(features, targets, weights)
        expected_probabilities = cpu.model().predict_proba(features, weights)
    probe = replace(problem.config.training, steps=probe_steps, evaluate_every=probe_steps)
    rows = []
    for candidate in candidates:
        row = {**asdict(candidate), "status": "failed"}
        try:
            with NoisyBatchService(problem.config.circuit, problem.config.backend, candidate, setting) as service:
                model = service.model()
                actual_loss, actual_gradient = model.loss_and_gradient(features, targets, weights)
                actual_probabilities = model.predict_proba(features, weights)
                np.testing.assert_allclose(actual_probabilities, expected_probabilities, atol=1e-9, rtol=0)
                np.testing.assert_allclose(actual_gradient, expected_gradient, atol=1e-9, rtol=0)
                np.testing.assert_allclose(actual_loss, expected_loss, atol=1e-9, rtol=0)
                row.update(max_probability_error=float(np.max(np.abs(actual_probabilities - expected_probabilities))),
                           max_gradient_error=float(np.max(np.abs(actual_gradient - expected_gradient))),
                           absolute_loss_error=float(abs(actual_loss - expected_loss)))
                with TemporaryDirectory(prefix="qml-noisy-probe-") as temporary:
                    def run_probe(i):
                        return training.fit_vqc(service.model(), arrays["train"], arrays["train_labels"],
                            arrays["validation"], arrays["validation_labels"], probe, DEFAULT_SEED,
                            checkpoint_path=Path(temporary) / f"probe_{i}.json")
                    started = perf_counter()
                    with ThreadPoolExecutor(max_workers=candidate.workers) as pool:
                        results = list(pool.map(run_probe, range(candidate.workers)))
                    elapsed = perf_counter() - started
                    for result in results[1:]:
                        np.testing.assert_allclose(result.last_weights, results[0].last_weights, atol=1e-9, rtol=0)
                    row.update(status="passed", seconds=elapsed,
                               updates_per_second=candidate.workers * probe_steps / elapsed,
                               max_parallel_weights_error=max(float(np.max(np.abs(r.last_weights - results[0].last_weights))) for r in results))
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}"
        rows.append(row)
        print(f"Noisy training benchmark batch={candidate.circuit_batch_size}, workers={candidate.workers}, "
              f"CPU threads={candidate.threads}, gather={candidate.gather_ms:g}ms: {row['status']}", flush=True)
        if automatic and len(rows) == initial_candidates:
            passing = [result for result in rows if result["status"] == "passed" and result["workers"] >= minimum_workers]
            if passing:
                best = max(passing, key=lambda result: result["updates_per_second"])
                base = AerExecution(**{key: best[key] for key in AerExecution.__dataclass_fields__})
                candidates.extend(execution_refinements(base))
    passed = [row for row in rows if row["status"] == "passed" and row["workers"] >= minimum_workers]
    if not passed:
        raise RuntimeError("No requested parallel noisy-training candidate passed; no research fits started.")
    best = max(passed, key=lambda row: row["updates_per_second"])
    selected = AerExecution(**{key: best[key] for key in AerExecution.__dataclass_fields__})
    save_json(directory / "benchmark.json", {"identity": identity, "selected": asdict(selected), "candidates": rows,
                                             "scope": "Synthetic-length performance probes, not research results."})
    pd.DataFrame(rows).to_csv(directory / "candidates.csv", index=False)
    return selected, pd.DataFrame(rows), directory


def _cancel_callback(cancellation, mirror):
    def callback(row):
        if row["step"] == 0:
            mirror.flush_now()
        if cancellation.is_set():
            raise InterruptedError("Noisy study interrupted; committed Adam/RNG state is saved.")
        if mirror.error is not None:
            raise RuntimeError("storage backup failed; noisy optimization stopped at a saved update.") from mirror.error
    return callback


def _save_fit(path, source, arrays, model, outcome, extra=None, probabilities=None):
    if probabilities is None:
        probabilities = model.predict_proba(arrays["test"], outcome.weights)
    metrics = {"models": {"model": classification_metrics(source["predictions"]["test_labels"], probabilities)}, **(extra or {})}
    save_training_result(path, "model", outcome)
    save_arrays(path / "weights.npz", model=outcome.weights)
    save_arrays(path / "features.npz", **arrays)
    save_arrays(path / "predictions.npz", model__test=probabilities,
                test_labels=source["predictions"]["test_labels"], test_sample_ids=source["predictions"]["test_sample_ids"])
    for name in ("preprocessing.joblib", "preprocessing.json"):
        shutil.copyfile(source["path"] / name, path / name)
    save_json(path / "metrics.json", metrics)
    save_json(path / "resources.json", {**model.metadata(), "training_seconds": float(outcome.history.elapsed_seconds.max())})
    outcome.history.assign(model="model").to_csv(path / "history.csv", index=False)
    finish_run(path, "complete")
    return load_run(path)


def _supervised(source, arrays, service, session, lock, root, common, cancellation):
    config = {**source["config"], "backend": asdict(service.executor.config), "classical_baselines": False}
    identity = {**common, "stage": "noisy_supervised", "config": config,
                "historical_problem": source["path"].name, "historical_artifacts": source["manifest"]["artifact_sha256"],
                "frozen_numeric_features": fingerprint(arrays)}
    with lock:
        path, cached = session.open(root / "results/runs/noisy_supervised", identity,
                                    allow_execution=True, slot={"historical_problem": source["path"].name})
    if cached is not None:
        return cached, "loaded"
    resumed = (path / "checkpoints/model.resume.json").exists()
    try:
        save_source_snapshot(path / "source.zip")
        model = service.model(DEFAULT_SEED)
        mirror = CheckpointMirror(path / "checkpoints/model.resume.json")
        with mirror as progress:
            outcome = training.fit_vqc(model, arrays["train"], arrays["train_labels"], arrays["validation"],
                arrays["validation_labels"], ExperimentConfig.from_dict(source["config"]).training, DEFAULT_SEED,
                callback=_cancel_callback(cancellation, mirror), checkpoint_path=progress)
        return _save_fit(path, source, arrays, model, outcome), "resumed" if resumed else "trained"
    except BaseException as error:
        finish_run(path, "failed", error=f"{type(error).__name__}: {error}")
        raise


def _teacher(problem, original, service, directory):
    file = directory / "teachers" / f"{problem.key}.json"
    identity = {"original": original["path"].name, "original_artifacts": original["manifest"]["artifact_sha256"],
                "membership": {key: getattr(problem.membership, key) for key in
                               ("forget_indices", "anchor_indices", "calibration_indices")}}
    if file.exists():
        saved = load_json(file)
        if saved["identity"] != fingerprint(identity) or fingerprint(saved["references"]) != saved["references_fingerprint"]:
            raise ValueError("The fixed noisy teacher references changed.")
        return UnlearningReferences(**{key: np.asarray(value) for key, value in saved["references"].items()})
    model = service.model(DEFAULT_SEED + 1000)
    member, weights = problem.membership, original["weights"]["model"]
    references = UnlearningReferences(forget_indices=member.forget_indices.copy(), anchor_indices=member.anchor_indices.copy(),
         calibration_indices=member.calibration_indices.copy(), original_weights=weights.copy(),
         anchor_probabilities=model.predict_proba(problem.arrays["train"][member.anchor_indices], weights),
         calibration_probabilities=model.predict_proba(problem.arrays["train"][member.calibration_indices], weights))
    save_json(file, {"identity": fingerprint(identity), "references": asdict(references),
                     "references_fingerprint": fingerprint(asdict(references))})
    return references


def _unlearning(problem, method, original, reference, refs, service, session, lock, root, common, cancellation):
    config = {**problem.config.to_dict(), "backend": asdict(service.executor.config), "classical_baselines": False}
    identity = {**common, "stage": "noisy_unlearning", "config": config, "method": method,
                "parent_run": original["path"].name, "reference_run": reference["path"].name,
                "fixed_teacher": fingerprint(asdict(refs)), "problem": problem.key}
    with lock:
        path, cached = session.open(root / "results/runs/noisy_unlearning", identity, allow_execution=True,
                                    slot={"problem": problem.key, "method": method})
    if cached is not None:
        return cached, "loaded"
    resumed = (path / "checkpoints/model.resume.json").exists()
    try:
        save_source_snapshot(path / "source.zip")
        file = path / "training_references.json"
        if file.exists() and fingerprint(load_json(file)) != fingerprint(asdict(refs)):
            raise ValueError("Noisy unlearning teacher must stay fixed across continuation.")
        save_json(file, asdict(refs))
        model = service.model(DEFAULT_SEED)
        mirror = CheckpointMirror(path / "checkpoints/model.resume.json")
        with mirror as progress:
            outcome, target = methods.unlearn(model, problem.arrays["train"], refs, problem.config.unlearning,
                method, DEFAULT_SEED + 2000, callback=_cancel_callback(cancellation, mirror), checkpoint_path=progress)
        probabilities = model.predict_proba(problem.arrays["test"], outcome.weights)
        scores = classification_metrics(original["predictions"]["test_labels"], probabilities)
        extra = {"target": target, "changes": forgetting_changes(original["metrics"]["models"]["model"], scores),
                 "reference_audit": {"n_forget": len(refs.forget_indices), "n_anchors": len(refs.anchor_indices),
                                     "n_calibration": len(refs.calibration_indices)},
                 "retraining_comparison": compare_to_retraining(original["predictions"]["test_labels"],
                                     reference["predictions"]["model__test"], probabilities)}
        return _save_fit(path, problem.original, problem.arrays, model, outcome, extra, probabilities), "resumed" if resumed else "trained"
    except BaseException as error:
        finish_run(path, "failed", error=f"{type(error).__name__}: {error}")
        raise


def _interleave(items, fold):
    buckets = {}
    for item in items:
        buckets.setdefault(fold(item), []).append(item)
    result = []
    while any(buckets.values()):
        for key in sorted(buckets):
            if buckets[key]:
                result.append(buckets[key].pop(0))
    return result


def _parallel(items, function, execution, cancellation, completed):
    with ThreadPoolExecutor(max_workers=execution.workers) as pool:
        futures = {pool.submit(function, item): item for item in items}
        failure = None
        try:
            for future in as_completed(futures):
                if future.cancelled():
                    continue
                try:
                    completed(futures[future], *future.result())
                except BaseException as error:
                    failure = failure or error
                    cancellation.set()
                    for pending in futures:
                        pending.cancel()
        except BaseException:
            cancellation.set()
            for pending in futures:
                pending.cancel()
            raise
        if failure is not None:
            raise failure


def _evaluation(file, identity, compute):
    marker = file.with_suffix(".json")
    if marker.exists():
        saved = load_json(marker)
        if saved["identity"] != fingerprint(identity) or artifacts._file_digest(file) != saved["sha256"]:
            raise ValueError("Cached noisy-training deployment evaluation changed.")
    else:
        save_arrays(file, **compute())
        save_json(marker, {"identity": fingerprint(identity), "sha256": artifacts._file_digest(file)})


def run_noisy_training(review, problems, historical, bindings, execution, *, setting=NoiseSetting("combined", 0.01),
                        allow_training=False, force_retrain=False, progress=None, performance_directory=None):
    """Run inside noisy_training_execution; original result directories are never fitted."""
    if not allow_training or setting.family == "none" or setting.level <= 0:
        raise RuntimeError("Enable noisy training with a positive, explicitly registered noise profile.")
    if len({fingerprint(p.config.circuit) for p in problems}) != 1:
        raise ValueError("A shared training service requires one unchanged circuit architecture.")
    root = review.root
    common = {"code_fingerprint": source_fingerprint(root), "environment": execution_environment(),
              "execution": asdict(execution), "training_noise": asdict(setting), "seed": DEFAULT_SEED}
    protocol = {**common, "original_report": review.report_path.relative_to(root).as_posix(),
                "original_source": review.source_fingerprint, "bindings": bindings.to_dict("records"),
                "problems": [p.key for p in problems], "plan": training_plan(problems).to_dict("records"),
                "historical_inputs": {key: model.identity(root) for key, model in historical.items()},
                "preprocessing": "frozen historical training-fitted PCA; no refit", "training_expectations": "exact noisy density matrix",
                "protocol_note": "New supervised fits start from seed-6304 initialization; new noisy teachers are frozen before unlearning."}
    directory = root / "results/noisy_training" / fingerprint(protocol)[:16]
    # Forced sessions get their own study/teacher namespace as well as fresh model attempts.
    if force_retrain:
        from uuid import uuid4
        directory = directory.with_name(directory.name + f"_{uuid4().hex[:8]}")
    index_file = directory / "study.json"
    previous = load_json(index_file) if index_file.exists() else None
    previous_executions = list(previous.get("previous_executions", [])) if previous else []
    if previous:
        previous_executions.append({key: previous.get(key) for key in ("complete", "elapsed_seconds", "service_resources")})
    index = {"protocol": protocol, "complete": False, "supervised": {}, "unlearned": {}, "evaluations": [],
             "previous_executions": previous_executions,
             "performance_directory": None if performance_directory is None else
                  Path(performance_directory).relative_to(root).as_posix()}
    directory.mkdir(parents=True, exist_ok=True)
    if not (directory / "source.zip").exists():
        save_source_snapshot(directory / "source.zip")
    save_json(index_file, index)
    session, lock, cancellation = CheckpointSession(force_retrain=force_retrain), Lock(), Event()
    supervised, unlearned, jobs = {}, {}, {}
    for p in problems:
        jobs[p.original["path"].name] = (p.original, p.arrays)
        jobs[p.reference["path"].name] = (p.reference, p.reference_arrays)
    plan_count, started = len(jobs) + 2 * len(problems), perf_counter()
    def report(stage, key, run, status):
        group = "supervised" if stage == "noisy_supervised" else "unlearned"
        index[group][key] = {"run": run["path"].relative_to(root).as_posix(), "status": status}
        save_json(index_file, index)
        if progress:
            progress({"stage": stage, "key": key, "status": status,
                      "completed": len(index["supervised"]) + len(index["unlearned"]),
                      "planned": plan_count, "elapsed_seconds": perf_counter() - started})
    with NoisyBatchService(problems[0].config.circuit, problems[0].config.backend, execution, setting) as service, GpuMonitor() as monitor:
        try:
            # Round-robin queueing starts work from different folds together.
            items = _interleave(list(jobs.items()), lambda item: item[1][0]["config"]["data"]["fold"])
            def fit_original(item):
                _, (source, arrays) = item
                return _supervised(source, arrays, service, session, lock, root, common, cancellation)
            def done_original(item, run, status):
                supervised[item[0]] = run
                report("noisy_supervised", item[0], run, status)
            _parallel(items, fit_original, execution, cancellation, done_original)
            refs = {p.key: _teacher(p, supervised[p.original["path"].name], service, directory) for p in problems}
            items = _interleave([(p, method) for p in problems for method in ("guided", "uniform")], lambda item: item[0].fold)
            def fit_unlearning(item):
                p, method = item
                return _unlearning(p, method, supervised[p.original["path"].name], supervised[p.reference["path"].name],
                    refs[p.key], service, session, lock, root, common, cancellation)
            def done_unlearning(item, run, status):
                p, method = item
                unlearned[(p.key, method)] = run
                report("noisy_unlearning", f"{p.key}_{method}", run, status)
            _parallel(items, fit_unlearning, execution, cancellation, done_unlearning)
            # Evaluate the old noiseless-trained models under the SAME training-noise
            # profile. This distinguishes noisy-training effects from inference noise.
            for key, old in historical.items():
                file = directory / "evaluation" / f"historical_noisy_{key}.npz"
                _evaluation(file, {"input": old.identity(root), "noise": setting, **common}, lambda old=old:
                    {"probabilities": service.model().predict_proba(old.features, old.run["weights"]["model"]),
                     "labels": old.labels, "sample_ids": old.sample_ids})
                index["evaluations"].append({"regime": "historical", "deployment": "noisy", "run_id": key,
                    "file": file.relative_to(root).as_posix(), "sha256": artifacts._file_digest(file)})
            # Ideal deployment is evaluated separately without changing any completed fit.
            with NoisyBatchService(problems[0].config.circuit, problems[0].config.backend, execution, NoiseSetting()) as ideal:
                new_inputs = {run["path"].name: (run, load_arrays(run["path"] / "features.npz")["test"])
                              for run in [*supervised.values(), *unlearned.values()]}
                for key, (run, features) in new_inputs.items():
                    file = directory / "evaluation" / f"noisy_trained_ideal_{key}.npz"
                    _evaluation(file, {"run": key, "artifacts": run["manifest"]["artifact_sha256"], "deployment": "ideal", **common},
                        lambda run=run, features=features: {"probabilities": ideal.model().predict_proba(features, run["weights"]["model"]),
                           "labels": run["predictions"]["test_labels"], "sample_ids": run["predictions"]["test_sample_ids"]})
                    index["evaluations"].append({"regime": "noisy_trained", "deployment": "ideal", "run_id": key,
                        "file": file.relative_to(root).as_posix(), "sha256": artifacts._file_digest(file)})
                index["ideal_deployment_resources"] = ideal.metadata()
            index["complete"] = True
        finally:
            index.update(elapsed_seconds=perf_counter() - started, service_resources=service.metadata())
            save_json(index_file, index)
            if monitor.rows:
                pd.DataFrame(monitor.rows).to_csv(directory / "gpu_telemetry.csv", index=False)
            # A global checkpoint scan is an explicit notebook diagnostic.
            # Save this study's small inventory without rereading every Adam state.
            rows = [{"stage": stage, "key": key, **record}
                    for stage in ("supervised", "unlearned")
                    for key, record in index[stage].items()]
            pd.DataFrame(rows).to_csv(directory / "checkpoint_health.csv", index=False)
    return index_file


def noisy_training_bundle(index_file, root):
    root, index_file = Path(root).resolve(), Path(index_file).resolve()
    if not index_file.is_relative_to(root):
        raise ValueError("Noisy study must stay inside the project root.")
    index = load_json(index_file)
    destination = index_file.parent / "noisy_training_download.zip"
    directories = [index_file.parent]
    for group in ("supervised", "unlearned"):
        directories.extend(root / record["run"] for record in index[group].values())
    if index.get("performance_directory"):
        directories.append(root / index["performance_directory"])
    files = {p for directory in directories for p in directory.rglob("*") if p.is_file() and p != destination and p.suffix != ".tmp"}
    files.update(root / name for name in ("src/noisy_training.py", "src/models/noisy_training.py",
                   "src/evaluation/noisy_training_analysis.py", "src/checkpoint_storage.py", "src/execution_resources.py",
                   "src/models/parallel_aer.py", "main_project.ipynb", "README.md"))
    files.update((root / "src").rglob("*.py"))
    with ZipFile(destination, "w", ZIP_DEFLATED) as archive:
        for file in sorted(files):
            if not file.resolve().is_relative_to(root):
                raise ValueError("Noisy bundle path leaves the project.")
            archive.write(file, file.relative_to(root).as_posix())
    return destination
