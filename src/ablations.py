"""Focused OFAT unlearning study using immutable historical model inputs.

No supervised training, PCA fitting, MNIST downloads or old-run mutation occur
here. A new exact-expectation Aer service executes independent optimizers.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager, ExitStack
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Lock
from time import perf_counter
from unittest.mock import patch
import os
import shutil
from importlib import metadata

import numpy as np
import pandas as pd

from src.artifacts import (environment_info, fingerprint, finish_run, load_arrays, load_json,
                           load_run, save_arrays, save_json, save_source_snapshot,
                           save_training_result, source_fingerprint)
from src.checkpoints import CheckpointSession
from src.checkpoint_storage import CheckpointMirror
from src.execution_resources import (memory_budget_mb, parallel_worker_counts, benchmark_policy,
                                     aer_cpu_threads, execution_refinements)
from src.config import DEFAULT_SEED, ExperimentConfig
from src.evaluation.metrics import classification_metrics, compare_to_retraining, forgetting_changes
from src.models.parallel_aer import AerBatchService, AerExecution, GpuMonitor, gpu_sample, require_gpu
from src.models.vqc import VQC
from src.unlearning import methods
from src.unlearning.targets import UnlearningReferences, make_forget_target


SWEEPS = {"alpha": (0.0, 0.5, 1.0, 2.0, 4.0),
          "regularization": (0.0, 0.001, 0.01, 0.1),
          "beta": (0.25, 0.5, 0.75, 1.0, 2.0, 4.0)}
DEFAULTS = {"alpha": 1.0, "regularization": 0.01, "beta": 1.0}


def execution_environment():
    import qiskit_aer
    result = environment_info()
    result["aer_runtime_version"] = qiskit_aer.__version__
    try:
        result["qiskit_aer_gpu"] = metadata.version("qiskit-aer-gpu")
    except metadata.PackageNotFoundError:
        result["qiskit_aer_gpu"] = None
    return result


def parameter_settings(base):
    """12 unique changes; defaults are loaded once rather than fitted per axis."""
    if any(getattr(base, key) != value for key, value in DEFAULTS.items()):
        raise ValueError("The study requires the registered default alpha/lambda/beta.")
    return [(parameter, value, replace(base, **{parameter: value}, methods=("guided",)))
            for parameter, values in SWEEPS.items() for value in values if value != DEFAULTS[parameter]]


@dataclass
class AblationProblem:
    name: str
    fold: int
    config: ExperimentConfig
    parent: dict
    retrained: dict
    baseline: dict
    uniform: dict
    arrays: dict
    references: UnlearningReferences
    baseline_origin: str = "historical"

    @property
    def key(self):
        return f"fold_{self.fold}_{self.name}"


def load_problems(review, folds=(0, 1, 2, 3, 4), *, include_pixel=False):
    """Historical features/teachers are numeric; never unpickle fitted sklearn objects."""
    folds = tuple(folds)
    if not folds or len(set(folds)) != len(folds) or not set(folds).issubset(review.split_paths):
        raise ValueError("Choose unique folds from the original five-fold study.")
    problems = []
    for fold in folds:
        for name in (("clean", "label_flip_50pct", "pixel_patch_50pct") if include_pixel
                     else ("clean", "label_flip_50pct")):
            candidates = [condition for condition in review.conditions.values()
                          if condition["config"]["data"]["fold"] == fold
                          and ((name == "clean" and condition["config"]["poison"]["kind"] == "none")
                               or (name != "clean" and condition["config"]["poison"]["kind"] == name.rsplit("_", 1)[0]
                                   and condition["config"]["poison"]["rate"] == 0.5))]
            if len(candidates) != 1:
                raise ValueError(f"Missing/ambiguous original condition: {fold}/{name}")
            condition = candidates[0]
            config = ExperimentConfig.from_dict(condition["config"])
            if (config.seed != DEFAULT_SEED or config.unlearning.training.steps != 100
                    or config.unlearning.training.batch_size != 10 or config.unlearning.anchor_fraction != 0.25
                    or config.backend.shots is not None or any(config.backend.noise_rates)):
                raise ValueError("Saved condition differs from the registered exact-expectation protocol.")
            parameter_settings(config.unlearning)
            case = "clean" if name == "clean" else "poisoned"
            parent, retrained = (condition["models"][key] for key in (case, f"retrained_{case}"))
            baseline, uniform = (condition["unlearned"][f"{case}_{method}"] for method in ("guided", "uniform"))
            problems.append(problem_from_runs(name, fold, config, parent, retrained, baseline, uniform,
                                              read_path=review._read_path))
    return problems


def problem_from_runs(name, fold, config, parent, retrained, baseline, uniform, *,
                      read_path=None, baseline_origin="historical"):
    """Validate saved fixed teachers for historical or separately registered new conditions."""
    read_path = read_path or (lambda path: path)
    arrays = load_arrays(read_path(parent["path"] / "features.npz"))
    saved = load_json(read_path(baseline["path"] / "training_references.json"))["references"]
    references = UnlearningReferences(**{key: np.asarray(value) for key, value in saved.items()})
    np.testing.assert_array_equal(references.original_weights, parent["weights"]["model"])
    forgotten = np.flatnonzero(arrays["train_labels"] == config.unlearning.forget_class)
    np.testing.assert_array_equal(references.forget_indices, forgotten)
    if (np.intersect1d(references.anchor_indices, forgotten).size
            or not np.isin(references.calibration_indices, forgotten).all()):
        raise ValueError("Historical calibration/anchor membership is invalid.")
    cached = load_arrays(read_path(baseline["path"] / "references.npz"))
    for index_key, id_key in (("forget_indices", "forget_ids"), ("anchor_indices", "anchor_ids"),
                              ("calibration_indices", "calibration_ids")):
        np.testing.assert_array_equal(arrays["train_ids"][getattr(references, index_key)], cached[id_key])
    for key in ("anchor_probabilities", "calibration_probabilities", "original_weights"):
        np.testing.assert_array_equal(getattr(references, key), cached[key])
    expected_target = make_forget_target(references.calibration_probabilities,
                                        config.unlearning.forget_class, config.unlearning.beta)
    np.testing.assert_allclose(expected_target, baseline["metrics"]["target"], atol=1e-14, rtol=0)
    identity = baseline["manifest"]["identity"]
    if identity["parent_run"] != parent["path"].name or identity["reference_run"] != retrained["path"].name:
        raise ValueError("Historical parent/reference links disagree.")
    retained = load_arrays(read_path(retrained["path"] / "features.npz"))
    np.testing.assert_array_equal(retained["train_ids"], arrays["train_ids"][~np.isin(np.arange(len(arrays["train"])), forgotten)])
    for run in (parent, retrained, baseline, uniform):
        np.testing.assert_array_equal(run["predictions"]["test_sample_ids"], parent["predictions"]["test_sample_ids"])
        np.testing.assert_array_equal(run["predictions"]["test_labels"], parent["predictions"]["test_labels"])
    return AblationProblem(name, fold, config, parent, retrained, baseline, uniform, arrays, references,
                           baseline_origin)


def plan_table(problems):
    return pd.DataFrame([{"problem": problem.key, "fold": problem.fold, "condition": problem.name,
                          "parameter": parameter, "value": value, **{k: getattr(setting, k) for k in DEFAULTS},
                          "updates": setting.training.steps, "seed": problem.config.seed,
                          "parent_run": problem.parent["path"].name}
                         for problem in problems for parameter, value, setting in parameter_settings(problem.config.unlearning)])


def _forbid(*args, **kwargs):
    raise RuntimeError("The focused study forbids supervised training, downloads, PCA fitting and cleanup.")


@contextmanager
def ablation_execution(root, review_guard=None):
    """Temporarily allow new unlearning only; restore the notebook's review guard."""
    from src.evaluation.saved_results import protect_review
    if review_guard is not None:
        review_guard.close()
    try:
        with ExitStack() as stack:
            for target in ("src.training.fit_vqc", "src.experiments.fit_vqc", "src.experiment_stages.fit_vqc",
                           "src.experiments.run_classical_cv", "src.experiments.run_clean_vqc",
                           "src.experiment_stages.train_condition", "src.models.classical.fit_classical_baselines",
                           "src.experiments.fit_classical_baselines", "src.experiment_stages.fit_classical_baselines",
                           "src.data.preprocessing.ImagePreprocessor.fit", "src.data.dataset.download_mnist",
                           "src.checkpoints.retire_pilot_checkpoints"):
                stack.enter_context(patch(target, _forbid))
            original_open = CheckpointSession.open
            allowed = [(Path(root) / "results/runs" / name).resolve() for name in ("ablations", "ablation_validation")]

            def restricted_open(session, directory, *args, **kwargs):
                if Path(directory).resolve() not in allowed:
                    raise RuntimeError("Ablation checkpoints must stay in their separate run directories.")
                return original_open(session, directory, *args, **kwargs)

            stack.enter_context(patch.object(CheckpointSession, "open", restricted_open))
            yield
    finally:
        if review_guard is not None:
            review_guard.enter_context(protect_review())


def validate_execution(problem, service, tolerance=1e-9):
    """Predictions, cached teachers, and all gradient entries; no optimization."""
    model = service.model(problem.config.seed)
    refs, arrays = problem.references, problem.arrays
    discrepancies = {}
    for name, features, expected in (
        ("test", arrays["test"], problem.parent["predictions"]["model__test"]),
        ("anchors", arrays["train"][refs.anchor_indices], refs.anchor_probabilities),
        ("calibration", arrays["train"][refs.calibration_indices], refs.calibration_probabilities),
    ):
        actual = model.predict_proba(features, refs.original_weights)
        discrepancies[name] = float(np.max(np.abs(actual - expected)))
        np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=0)
    cpu = VQC(problem.config.circuit, replace(problem.config.backend, device="CPU", max_parallel_threads=1), DEFAULT_SEED)
    features = arrays["train"][np.r_[refs.forget_indices[:2], refs.anchor_indices[:2]]]
    target = make_forget_target(refs.calibration_probabilities)
    targets = np.vstack([np.broadcast_to(target, (2, 3)), refs.anchor_probabilities[:2]])
    expected_loss, expected_gradient = cpu.loss_and_gradient(features, targets, refs.original_weights)
    actual_loss, actual_gradient = model.loss_and_gradient(features, targets, refs.original_weights)
    discrepancies["gradient"] = float(np.max(np.abs(actual_gradient - expected_gradient)))
    discrepancies["loss"] = float(abs(actual_loss - expected_loss))
    np.testing.assert_allclose(actual_gradient, expected_gradient, atol=tolerance, rtol=0)
    np.testing.assert_allclose(actual_loss, expected_loss, atol=tolerance, rtol=0)
    return {"problem": problem.key, "tolerance": tolerance, "maximum_errors": discrepancies, "passed": True}


def benchmark_execution(problem, root, *, candidates=None, require_gpu_device=True, probe_steps=3,
                        minimum_workers=1, worker_counts=None):
    """Time full short optimizer probes, including checkpoint I/O; never research results."""
    if require_gpu_device:
        require_gpu()
    sample = gpu_sample()
    memory = memory_budget_mb(sample)
    automatic = candidates is None
    if candidates is None:
        threads = aer_cpu_threads()
        candidates = [AerExecution(circuit_batch_size=batch, workers=workers, threads=threads,
                                   memory_mb=memory, runtime_binding=binding,
                                   gpu_parameter_batching=binding)
                      for batch, binding in ((2048, False), (8192, False), (32768, True))
                      for workers in (worker_counts or parallel_worker_counts())]
    initial_candidates = len(candidates)
    environment, source = execution_environment(), source_fingerprint(root)
    cache_key = fingerprint({"version": 1, "environment": environment, "source": source,
                             "parent": problem.parent["path"].name, "candidates": benchmark_policy(candidates, automatic=automatic),
                             "probe_steps": probe_steps, "minimum_workers": minimum_workers,
                             "refine_cpu_and_gather": automatic})[:16]
    destination = Path(root) / "results/ablations/performance" / cache_key
    cache = destination / "benchmark.json"
    if cache.exists():
        saved = load_json(cache)
        return AerExecution(**saved["selected"]), pd.DataFrame(saved["candidates"]), destination
    rows = []
    probe = replace(problem.config.unlearning,
                    training=replace(problem.config.unlearning.training, steps=probe_steps,
                                     evaluate_every=probe_steps), methods=("guided",))
    for execution in candidates:
        record = {**asdict(execution), "status": "failed"}
        try:
            with AerBatchService(problem.config.circuit, problem.config.backend, execution) as service:
                validate_execution(problem, service)
                with TemporaryDirectory(prefix="qml-aer-probe-") as temporary:
                    def fit_probe(index):
                        return methods.unlearn(service.model(), problem.arrays["train"], problem.references,
                                               probe, "guided", DEFAULT_SEED + 2000,
                                               checkpoint_path=Path(temporary) / f"probe_{index}.resume.json")[0]
                    started = perf_counter()
                    with ThreadPoolExecutor(max_workers=execution.workers) as workers:
                        outcomes = list(workers.map(fit_probe, range(execution.workers)))
                    elapsed = perf_counter() - started
                    for outcome in outcomes[1:]:
                        np.testing.assert_allclose(outcome.last_weights, outcomes[0].last_weights, atol=1e-9, rtol=0)
                    record.update(status="passed", seconds=elapsed,
                                  updates_per_second=execution.workers * probe_steps / elapsed,
                                  physical_jobs=service.executor.stats["jobs"])
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {error}"
        rows.append(record)
        print(f"Benchmark batch={execution.circuit_batch_size}, workers={execution.workers}, "
              f"CPU threads={execution.threads}, gather={execution.gather_ms:g}ms, "
              f"runtime binding={execution.runtime_binding}: {record['status']}", flush=True)
        if automatic and len(rows) == initial_candidates:
            passing = [row for row in rows if row["status"] == "passed" and row["workers"] >= minimum_workers]
            if passing:
                best = max(passing, key=lambda row: row["updates_per_second"])
                base = AerExecution(**{key: best[key] for key in AerExecution.__dataclass_fields__})
                candidates.extend(execution_refinements(base))
    passed = [row for row in rows if row["status"] == "passed" and row["workers"] >= minimum_workers]
    if not passed:
        raise RuntimeError("No Aer execution candidate passed numerical checks. No study runs started.")
    best = max(passed, key=lambda row: row["updates_per_second"])
    selected = AerExecution(**{key: best[key] for key in AerExecution.__dataclass_fields__})
    save_json(cache, {"scope": "performance probes, not research experiments", "selected": asdict(selected),
                      "candidates": rows, "environment": environment, "source_fingerprint": source})
    pd.DataFrame(rows).to_csv(destination / "candidates.csv", index=False)
    return selected, pd.DataFrame(rows), destination


def _run_variant(problem, settings, service, session, session_lock, root, environment, source, *,
                 validation=False, cancellation=None):
    config = replace(problem.config, unlearning=settings)
    identity = {"stage": "ablation_validation" if validation else "focused_guided_ablation",
                "config": config.to_dict(), "code_fingerprint": source, "environment": environment,
                "parent_run": problem.parent["path"].name, "reference_run": problem.retrained["path"].name,
                "teacher_run": problem.baseline["path"].name,
                "parent_artifacts": problem.parent["manifest"]["artifact_sha256"],
                "teacher_artifacts": problem.baseline["manifest"]["artifact_sha256"],
                "execution": asdict(service.execution), "method": "guided"}
    directory = Path(root) / "results/runs" / ("ablation_validation" if validation else "ablations")
    with session_lock:
        path, cached = session.open(directory, identity, allow_execution=True,
                                    slot={"problem": problem.key, "unlearning": asdict(settings)})
    if cached is not None:
        return cached, "loaded"
    resumed = (path / "checkpoints/model.resume.json").exists()
    try:
        save_source_snapshot(path / "source.zip")
        refs = problem.references
        reference_file = path / "training_references.json"
        teacher_payload = {"references": asdict(refs), "imported_from": problem.baseline["path"].name}
        if reference_file.exists():
            if load_json(reference_file) != json_normal(teacher_payload):
                raise ValueError("Persisted fixed teacher references changed.")
        else:
            save_json(reference_file, teacher_payload)
        model = service.model(problem.config.seed)
        mirror = CheckpointMirror(path / "checkpoints/model.resume.json")
        def check_cancellation(row):
            if row["step"] == 0:
                mirror.flush_now()
            if cancellation is not None and cancellation.is_set():
                raise InterruptedError("Study stopped; the last complete optimizer update is checkpointed.")
            if mirror.error is not None:
                raise RuntimeError("storage backup failed; optimization stopped at a saved update.") from mirror.error
        with mirror as progress_file:
            outcome, target = methods.unlearn(model, problem.arrays["train"], refs, settings, "guided",
                                              problem.config.seed + 2000, checkpoint_path=progress_file,
                                              callback=check_cancellation)
        probabilities = model.predict_proba(problem.arrays["test"], outcome.weights)
        labels = problem.parent["predictions"]["test_labels"]
        scores = classification_metrics(labels, probabilities, settings.forget_class)
        comparison = compare_to_retraining(labels, problem.retrained["predictions"]["model__test"],
                                            probabilities, settings.forget_class)
        audit = dict(problem.baseline["metrics"]["reference_audit"])
        save_json(path / "metrics.json", {"models": {"model": scores}, "target": target,
                  "reference_audit": audit, "retraining_comparison": comparison,
                  "changes": forgetting_changes(problem.parent["metrics"]["models"]["model"], scores)})
        save_arrays(path / "references.npz", **asdict(refs), target=target)
        save_training_result(path, "model", outcome)
        save_arrays(path / "weights.npz", model=outcome.weights)
        save_arrays(path / "predictions.npz", model__test=probabilities, test_labels=labels,
                    test_sample_ids=problem.parent["predictions"]["test_sample_ids"])
        for name in ("preprocessing.joblib", "preprocessing.json"):
            shutil.copyfile(problem.parent["path"] / name, path / name)
        save_json(path / "datasets.json", {"parent_run": problem.parent["path"].relative_to(root).as_posix(),
                  "features": "features.npz in parent_run", "training_features_refitted": False})
        save_json(path / "resources.json", {**model.metadata(), "training_seconds": float(outcome.history.elapsed_seconds.max()),
                  "teacher_predictions_reused": True, "supervised_updates": 0})
        outcome.history.assign(model="model").to_csv(path / "history.csv", index=False)
        finish_run(path, "complete")
    except BaseException as error:
        finish_run(path, "failed", error=f"{type(error).__name__}: {error}")
        raise
    return load_run(path), "resumed" if resumed else "trained"


def json_normal(value):
    from src.artifacts import json_value
    return json_value(value)


def run_study(review, problems, execution, *, force_retrain=False, baseline_checks=True, progress=None,
              performance_directory=None, support_index=None, support_indexes=None):
    """Call inside ablation_execution. All futures finish/stop before that scope exits."""
    root = review.root
    environment, source = execution_environment(), source_fingerprint(root)
    protocol = {"version": 1, "original_report": review.report_path.relative_to(root).as_posix(),
                "original_source": review.source_fingerprint, "problems": [problem.key for problem in problems],
                "teachers": [problem.baseline["path"].name for problem in problems],
                "sweeps": SWEEPS, "seed": DEFAULT_SEED, "updates": 100,
                "code_fingerprint": source, "environment": environment, "execution": asdict(execution)}
    study_id = fingerprint(protocol)[:16]
    destination = root / "results/ablations" / study_id
    index_file = destination / "study.json"
    records = []
    index = {"protocol": protocol, "complete": False, "records": records, "baseline_checks": [],
             "support_index": None if support_index is None else Path(support_index).relative_to(root).as_posix(),
             "support_indexes": [Path(path).relative_to(root).as_posix() for path in (support_indexes or [])],
             "performance_directory": None if performance_directory is None else
                 Path(performance_directory).relative_to(root).as_posix()}
    session, lock = CheckpointSession(force_retrain=force_retrain), Lock()
    start = perf_counter()
    with AerBatchService(problems[0].config.circuit, problems[0].config.backend, execution) as service, GpuMonitor() as monitor:
        try:
            checks = []
            for problem in problems:
                print("Checking saved predictions/teachers/gradients:", problem.key, flush=True)
                checks.append(validate_execution(problem, service))
            save_json(destination / "numerical_compatibility.json", checks)
            if baseline_checks:
                selected = [p for p in problems if p.fold == min(p.fold for p in problems)]
                for problem in selected:
                    print("Default trajectory compatibility fit:", problem.key, flush=True)
                    settings = replace(problem.config.unlearning, methods=("guided",))
                    run, status = _run_variant(problem, settings, service, session, lock, root, environment, source, validation=True)
                    actual = run["predictions"]["model__test"]
                    expected = problem.baseline["predictions"]["model__test"]
                    error = float(np.max(np.abs(actual - expected)))
                    check = {"problem": problem.key, "run": run["path"].relative_to(root).as_posix(),
                             "status": status, "max_probability_error": error, "tolerance": 1e-6,
                             "passed": bool(error <= 1e-6)}
                    index["baseline_checks"].append(check)
                    save_json(index_file, index)
                    if not check["passed"]:
                        raise RuntimeError("runtime baseline trajectory differs from the historical baseline; "
                                           "inspect baseline_checks before running ablations.")
                    print("Default trajectory passed:", problem.key, flush=True)
            for problem in problems:
                for method, run in (("baseline", problem.baseline), ("uniform", problem.uniform)):
                    records.append({"problem": problem.key, "fold": problem.fold, "condition": problem.name,
                                    "parameter": method, "value": None, "status": problem.baseline_origin,
                                    "run": run["path"].relative_to(root).as_posix()})
            save_json(index_file, index)
            tasks = [(problem, parameter, value, setting) for problem in problems
                     for parameter, value, setting in parameter_settings(problem.config.unlearning)]
            # Interleave folds/conditions instead of finishing an entire fold first.
            tasks.sort(key=lambda task: (list(SWEEPS).index(task[1]), task[2], task[0].name, task[0].fold))
            cancellation = Event()
            with ThreadPoolExecutor(max_workers=execution.workers) as workers:
                pending = {workers.submit(_run_variant, problem, setting, service, session, lock, root,
                                          environment, source, cancellation=cancellation): (problem, parameter, value)
                           for problem, parameter, value, setting in tasks}
                failure = None
                try:
                    for future in as_completed(pending):
                        problem, parameter, value = pending[future]
                        if future.cancelled():
                            continue
                        try:
                            run, status = future.result()
                            records.append({"problem": problem.key, "fold": problem.fold, "condition": problem.name,
                                            "parameter": parameter, "value": value, "status": status,
                                            "run": run["path"].relative_to(root).as_posix()})
                            save_json(index_file, index)
                            if progress:
                                done = sum(record["parameter"] not in ("baseline", "uniform") for record in records)
                                progress({"completed": done, "planned": len(tasks), "problem": problem.key,
                                          "parameter": parameter, "value": value, "status": status,
                                          "elapsed_seconds": perf_counter() - start})
                        except BaseException as error:
                            failure = failure or error
                            cancellation.set()
                            for candidate in pending:
                                candidate.cancel()
                except BaseException:
                    cancellation.set()
                    for candidate in pending:
                        candidate.cancel()
                    raise
                if failure is not None:
                    raise failure
            index["complete"] = True
        finally:
            index.update(elapsed_seconds=perf_counter() - start, service_resources=service.metadata())
            save_json(index_file, index)
            if monitor.rows:
                pd.DataFrame(monitor.rows).to_csv(destination / "gpu_telemetry.csv", index=False)
    return index_file


def load_study(index_file, root):
    """Portable analysis after download; no current-environment matching or execution."""
    root = Path(root).resolve()
    index_file = Path(index_file).resolve()
    if not index_file.is_relative_to(root):
        raise ValueError("Study index must be inside the project.")
    index = load_json(index_file)
    rows = []
    teachers = dict(zip(index["protocol"]["problems"], index["protocol"]["teachers"]))
    for record in index["records"]:
        path = (root / record["run"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Run path leaves the project.")
        run = load_run(path)
        scores, metrics = run["metrics"]["models"]["model"], run["metrics"]
        settings = run["config"]["unlearning"]
        if record["parameter"] in DEFAULTS:
            differences = [key for key, value in DEFAULTS.items() if settings[key] != value]
            if differences != [record["parameter"]] or settings[record["parameter"]] != record["value"]:
                raise ValueError("Run configuration does not match its one-factor ablation label.")
            identity = run["manifest"]["identity"]
            if identity["teacher_run"] != teachers[record["problem"]]:
                raise ValueError("Ablation uses a different historical teacher.")
        elif record["parameter"] == "baseline" and path.name != teachers[record["problem"]]:
            raise ValueError("Guided baseline does not match the registered teacher run.")
        rows.append({**record, "run_id": path.name, "seed": DEFAULT_SEED,
                     **{key: settings[key] for key in (*DEFAULTS, "anchor_fraction")},
                     **{key: scores[key] for key in ("retained_accuracy", "forget_recall", "forget_confidence")},
                     **{f"recall_digit_{digit}": scores["recall_by_class"][str(digit)] for digit in range(3)},
                     **{key: metrics["retraining_comparison"][key] for key in ("full_kl_mean", "conditional_kl_mean")},
                     **{f"target_{digit}": value for digit, value in enumerate(metrics["target"])}})
    columns = ["problem", "fold", "condition", "parameter", "value", "status", "run", "run_id", "seed",
               *DEFAULTS, "anchor_fraction", "retained_accuracy", "forget_recall", "forget_confidence",
               "recall_digit_0", "recall_digit_1", "recall_digit_2", "full_kl_mean", "conditional_kl_mean",
               "target_0", "target_1", "target_2"]
    return index, pd.DataFrame(rows, columns=columns)
