"""Visible progress and less storage I/O around the unchanged noisy trainer.

This execution-only helper deliberately lives outside src/. It does not change
the scientific-source fingerprint or any optimizer/checkpoint identity. Its own
source hash and source copy are recorded separately under runtime_support.
It never changes a gradient, parameter, RNG stream, fit budget or cache identity.
"""

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Lock, Thread, local
from time import monotonic
from unittest.mock import patch
from uuid import uuid4
import functools
import hashlib
import json
import os
import sys
import traceback

import pandas as pd

from src import artifacts, checkpoints, noisy_training
from src.config import DEFAULT_SEED
from src.execution_resources import aer_cpu_threads, memory_budget_mb
from src.models.parallel_aer import AerExecution, gpu_sample


def _inside(root, path):
    path = Path(path)
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(root):
        raise ValueError("Saved study path leaves the project.")
    return path


def completed_fresh_index(root, selected=None, *, label="feature_randomization_v1"):
    """Load-only discovery: ambiguity or missing inputs stops, never starts fitting."""
    root = Path(root).resolve()
    paths = ([_inside(root, selected)] if selected else
             sorted((root / "results/fresh_studies").glob("*/study.json")))
    matches = []
    for path in paths:
        record = artifacts.load_json(path)
        if record.get("complete") is True and record.get("protocol", {}).get("label") == label:
            matches.append(path)
    if len(matches) != 1:
        choices = [str(path.relative_to(root)) for path in matches]
        raise ValueError("Set FRESH_STUDY_INDEX to one completed fresh study; "
                         f"found {len(matches)} matches: {choices}. No fitting was started.")
    return matches[0]


def completed_extension_index(review, stage, selected=None):
    """Relink completed optional results without executing their earlier stages."""
    if stage not in ("ablations", "noise_studies"):
        raise ValueError("Unknown saved extension stage.")
    root = review.root.resolve()
    paths = ([_inside(root, selected)] if selected else
             sorted((root / "results" / stage).glob("*/study.json")))
    matches = []
    for path in paths:
        record = artifacts.load_json(path)
        protocol = record.get("protocol", {})
        if (record.get("complete") is True
                and protocol.get("original_report") == review.report_path.relative_to(root).as_posix()
                and protocol.get("original_source") == review.source_fingerprint):
            matches.append(path)
    if len(matches) > 1 or (selected and not matches):
        raise ValueError(f"Select one completed {stage} index belonging to the selected fresh study: "
                         + str([p.relative_to(root).as_posix() for p in matches]))
    return matches[0] if matches else None


def prepare_noisy_execution(review, problems, historical, bindings, setting, *, selected_index=None,
                            require_gpu_device=True):
    """Pin a compatible existing execution; otherwise benchmark one conservative mode.

Changing workers/memory/binding on an unfinished model changes its strict identity.
We therefore recover the old setting before considering a fresh benchmark.
"""
    root = review.root.resolve()
    expected = {"code_fingerprint": artifacts.source_fingerprint(root),
                "environment": noisy_training.execution_environment(), "training_noise": asdict(setting),
                "seed": DEFAULT_SEED, "original_report": review.report_path.relative_to(root).as_posix(),
                "original_source": review.source_fingerprint,
                "problems": [p.key for p in problems], "bindings": bindings.to_dict("records"),
                "historical_inputs": {key: model.identity(root) for key, model in historical.items()}}
    paths = ([_inside(root, selected_index)] if selected_index else
             sorted((root / "results/noisy_training").glob("*/study.json")))
    matching = []
    for path in paths:
        record = artifacts.load_json(path)
        protocol = record.get("protocol", {})
        if all(artifacts.fingerprint(protocol.get(key)) == artifacts.fingerprint(value)
               for key, value in expected.items()):
            matching.append((path, record))
    if len(matching) > 1:
        raise ValueError("Multiple compatible noisy attempts exist. Set FRESH_NOISY_INDEX explicitly: "
                         + str([p.relative_to(root).as_posix() for p, _ in matching]))
    if selected_index and not matching:
        raise ValueError("Selected noisy study differs in source/runtime/inputs/noise. "
                         "Restore its matching environment; no new fitting was started.")
    if matching:
        path, record = matching[0]
        execution = AerExecution(**record["protocol"]["execution"])
        print("Preserving the exact noisy checkpoint execution:", asdict(execution), flush=True)
        performance = record.get("performance_directory")
        if performance:
            directory = _inside(root, performance)
            file = directory / "benchmark.json"
            if file.is_file():
                benchmark = artifacts.load_json(file)
                identity = benchmark.get("identity", {})
                passed = any(row.get("status") == "passed"
                             and all(row.get(key) == value for key, value in asdict(execution).items())
                             for row in benchmark.get("candidates", []))
                if (benchmark.get("selected") == asdict(execution)
                        and passed
                        and identity.get("source") == expected["code_fingerprint"]
                        and identity.get("environment") == expected["environment"]
                        and identity.get("noise") == artifacts.json_value(setting)
                        and identity.get("original_problem") == problems[0].original["path"].name):
                    print("Loaded the already-passing noisy benchmark; no probes repeated.", flush=True)
                    return execution, pd.DataFrame(benchmark["candidates"]), directory
        # Missing benchmark metadata does not change the pinned execution.
        return noisy_training.benchmark_noisy_training(problems[0], setting, root, candidates=[execution],
            minimum_workers=min(5, execution.workers), require_gpu_device=require_gpu_device)
    existing_models = any(next((root / "results/runs" / stage).glob("*/manifest.json"), None)
                          for stage in ("noisy_supervised", "noisy_unlearning"))
    if paths or existing_models:
        raise ValueError("Existing noisy studies/checkpoints do not match this source, runtime or inputs. "
                         "Do not retrain blindly: restore the original settings/environment and select its index.")
    execution = AerExecution(device="GPU" if require_gpu_device else "CPU",
        circuit_batch_size=8192 if require_gpu_device else 2048,
        workers=8 if require_gpu_device else min(2, os.cpu_count() or 1),
        threads=aer_cpu_threads() if require_gpu_device else 1,
        memory_mb=memory_budget_mb(gpu_sample()) if require_gpu_device else 4096,
        runtime_binding=False, gpu_parameter_batching=False)
    print("No noisy checkpoints found. Testing one conservative parallel execution.", flush=True)
    return noisy_training.benchmark_noisy_training(problems[0], setting, root, candidates=[execution],
        minimum_workers=5 if require_gpu_device else 1, require_gpu_device=require_gpu_device)


class _ManifestCatalog:
    """Read existing noisy manifests once, retaining exact matching and hash checks."""

    def __init__(self, root, announce):
        self.allowed = {root / "results/runs" / stage for stage in ("noisy_supervised", "noisy_unlearning")}
        self.original = checkpoints.matching_manifests
        self.announce, self.cache, self.lock = announce, {}, Lock()

    def __call__(self, directory, identity):
        directory = Path(directory).resolve()
        if directory not in self.allowed:
            return self.original(directory, identity)
        with self.lock:
            if directory not in self.cache:
                self.announce(f"Indexing existing checkpoints once: {directory.name}")
                files = sorted(directory.glob("*/manifest.json"))
                with ThreadPoolExecutor(max_workers=4) as readers:
                    records = list(readers.map(artifacts.load_json, files))
                self.cache[directory] = [(file.parent, record) for file, record in zip(files, records)]
                self.announce(f"Indexed {len(files)} existing {directory.name} checkpoints.")
            matches = [(path, record) for path, record in self.cache[directory]
                       if record.get("status") != "retired" and record.get("identity") == artifacts.json_value(identity)]
        # CheckpointSession still reads the selected current manifest, verifies
        # completed artifact hashes, and checks the full optimizer identity.
        return sorted(matches, key=lambda item: item[1]["created_utc"], reverse=True)


class _Progress:
    def __init__(self, root, heartbeat_seconds, emit):
        if heartbeat_seconds <= 0:
            raise ValueError("heartbeat_seconds must be positive.")
        self.root, self.interval = root, heartbeat_seconds
        self.emit = emit or (lambda message: print(message, flush=True))
        self.lock, self.stop, self.context = Lock(), Event(), local()
        self.active, self.finished, self.phase = {}, 0, "preparing study metadata"
        self.services = []
        self.started = monotonic()
        self.audit = None

    def message(self, message):
        self.emit(f"[noisy {monotonic() - self.started:.0f}s] {message}")

    @contextmanager
    def job(self, key, fold, updates):
        entry = {"key": key, "fold": fold, "total_updates": updates, "step": None,
                 "phase": "checkpoint selection / source snapshot", "started": monotonic(), "mirror": None}
        self.context.entry = entry
        with self.lock:
            self.active[key] = entry
        self.message(f"START fold{fold} {key}: selecting/checking checkpoint.")
        try:
            yield entry
        finally:
            with self.lock:
                self.active.pop(key, None)
            self.context.entry = None

    def snapshot(self):
        with self.lock:
            entries = [dict(row) for row in self.active.values()]
            finished, phase = self.finished, self.phase
        rows = []
        for entry in entries:
            mirror = entry.pop("mirror")
            step = entry["step"]
            if mirror is not None:
                # On Windows a concurrent reader can block atomic replacement
                # of the optimizer file. Observe committed callback steps there;
                # POSIX permits the finer per-update runtime-local read below.
                if os.name != "nt":
                    try:
                        saved = json.loads(mirror.local.read_text(encoding="utf-8"))
                        step = saved["step"]
                    except (OSError, ValueError, KeyError):
                        pass
                entry["checkpoint"] = mirror.durable.relative_to(self.root).as_posix()
                entry["backup_error"] = None if mirror.error is None else str(mirror.error)
            started = entry.pop("started")
            rows.append({**entry, "step": step, "elapsed_seconds": monotonic() - started})
        return {"elapsed_seconds": monotonic() - self.started, "completed_fits": finished,
                "phase": phase, "active": rows, "aer_services": [
                    {"device": service.executor.actual_device,
                     "worker_thread_alive": service.thread.is_alive(),
                     "closed": service.closed, "queued_requests": service.queue.qsize(),
                     "merged_batches": service.batches, "physical_jobs": service.executor.stats["jobs"]}
                    for service in self.services]}

    def save_thread_diagnostic(self):
        """Locate a possible stall without cancelling fits or changing checkpoints."""
        text = "\n\n".join(f"THREAD {ident}\n" + "".join(traceback.format_stack(frame))
                           for ident, frame in sys._current_frames().items())
        if self.audit is not None:
            name = datetime.now(timezone.utc).strftime("thread_stacks_%Y%m%dT%H%M%SZ.txt")
            (self.audit / name).write_text(text, encoding="utf-8")

    def _heartbeat(self):
        previous, changed, warned = None, monotonic(), False
        while not self.stop.wait(self.interval):
            state = self.snapshot()
            signature = artifacts.fingerprint({key: state[key] for key in ("completed_fits", "phase", "aer_services")}
                | {"active": [{key: r[key] for key in ("key", "step", "phase")} for r in state["active"]]})
            if signature != previous:
                previous, changed, warned = signature, monotonic(), False
            state["seconds_without_observed_progress"] = monotonic() - changed
            details = "; ".join(f"fold{r['fold']} {r['step'] if r['step'] is not None else '?'}/{r['total_updates']} "
                                f"({r['phase']})" for r in state["active"])
            self.message(f"HEARTBEAT: {state['completed_fits']} fits completed, {len(state['active'])} active. "
                         + (details or state["phase"]))
            for service in state["aer_services"]:
                if not service["closed"]:
                    self.message(f"AER {service['device']}: {service['physical_jobs']} completed jobs, "
                                 f"{service['queued_requests']} queued; service alive={service['worker_thread_alive']}")
            if self.audit is not None:
                try:
                    # Disk I/O is outside the training-state lock and never
                    # blocks optimizer callbacks on a mounted-storage write.
                    artifacts.save_json(self.audit / "latest_progress.json", state)
                    if state["seconds_without_observed_progress"] >= 600 and not warned:
                        self.message("POSSIBLE STALL: steps and Aer job counters have not changed for 10 minutes. "
                                     "Saving thread stacks in runtime_support; no models were cancelled or reset.")
                        self.save_thread_diagnostic()
                        warned = True
                except OSError as error:
                    self.message(f"Progress-report write failed: {error}; model backup status is separate.")


@contextmanager
def noisy_training_monitor(root, *, heartbeat_seconds=30, emit=None):
    """Observe existing callbacks; cache metadata/source I/O without altering fits.

Use INSIDE noisy_training_execution. Keep this context alive until workers stop.
All original patched functions are restored on exit, including interruptions.
"""
    root = Path(root).resolve()
    progress = _Progress(root, heartbeat_seconds, emit)
    module = noisy_training
    supervised, unlearning, cancel = module._supervised, module._unlearning, module._cancel_callback
    save_fit, teacher, evaluation = module._save_fit, module._teacher, module._evaluation
    service_init = module.NoisyBatchService.__init__
    snapshot = module.save_source_snapshot
    helper = Path(__file__).read_bytes()
    helper_hash = hashlib.sha256(helper).hexdigest()
    session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    progress.audit = root / "results/noisy_training/runtime_support" / session_id
    progress.message("Preparing progress observer and one reusable source archive; optimizer code is unchanged.")
    with TemporaryDirectory(prefix="qml-source-cache-") as temporary:
        archive = Path(temporary) / "source.zip"
        snapshot(archive)
        archive_bytes = archive.read_bytes()
        progress.audit.mkdir(parents=True, exist_ok=True)
        (progress.audit / "noisy_training_runtime.py").write_bytes(helper)
        artifacts.save_json(progress.audit / "provenance.json", {
            "helper_sha256": helper_hash, "scientific_source_fingerprint": artifacts.source_fingerprint(root),
            "scientific_source_identity_changed": False, "optimizer_or_execution_settings_changed": False,
            "scope": "Progress/heartbeat, cached source ZIP and exact-match manifest lookup only; not new research results."})

        @functools.wraps(supervised)
        def observed_supervised(source, arrays, service, session, lock, root, common, cancellation):
            key = "supervised_" + source["path"].name
            with progress.job(key, source["config"]["data"]["fold"], source["config"]["training"]["steps"]):
                result = supervised(source, arrays, service, session, lock, root, common, cancellation)
                with progress.lock:
                    progress.finished += 1
                progress.message(f"DONE {key}: {result[1]}")
                return result

        @functools.wraps(service_init)
        def observed_service_init(service, *args, **kwargs):
            progress.phase = "initializing Aer density-matrix service"
            progress.message(progress.phase)
            service_init(service, *args, **kwargs)
            progress.services.append(service)
            progress.phase = "Aer ready; selecting noisy checkpoints"
            progress.message(progress.phase)

        @functools.wraps(unlearning)
        def observed_unlearning(problem, method, *args, **kwargs):
            key = problem.key + "_" + method
            with progress.job(key, problem.fold, problem.config.unlearning.training.steps):
                result = unlearning(problem, method, *args, **kwargs)
                with progress.lock:
                    progress.finished += 1
                progress.message(f"DONE {key}: {result[1]}")
                return result

        def observed_callback(cancellation, mirror):
            original = cancel(cancellation, mirror)
            entry = getattr(progress.context, "entry", None)
            if entry is not None:
                with progress.lock:
                    entry.update(mirror=mirror, phase="optimizer")
                progress.message(f"CHECKPOINT {entry['key']}: {mirror.durable.relative_to(root)}")
            def report(row):
                if entry is not None and row["step"] == 0:
                    with progress.lock:
                        entry["phase"] = "initial storage backup verification"
                    progress.message(f"INITIAL STATE saved locally for {entry['key']}; verifying storage backup.")
                original(row)  # Existing cancellation/storage checks stay mandatory.
                if entry is not None:
                    with progress.lock:
                        entry.update(step=row["step"], phase="optimizer")
                    loss = row.get("objective", row.get("validation_loss"))
                    progress.message(f"STEP fold{entry['fold']} {entry['key']}: "
                                     f"{row['step']}/{entry['total_updates']}; evaluation loss={loss}")
            return report

        @functools.wraps(save_fit)
        def observed_save(*args, **kwargs):
            entry = getattr(progress.context, "entry", None)
            if entry is not None:
                with progress.lock:
                    entry["phase"] = "test evaluation / artifact saving and verification"
            return save_fit(*args, **kwargs)

        @functools.wraps(teacher)
        def observed_teacher(problem, *args, **kwargs):
            progress.phase = "frozen noisy teacher: " + problem.key
            progress.message(progress.phase)
            return teacher(problem, *args, **kwargs)

        @functools.wraps(evaluation)
        def observed_evaluation(file, *args, **kwargs):
            progress.phase = "deployment evaluation: " + Path(file).name
            return evaluation(file, *args, **kwargs)

        def cached_source(path):
            from src.checkpoint_storage import _atomic_bytes
            _atomic_bytes(path, archive_bytes)

        with ExitStack() as stack:
            for name, replacement in (("_supervised", observed_supervised), ("_unlearning", observed_unlearning),
                    ("_cancel_callback", observed_callback), ("_save_fit", observed_save),
                    ("_teacher", observed_teacher), ("_evaluation", observed_evaluation),
                    ("save_source_snapshot", cached_source)):
                stack.enter_context(patch.object(module, name, replacement))
            stack.enter_context(patch.object(checkpoints, "matching_manifests", _ManifestCatalog(root, progress.message)))
            stack.enter_context(patch.object(module.NoisyBatchService, "__init__", observed_service_init))
            thread = Thread(target=progress._heartbeat, name="noisy-progress-heartbeat", daemon=True)
            thread.start()
            try:
                progress.message("Observer ready. Progress reports committed optimizer steps, not model completions only.")
                yield progress
            finally:
                progress.stop.set()
                thread.join(timeout=5)
                artifacts.save_json(progress.audit / "final_progress.json", progress.snapshot())
                progress.message("Observer stopped; completed models and committed optimizer checkpoints are preserved.")


def include_runtime_support(archive, root):
    """Carry this separately identified execution helper and its audit in the ZIP."""
    from zipfile import ZipFile, ZIP_DEFLATED
    root = Path(root).resolve()
    files = list((root / "tools").glob("*.py"))
    files += [p for p in (root / "results/noisy_training/runtime_support").rglob("*") if p.is_file()]
    with ZipFile(_inside(root, archive), "a", ZIP_DEFLATED) as bundle:
        existing = set(bundle.namelist())
        for file in sorted(files):
            file = _inside(root, file)
            name = file.relative_to(root).as_posix()
            if name not in existing:
                bundle.write(file, name)
                existing.add(name)
