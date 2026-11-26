"""CPU fold orchestration with immutable completed-model imports.

Device changes never convert an Adam checkpoint. Completed models retain their
actual identities; unfinished source fits stay in the original recovery store.
All new/unfinished CPU fits use the unchanged strict scientific checkpointing.
"""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import RLock
from unittest.mock import patch
import hashlib
import os

from src import artifacts, checkpoints, noisy_training
from src.config import DEFAULT_SEED
from src.models.noise import NoiseSetting
from src.models.parallel_aer import AerExecution
from tools.noisy_drive import DriveRecovery, STAGES, _read_verified, publish
from tools.noisy_workflow import locate_index
from tools.parallel_folds import (COMMON, FOLDS, _check_protocol, _copy_files, _fold,
    _inputs, _json_publish, _local, _protocol, _verified_json, collect_parallel,
    fold_recovery, select_review_fold)


CPU_NAMESPACE = "parallel_cpu"
HANDOFF = "results/noisy_training/handoff"
# These affect execution cost/device only. All other backend fields, including
# shots, precision/method, synthetic channel settings and transpilation, match.
RUNTIME_FIELDS = ("device", "allow_cpu_fallback", "circuit_batch_size",
                  "max_memory_mb", "max_parallel_threads")


def scientific_identity(identity):
    """Completed-model comparison only; never used for optimizer continuation."""
    value = deepcopy(artifacts.json_value(identity))
    value.pop("environment", None)
    value.pop("execution", None)
    backend = value["config"]["backend"]
    for key in RUNTIME_FIELDS:
        backend.pop(key, None)
    return value


def allocated_cores():
    """Respect CPU affinity and Linux cgroup quotas rather than host core count."""
    limits = [os.cpu_count() or 1]
    if hasattr(os, "sched_getaffinity"):
        limits.append(len(os.sched_getaffinity(0)))
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            limits.append(max(1, int(quota) // int(period)))
    except (OSError, ValueError):
        try:
            quota = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
            period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
            if quota > 0:
                limits.append(max(1, quota // period))
        except (OSError, ValueError):
            pass
    return max(1, min(limits))


def cpu_candidates(cores=None):
    """Two or three bounded choices; one Aer owner shares the allocated cores."""
    cores = allocated_cores() if cores is None else max(1, int(cores))
    memory = 4096
    try:
        available = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        limit = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        if limit != "max":
            available = min(available, int(limit))
        memory = max(256, min(16384, int(available / 1024**2 * .65)))
    except (AttributeError, OSError, ValueError):
        pass
    workers = sorted({1, min(2, cores), min(4, cores)})
    return [AerExecution(device="CPU", workers=w, threads=max(1, cores - min(w - 1, 2)),
               circuit_batch_size=1024 if w == 1 else 2048, memory_mb=memory,
               runtime_binding=False, gpu_parameter_batching=False, gather_ms=2)
            for w in workers]


def _source_store(drive, fold):
    # Once a GPU fold handoff exists its folds own the latest progress. Never
    # mix an older serial snapshot with a newer fold snapshot.
    plan = drive / "parallel/plan.json"
    if plan.exists():
        record = _verified_json(plan)
        if not record["ready"] or fold not in record["folds"]:
            raise ValueError("The GPU fold handoff is incomplete or omits a requested fold.")
        store = drive / "parallel" / f"fold_{fold}"
        if not (store / "handoff.json").is_file() or not any(
                (store / name).is_file() for name in ("recovery/state.zip", "recovery/state.zip.previous")):
            raise ValueError("An assigned GPU fold store is missing; no replacement CPU fitting was authorized by missing files.")
        return store
    return drive


def prepare_cpu_parallel(review, drive_root, *, folds=FOLDS, setting=NoiseSetting("combined", .01),
                         restart_unfinished=False, legacy_index=None, emit=print):
    """Read stopped GPU recovery; copy complete models into separate CPU stores.

    restart_unfinished must be explicitly True to start replacements for partial
    GPU fits. Their original checkpoint bytes remain untouched at the source.
    Repeating setup retains the CPU plan and all subsequent CPU progress.
    """
    drive = Path(drive_root).resolve()
    folds = tuple(_fold(f) for f in folds)
    if not folds or len(set(folds)) != len(folds):
        raise ValueError("Select unique folds.")
    problems, historical, bindings = _inputs(review, folds)
    base = {"version": 1, "folds": list(folds), "device": "CPU",
            "code_fingerprint": artifacts.source_fingerprint(review.root),
            "original_source": review.source_fingerprint, "training_noise": asdict(setting),
            "seed": DEFAULT_SEED,
            "historical_inputs_fingerprint": artifacts.fingerprint({k: m.identity(review.root) for k, m in historical.items()})}
    plan_file = drive / CPU_NAMESPACE / "plan.json"
    if plan_file.exists():
        plan = _verified_json(plan_file)
        _check_protocol(plan, base)
        emit("Existing CPU plan retained; no CPU progress was reset.")
        return plan
    plan_id = artifacts.fingerprint(base)[:16]
    coverage = []
    with TemporaryDirectory(prefix="qml-cpu-handoff-") as temporary:
        # Restore a shared serial source once, or each separate fold store once.
        restored = {}
        for fold in folds:
            selected = select_review_fold(review, fold)
            fp, models, fb = _inputs(selected, (fold,))
            source_drive = _source_store(drive, fold)
            if source_drive not in restored:
                source_root = Path(temporary) / f"source_{len(restored)}"
                source_root.mkdir()
                previous = DriveRecovery(source_root, source_drive, emit=emit)
                previous.restore(read_only=True)
                source_index = locate_index(source_root, legacy_index)
                source = artifacts.load_json(source_index) if source_index else None
                if source:
                    common = {k: source["protocol"][k] for k in COMMON}
                    _check_protocol(common, {k: base[k] for k in ("code_fingerprint", "training_noise", "seed")})
                    source_folds = sorted({r["fold"] for r in source["protocol"]["bindings"]})
                    sp, sh, sb = _inputs(review, source_folds)
                    _check_protocol(source["protocol"], _protocol(review, sp, sh, sb, common))
                restored[source_drive] = source_root, previous, source_index, source
            source_root, previous, source_index, source = restored[source_drive]
            target = drive / CPU_NAMESPACE / f"fold_{fold}"
            if (target / "handoff.json").exists():
                done = _verified_json(target / "handoff.json")
                if done["plan_id"] != plan_id:
                    raise ValueError("Another CPU handoff owns this fold; no files overwritten.")
                coverage.append(done["initial_coverage"])
                continue
            marker = target / "preparation.json"
            if marker.exists() and _verified_json(marker)["plan_id"] != plan_id:
                raise ValueError("Conflicting CPU preparation; no files overwritten.")
            original_keys = {p.original["path"].name for p in fp} | {p.reference["path"].name for p in fp}
            unlearn_keys = {f"{p.key}_{method}" for p in fp for method in ("guided", "uniform")}
            complete, partial = [], []
            for stage in STAGES:
                for file in sorted((source_root / "results/runs" / stage).glob("*/manifest.json")):
                    manifest = artifacts.load_json(file)
                    identity = manifest["identity"]
                    if identity["config"]["data"]["fold"] != fold:
                        continue
                    if source is None:
                        raise ValueError("Source models exist without a verified study identity.")
                    _check_protocol(identity, {k: source["protocol"][k] for k in COMMON})
                    allowed = (identity.get("historical_problem") in original_keys if stage == "noisy_supervised"
                               else f"{identity.get('problem')}_{identity.get('method')}" in unlearn_keys)
                    if not allowed:
                        raise ValueError("Unexpected source model in CPU handoff.")
                    if manifest["status"] == "complete":
                        artifacts.load_run(file.parent)
                        complete.append(file.parent)
                    else:
                        checkpoint = file.parent / "checkpoints/model.resume.json"
                        partial.append({"run": file.parent.relative_to(source_root).as_posix(),
                            "manifest_sha256": artifacts._file_digest(file),
                            "step": artifacts.load_json(checkpoint)["step"] if checkpoint.exists() else 0,
                            "checkpoint_sha256": artifacts._file_digest(checkpoint) if checkpoint.exists() else None})
            if partial and not restart_unfinished:
                raise ValueError(f"Fold {fold} has {len(partial)} unfinished source fits. Their Adam state cannot resume on CPU. "
                    "Keep the matching GPU runtime, or deliberately set RESTART_UNFINISHED_ON_CPU=True. Original GPU files stay untouched.")
            _json_publish(marker, {"plan_id": plan_id, "fold": fold})
            seed_root = Path(temporary) / f"cpu_{fold}"
            seed_root.mkdir()
            recovery = DriveRecovery(seed_root, target, emit=emit)
            imported = {}
            for path in complete:
                relative = path.relative_to(source_root).as_posix()
                _copy_files(path, seed_root / relative)
                record = previous.catalog["runs"].get(relative)
                if record is None:
                    raise ValueError("Completed source model has no verified archive.")
                content = _read_verified(_local(source_drive, record["archive"]))
                if hashlib.sha256(content).hexdigest() != record["sha256"]:
                    raise ValueError("Completed source archive checksum changed.")
                publish(_local(target, record["archive"]), content)
                recovery.catalog["runs"][relative] = record
                imported[relative] = {"manifest_sha256": artifacts._file_digest(path / "manifest.json"),
                    "identity": artifacts.load_json(path / "manifest.json")["identity"], "archive_sha256": record["sha256"]}
            # Reuse exact teachers only for an imported original. This allows
            # completed guided fits to retain their exact fixed-teacher identity.
            if source_index:
                _copy_files(source_index, seed_root / HANDOFF / "source_index.json")
                for p in fp:
                    teacher = source_index.parent / "teachers" / f"{p.key}.json"
                    if teacher.exists():
                        _copy_files(teacher, seed_root / HANDOFF / "teachers" / teacher.name)
            audit = {**base, "plan_id": plan_id, "fold": fold,
                     "fold_inputs_fingerprint": artifacts.fingerprint({k: m.identity(review.root) for k, m in models.items()}),
                     "source_store": str(source_drive), "source_index_sha256": artifacts._file_digest(source_index) if source_index else None,
                     "imported_completed": imported, "preserved_unfinished": partial,
                     "restart_unfinished": bool(restart_unfinished),
                     "policy": "Reuse immutable completed fits only. New CPU fits initialize normally; no GPU Adam state is converted."}
            artifacts.save_json(seed_root / HANDOFF / "cpu.json", audit)
            recovery.snapshot()
            initial = {"fold": fold, "planned": len(noisy_training.training_plan(fp)),
                       "completed": len(complete), "preserved_gpu_partial": len(partial),
                       "preserved_gpu_updates": sum(r["step"] for r in partial)}
            _json_publish(target / "handoff.json", {"plan_id": plan_id, "fold": fold,
                "code_fingerprint": base["code_fingerprint"], "initial_coverage": initial})
            coverage.append(initial)
            emit(f"CPU fold {fold}: reuse {len(complete)} completed models; preserve {len(partial)} unfinished GPU fits at their source.")
    plan = {**base, "plan_id": plan_id, "ready": True, "initial_coverage": coverage}
    _json_publish(plan_file, plan)
    return plan


def cpu_fold_recovery(runtime_root, drive_root, fold, *, emit=print):
    return fold_recovery(runtime_root, drive_root, fold, emit=emit, namespace=CPU_NAMESPACE)


def _record_support(root):
    source = Path(__file__).resolve()
    digest = artifacts._file_digest(source)
    directory = root / "results/noisy_training/runtime_support/cpu_folds" / digest[:16]
    _copy_files(source, directory / "cpu_folds.py")
    artifacts.save_json(directory / "provenance.json", {
        "helper_sha256": digest, "scientific_source": artifacts.source_fingerprint(root),
        "environment": noisy_training.execution_environment(),
        "scope": "CPU execution selection and verified immutable completed-model reuse; no cross-device Adam continuation."})


def prepare_cpu_execution(review, problems, historical, bindings, setting, *, selected_index=None):
    """Resume a CPU execution exactly, or measure a bounded core-aware choice."""
    audit = artifacts.load_json(review.root / HANDOFF / "cpu.json")
    if ({p.fold for p in problems} != {audit["fold"]}
            or audit["fold_inputs_fingerprint"] != artifacts.fingerprint({k: m.identity(review.root) for k, m in historical.items()})):
        raise ValueError("CPU fold membership differs from its handoff.")
    _record_support(review.root)
    if selected_index or locate_index(review.root):
        from tools.noisy_training_runtime import prepare_noisy_execution
        result = prepare_noisy_execution(review, problems, historical, bindings, setting,
            selected_index=selected_index, require_gpu_device=False)
        if result[0].device != "CPU" or result[0].threads > allocated_cores():
            raise ValueError("The saved CPU execution needs a matching CPU allocation; its settings were not changed.")
        return result
    imported = audit["imported_completed"]
    for stage in STAGES:
        for file in (review.root / "results/runs" / stage).glob("*/manifest.json"):
            relative = file.parent.relative_to(review.root).as_posix()
            if relative not in imported or artifacts._file_digest(file) != imported[relative]["manifest_sha256"]:
                raise ValueError("Unindexed CPU fits exist. Preserve them and recover their study index before continuing.")
    candidates = cpu_candidates()
    print(f"CPU allocation: {allocated_cores()} cores; checking {len(candidates)} bounded worker/thread settings.", flush=True)
    return noisy_training.benchmark_noisy_training(problems[0], setting, review.root,
        candidates=candidates, minimum_workers=1, require_gpu_device=False, probe_steps=1)


@contextmanager
def cpu_completed_models(root, *, audit_name="cpu.json", usage_name="usage.json"):
    """Reuse verified completed imports, never a partial or a changed problem."""
    root = Path(root).resolve()
    audit = artifacts.load_json(root / HANDOFF / audit_name)
    inventory = audit["imported_completed"]
    imported = {}
    for relative, entry in inventory.items():
        path = _local(root, relative)
        if artifacts._file_digest(path / "manifest.json") != entry["manifest_sha256"]:
            raise ValueError("Imported completed-model manifest changed.")
        run = artifacts.load_run(path)
        if run["manifest"]["identity"] != entry["identity"]:
            raise ValueError("Imported completed-model identity changed.")
        key = artifacts.fingerprint(scientific_identity(entry["identity"]))
        if key in imported:
            raise ValueError("Ambiguous completed attempts; select a single source attempt.")
        imported[key] = run
    original_open, original_teacher = checkpoints.CheckpointSession.open, noisy_training._teacher
    lock = RLock()
    usage_file = root / HANDOFF / usage_name
    usage = artifacts.load_json(usage_file) if usage_file.exists() else {}

    def open_completed(session, directory, identity, **kwargs):
        directory = Path(directory).resolve()
        permitted = directory in {root / "results/runs" / s for s in STAGES}
        if permitted and not session.force_retrain and kwargs.get("reuse", True):
            run = imported.get(artifacts.fingerprint(scientific_identity(identity)))
            if run is not None:
                if run["path"].parent != directory:
                    raise ValueError("Imported model stage differs.")
                with lock:
                    usage[run["path"].relative_to(root).as_posix()] = {"requested_identity": artifacts.json_value(identity)}
                    artifacts.save_json(usage_file, usage)
                return run["path"], run
        return original_open(session, directory, identity, **kwargs)

    def frozen_teacher(problem, original, service, directory):
        destination = directory / "teachers" / f"{problem.key}.json"
        source = root / HANDOFF / "teachers" / destination.name
        if not destination.exists() and source.exists() and original["path"].relative_to(root).as_posix() in inventory:
            saved = artifacts.load_json(source)
            expected = {"original": original["path"].name, "original_artifacts": original["manifest"]["artifact_sha256"],
                "membership": {k: getattr(problem.membership, k) for k in ("forget_indices", "anchor_indices", "calibration_indices")}}
            if saved["identity"] != artifacts.fingerprint(expected) or artifacts.fingerprint(saved["references"]) != saved["references_fingerprint"]:
                raise ValueError("Imported noisy teacher identity changed.")
            _copy_files(source, destination)
        return original_teacher(problem, original, service, directory)

    with patch.object(checkpoints.CheckpointSession, "open", open_completed), patch.object(noisy_training, "_teacher", frozen_teacher):
        yield


def _validate_import(root, run, index):
    audit = artifacts.load_json(root / HANDOFF / "cpu.json")
    relative = run["path"].relative_to(root).as_posix()
    entry = audit["imported_completed"].get(relative)
    if entry is None:
        return False
    if artifacts._file_digest(run["path"] / "manifest.json") != entry["manifest_sha256"] or run["manifest"]["identity"] != entry["identity"]:
        raise ValueError("Completed import provenance changed before collection.")
    usage = artifacts.load_json(root / HANDOFF / "usage.json")
    request = usage[relative]["requested_identity"]
    _check_protocol(request, {k: index["protocol"][k] for k in COMMON})
    if scientific_identity(request) != scientific_identity(entry["identity"]):
        raise ValueError("Completed import does not match the CPU scientific problem.")
    return True


def collect_cpu_parallel(review, drive_root, **kwargs):
    return collect_parallel(review, drive_root, namespace=CPU_NAMESPACE,
        imported_run_validator=_validate_import, **kwargs)
