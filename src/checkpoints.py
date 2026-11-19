"""Select per-model runs and preserve the state needed to resume local training."""

from dataclasses import dataclass, field
from pathlib import Path
import re

from src.artifacts import (create_run, fingerprint, json_value, load_json,
                           load_run, save_json)


def matching_manifests(directory, identity):
    """Newest attempt first, independent of random directory-name suffixes."""
    matches = []
    for file in Path(directory).glob("*/manifest.json"):
        manifest = load_json(file)
        if manifest.get("status") != "retired" and manifest.get("identity") == json_value(identity):
            matches.append((file.parent, manifest))
    return sorted(matches, key=lambda item: item[1]["created_utc"], reverse=True)


@dataclass
class CheckpointSession:
    """A notebook execution session; force each distinct fitting problem only once.

    Recreate this object when starting another forced experiment. Sharing it
    across conditions avoids retraining the same clean reference repeatedly.
    Existing completed run folders remain immutable research records. The
    active_checkpoints.json index is replaced atomically as active models change.
    """

    force_retrain: bool = False
    _runs: dict = field(default_factory=dict, init=False, repr=False)

    def owns(self, run):
        return run is not None and Path(run["path"]).resolve() in self._runs.values()

    def open(self, directory, identity, *, allow_execution, reuse=True, slot=None):
        """Return (path, completed_result), or (None, None) when execution is off."""
        directory = Path(directory).resolve()
        key = (str(directory), fingerprint(identity))
        path = self._runs.get(key)
        if path is None and reuse and not self.force_retrain:
            matches = matching_manifests(directory, identity)
            if matches:
                path = matches[0][0]
        if path is not None:
            manifest = load_json(path / "manifest.json")
            if manifest["status"] == "retired":
                self._runs.pop(key, None)
                path = None
        if path is not None:
            if manifest["status"] == "complete":
                result = load_run(path)  # Verify artifact checksums before reuse.
                self._runs[key] = path
                if allow_execution:
                    self._activate(directory, slot, identity, path)
                return path, result
        if not allow_execution:
            return None, None
        if path is None:
            path, _ = create_run(directory, identity, reuse=False)
        else:
            manifest.update(status="running", resumed=True)
            manifest.pop("error", None)
            manifest.pop("finished_utc", None)
            save_json(path / "manifest.json", manifest)
        self._runs[key] = path
        self._activate(directory, slot, identity, path)
        return path, None

    @staticmethod
    def _activate(directory, slot, identity, path):
        slot = slot or {"identity": fingerprint(identity)}
        index_file = directory / "active_checkpoints.json"
        index = load_json(index_file) if index_file.exists() else {}
        index[fingerprint(slot)[:24]] = {
            "slot": slot, "identity": fingerprint(identity), "run": path.name,
        }
        save_json(index_file, index)


def _checked_path(root, path):
    """Reject escapes and links, including Windows junctions, before mutations."""
    if not path.is_relative_to(root) or not path.resolve().is_relative_to(root):
        raise ValueError(f"Checkpoint cleanup path escapes its root: {path}")
    for part in (path, *path.parents):
        if part == root:
            break
        if part.is_symlink() or part.is_junction():
            raise ValueError(f"Checkpoint cleanup does not follow links: {part}")
    return path


def _is_model_checkpoint(relative):
    """Only model weights/progress and fitted classifiers; keep analysis files."""
    path = Path(relative)
    if len(path.parts) == 1:
        return path.name in ("weights.npz", "classical_models.joblib") or bool(
            re.fullmatch(r"fold_\d+_models\.joblib", path.name))
    return (len(path.parts) == 2 and path.parts[0] == "checkpoints"
            and (path.suffix == ".npz" or path.name.endswith(".resume.json")))


def retire_pilot_checkpoints(root, *, data_config, backend_config, seed,
                             migration_id="mnist420_exact_v1"):
    """One-time, deferred removal of incompatible pilot model checkpoints.

    Call explicitly from the notebook before training. A durable journal freezes
    the target list before any deletion, so interruptions are recoverable and
    rerunning this cell cannot delete newly trained models. Original manifests,
    including their checksums, remain in the journal. Metrics, predictions,
    histories, preprocessing, source snapshots, figures and data are retained.
    """
    if not re.fullmatch(r"[A-Za-z0-9_-]+", migration_id):
        raise ValueError("Use a simple checkpoint migration identifier.")
    root = Path(root).resolve()
    runs = _checked_path(root, root / "results/runs")
    journal_path = _checked_path(root, root / "results/maintenance" / f"{migration_id}.json")
    data = json_value(data_config)
    backend = json_value(backend_config)
    data_keys = ("train_per_class", "test_per_class", "split_strategy", "n_folds",
                 "validation_fraction", "seed")
    backend_keys = ("shots", "method", "use_ibm_hardware", "one_qubit_error",
                    "two_qubit_error", "readout_error")
    protocol = {"seed": seed, "data": {k: data[k] for k in data_keys},
                "backend": {k: backend[k] for k in backend_keys}}
    stages = ("classical_cv", "clean_vqc", "supervised_models", "unlearning")
    if journal_path.exists():
        journal = load_json(journal_path)
        if journal["status"] == "complete":
            return journal
        if journal["protocol"] != protocol:
            raise ValueError("Finish the pending checkpoint cleanup with its original settings first.")
    else:
        targets = []
        for stage in stages:
            directory = _checked_path(root, runs / stage)
            for file in sorted(directory.glob("*/manifest.json")):
                _checked_path(root, file)
                manifest = load_json(file)
                config = manifest.get("identity", {}).get("config", {})
                if manifest.get("status") == "retired" or not config.get("data"):
                    continue
                compatible = (config.get("seed") == seed
                    and all(config["data"].get(k) == v for k, v in protocol["data"].items())
                    and (stage == "classical_cv" or all(
                        config.get("backend", {}).get(k) == v for k, v in protocol["backend"].items())))
                if compatible:
                    continue
                candidates = [*file.parent.glob("*"), *file.parent.glob("checkpoints/*")]
                files = sorted(p.relative_to(file.parent).as_posix() for p in candidates
                               if _is_model_checkpoint(p.relative_to(file.parent)))
                targets.append({"run": file.parent.relative_to(runs).as_posix(),
                                "files": files, "original_manifest": manifest})
        journal = {"migration_id": migration_id, "status": "pending",
                   "protocol": protocol, "targets": targets}

    # Validate the entire plan before deleting anything. Never recursively delete.
    for target in journal["targets"]:
        relative = Path(target["run"])
        if len(relative.parts) != 2 or relative.parts[0] not in stages or ".." in relative.parts:
            raise ValueError("Invalid run path in checkpoint cleanup journal.")
        run = _checked_path(root, runs / relative)
        manifest_path = _checked_path(root, run / "manifest.json")
        current = load_json(manifest_path)
        if current["identity"] != target["original_manifest"]["identity"]:
            raise ValueError("Run identity changed after checkpoint cleanup was planned.")
        _checked_path(root, run.parent / "active_checkpoints.json")
        for relative_file in target["files"]:
            if not _is_model_checkpoint(relative_file):
                raise ValueError("Cleanup journal contains a non-checkpoint file.")
            file = _checked_path(root, run / relative_file)
            if file.exists() and not file.is_file():
                raise ValueError(f"Expected a checkpoint file: {file}")
    save_json(journal_path, journal)
    for target in journal["targets"]:
        run = runs / target["run"]
        manifest = load_json(run / "manifest.json")
        manifest.update(status="retired", checkpoint_retirement={
            "journal": journal_path.relative_to(root).as_posix(), "removed_files": target["files"]})
        save_json(run / "manifest.json", manifest)
        for relative_file in target["files"]:
            _checked_path(root, run / relative_file).unlink(missing_ok=True)
        index_path = run.parent / "active_checkpoints.json"
        if index_path.exists():
            index = load_json(index_path)
            save_json(index_path, {key: value for key, value in index.items() if value["run"] != run.name})
    journal["status"] = "complete"
    save_json(journal_path, journal)
    return journal


def training_streams(model, random):
    """Callbacks for minibatch RNG, Aer sampling RNG, and accumulated resources."""
    def capture():
        return {"minibatch_rng": random.bit_generator.state,
                "simulator_rng": model.executor.random.bit_generator.state,
                "resources": dict(model.executor.stats),
                "job_ids": list(model.executor.job_ids),
                "actual_device": model.executor.actual_device}

    def restore(state):
        if state["actual_device"] != model.executor.actual_device:
            raise ValueError("Checkpoint device changed; start a fresh run with FORCE_RETRAIN=True.")
        random.bit_generator.state = state["minibatch_rng"]
        model.executor.random.bit_generator.state = state["simulator_rng"]
        model.executor.stats = dict(state["resources"])
        model.executor.job_ids = list(state["job_ids"])

    return capture, restore
