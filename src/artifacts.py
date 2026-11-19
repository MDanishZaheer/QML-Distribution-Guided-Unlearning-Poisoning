"""Reproducible dataset variants, immutable run directories, and reloadable outputs."""

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import uuid
import zipfile

import joblib
import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT


def json_value(value):
    if is_dataclass(value):
        return json_value(asdict(value))
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("Non-finite values cannot be stored in research metadata.")
    return value


def fingerprint(value) -> str:
    serialized = json.dumps(json_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def save_json(path: Path | str, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(json_value(value), indent=2, sort_keys=True,
                                        allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_json(path: Path | str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_arrays(path: Path | str, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for name, array in arrays.items():
        if np.asarray(array).dtype.hasobject:
            raise ValueError(f"Object arrays are not supported: {name}")
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_arrays(path: Path | str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as arrays:
        return {name: arrays[name].copy() for name in arrays.files}


def source_fingerprint(root: Path | str = PROJECT_ROOT) -> str:
    root = Path(root)
    files = sorted((root / "src").rglob("*.py")) + [root / "requirements.txt"]
    digest = hashlib.sha256()
    for path in files:
        if path.exists():
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def save_source_snapshot(path: Path | str):
    """Keep the exact source files available even before the project has Git commits."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file in sorted((PROJECT_ROOT / "src").rglob("*.py")):
            archive.write(file, file.relative_to(PROJECT_ROOT).as_posix())
        archive.write(PROJECT_ROOT / "requirements.txt", "requirements.txt")


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_info() -> dict:
    packages = {}
    for name in ("qiskit", "qiskit-aer", "qiskit-ibm-runtime", "numpy", "scipy",
                 "scikit-learn", "pandas", "matplotlib", "joblib"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    from qiskit_aer import AerSimulator
    try:
        report = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        gpu = report.stdout.strip() if report.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        gpu = None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": packages, "aer_devices": list(AerSimulator().available_devices()),
            "nvidia_gpu": gpu}


def save_split_manifest(splits, directory: Path | str) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"split_{splits.fingerprint()[:16]}.json"
    content = {"fingerprint": splits.fingerprint()}
    for name in ("train", "validation", "test"):
        partition = getattr(splits, name)
        content[name] = {"sample_ids": partition.sample_ids, "labels": partition.labels}
    if path.exists():
        if load_json(path) != json_value(content):
            raise ValueError("Existing split manifest does not match these records.")
    else:
        save_json(path, content)
    return path


def save_poisoned_dataset(dataset, directory: Path | str) -> Path:
    """Create data/annotated/<condition>_<identity>/ only when explicitly called."""
    information = dataset.metadata()
    identity = fingerprint(information)[:16]
    label = "clean" if dataset.config.kind == "none" else dataset.config.kind
    name = f"{label}_p{dataset.config.rate * 100:g}_{identity}"
    destination = Path(directory) / name
    if destination.exists():
        if (destination / "metadata.json").exists() and load_json(destination / "metadata.json") == json_value(information):
            if (destination / "images.npz").exists() and (destination / "annotations.csv").exists():
                return destination
        raise FileExistsError(f"Incomplete or conflicting annotated dataset: {destination}")
    destination.mkdir(parents=True)
    save_arrays(destination / "images.npz", images=dataset.observed.images,
                original_labels=dataset.original.labels, observed_labels=dataset.observed.labels,
                sample_ids=dataset.original.sample_ids, is_poisoned=dataset.is_poisoned)
    dataset.manifest().to_csv(destination / "annotations.csv", index=False)
    # Write the completion marker last.
    save_json(destination / "metadata.json", information)
    return destination


def create_run(directory: Path | str, identity: dict, *, reuse: bool = True):
    """Return (path, cached). Never silently overwrite a previous experiment."""
    config = identity["config"]
    name = (f"{config['poison']['kind']}_p{config['poison']['rate'] * 100:g}"
            f"_seed{config['seed']}_fold{config['data']['fold']}_{fingerprint(identity)[:16]}")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / name
    if destination.exists():
        if reuse:
            manifest = load_json(destination / "manifest.json")
            if manifest["status"] == "complete" and manifest["identity"] == json_value(identity):
                return destination, True
            raise RuntimeError(f"Run is incomplete. Use reuse=False to create a new attempt: {destination}")
        destination = root / f"{name}_{uuid.uuid4().hex[:8]}"
    destination.mkdir()
    save_json(destination / "manifest.json", {
        "status": "running", "created_utc": datetime.now(timezone.utc).isoformat(), "identity": identity,
    })
    save_json(destination / "config.json", config)
    save_json(destination / "provenance.json", {key: value for key, value in identity.items() if key != "config"})
    return destination, False


def finish_run(directory: Path | str, status: str, **details):
    path = Path(directory) / "manifest.json"
    manifest = load_json(path)
    manifest.update(status=status, finished_utc=datetime.now(timezone.utc).isoformat(), **details)
    if status == "complete":
        root = Path(directory)
        manifest["artifact_sha256"] = {
            file.relative_to(root).as_posix(): _file_digest(file)
            for file in sorted(root.rglob("*")) if file.is_file() and file != path}
    save_json(path, manifest)


def save_training_result(directory: Path | str, name: str, result):
    directory = Path(directory)
    save_arrays(directory / "checkpoints" / f"{name}.npz",
                best_weights=result.weights, last_weights=result.last_weights)
    history_dir = directory / "histories"
    history_dir.mkdir(exist_ok=True)
    result.history.to_csv(history_dir / f"{name}.csv", index=False)
    save_json(directory / "checkpoints" / f"{name}.json", {
        "best_step": result.best_step, "best_score": result.best_score,
        "selection_metric": result.selection_metric,
    })


def load_run(directory: Path | str) -> dict:
    """Load numeric outputs only; joblib files are never implicitly unpickled."""
    directory = Path(directory)
    manifest = load_json(directory / "manifest.json")
    if manifest["status"] != "complete":
        raise ValueError("Only completed runs may be used as cached research results.")
    for relative, expected in manifest.get("artifact_sha256", {}).items():
        file = (directory / relative).resolve()
        if not file.is_relative_to(directory.resolve()) or not file.is_file() or _file_digest(file) != expected:
            raise ValueError(f"Saved run artifact is missing or changed: {relative}")
    return {"path": directory, "config": load_json(directory / "config.json"),
            "metrics": load_json(directory / "metrics.json"), "manifest": manifest,
            "weights": load_arrays(directory / "weights.npz"),
            "predictions": load_arrays(directory / "predictions.npz"),
            "history": pd.read_csv(directory / "history.csv")}


def find_completed_run(directory, identity):
    """Reuse only an identical, completed run with verified artifact hashes."""
    manifests = [(file, load_json(file)) for file in Path(directory).glob("*/manifest.json")]
    for file, manifest in sorted(manifests, key=lambda item: item[1]["created_utc"], reverse=True):
        if manifest.get("status") == "complete" and manifest.get("identity") == json_value(identity):
            return load_run(file.parent)
    return None


def save_fold_artifacts(path, splits, features, root, prefix):
    """Keep labels, record IDs, fitted PCA, and features alongside the results."""
    manifest = save_split_manifest(splits, root / "data/splits")
    arrays = {}
    for partition in ("train", "validation", "test"):
        dataset = getattr(splits, partition)
        arrays[f"{partition}_features"] = features[partition]
        arrays[f"{partition}_labels"] = dataset.labels
        arrays[f"{partition}_sample_ids"] = dataset.sample_ids
    save_arrays(path / f"{prefix}_features.npz", **arrays)
    joblib.dump(features["preprocessing"], path / f"{prefix}_preprocessing.joblib")
    save_json(path / f"{prefix}_preprocessing.json", features["preprocessing"].diagnostics())
    return manifest.relative_to(root).as_posix()
