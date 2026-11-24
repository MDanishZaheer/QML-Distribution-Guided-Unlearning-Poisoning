"""Atomic checkpoint storage and optional mirrors on a selected filesystem."""

from pathlib import Path
from threading import Event, Thread, Lock
import json
import os
import hashlib
from datetime import datetime, timezone
from uuid import uuid4


def _atomic_bytes(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{uuid4().hex}.tmp")
    try:
        temporary.write_bytes(content)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _receipt(path):
    return Path(path).with_name(Path(path).name + ".receipt.json")


def _read_checkpoint(path):
    path = Path(path)
    content = path.read_bytes()
    state = json.loads(content)
    receipt = _receipt(path)
    if receipt.exists():
        recorded = json.loads(receipt.read_text())
        if (recorded["sha256"] != hashlib.sha256(content).hexdigest()
                or recorded["identity"] != state["identity"] or recorded["step"] != state["step"]):
            raise ValueError(f"Checkpoint checksum/receipt mismatch: {path}")
    return state, content


def _publish_checkpoint(path, content):
    state = json.loads(content)
    _atomic_bytes(path, content)
    if Path(path).read_bytes() != content:
        raise OSError("Checkpoint read-back verification failed.")
    receipt = {"identity": state["identity"], "step": state["step"],
               "sha256": hashlib.sha256(content).hexdigest(),
               "verified_utc": datetime.now(timezone.utc).isoformat()}
    _atomic_bytes(_receipt(path), json.dumps(receipt, indent=2).encode())
    _read_checkpoint(path)


class CheckpointMirror:
    """Save each update directly, or use an explicitly configured runtime mirror.

    By default the optimizer writes atomically to its final project path.
    Set QML_RUNTIME_CHECKPOINT_DIR to a fast scratch directory to enable periodic
    backups (default 5 s). Normal interruption flushes the last complete update.
    Losing the scratch filesystem can lose updates since the last backup.
    """

    def __init__(self, durable_path, interval=5.0):
        self.durable = Path(durable_path)
        self.interval = interval
        if interval <= 0:
            raise ValueError("Checkpoint backup interval must be positive.")
        scratch = os.environ.get("QML_RUNTIME_CHECKPOINT_DIR")
        self.enabled = bool(scratch)
        self.stop = Event()
        self.flush_lock = Lock()
        self.error = None
        if self.enabled:
            namespace = hashlib.sha256(str(self.durable.parent.parent.parent).encode()).hexdigest()[:12]
            self.local = Path(scratch).expanduser().resolve() / namespace / self.durable.parent.parent.name / self.durable.name
            if self.local.resolve() == self.durable.resolve():
                raise ValueError("Runtime mirror and durable checkpoint must be different paths.")
        else:
            self.local = self.durable

    def _flush(self):
        with self.flush_lock:
            self._flush_unlocked()

    def flush_now(self):
        """Publish initial state immediately, in addition to periodic/final backups."""
        self._flush()

    def _flush_unlocked(self):
        if not self.local.exists() or not self.enabled:
            return
        self.durable.parent.mkdir(parents=True, exist_ok=True)
        # The optimizer replaces the local file atomically, so this read is committed state.
        content = self.local.read_bytes()
        json.loads(content)
        if self.durable.exists():
            _, previous = _read_checkpoint(self.durable)
            if previous == content:
                if not _receipt(self.durable).exists():
                    _publish_checkpoint(self.durable, content)
                return
            _publish_checkpoint(self.durable.with_name(self.durable.name + ".previous"), previous)
        _publish_checkpoint(self.durable, content)

    def _loop(self):
        while not self.stop.wait(self.interval):
            try:
                self._flush()
            except Exception as error:
                self.error = error
                self.stop.set()

    def __enter__(self):
        if self.enabled:
            self.local.parent.mkdir(parents=True, exist_ok=True)
            candidates, errors = [], []
            for file in (self.durable, self.durable.with_name(self.durable.name + ".previous")):
                if file.exists():
                    try:
                        saved, content = _read_checkpoint(file)
                        candidates.append((saved["step"], saved, content))
                    except (ValueError, KeyError, OSError) as error:
                        errors.append(error)
            if candidates:
                _, saved, content = max(candidates, key=lambda item: item[0])
                if errors or not self.durable.exists():
                    _publish_checkpoint(self.durable, content)
                local = json.loads(self.local.read_text()) if self.local.exists() else None
                if local is None or local.get("identity") != saved["identity"] or local["step"] < saved["step"]:
                    _atomic_bytes(self.local, content)
            elif errors:
                raise ValueError("No verified checkpoint generation is recoverable.") from errors[0]
            else:
                # A forced fitting session never resumes a leftover local attempt.
                self.local.unlink(missing_ok=True)
            self.thread = Thread(target=self._loop, name="checkpoint-backup", daemon=True)
            self.thread.start()
        return self.local

    def __exit__(self, *args):
        if self.enabled:
            self.stop.set()
            self.thread.join()
            self._flush()
            if self.error is not None:
                raise RuntimeError("Checkpoint backup failed; runtime progress remains available.") from self.error


def verify_checkpoint_storage(root):
    """Exercise atomic save, read-back and recovery on the selected durable filesystem."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Project directory does not exist: {root}")
    destination = root / "results/maintenance"
    probe = destination / f"storage_probe_{uuid4().hex}.resume.json"
    mirror = CheckpointMirror(probe)
    try:
        with mirror as local:
            _atomic_bytes(local, json.dumps({"identity": "storage_probe", "step": 1,
                          "weights": [0.0], "first_moment": [0.0], "second_moment": [0.0]}).encode())
        saved, content = _read_checkpoint(probe)
        if saved["step"] != 1:
            raise OSError("Durable checkpoint probe did not round-trip.")
        report = {"verified": True, "project_root": str(root), "runtime_mirror": mirror.enabled,
                  "verified_utc": datetime.now(timezone.utc).isoformat(),
                  "probe_sha256": hashlib.sha256(content).hexdigest(),
                  "backup_interval_seconds": 5,
                  "limitation": "Read-back verifies this filesystem. With a runtime mirror, scratch loss can discard updates since its last backup."}
        _atomic_bytes(destination / "checkpoint_storage_check.json", json.dumps(report, indent=2).encode())
        return report
    finally:
        for file in (probe, _receipt(probe), probe.with_name(probe.name + ".previous"),
                     _receipt(probe.with_name(probe.name + ".previous")), mirror.local):
            file.unlink(missing_ok=True)


def checkpoint_status(root):
    """Optional full checkpoint inventory; normal loading verifies individual runs."""
    rows = []
    for stage in ("ablations", "ablation_validation", "noisy_supervised", "noisy_unlearning",
                  "ablation_inputs_supervised", "ablation_inputs_unlearning",
                  "fresh_supervised", "fresh_unlearning"):
        directory = Path(root) / "results/runs" / stage
        for manifest_file in sorted(directory.glob("*/manifest.json")):
            manifest = json.loads(manifest_file.read_text())
            file = manifest_file.parent / "checkpoints/model.resume.json"
            row = {"stage": stage, "run": manifest_file.parent.name, "status": manifest["status"],
                   "saved_step": None, "local_step": None, "backup_lag_updates": None,
                   "verified_receipt": False, "backup_utc": None}
            if file.exists():
                try:
                    state, _ = _read_checkpoint(file)
                    row.update(saved_step=state["step"], verified_receipt=_receipt(file).exists())
                    local = CheckpointMirror(file).local
                    if local.exists():
                        current = json.loads(local.read_text())
                        row["local_step"] = current["step"]
                        row["backup_lag_updates"] = current["step"] - state["step"]
                    if _receipt(file).exists():
                        row["backup_utc"] = json.loads(_receipt(file).read_text())["verified_utc"]
                except (ValueError, KeyError, OSError) as error:
                    row["error"] = str(error)
            rows.append(row)
    return rows
