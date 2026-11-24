"""Local read cache for historical inputs; new results still belong on storage."""

from hashlib import sha256
from pathlib import Path, PurePosixPath
import json
import os
import tempfile
from zipfile import ZipFile, ZIP_DEFLATED


def _digest(path):
    value = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _say(progress, message):
    if progress is not None:
        progress(message)


def build_review_input_archive(review, destination=None):
    """Package already verified inputs, without changing any historical artifact.

    Run this on the local project copy before uploading it. The archive is a
    transport cache, not an experimental output or a replacement for checkpoints.
    """
    root = review.root
    destination = Path(destination or root / "results/review_inputs" /
                       f"{review.report_path.name}.zip")
    files = set()
    for run in review.runs.values():
        directory = run["path"]
        files.add(directory / "manifest.json")
        files.update(directory / name for name in run["manifest"]["artifact_sha256"])
    files.update(condition["manifest_path"] for condition in review.conditions.values())
    files.update(review.split_paths.values())
    files.update(file for file in review.report_path.iterdir() if file.is_file())
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".zip.tmp")
    try:
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED) as archive:
            for file in sorted(files):
                relative = file.resolve().relative_to(root).as_posix()
                archive.write(file, relative)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    metadata = {
        "schema": 1, "archive_sha256": _digest(destination),
        "report": review.report_path.relative_to(root).as_posix(),
        "gold_run": review.gold["path"].relative_to(root).as_posix(),
        "classical_run": review.classical["path"].relative_to(root).as_posix(),
        "file_count": len(files), "purpose": "historical-input transport cache",
    }
    destination.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf8")
    return destination


def prepare_review_inputs(root, report, *, gold_run, classical_run,
                          cache_directory=None, progress=None):
    """Copy one archive from storage, verify it, and extract to local runtime disk.

    Each run's original artifact hashes are still checked by SavedResultsReview.
    Its public paths and all new output paths continue to point to the project.
    Absence of an archive uses the ordinary strict loader; a broken archive fails.
    """
    root = Path(root).resolve()
    archive_path = root / "results/review_inputs" / f"{Path(report).name}.zip"
    metadata_path = archive_path.with_suffix(".json")
    _say(progress, "Locating historical-input archive (no fitting).")
    if not archive_path.is_file() and not metadata_path.is_file():
        _say(progress, "Archive not present; verifying individual files with concurrent reads.")
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf8"))
    selection = {"report": str(Path(report).as_posix()),
                 "gold_run": str(Path(gold_run).as_posix()),
                 "classical_run": str(Path(classical_run).as_posix())}
    if metadata.get("schema") != 1 or any(metadata.get(k) != v for k, v in selection.items()):
        raise ValueError("Historical-input archive selects different report/baseline runs.")
    expected = metadata["archive_sha256"]
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError("Invalid historical-input archive checksum.")
    base = Path(cache_directory or Path(tempfile.gettempdir()) / "qml_saved_review").resolve()
    if base == root or base.is_relative_to(root):
        raise ValueError("The historical read cache must be outside the storage project.")
    destination = base / expected
    marker = destination / ".ready.json"
    if marker.is_file() and json.loads(marker.read_text()).get("archive_sha256") == expected:
        _say(progress, "Using local historical-input cache; original run checksums will be rechecked.")
        return destination
    destination.mkdir(parents=True, exist_ok=True)
    local_zip = destination / ".inputs.zip"
    _say(progress, "Copying historical-input archive to local disk; this is one storage file.")
    value = sha256()
    with archive_path.open("rb") as source, local_zip.open("wb") as target:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            target.write(chunk)
            value.update(chunk)
    if value.hexdigest() != expected:
        local_zip.unlink(missing_ok=True)
        raise ValueError("Historical-input archive checksum mismatch; no inputs were loaded.")
    _say(progress, "Archive checksum passed; extracting on local disk.")
    try:
        with ZipFile(local_zip) as archive:
            names = archive.namelist()
            if len(set(names)) != len(names) or len(names) != metadata["file_count"]:
                raise ValueError("Historical-input archive has duplicate or missing entries.")
            for entry in archive.infolist():
                name = PurePosixPath(entry.filename)
                target = (destination / entry.filename).resolve()
                if (name.is_absolute() or ".." in name.parts or "\\" in entry.filename
                        or not target.is_relative_to(destination) or entry.is_dir()
                        or entry.filename.startswith(".") or (entry.external_attr >> 16) & 0o170000 == 0o120000):
                    raise ValueError("Historical-input archive contains an invalid file path.")
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(entry) as source, target.open("wb") as output:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        output.write(chunk)
        marker.write_text(json.dumps({"archive_sha256": expected}), encoding="utf8")
    finally:
        local_zip.unlink(missing_ok=True)
    _say(progress, f"Local inputs ready: {len(names):,} files. New checkpoints will still save to storage.")
    return destination
