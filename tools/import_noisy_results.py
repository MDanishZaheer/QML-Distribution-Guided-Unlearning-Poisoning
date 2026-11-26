"""Verify a downloaded noisy-study export before importing immutable artifacts.

Only numeric files and JSON are read. Incoming Python, notebooks and joblib
files are never executed. Archived training provenance is kept byte-for-byte;
active project source is never replaced by downloaded code.
"""

import argparse
from dataclasses import asdict
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import shutil
from tempfile import TemporaryDirectory
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED

import numpy as np

from src import artifacts, noisy_training
from src.config import DEFAULT_SEED
from src.data.feature_randomization import experiment_config
from src.evaluation.metrics import classification_metrics
from src.project_workflow import load_completed_study, protect_saved_review
from src.unlearning.targets import make_forget_target


COMMON = ("code_fingerprint", "environment", "execution", "training_noise", "seed")
STAGES = ("noisy_supervised", "noisy_unlearning")


def checked_members(archive):
    """Validate the complete central directory, even for files we will skip."""
    seen = set()
    for entry in archive.infolist():
        path = PurePosixPath(entry.filename)
        key = entry.filename.casefold()
        if (entry.orig_filename != entry.filename or path.is_absolute() or ".." in path.parts
                or "\\" in entry.filename or ":" in entry.filename
                or key in seen or (entry.external_attr >> 16) & 0o170000 == 0o120000):
            raise ValueError("Unsafe or duplicate ZIP entry: " + entry.filename)
        seen.add(key)
    damaged = archive.testzip()
    if damaged:
        raise ValueError("ZIP CRC failure: " + damaged)
    return [entry.filename for entry in archive.infolist() if not entry.is_dir()]


def _same(actual, expected, description):
    if artifacts.fingerprint(actual) != artifacts.fingerprint(expected):
        raise ValueError("Noisy import mismatch: " + description)


def _arrays(actual, expected):
    if set(actual) != set(expected):
        raise ValueError("Frozen numeric feature keys changed.")
    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)


def _probabilities(arrays, labels, ids, *, key="probabilities"):
    np.testing.assert_array_equal(arrays["labels"], labels)
    np.testing.assert_array_equal(arrays["sample_ids"], ids)
    p = arrays[key]
    if p.shape != (len(labels), 3) or not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1):
        raise ValueError("Invalid saved class probabilities.")
    np.testing.assert_allclose(p.sum(axis=1), 1., atol=1e-12, rtol=0)


def source_digest(file):
    with ZipFile(file) as archive:
        names = checked_members(archive)
        paths = sorted(n for n in names if n.startswith("src/") and n.endswith(".py")) + ["requirements.txt"]
        digest = sha256()
        for name in paths:
            digest.update(name.encode())
            digest.update(archive.read(name))
    return digest.hexdigest()


def validate_study(staging, project, *, emit=print):
    staging, project = Path(staging), Path(project).resolve()
    indexes = list((staging / "results/noisy_training").glob("*/study.json"))
    if len(indexes) != 1:
        raise ValueError("The export must select exactly one noisy study.")
    path = indexes[0]
    record = artifacts.load_json(path)
    if not record.get("complete"):
        raise ValueError("An unfinished study is not imported as complete.")
    protocol = record["protocol"]
    if (protocol["seed"] != DEFAULT_SEED or protocol["training_noise"] != {"family": "combined", "level": .01}
            or len(record["supervised"]) != 100 or len(record["unlearned"]) != 130):
        raise ValueError("The completed five-fold noisy protocol/counts differ.")
    if path.parent.name != artifacts.fingerprint(protocol)[:16]:
        raise ValueError("Study directory differs from its protocol fingerprint.")
    original = load_completed_study(project, protocol["original_report"], progress=None).select_rates((.05, .1, .2, .3, .4, .5))
    _same(protocol["original_source"], original.source_fingerprint, "original training source")
    problems, historical, bindings = noisy_training.load_training_problems(original, (0, 1, 2, 3, 4))
    _same(protocol["bindings"], bindings.to_dict("records"), "historical condition/model bindings")
    _same(protocol["problems"], [p.key for p in problems], "problem membership")
    _same(protocol["plan"], noisy_training.training_plan(problems).to_dict("records"), "fit budgets")
    _same(protocol["historical_inputs"], {key: value.identity(project) for key, value in historical.items()}, "historical numeric artifacts")
    runtime = record["service_resources"].get("torch_runtime")
    if runtime:
        # Preserve/verify the recorded engine rather than pretending it was Aer.
        fingerprint = sha256(json.dumps(runtime, sort_keys=True).encode()).hexdigest()
        _same(fingerprint, protocol["execution"]["runtime_fingerprint"], "training runtime fingerprint")
        support = staging / "results/noisy_training/runtime_support/torch" / fingerprint[:16]
        for name, digest in runtime["source"].items():
            _same(artifacts._file_digest(support / name), digest, "archived execution helper " + name)
    else:
        raise ValueError("Missing recorded noisy execution provenance.")
    expected_supervised = {}
    for p in problems:
        expected_supervised[p.original["path"].name] = p.original, p.arrays
        expected_supervised[p.reference["path"].name] = p.reference, p.reference_arrays
    _same(sorted(record["supervised"]), sorted(expected_supervised), "supervised problem IDs")
    runs, source_hashes, manifest_files = {}, set(), []

    def read_run(link, stage, arrays, original_prediction):
        relative = PurePosixPath(link)
        if relative.parts[:3] != ("results", "runs", stage) or len(relative.parts) != 4:
            raise ValueError("Invalid model path in the noisy index.")
        run = artifacts.load_run(staging / link)
        identity = run["manifest"]["identity"]
        _same({key: identity[key] for key in COMMON}, {key: protocol[key] for key in COMMON}, "model provenance")
        _same(identity["stage"], stage, "model stage")
        _same(identity["config"], run["config"], "model configuration")
        _arrays(artifacts.load_arrays(run["path"] / "features.npz"), arrays)
        pred = run["predictions"]
        _probabilities({"labels": pred["test_labels"], "sample_ids": pred["test_sample_ids"], "probabilities": pred["model__test"]},
            original_prediction["test_labels"], original_prediction["test_sample_ids"])
        actual_scores = classification_metrics(pred["test_labels"], pred["model__test"])
        scores = run["metrics"]["models"]["model"]
        for name, value in actual_scores.items():
            if isinstance(value, dict):
                np.testing.assert_allclose(list(scores[name].values()), list(value.values()), atol=1e-12, rtol=0)
            else:
                np.testing.assert_allclose(scores[name], value, atol=1e-12, rtol=0)
        checkpoint = artifacts.load_json(run["path"] / "checkpoints/model.resume.json")
        budget = 300 if stage == "noisy_supervised" else 100
        if checkpoint["step"] != budget or checkpoint["stopped"] or checkpoint["streams"]["actual_device"] != "torch:cuda:0":
            raise ValueError("Optimizer budget/device differs from the complete noisy fit.")
        np.testing.assert_array_equal(run["weights"]["model"], checkpoint["best_weights"])
        if len(checkpoint["weights"]) != 72 or not np.isfinite(checkpoint["weights"]).all():
            raise ValueError("Invalid model parameters.")
        if int(run["history"].step.max()) != budget:
            raise ValueError("Training history does not reach the recorded budget.")
        source = run["path"] / "source.zip"
        digest = artifacts._file_digest(source)
        if digest not in source_hashes:
            _same(source_digest(source), protocol["code_fingerprint"], "archived scientific source")
            source_hashes.add(digest)
        manifest_files.append(run["path"] / "manifest.json")
        runs[link] = run
        return run

    for key, entry in record["supervised"].items():
        source, arrays = expected_supervised[key]
        run = read_run(entry["run"], "noisy_supervised", arrays, source["predictions"])
        identity = run["manifest"]["identity"]
        _same(identity["historical_problem"], key, "supervised source model")
        _same(identity["historical_artifacts"], source["manifest"]["artifact_sha256"], "source artifact hashes")
        _same(identity["frozen_numeric_features"], artifacts.fingerprint(arrays), "frozen numeric features")
    for p in problems:
        parent = runs[record["supervised"][p.original["path"].name]["run"]]
        reference = runs[record["supervised"][p.reference["path"].name]["run"]]
        teacher = artifacts.load_json(path.parent / "teachers" / f"{p.key}.json")
        refs = teacher["references"]
        expected_teacher = {"original": parent["path"].name, "original_artifacts": parent["manifest"]["artifact_sha256"],
            "membership": {k: getattr(p.membership, k) for k in ("forget_indices", "anchor_indices", "calibration_indices")}}
        _same(teacher["identity"], artifacts.fingerprint(expected_teacher), "frozen teacher identity")
        _same(teacher["references_fingerprint"], artifacts.fingerprint(refs), "frozen teacher checksum")
        for k in expected_teacher["membership"]:
            np.testing.assert_array_equal(refs[k], expected_teacher["membership"][k])
        np.testing.assert_array_equal(refs["original_weights"], parent["weights"]["model"])
        for method in ("guided", "uniform"):
            entry = record["unlearned"][f"{p.key}_{method}"]
            run = read_run(entry["run"], "noisy_unlearning", p.arrays, p.original["predictions"])
            identity = run["manifest"]["identity"]
            _same(identity["parent_run"], parent["path"].name, "unlearning parent")
            _same(identity["reference_run"], reference["path"].name, "retained retraining reference")
            _same(identity["method"], method, "unlearning method")
            _same(identity["fixed_teacher"], artifacts.fingerprint(refs), "unlearning teacher fingerprint")
            _same(artifacts.load_json(run["path"] / "training_references.json"), refs, "saved unlearning references")
            target = make_forget_target(np.asarray(refs["calibration_probabilities"]), p.config.unlearning.forget_class,
                                       p.config.unlearning.beta, method)
            np.testing.assert_allclose(run["metrics"]["target"], target, atol=1e-12, rtol=0)
    actual_manifests = {p.resolve() for stage in STAGES for p in (staging / "results/runs" / stage).glob("*/manifest.json")}
    if actual_manifests != {p.resolve() for p in manifest_files} or len(runs) != 230:
        raise ValueError("Unlinked or missing noisy models in the export.")
    expected_evaluations = {("historical", "noisy", key) for key in historical}
    expected_evaluations |= {("noisy_trained", "ideal", run["path"].name) for run in runs.values()}
    actual_evaluations = set()
    by_name = {run["path"].name: run for run in runs.values()}
    for entry in record["evaluations"]:
        key = entry["regime"], entry["deployment"], entry["run_id"]
        if key in actual_evaluations:
            raise ValueError("Duplicate deployment evaluation.")
        actual_evaluations.add(key)
        file = (staging / entry["file"]).resolve()
        if not file.is_relative_to(staging.resolve()):
            raise ValueError("Deployment file leaves the import staging directory.")
        _same(artifacts._file_digest(file), entry["sha256"], "deployment prediction checksum")
        arrays = artifacts.load_arrays(file)
        if entry["regime"] == "historical":
            old = historical[entry["run_id"]]
            labels, ids = old.labels, old.sample_ids
        else:
            pred = by_name[entry["run_id"]]["predictions"]
            labels, ids = pred["test_labels"], pred["test_sample_ids"]
        _probabilities(arrays, labels, ids)
    _same(sorted(actual_evaluations), sorted(expected_evaluations), "complete deployment coverage")
    emit("Verified 230 model manifests/artifacts, complete update budgets, frozen inputs/teachers, and 460 evaluations.")
    return {"study_index": path.relative_to(staging).as_posix(), "complete": True,
            "supervised": 100, "unlearning": 130, "deployment_prediction_files": len(actual_evaluations),
            "bindings": len(bindings), "folds": sorted({p.fold for p in problems}),
            "training_engine": protocol["execution"]["engine"], "actual_training_runtime": runtime,
            "training_elapsed_seconds": record["elapsed_seconds"], "source_fingerprint": protocol["code_fingerprint"],
            "original_source": original.source_fingerprint, "research_training_executed_during_import": False}


def import_export(archive_file, project, *, apply=False, emit=print):
    archive_file, project = Path(archive_file).resolve(), Path(project).resolve()
    digest = artifacts._file_digest(archive_file)
    audit = project / "results/maintenance" / ("import_" + digest[:16])
    with TemporaryDirectory(prefix="qml-noisy-import-") as temporary:
        staging = Path(temporary).resolve()
        with ZipFile(archive_file) as outer:
            names = checked_members(outer)
            exports = [n for n in names if n.endswith("/noisy_models_results.zip")]
            if len(exports) != 1:
                raise ValueError("Select an archive containing one final noisy-model export.")
            content = outer.read(exports[0])
        with ZipFile(BytesIO(content)) as inner:
            names = checked_members(inner)
            selected = [n for n in names if n.startswith("results/noisy_training/")
                        or any(n.startswith(f"results/runs/{s}/") for s in STAGES)]
            for name in selected:
                target = staging / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(inner.read(name))
            with protect_saved_review():
                report = validate_study(staging, project, emit=emit)
            # Additional provenance keeps the original implementation available
            # without installing/importing it in the active Aer-only project.
            snapshot = staging / Path(report["study_index"]).parent / "execution_source.zip"
            with ZipFile(snapshot, "w", ZIP_DEFLATED) as source:
                for name in names:
                    if name.startswith(("src/", "tools/")) or name in ("requirements.txt", "README.md", "main_project.ipynb"):
                        entry = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                        entry.compress_type = ZIP_DEFLATED
                        source.writestr(entry, inner.read(name))
        files = {p.relative_to(staging).as_posix(): artifacts._file_digest(p) for p in staging.rglob("*") if p.is_file()}
        conflicts = [n for n, digest_value in files.items()
                     if (project / n).exists() and artifacts._file_digest(project / n) != digest_value]
        if conflicts:
            raise ValueError("Import would overwrite different artifacts: " + repr(conflicts[:5]))
        report.update(archive=archive_file.name, archive_sha256=digest, inner_export_sha256=sha256(content).hexdigest(),
                      file_count=len(files), active_source_replaced=False, applied=bool(apply))
        if apply:
            for name in files:
                target = project / name
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(staging / name, target)
                if artifacts._file_digest(target) != files[name]:
                    raise OSError("Imported bytes failed verification: " + name)
            artifacts.save_json(audit / "files.json", files)
            artifacts.save_json(audit / "report.json", report)
        emit(json.dumps({k: report[k] for k in ("study_index", "complete", "supervised", "unlearning", "file_count", "applied")}, indent=2))
        return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    import_export(args.archive, args.root, apply=args.apply)
