"""Read-only model audit: independently evaluate saved runs, never fit models.

Run with --provenance <report provenance.json> --output <audit.json>.
Only the explicitly requested audit JSON is written. Research artifacts and
checkpoint state are not modified, and no joblib objects are deserialized.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.artifacts import load_arrays, load_json, load_run, save_json, source_fingerprint
from test_guided_paper_audit import paper_probabilities, paper_loss_parts


def audit(provenance_file):
    provenance = load_json(provenance_file)
    current_source = source_fingerprint()
    assert current_source == provenance["source_fingerprint"]
    conditions = [load_json(PROJECT / p) for p in provenance["condition_manifests"]]
    runs, parents, original_probabilities, source_archives = {}, {}, {}, set()
    error_bounds = Counter()
    methods, best_steps, completed_steps = Counter(), Counter(), Counter()
    pairs = set()

    def close(name, actual, expected, tolerance=2e-10):
        actual, expected = np.asarray(actual), np.asarray(expected)
        assert actual.shape == expected.shape, (name, actual.shape, expected.shape)
        error = float(np.max(np.abs(actual - expected))) if actual.size else 0.0
        error_bounds[name] = max(error_bounds[name], error)
        assert error <= tolerance, (name, error)

    def verify_source(run):
        assert run["manifest"]["identity"]["code_fingerprint"] == current_source
        archive = run["path"] / "source.zip"
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        if digest in source_archives:
            return
        with zipfile.ZipFile(archive) as snapshot:
            names = snapshot.namelist()
            expected = [p.relative_to(PROJECT).as_posix() for p in (PROJECT / "src").rglob("*.py")]
            expected.append("requirements.txt")
            assert set(names) == set(expected)
            for name in names:
                assert snapshot.read(name) == (PROJECT / name).read_bytes(), name
        source_archives.add(digest)

    for condition in conditions:
        config = condition["config"]
        assert config["seed"] == 6304
        assert config["circuit"] == {"ansatz_reps": 3, "entanglement": True,
                                      "logit_scale": 1.0, "reuploads": 2}
        assert config["backend"]["shots"] is None and config["backend"]["method"] == "statevector"
        assert not any(config["backend"][k] for k in ("one_qubit_error", "two_qubit_error", "readout_error"))
        unlearning = config["unlearning"]
        assert (unlearning["alpha"], unlearning["regularization"], unlearning["beta"]) == (1.0, 0.01, 1.0)
        split = load_json(PROJECT / condition["split_manifest"])
        train_ids = np.asarray(split["train"]["sample_ids"])
        heldout_ids = set(split["validation"]["sample_ids"] + split["test"]["sample_ids"])
        for case in ("clean", "poisoned"):
            pair = []
            for method in ("guided", "uniform"):
                path = condition["unlearned"][f"{case}_{method}"]
                pair.append(path)
                if path in runs:
                    continue
                run = load_run(PROJECT / path)  # Includes verification of all saved artifact hashes.
                verify_source(run)
                identity = run["manifest"]["identity"]
                assert identity["method"] == method
                parent_path = condition["models"][case]
                assert identity["parent_run"] == Path(parent_path).name
                if parent_path not in parents:
                    parent = load_run(PROJECT / parent_path)
                    verify_source(parent)
                    arrays = load_arrays(parent["path"] / "features.npz")
                    parents[parent_path] = (parent, arrays)
                    original_probabilities[parent_path] = paper_probabilities(arrays["train"], parent["weights"]["model"])
                parent, arrays = parents[parent_path]
                np.testing.assert_array_equal(arrays["train_ids"], train_ids)
                refs = load_json(run["path"] / "training_references.json")["references"]
                refs = {k: np.asarray(v) for k, v in refs.items()}
                saved = load_arrays(run["path"] / "references.npz")
                forgotten, anchors, calibration = (refs[k] for k in
                    ("forget_indices", "anchor_indices", "calibration_indices"))
                np.testing.assert_array_equal(forgotten, np.flatnonzero(arrays["train_labels"] == 2))
                assert set(calibration) <= set(forgotten)
                assert set(anchors).isdisjoint(forgotten)
                assert set(train_ids[anchors]).isdisjoint(heldout_ids)
                assert set(train_ids[calibration]).isdisjoint(heldout_ids)
                assert len(calibration) == min(32, len(forgotten))
                for label in (0, 1):
                    count = np.sum(arrays["train_labels"] == label)
                    assert np.sum(arrays["train_labels"][anchors] == label) == int(np.ceil(0.25 * count))
                for name, indices in (("forget", forgotten), ("anchor", anchors), ("calibration", calibration)):
                    np.testing.assert_array_equal(saved[f"{name}_ids"], train_ids[indices])
                original = parent["weights"]["model"]
                close("original_weights", refs["original_weights"], original, 0)
                close("saved_original_weights", saved["original_weights"], original, 0)
                expected_original = original_probabilities[parent_path]
                for name, indices in (("anchor", anchors), ("calibration", calibration)):
                    close(f"{name}_teacher", refs[f"{name}_probabilities"], expected_original[indices])
                    close(f"{name}_teacher_unchanged", saved[f"{name}_probabilities"], refs[f"{name}_probabilities"], 0)
                target = expected_original[calibration].mean(axis=0)**unlearning["beta"] if method == "guided" else np.ones(3)
                target[2] = 0
                target /= target.sum()
                close("forget_target", saved["target"], target)
                close("reported_target", run["metrics"]["target"], target)

                progress = load_json(run["path"] / "checkpoints/model.resume.json")
                checkpoint = load_json(run["path"] / "checkpoints/model.json")
                weights = load_arrays(run["path"] / "checkpoints/model.npz")
                assert progress["step"] == 100 and not progress["stopped"]
                assert checkpoint["selection_metric"] == progress["selection_metric"] == "objective"
                close("best_weights", weights["best_weights"], run["weights"]["model"], 0)
                close("progress_best_weights", progress["best_weights"], weights["best_weights"], 0)
                close("progress_last_weights", progress["weights"], weights["last_weights"], 0)
                history = run["history"].set_index("step")
                assert list(history.index) == list(range(0, 101, 10))
                assert checkpoint["best_step"] == progress["best_step"] == int(history.objective.idxmin())
                close("best_score", checkpoint["best_score"], history.objective.min())
                # Independently evaluate the fixed teacher objective at initialization,
                # selected checkpoint, and final update, without running optimization.
                evaluated = {}
                for step, theta in ((0, original), (checkpoint["best_step"], weights["best_weights"]),
                                    (100, weights["last_weights"])):
                    if step not in evaluated:
                        probabilities = expected_original if step == 0 else paper_probabilities(arrays["train"], theta)
                        evaluated[step] = paper_loss_parts(probabilities, forgotten, anchors, target,
                            refs["anchor_probabilities"], theta, original, unlearning["alpha"], unlearning["regularization"])
                    for name, value in evaluated[step].items():
                        close(f"history_{name}", history.loc[step, name], value)
                test_probabilities = paper_probabilities(arrays["test"], weights["best_weights"])
                close("saved_test_probabilities", run["predictions"]["model__test"], test_probabilities)
                # Reproduce only sample draws, not training: verifies the saved RNG
                # consumed exactly the declared 100 paired mini-batch updates.
                random = np.random.default_rng(config["seed"] + 2000)
                for _ in range(100):
                    random.choice(len(forgotten), min(10, len(forgotten)), replace=False)
                    random.choice(len(anchors), min(10, len(anchors)), replace=False)
                assert random.bit_generator.state == progress["streams"]["minibatch_rng"]
                methods[method] += 1
                best_steps[checkpoint["best_step"]] += 1
                completed_steps[progress["step"]] += 1
                runs[path] = (run, refs)
            pairs.add(tuple(pair))

    for guided, uniform in pairs:
        guided_run, guided_refs = runs[guided]
        uniform_run, uniform_refs = runs[uniform]
        assert guided_run["manifest"]["identity"]["parent_run"] == uniform_run["manifest"]["identity"]["parent_run"]
        for key in guided_refs:
            np.testing.assert_array_equal(guided_refs[key], uniform_refs[key])
    assert len(conditions) == 65 and len(runs) == 130 and len(pairs) == 65
    assert source_fingerprint() == current_source
    return {"all_checks_passed": True, "report_provenance": str(Path(provenance_file).resolve()),
            "source_fingerprint": current_source, "conditions": len(conditions),
            "unique_unlearning_runs": len(runs), "unique_parents": len(parents),
            "paired_methods": len(pairs), "methods": dict(methods),
            "best_step_counts": dict(best_steps), "completed_step_counts": dict(completed_steps),
            "max_absolute_errors": dict(error_bounds),
            "checks": ["saved artifact hashes and exact source snapshots", "paper circuit evaluated independently in NumPy",
                       "fixed original teacher probabilities and Eq. 6 targets", "training-only forget/calibration/anchor IDs",
                       "identical original weights and references for guided/uniform", "Eq. 7 objective and checkpoint selection",
                       "saved best-weight test predictions", "100-update minibatch RNG accounting"],
            "limitations": ["Numerical verification, not a formal proof or certified deletion guarantee",
                            "Authors' private source code unavailable; Eq. 5 shorthand differs from our expectation-level chain rule",
                            "MNIST protocol and unspecified optimization choices are documented adaptations"],
            "research_training_executed": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = audit(args.provenance)
    save_json(args.output, report)
    print(json.dumps(report, indent=2))
