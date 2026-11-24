"""Fresh Phase 1 parents/references/teachers with storage-backed per-model recovery.

The shared stage routines use the normalized-input protocol and separate
checkpoint namespaces. Completed studies can be reviewed across machines;
optimizer continuation retains strict source and runtime matching.
"""

from contextlib import contextmanager, ExitStack
from copy import copy
from dataclasses import asdict, replace
from pathlib import Path
from threading import Lock
from unittest.mock import patch
from uuid import uuid4

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src import artifacts, experiment_stages as stages
from src.ablation_conditions import _parallel
from src.ablations import (AblationProblem, execution_environment, problem_from_runs,
                           parameter_settings)
from src.checkpoints import CheckpointSession
from src.checkpoint_storage import CheckpointMirror
from src.config import (DEFAULT_SEED, BackendConfig, CircuitConfig, DataConfig, ExperimentConfig,
                        PoisonConfig, TrainingConfig, UnlearningConfig)
from src.data.feature_randomization import (INPUT_POLICY, FeaturePoisonConfig, UnitInputPreprocessor,
                                           experiment_config, feature_variant)
from src.data.poisoning import forget_mask
from src.models.parallel_aer import AerBatchService, GpuMonitor
from src.models.vqc import VQC
from src.unlearning.targets import prepare_references


PRIMARY_RATES = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50)
RATES = tuple(sorted((*PRIMARY_RATES, 0.25)))  # 25% supplies the requested OFAT inputs.
FRESH_STAGES = ("fresh_supervised", "fresh_unlearning")


def base_config():
    return ExperimentConfig(seed=DEFAULT_SEED,
        data=DataConfig(train_per_class=140, test_per_class=None, split_strategy="stratified_kfold",
                        n_folds=5, validation_fraction=0.2, seed=DEFAULT_SEED),
        backend=BackendConfig(shots=None, method="statevector", device="auto"),
        circuit=CircuitConfig(reuploads=2, ansatz_reps=3), training=TrainingConfig(steps=300),
        unlearning=UnlearningConfig(training=TrainingConfig(steps=100)), classical_baselines=True)


def condition_configs(base, folds=(0, 1, 2, 3, 4), rates=RATES):
    folds, rates = tuple(folds), tuple(rates)
    if (base.seed != DEFAULT_SEED or base.data.seed != DEFAULT_SEED or base.backend.shots is not None
            or any(base.backend.noise_rates) or base.backend.use_ibm_hardware):
        raise ValueError("Fresh Phase 1 uses seed6304 and noiseless exact local Aer.")
    if not folds or len(set(folds)) != len(folds) or not set(folds).issubset(range(base.data.n_folds)):
        raise ValueError("Choose unique valid folds.")
    if not rates or len(set(rates)) != len(rates) or any(not 0 < rate <= 1 for rate in rates):
        raise ValueError("Choose unique positive poisoning rates <=1.")
    parameter_settings(base.unlearning)
    result = {}
    for fold in folds:
        config = replace(base, data=replace(base.data, fold=fold), poison=PoisonConfig(),
                         label_shuffle_control=False)
        result[fold, "clean"] = config
        for attack in ("label_flip", "feature_randomization"):
            for rate in sorted(rates):
                poison = (FeaturePoisonConfig(rate=rate, seed=DEFAULT_SEED) if attack == "feature_randomization"
                          else PoisonConfig(kind="label_flip", rate=rate, seed=DEFAULT_SEED))
                result[fold, f"{attack}_{rate * 100:g}pct"] = replace(config, poison=poison)
    return result


def study_plan(base=None, folds=(0, 1, 2, 3, 4), rates=RATES):
    configs = condition_configs(base or base_config(), folds, rates)
    rows = []
    for (fold, name), config in configs.items():
        roles = ["original", "guided", "uniform"]
        if name == "clean" or config.poison.kind == "label_flip":
            roles.insert(1, "retrained")
        for role in roles:
            rows.append({"fold": fold, "condition": name, "role": role, "seed": DEFAULT_SEED,
                         "stage": "supervised" if role in ("original", "retrained") else "unlearning",
                         "updates": config.training.steps if role in ("original", "retrained") else config.unlearning.training.steps})
    return pd.DataFrame(rows)


def _blocked(*args, **kwargs):
    raise RuntimeError("Fresh execution permits only its own stages and shared-service models.")


@contextmanager
def fresh_execution(root, service, label, review_guard=None):
    """Preserve original routines; change their storage, preprocessing and model factory."""
    from src.evaluation.saved_results import protect_review
    import src.checkpoints as checkpoints
    root = Path(root).resolve()
    if review_guard is not None:
        review_guard.close()
    try:
        with ExitStack() as stack:
            opened, create = CheckpointSession.open, artifacts.create_run
            fit, unlearn = stages.fit_vqc, stages.unlearn
            lock = Lock()
            mapping = {root / "results/runs" / old: root / "results/runs" / new for old, new in
                       zip(("supervised_models", "unlearning"), FRESH_STAGES)}
            allowed = set(mapping.values())
            context = {"protocol_label": label, "input_policy": INPUT_POLICY,
                       "execution": asdict(service.execution), "environment": execution_environment()}
            source = artifacts.source_fingerprint()

            def identity(stage, config, **details):
                return {"stage": stage, "config": config, "code_fingerprint": source,
                        "environment": context["environment"], "fresh_protocol": context, **details}

            def redirected_open(session, directory, identity, **kwargs):
                if Path(directory).resolve() not in mapping or identity.get("fresh_protocol") != context:
                    return _blocked()
                with lock:
                    return opened(session, mapping[Path(directory).resolve()], identity, **kwargs)

            def restricted_create(directory, *args, **kwargs):
                if Path(directory).resolve() not in allowed:
                    return _blocked()
                return create(directory, *args, **kwargs)

            def factory(circuit, backend, seed=DEFAULT_SEED):
                if circuit != service.circuit_config or backend != service.backend_config or seed != DEFAULT_SEED:
                    return _blocked()
                return service.model(seed)

            def mirrored(function):
                def execute(model, *args, checkpoint_path, callback=None, **kwargs):
                    if (getattr(model.executor, "service", None) is not service
                            or not any(Path(checkpoint_path).resolve().is_relative_to(p) for p in allowed)):
                        return _blocked()
                    mirror = CheckpointMirror(checkpoint_path)
                    def report(row):
                        if row["step"] == 0:
                            mirror.flush_now()
                        if mirror.error is not None:
                            raise RuntimeError("storage backup failed; stop with saved progress.") from mirror.error
                        if callback:
                            callback(row)
                    with mirror as local:
                        return function(model, *args, checkpoint_path=local, callback=report, **kwargs)
                return execute

            stack.enter_context(patch.object(CheckpointSession, "open", redirected_open))
            stack.enter_context(patch.object(artifacts, "create_run", restricted_create))
            stack.enter_context(patch.object(checkpoints, "create_run", restricted_create))
            stack.enter_context(patch.object(stages, "_identity", identity))
            stack.enter_context(patch.object(stages, "ImagePreprocessor", UnitInputPreprocessor))
            stack.enter_context(patch.object(stages, "VQC", factory))
            stack.enter_context(patch.object(stages, "fit_vqc", mirrored(fit)))
            stack.enter_context(patch.object(stages, "unlearn", mirrored(unlearn)))
            for target in ("src.experiments.run_clean_vqc", "src.experiments.run_classical_cv",
                           "src.experiment_stages.train_condition", "src.experiment_stages.unlearn_condition",
                           "src.data.dataset.download_mnist",
                           "src.checkpoints.retire_pilot_checkpoints"):
                stack.enter_context(patch(target, _blocked))
            stack.enter_context(threadpool_limits(limits=1, user_api="blas"))
            yield
    finally:
        if review_guard is not None:
            review_guard.enter_context(protect_review())


def benchmark_problem(config, splits):
    """Untrained initialization for performance probes; not a research checkpoint."""
    processor = UnitInputPreprocessor(DEFAULT_SEED).fit(splits.train.images, splits.train.sample_ids)
    arrays = {name: processor.transform(getattr(splits, name).images) for name in ("train", "validation", "test")}
    model = VQC(config.circuit, replace(config.backend, device="CPU", max_parallel_threads=1), DEFAULT_SEED)
    weights = np.random.default_rng(DEFAULT_SEED).normal(0.0, config.training.initialization_std, model.n_parameters)
    refs = prepare_references(model, arrays["train"], splits.train.labels, splits.train.labels == 2,
                              weights, config.unlearning, DEFAULT_SEED)
    parent = {"path": Path("untrained_probe_" + artifacts.fingerprint({"split": splits.fingerprint(), "policy": INPUT_POLICY,
                      "circuit": asdict(config.circuit)})[:16]),
              "predictions": {"model__test": model.predict_proba(arrays["test"], weights)}}
    return AblationProblem("untrained_performance_probe", config.data.fold, config, parent, {}, {}, {}, arrays, refs)


def run_fresh_study(root, splits, execution, *, base=None, rates=RATES,
                    label="feature_randomization_v1", force_retrain=False, progress=None):
    """Fresh once; subsequent Run All loads completed fits and resumes unfinished fits."""
    root = Path(root).resolve()
    base = base or base_config()
    configs = condition_configs(base, tuple(splits), rates)
    variants = {key: feature_variant(splits[key[0]].train, config.poison) for key, config in configs.items()}
    protocol = {"version": 1, "label": label, "input_policy": INPUT_POLICY,
                "configs": {f"fold_{fold}_{name}": config.to_dict() for (fold, name), config in configs.items()},
                "splits": {str(f): s.fingerprint() for f, s in splits.items()},
                "source_fingerprint": artifacts.source_fingerprint(), "environment": execution_environment(),
                "execution": asdict(execution), "historical_weight_imports": False}
    if force_retrain:
        protocol["forced_attempt"] = uuid4().hex
        label += "_" + protocol["forced_attempt"]
    protocol["run_label"] = label
    destination = root / "results/fresh_studies" / artifacts.fingerprint(protocol)[:16]
    index_file = destination / "study.json"
    if index_file.exists():
        saved = artifacts.load_json(index_file)
        if saved["protocol"] != artifacts.json_value(protocol):
            raise ValueError("Fresh study identity conflict.")
        if saved["complete"]:
            return FreshStudy(root, index_file), index_file
    index = {"protocol": protocol, "complete": False, "conditions": {},
             "plan": study_plan(base, tuple(splits), rates).to_dict("records")}
    artifacts.save_json(index_file, index)
    session = CheckpointSession(force_retrain=force_retrain)
    def callback(cancel, fold, name, role, updates):
        def report(row):
            if cancel.is_set():
                raise InterruptedError("Fresh study stopped at a committed update.")
            step = row.get("step")
            if progress and (step is None or step in (0, 1, updates) or step % 10 == 0):
                progress({**row, "fold": fold, "condition": name, "role": role, "updates": updates})
        return report
    with AerBatchService(base.circuit, base.backend, execution) as service, GpuMonitor() as monitor:
        try:
            with fresh_execution(root, service, label):
                jobs, aliases, unique = [], {}, {}
                # Interleave folds for each independent fitting problem.
                ordered = sorted(configs, key=lambda key: (key[1], key[0]))
                for fold, name in ordered:
                    config, variant = configs[fold, name], variants[fold, name]
                    roles = ("original", "retrained") if name == "clean" or config.poison.kind == "label_flip" else ("original",)
                    for role in roles:
                        mask = forget_mask(variant, config.unlearning.forget_class, config.unlearning.forget_definition)
                        data = variant.observed if role == "original" else splits[fold].train.subset(~mask)
                        if name == "clean" and role == "original":
                            data = splits[fold].train
                        problem_key = (fold, data.fingerprint(), role == "retrained")
                        key = (fold, name, role)
                        if problem_key in unique:
                            aliases[key] = unique[problem_key]
                            continue
                        unique[problem_key] = key
                        def supervised(cancel, fold=fold, name=name, role=role, data=data, config=config):
                            return stages._supervised(config, splits[fold], data, role == "retrained", allow_training=True,
                                root=root, role="poisoned" if role == "original" else "retrained_poisoned", checkpoints=session,
                                callback=callback(cancel, fold, name, role, config.training.steps))
                        jobs.append((key, supervised))
                jobs.sort(key=lambda item: (item[0][1], item[0][2] != "original", item[0][0]))
                trained = _parallel(jobs, execution.workers, progress)
                trained.update({key: trained[original] for key, original in aliases.items()})
                for (fold, name), config in configs.items():
                    if config.poison.kind == "feature_randomization":
                        trained[fold, name, "retrained"] = trained[fold, "clean", "retrained"]
                jobs, aliases, unique = [], {}, {}
                for method in ("guided", "uniform"):
                    for fold, name in ordered:
                        config, variant = configs[fold, name], variants[fold, name]
                        mask = forget_mask(variant, config.unlearning.forget_class, config.unlearning.forget_definition)
                        data = splits[fold].train if name == "clean" else variant.observed
                        key = (fold, name, method)
                        problem_key = (trained[fold, name, "original"]["path"], trained[fold, name, "retrained"]["path"], method)
                        if problem_key in unique:
                            aliases[key] = unique[problem_key]
                            continue
                        unique[problem_key] = key
                        def unlearning(cancel, fold=fold, name=name, config=config, mask=mask, data=data, method=method):
                            return stages._unlearn_stage(trained[fold, name, "original"], trained[fold, name, "retrained"],
                                config, splits[fold], data, mask, method, allow_execution=True, root=root,
                                role="poisoned_" + method, checkpoints=session,
                                callback=callback(cancel, fold, name, method, config.unlearning.training.steps))
                        jobs.append((key, unlearning))
                unlearned = _parallel(jobs, execution.workers, progress)
                unlearned.update({key: unlearned[original] for key, original in aliases.items()})
                for (fold, name), config in configs.items():
                    models = {"clean": trained[fold, "clean", "original"], "retrained_clean": trained[fold, "clean", "retrained"],
                              "poisoned": trained[fold, name, "original"], "retrained_poisoned": trained[fold, name, "retrained"]}
                    methods = {f"{case}_{method}": unlearned[fold, "clean" if case == "clean" else name, method]
                               for case in ("clean", "poisoned") for method in ("guided", "uniform")}
                    variant = variants[fold, name]
                    split_path = artifacts.save_split_manifest(splits[fold], root / "data/splits")
                    annotated = artifacts.save_poisoned_dataset(variant, root / "data/annotated")
                    manifest = {"config": config.to_dict(), "split_fingerprint": splits[fold].fingerprint(),
                                "poisoning": variant.metadata(), "input_policy": INPUT_POLICY,
                                "split_manifest": split_path.relative_to(root).as_posix(),
                                "annotated_dataset": annotated.relative_to(root).as_posix(),
                                **{group: {role: run["path"].relative_to(root).as_posix() for role, run in runs.items()}
                                   for group, runs in (("models", models), ("unlearned", methods))}}
                    manifest_path = destination / "conditions" / f"fold_{fold}_{name}.json"
                    artifacts.save_json(manifest_path, manifest)
                    index["conditions"][f"fold_{fold}_{name}"] = {"path": manifest_path.relative_to(root).as_posix(),
                                                              "sha256": artifacts._file_digest(manifest_path)}
                index["complete"] = True
        finally:
            artifacts.save_json(index_file, index)
            if monitor.rows:
                pd.DataFrame(monitor.rows).to_csv(destination / "gpu_telemetry.csv", index=False)
    return FreshStudy(root, index_file), index_file


class FreshStudy:
    """Portable, checked view accepted by the existing ablation and noise pipelines."""

    def __init__(self, root, index_file, *, progress=None):
        self.root, self.report_path = Path(root).resolve(), Path(index_file).resolve()
        self.index = artifacts.load_json(self.report_path)
        if not self.index["complete"]:
            raise ValueError("Fresh study is incomplete; enable its run stage to load/resume each model.")
        self.source_fingerprint = self.index["protocol"]["source_fingerprint"]
        if progress:
            progress("Verifying saved conditions and model artifacts; no fitting.")
        self.conditions, self.runs, self.split_paths = {}, {}, {}
        for key, entry in self.index["conditions"].items():
            manifest_path = self._read_path(self.root / entry["path"])
            if artifacts._file_digest(manifest_path) != entry["sha256"]:
                raise ValueError("Fresh condition manifest checksum mismatch.")
            condition = artifacts.load_json(manifest_path)
            config = experiment_config(condition["config"])
            protocol = self.index["protocol"]
            if (condition["config"] != protocol["configs"].get(key)
                    or condition["split_fingerprint"] != protocol["splits"].get(str(config.data.fold))):
                raise ValueError("Fresh condition config/split differs from the registered study.")
            if condition["input_policy"] != INPUT_POLICY:
                raise ValueError("Fresh input policy mismatch.")
            for group in ("models", "unlearned"):
                paths = condition[group]
                loaded = {}
                for role, relative in paths.items():
                    path = self._read_path(self.root / relative)
                    if path not in self.runs:
                        self.runs[path] = artifacts.load_run(path)
                        if progress and len(self.runs) % 10 == 0:
                            progress(f"Verified {len(self.runs)} unique model runs.")
                    loaded[role] = self.runs[path]
                    identity = loaded[role]["manifest"]["identity"]
                    context = identity.get("fresh_protocol", {})
                    if (context.get("input_policy") != INPUT_POLICY
                            or context.get("protocol_label") != protocol["run_label"]
                            or identity["code_fingerprint"] != protocol["source_fingerprint"]
                            or context.get("execution") != protocol["execution"]
                            or identity["environment"] != protocol["environment"]):
                        raise ValueError("A historical/foreign run was substituted into the fresh study.")
                    if group == "models" and artifacts.fingerprint(loaded[role]["config"]) != artifacts.fingerprint(stages._fit_config(config)):
                        raise ValueError("Fresh supervised configuration mismatch.")
                condition[group] = loaded
            condition["manifest_path"] = manifest_path
            self.conditions[key] = condition
            fold = config.data.fold
            split_path = self._read_path(self.root / condition["split_manifest"])
            split = artifacts.load_json(split_path)
            if split["fingerprint"] != condition["split_fingerprint"]:
                raise ValueError("Split fingerprint mismatch.")
            for run in (*condition["models"].values(), *condition["unlearned"].values()):
                identity = run["manifest"]["identity"]
                # Supervised IDs carry the split directly; unlearning links are
                # verified below against their exact completed parents/references.
                if "split_fingerprint" in identity and identity["split_fingerprint"] != split["fingerprint"]:
                    raise ValueError("Fresh model belongs to a different split.")
                np.testing.assert_array_equal(run["predictions"]["test_sample_ids"], split["test"]["sample_ids"])
                np.testing.assert_array_equal(run["predictions"]["test_labels"], split["test"]["labels"])
            if fold in self.split_paths and self.split_paths[fold] != split_path:
                raise ValueError("Conflicting fold splits.")
            self.split_paths[fold] = split_path
            for case in ("clean", "poisoned"):
                parent, reference = (condition["models"][r] for r in (case, f"retrained_{case}"))
                for method in ("guided", "uniform"):
                    identity = condition["unlearned"][f"{case}_{method}"]["manifest"]["identity"]
                    if identity["parent_run"] != parent["path"].name or identity["reference_run"] != reference["path"].name:
                        raise ValueError("Fresh unlearning parent/reference links disagree.")
        if progress:
            progress(f"Verified {len(self.runs)} unique fits across {len(self.split_paths)} folds.")

    def _read_path(self, path):
        path = Path(path).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Fresh artifact path leaves the project.")
        return path

    def select_rates(self, rates=PRIMARY_RATES):
        """Keep original primary/noise coverage; 25% remains separate ablation support."""
        selected = copy(self)
        selected.conditions = {key: condition for key, condition in self.conditions.items()
                               if condition["config"]["poison"]["kind"] == "none"
                               or condition["config"]["poison"]["rate"] in rates}
        return selected

    def ablation_problems(self, folds=None):
        folds = set(self.split_paths if folds is None else folds)
        problems = []
        for condition in self.conditions.values():
            config = experiment_config(condition["config"])
            if config.data.fold not in folds or (config.poison.kind != "none" and config.poison.rate not in (.25, .5)):
                continue
            case = "clean" if config.poison.kind == "none" else "poisoned"
            name = "clean" if case == "clean" else f"{config.poison.kind}_{config.poison.rate * 100:g}pct"
            problems.append(problem_from_runs(name, config.data.fold, config,
                condition["models"][case], condition["models"][f"retrained_{case}"],
                condition["unlearned"][f"{case}_guided"], condition["unlearned"][f"{case}_uniform"],
                baseline_origin="fresh_protocol"))
        if len(problems) != 5 * len(folds):
            raise ValueError("Ablations require clean and both attacks at 25%/50% in each selected fold.")
        return problems
