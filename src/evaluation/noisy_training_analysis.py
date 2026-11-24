"""Compare noisy-trained and historical models under matched deployment noise."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.artifacts import _file_digest, load_arrays, load_json, load_run, save_json
from src.evaluation.metrics import classification_metrics, compare_to_retraining
from src.evaluation.noise_analysis import _fold_summary
from src.evaluation.plots import save_figure


METRICS = ("retained_accuracy", "forget_recall", "forget_confidence", "forgotten_mass_on_retained",
           "recall_digit_0", "recall_digit_1", "recall_digit_2", "full_kl_mean", "conditional_kl_mean")


def _local(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Noisy-training artifact leaves the project root.")
    return path


def noisy_training_tables(index_file, root):
    root = Path(root).resolve()
    index = load_json(_local(root, index_file))
    old = {key: load_run(_local(root, value["run"])) for key, value in index["protocol"]["historical_inputs"].items()}
    new = {record["run"]: load_run(_local(root, record["run"]))
           for group in ("supervised", "unlearned") for record in index[group].values()}
    evaluations = {}
    for record in index["evaluations"]:
        file = _local(root, record["file"])
        if _file_digest(file) != record["sha256"]:
            raise ValueError("Noisy-training deployment predictions changed.")
        key = (record["regime"], record["deployment"], record["run_id"])
        if key in evaluations:
            raise ValueError("Duplicate noisy-training deployment evaluation.")
        evaluations[key] = load_arrays(file)
    def select(binding, regime):
        if regime == "historical":
            return old[binding["run_id"]], old[binding["reference_id"]]
        if binding["role"] in ("original", "retrained"):
            candidate = index["supervised"].get(binding["run_id"])
        else:
            candidate = index["unlearned"].get(f"fold_{binding['fold']}_{binding['condition']}_{binding['role']}")
        reference = index["supervised"].get(binding["reference_id"])
        return (new[candidate["run"]] if candidate else None, new[reference["run"]] if reference else None)
    def arrays(run, regime, deployment):
        if (regime == "historical" and deployment == "ideal") or (regime == "noisy_trained" and deployment == "noisy"):
            pred = run["predictions"]
            return {"probabilities": pred["model__test"], "labels": pred["test_labels"], "sample_ids": pred["test_sample_ids"]}
        return evaluations.get((regime, deployment, run["path"].name))
    rows, coverage, targets = [], [], []
    for binding in index["protocol"]["bindings"]:
        for regime in ("historical", "noisy_trained"):
            run, reference = select(binding, regime)
            if run is not None and binding["role"] in ("guided", "uniform"):
                targets.append({**binding, "training_regime": regime, "run_id": run["path"].name,
                                **{f"target_{k}": v for k, v in enumerate(run["metrics"]["target"])}})
            for deployment in ("ideal", "noisy"):
                candidate = arrays(run, regime, deployment) if run is not None else None
                matched = arrays(reference, regime, deployment) if reference is not None else None
                present = candidate is not None and matched is not None
                coverage.append({**binding, "training_regime": regime, "deployment": deployment, "completed": present})
                if not present:
                    continue
                np.testing.assert_array_equal(candidate["sample_ids"], matched["sample_ids"])
                np.testing.assert_array_equal(candidate["labels"], matched["labels"])
                scores = classification_metrics(candidate["labels"], candidate["probabilities"])
                comparison = compare_to_retraining(candidate["labels"], matched["probabilities"], candidate["probabilities"])
                rows.append({**binding, "source_historical_run": binding["run_id"], "run_id": run["path"].name,
                             "training_regime": regime, "deployment": deployment, "seed": index["protocol"]["seed"],
                             **{k: v for k, v in scores.items() if not isinstance(v, (dict, list))},
                             **{f"recall_digit_{d}": scores["recall_by_class"][str(d)] for d in range(3)}, **comparison})
    summary = pd.DataFrame(rows)
    groups = ["condition", "attack", "poison_rate", "role", "training_regime", "deployment"]
    folds = {b["fold"] for b in index["protocol"]["bindings"]}
    means = _fold_summary(summary, groups, METRICS, folds)
    gains = []
    if not summary.empty:
        lookup = {(r.condition, r.fold, r.role, r.training_regime, r.deployment): r for r in summary.itertuples()}
        for r in summary.loc[summary.training_regime == "noisy_trained"].itertuples():
            baseline = lookup.get((r.condition, r.fold, r.role, "historical", r.deployment))
            if baseline is not None:
                gains.append({**{key: getattr(r, key) for key in ("condition", "attack", "poison_rate", "role", "deployment", "fold")},
                              **{f"gain_{metric}": getattr(r, metric) - getattr(baseline, metric) for metric in METRICS}})
    gains = pd.DataFrame(gains)
    return index, {"summary": summary, "fold_summary": means, "coverage": pd.DataFrame(coverage),
                   "targets": pd.DataFrame(targets), "training_changes": gains,
                   "training_change_fold_summary": _fold_summary(gains,
                       ["condition", "attack", "poison_rate", "role", "deployment"], [f"gain_{m}" for m in METRICS], folds)}


def noisy_training_figures(tables, conditions=("clean", "label_flip_50pct", "feature_randomization_50pct")):
    frame = tables["summary"]
    if frame.empty:
        return {}
    conditions = [c for c in conditions if c in set(frame.condition)]
    if not conditions:
        return {}
    figures = {}
    colors = {"original": "tab:gray", "guided": "tab:blue", "uniform": "tab:orange", "retrained": "tab:green"}
    for deployment in ("ideal", "noisy"):
        figure, axes = plt.subplots(len(conditions), 3, figsize=(15, 3.5 * len(conditions)), squeeze=False)
        for row_axes, condition in zip(axes, conditions):
            selected = frame[(frame.condition == condition) & (frame.deployment == deployment)]
            for axis, metric in zip(row_axes, ("retained_accuracy", "forget_confidence", "conditional_kl_mean")):
                for regime, style, label in (("historical", "--", "Noiseless-trained"), ("noisy_trained", "-", "Noisy-trained")):
                    data = selected[selected.training_regime == regime]
                    stats = data.groupby("role")[metric].agg(["mean", "std"]).reindex(colors)
                    axis.errorbar(range(4), stats["mean"], yerr=stats["std"].fillna(0), fmt="o" + style,
                                  capsize=3, label=label)
                axis.set_xticks(range(4), colors)
                axis.set(title=condition.replace("_", " "), ylabel=metric.replace("_", " "))
                axis.grid(alpha=0.2)
                if metric != "conditional_kl_mean":
                    axis.set_ylim(-0.03, 1.03)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="lower center", ncol=2)
        figure.suptitle(f"Training under noise: {deployment} deployment; seed 6304; descriptive fold SD")
        figure.tight_layout(rect=(0, 0.05, 1, 0.96))
        figures[f"noisy_training_{deployment}_comparison"] = figure
    gains = tables["training_changes"]
    attacks = sorted(set(gains.attack) - {"none"}) if not gains.empty else []
    if attacks:
        figure, axes = plt.subplots(len(attacks), 3, figsize=(15, 3.5 * len(attacks)), squeeze=False)
        for row_axes, attack in zip(axes, attacks):
            selected = gains[(gains.attack == attack) & (gains.deployment == "noisy")]
            for axis, metric in zip(row_axes, ("retained_accuracy", "forget_confidence", "conditional_kl_mean")):
                for role, color in colors.items():
                    stats = selected[selected.role == role].groupby("poison_rate")[f"gain_{metric}"].agg(["mean", "std"])
                    axis.errorbar(stats.index * 100, stats["mean"], yerr=stats["std"].fillna(0), color=color,
                                  marker="o", capsize=3, label=role)
                axis.axhline(0, color="black", linewidth=0.6)
                axis.set(title=attack.replace("_", " "), xlabel="Poisoning: % of eligible class",
                         ylabel="Noisy-trained minus noiseless-trained\n" + metric.replace("_", " "))
                axis.grid(alpha=0.2)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="lower center", ncol=4)
        figure.suptitle("Training under noise: matched noisy deployment; paired changes across all poisoning rates")
        figure.tight_layout(rect=(0, 0.07, 1, 0.94))
        figures["noisy_training_poisoning_curves"] = figure
    return figures


def export_noisy_training(index_file, root, figures=None, *, tables=None, destination=None):
    if tables is None:
        index, tables = noisy_training_tables(index_file, root)
    else:
        index = load_json(index_file)
    destination = Path(destination) if destination is not None else Path(index_file).parent / "analysis"
    destination.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        if len(table.columns):
            table.to_csv(destination / f"{name}.csv", index=False)
    for name, figure in (figures or {}).items():
        save_figure(figure, destination / "figures" / name, source_run_ids=sorted(tables["summary"].run_id.unique()),
                    description="Separate noisy-training study; exact expectations; descriptive fold SD.")
    save_json(destination / "provenance.json", {"protocol": index["protocol"], "complete": index["complete"],
              "completed_model_fits": len(index["supervised"]) + len(index["unlearned"]),
              "planned_model_fits": len(index["protocol"]["plan"]),
              "comparison": "Historical and noisy-trained checkpoints compared under matched ideal/noisy deployment.",
              "limitations": "One seed; descriptive fold SD; synthetic noise; exact noisy expectations rather than finite-shot optimization."})
    return destination
