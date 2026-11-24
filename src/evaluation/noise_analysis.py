"""Matched-noise comparisons, descriptive fold summaries and portable exports."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.artifacts import save_json, load_json
from src.evaluation.metrics import classification_metrics, compare_to_retraining
from src.evaluation.plots import save_figure
from src.noise_study import load_noise_predictions


METRICS = ("retained_accuracy", "forget_recall", "forget_confidence", "forgotten_mass_on_retained",
           "recall_digit_0", "recall_digit_1", "recall_digit_2", "conditional_kl_mean", "full_kl_mean",
           "mean_prediction_margin", "mean_abs_expectation")


def _scores(arrays):
    metrics = classification_metrics(arrays["labels"], arrays["probabilities"])
    ordered = np.sort(arrays["probabilities"], axis=1)
    return {**{key: value for key, value in metrics.items() if not isinstance(value, (list, dict))},
            **{f"recall_digit_{digit}": metrics["recall_by_class"][str(digit)] for digit in range(3)},
            "mean_prediction_margin": float(np.mean(ordered[:, -1] - ordered[:, -2])),
            "mean_abs_expectation": float(np.mean(np.abs(arrays["expectations"])))}


def _fold_summary(frame, groups, metrics, planned_folds):
    rows = []
    if frame.empty:
        return pd.DataFrame()
    if frame.duplicated(groups + ["fold"]).any():
        raise ValueError("Duplicate folds would receive extra statistical weight.")
    for keys, group in frame.groupby(groups, dropna=False):
        for metric in metrics:
            values = group[metric].dropna()
            rows.append({**dict(zip(groups, keys)), "metric": metric, "mean": values.mean(),
                         "fold_sd": values.std(ddof=1), "n_folds": len(values),
                         "planned_folds": len(planned_folds),
                         "complete_folds": set(group.loc[group[metric].notna(), "fold"]) == planned_folds})
    return pd.DataFrame(rows)


def noise_tables(index_file, root):
    index, predictions = load_noise_predictions(index_file, root)
    protocol = index["protocol"]
    modes = ["exact" if n is None else f"shots_{n}" for n in protocol["shots"]]
    rows, coverage, scores = [], [], {}
    for binding in protocol["bindings"]:
        for setting in protocol["settings"]:
            for mode in modes:
                key = (binding["run_id"], setting["family"], setting["level"], mode)
                reference_key = (binding["reference_id"], setting["family"], setting["level"], mode)
                present = key in predictions and reference_key in predictions
                coverage.append({**binding, **setting, "mode": mode, "completed": present})
                if not present:
                    continue
                record, arrays = predictions[key]
                _, reference = predictions[reference_key]
                np.testing.assert_array_equal(arrays["sample_ids"], reference["sample_ids"])
                np.testing.assert_array_equal(arrays["labels"], reference["labels"])
                if key not in scores:
                    scores[key] = _scores(arrays)
                comparison = compare_to_retraining(arrays["labels"], reference["probabilities"], arrays["probabilities"])
                rows.append({**binding, "family": setting["family"], "level": setting["level"],
                             "mode": mode, "shots": record["shots"], "seed": protocol["seed"],
                             "one_qubit_strength": record["one_qubit_strength"],
                             "two_qubit_strength": record["two_qubit_strength"],
                             "readout_flip_probability": record["readout_flip_probability"],
                             **scores[key], **comparison})
    summary = pd.DataFrame(rows)
    groups = ["condition", "attack", "poison_rate", "role", "family", "level", "mode"]
    folds = {binding["fold"] for binding in protocol["bindings"]}
    means = _fold_summary(summary, groups, METRICS, folds)
    changes, effects = [], []
    if not summary.empty:
        lookup = {(r.condition, r.fold, r.role, r.family, r.level, r.mode): r for r in summary.itertuples()}
        for row in summary.itertuples():
            baseline = lookup.get((row.condition, row.fold, row.role, "none", 0.0, row.mode))
            if baseline is not None:
                changes.append({**{key: getattr(row, key) for key in (*groups, "fold", "run_id")},
                                **{f"delta_{key}": getattr(row, key) - getattr(baseline, key) for key in METRICS}})
            if row.role not in ("guided", "uniform"):
                continue
            original = lookup.get((row.condition, row.fold, "original", row.family, row.level, row.mode))
            original_zero = lookup.get((row.condition, row.fold, "original", "none", 0.0, row.mode))
            if original is not None and baseline is not None and original_zero is not None:
                effects.append({**{key: getattr(row, key) for key in (*groups, "fold", "run_id")},
                    **{f"unlearning_delta_{key}": getattr(row, key) - getattr(original, key) for key in METRICS},
                    **{f"noise_interaction_{key}": (getattr(row, key) - getattr(original, key)) -
                            (getattr(baseline, key) - getattr(original_zero, key)) for key in METRICS}})
    changes, effects = pd.DataFrame(changes), pd.DataFrame(effects)
    return index, {"summary": summary, "fold_summary": means, "coverage": pd.DataFrame(coverage),
                   "noise_changes": changes, "unlearning_effects": effects,
                   "noise_change_fold_summary": _fold_summary(changes, groups, [f"delta_{m}" for m in METRICS], folds),
                   "unlearning_effect_fold_summary": _fold_summary(effects, groups,
                         [f"unlearning_delta_{m}" for m in METRICS] + [f"noise_interaction_{m}" for m in METRICS], folds)}


def noise_figures(tables, conditions=("clean", "label_flip_50pct", "feature_randomization_50pct")):
    """Selected-condition curves; every condition/rate remains in the exported tables."""
    frame = tables["summary"]
    if frame.empty:
        return {}
    conditions = [name for name in conditions if name in set(frame.condition)]
    if not conditions:
        return {}
    figures = {}
    metrics = ("retained_accuracy", "forget_recall", "forget_confidence", "conditional_kl_mean")
    colors = {"original": "tab:gray", "guided": "tab:blue", "uniform": "tab:orange", "retrained": "tab:green"}
    for mode in sorted(frame["mode"].unique()):
        for family in sorted(set(frame.family) - {"none"}):
            figure, axes = plt.subplots(len(conditions), 4, figsize=(16, 3.4 * len(conditions)), squeeze=False)
            for row_axes, condition in zip(axes, conditions):
                selected = frame[(frame.condition == condition) & (frame["mode"] == mode) &
                                 frame.family.isin([family, "none"])]
                for axis, metric in zip(row_axes, metrics):
                    for role in ("original", "guided", "uniform", "retrained"):
                        stats = selected[selected.role == role].groupby("level")[metric].agg(["mean", "std"])
                        axis.errorbar(stats.index, stats["mean"], yerr=stats["std"].fillna(0),
                                      marker="o", capsize=3, color=colors[role], label=role)
                    axis.set(xlabel="Noise level p", ylabel=metric.replace("_", " "), title=condition.replace("_", " "))
                    if metric != "conditional_kl_mean":
                        axis.set_ylim(-0.03, 1.03)
                    axis.grid(alpha=0.2)
            handles, labels = axes[0, 0].get_legend_handles_labels()
            figure.legend(handles, labels, loc="lower center", ncol=4)
            figure.suptitle(f"Frozen weights: {family}; {mode}; means +/- descriptive fold SD; seed 6304")
            figure.tight_layout(rect=(0, 0.045, 1, 0.96))
            figures[f"noise_{family}_{mode}"] = figure
    # A complete condition/rate view supplements the selected-condition curves.
    available = frame[(frame.family == "combined") & (frame["mode"] == "exact")]
    if not available.empty:
        level = min(available.level.unique(), key=lambda p: abs(p - 0.01))
        selected = available[available.level == level]
        names = selected[["condition", "attack", "poison_rate"]].drop_duplicates().sort_values(["attack", "poison_rate"]).condition
        figure, axes = plt.subplots(1, 4, figsize=(14, max(4, 0.35 * len(names))), squeeze=False)
        for axis, role in zip(axes[0], ("original", "guided", "uniform", "retrained")):
            heat = selected[selected.role == role].groupby("condition")[[f"recall_digit_{d}" for d in range(3)]].mean().reindex(names)
            values = heat.to_numpy()
            shown = axis.imshow(values, vmin=0, vmax=1, cmap="viridis", aspect="auto")
            axis.set_xticks(range(3), ["Digit 0", "Digit 1", "Digit 2"])
            axis.set_yticks(range(len(names)), names if axis is axes[0, 0] else [])
            axis.set_title(role)
            for r in range(len(names)):
                for c in range(3):
                    if np.isfinite(values[r, c]):
                        axis.text(c, r, f"{values[r,c]:.2f}", ha="center", va="center", fontsize=8,
                                  color="white" if values[r, c] < 0.55 else "black")
        figure.colorbar(shown, ax=list(axes[0]), label="Mean class recall", fraction=0.025)
        figure.suptitle(f"All poisoning conditions: combined noise p={level:g}; exact density; seed 6304")
        figure.subplots_adjust(left=0.22, right=0.88, bottom=0.1, top=0.88, wspace=0.25)
        figures["noise_all_conditions_class_recall"] = figure
    return figures


def export_noise_study(index_file, root, figures=None, *, tables=None, destination=None):
    if tables is None:
        index, tables = noise_tables(index_file, root)
    else:
        index = load_json(index_file)
    destination = Path(destination) if destination is not None else Path(index_file).parent / "analysis"
    destination.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        if len(frame.columns):
            frame.to_csv(destination / f"{name}.csv", index=False)
    for name, figure in (figures or {}).items():
        save_figure(figure, destination / "figures" / name,
                    source_run_ids=sorted(tables["summary"].run_id.unique()),
                    description="Frozen-weight noise robustness; one seed; descriptive fold SD.")
    coverage = tables["coverage"]
    save_json(destination / "provenance.json", {"protocol": index["protocol"], "complete": index["complete"],
              "planned_rows": len(coverage), "completed_rows": int(coverage.completed.sum()),
              "training_updates": 0, "matched_noise_retraining_reference": True,
              "limitations": ["Synthetic noise, not an IBM calibration or hardware execution.",
                              "Inference robustness, not training/unlearning with noise in the optimizer.",
                              "Fold SD is descriptive; shot draws are not independent training seeds.",
                              "Noise can lower forgotten-class confidence without successful unlearning."]})
    return destination
