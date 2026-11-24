"""Portable tables and figures for the focused guided-unlearning ablations."""

from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.ablations import DEFAULTS, SWEEPS, load_study
from src.artifacts import save_json, load_json
from src.evaluation.plots import save_figure


METRICS = ("retained_accuracy", "forget_recall", "forget_confidence", "full_kl_mean",
           "conditional_kl_mean", "recall_digit_0", "recall_digit_1", "recall_digit_2")


def study_tables(index_file, root):
    index, summary = load_study(index_file, root)
    expected_folds = sorted({int(key.split("_")[1]) for key in index["protocol"]["problems"]})
    groups = ["condition", "parameter", "value"]
    if summary.duplicated(groups + ["fold"]).any():
        raise ValueError("Duplicate parameter/fold rows must not receive extra statistical weight.")
    aggregates = []
    for keys, group in summary.groupby(groups, dropna=False):
        for metric in METRICS:
            values = group[metric].dropna()
            aggregates.append({**dict(zip(groups, keys)), "metric": metric, "mean": values.mean(),
                               "fold_sd": values.std(ddof=1), "n_folds": len(values),
                               "expected_folds": len(expected_folds),
                               "complete_folds": set(group.loc[group[metric].notna(), "fold"]) == set(expected_folds)})
    paired = []
    for _, row in summary.loc[~summary.parameter.isin(["baseline", "uniform"])].iterrows():
        baseline = summary[(summary.problem == row.problem) & (summary.parameter == "baseline")]
        if len(baseline) != 1:
            raise ValueError("Every ablation needs one matched historical guided baseline.")
        paired.append({key: row[key] for key in (*groups, "fold", "problem", "run_id")} |
                      {f"delta_{metric}": row[metric] - baseline.iloc[0][metric] for metric in METRICS})
    paired = pd.DataFrame(paired)
    changes = []
    if not paired.empty:
        for keys, group in paired.groupby(groups):
            for metric in METRICS:
                values = group[f"delta_{metric}"].dropna()
                changes.append({**dict(zip(groups, keys)), "metric": metric,
                                "mean_delta": values.mean(), "fold_sd": values.std(ddof=1),
                                "n_folds": len(values), "expected_folds": len(expected_folds)})
    coverage = []
    for problem in index["protocol"]["problems"]:
        for parameter, values in SWEEPS.items():
            for value in values:
                if value == DEFAULTS[parameter]:
                    continue
                match = summary[(summary.problem == problem) & (summary.parameter == parameter)
                                & (summary.value == value)]
                coverage.append({"problem": problem, "parameter": parameter, "value": value,
                                 "completed": len(match) == 1,
                                 "run_id": match.iloc[0].run_id if len(match) else None})
    return index, {"summary": summary, "fold_summary": pd.DataFrame(aggregates),
                   "paired_changes": paired, "paired_fold_summary": pd.DataFrame(changes),
                   "coverage": pd.DataFrame(coverage)}


def comparison_figures(tables):
    """One figure per parameter, plus beta targets; SD is descriptive fold variability."""
    summary = tables["summary"]
    if summary.empty:
        return {}
    conditions = sorted(summary.condition.unique())
    panels = (("retained_accuracy", "Retained accuracy (0/1)"),
              ("forget_recall", "Digit-2 recall"),
              ("forget_confidence", "Mean p(2) on digit 2"),
              ("conditional_kl_mean", "Retained-label KL to retraining"))
    figures = {}
    for parameter, values in SWEEPS.items():
        figure, axes = plt.subplots(len(conditions), len(panels),
                                    figsize=(18, 4 * len(conditions)), squeeze=False)
        for row, condition in enumerate(conditions):
            selected = summary[(summary.condition == condition) & summary.parameter.isin([parameter, "baseline"])].copy()
            selected.loc[selected.parameter == "baseline", "value"] = DEFAULTS[parameter]
            uniform = summary[(summary.condition == condition) & (summary.parameter == "uniform")]
            for column, (metric, label) in enumerate(panels):
                axis = axes[row, column]
                stats = selected.groupby("value")[metric].agg(["mean", "std", "count"]).reindex(values)
                axis.errorbar(values, stats["mean"], yerr=stats["std"].fillna(0),
                              marker="o", capsize=3, label="Guided: mean +/- fold SD")
                if not uniform.empty:
                    axis.axhline(uniform[metric].mean(), color="tab:orange", linestyle="--", label="Matched uniform mean")
                baseline_mean = selected[selected.parameter == "baseline"][metric].mean()
                axis.scatter([DEFAULTS[parameter]], [baseline_mean], marker="*", s=140, color="black", zorder=4,
                             label="Guided default")
                if parameter == "regularization":
                    axis.set_xscale("symlog", linthresh=0.001)
                axis.set_xticks(values, [f"{value:g}" for value in values])
                axis.set_xlabel("lambda" if parameter == "regularization" else parameter)
                axis.set_ylabel(label)
                axis.set_title(condition.replace("_", " "))
                axis.grid(alpha=0.2)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="lower center", ncol=3)
        figure.suptitle(f"One-factor ablation: {parameter}; seed 6304; SD is not a confidence interval")
        figure.tight_layout(rect=(0, 0.07, 1, 0.94))
        figures[f"ablation_{parameter}"] = figure
    figure, axes = plt.subplots(1, len(conditions), figsize=(6 * len(conditions), 4), squeeze=False)
    for axis, condition in zip(axes[0], conditions):
        selected = summary[(summary.condition == condition) & summary.parameter.isin(["beta", "baseline"])].copy()
        selected.loc[selected.parameter == "baseline", "value"] = 1.0
        for digit in (0, 1):
            stats = selected.groupby("value")[f"target_{digit}"].agg(["mean", "std"]).reindex(SWEEPS["beta"])
            axis.errorbar(stats.index, stats["mean"], yerr=stats["std"].fillna(0), marker="o", capsize=3,
                          label=f"Target q({digit})")
        axis.axhline(0.5, color="grey", linestyle="--", label="Uniform target")
        axis.set(xlabel="beta", ylabel="Target probability", title=condition.replace("_", " "))
        axis.legend()
    figure.suptitle("Guided target sharpness; q(2)=0; means +/- descriptive fold SD")
    figure.tight_layout()
    figures["ablation_beta_targets"] = figure
    return figures


def export_study(index_file, root, figures=None, *, tables=None, destination=None):
    root, index_file = Path(root).resolve(), Path(index_file).resolve()
    if tables is None:
        index, tables = study_tables(index_file, root)
    else:
        index = load_json(index_file)
    destination = Path(destination) if destination is not None else index_file.parent / "analysis"
    destination.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        if len(table.columns):
            table.to_csv(destination / f"{name}.csv", index=False)
    for name, figure in (figures or {}).items():
        save_figure(figure, destination / "figures" / name,
                    source_run_ids=tables["summary"].run_id.tolist(), description="Focused OFAT ablation; seed 6304.")
    counts = tables["summary"].status.value_counts().to_dict()
    save_json(destination / "provenance.json", {"original_training_environment_preserved": True,
              "study_index": index_file.relative_to(root).as_posix(), "protocol": index["protocol"],
              "complete": index["complete"], "record_status_counts": counts,
              "planned_new_runs": len(tables["coverage"]),
              "completed_new_runs": int(tables["coverage"].completed.sum()),
              "supervised_models_fitted_in_parameter_sweep": 0,
              "new_condition_input_index": index.get("support_index"),
              "new_condition_input_indexes": index.get("support_indexes", []),
              "interpretation": "Report every tested value. Fold variability is descriptive, not independent-seed uncertainty."})
    return destination


def download_bundle(index_file, root):
    """Incremental portable archive; old supervised files already exist in the main project."""
    root, index_file = Path(root).resolve(), Path(index_file).resolve()
    index = load_json_local(index_file)
    destination = index_file.parent / "ablation_download.zip"
    directories = [index_file.parent]
    if index.get("performance_directory"):
        directories.append(root / index["performance_directory"])
    support_files = set(index.get("support_indexes", []))
    if index.get("support_index"):
        support_files.add(index["support_index"])
    for relative in support_files:
        support_file = root / relative
        support = load_json_local(support_file)
        directories.append(support_file.parent)
        for condition in support["conditions"].values():
            # Historical reference runs already exist in the main project.
            directories.extend(root / relative for relative in condition["runs"].values()
                               if relative.startswith("results/runs/ablation_inputs_"))
            directories.append(root / condition["annotated_dataset"])
    for record in [*index["records"], *index["baseline_checks"]]:
        relative = record.get("run")
        if relative and relative.startswith(("results/runs/ablations/", "results/runs/ablation_validation/")):
            directories.append(root / relative)
    files = {file for directory in directories for file in directory.rglob("*")
             if file.is_file() and file != destination and file.suffix != ".tmp"}
    files.update(root / relative for relative in ("src/ablations.py", "src/models/parallel_aer.py",
                 "src/evaluation/ablation_analysis.py", "src/checkpoint_storage.py", "main_project.ipynb", "README.md"))
    files.update((root / "src").rglob("*.py"))
    with ZipFile(destination, "w", ZIP_DEFLATED) as archive:
        for file in sorted(files):
            if not file.resolve().is_relative_to(root):
                raise ValueError("Bundle file leaves the project root.")
            archive.write(file, file.relative_to(root).as_posix())
    return destination


def load_json_local(path):
    from src.artifacts import load_json
    return load_json(path)
