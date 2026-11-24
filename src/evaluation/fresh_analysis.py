"""Signed-input diagnostics and fresh-protocol summaries; never fabricate metrics."""

from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import artifacts
from src.data.feature_randomization import INPUT_POLICY, UnitInputPreprocessor, feature_variant, unit_inputs
from src.evaluation.analysis import phase1_tables, fold_summaries
from src.evaluation.cached_analysis import cached_phase1_tables
from src.evaluation.plots import save_figure


def input_diagnostics(configs, splits):
    """Changes reaching frozen clean PCA, measured without looking at test outcomes."""
    processors, rows = {}, []
    for (fold, name), config in configs.items():
        if fold not in processors:
            data = splits[fold].train
            processors[fold] = UnitInputPreprocessor(config.seed).fit(data.images, data.sample_ids)
        variant = feature_variant(splits[fold].train, config.poison)
        selected = variant.is_poisoned
        if not selected.any():
            continue
        before, after = variant.original.images[selected], variant.observed.images[selected]
        delta = unit_inputs(after) - unit_inputs(before)
        processor = processors[fold]
        angles = processor.transform(after) - processor.transform(before)
        rows.append({"fold": fold, "condition": name, "attack": config.poison.kind,
                     "poison_rate": config.poison.rate, "n_selected": int(selected.sum()),
                     "mean_input_l2": float(np.linalg.norm(delta, axis=1).mean()),
                     "mean_pca_l2": float(np.linalg.norm(delta @ processor.pca.components_.T, axis=1).mean()),
                     "mean_angle_l2": float(np.linalg.norm(angles, axis=1).mean()),
                     "unchanged_angle_records": int(np.all(np.isclose(angles, 0, atol=1e-12), axis=1).sum()),
                     "realized_eligible_fraction": variant.metadata()["actual_fraction_eligible"],
                     "realized_training_fraction": variant.metadata()["actual_fraction_all_training"]})
    return pd.DataFrame(rows), processors


def plot_feature_examples(variant, count=3):
    selected = np.flatnonzero(variant.is_poisoned)[:count]
    if not len(selected):
        raise ValueError("No selected records at this rate.")
    figure, axes = plt.subplots(len(selected), 3, figsize=(9, 2.8 * len(selected)), squeeze=False, layout="constrained")
    for row, index in zip(axes, selected):
        before = unit_inputs(variant.original.images[index:index+1])[0].reshape(28, 28)
        after = unit_inputs(variant.observed.images[index:index+1])[0].reshape(28, 28)
        row[0].imshow(before, cmap="gray", vmin=0, vmax=max(float(before.max()), .01))
        limit = max(float(np.abs(after).max()), .01)
        signed = row[1].imshow(after, cmap="coolwarm", vmin=-limit, vmax=limit)
        figure.colorbar(signed, ax=row[1], shrink=.65)
        difference = row[2].imshow(after - before, cmap="coolwarm")
        figure.colorbar(difference, ax=row[2], shrink=.65)
        for axis, title in zip(row, ("Clean unit vector", "Observed real unit vector", "Signed difference")):
            axis.set(title=title)
            axis.axis("off")
        row[0].set_title(f"Clean unit vector: label {variant.original.labels[index]}\n{variant.original.sample_ids[index]}", fontsize=9)
        row[1].set_title(f"Observed real unit vector\nLabel {variant.original.labels[index]} → {variant.observed.labels[index]}", fontsize=9)
    figure.suptitle(f"{variant.config.kind}: {variant.config.rate * 100:g}% of eligible records; signed entries preserved")
    return figure


def diagnostic_figure(table):
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for axis, attack in zip(axes, ("feature_randomization", "label_flip")):
        for metric in ("mean_input_l2", "mean_pca_l2", "mean_angle_l2"):
            values = table[table.attack == attack].groupby("poison_rate")[metric].agg(["mean", "std"])
            axis.errorbar(values.index * 100, values["mean"], yerr=values["std"].fillna(0), marker="o", capsize=3, label=metric)
        axis.set(title=attack.replace("_", " "), xlabel="Eligible-class poisoning (%)", ylabel="Paired change (L2)")
        axis.legend(fontsize=8)
    return figure


def study_tables(study):
    tables = cached_phase1_tables(study.conditions)
    aggregates = fold_summaries(tables, expected_folds=tuple(study.split_paths))
    return {**tables, "fold_summary": aggregates["metrics"], "fold_effects": aggregates["effects"]}


def comparison_figures(tables):
    summary = tables["fold_summary"]
    figures = {}
    roles = ("poisoned", "poisoned_guided", "poisoned_uniform", "retrained_poisoned")
    for metric in ("retained_accuracy", "forget_recall", "forget_confidence", "conditional_kl_mean"):
        selected = summary[summary.metric == metric]
        figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
        for axis, attack in zip(axes, ("feature_randomization", "label_flip")):
            for role in roles:
                rows = selected[(selected.model == role) & selected.attack.isin(["none", attack])].sort_values("poison_rate")
                axis.errorbar(rows.poison_rate * 100, rows["mean"], yerr=rows.fold_sd.fillna(0), marker="o", capsize=3, label=role)
            axis.set(title=attack.replace("_", " "), xlabel="Eligible-class poisoning (%)", ylabel=metric.replace("_", " "))
            axis.legend(fontsize=8)
        figure.suptitle("Seed6304; mean ± descriptive fold SD")
        figures[metric] = figure
    return figures


def detailed_figures(study, fold=0):
    """Selected-fold trajectories and class errors next to the aggregate results."""
    from sklearn.metrics import ConfusionMatrixDisplay
    figures = {}
    names = ("clean", "label_flip_50pct", "feature_randomization_50pct")
    for name in names:
        condition = study.conditions.get(f"fold_{fold}_{name}")
        if condition is None:
            continue
        runs = {"original": condition["models"]["poisoned"], "guided": condition["unlearned"]["poisoned_guided"],
                "uniform": condition["unlearned"]["poisoned_uniform"], "retrained": condition["models"]["retrained_poisoned"]}
        history_figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
        for role, run in runs.items():
            history = run["history"]
            column = "objective" if role in ("guided", "uniform") else "validation_loss"
            axis = axes[1] if role in ("guided", "uniform") else axes[0]
            axis.plot(history.step, history[column], label=role)
        for axis, title in zip(axes, ("Supervised validation loss", "Unlearning objective")):
            axis.set(title=title, xlabel="Mini-batch update", ylabel="Loss")
            axis.legend()
        history_figure.suptitle(f"Fold{fold}: {name}; checkpoint selection uses no test labels")
        figures[f"{name}_histories"] = history_figure
        figure, axes = plt.subplots(1, 4, figsize=(14, 3.7), layout="constrained")
        for axis, (role, run) in zip(axes, runs.items()):
            matrix = np.asarray(run["metrics"]["models"]["model"]["confusion_matrix"])
            ConfusionMatrixDisplay(matrix, display_labels=[0, 1, 2]).plot(ax=axis, colorbar=False)
            axis.set_title(role)
        figure.suptitle(f"Fold{fold}: {name}; clean held-out test counts")
        figures[f"{name}_confusions"] = figure
    return figures


def export_study(study, tables, figures=None, diagnostics=None, *, destination=None):
    destination = Path(destination) if destination is not None else study.report_path.parent / "analysis"
    destination.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        if len(frame.columns):
            frame.to_csv(destination / f"{name}.csv", index=False)
    if diagnostics is not None:
        diagnostics.to_csv(destination / "input_diagnostics.csv", index=False)
    run_ids = [path.name for path in study.runs]
    for name, figure in (figures or {}).items():
        save_figure(figure, destination / "figures" / name, source_run_ids=run_ids,
                    description="Fresh normalized-input protocol; one seed; descriptive fold variability.")
    artifacts.save_json(destination / "provenance.json", {"protocol": study.index["protocol"],
                        "study_index": study.report_path.relative_to(study.root).as_posix(),
                        "input_policy": INPUT_POLICY, "completed_unique_fits": len(study.runs),
                        "historical_results_replaced": False})
    return destination


def download_bundle(study, related_indexes=()):
    """Include new parents, references, annotations and optional ablation/noise results."""
    root = study.root
    destination = study.report_path.parent / "fresh_study_download.zip"
    directories = [study.report_path.parent, root / "src", *study.runs]
    files = {root / "main_project.ipynb", root / "README.md", root / "requirements.txt"}
    for condition in study.conditions.values():
        directories.append(root / condition["annotated_dataset"])
        files.add(root / condition["split_manifest"])
    for index_file in related_indexes:
        if index_file is None:
            continue
        index_file = study._read_path(index_file)
        record = artifacts.load_json(index_file)
        directories.append(index_file.parent)
        if record.get("performance_directory"):
            directories.append(study._read_path(root / record["performance_directory"]))
        for row in [*record.get("records", []), *record.get("baseline_checks", []),
                    *record.get("supervised", {}).values(), *record.get("unlearned", {}).values()]:
            if row.get("run"):
                directories.append(study._read_path(root / row["run"]))
        for row in record.get("density_records", []):
            directories.append(study._read_path(root / row["cache"]))
    for directory in directories:
        study._read_path(directory)
        files.update(p for p in directory.rglob("*") if p.is_file() and p.suffix not in (".zip", ".tmp", ".pyc")
                     and "__pycache__" not in p.parts)
        # Each completed run's exact source snapshot remains in its portable folder.
        if (directory / "source.zip").is_file():
            files.add(directory / "source.zip")
    with ZipFile(destination, "w", ZIP_DEFLATED) as archive:
        for file in sorted(files):
            study._read_path(file)
            archive.write(file, file.relative_to(root).as_posix())
    return destination
