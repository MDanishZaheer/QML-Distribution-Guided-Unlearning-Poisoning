"""Small, reusable research figures. Functions return figures instead of calling show()."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from src.artifacts import save_json


def plot_examples(original, observed=None, indices=None, max_examples: int = 6):
    """Show the same record IDs before/after poisoning with original/observed labels."""
    indices = np.arange(min(max_examples, len(original))) if indices is None else np.asarray(indices)[:max_examples]
    if not len(indices):
        raise ValueError("Select at least one image to display.")
    if observed is not None and not np.array_equal(original.sample_ids, observed.sample_ids):
        raise ValueError("Before/after examples must have the same sample IDs and ordering.")
    rows = 1 if observed is None else 2
    figure, axes = plt.subplots(rows, len(indices), figsize=(2.2 * len(indices), 2.7 * rows), squeeze=False)
    for column, index in enumerate(indices):
        for row, dataset in enumerate([original] if observed is None else [original, observed]):
            axis = axes[row, column]
            axis.imshow(dataset.images[index], cmap="gray", vmin=0, vmax=255)
            description = "Original" if row == 0 else "Observed"
            axis.set_title(f"{description}: label {dataset.labels[index]}\n{dataset.sample_ids[index]}", fontsize=8)
            axis.axis("off")
    figure.tight_layout()
    return figure


def plot_poisoned_examples(variant, max_examples: int = 3):
    """Inspect selected records with paired images and actual pixel differences.

    Use the prepared variant directly: this does not create another attack or
    change its data. A patch outline is a display annotation, not an image edit.
    """
    if max_examples < 1:
        raise ValueError("max_examples must be positive.")
    if not np.array_equal(variant.original.sample_ids, variant.observed.sample_ids):
        raise ValueError("Before/after examples must have matching sample IDs and ordering.")
    indices = np.flatnonzero(variant.is_poisoned)[:max_examples]
    if not len(indices):
        raise ValueError("This variant has no selected poisoned records to display.")

    figure, axes = plt.subplots(len(indices), 3, figsize=(9, 2.8 * len(indices) + 1),
                                squeeze=False, layout="constrained")
    for row, index in enumerate(indices):
        original = variant.original.images[index]
        observed = variant.observed.images[index]
        original_label = int(variant.original.labels[index])
        observed_label = int(variant.observed.labels[index])
        # Convert before subtracting to avoid unsigned pixel-value wraparound.
        difference = np.abs(observed.astype(float) - original.astype(float))
        changed_pixels = np.count_nonzero(difference)
        for column, pixels in enumerate((original, observed)):
            axes[row, column].imshow(pixels, cmap="gray", vmin=0, vmax=255,
                                     interpolation="nearest")
        axes[row, 0].set_title(
            f"Original: label {original_label}\n{variant.original.sample_ids[index]}", fontsize=9)
        axes[row, 1].set_title(
            f"Poisoned: observed label {observed_label}\n"
            f"Label {original_label} -> {observed_label}", fontsize=9,
            color="crimson" if original_label != observed_label else "black")
        if variant.config.kind == "pixel_patch":
            size = variant.config.patch_size
            height, width = observed.shape
            axes[row, 1].add_patch(Rectangle(
                (width - size - 0.5, height - size - 0.5), size, size,
                fill=False, edgecolor="darkorange", linewidth=1.8))
        difference_image = axes[row, 2].imshow(
            difference, cmap="magma", vmin=0, vmax=255, interpolation="nearest")
        axes[row, 2].set_title(
            f"Absolute pixel change\n{changed_pixels} of {difference.size} pixels changed", fontsize=9)
        for axis in axes[row]:
            axis.axis("off")

    figure.colorbar(difference_image, ax=list(axes[:, 2]), shrink=0.7,
                   label="Absolute pixel difference (0-255)")
    denominator = "eligible class" if variant.config.denominator == "eligible" else "training set"
    figure.suptitle(
        f"{variant.config.kind.replace('_', ' ').title()}: requested "
        f"{100 * variant.config.rate:g}% of {denominator}\n"
        f"{int(variant.is_poisoned.sum())} selected / {len(variant.original)} training records; "
        f"seed {variant.config.seed}", fontsize=11)
    note = ("Orange outline marks the configured patch area; it is a display annotation."
            if variant.config.kind == "pixel_patch" else
            "Compare the labels: label flipping leaves the image pixels unchanged.")
    figure.supxlabel(note, fontsize=9)
    return figure


def plot_class_balance(partitions: dict):
    """partitions maps a descriptive split name to its labels."""
    figure, axis = plt.subplots(figsize=(7, 4))
    width = 0.8 / max(1, len(partitions))
    for index, (name, labels) in enumerate(partitions.items()):
        counts = [np.count_nonzero(np.asarray(labels) == label) for label in range(3)]
        positions = np.arange(3) - 0.4 + width / 2 + index * width
        axis.bar(positions, counts, width, label=name)
    axis.set(xticks=[0, 1, 2], xlabel="Digit", ylabel="Number of records", title="Class counts by partition")
    axis.legend()
    figure.tight_layout()
    return figure


def plot_pca_variance(preprocessor):
    information = preprocessor.diagnostics()
    ratios = np.asarray(information["explained_variance_ratio"])
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.bar(np.arange(1, 5), ratios, label="Individual component")
    axis.plot(np.arange(1, 5), ratios.cumsum(), "o-", label="Cumulative")
    axis.set(xlabel="PCA component", ylabel="Explained variance fraction", xticks=[1, 2, 3, 4], ylim=(0, 1),
             title=f"PCA fitted on {information['n_fit_samples']} training records")
    axis.legend()
    figure.tight_layout()
    return figure


def plot_training_history(history, columns=None, title="Optimization history"):
    """Pass one model's history; multi-model history must be filtered first."""
    if "model" in history and history["model"].nunique() > 1:
        raise ValueError("Filter the history to one model before plotting.")
    columns = columns or [name for name in ("train_loss", "validation_loss", "objective")
                          if name in history and history[name].notna().any()]
    if not columns:
        raise ValueError("No requested loss columns are available.")
    figure, axis = plt.subplots(figsize=(7, 4))
    for column in columns:
        axis.plot(history["step"], history[column], label=column.replace("_", " "))
    axis.set(xlabel="Mini-batch update", ylabel="Loss", title=title)
    axis.legend()
    figure.tight_layout()
    return figure


def plot_confusion_comparison(before_matrix, after_matrix, title="Before and after unlearning"):
    matrices = [np.asarray(before_matrix), np.asarray(after_matrix)]
    if any(matrix.shape != (3, 3) for matrix in matrices):
        raise ValueError("Expected two 3-by-3 confusion matrices.")
    maximum = max(1, max(matrix.max() for matrix in matrices))
    figure, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
    for axis, matrix, name in zip(axes, matrices, ("Before", "After")):
        image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=maximum)
        for row in range(3):
            for column in range(3):
                axis.text(column, row, str(matrix[row, column]), ha="center", va="center",
                          color="white" if matrix[row, column] > maximum / 2 else "black")
        axis.set(xticks=[0, 1, 2], yticks=[0, 1, 2], xlabel="Predicted digit", ylabel="True digit",
                 title=f"{name} (n={matrix.sum()})")
    figure.colorbar(image, ax=list(axes), label="Records")
    figure.suptitle(title)
    return figure


def plot_forget_confidence(labels, before_probabilities, after_probabilities, forget_class: int = 2):
    mask = np.asarray(labels) == forget_class
    if not mask.any():
        raise ValueError("No forgotten-class examples in this evaluation set.")
    figure, axis = plt.subplots(figsize=(7, 4))
    for probabilities, name in ((before_probabilities, "Before"), (after_probabilities, "After")):
        values = np.asarray(probabilities)[mask, forget_class]
        axis.hist(values, bins=np.linspace(0, 1, 21), alpha=0.55, label=name)
    axis.set(xlabel=f"Predicted probability of digit {forget_class}", ylabel="Records",
             title=f"Forgotten-class confidence on clean digit {forget_class} (n={mask.sum()})", xlim=(0, 1))
    axis.legend()
    figure.tight_layout()
    return figure


def plot_metric_summary(summary, category="model", ylabel="Retained accuracy", title="Model comparison"):
    """Input is a summarize_runs() table; error bars are SD across seed means."""
    figure, axis = plt.subplots(figsize=(max(7, 0.9 * len(summary)), 4))
    positions = np.arange(len(summary))
    axis.bar(positions, summary["mean"])
    valid = summary["std"].notna().to_numpy()
    if valid.any():
        axis.errorbar(positions[valid], summary["mean"].to_numpy()[valid],
                      yerr=summary["std"].to_numpy()[valid], fmt="none", color="black", capsize=4)
    axis.set(xticks=positions, xticklabels=summary[category].astype(str), ylabel=ylabel,
             title=title + " (SD bars when at least two seeds)")
    axis.tick_params(axis="x", labelrotation=35)
    figure.tight_layout()
    return figure


def plot_poisoning_curve(summary, ylabel="Retained accuracy", group="model"):
    """Summary must retain poison_rate, group, mean, std, and n_seeds columns."""
    figure, axis = plt.subplots(figsize=(7, 4))
    for name, rows in summary.groupby(group):
        rows = rows.sort_values("poison_rate")
        if rows["poison_rate"].duplicated().any():
            raise ValueError("Filter to one attack/ablation condition before plotting a curve.")
        line, = axis.plot(100 * rows["poison_rate"], rows["mean"], marker="o", label=str(name))
        valid = rows["std"].notna()
        if valid.any():
            axis.errorbar(100 * rows.loc[valid, "poison_rate"], rows.loc[valid, "mean"],
                          yerr=rows.loc[valid, "std"], fmt="none", color=line.get_color(), capsize=4)
    axis.set(xlabel="Requested poisoning rate (%)", ylabel=ylabel,
             title="Poisoning response (SD bars when at least two seeds)")
    axis.legend()
    figure.tight_layout()
    return figure


def draw_circuit(circuit, fold: int = 30):
    return circuit.draw(output="mpl", fold=fold)


def save_figure(figure, path: Path | str, *, source_run_ids, description: str,
                formats=("png", "pdf"), dpi: int = 300):
    """Export a figure plus its provenance. path is a filename stem, without extension."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exported = []
    for extension in formats:
        destination = path.with_suffix(f".{extension}")
        figure.savefig(destination, dpi=dpi, bbox_inches="tight")
        exported.append(destination)
    save_json(path.with_suffix(".json"), {"source_run_ids": list(source_run_ids),
                                         "description": description, "files": [p.name for p in exported]})
    return exported
