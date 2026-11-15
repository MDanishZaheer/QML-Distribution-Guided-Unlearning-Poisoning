"""Explicit denominators for class forgetting, retention, and retraining agreement."""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

from src.models.vqc import validate_targets


def classification_metrics(labels, probabilities, forget_class: int = 2) -> dict:
    labels = np.asarray(labels)
    probabilities = validate_targets(probabilities, len(labels))
    if not np.isin(labels, [0, 1, 2]).all():
        raise ValueError("Expected clean labels 0, 1, and 2.")
    predicted = probabilities.argmax(axis=1)
    forgotten, retained = labels == forget_class, labels != forget_class

    def average(values, mask):
        return float(np.mean(values[mask])) if np.any(mask) else None

    return {
        "n_samples": len(labels), "n_retained": int(retained.sum()), "n_forget": int(forgotten.sum()),
        "accuracy": float(accuracy_score(labels, predicted)),
        "retained_accuracy": average(predicted == labels, retained),
        "forget_recall": average(predicted == forget_class, forgotten),
        "forget_confidence": average(probabilities[:, forget_class], forgotten),
        "forgotten_mass_on_retained": average(probabilities[:, forget_class], retained),
        "recall_by_class": {str(k): average(predicted == k, labels == k) for k in range(3)},
        "confusion_matrix": confusion_matrix(labels, predicted, labels=[0, 1, 2]).tolist(),
    }


def kl_divergence(reference, candidate, retained_classes=None) -> np.ndarray:
    """Per-sample KL(reference || candidate); optional explicit conditioning."""
    reference = validate_targets(reference, len(reference))
    candidate = validate_targets(candidate, len(candidate))
    if reference.shape != candidate.shape:
        raise ValueError("KL distributions must be aligned by sample and class.")
    if retained_classes is not None:
        reference, candidate = reference[:, retained_classes], candidate[:, retained_classes]
        if (reference.sum(axis=1) <= 0).any() or (candidate.sum(axis=1) <= 0).any():
            raise ValueError("Cannot condition a distribution with zero retained mass.")
        reference = reference / reference.sum(axis=1, keepdims=True)
        candidate = candidate / candidate.sum(axis=1, keepdims=True)
    epsilon = 1e-12
    terms = reference * (np.log(np.clip(reference, epsilon, 1)) - np.log(np.clip(candidate, epsilon, 1)))
    return np.maximum(terms.sum(axis=1), 0.0)


def compare_to_retraining(labels, reference_probabilities, candidate_probabilities,
                          forget_class: int = 2) -> dict:
    labels = np.asarray(labels)
    mask = labels != forget_class
    if not mask.any():
        raise ValueError("Retraining comparison requires retained test examples.")
    reference, candidate = np.asarray(reference_probabilities)[mask], np.asarray(candidate_probabilities)[mask]
    conditional = kl_divergence(reference, candidate, [k for k in range(3) if k != forget_class])
    full = kl_divergence(reference, candidate)
    return {"kl_direction": "reference || candidate", "evaluation_subset": "clean retained test",
            "full_kl_mean": float(full.mean()), "conditional_kl_mean": float(conditional.mean()),
            "conditional_kl_median": float(np.median(conditional)), "conditional_kl_max": float(conditional.max()),
            "reference_forgotten_mass": float(reference[:, forget_class].mean()),
            "candidate_forgotten_mass": float(candidate[:, forget_class].mean())}


def forgetting_changes(before: dict, after: dict) -> dict:
    """Positive confidence reduction means stronger suppression; retention delta is signed."""
    result = {}
    for key in ("retained_accuracy", "forget_confidence", "forget_recall"):
        first, last = before[key], after[key]
        result[f"{key}_delta"] = None if first is None or last is None else last - first
    delta = result["forget_confidence_delta"]
    result["forget_confidence_reduction"] = None if delta is None else -delta
    return result


def summarize_runs(frame: pd.DataFrame, metric: str, group_columns=None,
                   seed_column: str = "seed") -> pd.DataFrame:
    """Average folds within a seed first; report SD across seed means (not test samples).

    Fold means receive equal weight. Use equal-sized folds or explicitly weight
    unequal folds before calling this helper. A single seed has undefined SD.
    """
    groups = list(group_columns) if group_columns is not None else [
        name for name in ("protocol_id", "condition", "poison_rate", "model") if name in frame.columns]
    if "run_id" in frame and frame.duplicated(["run_id", "model"]).any():
        raise ValueError("Duplicate run/model rows would give the same run extra statistical weight.")
    per_seed = frame.groupby(groups + [seed_column], dropna=False)[metric].mean().reset_index()
    summary = per_seed.groupby(groups, dropna=False)[metric].agg(["mean", "std", "count"]).reset_index()
    return summary.rename(columns={"count": "n_seeds"})
