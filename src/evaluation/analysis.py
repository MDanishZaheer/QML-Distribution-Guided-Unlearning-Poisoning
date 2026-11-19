"""Compact, run-linked notebook tables; no model execution or fitting."""

import numpy as np
import pandas as pd

from src.artifacts import load_json
from src.evaluation.metrics import compare_to_retraining


def accuracy_interval(correct, count):
    """95% Wilson binomial interval; not uncertainty across folds or seeds."""
    if count == 0:
        return np.nan, np.nan
    z = 1.959963984540054
    proportion = correct / count
    denominator = 1 + z*z / count
    center = (proportion + z*z / (2*count)) / denominator
    half = z * np.sqrt(proportion*(1-proportion)/count + z*z/(4*count*count)) / denominator
    return max(0.0, center-half), min(1.0, center+half)


def phase1_tables(conditions):
    """Keep attack/rate/fold identities and do not average duplicated clean runs."""
    scores, changes, targets, comparisons, resources, per_class, checkpoints = [], [], [], [], {}, [], []
    for condition_name, condition in conditions.items():
        config = condition["config"]
        common = {"condition": condition_name, "attack": config["poison"]["kind"],
                  "poison_rate": config["poison"]["rate"], "seed": config["seed"],
                  "fold": config["data"]["fold"],
                  "realized_training_fraction": condition["poisoning"]["actual_fraction_all_training"],
                  "realized_eligible_fraction": condition["poisoning"]["actual_fraction_eligible"]}
        for role, run in {**condition["models"], **condition["unlearned"]}.items():
            if run is None:
                continue
            is_unlearning = role in condition["unlearned"]
            for estimator, values in run["metrics"]["models"].items():
                # Classical retained-reference estimators are saved, but omit them
                # from the main quantum unlearning comparison to keep it readable.
                if estimator != "model" and role not in ("clean", "poisoned"):
                    continue
                model_name = role if estimator == "model" else f"{role}_{estimator}"
                matrix = np.asarray(values["confusion_matrix"])
                low, high = accuracy_interval(int(np.trace(matrix)), int(matrix.sum()))
                scores.append({**common, "model": model_name, "run_id": run["path"].name,
                               **{k: v for k, v in values.items() if not isinstance(v, (list, dict))},
                               "accuracy_ci95_low": low, "accuracy_ci95_high": high})
                for digit in range(3):
                    per_class.append({**common, "model": model_name, "digit": digit,
                                      "n_test": int(matrix[digit].sum()), "correct": int(matrix[digit, digit]),
                                      "recall": values["recall_by_class"][str(digit)]})
            checkpoint = load_json(run["path"] / "checkpoints/model.json")
            checkpoints.append({**common, "model": role, "run_id": run["path"].name, **checkpoint})
            if run["path"].name not in resources:
                information = load_json(run["path"] / "resources.json")
                resources[run["path"].name] = {"run_id": run["path"].name,
                    "first_role": role, "actual_device": information["actual_device"],
                    "training_seconds": information.get("training_seconds"),
                    "compiled_depth": information["compiled_circuit"]["depth"],
                    "two_qubit_gates": information["compiled_circuit"]["two_qubit_gates"],
                    "imported_from": information.get("imported_from"),
                    **information["resources"]}
            if is_unlearning:
                targets.append({**common, "model": role,
                                **{f"target_{i}": p for i, p in enumerate(run["metrics"]["target"])},
                                **run["metrics"]["reference_audit"]})
                changes.append({**common, "model": role, **run["metrics"]["changes"]})
                comparisons.append({**common, "model": role, **run["metrics"]["retraining_comparison"]})
            elif role in ("clean", "poisoned"):
                reference = condition["models"].get(f"retrained_{role}")
                if reference is not None:
                    comparison = compare_to_retraining(run["predictions"]["test_labels"],
                        reference["predictions"]["model__test"], run["predictions"]["model__test"],
                        config["unlearning"]["forget_class"])
                    comparisons.append({**common, "model": role, **comparison})
    effects = []
    for name, condition in conditions.items():
        clean, poisoned = (condition["models"].get(case) for case in ("clean", "poisoned"))
        if clean is None or poisoned is None:
            continue
        for method in condition["config"]["unlearning"]["methods"]:
            clean_after, poisoned_after = (condition["unlearned"].get(f"{case}_{method}") for case in ("clean", "poisoned"))
            if clean_after is None or poisoned_after is None:
                continue
            for metric in ("retained_accuracy", "forget_confidence", "forget_recall"):
                before_clean, before_poisoned, after_clean, after_poisoned = [
                    run["metrics"]["models"]["model"][metric]
                    for run in (clean, poisoned, clean_after, poisoned_after)]
                config = condition["config"]
                effects.append({"condition": name, "method": method, "metric": metric,
                    "attack": config["poison"]["kind"], "poison_rate": config["poison"]["rate"],
                    "seed": config["seed"], "fold": config["data"]["fold"],
                    "poisoning_damage_before": before_poisoned-before_clean,
                    "clean_unlearning_change": after_clean-before_clean,
                    "poisoned_unlearning_change": after_poisoned-before_poisoned,
                    "additional_change_under_poisoning": (after_poisoned-before_poisoned)-(after_clean-before_clean)})
    return {"metrics": pd.DataFrame(scores), "per_class": pd.DataFrame(per_class),
            "changes": pd.DataFrame(changes), "targets": pd.DataFrame(targets),
            "retraining": pd.DataFrame(comparisons), "effects": pd.DataFrame(effects),
            "resources": pd.DataFrame(list(resources.values())), "checkpoints": pd.DataFrame(checkpoints)}


def _fold_summary(frame, groups, metrics, expected_folds):
    """Equal-weight fold means and descriptive sample SD, never seed intervals."""
    if frame.empty:
        return pd.DataFrame()
    if frame.duplicated(groups + ["fold"]).any():
        raise ValueError("Duplicate fold rows: separate ablations/protocols before summarizing.")
    expected = set(expected_folds)
    if not expected or not set(frame["fold"]).issubset(expected):
        raise ValueError("Observed folds must belong to the planned fold set.")
    rows = []
    for key, group in frame.groupby(groups, dropna=False):
        for metric in metrics:
            if metric not in group:
                continue
            values = group.dropna(subset=[metric])
            rows.append({**dict(zip(groups, key)), "metric": metric,
                "mean": values[metric].mean(), "fold_sd": values[metric].std(ddof=1),
                "n_folds": values["fold"].nunique(), "expected_folds": len(expected),
                "complete_folds": set(values["fold"]) == expected})
    return pd.DataFrame(rows)


def fold_summaries(tables, *, expected_folds):
    """Summarize scores, per-class recalls, KL, and paired poisoning effects.

    Keep each seed/attack/rate/model separate. Clean references reused across
    rates are never pooled as additional independent observations. Compute
    differences within each fold before aggregating their mean and SD.
    """
    keys = ["seed", "attack", "poison_rate", "model"]
    summaries = []
    for table, metrics in (
        (tables["metrics"], ["accuracy", "retained_accuracy", "forget_recall", "forget_confidence"]),
        (tables["retraining"], ["full_kl_mean", "conditional_kl_mean"]),
    ):
        summaries.append(_fold_summary(table, keys, metrics, expected_folds))
    if not tables["per_class"].empty:
        for digit, group in tables["per_class"].groupby("digit"):
            metric = f"recall_digit_{digit}"
            summaries.append(_fold_summary(group.rename(columns={"recall": metric}),
                                           keys, [metric], expected_folds))
    metrics = pd.concat([table for table in summaries if not table.empty], ignore_index=True) if any(
        not table.empty for table in summaries) else pd.DataFrame()
    effects = _fold_summary(tables["effects"].rename(columns={"metric": "outcome"}),
        ["seed", "attack", "poison_rate", "method", "outcome"],
        ["poisoning_damage_before", "clean_unlearning_change", "poisoned_unlearning_change",
         "additional_change_under_poisoning"], expected_folds)
    return {"metrics": metrics, "effects": effects}
