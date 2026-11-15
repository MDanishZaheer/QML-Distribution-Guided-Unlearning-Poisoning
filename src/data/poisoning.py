"""Controlled training-only poisoning with original labels and stable record IDs."""

from dataclasses import asdict, dataclass
import warnings

import numpy as np
import pandas as pd

from src.config import PoisonConfig
from src.data.dataset import ImageDataset


@dataclass(frozen=True)
class PoisonedDataset:
    original: ImageDataset
    observed: ImageDataset
    is_poisoned: np.ndarray
    config: PoisonConfig
    eligible_count: int

    def manifest(self) -> pd.DataFrame:
        return pd.DataFrame({
            "sample_id": self.original.sample_ids,
            "original_label": self.original.labels,
            "observed_label": self.observed.labels,
            "is_poisoned": self.is_poisoned,
            "poison_type": np.where(self.is_poisoned, self.config.kind, "none"),
            "poison_seed": self.config.seed,
        })

    def metadata(self) -> dict:
        count = int(self.is_poisoned.sum())
        return {"config": asdict(self.config), "n_training": len(self.original),
                "eligible_count": self.eligible_count, "poisoned_count": count,
                "actual_fraction_all_training": count / len(self.original),
                "actual_fraction_eligible": count / self.eligible_count if self.eligible_count else 0.0,
                "rounding": "floor", "original_fingerprint": self.original.fingerprint(),
                "observed_fingerprint": self.observed.fingerprint()}


def apply_poisoning(training_data: ImageDataset, config: PoisonConfig) -> PoisonedDataset:
    """Modify copies of a TRAINING partition; validation/test stay outside this API.

    Pixel poisoning uses a bottom-right square patch. This is our configurable
    experimental attack, not a pixel-level specification claimed from either paper.
    Rates use eligible-class records by default; exact realized counts are saved.
    """
    images = training_data.images.astype(np.float32, copy=True)
    labels = training_data.labels.copy()
    marked = np.zeros(len(training_data), dtype=bool)
    eligible_class = config.source_class if config.kind == "label_flip" else config.target_class
    eligible = np.flatnonzero(training_data.labels == eligible_class)
    denominator = len(eligible) if config.denominator == "eligible" else len(training_data)
    count = int(np.floor(config.rate * denominator))
    if count > len(eligible):
        raise ValueError("Requested poisoning exceeds the number of eligible training records.")
    if config.rate > 0 and count == 0:
        warnings.warn("This rate rounds down to zero poisoned samples; increase the dataset or rate.", stacklevel=2)
    if count:
        selected = np.random.default_rng(config.seed).permutation(eligible)[:count]
        marked[selected] = True
        if config.kind == "label_flip":
            labels[selected] = config.target_class
        elif config.kind == "pixel_patch":
            size = config.patch_size
            images[selected, -size:, -size:] = config.patch_value
    observed = ImageDataset(images, labels, training_data.sample_ids)
    return PoisonedDataset(training_data, observed, marked, config, len(eligible))


def forget_mask(data: PoisonedDataset, forget_class: int = 2,
                definition: str = "observed") -> np.ndarray:
    """Observed-label forgetting includes digit-0 records relabeled as digit 2."""
    if definition not in ("observed", "original"):
        raise ValueError("definition must be observed or original.")
    labels = data.observed.labels if definition == "observed" else data.original.labels
    return labels == forget_class
