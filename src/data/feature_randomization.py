"""Paper-inspired real-vector replacement before our four-component PCA.

Chen/Zhang datar.py:data_poison(ordered=False) draws real N(0, .1**2)
vectors for MNIST and L2-normalizes each replacement. We retain our class-2
targeting and eligible-class rates, rather than their balanced binary sampling.
All inputs receive per-record L2 normalization before PCA in this NEW protocol.
Signed vectors are stored as feature surrogates, never clipped into pixel values.
"""

from dataclasses import dataclass, replace
import hashlib

import numpy as np
from sklearn.utils.validation import check_is_fitted

from src.config import DEFAULT_SEED, ExperimentConfig, PoisonConfig
from src.data.dataset import ImageDataset
from src.data.poisoning import PoisonedDataset, apply_poisoning
from src.data.preprocessing import ImagePreprocessor


INPUT_POLICY = {
    "version": "real_gaussian_unit_input_v1",
    "replacement": "full 784-vector, real N(0, 0.1**2), then unit L2 norm",
    "eligible_class": 2, "labels_preserved": True,
    "normalization": "all clean/corrupt inputs divided by 255 then per-record L2 normalized before PCA",
    "zero_input": "leave a zero vector unchanged",
    "pca": "four components; fit on the model's training records only; original refit policy",
    "angle_scaling": "training-fitted MinMaxScaler [0, pi], clip=True",
    "adaptation": "Malik six-qubit angle circuit and class-level forgetting; not amplitude encoding or binary sampling",
    "source": "https://github.com/yutuer21/quantum-machine-unlearning/blob/main/src/datar.py",
}


@dataclass(frozen=True)
class FeaturePoisonConfig(PoisonConfig):
    kind: str = "feature_randomization"
    gaussian_std: float = 0.1

    def __post_init__(self):
        if self.kind != "feature_randomization" or self.target_class != 2:
            raise ValueError("Feature randomization targets digit 2 in this protocol.")
        # Validate the common settings using the original config, without changing it.
        PoisonConfig(kind="pixel_patch", rate=self.rate, denominator=self.denominator,
                     seed=self.seed, source_class=self.source_class, target_class=self.target_class,
                     patch_size=self.patch_size, patch_value=self.patch_value)
        if self.gaussian_std != 0.1:
            raise ValueError("The registered normalized Gaussian generator uses std=0.1.")


def experiment_config(values):
    """Read both original configs and the new explicit attack without editing archives."""
    if values.get("poison", {}).get("kind") != "feature_randomization":
        return ExperimentConfig.from_dict(values)
    values = dict(values)
    poison = FeaturePoisonConfig(**values.pop("poison"))
    return replace(ExperimentConfig.from_dict({**values, "poison": {"kind": "none"}}), poison=poison)


@dataclass(frozen=True)
class SignedFeatureDataset(ImageDataset):
    """28x28 layout for a real feature vector; negative entries are valid here."""

    def __post_init__(self):
        images = np.asarray(self.images, dtype=np.float64)
        # Reuse original label/ID/shape checks on a harmless nonnegative placeholder.
        checked = ImageDataset(np.zeros_like(images), self.labels, self.sample_ids)
        if not np.isfinite(images).all() or np.max(np.abs(images)) > 255:
            raise ValueError("Signed feature surrogates must be finite and within [-255,255].")
        object.__setattr__(self, "images", images.copy())
        object.__setattr__(self, "labels", checked.labels)
        object.__setattr__(self, "sample_ids", checked.sample_ids)

    def subset(self, indices):
        return SignedFeatureDataset(self.images[indices], self.labels[indices], self.sample_ids[indices])


def unit_inputs(images):
    values = np.asarray(images, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (28, 28) or not len(values) or not np.isfinite(values).all():
        raise ValueError("Expected nonempty finite real input arrays (n,28,28).")
    vectors = values.reshape(len(values), -1) / 255.0
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0)


class UnitInputPreprocessor(ImagePreprocessor):
    """Same PCA/angle mapping, with explicit signed, per-record unit normalization."""

    def fit(self, images, sample_ids=None):
        values = unit_inputs(images)
        if len(values) < 4:
            raise ValueError("Four-component PCA requires at least four training examples.")
        self.scaler.fit(self.pca.fit_transform(values))
        self.fit_sample_ids_ = None if sample_ids is None else np.asarray(sample_ids, dtype=str).copy()
        if self.fit_sample_ids_ is not None and len(self.fit_sample_ids_) != len(values):
            raise ValueError("sample_ids must match the fitted inputs.")
        self.n_fit_samples_ = len(values)
        return self

    def transform(self, images):
        check_is_fitted(self.pca)
        return self.scaler.transform(self.pca.transform(unit_inputs(images)))

    def diagnostics(self):
        return {**super().diagnostics(), "input_policy": INPUT_POLICY}


def feature_variant(training_data, config):
    """Nested selected IDs and identical replacement vectors at every poison rate."""
    if config.kind != "feature_randomization":
        return apply_poisoning(training_data, config)
    eligible = np.flatnonzero(training_data.labels == config.target_class)
    denominator = len(eligible) if config.denominator == "eligible" else len(training_data)
    count = int(np.floor(config.rate * denominator))
    if count > len(eligible):
        raise ValueError("Requested rate exceeds eligible training records.")
    selected = np.random.default_rng(config.seed).permutation(eligible)[:count]
    images = training_data.images.astype(np.float64, copy=True)
    for index in selected:
        # Record-derived streams avoid changing a shared record's corruption with
        # the rate, fold membership, ordering or number of simultaneous workers.
        token = hashlib.sha256(str(training_data.sample_ids[index]).encode()).digest()
        stream = np.random.default_rng(np.random.SeedSequence(
            [config.seed, 6304, int.from_bytes(token[:4], "little"), int.from_bytes(token[4:8], "little")]))
        vector = stream.normal(0.0, config.gaussian_std, 784)
        images[index] = (255.0 * vector / np.linalg.norm(vector)).reshape(28, 28)
    marked = np.zeros(len(training_data), dtype=bool)
    marked[selected] = True
    observed = SignedFeatureDataset(images, training_data.labels, training_data.sample_ids)
    return PoisonedDataset(training_data, observed, marked, config, len(eligible))
