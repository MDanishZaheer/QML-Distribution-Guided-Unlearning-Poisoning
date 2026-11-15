"""MNIST acquisition and reproducible, disjoint sample partitions."""

from dataclasses import dataclass
import hashlib
from pathlib import Path
import urllib.request

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.config import DEFAULT_SEED, DataConfig, PROJECT_ROOT

MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
# Published in keras/src/datasets/mnist.py; TensorFlow/Keras is NOT a dependency.
MNIST_SHA256 = "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1"


@dataclass(frozen=True)
class ImageDataset:
    """Images use the original 0..255 scale; IDs never change after poisoning."""

    images: np.ndarray
    labels: np.ndarray
    sample_ids: np.ndarray

    def __post_init__(self):
        images = np.asarray(self.images)
        labels = np.asarray(self.labels)
        ids = np.asarray(self.sample_ids, dtype=str)
        if images.ndim != 3 or images.shape[1:] != (28, 28):
            raise ValueError("images must have shape (n_samples, 28, 28).")
        if labels.shape != (len(images),) or ids.shape != labels.shape or len(images) == 0:
            raise ValueError("Images, labels, and IDs must have matching nonempty lengths.")
        if not np.isfinite(images).all() or images.min() < 0 or images.max() > 255:
            raise ValueError("Pixels must be finite and in the original 0..255 range.")
        if not np.issubdtype(labels.dtype, np.integer) or not np.isin(labels, range(10)).all():
            raise ValueError("Digit labels must be integers 0..9.")
        if len(np.unique(ids)) != len(ids):
            raise ValueError("Sample IDs must be unique.")
        object.__setattr__(self, "images", images.copy())
        object.__setattr__(self, "labels", labels.astype(np.int64, copy=True))
        object.__setattr__(self, "sample_ids", ids.copy())

    def subset(self, indices) -> "ImageDataset":
        return ImageDataset(self.images[indices], self.labels[indices], self.sample_ids[indices])

    def __len__(self):
        return len(self.labels)

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        for array in (self.images, self.labels, self.sample_ids):
            digest.update(str((array.shape, array.dtype.str)).encode())
            digest.update(np.ascontiguousarray(array).tobytes())
        return digest.hexdigest()


@dataclass(frozen=True)
class DatasetSplits:
    train: ImageDataset
    validation: ImageDataset
    test: ImageDataset

    def __post_init__(self):
        sets = [set(part.sample_ids) for part in (self.train, self.validation, self.test)]
        if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
            raise ValueError("Training, validation, and test sample IDs must be disjoint.")
        if any(not np.isin(part.labels, [0, 1, 2]).all()
               for part in (self.train, self.validation, self.test)):
            raise ValueError("The Phase 1 splits must contain only digits 0, 1, and 2.")

    def fingerprint(self) -> str:
        joined = "|".join(part.fingerprint() for part in (self.train, self.validation, self.test))
        return hashlib.sha256(joined.encode()).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_mnist(raw_dir: Path | str = PROJECT_ROOT / "data/raw") -> Path:
    """Download only when explicitly called; verify the official archive checksum."""
    directory = Path(raw_dir)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "mnist.npz"
    if destination.exists():
        if _file_sha256(destination) != MNIST_SHA256:
            raise ValueError(f"MNIST checksum mismatch: {destination}")
        return destination
    temporary = destination.with_suffix(".npz.part")
    try:
        with urllib.request.urlopen(MNIST_URL, timeout=60) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        if _file_sha256(temporary) != MNIST_SHA256:
            raise ValueError("Downloaded MNIST failed its checksum verification.")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def load_mnist(raw_dir: Path | str = PROJECT_ROOT / "data/raw", *, download: bool = False):
    """Return official training/test pools, with source-prefixed immutable IDs."""
    archive = Path(raw_dir) / "mnist.npz"
    if download:
        archive = download_mnist(raw_dir)
    if not archive.exists():
        raise FileNotFoundError("MNIST is not downloaded. Call load_mnist(download=True) explicitly.")
    if _file_sha256(archive) != MNIST_SHA256:
        raise ValueError("MNIST archive checksum mismatch.")
    with np.load(archive, allow_pickle=False) as arrays:
        pools = []
        for split in ("train", "test"):
            images, labels = arrays[f"x_{split}"], arrays[f"y_{split}"]
            indices = np.flatnonzero(np.isin(labels, [0, 1, 2]))
            ids = np.array([f"mnist_{split}_{i:05d}" for i in indices])
            pools.append(ImageDataset(images[indices], labels[indices], ids))
    return tuple(pools)


def balanced_subset(dataset: ImageDataset, per_class: int | None, seed: int) -> ImageDataset:
    if per_class is None:
        return dataset
    random = np.random.default_rng(seed)
    selected = []
    for label in (0, 1, 2):
        eligible = np.flatnonzero(dataset.labels == label)
        if len(eligible) < per_class:
            raise ValueError(f"Class {label} has fewer than {per_class} examples.")
        selected.extend(random.choice(eligible, per_class, replace=False))
    return dataset.subset(np.sort(selected))


def make_cv_splits(dataset: ImageDataset, n_folds: int = 5,
                   validation_fraction: float = 0.2, seed: int = DEFAULT_SEED) -> list[DatasetSplits]:
    """Outer test folds; each outer training partition gets its own inner holdout.

    This is a fixed validation holdout, not a nested hyperparameter search.
    """
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    for outer_train, test in splitter.split(dataset.images, dataset.labels):
        train, validation = train_test_split(
            outer_train, test_size=validation_fraction, random_state=seed,
            stratify=dataset.labels[outer_train],
        )
        folds.append(DatasetSplits(dataset.subset(train), dataset.subset(validation), dataset.subset(test)))
    return folds


def prepare_splits(config: DataConfig, root: Path | str = PROJECT_ROOT,
                   *, download: bool = False) -> DatasetSplits:
    train_pool, test_pool = load_mnist(Path(root) / "data/raw", download=download)
    pool = balanced_subset(train_pool, config.train_per_class, config.seed)
    if config.split_strategy == "stratified_kfold":
        # The official test pool remains unused in this cross-validation protocol.
        return make_cv_splits(pool, config.n_folds, config.validation_fraction, config.seed)[config.fold]
    train, validation = train_test_split(
        np.arange(len(pool)), test_size=config.validation_fraction,
        stratify=pool.labels, random_state=config.seed,
    )
    test = balanced_subset(test_pool, config.test_per_class, config.seed)
    return DatasetSplits(pool.subset(train), pool.subset(validation), test)
