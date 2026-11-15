"""Training-only PCA and angle scaling. Validation/test data only call transform."""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import check_is_fitted

from src.config import DEFAULT_SEED


def normalize_images(images: np.ndarray) -> np.ndarray:
    values = np.asarray(images, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (28, 28) or not len(values):
        raise ValueError("Expected nonempty raw images with shape (n, 28, 28).")
    if not np.isfinite(values).all() or values.min() < 0 or values.max() > 255:
        raise ValueError("Expected raw pixel values in [0, 255].")
    return values.reshape(len(values), -1) / 255.0


class ImagePreprocessor:
    """Four PCA features, followed by training-fitted min/max scaling to [0, pi].

    Out-of-training-range values are clipped. This policy is explicit and is
    saved with the fitted transformer; it never refits on validation/test data.
    """

    def __init__(self, seed: int = DEFAULT_SEED):
        self.seed = seed
        self.pca = PCA(n_components=4, svd_solver="randomized", random_state=seed)
        self.scaler = MinMaxScaler(feature_range=(0.0, np.pi), clip=True)

    def fit(self, images: np.ndarray, sample_ids=None):
        values = normalize_images(images)
        if len(values) < 4:
            raise ValueError("Four-component PCA requires at least four training examples.")
        components = self.pca.fit_transform(values)
        self.scaler.fit(components)
        self.fit_sample_ids_ = None if sample_ids is None else np.asarray(sample_ids, dtype=str).copy()
        if self.fit_sample_ids_ is not None and len(self.fit_sample_ids_) != len(values):
            raise ValueError("sample_ids must match the fitted images.")
        self.n_fit_samples_ = len(values)
        return self

    def transform(self, images: np.ndarray) -> np.ndarray:
        check_is_fitted(self.pca)
        return self.scaler.transform(self.pca.transform(normalize_images(images)))

    def fit_transform(self, images: np.ndarray, sample_ids=None) -> np.ndarray:
        return self.fit(images, sample_ids).transform(images)

    def diagnostics(self) -> dict:
        check_is_fitted(self.pca)
        return {"n_fit_samples": self.n_fit_samples_,
                "explained_variance_ratio": self.pca.explained_variance_ratio_.tolist(),
                "total_explained_variance": float(self.pca.explained_variance_ratio_.sum()),
                "angle_range": [0.0, float(np.pi)], "clip": True}


def prepare_fold_features(splits, seed=DEFAULT_SEED):
    """Fit one transformer on this fold's training records, then only transform."""
    preprocessing = ImagePreprocessor(seed=seed)
    preprocessing.fit(splits.train.images, splits.train.sample_ids)
    return {
        "preprocessing": preprocessing,
        "train": preprocessing.transform(splits.train.images),
        "validation": preprocessing.transform(splits.validation.images),
        "test": preprocessing.transform(splits.test.images),
    }
