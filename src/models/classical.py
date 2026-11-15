"""Fixed-hyperparameter classical baselines on the same four input features."""

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import DEFAULT_SEED


def fit_classical_baselines(features, labels, seed: int = DEFAULT_SEED) -> dict:
    """Any future tuning must occur inside the training partition, not on test data."""
    _, counts = np.unique(labels, return_counts=True)
    if len(counts) < 2 or counts.min() < 2:
        raise ValueError("Calibrated baselines need at least two classes and two training records per class.")
    calibration = StratifiedKFold(n_splits=min(5, int(counts.min())), shuffle=True, random_state=seed)

    def calibrated_svm(kernel):
        estimator = make_pipeline(StandardScaler(), SVC(kernel=kernel, C=1.0, gamma="scale", random_state=seed))
        return CalibratedClassifierCV(estimator, method="sigmoid", cv=calibration, ensemble=False)

    models = {
        "majority": DummyClassifier(strategy="most_frequent", random_state=seed),
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed)),
        "linear_svm": calibrated_svm("linear"),
        "rbf_svm": calibrated_svm("rbf"),
    }
    for model in models.values():
        model.fit(features, labels)
    return models


def aligned_probabilities(model, features, n_classes: int = 3) -> np.ndarray:
    """Preserve the 0/1/2 output order even when training contains only two labels."""
    probabilities = np.zeros((len(features), n_classes))
    classes = np.asarray(model.classes_, dtype=int)
    if np.any(classes < 0) or np.any(classes >= n_classes):
        raise ValueError("Model classes do not match the requested output space.")
    probabilities[:, classes] = model.predict_proba(features)
    return probabilities
