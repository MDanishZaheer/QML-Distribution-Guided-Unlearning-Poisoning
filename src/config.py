"""Explicit, serializable settings. Defaults are a starting protocol, not tuned results."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEED = 6304


@dataclass(frozen=True)
class DataConfig:
    train_per_class: int | None = 100
    test_per_class: int | None = 50
    validation_fraction: float = 0.2
    split_strategy: Literal["official", "stratified_kfold"] = "official"
    n_folds: int = 5
    fold: int = 0
    seed: int = DEFAULT_SEED
    preprocessing: Literal["refit", "shared_clean_train"] = "refit"

    def __post_init__(self):
        for size in (self.train_per_class, self.test_per_class):
            if size is not None and size < 3:
                raise ValueError("Per-class sample limits must be at least 3, or None.")
        if not 0 < self.validation_fraction < 0.5:
            raise ValueError("validation_fraction must be between 0 and 0.5.")
        if self.split_strategy not in ("official", "stratified_kfold"):
            raise ValueError("Unknown split strategy.")
        if self.n_folds < 2 or not 0 <= self.fold < self.n_folds:
            raise ValueError("Choose n_folds >= 2 and a valid zero-based fold.")
        if self.split_strategy == "official" and self.fold != 0:
            raise ValueError("The official holdout protocol uses fold=0; use stratified_kfold for other folds.")
        if self.preprocessing not in ("refit", "shared_clean_train"):
            raise ValueError("Unknown preprocessing policy.")


@dataclass(frozen=True)
class PoisonConfig:
    kind: Literal["none", "pixel_patch", "label_flip"] = "none"
    rate: float = 0.0
    denominator: Literal["eligible", "all_training"] = "eligible"
    seed: int = DEFAULT_SEED
    source_class: int = 0
    target_class: int = 2
    patch_size: int = 3
    patch_value: float = 255.0

    def __post_init__(self):
        if self.kind not in ("none", "pixel_patch", "label_flip"):
            raise ValueError("Unknown poisoning kind.")
        if not 0 <= self.rate <= 1 or (self.kind == "none" and self.rate != 0):
            raise ValueError("rate must be in [0, 1]; kind='none' requires rate=0.")
        if self.denominator not in ("eligible", "all_training"):
            raise ValueError("Unknown poisoning denominator.")
        if self.source_class not in (0, 1, 2) or self.target_class not in (0, 1, 2):
            raise ValueError("Phase 1 uses classes 0, 1, and 2.")
        if self.kind == "label_flip" and self.source_class == self.target_class:
            raise ValueError("Label flipping requires different source and target classes.")
        if not 1 <= self.patch_size <= 28 or not 0 <= self.patch_value <= 255:
            raise ValueError("Invalid patch size or pixel value.")


@dataclass(frozen=True)
class CircuitConfig:
    reuploads: int = 2
    ansatz_reps: int = 3
    entanglement: bool = True
    logit_scale: float = 1.0

    def __post_init__(self):
        if self.reuploads < 1 or self.ansatz_reps < 1 or self.logit_scale <= 0:
            raise ValueError("Circuit repetitions and logit_scale must be positive.")

    @property
    def n_parameters(self) -> int:
        return self.reuploads * self.ansatz_reps * 6 * 2


@dataclass(frozen=True)
class BackendConfig:
    shots: int | None = 2048  # None explicitly requests exact expectations.
    device: Literal["auto", "CPU", "GPU"] = "auto"
    method: Literal["automatic", "statevector", "density_matrix"] = "automatic"
    allow_cpu_fallback: bool = True
    circuit_batch_size: int = 128
    max_memory_mb: int = 4096
    max_parallel_threads: int = 4
    one_qubit_error: float = 0.0
    two_qubit_error: float = 0.0
    readout_error: float = 0.0
    use_ibm_hardware: bool = False
    ibm_backend_name: str | None = None
    optimization_level: int = 0

    def __post_init__(self):
        if self.shots is not None and self.shots < 1:
            raise ValueError("shots must be positive, or None for exact diagnostics.")
        if self.device not in ("auto", "CPU", "GPU"):
            raise ValueError("device must be auto, CPU, or GPU.")
        if self.method not in ("automatic", "statevector", "density_matrix"):
            raise ValueError("Unsupported simulation method.")
        if min(self.circuit_batch_size, self.max_memory_mb, self.max_parallel_threads) < 1:
            raise ValueError("Batch, memory, and thread limits must be positive.")
        if any(not 0 <= p <= 1 for p in self.noise_rates):
            raise ValueError("Noise probabilities must be in [0, 1].")
        if self.shots is None and self.readout_error:
            raise ValueError("Readout noise requires finite shots in this implementation.")
        if self.shots is None and self.method == "statevector" and any(self.noise_rates):
            raise ValueError("Exact noisy evaluation requires density_matrix or automatic.")
        if self.optimization_level not in (0, 1, 2, 3):
            raise ValueError("optimization_level must be 0, 1, 2, or 3.")
        if self.use_ibm_hardware:
            if self.shots is None or not self.ibm_backend_name:
                raise ValueError("IBM execution requires shots and an explicit backend name.")
            if any(self.noise_rates):
                raise ValueError("Synthetic Aer noise cannot be applied to IBM hardware.")

    @property
    def noise_rates(self) -> tuple[float, float, float]:
        return self.one_qubit_error, self.two_qubit_error, self.readout_error


@dataclass(frozen=True)
class TrainingConfig:
    steps: int = 300
    batch_size: int = 10
    learning_rate: float = 0.02
    minimum_learning_rate: float = 0.001
    initialization_std: float = 0.1
    evaluate_every: int = 10
    patience: int | None = None  # Number of evaluation checks without improvement.
    gradient_clip: float | None = None

    def __post_init__(self):
        if min(self.steps, self.batch_size, self.evaluate_every) < 1:
            raise ValueError("steps, batch_size, and evaluate_every must be positive.")
        if not 0 <= self.minimum_learning_rate <= self.learning_rate or self.learning_rate <= 0:
            raise ValueError("Require 0 <= minimum_learning_rate <= learning_rate, with lr > 0.")
        if self.initialization_std <= 0:
            raise ValueError("initialization_std must be positive.")
        if self.patience is not None and self.patience < 1:
            raise ValueError("patience must be positive or None.")
        if self.gradient_clip is not None and self.gradient_clip <= 0:
            raise ValueError("gradient_clip must be positive or None.")


@dataclass(frozen=True)
class UnlearningConfig:
    forget_class: int = 2
    forget_definition: Literal["observed", "original"] = "observed"
    alpha: float = 1.0
    regularization: float = 0.01  # The paper's lambda.
    beta: float = 1.0
    anchor_fraction: float = 0.25
    calibration_size: int | None = 32
    methods: tuple[str, ...] = ("guided", "uniform")
    training: TrainingConfig = field(default_factory=lambda: TrainingConfig(steps=100))

    def __post_init__(self):
        if self.forget_class not in (0, 1, 2):
            raise ValueError("forget_class must be 0, 1, or 2.")
        if self.forget_definition not in ("observed", "original"):
            raise ValueError("Unknown forget-set definition.")
        if self.alpha < 0 or self.regularization < 0 or self.beta <= 0:
            raise ValueError("Require alpha/lambda >= 0 and beta > 0.")
        if not 0 < self.anchor_fraction <= 1:
            raise ValueError("anchor_fraction must be in (0, 1].")
        if self.calibration_size is not None and self.calibration_size < 1:
            raise ValueError("calibration_size must be positive or None.")
        if not self.methods or len(set(self.methods)) != len(self.methods):
            raise ValueError("Choose unique unlearning methods.")
        if any(method not in ("guided", "uniform") for method in self.methods):
            raise ValueError("Supported methods are guided and uniform.")


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = DEFAULT_SEED
    data: DataConfig = field(default_factory=DataConfig)
    poison: PoisonConfig = field(default_factory=PoisonConfig)
    circuit: CircuitConfig = field(default_factory=CircuitConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    unlearning: UnlearningConfig = field(default_factory=UnlearningConfig)
    classical_baselines: bool = True
    label_shuffle_control: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "ExperimentConfig":
        """Reconstruct nested settings from a saved config.json."""
        values = dict(values)
        unlearning = dict(values.pop("unlearning", {}))
        if "training" in unlearning:
            unlearning["training"] = TrainingConfig(**unlearning["training"])
        if "methods" in unlearning:
            unlearning["methods"] = tuple(unlearning["methods"])
        return cls(data=DataConfig(**values.pop("data", {})),
                   poison=PoisonConfig(**values.pop("poison", {})),
                   circuit=CircuitConfig(**values.pop("circuit", {})),
                   backend=BackendConfig(**values.pop("backend", {})),
                   training=TrainingConfig(**values.pop("training", {})),
                   unlearning=UnlearningConfig(**unlearning), **values)
