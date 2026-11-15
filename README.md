# Distribution-guided quantum machine unlearning under poisoning

Phase 1 evaluates an existing class-level unlearning method on MNIST digits **0, 1, and 2**. We forget digit **2** and measure what happens to retained digits **0 and 1**, before and after controlled training-data poisoning.

**Current focus:** clean data preparation and baseline training, plus preparation of poisoned datasets. The **gold reference model** is the unpoisoned classifier trained on original digit-0/1/2 images and labels. Poisoned-model training will follow. Models retrained after removing designated forget records are called **retained-data retraining references** throughout the implementation.

The implementation is in `src/`. [main_project.ipynb](main_project.ipynb) is the central research workflow, from data preparation and clean baselines through poisoning, unlearning, and evaluation. It contains saved data/preprocessing and classical baseline outputs; newly added analysis and poisoned-data preparation cells remain unexecuted. Sections 1–9 cover clean baselines and poisoned-dataset preparation; Sections 10–15 have Markdown outlines with empty code cells. Importing the package does not download data, train models, or contact IBM.

### Research notebook

The fixed project seed is **6304**, defined as `DEFAULT_SEED` in `src/config.py` and imported by the notebook. Data selection/splits, PCA, poisoning, classical models, quantum initialization, and training use this seed. Simulator batch seeds and later stage streams are generated deterministically from the experiment seed; the noise-sweep default uses only 6304. Restart the kernel after changing the configuration. Saved outputs from earlier seeds retain their provenance; rerunning with 6304 creates matching new runs instead of reusing those earlier results.

Open the notebook from the project root and select the QML Python environment. Sections 1–8 cover the research question, GPU/Aer availability, MNIST data pipeline, five fixed stratified folds, training-only PCA, four classical baselines, and one clean six-qubit VQC on fold 0. Short cells include comments and guidance for interpreting future outputs.

The actual circuit and every generated figure are displayed inline beside the relevant analysis. Tables show split counts, PCA variance, gate counts, individual-fold accuracy, aggregate baseline performance, and per-digit recall/error counts. When a quantum result exists, the notebook also displays its learning curve, initial/selected/final checkpoint comparison, confusion matrix, and computation costs. Section 8 saves the figures and analysis tables for reporting; rerun the new analysis/export cells in order to populate their outputs and saved copies.

Section 9 prepares a clean control and two separate attacks at 5%, 10%, and 20% for the selected training fold: a configurable white 3-by-3 bottom-right pixel patch on digit 2, and label flips from digit 0 to 2. Rates default to fractions of the eligible class and round down. Validation/test data stay clean. Short cells check unchanged originals, record alignment, intended modifications, and nested selections, then display actual counts, label balance, before/after examples, annotations, and potential forget-set membership. These are data audits; poisoned-model preprocessing and training remain deferred. After Sections 2–4, Section 9 can run independently of model training.

Section 9.7.1 adds detailed sample figures with original and poisoned images, original/observed labels, and absolute pixel differences on a shared 0–255 scale. Pixel-patch images have an orange annotation around the patch; label-flip figures show identical images with different labels and zero pixel difference. The figures use the configured preview rate (20% by default), appear inline, and join the Section 9.9 PNG/PDF exports. These new cells are unexecuted; existing notebook cells and outputs are preserved.

Executing the Section 9 save cells writes annotated variants, a collection index under `data/annotated/collections/`, and CSV/PNG/PDF analysis exports under `results/tables/poison_preparation_<id>/` and `results/figures/poison_preparation_<id>/`. The index records split identity, settings, source identity, and variant paths. New preparation cells have not been executed while adding this code.

The development pool contains 100 records per digit (0/1/2) from the official training pool. Each default fold has 192 training, 48 validation, and 60 test records. The official test pool remains unused. Classical baselines run on all five folds; the first quantum model uses only fold 0 and seed 6304. Its same-fold comparison must not be described as a five-fold quantum result. The default quantum budget is 300 mini-batch updates with 2,048 shots and no device-noise model.

- `DOWNLOAD_MNIST=True` permits download when the loading cell is executed; existing data is verified and reused.
- `RUN_QUANTUM_TRAINING=True` enables one clean gold VQC for the selected fold and seed when Section 7 is executed. A matching completed run is reused; otherwise the model is trained and saved. Set it to `False` to load matching saved quantum results without starting fresh training.
- `REUSE_SAVED_RUNS=True` reuses matching completed baseline/quantum runs. Set it to `False` for a fresh attempt. Interrupted attempts are preserved.

To train the current gold reference, restart the kernel and run Sections 2–8 in order. The notebook prints the selected fold, record counts, and training budget before the single quantum training call. Current settings are fold 0, seed 6304, 300 updates, batch size 10, and 2,048 shots. Enabling the flag in the notebook file has not started a training run. Section 9 remains data preparation only; Sections 10–15 remain empty.

Code is organized by research responsibility. `prepare_fold_features()` belongs to `src/data/preprocessing.py`; `run_classical_cv()` and `run_clean_vqc()` belong to `src/experiments.py`; saving and loading belong to `src/artifacts.py`. The clean baseline stages can run independently before the complete poisoning/unlearning experiment. They save records, preprocessing, models, predictions, metrics, source snapshots, and environment metadata under `results/runs/classical_cv/` and `results/runs/clean_vqc/`. Section 8 exports CSV tables to `results/tables/<view_id>/` and PNG/PDF figures with provenance to `results/figures/<view_id>/`. No data or result artifacts have been generated by preparing the notebook.

Restart the kernel and run cells in order after changing code or settings. Paths are relative to the project root, including saved split references, so the project can move to WSL2. The early check distinguishes physical GPU detection from GPU support in the selected Aer environment; Section 6 contains a small execution probe. WSL2 GPU setup is separate and has not been performed.

## 1. Environment

Tested with Python **3.12.14** in the existing **QML** conda environment. Direct dependencies are pinned to the verified installed versions in [requirements.txt](requirements.txt). No environment packages were changed during implementation.

From the project root:

```powershell
conda activate QML
python -m unittest discover -s tests -v
```

For a separate Python 3.12 environment:

```powershell
python -m pip install -r requirements.txt
```

Tests use standard-library unittest; pytest is not required. ipykernel supports the notebook, and pylatexenc supports circuit drawings.

### GPU and IBM

The laptop has an **NVIDIA RTX 3060 Laptop GPU with 6 GB VRAM**. Its installed Aer 0.17.2 build currently exposes only **CPU**. The code checks Aer capabilities rather than assuming that an NVIDIA GPU implies Aer support.

`BackendConfig(device="auto")` selects GPU when Aer exposes it, otherwise CPU. Requesting `device="GPU"` permits an explicit warning and CPU fallback; `allow_cpu_fallback=False` makes an unavailable/failed GPU a hard error. Runtime fallbacks are recorded. Circuit batches and memory limits are configurable; the default memory limit is 4,096 MB. `benchmark_aer_devices()` compares available devices on supplied inputs so selection can be based on measured timings.

Aer documents its prebuilt GPU package for Linux. A compatible Linux/WSL environment and GPU-enabled Aer installation require separate setup and validation; the current Windows environment was not modified. [Aer simulator documentation](https://qiskit.github.io/qiskit-aer/tutorials/1_aersimulator.html)

Local runs require no IBM account. The optional IBM adapter imports qiskit-ibm-runtime only when `use_ibm_hardware=True`, requires an explicit backend name, and uses an already configured account. That interface was tested with a mocked service and local backend, **not an actual IBM device**. The optional dependency version is commented in requirements.txt. Once access exists, use saved weights for a small hardware evaluation before considering hardware training.

## 2. Modules

| Location | Responsibility |
|---|---|
| `src/config.py` | Serializable experiment settings |
| `src/data/dataset.py` | Verified MNIST download, stable IDs, subsets, holdout/CV splits |
| `src/data/preprocessing.py` | Training-only normalization, PCA, angle scaling, fold features |
| `src/data/poisoning.py` | Pixel patches, label flips, annotation manifests, forget masks |
| `src/models/circuits.py` | Malik circuit and resource counts |
| `src/models/backends.py` | Batched Aer, device noise, GPU detection, optional IBM |
| `src/models/vqc.py` | Expectations, probabilities, chain-rule parameter-shift gradients |
| `src/models/classical.py` | Majority, logistic, linear SVM, RBF SVM |
| `src/training.py` | Mini-batch Adam, cosine schedule, checkpoint selection |
| `src/unlearning/targets.py` | Fixed targets and retained anchors |
| `src/unlearning/methods.py` | Guided/uniform class unlearning |
| `src/evaluation/metrics.py` | Retention, forgetting, divergence, seed summaries |
| `src/evaluation/plots.py` | Explanatory figures and comparison plots |
| `src/artifacts.py` | Run identities, fold artifacts, snapshots, integrity checks, reload |
| `src/experiments.py` | Classical CV, clean VQC, complete experiments, studies, restoration, noise sweeps |

## 3. Data and saved results

`load_mnist(download=True)` explicitly downloads `data/raw/mnist.npz` when it is missing, and verifies an existing copy before reuse. Its URL and SHA-256 come from the [Keras MNIST loader](https://github.com/keras-team/keras/blob/master/keras/src/datasets/mnist.py); Keras/TensorFlow are not dependencies. A checksum mismatch raises an error. Downloader tests use a small mocked archive; the project now also contains the archive used by the saved data-preparation outputs.

Original labels and sample IDs remain available after poisoning. IDs distinguish official training and test records. The default protocol uses official training images for training/validation and the separate official test pool for evaluation. With `split_strategy="stratified_kfold"`, outer test folds instead come from the selected official training pool; each outer training partition gets its own validation holdout. The official test pool is unused in that mode. This is fixed-fold evaluation, not a nested hyperparameter search.

```text
data/
  raw/mnist.npz
  splits/split_<fingerprint>.json
  annotated/<condition>_p<percentage>_<fingerprint>/
    images.npz
    annotations.csv
    metadata.json
  annotated/collections/<collection_id>.json

results/
  runs/<condition>_<seed>_<fold>_<identity>/
    manifest.json
    config.json
    provenance.json
    source.zip
    datasets.json
    preprocessing.joblib
    preprocessing.json
    features.npz
    retraining_ids.npz
    weights.npz
    predictions.npz
    metrics.json
    resources.json
    history.csv
    checkpoints/
    histories/
    references/
  figures/
  tables/
```

Annotation variants are created on demand. They contain observed images, original/observed labels, IDs, poison flags, settings, seeds, and realized corruption counts. Original downloads and annotated image archives are ignored by Git; split manifests and result artifacts remain available for deliberate versioning.

Run identity includes configuration, dataset content, code, and environment. Completed matching runs can be reused; incomplete runs cannot become completed cached results. `reuse=False` creates a new attempt rather than overwriting an existing one. Saved artifacts have integrity checksums verified on reload. `source.zip` preserves source and requirements. Restart the Python kernel after changing implementation files so imported code matches the saved source identity.

## 4. Explicit research defaults

These are starting settings, not optimized choices or reproduced paper results.

| Setting | Default |
|---|---|
| Selected official training-pool images | 100 per digit, with 20% reserved for validation |
| Selected official test images | 50 per digit |
| Data/split seed | 0, separate from optimization seed |
| Circuit | Six qubits; two feature-map/ansatz blocks; three ansatz repetitions per block |
| Weights / inputs | 72 trainable weights plus four input parameters |
| Readout | Z on q3/q4/q5, followed by unscaled three-class softmax |
| Simulation | Aer, 2,048 shots; `shots=None` explicitly requests exact diagnostics |
| Initial training | 300 mini-batch updates; batch 10; Adam; cosine learning rate 0.02 to 0.001 |
| Initialization | Gaussian standard deviation 0.1 |
| Unlearning | 100 mini-batch updates, with the same optimizer defaults |
| Preservation / sharpness | alpha=1, lambda=0.01, beta=1 |
| Anchors / calibration | 25% per retained observed class, at least one each / up to 32 forget records |
| Checkpoint evaluation | Every 10 updates, including initial and final checkpoints |
| Poisoning | Disabled until a condition is selected |

Learning-rate endpoints, initialization scale, unlearning budget, exact splits, and calibration/anchor sizes are explicit implementation choices where the paper does not supply a complete reproducible setup.

### Poisoning and forgetting

- `kind="pixel_patch"` places a configurable bottom-right square in selected target-class images. Defaults are 3 by 3 pixels and value 255. This is our experimental attack, not a pixel pattern attributed to either paper.
- `kind="label_flip"` changes selected digit-0 training labels to digit 2.
- `denominator="eligible"` applies the rate to eligible-class records. `"all_training"` uses the full training size and rejects counts exceeding the eligible population. Counts are rounded down; zero realized corruption is warned about.
- With `forget_definition="observed"`, relabeled digit-0 records join the poisoned model's class-2 forget set. Its retained-data retraining reference excludes those exact records. The clean model forgets true digit 2 and has its own matched retraining reference. Identical retraining problems are reused and recorded.
- The `"original"` forget definition is an explicit alternative that changes the question being evaluated.

Validation/test images remain clean. Model roles share architecture, optimization seed, and initialization procedure. Retained-data references start from fresh initialization on clean retained records; their saved roles are `retrained_clean` and `retrained_poisoned`. The unpoisoned gold reference uses the saved role `clean`.

### Preprocessing and leakage

`preprocessing="refit"` is the default: each original model fits PCA/scaling on its own training inputs; each retrained reference fits them on its clean retained training records. Unlearning keeps its original representation fixed. Thus retained-data retraining updates the full pipeline while unlearning updates classifier parameters. Report that limitation when discussing removal of all data influence.

`"shared_clean_train"` instead fits one representation on clean training images and shares it. This isolates other changes but uses clean-data knowledge and includes forget records in the representation; it is not full-pipeline deletion.

Neither policy fits preprocessing on validation/test data. PCA values outside the training-fitted range are clipped to [0, pi].

### Objective and optimization

The guided target is computed once from original-model predictions on a training calibration subset. Its forgotten component is zero; retained components are sharpened by beta and normalized. For forgetting class 2, the uniform comparison is (0.5, 0.5, 0). Full original predictions on training anchors are cached.

```text
mean_forget CE(target, prediction)
+ alpha * mean_anchor CE(original_anchor_prediction, prediction)
+ lambda * squared_distance(weights, original_weights)
```

Each method starts from the same original checkpoint. Supervised checkpoint selection uses validation cross-entropy; unlearning selection uses the complete training unlearning objective. Test metrics never select checkpoints. A step is a mini-batch update, not an epoch.

Parameter shifts differentiate expectations, followed by the softmax/cross-entropy chain rule. The implementation does not shift the complete nonlinear loss. All 72 components are checked against finite differences. The final six RZ angles remain for fidelity even though they do not affect ideal Z-readout probabilities.

With three unscaled Z logits, probabilities have a generic lower bound of approximately 0.06338. A zero target does not imply attainable zero predicted confidence. Changing logit scale, repetitions, or entanglement is an architectural ablation.

### Classical baselines and summaries

Classical models use the same four features as the corresponding VQC. SVM probabilities use training-only stratified sigmoid calibration with up to five folds. Hyperparameters are fixed, and metrics consistently use probability argmax. [Calibration API](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html)

Metrics include retained-only accuracy under the normal three-class decision rule; forgotten-class confidence/recall on clean test digit 2; and separate before/after changes. Retraining comparisons include full and retained-conditioned **KL(reference || candidate)**, where the reference was retrained on matched retained records. Forgotten-class mass on retained examples is reported separately.

`results_table()` records protocol IDs, conditions, rates, roles, seeds, and folds. `summarize_runs()` averages folds within each seed before reporting mean/SD across seed means. Folds have equal weight; unequal folds need an explicitly weighted analysis. Single-seed SD is undefined and its error bars are omitted. Different code/environment/configuration protocols remain separate by default.

## 5. Calling the code later

The notebook will call these functions in smaller stages. This deliberately short diagnostic uses the full paper circuit, but is not the final research budget:

```python
from src.config import (
    BackendConfig, DataConfig, ExperimentConfig, PoisonConfig,
    TrainingConfig, UnlearningConfig,
)
from src.experiments import run_experiment

short_training = TrainingConfig(steps=2, batch_size=2, evaluate_every=1)
config = ExperimentConfig(
    data=DataConfig(train_per_class=20, test_per_class=5),
    poison=PoisonConfig(kind="label_flip", rate=0.2, seed=6304),
    backend=BackendConfig(shots=256, device="auto"),
    training=short_training,
    unlearning=UnlearningConfig(training=short_training),
)

# This explicit call downloads MNIST if needed and performs training.
result = run_experiment(config, download=True)
```

The default circuit requires 145 bindings per sample for a forward value and all shifted derivatives, before extra validation/reference work. Start with a diagnostic budget before launching complete studies.

Stage-level functions include `prepare_splits`, `apply_poisoning`, `ImagePreprocessor`, `build_malik_circuit`, `VQC`, `fit_vqc`, `prepare_references`, and `unlearn`. They allow inspection at each stage.

To analyze a saved run:

```python
from src.artifacts import load_run
from src.experiments import results_table, restore_vqc

result = load_run("results/runs/YOUR_RUN_ID")
table = results_table([result])

# Optional new predictions from a trusted local checkpoint; no training.
model, preprocessing, weights = restore_vqc(
    result["path"], "poisoned_guided"
)
```

`load_run()` reads numeric/JSON outputs and checks integrity. `restore_vqc()` explicitly loads the trusted local preprocessing.joblib object; use it only with trusted runs.

Build repeated conditions with `dataclasses.replace` and pass the configurations to `run_study()`. Keep data, poison, and optimization seeds fixed at 6304 for the current protocol; multi-seed studies are deferred. Select CV strategy/fold explicitly. Change alpha, lambda, beta, anchor fraction, target methods, architecture, or noise through configuration. `label_shuffle_control=True` adds a separate quantum negative control.

`evaluate_noise_sweep()` evaluates **fixed weights** at one-qubit error p and two-qubit error 2p. For noisy training, configure noise in the main experiment instead. Device noise and poisoning are distinct variables. Exact readout-noise evaluation is disallowed; use finite shots for it.

## 6. Figures and tables

Plot helpers return Matplotlib figures. They cover paired images, partition counts, PCA variance, circuits, optimization curves, paired confusion matrices with a common scale, confidence histograms, seed summaries, and poisoning-response curves.

```python
from src.evaluation.plots import plot_confusion_comparison, save_figure

models = result["metrics"]["models"]
figure = plot_confusion_comparison(
    models["poisoned"]["confusion_matrix"],
    models["poisoned_guided"]["confusion_matrix"],
)
save_figure(
    figure, "results/figures/poisoned_guided_confusion",
    source_run_ids=[result["path"].name],
    description="Clean test predictions before and after class-2 unlearning.",
)
```

Figure exports include PNG/PDF and JSON provenance. `run_study()` exports a run-linked CSV to results/tables; DataFrames also support `to_csv()` and `to_latex()`. Filter to the intended attack/protocol before combining results. Interpret favorable, unfavorable, and inconclusive outcomes.

## 7. Validation and sources

The suite covers download integrity with mocked data, poisoning records, split disjointness, training-only PCA, gate counts, readout ordering, all 72 loss gradients, the complete unlearning gradient, finite shots/noise, GPU fallback, the IBM interface offline, checkpoint selection, metric semantics, complete run storage/reload, integrity failures, and figure exports.

Integration tests use synthetic images and short budgets, including finite-shot training with the full paper circuit. They are software checks, not MNIST research findings. No real IBM job, GPU simulation, or full MNIST study has been executed.

The notebook and the independently callable baseline stages have received structure, syntax, and import checks only. No notebook cells, downloads, or training were executed while preparing or reorganizing them, as requested. Their first runtime validation remains pending.

- Malik, Khalid, and Faryad, [Distribution-Guided and Constrained Quantum Machine Unlearning](https://arxiv.org/abs/2601.04413), supplied v2: core circuit and method.
- Chen and Zhang, [Superior resilience to poisoning and amenability to unlearning in quantum machine learning](https://doi.org/10.1038/s41467-026-70420-4): poisoning/repair motivation; its binary amplitude-encoded QNN is a different task and model.
- [Local research context](RESEARCH_CONTEXT.md): detailed reading notes, course requirements, and scope.

