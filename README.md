# Distribution-guided quantum machine unlearning under poisoning

Phase 1 evaluates an existing class-level unlearning method on MNIST digits **0, 1, and 2**. We forget digit **2** and measure what happens to retained digits **0 and 1**, before and after controlled training-data poisoning.

**Current focus:** the complete Phase 1 workflow: clean and poisoned training, matched retained-data references, guided/uniform class unlearning, and evaluation. The **gold reference model** is the unpoisoned classifier trained on original digit-0/1/2 images and labels. The revised protocol uses 420 images, exact expectations, six nonzero poisoning rates, and all five folds. The revised experiment completed on Linux/RTX 3070 and its guided-unlearning implementation was audited. The notebook now defaults to reviewing those saved results on Windows without fitting models. Models retrained after removing designated forget records are called **retained-data retraining references** throughout the implementation.

The implementation is in `src/`. [main_project.ipynb](main_project.ipynb) is the central research workflow. Sections 1-9 retain existing code and saved outputs; Sections 10-14 cover poisoned-model training, retained-data references, guided/uniform unlearning, optional experiments, and exports. Only Phase 2 (Section 15) remains empty. Importing modules does not download data, train models, or contact IBM. The notebook has not been executed during this update.

### Research notebook

**Saved-results review on Windows (19 November 2026):** keep `SAVED_RESULTS_ONLY=True`, open `main_project.ipynb` with the Windows QML kernel, restart the kernel, and Run All. The original experiment is explicitly selected by `SAVED_REPORT`, `SAVED_GOLD_RUN`, and `SAVED_CLASSICAL_RUN`. All fitting/unlearning, force-retraining, download, cleanup, and optional experiment flags remain `False`; `RUN_FOLD_STUDY=True` loads all five folds. Missing, changed, or incompatible artifacts stop review instead of starting training. Classical estimators and PCA are not fitted, and no new quantum predictions are computed.

The loader in `src/evaluation/saved_results.py` verifies the 230 linked stage runs plus the original gold/classical baselines, restores original split IDs and saved numeric features, and preserves Linux training provenance. Current Windows environment information is recorded separately. Saved arrays and metrics are read directly without unpickling cross-version sklearn objects. Primary tables are rebuilt and compared against the original report before export. New analysis goes to `results/tables/phase1_review_<id>/` and `results/figures/phase1_review_<id>/`; original training artifacts are unchanged. The read-only guard remains active until kernel restart. New training still uses strict source/environment identity checks; this review path does not resume a Linux optimizer on Windows.

The original experiment report is `results/tables/phase1_0e7dbe83c914e5f8/`. Later sections that describe enabling training document the experiment workflow, not the current review defaults.

The fixed project seed is **6304**, defined as `DEFAULT_SEED` in `src/config.py` and imported by the notebook. Data selection/splits, PCA, poisoning, classical models, quantum initialization, and training use this seed. Simulator batch seeds and later stage streams are generated deterministically from the experiment seed; the noise-sweep default uses only 6304. Restart the kernel after changing the configuration. Saved outputs from earlier seeds retain their provenance; rerunning with 6304 creates matching new runs instead of reusing those earlier results.

Open the notebook from the project root and select the QML Python environment. Sections 1–8 cover the research question, GPU/Aer availability, MNIST data pipeline, five fixed stratified folds, training-only PCA, four classical baselines, and one clean six-qubit VQC on fold 0. Short cells include comments and guidance for interpreting future outputs.

The actual circuit and every generated figure are displayed inline beside the relevant analysis. Tables show split counts, PCA variance, gate counts, individual-fold accuracy, aggregate baseline performance, and per-digit recall/error counts. When a quantum result exists, the notebook also displays its learning curve, initial/selected/final checkpoint comparison, confusion matrix, and computation costs. Section 8 saves the figures and analysis tables for reporting; rerun the new analysis/export cells in order to populate their outputs and saved copies.

Section 9 prepares a clean control and two separate attacks at 5%, 10%, 20%, 30%, 40%, and 50% for the selected training fold: a configurable white 3-by-3 bottom-right pixel patch on digit 2, and label flips from digit 0 to 2. Rates default to fractions of the eligible class and round down. Validation/test data stay clean. Short cells check unchanged originals, record alignment, intended modifications, and nested selections, then display actual counts, label balance, before/after examples, annotations, and potential forget-set membership. These are data audits; poisoned-model preprocessing and training execute separately in Section 10. After Sections 2–4, Section 9 can run independently of model training.

Section 9.7.1 adds detailed sample figures with original and poisoned images, original/observed labels, and absolute pixel differences on a shared 0–255 scale. Pixel-patch images have an orange annotation around the patch; label-flip figures show identical images with different labels and zero pixel difference. The figures use the configured preview rate (50% by default), appear inline, and join the Section 9.9 PNG/PDF exports. Existing outputs are preserved as pilot evidence until the revised cells are executed.

Executing the Section 9 save cells writes annotated variants, a collection index under `data/annotated/collections/`, and CSV/PNG/PDF analysis exports under `results/tables/poison_preparation_<id>/` and `results/figures/poison_preparation_<id>/`. The index records split identity, settings, source identity, and variant paths. New preparation cells have not been executed while adding this code.

The development pool contains **420 images: 140 per digit (0/1/2)** from the official training pool. Each of five fixed folds has **268 training, 68 validation, and 84 test records** (28 test records per digit). The official test pool remains unused. Section 7 inspects fold 0; Section 12.1 completes all remaining folds for the main quantum study. The budget remains 300 supervised updates, batch size 10, and 100 unlearning updates. Local Aer uses **statevector with `shots=None`**, giving exact noiseless expectations. This matches the paper's expectation setting; MNIST, the sample count, and our folds are our own design choices.

- `DOWNLOAD_MNIST=True` permits download when the loading cell is executed; existing data is verified and reused.
- `RUN_QUANTUM_TRAINING=True` enables one clean gold VQC for the selected fold and seed when Section 7 is executed. A matching completed run is reused; otherwise the model is trained and saved. Set it to `False` to load matching saved quantum results without starting fresh training.
- `FORCE_RETRAIN=False` loads completed compatible models, resumes interrupted quantum models from their last saved update, and trains missing models when their stage is enabled. Set it to `True` to train each distinct enabled model from scratch. Re-run the control cell to create a new `CheckpointSession`; share that session across condition loops so identical clean/reference models train only once.

To train the current gold reference, restart the kernel and run Sections 2–8 in order. The notebook prints the selected fold, record counts, and training budget before the single quantum training call. The selected-fold illustration uses fold 0, seed 6304, 300 updates, batch size 10, and exact expectations. Run through Section 14 for the full five-fold study. Enabling the flag in the notebook file has not started a training run. Section 9 remains data preparation only; Sections 10 and 11 now implement the subsequent training and unlearning stages.

Code is organized by research responsibility. `prepare_fold_features()` belongs to `src/data/preprocessing.py`; `run_classical_cv()` and `run_clean_vqc()` belong to `src/experiments.py`; saving and loading belong to `src/artifacts.py`. The clean baseline stages can run independently before the complete poisoning/unlearning experiment. They save records, preprocessing, models, predictions, metrics, source snapshots, and environment metadata under `results/runs/classical_cv/` and `results/runs/clean_vqc/`. Section 8 exports CSV tables to `results/tables/<view_id>/` and PNG/PDF figures with provenance to `results/figures/<view_id>/`. No data or result artifacts have been generated by preparing the notebook.

Restart the kernel and run cells in order after changing code or settings. Paths are relative to the project root, including saved split references, so the project can move to WSL2. The early check distinguishes physical GPU detection from GPU support in the selected Aer environment; Section 6 contains a small execution probe. WSL2 GPU setup is separate and has not been performed.

## Completing Phase 1 in the notebook

After preparing the variants in Section 9, run Sections 10 and 11 in order when ready. `RUN_POISONED_TRAINING=True` and `RUN_UNLEARNING=True` enable the core work when their cells execute. Set either to `False` for matching-cache lookup only. Main conditions are the clean control and each prepared pixel-patch/label-flip rate. Keep seed 6304 and local Aer. New code has not launched any MNIST training or unlearning.

Section 10 reuses `quantum_run` from Section 7 when supplied. If a source change prevents the old run matching the automatic cache, set `RUN_QUANTUM_TRAINING=False` and explicitly set `GOLD_REFERENCE_PATH` to a trusted existing clean run. Its split, seed, data/circuit/backend/training settings and artifact integrity are checked; original predictions and provenance are preserved. Without a supplied gold run, the stage cache fits one clean model for matching conditions, not a new one per poison rate.

Supervised models are cached by their actual training data and fitting settings under `results/runs/supervised_models/`. Matched retained-data references use original images/labels after excluding exact forget IDs. Identical reference problems are reused across conditions. Unlearning stages are cached separately under `results/runs/unlearning/`, with each new method starting from the parent checkpoint. `results/conditions/condition_<id>.json` links scientific roles (`clean`, `poisoned`, `retrained_clean`, `retrained_poisoned`, and method roles) to numeric stage artifacts whose internal weight key is `model`.

Quantum progress is atomically saved after every completed update to `checkpoints/clean.resume.json` (Section 7) or `checkpoints/model.resume.json` (Sections 10-11) inside the run folder. It includes current/best weights, both Adam moments, step/schedule position, early-stopping state, history, minibatch/Aer random streams, and resource counters. Resuming keeps the original training budget and redoes only any update interrupted before its checkpoint was committed. Unlearning also saves its fixed teacher distributions before optimization. Completed best weights remain in `checkpoints/*.npz`. Classical estimators restart an unfinished fit; completed classical stage artifacts are reused.

For example, after model 1 finishes and model 2 stops, rerunning with `FORCE_RETRAIN=False` loads model 1, continues model 2 from its own checkpoint, then starts any missing model 3. This does not initialize unrelated models from model 1. Changed settings/data/splits/source/environment start a new matching run instead of continuing incompatible optimizer state. Only changes affecting a stage invalidate that stage (unlearning-only settings leave compatible supervised fits reusable).

Each stage directory has `active_checkpoints.json`, whose links are overwritten as new attempts become active. Forced retraining replaces the active model selection and writes fresh checkpoints; completed run folders remain intact except for the separately authorized one-time pilot checkpoint cleanup below. The latest matching attempt is preferred on rerun, including an interrupted forced attempt. Execution flags still apply: force does not enable a disabled stage. During a forced session, an old `GOLD_REFERENCE_PATH` is ignored; only a gold model produced in that session can be imported into Section 10.

Section 10 includes a frozen-clean-PCA diagnostic of actual image/angle changes. Section 11 displays targets, anchor quality, optimization curves, confusion matrices, confidence distributions, per-class scores, KL against matched retraining, poisoning damage before unlearning, and additional changes after unlearning. Wilson accuracy intervals describe finite-test-sample uncertainty under a binomial approximation, not uncertainty across seeds. References refit preprocessing by default while unlearning keeps parent preprocessing fixed; report this distinction.

Section 12.1 is enabled with `RUN_FOLD_STUDY=True` and evaluates every paired condition on all five folds. Exports use all-fold scores, per-class recall, full/conditional KL, and within-fold poisoning-effect differences, with means, descriptive fold SD, and completion counts. Missing runs remain explicit; shared clean models are not pooled as extra independent observations. Subsequent one-factor ablations, shuffled-label controls, noise/shot sweeps, and IBM inference remain disabled. Optional noise and IBM evaluation explicitly use 2,048 shots. Fold variability is descriptive; the project still uses only seed 6304. IBM requires a named accessible backend and configured account and is unnecessary for local execution. Phase 2 algorithm extensions remain deferred.

Section 14 exports all newly displayed figures as PNG/PDF and tables as CSV, plus source run/condition provenance, under `results/figures/phase1_<id>/` and `results/tables/phase1_<id>/`. Missing runs are listed explicitly. Interpretation notes stay pending until experimental results are inspected; no favorable result is assumed.

### One-time pilot checkpoint cleanup

`REMOVE_PILOT_CHECKPOINTS=True` authorizes Section 3.4 to remove incompatible pilot model checkpoints on the **next user execution**, before training. It removes saved weights, optimizer progress, and fitted classifier files under the four notebook stage directories. Metrics, predictions, histories, preprocessing, source snapshots, annotated data, and report figures stay available. Original manifests/checksums are recorded in `results/maintenance/mnist420_exact_v1.json`; affected runs are marked `retired` and cannot be resumed or imported as models.

The journal freezes the deletion list before any files are removed and resumes an interrupted cleanup. It runs once; rerunning the notebook does not erase new checkpoints. Compatible 420-image/exact runs are protected. No recursive directory removal is used. `FORCE_RETRAIN=False` then resumes compatible interrupted work and trains missing models. The changed protocol starts fresh automatically; set `FORCE_RETRAIN=True` only to deliberately retrain the new experiment again.

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
| `src/checkpoints.py` | Per-model resume/retrain policy, active checkpoint links, random-stream restoration |
| `src/unlearning/targets.py` | Fixed targets and retained anchors |
| `src/unlearning/methods.py` | Guided/uniform class unlearning |
| `src/evaluation/metrics.py` | Retention, forgetting, divergence, seed summaries |
| `src/evaluation/plots.py` | Explanatory figures and comparison plots |
| `src/artifacts.py` | Run identities, fold artifacts, snapshots, integrity checks, reload |
| `src/experiments.py` | Classical CV, clean VQC, complete experiments, studies, restoration, noise sweeps |
| `src/experiment_stages.py` | Independently saved supervised/unlearning stages, exact-record reference reuse, PCA poisoning diagnostics |
| `src/evaluation/analysis.py` | Run-linked Phase 1 metrics, changes, targets, intervals, and unique resource tables |

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

Run identity includes configuration, dataset content, code, and environment. Completed matching runs can be reused; incomplete runs are resumed only by the staged notebook APIs and never treated as completed research results. `reuse=False` creates a fresh attempt; the notebook uses a shared `CheckpointSession(force_retrain=FORCE_RETRAIN)` instead. The legacy all-in-one `run_experiment` retains its new-attempt behavior and is not the notebook execution path. Saved artifacts have integrity checksums verified on reload. `source.zip` preserves source and requirements. Restart the Python kernel after changing implementation files so imported code matches the saved source identity.

## 4. Explicit research defaults

The table below describes reusable API defaults, not the notebook protocol above. The notebook explicitly overrides data selection/splits and simulation. These are starting settings, not optimized choices or reproduced paper results.

| Setting | Default |
|---|---|
| Selected official training-pool images | 100 per digit, with 20% reserved for validation |
| Selected official test images | 50 per digit |
| Data/split seed | 6304, also the root optimization/poisoning seed |
| Circuit | Six qubits; two feature-map/ansatz blocks; three ansatz repetitions per block |
| Weights / inputs | 72 trainable weights plus four input parameters |
| Readout | Z on q3/q4/q5, followed by unscaled three-class softmax |
| Simulation | Aer, 2,048 shots; `shots=None` requests exact expectations for training or evaluation |
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

The notebook calls the staged APIs above; the original all-in-one runner also remains available. This deliberately short diagnostic uses the full paper circuit, but is not the final research budget:

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

The notebook and the independently callable baseline stages have received structure, syntax, and import checks only. No notebook cells, downloads, or training were executed while preparing or reorganizing them, as requested. The new stage APIs have also been exercised with short synthetic integration tests; full MNIST notebook execution remains the user's next step.

- Malik, Khalid, and Faryad, [Distribution-Guided and Constrained Quantum Machine Unlearning](https://arxiv.org/abs/2601.04413), supplied v2: core circuit and method.
- Chen and Zhang, [Superior resilience to poisoning and amenability to unlearning in quantum machine learning](https://doi.org/10.1038/s41467-026-70420-4): poisoning/repair motivation; its binary amplitude-encoded QNN is a different task and model.
- [Local research context](RESEARCH_CONTEXT.md): detailed reading notes, course requirements, and scope.


Protocol-update verification (18 November): **38 synthetic unittest checks passed**. New checks cover exact-expectation gold training/cache reuse, 420-image fold sizes, nested poison selections through 50%, fold statistics and missing/duplicate folds, and one-time checkpoint retirement including interrupted cleanup and path safety. Existing Adam/RNG resume and research-pipeline checks also passed. Notebook schema and code syntax passed; existing outputs were preserved. No notebook execution, new MNIST training, or cleanup of real checkpoints occurred during this update.


## Guided-unlearning audit of the completed experiment (19 November 2026)

The completed five-fold report is `results/tables/phase1_0e7dbe83c914e5f8/`. The audit found no algorithmic error in the implemented guided objective or saved-run execution. All 130 unique unlearning runs (65 guided, 65 uniform) were checked against their saved source snapshots and parents. The source fingerprint remains `07b6f070a9899567386dd122a0bdcbbc8b06ac94e8c5ace11692ddced0acdc25`; no production source, notebook, model checkpoint, or experimental result was changed by the audit.

An independent NumPy state-amplitude implementation of the paper's full six-qubit circuit reproduced the saved original-model anchor/calibration distributions, Eq. 6 targets, Eq. 7 objective at initialization and the selected/final checkpoint, and held-out predictions. Maximum saved-prediction discrepancy was below `5e-16`; objective discrepancy was below `9e-16`. All methods used fixed original teachers, training-only records, identical guided/uniform starting weights/references, and the declared 100 updates. Every selected checkpoint in this report was update 100. This verifies execution of the fixed budget, not convergence to a global optimum.

The full 72-component unlearning gradient was independently checked by finite differences of a separate NumPy objective. Maximum absolute error at alpha=1/lambda=0.01 was `4.88e-11`. The complete synthetic suite now passes **41 tests**. Audit evidence: [completed-run checks](results/audits/guided_unlearning_0e7dbe83c914e5f8.json) and [test results](results/audits/guided_unlearning_tests.json).

Report the implementation accurately: Eqs. 6-8 are implemented as a fixed mean-probability target and minimization of the negative log-likelihood objective, with alpha=1, lambda=0.01, beta=1. The gradient uses parameter shift on expectation values followed by the softmax/cross-entropy chain rule. This is a deliberate mathematical clarification of Eq. 5, whose literal shift of the entire nonlinear loss is not generally its derivative; the authors' private source cannot be checked. MNIST/folds, 32 calibration records, 25% stratified anchors, 100 unlearning updates, learning-rate settings and training-objective checkpoint selection are explicit implementation choices where the paper does not fully specify the main experiment. Describe the work as an independent implementation/adaptation of the published objective, not an exact numerical reproduction of the private code.

To repeat only the saved-run audit (no fitting):

```powershell
python -B tests/audit_guided_runs.py --provenance results/tables/phase1_0e7dbe83c914e5f8/provenance.json --output results/audits/guided_unlearning_0e7dbe83c914e5f8.json
```

Verification completed on 20 November 2026: the authorized Windows review executed all 105 nonempty code cells in 42.014 seconds, with 32 inline images and no notebook errors. All primary tables matched the original Linux experiment, all 65 conditions/230 unique stages were present, and SHA-256 comparison confirmed all 4,917 protected data/run/condition/maintenance/original-report files unchanged. All 48 synthetic software tests passed. Review exports are in `results/tables/phase1_review_fce16cfefb8f0456/` and the matching figures directory. Execution and test evidence is in `results/reviews/phase1_0e7dbe83c914e5f8/`. No research-model training, optimizer continuation, or new quantum predictions were performed.
