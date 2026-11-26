# Distribution-guided quantum unlearning under data poisoning

Phase 1 evaluates Malik et al.'s existing class-level unlearning method under controlled MNIST poisoning. Forget digit 2 and retain digits 0 and 1. This is an evaluation of an existing algorithm; Phase 2 method extensions are deferred.

`main_project.ipynb` is the single entry point for execution, figures, tables, and interpretation. It uses short cells and numbered research steps. Reusable implementation lives under `src/`.

## Project layout

```text
main_project.ipynb             Run and analyze the research
README.md                     Setup, protocol, and execution guidance
requirements.txt              Python dependencies
src/
  project_workflow.py         Local execution and portable review helpers
  fresh_study.py              Current normalized-input protocol and orchestration
  experiment_stages.py        Shared supervised/unlearning stage implementation
  config.py                  Dataclasses and DEFAULT_SEED = 6304
  data/                      Raw data, splits, preprocessing, poisoning
  models/                    Circuit, Aer, classical models, batching, noise
  unlearning/                Fixed teachers, targets, objective
  training.py                Adam and full optimizer continuation
  checkpoints.py             Cache identity and per-model reuse/resume
  checkpoint_storage.py      Atomic storage and optional runtime mirrors
  artifacts.py               Hashes, manifests, source snapshots, numeric artifacts
  ablations.py               Guided one-factor parameter experiments
  noise_study.py             Fixed-weight synthetic noise inference
  noisy_training.py          Separate training and unlearning under noise
  evaluation/                Metrics, checked loading, summaries, plots, exports
tools/noisy_training_runtime.py  Progress, heartbeat, compatible execution recovery
tests/                       Synthetic software checks, not research results
data/raw/                    Original verified MNIST archive
data/splits/                 Exact memberships, created on demand
data/annotated/              Poisoned arrays, labels, IDs, metadata, created on demand
results/
  runs/                      Trained weights, checkpoints, predictions and run metadata
  fresh_studies/             Primary study indexes, conditions and analysis
  ablations/                 Guided parameter ablations
  noise_studies/             Fixed-weight synthetic noise evaluation
  noisy_training/            Training and unlearning under synthetic noise
  reviews/                   Rebuilt analysis tables, figures and review provenance
Relevant_Papers/              Local reading material
report/                      Report material
```

## Setup

Use Python 3.12, create an environment, and install `requirements.txt`. Select its kernel and open the notebook from this project root. On this Windows machine, the existing environment is `C:/Users/DanishZaheer/anaconda3/envs/QML/python.exe`. No automatic package installation occurs in the notebook.

The first environment cells report the physical GPU and Aer devices separately. CPU-only Aer is supported. A compatible GPU-enabled Aer build can accelerate execution on supported systems; its recorded version and device remain part of training provenance. Hardware access is unnecessary. IBM execution is disabled by default.

## Review saved results

The default is `SAVED_RESULTS_ONLY=True`, with all fitting, download, force, noise-prediction, and hardware flags False. Restore saved artifacts under their original relative `results/` and `data/` paths. Do not copy old notebook outputs or replace the current source when importing an archive.

`STUDY_INDEX=None` automatically selects exactly one completed `results/fresh_studies/<id>/study.json` with label `feature_randomization_v1`. Set an explicit relative path if multiple studies exist. Optional ablation, fixed-weight noise, and noisy-training indexes link to that exact study; path overrides are in the controls cell. A partially completed noisy study can be reviewed with missing coverage shown explicitly.

Restart Kernel and Run All loads saved numeric metrics, predictions, histories, and features. It verifies hashes, folds, labels, and parent/reference links. It does not unpickle trained estimators, refit PCA, simulate predictions, resume optimizers, or train on a cache miss. Missing or incompatible required artifacts stop review. Original source/environment/device metadata is preserved; the current review is recorded separately under `results/reviews/<id>/`.

The notebook selects these completed experiments, each covering five folds:

| Experiment | Completed coverage | Study index |
|---|---|---|
| Primary poisoning and unlearning | 265 VQC fits, including ablation support | [Primary study](results/fresh_studies/fa1604cdd38207d9/study.json) |
| Guided parameter ablations | 300 nondefault fits and five compatibility fits | [Ablation study](results/ablations/e4a96e40f818e487/study.json) |
| Fixed-weight noise evaluation | 9,660 prediction files | [Noise evaluation](results/noise_studies/11aa94ebc456b91d/study.json) |
| Training and unlearning under noise | 100 supervised and 130 unlearning fits | [Noisy training](results/noisy_training/9b54f30ef300facf/study.json) |

Restart Kernel and Run All reviews all four studies with fitting disabled. Poison previews load the exact saved arrays: regenerating normalized random vectors on another platform can differ in the last floating-point bits. Per-run manifests retain artifact checksums, settings, source snapshots, training environments and parent/reference links. Analysis exports stay under `results/reviews/<id>/`.

The completed noisy study includes 65 frozen teachers and 460 matched deployment-prediction files. Its recorded training backend was PyTorch exact density-matrix evolution/autodiff on a Tesla T4, numerically validated against Aer. The recorded training-cell duration was approximately 2 hours 3 minutes.

Saved review and the active execution code require no PyTorch installation. Future noisy-model fitting uses Qiskit Aer, exact density matrices and expectation-level parameter shifts. Forced Aer fits create separate model/study identities. Original implementation files are preserved with the saved study, including `execution_source.zip`.

## Execute a new local study

Restart the kernel, set `SAVED_RESULTS_ONLY=False`, and enable `RUN_PRIMARY_STUDY=True`. Enable `DOWNLOAD_MNIST=True` only when the verified raw archive is missing. Keep force flags False for normal continuation. Clean gold originals are prepared first; the full pipeline reuses them and completes remaining originals, retained-data references, and default guided/uniform fits.

Enable extensions deliberately with `RUN_ABLATIONS`, `RUN_FIXED_WEIGHT_NOISE`, and `RUN_NOISY_TRAINING`. They use completed primary inputs. Each stage saves its own index and artifacts. Analysis flags can be disabled to skip expensive saved-prediction reads while running another stage. A full recursive checkpoint inventory is optional and disabled by default.

For a new Aer noisy experiment, restart the kernel and set `SAVED_RESULTS_ONLY=False`, `RUN_NOISY_TRAINING=True` and `FORCE_NOISY_TRAINING=True`. `FORCE_RETRAIN=True` also forces noisy fitting when its run flag is enabled. With force flags False, the selected completed noisy study is reviewed without retraining. After creating a new Aer attempt, select its printed relative `NOISY_TRAINING_INDEX` and turn force flags False to resume an interruption with its matching source/environment/execution. An imported partial optimizer from another backend cannot resume as Aer.

Completed compatible models are reused. Interrupted quantum fits continue from their last committed Adam/RNG checkpoint, not the last evaluation checkpoint. Each complete update is saved atomically. Changes in data, settings, scientific source, runtime, or execution prevent incompatible continuation. Completed results remain portable for review across environments. An imported unfinished optimizer needs its matching source/environment; it is never silently converted to a new experiment. Force flags deliberately create new attempts and retain distinct completed artifacts.

Independent folds/models run through one shared Aer service; each optimizer has its own state. `DEVICE="auto"` checks actual Aer support. Conservative defaults use up to two CPU workers or eight GPU workers; `PARALLEL_WORKERS` is explicit. VRAM limits prevent over-allocation and do not promise high utilization. Noisy training reports model starts, steps, 30-second heartbeats, Aer jobs, and possible stalls. Runtime logs are stored with its study.

Direct project-path checkpoint saves are the default. Optionally set `QML_RUNTIME_CHECKPOINT_DIR` to a fast scratch directory to mirror updates with verified periodic backups (5 seconds) and a final flush. Losing scratch storage may lose progress since the latest backup.

## Registered protocol

- Seed **6304**; 420 MNIST images, 140 each of 0/1/2; five stratified outer folds and an inner validation holdout: 268 train, 68 validation, 84 test per fold.
- Flatten, divide by 255, and normalize all inputs to unit L2 length (zero remains zero). Fit four PCA components and clipped `[0, pi]` angle scaling on each model's training records only.
- Feature randomization replaces selected digit-2 vectors with independent real `N(0, 0.1²)` vectors normalized to unit length; labels stay unchanged. Signed entries are preserved. Label flipping changes selected digit-0 labels to 2.
- Primary rates: 5%, 10%, 20%, 30%, 40%, 50% of the eligible class, rounded down. Additional 25% conditions support ablations. Validation/test remain clean.
- Malik-style six-qubit circuit: two reuploads, three ansatz repeats, 72 nominal parameters. Noiseless statevector exact expectations; 300 supervised updates, batch size 10; 100 unlearning updates.
- The **gold reference** is clean three-class training. **Retained-data retraining** removes exact observed-label-2 records, including flipped zeros, and refits its own preprocessing. Unlearning freezes its parent's features.
- Guided defaults: alpha=1, lambda=0.01, beta=1, anchor fraction=0.25. The global forget target and original soft-anchor predictions are frozen before fitting. Uniform uses `(0.5, 0.5, 0)`. Gradients shift expectations before the classical chain rule.
- OFAT: alpha `[0, .5, 1, 2, 4]`; lambda `[0, .001, .01, .1]`; beta `[.25, .5, .75, 1, 2, 4]`. Reuse the default: 12 nondefault settings × five conditions (clean and both attacks at 25%/50%) × five folds = **300 fits**. Five optional first-fold compatibility checks are separate.
- Primary plus support pipeline: **265 distinct VQC fits**. Separate combined gate/readout p=0.01 noisy optimization: **230 fits** across all primary conditions/folds, comprising 100 supervised and 130 unlearning fits.
- Fixed-weight noise uses zero plus five channel families at four nonzero levels, exact density probabilities, and separate 2048-shot readouts. Noisy optimization instead uses exact noisy expectations and new frozen noisy teachers.

Report retained accuracy, digit-wise recall, forgotten confidence, full and retained-conditional KL, targets, paired poisoning effects, resources, and coverage. Fold SD is descriptive; reused clean references are not independent seeds. Noise inference and noisy training answer different questions. Phase 1 does not claim certified deletion or require quantum superiority.

## Software checks

Run `python -m unittest discover -s tests -v` in the project environment. Tests use tiny synthetic data/circuits and temporary directories, checking gradients, frozen teachers, membership, reuse, strict continuation, portable review, and noise handling. They do not execute the research notebook or produce experimental findings.
