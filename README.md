# Qiskit Fall Fest 2026 — QML Track v3

## Phase 2 submission: PS-Q2, Task A, and Task B

This repository's Phase 2 submission addresses **PS-Q2 — high-dimensional
quantum-kernel classification** and the two required common tasks:

- **Task A:** official Bell-state fidelity benchmark;
- **Task B:** official Processor D QFT-3 benchmark; and
- **PS-Q2:** ideal/noisy quantum-kernel models, an all-feature classical
  RBF-SVM baseline, and predictions for the unlabeled test set.

The Phase 2 entry point is [`solution.ipynb`](./solution.ipynb). It checks the
pinned QML environment, regenerates all three Phase 2 result groups, validates
the Processor D numbers against the JSON rather than trusting the benchmark's
exit code, and runs the kit's official prediction validator.

## Reproduce from a clean clone

Use Python 3.11, 3.12, or 3.13. From the repository root, the following is the
complete **one-command** Linux/WSL reproduction:

```bash
python3 -m venv .venv && . .venv/bin/activate && python -m pip install -r qml-requirements.txt && python -m nbconvert --to notebook --execute solution.ipynb --inplace --ExecutePreprocessor.timeout=-1
```

PowerShell 7 equivalent:

```powershell
py -3.12 -m venv .venv && .\.venv\Scripts\Activate.ps1 && python -m pip install -r qml-requirements.txt && python -m nbconvert --to notebook --execute solution.ipynb --inplace --ExecutePreprocessor.timeout=-1
```

For an already prepared pinned environment, reproduction is one command:

```bash
python -m nbconvert --to notebook --execute solution.ipynb --inplace --ExecutePreprocessor.timeout=-1
```

The notebook fails immediately if any package version differs from
[`qml-requirements.txt`](./qml-requirements.txt). A successful run regenerates:

- `results/phase2/task_a_results.json`;
- `results/phase2/processor_d_benchmark.txt`;
- `results/phase2/q2_metrics.json`; and
- `predictions_q2.csv`.

It then invokes
`python tools/validate_predictions.py q2 predictions_q2.csv`; any validator
error fails the notebook. There is no alternate or substitute validation path.

## PS-Q2 method

### Data and leakage control

The official QML-HIGHD-KER files contain 800 labeled training rows, 200
unlabeled test rows, and 50 standardized numeric features (`x01`–`x50`). The
experiment uses **all 800 training rows and all 50 features**. No test labels
are available or used.

Model selection uses five fixed stratified folds. For every fold, all learned
preprocessing is fitted on that fold's training partition only:

- the quantum pipeline standardizes all 50 features and fits a six-component
  PCA, then applies those fitted transformations to the validation partition;
- the classical RBF-SVM baseline standardizes and retains all 50 features; and
- both models use the same rows and fold assignments.

After selection, preprocessing and the selected model are refitted on all 800
labeled rows before producing the 200 test predictions.

### Six-qubit weak-ZZ ring kernel

The quantum feature map encodes the six fold-fitted PCA coordinates on six
qubits. It is a shallow, fixed (non-variational) weak-ZZ ring with
`theta_scale = 1.2` and interaction strength `beta = 0.1`. The ideal kernel is
the squared state overlap. The noisy kernel uses density matrices generated
under the unmodified **QFF2026-v2** model and the normalized Hilbert–Schmidt
overlap:

```text
K_noisy(i,j) = Tr(rho_i rho_j) / sqrt(Tr(rho_i^2) Tr(rho_j^2))
```

The circuit has **0 trainable quantum parameters**. Its six PCA coordinates
are per-sample data inputs (six data-encoding parameters), not optimized
variational weights. The SVM still fits its classical dual coefficients.

### Cross-validation reporting

Balanced accuracy is the official PS-Q2 metric. For the ideal quantum kernel,
noisy quantum kernel, and RBF baseline, the report must show all five held-out
fold scores, their mean, and a 95% Student-*t* interval computed from those five
scores. This interval describes variation across the fixed training folds; it
is not a test-set confidence interval and does not replace evaluation by the
organizers. The authoritative generated values are in
[`results/phase2/q2_metrics.json`](./results/phase2/q2_metrics.json) and are
displayed dynamically near the end of `solution.ipynb`, avoiding copied values
drifting out of sync.

## Processor D and the physical cycle

Processor D has six qubits and includes the physical cycle

```text
4 — 0 — 2 — 5 — 1 — 3 — 4
```

The weak-ZZ ring follows this cycle, so each intended nearest-neighbor
interaction is a physical link. The sparse ring uses six interactions per
layer rather than the 15 interactions of a six-qubit all-to-all layer. This
reduces routing pressure, two-qubit-gate depth, and accumulated gate noise, at
the cost of less direct long-range expressivity.

The coupling map is **directed as written** by the kit. Every physical link in
`processor_d_config.json` is therefore listed in both directions. That allows
either CX orientation without direction-reversal synthesis, which would add
single-qubit gates, depth, and noise. The extra K4 chords on qubits 0–3 retain
useful QFT-3 placement and routing options, while the PS-Q2 circuit deliberately
uses the topology-aware six-qubit cycle.

The saved official Task B measurement remains:

| Official QFT-3 measurement | Value |
|---|---:|
| Circuit depth before transpilation | 6 |
| Circuit depth after transpilation | 18 |
| Two-qubit gate count | 9 |
| Routing SWAP overhead | 0 |
| Ideal fidelity | 1.000000 |
| Noisy fidelity, QFF2026-v2 | 0.668377 |

Zero routing SWAP overhead is a topology result for this benchmark. It does
not remove native-gate noise and is not evidence of classifier superiority.

## Task A — Bell-state fidelity

The official `task_a_bell()` function uses 8,192 shots for each of the XX, YY,
and ZZ measurement circuits. It evaluates

```text
F = (1 + <XX> - <YY> + <ZZ>) / 4
```

with the organizer's fixed simulator settings:

| Condition | `<XX> ± SE` | `<YY> ± SE` | `<ZZ> ± SE` | `F ± SE` |
|---|---:|---:|---:|---:|
| Ideal | 1.0000 ± 0.0000 | -1.0000 ± 0.0000 | 1.0000 ± 0.0000 | **1.0000 ± 0.0000** |
| QFF2026-v2 noisy | 0.8191 ± 0.0063 | -0.8279 ± 0.0062 | 0.8735 ± 0.0054 | **0.8801 ± 0.0026** |

These are seeded simulator estimates under the organizer's benchmark model,
not hardware measurements.

## Submission files versus Phase 1 archive

| Scope | Files |
|---|---|
| **Phase 2 authoritative** | `solution.ipynb`, `q2_solution.py`, `processor_d_config.json`, `results.md`, `predictions_q2.csv`, and `results/phase2/` |
| **Official kit v3 (unchanged)** | `benchmark/`, `noise_model/`, `schemas/`, `tools/`, and `data/q2/` |
| **Phase 2 environment** | `qml-requirements.txt` (pins) and `pip_freeze.txt` (recorded full environment) |
| **Phase 1 archive only** | `main.ipynb`, `report.md`, `run_all.py`, `generate_notebook.py`, `requirements.txt`, `processors/`, `src/`, `assets/`, `data/raw/`, `data/processed/`, `results/figures/`, and `results/tables/` |

Do not use Phase 1 `requirements.txt`, `run_all.py`, `main.ipynb`, or
`processors/processor_D.json` to reproduce Phase 2. In particular,
`processor_d_config.json` is the only Phase 2 Processor D configuration.

## Results and limitations

See [`results.md`](./results.md) for the submission-facing methodology,
benchmark interpretation, resource-accounting rules, and limitations. Final
PS-Q2 performance and resource numbers are read from the generated
`results/phase2/q2_metrics.json`; they are intentionally not duplicated here
before the final model run.
