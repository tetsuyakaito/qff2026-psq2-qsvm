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

### Final results at a glance

| Model / benchmark | Result |
|---|---:|
| Ideal quantum-kernel CV balanced accuracy | **0.7290** |
| QFF2026-v2 noisy quantum-kernel CV balanced accuracy | **0.7266** |
| Classical RBF-SVM CV balanced accuracy | **0.7274** |
| Task A noisy Bell fidelity | **0.8801 ± 0.0026** |
| Task B noisy QFT-3 fidelity | **0.668377** |
| Task B routing SWAP overhead | **0** |

The three classification values are five-fold training-only estimates, not
hidden-test scores. The noisy quantum result is statistically indistinguishable
from the classical result under the reported descriptive fold intervals; no
quantum advantage is claimed.

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

The root [`requirements.txt`](./requirements.txt) and
[`qml-requirements.txt`](./qml-requirements.txt) are identical copies of the
official kit-v3 pins. The notebook fails immediately if any direct package
version differs. A successful run regenerates:

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

Balanced accuracy is the official PS-Q2 metric. The final run produced:

| Model | Selected settings | Fold scores | Mean | Descriptive 95% *t* interval |
|---|---|---|---:|---:|
| Quantum, ideal | `C=1` | 0.7072, 0.7061, 0.7631, 0.7186, 0.7501 | **0.7290** | 0.6966–0.7614 |
| Quantum, QFF2026-v2 noisy | `C=1` | 0.6943, 0.7061, 0.7631, 0.7256, 0.7441 | **0.7266** | 0.6921–0.7612 |
| Classical RBF-SVM | `C=1`, `gamma=0.01` | 0.7194, 0.7561, 0.6926, 0.7450, 0.7237 | **0.7274** | 0.6968–0.7579 |

These intervals describe variation across the five fixed training folds; they
are not test-set confidence intervals and do not replace evaluation by the
organizers. The authoritative full-precision values are in
[`results/phase2/q2_metrics.json`](./results/phase2/q2_metrics.json) and are
displayed dynamically near the end of `solution.ipynb`, avoiding copied values
drifting out of sync.

The compiled feature map has depth **9**, **12** two-qubit gates, and **zero
routing-added two-qubit gates** on the declared Processor D layout. The complete
run used 9,000 state preparations and computed 3,555,600 unique pairwise kernel
entries. Those overlaps are classical post-processing of reusable simulated
states, not 3,555,600 separate circuit executions.

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
| **Phase 2 environment** | `requirements.txt` and `qml-requirements.txt` (identical official pins), plus `pip_freeze.txt` (recorded full environment) |
| **Phase 1 archive only** | `main.ipynb`, `report.md`, `run_all.py`, `generate_notebook.py`, `processors/`, `src/`, `assets/`, `data/raw/`, `data/processed/`, `results/figures/`, and `results/tables/` |

Do not use Phase 1 `run_all.py`, `main.ipynb`, or
`processors/processor_D.json` to reproduce Phase 2. In particular,
`processor_d_config.json` is the only Phase 2 Processor D configuration.

## Results and limitations

See [`results.md`](./results.md) for the submission-facing methodology,
benchmark interpretation, resource-accounting rules, and limitations. The
generated `results/phase2/q2_metrics.json` remains the full-precision source of
truth for PS-Q2 performance and resources.

## Recorded `pip freeze` output

Generated from the clean Python 3.11 environment used for the final pinned
notebook execution. The same output is stored in [`pip_freeze.txt`](./pip_freeze.txt).

<details>
<summary>Show complete environment</summary>

```text
annotated-types==0.8.0
anyio==4.15.1
appdirs==1.4.4
argon2-cffi==25.1.0
argon2-cffi-bindings==26.1.0
arrow==1.4.0
asttokens==3.0.2
astunparse==1.6.3
async-lru==2.4.0
attrs==26.1.0
autograd==1.9.1
autoray==0.8.2
babel==2.18.0
beautifulsoup4==4.15.0
bleach==6.4.0
cachetools==7.2.7
certifi==2026.10.23
cffi==2.1.1
charset-normalizer==3.5.2
cloudpickle==3.1.2
comm==0.2.3
contourpy==1.3.3
cryptography==50.0.2
cycler==0.12.1
debugpy==1.8.22
defusedxml==0.7.1
diastatic-malt==2.15.2
dill==0.4.1
executing==2.3.0
fastjsonschema==2.22.2
fonttools==4.68.0
fqdn==1.6.0
gast==0.7.0
h11==0.16.0
httpcore==1.0.9
httpx==0.28.1
ibm-cloud-sdk-core==3.28.1
ibm-platform-services==0.77.1
idna==3.20
ipykernel==7.4.0
ipython==9.17.1
ipython_pygments_lexers==1.1.1
isoduration==20.11.0
jedi==0.20.1
Jinja2==3.1.6
joblib==1.6.0
json5==0.17.3
jsonpointer==3.6
jsonschema==4.26.0
jsonschema-specifications==2025.9.1
jupyter-events==0.12.1
jupyter-lsp==2.3.1
jupyter_builder==1.2.4
jupyter_client==8.10.0
jupyter_core==5.9.1
jupyter_server==2.21.1
jupyter_server_terminals==0.5.4
jupyterlab==4.6.5
jupyterlab_pygments==0.3.0
jupyterlab_server==2.28.1
kiwisolver==1.5.1
lark==1.3.1
MarkupSafe==3.0.4
matplotlib==3.11.2
matplotlib-inline==0.2.2
mistune==3.4.0
mpmath==1.3.0
narwhals==2.28.0
nbclient==0.11.0
nbconvert==7.17.2
nbformat==5.11.1
nest-asyncio2==1.7.4
networkx==3.6.1
notebook==7.6.3
notebook_shim==0.2.4
numpy==2.4.6
overrides==7.7.0
packaging==26.3
pandas==3.1.0
pandocfilters==1.5.1
parso==0.8.7
pennylane==0.44.1
PennyLane-qiskit==0.44.1
pennylane_lightning==0.45.0
pexpect==4.9.0
pillow==13.0.0
platformdirs==4.14.0
prometheus_client==0.26.0
prompt_toolkit==3.0.53
psutil==7.2.2
ptyprocess==0.7.0
pure_eval==0.2.4
pycparser==3.12
pydantic==2.14.2
pydantic_core==2.50.2
Pygments==2.21.0
PyJWT==2.15.1
pyparsing==3.3.3
pyspnego==0.12.4
python-dateutil==3.0.0
python-json-logger==4.2.0
PyYAML==6.0.3
pyzmq==27.2.0
qiskit==2.3.0
qiskit-aer==0.17.2
qiskit-ibm-runtime==0.45.1
referencing==0.37.0
requests==2.34.2
requests_ntlm==1.3.0
rfc3339-validator==0.1.4
rfc3986-validator==0.1.1
rfc3987-syntax==1.1.0
rpds-py==2026.9.1
rustworkx==0.18.1
scikit-learn==1.9.1
scipy==1.17.1
scipy-openblas32==0.3.34.237.2
Send2Trash==2.1.0
six==1.17.0
soupsieve==3.0.4
stack-data==0.6.3
stevedore==5.9.1
sympy==1.14.0
termcolor==3.3.0
terminado==0.18.1
threadpoolctl==3.7.0
tinycss2==1.5.1
tomlkit==0.15.2
tornado==6.6
traitlets==5.16.1
typing-inspection==0.4.4
typing_extensions==4.16.0
tzdata==2026.6
uri-template==1.3.0
urllib3==2.8.0
wcwidth==0.9.4
webcolors==25.10.0
webencodings==0.6.1
websocket-client==1.9.2
```

</details>
