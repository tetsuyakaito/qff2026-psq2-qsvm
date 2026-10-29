# Qiskit Fall Fest 2026 — Quantum Machine Learning

## PS-Q2: Quantum Kernel Classification

This repository contains the Phase 2 PS-Q2 submission and its two common benchmark tasks: Bell-state fidelity (Task A) and Processor D design/QFT-3 benchmarking (Task B). Phase 1 materials are retained as project history; use **solution.ipynb** for the Phase 2 run.

## Results at a glance

| Experiment | Result |
|---|---:|
| Task A Bell fidelity, ideal | 1.0000 |
| Task A Bell fidelity, QFF2026-v2 noisy | **0.8801 ± 0.0026** |
| Processor D QFT-3 SWAP overhead | **0** |
| Processor D QFT-3 noisy fidelity | 0.668377 |
| PS-Q2 ideal quantum-kernel CV balanced accuracy | 0.5148 |
| PS-Q2 noisy quantum-kernel CV balanced accuracy | 0.5099 |
| PS-Q2 classical RBF-SVM CV balanced accuracy | **0.6078** |

Task A and Task B values are simulations under the organizer's benchmark conditions, not hardware measurements. PS-Q2 scores are mean training-only cross-validation estimates, not test-set scores. The current experiment does not show a quantum-classification advantage over the classical RBF baseline; that result is reported directly.

## Project contents

| File or directory | Purpose |
|---|---|
| [solution.ipynb](./solution.ipynb) | Phase 2 entry point: Task A, Task B, PS-Q2 model, results, and predictions |
| [processor_d_config.json](./processor_d_config.json) | Phase 2 Processor D configuration and official QFT-3 benchmark results |
| [q2_solution.py](./q2_solution.py) | PS-Q2 preprocessing, ideal/noisy quantum kernels, classical baseline, CV, and test predictions |
| [results.md](./results.md) | Results, interpretation, resource costs, and limitations |
| [predictions_q2.csv](./predictions_q2.csv) | Predictions for the 200 unlabeled Q2 test samples |
| [results/phase2/](./results/phase2/) | Saved Bell, Processor D, and PS-Q2 outputs |
| **benchmark/**, **noise_model/**, **schemas/**, **tools/** | QFF2026 QML kit v3 benchmark, noise model, schemas, and validator |
| **data/q2/** | Official Q2 training and unlabeled test-feature files |
| **qml-requirements.txt** | Pinned QML-track environment |
| [pip_freeze.txt](./pip_freeze.txt) | Recorded environment package freeze |
| **main.ipynb**, **report.md**, **processors/** | Retained Phase 1 work; these are not the Phase 2 notebook/config |

Use **processor_d_config.json** for the Phase 2 benchmark. **processors/processor_D.json** is the earlier Phase 1 format and does not replace the Phase 2 configuration.

## PS-Q2: dataset and method

The QML-HIGHD-KER dataset has 800 labeled training rows, 200 unlabeled test rows, and 50 standardized features. The feature dictionary identifies x01–x06 as the signal-bearing features. The official kit provides no validation partition and no test labels.

To control the cost of a full quadratic kernel matrix, the experiment selects a stratified subset of 200 training rows (97 class A, 103 class B). The same subset and three stratified folds are used for quantum and classical model selection. Within each fold, standardization and PCA are fitted only on that fold's training rows. The quantum model projects the six signal features to four PCA components and encodes them with a four-qubit all-pairs ZZ feature map. The classical RBF-SVM retains all six standardized signal features; this feature-dimension difference is documented in the results.

The ideal kernel is the squared state overlap. The noisy kernel uses density matrices generated under the unmodified QFF2026-v2 gate-noise model and the normalized Hilbert–Schmidt overlap:

~~~~text
K_noisy(i,j) = Tr(rho_i rho_j) / sqrt(Tr(rho_i^2) Tr(rho_j^2))
~~~~

The noisy circuit is transpiled to the organizer's covered basis, and the code checks that all circuit gates are noise-covered. Test labels are not loaded or used. The final model writes one sample_id,label prediction for each test ID.

### PS-Q2 cross-validation results

| Model | Selected settings | Balanced accuracy |
|---|---|---:|
| Quantum kernel, ideal | C = 100; 4 PCA components | 0.5148 |
| Quantum kernel, QFF2026-v2 noisy | C = 100; 4 PCA components | 0.5099 |
| Classical RBF-SVM | C = 1; gamma = 0.3 | **0.6078** |

The noisy quantum kernel has a mean absolute training-kernel difference of 0.1191 from the ideal kernel and centered kernel alignment of 0.9895. Its off-diagonal standard deviation changes from 0.1220 (ideal) to 0.1088 (noisy). The reported balanced-accuracy values are model-selection estimates on training folds; test accuracy is unavailable because Q2 test labels are held by the organizers.

### PS-Q2 resource use

- 4 qubits; 4 PCA components.
- Compiled feature-map depth: 17.
- Two-qubit gates per feature-map circuit: 12.
- 2,000 state-preparation circuit simulations and 281,000 unique symmetric kernel entries across model selection and final inference.
- The recorded run took about 98 seconds; runtime depends on the machine.

## Processor D: topology and benchmark

Processor D has six qubits. Qubits 0–3 form a complete, bidirectional K₄ computational core; qubits 4 and 5 are connected ancillae. The K₄ gives every pair in the four-qubit all-pairs feature map a direct link. Native gates are rx, ry, rz, and cx. The coupling map lists both directions for every link.

The JSON's noise assumptions describe the design. The Task B benchmark itself always uses the organizer's QFF2026-v2 noise model: 1% single-qubit depolarizing error, 5% two-qubit depolarizing error, and 2% readout error. Task B has no measurement, so readout error does not enter its fidelity calculation.

| Official QFT-3 measurement | Result |
|---|---:|
| Depth before transpilation | 6 |
| Depth after transpilation | 18 |
| Two-qubit gate count | 9 |
| Routing SWAP overhead | **0** |
| Ideal fidelity | 1.000000 |
| Noisy fidelity | **0.668377** |

Zero routing SWAP overhead demonstrates that this topology embeds the benchmark circuit without extra routing SWAPs. The noisy fidelity shows that the benchmark still accumulates gate noise; this result does not establish classifier accuracy or prove that the topology is optimal.

## Task A: Bell-state fidelity

Task A prepares |Phi+> = (|00> + |11>) / sqrt(2) using a Hadamard and CNOT, then measures XX, YY, and ZZ in separate circuits. Each basis uses exactly 8,192 shots. The official task_a_bell() implementation imports the organizer noise model and transpiles to its fully-noised basis.

For each Pauli product P, the correlation is computed from measurement parity:

~~~~text
<P> = (N_even - N_odd) / N
~~~~

The required Bell fidelity is:

~~~~text
F = (1 + <XX> - <YY> + <ZZ>) / 4
~~~~

| Condition | &lt;XX&gt; ± SE | &lt;YY&gt; ± SE | &lt;ZZ&gt; ± SE | Fidelity ± SE |
|---|---:|---:|---:|---:|
| Ideal | 1.0000 ± 0.0000 | −1.0000 ± 0.0000 | 1.0000 ± 0.0000 | **1.0000 ± 0.0000** |
| QFF2026-v2 noisy | 0.8191 ± 0.0063 | −0.8279 ± 0.0062 | 0.8735 ± 0.0054 | **0.8801 ± 0.0026** |

The noisy correlations are reduced by gate and readout errors. The values are estimates from the organizer's fixed simulator model and shot count, not experimental hardware measurements. P(00) + P(11) is a population measure and is not used as the fidelity.

## Reproduce the Phase 2 submission

Use Python 3.11, 3.12, or 3.13 and install the dedicated QML-track pins from **qml-requirements.txt**. Do not use the Phase 1 **requirements.txt** for this workflow.

### Windows PowerShell

~~~~powershell
py -3.12 -m venv qml-env
.\qml-env\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r qml-requirements.txt
~~~~

### Linux or WSL

~~~~bash
python3 -m venv qml-env
source qml-env/bin/activate
python -m pip install --upgrade pip
python -m pip install -r qml-requirements.txt
~~~~

### Run the checks and notebook

~~~~bash
python noise_model/benchmark_noise_model.py
python benchmark/qff_qml_benchmark.py processor_d_config.json
python tools/validate_predictions.py q2 predictions_q2.csv
python -m nbconvert --to notebook --execute solution.ipynb --inplace
pip freeze > pip_freeze.txt
~~~~

The expected Processor D output is saved in **results/phase2/processor_d_benchmark.txt**; the Bell output is in **results/phase2/task_a_results.json**; PS-Q2 metrics are in **results/phase2/q2_metrics.json**. The official prediction validator checks the format and IDs in **predictions_q2.csv**.

## Reproducibility and limitations

- Random seed and transpiler seed: 42.
- The PS-Q2 balanced-accuracy results are cross-validation estimates, not test-set measurements.
- Only 200 of the 800 labeled training rows are used for quantum model selection and final fitting; conclusions may vary with the subset and folds.
- The classical baseline keeps six input features while the quantum model reduces them to four components.
- The noisy Q2 density-matrix kernel is a normalized Hilbert–Schmidt overlap, not a shot-based compute-uncompute kernel; readout noise therefore does not enter that classifier calculation.
- The current results do not show a quantum classification advantage. A successful Processor D QFT-3 benchmark is a topology result, not evidence of classifier superiority.
- Timing is hardware-dependent.

The complete recorded environment freeze from the executed Python 3.12.15 environment follows.

~~~~text
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
colorama==0.4.6
comm==0.2.3
contourpy==1.4.0
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
ibm-cloud-sdk-core==3.28.0
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
jupyterlab==4.6.4
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
networkx==3.7
notebook==7.6.3
notebook_shim==0.2.4
numpy==2.4.6
packaging==26.3
pandas==3.1.0
pandocfilters==1.5.1
parso==0.8.7
pennylane==0.44.1
PennyLane-qiskit==0.44.1
pennylane_lightning==0.45.0
pillow==13.0.0
platformdirs==4.14.0
prometheus_client==0.26.0
prompt_toolkit==3.0.53
psutil==7.2.2
pure_eval==0.2.4
pycparser==3.12
pydantic==2.14.1
pydantic_core==2.50.1
Pygments==2.21.0
PyJWT==2.15.1
pyparsing==3.3.3
pyspnego==0.12.4
python-dateutil==2.9.0.post0
python-json-logger==4.2.0
pywinpty==3.1.2
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
sspilib==0.6.0
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
wheel==0.48.0
~~~~
