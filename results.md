# QFF2026 Phase 2 — PS-Q2 Results

**Selected problem:** PS-Q2 — High-dimensional quantum-kernel classification
**Kit:** QFF2026 QML Track v3
**Noise model:** QFF2026-v2
**Random seed:** 42
**Environment:** Python 3.12.15; pinned packages are listed in `qml-requirements.txt`, with the installed environment recorded in `pip_freeze.txt`.

## 1. Approach

The official feature dictionary identifies `x01`–`x06` as the signal-bearing columns, so both models use those six features and the same stratified 200-row training subset. The quantum model fits a standardizer and four-component PCA on each training fold, then evaluates an all-pairs ZZ feature map on four qubits, matching the K4 core of Processor D. The classical baseline uses the same rows and folds with the six standardized signal features and an RBF-SVM. Since the official Q2 kit has no validation split, hyperparameters are selected by three-fold stratified cross-validation on training data only.

The ideal quantum kernel is the squared pure-state overlap. For the noisy run, each encoded state is simulated as a density matrix with the official QFF2026-v2 gate-noise model; the kernel is the normalized Hilbert–Schmidt overlap, `Tr(rho_i rho_j) / sqrt(Tr(rho_i^2) Tr(rho_j^2))`. This kernel uses gate noise but no measurement, so readout error does not enter the kernel calculation. The official training and unlabeled test-feature files are used; no test labels are present or used.

## 2. PS-Q2 Results

Scores below are mean balanced accuracy across the three training-only folds. They are model-selection estimates, not test-set measurements.

| Model | Selected settings | Training CV balanced accuracy |
|---|---|---:|
| Quantum kernel, ideal | `C=100`; 4 PCA components, 4 qubits | 0.5148 |
| Quantum kernel, QFF2026-v2 noisy | `C=100`; same folds and settings | 0.5099 |
| Classical RBF-SVM | `C=1`, `gamma=0.3`; 6 standardized features | 0.6078 |

The noisy quantum kernel's mean absolute training-matrix difference from the ideal kernel is **0.1191**; their centered kernel alignment is **0.9895**. Off-diagonal kernel standard deviation changes from **0.1220** ideal to **0.1088** noisy. At the selected quantum setting, mean CV balanced accuracy changes from **0.5148** ideal to **0.5099** noisy. The kernel shift is visible, but this experiment does not show a quantum classification advantage; the classical baseline performed better on these training folds.

The quantum subset contains 200 of the 800 official training rows (97 A, 103 B). It uses the six documented signal features, standardized within each fold and reduced to four PCA components; the final scaler/PCA is refit on the 200-row subset. The final noisy-kernel model generates the 200 predictions in `predictions_q2.csv`, preserving `s0`–`s199` and the required `sample_id,label` columns. The file's labels are predictions only; no test balanced accuracy is reported because test labels are unavailable.

**Resource use:** four qubits; the transpiled feature-map circuit has depth **17** and **12 two-qubit gates**. Across model-selection folds and final test inference, the pipeline performs **2,000 state-preparation circuit simulations** and computes **281,000 unique symmetric kernel entries**. The measured pipeline runtime was **98.4 seconds** in the final notebook run on this environment. Timing is machine-dependent. The custom SciPy dual optimizer implements the RBF-SVM and precomputed-kernel SVM; the pinned scikit-learn package is installed but its native extension cannot load under this machine's Windows Application Control policy.

## 3. Task A — Bell-State Fidelity

The official `task_a_bell()` benchmark was run with 8,192 shots per measurement basis. The required fidelity is `F = (1 + <XX> - <YY> + <ZZ>) / 4`; each correlation is `(N_even - N_odd) / N`.

| Condition | `<XX> ± SE` | `<YY> ± SE` | `<ZZ> ± SE` | Fidelity `± SE` |
|---|---:|---:|---:|---:|
| Ideal | 1.0000 ± 0.0000 | -1.0000 ± 0.0000 | 1.0000 ± 0.0000 | **1.0000 ± 0.0000** |
| Noisy, QFF2026-v2 | 0.8191 ± 0.0063 | -0.8279 ± 0.0062 | 0.8735 ± 0.0054 | **0.8801 ± 0.0026** |

The noisy fidelity is lower than the ideal value because depolarizing gate noise and readout errors reduce the measured Bell-state correlations. It remains well above the uncorrelated-state reference, but the reduction is a measurable loss of entanglement fidelity. These estimates use the organizer's fixed noise model and shot count; they do not represent hardware measurements.

## 4. Task B — Processor D QFT-3 Benchmark

The configuration is a six-qubit custom topology: qubits 0–3 form a complete, bidirectional K4; qubits 4 and 5 are connected ancillae. Native gates are `rx`, `ry`, `rz`, and `cx`, each covered by the official noise model. The six K4 pairs support the four-qubit all-pairs feature map without routing SWAPs; the QFT-3 benchmark can also be placed within that core. All links are listed in both directions, so the benchmark does not need direction-reversal gates for the core interactions.

| Official QFT-3 measurement | Value |
|---|---:|
| Circuit depth before transpilation | 6 |
| Circuit depth after transpilation | 18 |
| Two-qubit gate count | 9 |
| SWAP overhead | 0 |
| Ideal fidelity | 1.000000 |
| Noisy fidelity, QFF2026-v2 | 0.668377 |

The benchmark output was produced by `python benchmark/qff_qml_benchmark.py processor_d_config.json` and is preserved in `results/phase2/processor_d_benchmark.txt`. The ideal fidelity is exactly 1.0, and the reported SWAP overhead is zero; the depth and two-qubit count still reflect compiling the QFT and its final reversal into the native basis. Noisy fidelity is substantially lower under the fixed 1% single-qubit and 5% two-qubit depolarizing rates. These results validate this particular QFT-3 configuration, not Processor D's optimality or the classifier's accuracy.

## 5. Limitations and Assessment

- The Q2 kit has no validation partition. Three-fold cross-validation is used for model selection, and reported balanced accuracies are not test scores.
- Quantum model selection and final fitting use 200 of 800 training rows to constrain the circuit-simulation cost; conclusions may vary with the subset and folds.
- The quantum model reduces six signal features to four PCA components, while the RBF baseline retains all six standardized features. Both use the same training rows and CV folds, but their feature dimensionalities differ because the quantum circuit uses four qubits.
- The noisy density-matrix kernel is a normalized Hilbert–Schmidt overlap, not a shot-based compute-uncompute kernel. Readout error is therefore not applied in this classifier experiment; the Task A benchmark does include its official measurement procedure.
- The results do not demonstrate improved balanced accuracy for the quantum model. Passing the QFT-3 benchmark does not imply the quantum-kernel classifier achieves higher balanced accuracy.
- The official prediction-validator command is blocked in this Windows environment because Windows Application Control prevents loading the pinned pandas native extension (`pandas._libs.tslibs.timezones`). A separate structural check confirmed the prediction file has exactly 200 unique IDs (`s0`–`s199`), the required columns, and only labels A/B; this is not a substitute for the official validator.

## 6. Reproduction

Run from the repository root in the dedicated environment:

```powershell
python benchmark\qff_qml_benchmark.py processor_d_config.json
python tools\validate_predictions.py q2 predictions_q2.csv
python -m nbconvert --to notebook --execute solution.ipynb --inplace
```

The official benchmark command and notebook completed successfully. The prediction validator must be rerun on a machine/environment where the pinned pandas extension is permitted by application control. Raw benchmark and Task A outputs are under `results/phase2/`; Q2 fold scores and resource measurements are in `results/phase2/q2_metrics.json`.
