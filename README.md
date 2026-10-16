# Qiskit Fall Fest 2026 — Track 4 (QML)
## PS-Q2: Quantum Kernel Estimator & Support Vector Machine (QSVM)

Benchmark and architectural evaluation of 4-qubit Quantum Support Vector Machines (QSVM) using parameterized `ZZFeatureMap` circuits evaluated under frozen QFF 2026 synthetic noise on **Processor A** (5 physical qubits, branching-tree topology) and **Processor B** (9 physical qubits, heavy-hex topology).

---

##  Repository Architecture

```text
├── .gitignore
├── README.md
├── requirements.txt
├── processors/
│   ├── processor_A.json   # 5-qubit branching tree + custom ring alternative
│   └── processor_B.json   # 9-qubit heavy-hex with optimal 4-qubit cycle [1,2,5,6]
├── src/
│   ├── __init__.py        # Public package API re-exports
│   ├── processors.py      # Processor configs, AerSimulator, frozen NoiseModel
│   ├── problem.py         # Primary (make_moons), secondary & quantum datasets, circuits
│   ├── routing.py         # Transpilation analysis, depth, CX, and SWAP overhead
│   ├── metrics.py         # Ideal & noisy density-matrix kernel estimation, PSD mitigation, SVM benchmarking
│   └── visualization.py   # Publication-grade figures (300 DPI)
├── data/
│   ├── raw/               # Raw unscaled/unexpanded numpy arrays
│   └── processed/         # Processed [0, π]^4 feature matrices and labels (npy & CSV)
└── results/
    ├── figures/           # Generated figures (fig1-fig5)
    └── tables/            # Architecture routing & classification metrics CSVs
```

---

##  Frozen Hardware & Noise Specifications (Phase 1)

- **Basis Gates**: `['rz', 'sx', 'x', 'cx']`
- **Synthetic Noise Model**:
  - 1-qubit depolarizing error: $p_{1q} = 0.001$ on `['rz', 'sx', 'x']`
  - 2-qubit depolarizing error: $p_{2q} = 0.005$ on `['cx']`
  - Symmetric readout error: $p_{ro} = 0.01$ ($[[0.99, 0.01], [0.01, 0.99]]$) on all physical qubits
- **Transpilation Standard**: `optimization_level = 1`, `seed_transpiler = 42`, `shots = 4096`.
- **Qiskit Standard**: Modern Qiskit 1.x / 2.x standard (`qiskit.transpile`, `qiskit_aer.AerSimulator`). No deprecated 0.x methods. Parameterized circuits are transpiled once outside pair loops.

---

##  Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run modules & self-tests
```bash
# Validate processors & noise models
python3 -m src.processors

# Generate and verify datasets & feature maps
python3 -m src.problem

# Execute routing analysis and compute SWAP overhead
python3 -m src.routing

# Compute ideal & noisy hardware kernels, PSD mitigation & SVM benchmarks
python3 -m src.metrics

# Render all 5 publication figures
python3 -m src.visualization
```

---

##  Benchmarking & Statistical Evaluation
The pipeline evaluates 7 models across 5 random seeds (`[42, 100, 2026, 7, 99]`):
1. **QSVM (Ideal)**
2. **QSVM (Processor A - Noisy)**
3. **QSVM (Processor B - Noisy)**
4. **QSVM (Processor A - Mitigated)**
5. **QSVM (Processor B - Mitigated)**
6. **Classical SVM (RBF - Tuned via GridSearchCV)**
7. **Classical SVM (Polynomial - Tuned via GridSearchCV)**

Metrics captured with Mean, Standard Deviation, and 95% Confidence Intervals:
- Test Accuracy, F1-Score, ROC-AUC
- Kernel-Target Alignment (KTA)
- Off-Diagonal Kernel Variance (Kernel Concentration indicator)
- Quantum Circuit Execution counts ($N(N-1)/2$)
