"""Reproducible full-training PS-Q2 quantum-kernel workflow.

Only labels from ``hd_data_train.csv`` are read.  The official Q2 test file is
treated strictly as an unlabeled feature matrix.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter, ParameterVector
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from noise_model.benchmark_noise_model import (
    DEFAULT_BASIS,
    SEED,
    VERSION,
    assert_all_gates_noisy,
    get_benchmark_noise_model,
)

ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "data" / "q2" / "hd_data_train.csv"
TEST_PATH = ROOT / "data" / "q2" / "hd_data_test_features.csv"
PREDICTIONS_PATH = ROOT / "predictions_q2.csv"
METRICS_PATH = ROOT / "results" / "phase2" / "q2_metrics.json"

N_QUBITS = 6
N_PCA_FEATURES = 6
N_INPUT_FEATURES = 50
CV_FOLDS = 5
ANGLE_SCALE = 1.2
ANGLE_DIVISOR = 2.0
ZZ_BETA = 0.1
SVM_C_VALUES = (0.1, 1.0, 10.0)
RBF_GAMMA_VALUES = (0.01, 0.02, 0.05)
LABELS = {"A": 0, "B": 1}

# This order executes disjoint ring edges together where possible.  The edge
# set is exactly the logical six-cycle 0-1-2-3-4-5-0.
RING_EDGES = ((0, 1), (2, 3), (4, 5), (1, 2), (3, 4), (5, 0))
SIMULATION_BATCH_SIZE = 100
T_975_DF4 = 2.7764451051977987

_COMPILED_FEATURE_MAP: QuantumCircuit | None = None
_FEATURE_PARAMETERS: tuple[Parameter, ...] | None = None
_NOISY_SIMULATOR: AerSimulator | None = None


def _angles(features: np.ndarray) -> np.ndarray:
    """Bound PCA components while retaining their sign and local geometry."""

    return ANGLE_SCALE * np.tanh(features / ANGLE_DIVISOR)


def _feature_map(angles: Sequence[object]) -> QuantumCircuit:
    """Create the fixed six-qubit H/RZ plus weak ring-ZZ feature map."""

    if len(angles) != N_QUBITS:
        raise ValueError(f"Expected {N_QUBITS} feature-map angles.")
    circuit = QuantumCircuit(N_QUBITS)
    circuit.h(range(N_QUBITS))
    for qubit, angle in enumerate(angles):
        circuit.rz(angle, qubit)
    for first, second in RING_EDGES:
        # CX-RZ(phi)-CX implements RZZ(phi), with phi=beta*theta_i*theta_j.
        circuit.cx(first, second)
        circuit.rz(ZZ_BETA * angles[first] * angles[second], second)
        circuit.cx(first, second)
    return circuit


def _compiled_feature_map() -> tuple[QuantumCircuit, tuple[Parameter, ...]]:
    """Compile one parameterized template so thousands of binds stay cheap."""

    global _COMPILED_FEATURE_MAP, _FEATURE_PARAMETERS
    if _COMPILED_FEATURE_MAP is None:
        parameters = ParameterVector("theta", N_QUBITS)
        compiled = transpile(
            _feature_map(parameters),
            basis_gates=DEFAULT_BASIS,
            optimization_level=1,
            seed_transpiler=SEED,
        )
        assert_all_gates_noisy(compiled)
        _COMPILED_FEATURE_MAP = compiled
        _FEATURE_PARAMETERS = tuple(parameters)
    if _FEATURE_PARAMETERS is None:  # pragma: no cover - defensive invariant
        raise RuntimeError("Feature-map parameter cache was not initialized.")
    return _COMPILED_FEATURE_MAP, _FEATURE_PARAMETERS


def _bound_feature_maps(features: np.ndarray) -> list[QuantumCircuit]:
    if features.ndim != 2 or features.shape[1] != N_QUBITS:
        raise ValueError(f"Expected a two-dimensional array with {N_QUBITS} columns.")
    template, parameters = _compiled_feature_map()
    return [
        template.assign_parameters(
            dict(zip(parameters, row, strict=True)), inplace=False
        )
        for row in _angles(features)
    ]


def _noisy_simulator() -> AerSimulator:
    global _NOISY_SIMULATOR
    if _NOISY_SIMULATOR is None:
        _NOISY_SIMULATOR = AerSimulator(
            method="density_matrix",
            noise_model=get_benchmark_noise_model(),
            seed_simulator=SEED,
        )
    return _NOISY_SIMULATOR


def _simulate_states(features: np.ndarray, noisy: bool) -> np.ndarray:
    """Simulate one state preparation per row, exactly and without shots."""

    circuits = _bound_feature_maps(features)
    if not noisy:
        return np.asarray(
            [Statevector.from_instruction(circuit).data for circuit in circuits],
            dtype=complex,
        )

    density_matrices: list[np.ndarray] = []
    simulator = _noisy_simulator()
    for start in range(0, len(circuits), SIMULATION_BATCH_SIZE):
        batch = circuits[start : start + SIMULATION_BATCH_SIZE]
        for circuit in batch:
            # The save instruction is added only after checking the fully-noised
            # compiled gate basis; it is an Aer instruction, not a physical gate.
            assert_all_gates_noisy(circuit)
            circuit.save_density_matrix(label="rho")
        result = simulator.run(batch).result()
        density_matrices.extend(
            np.asarray(result.data(index)["rho"], dtype=complex)
            for index in range(len(batch))
        )
    return np.asarray(density_matrices, dtype=complex)


def _kernel_between(
    first_states: np.ndarray, second_states: np.ndarray, noisy: bool
) -> np.ndarray:
    """Evaluate pure-state fidelity or normalized Hilbert--Schmidt overlap."""

    if not noisy:
        kernel = np.abs(first_states @ second_states.conj().T) ** 2
    else:
        first_flat = first_states.reshape(len(first_states), -1)
        second_flat = second_states.reshape(len(second_states), -1)
        overlaps = np.real(first_flat.conj() @ second_flat.T)
        first_purities = np.real(
            np.einsum("ij,ij->i", first_flat.conj(), first_flat)
        )
        second_purities = np.real(
            np.einsum("ij,ij->i", second_flat.conj(), second_flat)
        )
        denominator = np.sqrt(
            np.maximum(first_purities[:, np.newaxis], 0.0)
            * np.maximum(second_purities[np.newaxis, :], 0.0)
        )
        kernel = np.divide(
            overlaps,
            denominator,
            out=np.zeros_like(overlaps),
            where=denominator > 0.0,
        )
    return np.clip(np.real(kernel), 0.0, 1.0)


def _training_and_validation_kernels(
    states: np.ndarray, training_size: int, noisy: bool
) -> tuple[np.ndarray, np.ndarray]:
    training_states = states[:training_size]
    validation_states = states[training_size:]
    training_kernel = _kernel_between(training_states, training_states, noisy)
    training_kernel = (training_kernel + training_kernel.T) / 2.0
    np.fill_diagonal(training_kernel, 1.0)
    validation_kernel = _kernel_between(
        validation_states, training_states, noisy
    )
    return training_kernel, validation_kernel


def _kernel_alignment(first: np.ndarray, second: np.ndarray) -> float:
    """Centered kernel alignment without constructing a centering matrix."""

    first_centered = (
        first
        - first.mean(axis=0, keepdims=True)
        - first.mean(axis=1, keepdims=True)
        + first.mean()
    )
    second_centered = (
        second
        - second.mean(axis=0, keepdims=True)
        - second.mean(axis=1, keepdims=True)
        + second.mean()
    )
    denominator = np.linalg.norm(first_centered) * np.linalg.norm(second_centered)
    if denominator == 0.0:
        return 0.0
    return float(np.sum(first_centered * second_centered) / denominator)


def _stratified_folds(
    x: np.ndarray, y: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=SEED
    )
    return [(train.copy(), validation.copy()) for train, validation in splitter.split(x, y)]


def _fit_fold_preprocessing(
    x_train: np.ndarray, x_other: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, PCA]:
    """Fit the all-feature scaler and PCA on one training partition only."""

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(x_train)
    other_scaled = scaler.transform(x_other)
    pca = PCA(n_components=N_PCA_FEATURES, svd_solver="full")
    train_reduced = pca.fit_transform(train_scaled)
    other_reduced = pca.transform(other_scaled)
    return train_reduced, other_reduced, train_scaled, other_scaled, pca


def _precomputed_predictions(
    training_kernel: np.ndarray,
    validation_kernel: np.ndarray,
    y_train: np.ndarray,
    c_value: float,
) -> tuple[np.ndarray, SVC]:
    model = SVC(
        C=c_value,
        kernel="precomputed",
        class_weight="balanced",
        cache_size=1024,
    )
    model.fit(training_kernel, y_train)
    return model.predict(validation_kernel), model


def _score_summary(scores: Sequence[float]) -> dict:
    values = np.asarray(scores, dtype=float)
    sample_std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    standard_error = sample_std / np.sqrt(len(values))
    half_width = T_975_DF4 * standard_error if len(values) == 5 else 1.96 * standard_error
    return {
        "fold_scores": [float(value) for value in values],
        "mean": float(np.mean(values)),
        "sample_standard_deviation": sample_std,
        "standard_error": float(standard_error),
        "confidence_interval_95": [
            float(max(0.0, np.mean(values) - half_width)),
            float(min(1.0, np.mean(values) + half_width)),
        ],
    }


def _mean_score(scores: dict, key: object) -> float:
    return float(np.mean(scores[key]))


def _cross_validate(x: np.ndarray, y: np.ndarray) -> dict:
    folds = _stratified_folds(x, y)
    quantum_scores = {
        "ideal": {value: [] for value in SVM_C_VALUES},
        "noisy": {value: [] for value in SVM_C_VALUES},
    }
    classical_scores = {
        (c_value, gamma): []
        for c_value in SVM_C_VALUES
        for gamma in RBF_GAMMA_VALUES
    }
    fold_diagnostics = []

    for fold_number, (train_indices, validation_indices) in enumerate(folds, start=1):
        x_train, x_validation = x[train_indices], x[validation_indices]
        y_train, y_validation = y[train_indices], y[validation_indices]
        (
            train_reduced,
            validation_reduced,
            train_scaled,
            validation_scaled,
            pca,
        ) = _fit_fold_preprocessing(x_train, x_validation)
        combined = np.vstack((train_reduced, validation_reduced))
        training_size = len(train_indices)

        ideal_states = _simulate_states(combined, noisy=False)
        ideal_training, ideal_validation = _training_and_validation_kernels(
            ideal_states, training_size, noisy=False
        )
        del ideal_states

        noisy_states = _simulate_states(combined, noisy=True)
        noisy_training, noisy_validation = _training_and_validation_kernels(
            noisy_states, training_size, noisy=True
        )
        del noisy_states

        for condition, training_kernel, validation_kernel in (
            ("ideal", ideal_training, ideal_validation),
            ("noisy", noisy_training, noisy_validation),
        ):
            for c_value in SVM_C_VALUES:
                prediction, _ = _precomputed_predictions(
                    training_kernel, validation_kernel, y_train, c_value
                )
                quantum_scores[condition][c_value].append(
                    float(balanced_accuracy_score(y_validation, prediction))
                )

        for (c_value, gamma), scores in classical_scores.items():
            model = SVC(
                C=c_value,
                kernel="rbf",
                gamma=gamma,
                class_weight="balanced",
                cache_size=1024,
            )
            model.fit(train_scaled, y_train)
            scores.append(
                float(
                    balanced_accuracy_score(
                        y_validation, model.predict(validation_scaled)
                    )
                )
            )

        off_diagonal = ~np.eye(training_size, dtype=bool)
        fold_diagnostics.append(
            {
                "fold": fold_number,
                "training_rows": int(training_size),
                "validation_rows": int(len(validation_indices)),
                "training_class_counts": {
                    label: int(np.sum(y_train == code))
                    for label, code in LABELS.items()
                },
                "validation_class_counts": {
                    label: int(np.sum(y_validation == code))
                    for label, code in LABELS.items()
                },
                "pca_explained_variance_ratio_sum": float(
                    pca.explained_variance_ratio_.sum()
                ),
                "centered_ideal_noisy_kernel_alignment": _kernel_alignment(
                    ideal_training, noisy_training
                ),
                "ideal_noisy_kernel_mean_absolute_difference": float(
                    np.mean(np.abs(ideal_training - noisy_training))
                ),
                "ideal_kernel_off_diagonal_standard_deviation": float(
                    np.std(ideal_training[off_diagonal])
                ),
                "noisy_kernel_off_diagonal_standard_deviation": float(
                    np.std(noisy_training[off_diagonal])
                ),
            }
        )

    selected_ideal_c = max(
        SVM_C_VALUES, key=lambda value: _mean_score(quantum_scores["ideal"], value)
    )
    selected_noisy_c = max(
        SVM_C_VALUES, key=lambda value: _mean_score(quantum_scores["noisy"], value)
    )
    selected_classical = max(
        classical_scores,
        key=lambda value: _mean_score(classical_scores, value),
    )
    return {
        "folds": folds,
        "quantum_scores": quantum_scores,
        "classical_scores": classical_scores,
        "selected_ideal_c": selected_ideal_c,
        "selected_noisy_c": selected_noisy_c,
        "selected_classical": selected_classical,
        "fold_diagnostics": fold_diagnostics,
    }


def _read_official_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    feature_columns = [f"x{index:02d}" for index in range(1, N_INPUT_FEATURES + 1)]
    with TRAIN_PATH.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != feature_columns + ["label"]:
            raise ValueError("Official Q2 training columns do not match x01-x50,label.")
        training_rows = list(reader)
    with TEST_PATH.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != ["sample_id", *feature_columns]:
            raise ValueError("Official Q2 test columns do not match sample_id,x01-x50.")
        test_rows = list(reader)

    if len(training_rows) != 800:
        raise ValueError("Official Q2 training file must contain exactly 800 rows.")
    labels = [row["label"] for row in training_rows]
    if any(label not in LABELS for label in labels) or set(labels) != set(LABELS):
        raise ValueError("Training labels must contain exactly the official A/B classes.")
    test_ids = [row["sample_id"] for row in test_rows]
    if len(test_rows) != 200 or set(test_ids) != {
        f"s{index}" for index in range(200)
    }:
        raise ValueError("Official test IDs must contain exactly s0 through s199.")
    if len(set(test_ids)) != len(test_ids):
        raise ValueError("Official test sample_id values must be unique.")

    x_train = np.asarray(
        [[float(row[column]) for column in feature_columns] for row in training_rows],
        dtype=float,
    )
    y_train = np.asarray([LABELS[label] for label in labels], dtype=int)
    x_test = np.asarray(
        [[float(row[column]) for column in feature_columns] for row in test_rows],
        dtype=float,
    )
    if not np.isfinite(x_train).all() or not np.isfinite(x_test).all():
        raise ValueError("Official Q2 feature matrices must contain only finite values.")
    return x_train, y_train, x_test, test_ids, feature_columns


def _write_predictions(test_ids: list[str], predictions: np.ndarray) -> None:
    prediction_labels = np.where(predictions == 1, "B", "A")
    temporary_path = PREDICTIONS_PATH.with_suffix(".csv.tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["sample_id", "label"])
        writer.writerows(zip(test_ids, prediction_labels, strict=True))
    temporary_path.replace(PREDICTIONS_PATH)


def _string_key(c_value: float, gamma: float | None = None) -> str:
    if gamma is None:
        return str(c_value)
    return f"C={c_value},gamma={gamma}"


def run_experiment() -> dict:
    """Run training-only model selection, fit all 800 rows, and predict Q2 test.

    Returns the metrics dictionary also written to
    ``results/phase2/q2_metrics.json``.  This public entry point intentionally
    takes no test labels and has no code path for loading them.
    """

    started = time.perf_counter()
    x_train, y_train, x_test, test_ids, feature_columns = _read_official_data()
    cv = _cross_validate(x_train, y_train)

    # Refit every preprocessing step and the final noisy quantum SVM on all 800
    # labeled rows.  The 200 test rows participate only in transformation and
    # state preparation after model selection is complete.
    (
        train_reduced,
        test_reduced,
        _,
        _,
        final_pca,
    ) = _fit_fold_preprocessing(x_train, x_test)
    combined_reduced = np.vstack((train_reduced, test_reduced))
    final_noisy_states = _simulate_states(combined_reduced, noisy=True)
    final_training_kernel, final_test_kernel = _training_and_validation_kernels(
        final_noisy_states, len(x_train), noisy=True
    )
    del final_noisy_states
    noisy_predictions, final_model = _precomputed_predictions(
        final_training_kernel,
        final_test_kernel,
        y_train,
        cv["selected_noisy_c"],
    )
    _write_predictions(test_ids, noisy_predictions)

    compiled_circuit, _ = _compiled_feature_map()
    quantum_scores = cv["quantum_scores"]
    classical_scores = cv["classical_scores"]
    selected_ideal_c = cv["selected_ideal_c"]
    selected_noisy_c = cv["selected_noisy_c"]
    selected_classical_c, selected_classical_gamma = cv["selected_classical"]
    ideal_selected_summary = _score_summary(
        quantum_scores["ideal"][selected_ideal_c]
    )
    noisy_selected_summary = _score_summary(
        quantum_scores["noisy"][selected_noisy_c]
    )
    classical_selected_summary = _score_summary(
        classical_scores[(selected_classical_c, selected_classical_gamma)]
    )
    alignments = [
        fold["centered_ideal_noisy_kernel_alignment"]
        for fold in cv["fold_diagnostics"]
    ]
    kernel_differences = [
        fold["ideal_noisy_kernel_mean_absolute_difference"]
        for fold in cv["fold_diagnostics"]
    ]

    fold_training_rows = len(x_train) * (CV_FOLDS - 1) // CV_FOLDS
    fold_validation_rows = len(x_train) // CV_FOLDS
    cv_unique_pairs_one_condition = CV_FOLDS * (
        fold_training_rows * (fold_training_rows + 1) // 2
        + fold_validation_rows * fold_training_rows
    )
    final_unique_pairs = (
        len(x_train) * (len(x_train) + 1) // 2
        + len(x_test) * len(x_train)
    )
    cv_matrix_cells_one_condition = CV_FOLDS * (
        fold_training_rows**2 + fold_validation_rows * fold_training_rows
    )
    final_matrix_cells = len(x_train) ** 2 + len(x_test) * len(x_train)

    result = {
        "dataset": {
            "training_rows_available": len(x_train),
            "training_rows_used": len(x_train),
            # Retained for consumers of the previous result schema; no subset is
            # taken, so this is the complete training set.
            "training_subset_rows": len(x_train),
            "training_class_counts": {
                label: int(np.sum(y_train == code))
                for label, code in LABELS.items()
            },
            "test_rows": len(x_test),
            "test_ids_preserved": True,
            "test_labels_available": False,
            "test_labels_used": False,
            "features_used_for_reduction": feature_columns,
            "pca_components": N_PCA_FEATURES,
            "classical_baseline_features": feature_columns,
        },
        "model_selection": {
            "method": (
                f"{CV_FOLDS}-fold shuffled stratified cross-validation on all "
                "official training rows only"
            ),
            "shared_fold_splits_for_all_models": True,
            "fold_preprocessing_fit_on_training_partition_only": True,
            "random_seed": SEED,
            "quantum_C_grid": list(SVM_C_VALUES),
            "classical_C_grid": list(SVM_C_VALUES),
            "classical_gamma_grid": list(RBF_GAMMA_VALUES),
            "ideal_quantum_selected_C": selected_ideal_c,
            "quantum_selected_C": selected_noisy_c,
            "classical_selected_parameters": {
                "kernel": "RBF",
                "C": selected_classical_c,
                "gamma": selected_classical_gamma,
            },
            "balanced_accuracy_cv": {
                "quantum_ideal": ideal_selected_summary["mean"],
                "quantum_noisy": noisy_selected_summary["mean"],
                "classical_rbf": classical_selected_summary["mean"],
            },
            "selected_model_uncertainty": {
                "quantum_ideal": ideal_selected_summary,
                "quantum_noisy": noisy_selected_summary,
                "classical_rbf": classical_selected_summary,
                "method": (
                    "Descriptive 95% t interval: mean +/- t(0.975, df=4) "
                    "times fold-score sample SD/sqrt(5); CV folds are not "
                    "independent test sets."
                ),
            },
            "quantum_cv_scores_by_C": {
                condition: {
                    _string_key(c_value): float(np.mean(scores))
                    for c_value, scores in by_c.items()
                }
                for condition, by_c in quantum_scores.items()
            },
            "classical_cv_scores_by_parameters": {
                _string_key(c_value, gamma): float(np.mean(scores))
                for (c_value, gamma), scores in classical_scores.items()
            },
            "test_labels_used_for_selection": False,
        },
        "quantum_kernel": {
            "qubits": N_QUBITS,
            "feature_map": (
                "H-RZ encoding with theta=1.2*tanh(PCA_component/2), plus "
                "weak ring RZZ interactions implemented as CX-RZ-CX"
            ),
            "angle_formula": "theta_i = 1.2 * tanh(component_i / 2)",
            "zz_angle_formula": "phi_ij = 0.1 * theta_i * theta_j",
            "zz_beta": ZZ_BETA,
            "logical_ring_edges": [list(edge) for edge in RING_EDGES],
            "entangling_pairs": len(RING_EDGES),
            "logical_feature_map_depth": int(_feature_map(np.zeros(N_QUBITS)).depth()),
            "compiled_feature_map_depth": int(compiled_circuit.depth()),
            "compiled_two_qubit_gate_count": int(
                sum(
                    instruction.operation.num_qubits == 2
                    for instruction in compiled_circuit.data
                )
            ),
            "trainable_quantum_parameters": 0,
            "data_bound_parameters_per_circuit": N_QUBITS,
            "mean_fold_ideal_noisy_kernel_absolute_difference": float(
                np.mean(kernel_differences)
            ),
            "mean_fold_centered_ideal_noisy_kernel_alignment": float(
                np.mean(alignments)
            ),
            "kernel_definition": (
                "Ideal squared state overlap; noisy normalized Hilbert-Schmidt "
                "overlap Tr(rho_i rho_j)/sqrt(Tr(rho_i^2)Tr(rho_j^2))."
            ),
            "noise_model_version": VERSION,
            "fully_noised_basis": list(DEFAULT_BASIS),
            "simulation": (
                "Exact statevector (ideal) and density matrix (noisy), without "
                "measurement shots; readout noise therefore does not enter."
            ),
        },
        "final_model": {
            "model": "Noisy QFF2026-v2 precomputed-kernel SVC",
            "training_rows": len(x_train),
            "selected_C": selected_noisy_c,
            "support_vectors": int(final_model.support_.size),
            "support_vectors_by_class": {
                label: int(final_model.n_support_[code])
                for label, code in LABELS.items()
            },
            "pca_explained_variance_ratio_sum": float(
                final_pca.explained_variance_ratio_.sum()
            ),
            "prediction_class_counts": {
                label: int(np.sum(noisy_predictions == code))
                for label, code in LABELS.items()
            },
            "test_labels_used": False,
        },
        "resource_cost": {
            "folds": CV_FOLDS,
            "state_preparation_circuit_evaluations": {
                "cv_ideal": CV_FOLDS * len(x_train),
                "cv_noisy": CV_FOLDS * len(x_train),
                "final_noisy_train_and_test": len(x_train) + len(x_test),
                "total": 2 * CV_FOLDS * len(x_train) + len(x_train) + len(x_test),
            },
            "unique_quantum_pairwise_kernel_entries": {
                "cv_ideal": cv_unique_pairs_one_condition,
                "cv_noisy": cv_unique_pairs_one_condition,
                "final_noisy": final_unique_pairs,
                "total": 2 * cv_unique_pairs_one_condition + final_unique_pairs,
            },
            "quantum_kernel_matrix_cells_materialized": {
                "cv_ideal": cv_matrix_cells_one_condition,
                "cv_noisy": cv_matrix_cells_one_condition,
                "final_noisy": final_matrix_cells,
                "total": 2 * cv_matrix_cells_one_condition + final_matrix_cells,
            },
            "accounting_note": (
                "A state preparation produces one reusable state. Pairwise "
                "overlaps are classical post-processing of those states; C-grid "
                "SVM fits reuse each kernel and add no circuit evaluations. "
                "Matrix-cell counts include both halves of stored symmetric "
                "training matrices; unique-entry counts do not."
            ),
            "trainable_quantum_parameters": 0,
            "compiled_feature_map_depth": int(compiled_circuit.depth()),
            "compiled_two_qubit_gate_count_per_state_preparation": int(
                sum(
                    instruction.operation.num_qubits == 2
                    for instruction in compiled_circuit.data
                )
            ),
            "training_wall_time_seconds": round(time.perf_counter() - started, 3),
            "transpiler_seed": SEED,
        },
        "prediction_file": str(PREDICTIONS_PATH.relative_to(ROOT)),
        "prediction_rows": len(x_test),
        "diagnostics": {
            "fold_details": cv["fold_diagnostics"],
            "cv_fold_scores": {
                "quantum_ideal": {
                    _string_key(c_value): [float(score) for score in scores]
                    for c_value, scores in quantum_scores["ideal"].items()
                },
                "quantum_noisy": {
                    _string_key(c_value): [float(score) for score in scores]
                    for c_value, scores in quantum_scores["noisy"].items()
                },
                "classical_rbf": {
                    _string_key(c_value, gamma): [float(score) for score in scores]
                    for (c_value, gamma), scores in classical_scores.items()
                },
            },
            "cv_score_summaries_by_hyperparameter": {
                "quantum_ideal": {
                    _string_key(c_value): _score_summary(scores)
                    for c_value, scores in quantum_scores["ideal"].items()
                },
                "quantum_noisy": {
                    _string_key(c_value): _score_summary(scores)
                    for c_value, scores in quantum_scores["noisy"].items()
                },
                "classical_rbf": {
                    _string_key(c_value, gamma): _score_summary(scores)
                    for (c_value, gamma), scores in classical_scores.items()
                },
            },
        },
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_metrics = METRICS_PATH.with_suffix(".json.tmp")
    temporary_metrics.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    temporary_metrics.replace(METRICS_PATH)
    return result


if __name__ == "__main__":
    print(json.dumps(run_experiment(), indent=2))
