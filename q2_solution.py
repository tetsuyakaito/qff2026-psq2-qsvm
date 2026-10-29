"""Reproducible PS-Q2 quantum-kernel and classical-baseline workflow."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from scipy.optimize import minimize

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

N_QUBITS = 4
N_PCA_FEATURES = 4
N_SIGNAL_FEATURES = 6
TRAINING_SUBSET_SIZE = 200
CV_FOLDS = 3
SVM_C_VALUES = (0.01, 0.1, 1.0, 10.0, 100.0)
RBF_GAMMA_VALUES = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
LABELS = {"A": 0, "B": 1}


def _fit_standardized(
    x_train: np.ndarray, x_other: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    train_signal = x_train[:, :N_SIGNAL_FEATURES]
    other_signal = x_other[:, :N_SIGNAL_FEATURES]
    mean = train_signal.mean(axis=0)
    scale = train_signal.std(axis=0)
    scale[scale == 0] = 1.0
    train_scaled = (train_signal - mean) / scale
    other_scaled = (other_signal - mean) / scale
    return train_scaled, other_scaled


def _fit_transform(
    x_train: np.ndarray, x_other: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    train_scaled, other_scaled = _fit_standardized(x_train, x_other)
    pca_mean = train_scaled.mean(axis=0)
    centered_train = train_scaled - pca_mean
    _, _, components = np.linalg.svd(centered_train, full_matrices=False)
    projection = components[:N_PCA_FEATURES].copy()
    largest = np.argmax(np.abs(projection), axis=1)
    signs = np.sign(projection[np.arange(N_PCA_FEATURES), largest])
    signs[signs == 0] = 1.0
    projection *= signs[:, np.newaxis]
    return (
        centered_train @ projection.T,
        (other_scaled - pca_mean) @ projection.T,
    )


def _angles(features: np.ndarray) -> np.ndarray:
    return np.pi * np.tanh(features / 2.0)


def _feature_map(angles: np.ndarray) -> QuantumCircuit:
    circuit = QuantumCircuit(N_QUBITS)
    circuit.h(range(N_QUBITS))
    for qubit, angle in enumerate(angles):
        circuit.rz(float(angle), qubit)
    for control in range(N_QUBITS):
        for target in range(control + 1, N_QUBITS):
            circuit.cx(control, target)
            circuit.rz(float(0.5 * angles[control] * angles[target]), target)
            circuit.cx(control, target)
    return circuit


def _simulate_states(features: np.ndarray, noisy: bool) -> np.ndarray:
    circuits = [_feature_map(row) for row in _angles(features)]
    compiled = transpile(
        circuits,
        basis_gates=DEFAULT_BASIS,
        optimization_level=1,
        seed_transpiler=SEED,
    )
    if not isinstance(compiled, list):
        compiled = [compiled]
    for circuit in compiled:
        assert_all_gates_noisy(circuit)

    if not noisy:
        return np.asarray(
            [Statevector.from_instruction(circuit).data for circuit in compiled]
        )

    for circuit in compiled:
        circuit.save_density_matrix(label="rho")
    simulator = AerSimulator(
        method="density_matrix",
        noise_model=get_benchmark_noise_model(),
        seed_simulator=SEED,
    )
    result = simulator.run(compiled).result()
    return np.asarray([result.data(index)["rho"] for index in range(len(compiled))])


def _kernel(states: np.ndarray, noisy: bool) -> np.ndarray:
    if not noisy:
        return np.abs(states @ states.conj().T) ** 2
    gram = np.real(np.einsum("aij,bji->ab", states, states))
    diagonal = np.sqrt(np.maximum(np.diag(gram), 0.0))
    normalized = gram / np.outer(diagonal, diagonal)
    return np.clip((normalized + normalized.T) / 2.0, 0.0, 1.0)


def _kernel_alignment(first: np.ndarray, second: np.ndarray) -> float:
    centering = np.eye(len(first)) - np.ones_like(first) / len(first)
    first_centered = centering @ first @ centering
    second_centered = centering @ second @ centering
    denominator = np.linalg.norm(first_centered) * np.linalg.norm(second_centered)
    return float(np.sum(first_centered * second_centered) / denominator)


def _select_training_subset(
    x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    classes, class_counts = np.unique(y, return_counts=True)
    allocations = np.floor(
        TRAINING_SUBSET_SIZE * class_counts / len(y)
    ).astype(int)
    for index in range(TRAINING_SUBSET_SIZE - int(allocations.sum())):
        allocations[index % len(allocations)] += 1
    selected = [
        rng.permutation(np.flatnonzero(y == label))[:count]
        for label, count in zip(classes, allocations, strict=True)
    ]
    indices = np.concatenate(selected)
    rng.shuffle(indices)
    return x[indices], y[indices]


def _stratified_folds(y: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(SEED)
    validation_folds: list[list[np.ndarray]] = [
        [] for _ in range(CV_FOLDS)
    ]
    for label in np.unique(y):
        indices = rng.permutation(np.flatnonzero(y == label))
        for fold_index, fold_indices in enumerate(
            np.array_split(indices, CV_FOLDS)
        ):
            validation_folds[fold_index].append(fold_indices)
    all_indices = np.arange(len(y))
    splits = []
    for parts in validation_folds:
        validation_indices = np.concatenate(parts)
        train_mask = np.ones(len(y), dtype=bool)
        train_mask[validation_indices] = False
        splits.append((all_indices[train_mask], validation_indices))
    return splits


def _balanced_accuracy(y_true: np.ndarray, y_predicted: np.ndarray) -> float:
    recalls = [
        float(np.mean(y_predicted[y_true == label] == label))
        for label in np.unique(y_true)
    ]
    return float(np.mean(recalls))


def _fit_kernel_svm(
    kernel_train: np.ndarray, labels: np.ndarray, c_value: float
) -> tuple[np.ndarray, np.ndarray, float]:
    signed_labels = np.where(labels == 1, 1.0, -1.0)
    quadratic = np.outer(signed_labels, signed_labels) * kernel_train
    sample_count = len(labels)

    def objective(alpha: np.ndarray) -> float:
        return float(0.5 * alpha @ quadratic @ alpha - alpha.sum())

    def gradient(alpha: np.ndarray) -> np.ndarray:
        return quadratic @ alpha - 1.0

    result = minimize(
        objective,
        np.zeros(sample_count),
        jac=gradient,
        bounds=[(0.0, c_value)] * sample_count,
        constraints={
            "type": "eq",
            "fun": lambda alpha: float(alpha @ signed_labels),
            "jac": lambda alpha: signed_labels + np.zeros_like(alpha),
        },
        method="SLSQP",
        options={"ftol": 1e-9, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(f"Kernel SVM optimization failed: {result.message}")
    alpha = np.clip(result.x, 0.0, c_value)
    support = alpha > 1e-7
    free_support = (alpha > 1e-7) & (alpha < c_value - 1e-7)
    margin_indices = np.flatnonzero(free_support)
    if not len(margin_indices):
        margin_indices = np.flatnonzero(support)
    if not len(margin_indices):
        raise RuntimeError("Kernel SVM optimization returned no support vectors.")
    support_indices = np.flatnonzero(support)
    coefficients = alpha[support_indices] * signed_labels[support_indices]
    intercepts = signed_labels[margin_indices] - (
        kernel_train[np.ix_(margin_indices, support_indices)] @ coefficients
    )
    return alpha, signed_labels, float(np.mean(intercepts))


def _predict_kernel_svm(
    kernel_test_train: np.ndarray,
    alpha: np.ndarray,
    signed_labels: np.ndarray,
    intercept: float,
) -> np.ndarray:
    return (
        kernel_test_train @ (alpha * signed_labels) + intercept >= 0.0
    ).astype(int)


def _rbf_kernel(x_first: np.ndarray, x_second: np.ndarray, gamma: float) -> np.ndarray:
    first_norm = np.sum(x_first**2, axis=1)[:, np.newaxis]
    second_norm = np.sum(x_second**2, axis=1)[np.newaxis, :]
    squared_distances = np.maximum(
        first_norm + second_norm - 2.0 * x_first @ x_second.T,
        0.0,
    )
    return np.exp(-gamma * squared_distances)


def _cross_validate(
    x: np.ndarray, y: np.ndarray
) -> tuple[dict, dict, dict]:
    folds = _stratified_folds(y)
    quantum_scores = {
        "ideal": {value: [] for value in SVM_C_VALUES},
        "noisy": {value: [] for value in SVM_C_VALUES},
    }
    classical_scores = {
        (c_value, gamma): []
        for c_value in SVM_C_VALUES
        for gamma in RBF_GAMMA_VALUES
    }
    alignment_scores = []

    for train_indices, validation_indices in folds:
        x_train, x_validation = x[train_indices], x[validation_indices]
        y_train, y_validation = y[train_indices], y[validation_indices]
        train_reduced, validation_reduced = _fit_transform(
            x_train, x_validation
        )
        classical_train, classical_validation = _fit_standardized(
            x_train, x_validation
        )
        combined = np.vstack((train_reduced, validation_reduced))
        split_at = len(train_reduced)

        ideal_states = _simulate_states(combined, noisy=False)
        ideal_kernel = _kernel(ideal_states, noisy=False)
        noisy_states = _simulate_states(combined, noisy=True)
        noisy_kernel = _kernel(noisy_states, noisy=True)
        alignment_scores.append(
            _kernel_alignment(
                ideal_kernel[:split_at, :split_at],
                noisy_kernel[:split_at, :split_at],
            )
        )

        for condition, matrix in (
            ("ideal", ideal_kernel),
            ("noisy", noisy_kernel),
        ):
            train_kernel = matrix[:split_at, :split_at]
            validation_kernel = matrix[split_at:, :split_at]
            for c_value in SVM_C_VALUES:
                alpha, signed_labels, intercept = _fit_kernel_svm(
                    train_kernel, y_train, c_value
                )
                prediction = _predict_kernel_svm(
                    validation_kernel, alpha, signed_labels, intercept
                )
                quantum_scores[condition][c_value].append(
                    _balanced_accuracy(y_validation, prediction)
                )

        for (c_value, gamma), scores in classical_scores.items():
            train_kernel = _rbf_kernel(
                classical_train, classical_train, gamma
            )
            validation_kernel = _rbf_kernel(
                classical_validation, classical_train, gamma
            )
            alpha, signed_labels, intercept = _fit_kernel_svm(
                train_kernel, y_train, c_value
            )
            prediction = _predict_kernel_svm(
                validation_kernel, alpha, signed_labels, intercept
            )
            scores.append(_balanced_accuracy(y_validation, prediction))

    quantum_means = {
        condition: {
            str(c_value): float(np.mean(scores))
            for c_value, scores in by_c.items()
        }
        for condition, by_c in quantum_scores.items()
    }
    classical_means = {
        f"C={c_value},gamma={gamma}": float(np.mean(scores))
        for (c_value, gamma), scores in classical_scores.items()
    }
    return (
        {"quantum": quantum_means, "classical": classical_means},
        {"quantum": quantum_scores, "classical": classical_scores},
        {"fold_kernel_alignment": alignment_scores},
    )


def run_experiment() -> dict:
    started = time.perf_counter()
    feature_columns = [f"x{index:02d}" for index in range(1, 51)]
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

    x_all = np.asarray(
        [[float(row[column]) for column in feature_columns] for row in training_rows],
        dtype=float,
    )
    y_all = np.asarray([LABELS[label] for label in labels], dtype=int)
    x_train, y_train = _select_training_subset(x_all, y_all)
    x_test = np.asarray(
        [[float(row[column]) for column in feature_columns] for row in test_rows],
        dtype=float,
    )

    cv_means, cv_scores, diagnostics = _cross_validate(x_train, y_train)
    best_quantum_c = max(
        SVM_C_VALUES,
        key=lambda value: cv_means["quantum"]["noisy"][str(value)],
    )
    best_classical_key = max(
        cv_means["classical"],
        key=cv_means["classical"].get,
    )
    best_classical_c, best_classical_gamma = (
        float(part.split("=")[1])
        for part in best_classical_key.split(",")
    )

    train_reduced, test_reduced = _fit_transform(x_train, x_test)
    classical_train, classical_test = _fit_standardized(x_train, x_test)
    combined_reduced = np.vstack((train_reduced, test_reduced))
    split_at = len(train_reduced)
    ideal_states = _simulate_states(combined_reduced, noisy=False)
    noisy_states = _simulate_states(combined_reduced, noisy=True)
    ideal_kernel = _kernel(ideal_states, noisy=False)
    noisy_kernel = _kernel(noisy_states, noisy=True)
    noisy_alpha, noisy_signed_labels, noisy_intercept = _fit_kernel_svm(
        noisy_kernel[:split_at, :split_at], y_train, best_quantum_c
    )
    noisy_predictions = _predict_kernel_svm(
        noisy_kernel[split_at:, :split_at],
        noisy_alpha,
        noisy_signed_labels,
        noisy_intercept,
    )

    classical_train_kernel = _rbf_kernel(
        classical_train, classical_train, best_classical_gamma
    )
    classical_test_kernel = _rbf_kernel(
        classical_test, classical_train, best_classical_gamma
    )
    classical_alpha, classical_signed_labels, classical_intercept = _fit_kernel_svm(
        classical_train_kernel, y_train, best_classical_c
    )
    _ = _predict_kernel_svm(
        classical_test_kernel,
        classical_alpha,
        classical_signed_labels,
        classical_intercept,
    )

    prediction_labels = np.where(noisy_predictions == 1, "B", "A")
    if len(prediction_labels) != len(test_rows) or len(test_ids) != len(test_rows):
        raise ValueError("Prediction rows or sample IDs do not match the official test file.")
    with PREDICTIONS_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["sample_id", "label"])
        writer.writerows(zip(test_ids, prediction_labels, strict=True))

    training_kernel_ideal = ideal_kernel[:split_at, :split_at]
    training_kernel_noisy = noisy_kernel[:split_at, :split_at]
    off_diagonal = ~np.eye(split_at, dtype=bool)
    circuit = transpile(
        _feature_map(_angles(train_reduced[0])),
        basis_gates=DEFAULT_BASIS,
        optimization_level=1,
        seed_transpiler=SEED,
    )
    result = {
        "dataset": {
            "training_rows_available": len(training_rows),
            "training_subset_rows": len(x_train),
            "training_subset_class_counts": {
                label: int(np.sum(y_train == code))
                for label, code in LABELS.items()
            },
            "test_rows": len(test_rows),
            "test_ids_preserved": True,
            "test_labels_used": False,
            "features_used_for_reduction": feature_columns[:N_SIGNAL_FEATURES],
            "pca_components": N_PCA_FEATURES,
            "classical_baseline_features": feature_columns[:N_SIGNAL_FEATURES],
        },
        "model_selection": {
            "method": f"{CV_FOLDS}-fold stratified cross-validation on training subset only",
            "quantum_selected_C": best_quantum_c,
            "classical_selected_parameters": {
                "kernel": "RBF",
                "C": best_classical_c,
                "gamma": best_classical_gamma,
            },
            "balanced_accuracy_cv": {
                "quantum_ideal": cv_means["quantum"]["ideal"][str(best_quantum_c)],
                "quantum_noisy": cv_means["quantum"]["noisy"][str(best_quantum_c)],
                "classical_rbf": cv_means["classical"][best_classical_key],
            },
            "quantum_cv_scores_by_C": cv_means["quantum"],
            "classical_cv_scores_by_parameters": cv_means["classical"],
            "test_labels_used_for_selection": False,
        },
        "quantum_kernel": {
            "qubits": N_QUBITS,
            "feature_map": "H-RZ data encoding plus all-pairs ZZ interactions on K4",
            "entangling_pairs": N_QUBITS * (N_QUBITS - 1) // 2,
            "compiled_feature_map_depth": int(circuit.depth()),
            "compiled_two_qubit_gate_count": int(
                sum(1 for instruction in circuit.data if instruction.operation.num_qubits == 2)
            ),
            "ideal_noisy_kernel_mean_absolute_difference": float(
                np.mean(np.abs(training_kernel_ideal - training_kernel_noisy))
            ),
            "ideal_noisy_training_kernel_alignment": _kernel_alignment(
                training_kernel_ideal, training_kernel_noisy
            ),
            "ideal_training_kernel_off_diagonal_std": float(
                np.std(training_kernel_ideal[off_diagonal])
            ),
            "noisy_training_kernel_off_diagonal_std": float(
                np.std(training_kernel_noisy[off_diagonal])
            ),
            "kernel_definition": (
                "Ideal squared state overlap; noisy normalized Hilbert-Schmidt "
                "overlap Tr(rho_i rho_j)/sqrt(Tr(rho_i^2)Tr(rho_j^2))."
            ),
            "noise_model_version": VERSION,
        },
        "resource_cost": {
            "folds": CV_FOLDS,
            "state_preparation_circuit_evaluations": (
                CV_FOLDS * len(x_train) * 2 + (len(x_train) + len(test_rows)) * 2
            ),
            "unique_symmetric_kernel_values": (
                CV_FOLDS * 2 * (len(x_train) * (len(x_train) + 1) // 2)
                + 2
                * (
                    (len(x_train) + len(test_rows))
                    * (len(x_train) + len(test_rows) + 1)
                    // 2
                )
            ),
            "test_prediction_model": "Noisy QFF2026-v2 kernel",
            "training_wall_time_seconds": round(time.perf_counter() - started, 3),
            "transpiler_seed": SEED,
        },
        "prediction_file": str(PREDICTIONS_PATH.relative_to(ROOT)),
        "prediction_rows": len(test_rows),
        "diagnostics": {
            "mean_fold_centered_kernel_alignment": float(
                np.mean(diagnostics["fold_kernel_alignment"])
            ),
            "cv_fold_scores": {
                "quantum_ideal": {
                    str(value): [
                        float(score)
                        for score in cv_scores["quantum"]["ideal"][value]
                    ]
                    for value in SVM_C_VALUES
                },
                "quantum_noisy": {
                    str(value): [
                        float(score)
                        for score in cv_scores["quantum"]["noisy"][value]
                    ]
                    for value in SVM_C_VALUES
                },
            },
        },
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run_experiment(), indent=2))
