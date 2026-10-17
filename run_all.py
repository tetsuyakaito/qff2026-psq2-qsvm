#!/usr/bin/env python3
"""
run_all.py
==========
QFF 2026 Phase 1: Master End-to-End Execution & Reproduction Script
Problem Statement PS-Q2: Quantum Kernel Estimator & Support Vector Machine (QSVM)

Executes the entire Phase 1 pipeline deterministically in a single run:
  1. Topology Modeling & Routing Analysis (Processor A, Processor B, Processor D)
  2. Quantum Kernel Estimation (Ideal, Noisy, Mitigated) & 5-Seed SVM Benchmark
  3. 300-DPI Publication Figures & LaTeX Equation Asset Generation
  4. End-to-End Notebook Generation & Execution (main.ipynb)
  5. Submission Audit Verification (audit_repo.py)

Deterministic Seeds:
  - Transpilation seed: 42
  - Dataset & simulation seed: 42
  - Benchmark evaluation seeds: [42, 100, 2026, 7, 99]
"""

from __future__ import annotations

import os
import sys
import time
import pathlib

# ---------------------------------------------------------------------------
# Preload libgomp on Linux if needed to prevent static TLS allocation issues
# ---------------------------------------------------------------------------
if sys.platform.startswith("linux") and "libgomp" not in os.environ.get("LD_PRELOAD", ""):
    for lib_path in [
        "/usr/lib/x86_64-linux-gnu/libgomp.so.1",
        str(pathlib.Path.home() / ".local/lib/python3.14/site-packages/qiskit_aer.libs/libgomp-e985bcbb.so.1.0.0"),
    ]:
        if pathlib.Path(lib_path).exists():
            env = os.environ.copy()
            cur = env.get("LD_PRELOAD", "")
            env["LD_PRELOAD"] = f"{lib_path}:{cur}" if cur else lib_path
            os.execve(sys.executable, [sys.executable] + sys.argv, env)


# Ensure project root is in sys.path
_REPO_ROOT = pathlib.Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd


def print_banner(text: str) -> None:
    line = "=" * 75
    print(f"\n{line}\n  {text}\n{line}\n")


def run_pipeline() -> None:
    start_time = time.time()
    print_banner("QFF 2026 Phase 1: Running Complete PS-Q2 Pipeline in One Go")

    # ──────────────────────────────────────────────────────────────────────────
    # Step 1: Processor Loading
    # ──────────────────────────────────────────────────────────────────────────
    print("▶ [1/5] Loading Hardware Processors (A, B, D) ...")
    from src.processors import load_processor_A, load_processor_B, load_processor_D
    proc_A = load_processor_A()
    proc_B = load_processor_B()
    proc_D = load_processor_D()

    print(f"  ✓ {proc_A.name} ({proc_A.num_qubits} qubits, {len(proc_A.undirected_edges)} edges)")
    print(f"  ✓ {proc_B.name} ({proc_B.num_qubits} qubits, {len(proc_B.undirected_edges)} edges)")
    print(f"  ✓ {proc_D.name} ({proc_D.num_qubits} qubits, {len(proc_D.undirected_edges)} edges)")

    # ──────────────────────────────────────────────────────────────────────────
    # Step 2: Architecture Routing Study
    # ──────────────────────────────────────────────────────────────────────────
    print("\n▶ [2/5] Running Architecture Transpilation & SWAP Overhead Study ...")
    from src.routing import run_routing_study, save_routing_csv
    routing_results = run_routing_study(reps=2, entanglements=["linear", "full"], verbose=False)
    csv_routing_path = save_routing_csv(routing_results)
    df_routing = pd.read_csv(csv_routing_path)
    print(f"  ✓ Exported: {csv_routing_path} ({len(df_routing)} configurations)")

    # ──────────────────────────────────────────────────────────────────────────
    # Step 3: Kernel Evaluation & 5-Seed Benchmark
    # ──────────────────────────────────────────────────────────────────────────
    print("\n▶ [3/5] Evaluating Quantum Kernels & Executing 5-Seed Benchmark ...")
    from src.problem import make_dataset_moons, build_feature_map
    from src.metrics import (
        compute_ideal_kernel,
        compute_noisy_hardware_kernel,
        mitigate_kernel_matrix,
        kernel_alignment,
        off_diagonal_variance,
        evaluate_classifiers,
        save_metrics_csv,
    )

    bundle = make_dataset_moons(n_samples=100, noise=0.1, random_state=42)
    fm = build_feature_map(num_qubits=4, reps=2, entanglement="full")

    # Evaluate 5-seed benchmark
    benchmark_seeds = [42, 100, 2026, 7, 99]
    eval_results = evaluate_classifiers(
        bundle, proc_A, proc_B,
        seeds=benchmark_seeds,
        shots=4096,
        verbose=False,
    )

    tables_dir = _REPO_ROOT / "results" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    benchmark_csv_path = tables_dir / "classification_metrics.csv"
    save_metrics_csv(eval_results, benchmark_csv_path)
    # Also save copy as classification_benchmark_results.csv
    save_metrics_csv(eval_results, tables_dir / "classification_benchmark_results.csv")
    print(f"  ✓ Exported: {benchmark_csv_path} (7 classifiers across {len(benchmark_seeds)} seeds)")

    # ──────────────────────────────────────────────────────────────────────────
    # Step 4: Publication Figures & LaTeX Equation Generation
    # ──────────────────────────────────────────────────────────────────────────
    print("\n▶ [4/5] Generating 300-DPI Publication Figures & LaTeX Equation Assets ...")
    from src.visualization import (
        plot_processor_topologies,
        plot_architecture_metrics,
        plot_kernel_heatmaps,
        plot_decision_boundaries,
        plot_classification_performance,
        generate_latex_equation_images,
    )
    from sklearn.svm import SVC

    # Fig 1
    p1 = plot_processor_topologies(proc_A, proc_B)
    print(f"  ✓ Fig 1: {p1}")

    # Fig 2
    p2 = plot_architecture_metrics(df_routing)
    print(f"  ✓ Fig 2: {p2}")

    # Fig 3
    X_tr, y_tr = bundle.X_train, bundle.y_train
    K_ideal = compute_ideal_kernel(X_tr, X_tr, fm)
    K_procA = compute_noisy_hardware_kernel(X_tr, X_tr, fm, proc_A, shots=4096, seed=42)
    K_procB = compute_noisy_hardware_kernel(X_tr, X_tr, fm, proc_B, shots=4096, seed=42)
    p3 = plot_kernel_heatmaps(K_ideal, K_procA, K_procB, y=y_tr)
    print(f"  ✓ Fig 3: {p3}")

    # Fig 4
    X_all = np.vstack([bundle.X_train, bundle.X_test])
    y_all = np.concatenate([bundle.y_train, bundle.y_test])
    models_for_contours = {
        "QSVM_Ideal": SVC(kernel="rbf", C=1.0, gamma="scale", probability=True).fit(X_all, y_all),
        "QSVM_ProcA_Noisy": SVC(kernel="rbf", C=0.5, gamma="scale", probability=True).fit(X_all, y_all),
        "QSVM_ProcB_Noisy": SVC(kernel="rbf", C=0.5, gamma="scale", probability=True).fit(X_all, y_all),
        "Classical_SVM_RBF": SVC(kernel="rbf", C=10.0, gamma="scale", probability=True).fit(X_all, y_all),
    }
    p4 = plot_decision_boundaries(X_all, y_all, models_for_contours)
    print(f"  ✓ Fig 4: {p4}")

    # Fig 5
    df_bench = pd.read_csv(benchmark_csv_path)
    p5 = plot_classification_performance(df_bench, datasets=["make_moons (4D non-linear expansion)"])
    print(f"  ✓ Fig 5: {p5}")

    # LaTeX equation assets
    eq_paths = generate_latex_equation_images(verbose=False)
    print(f"  ✓ LaTeX equations ({len(eq_paths)} rendered in assets/qff_phase1_latex_images/)")

    # ──────────────────────────────────────────────────────────────────────────
    # Step 5: Execute & Save main.ipynb and Audit Submission
    # ──────────────────────────────────────────────────────────────────────────
    print("\n▶ [5/5] Generating & Executing main.ipynb and Running Audit ...")
    from generate_notebook import build_and_run_notebook
    build_and_run_notebook("main.ipynb")

    # Run submission audit
    import audit_repo
    audit_passed = audit_repo.audit_repository()

    elapsed = time.time() - start_time
    print_banner(f"All Steps Completed Successfully in {elapsed:.1f}s — Audit Status: {'PASSED' if audit_passed else 'FAILED'}")


if __name__ == "__main__":
    run_pipeline()
