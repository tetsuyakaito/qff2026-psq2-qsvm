#!/usr/bin/env python3
"""
audit_repo.py
=============
QFF 2026 Phase 1 Submission Auditor
Verifies that all 19 mandatory submission files exist and are non-empty,
confirms all mandatory rubric keywords ("Processor D", "Ideology", "SWAP",
"kernel concentration") are present in the documentation, and checks that
main.ipynb has executed cleanly without error outputs.
"""

import sys
import json
import pathlib
import pandas as pd

# The 19 Mandatory Files for QFF 2026 Phase 1 PS-Q2
MANDATORY_19_FILES = [
    "README.md",
    "main.ipynb",
    "requirements.txt",
    "report.md",
    "processors/processor_A.json",
    "processors/processor_B.json",
    "src/__init__.py",
    "src/processors.py",
    "src/problem.py",
    "src/routing.py",
    "src/metrics.py",
    "src/visualization.py",
    "results/tables/architecture_routing_comparison.csv",
    "results/tables/classification_metrics.csv",
    "results/figures/fig1_processor_topologies.png",
    "results/figures/fig2_architecture_comparison.png",
    "results/figures/fig3_kernel_matrices_and_concentration.png",
    "results/figures/fig4_decision_boundaries.png",
    "results/figures/fig5_accuracy_auc_comparison.png",
]

# Mandatory Rubric Keywords
MANDATORY_KEYWORDS = [
    "Processor D",
    "Ideology",
    "SWAP",
    "kernel concentration",
]


def audit_repository() -> bool:
    print("=" * 75)
    print("  QFF 2026 Phase 1 Submission Audit: PS-Q2 Quantum Kernel SVM")
    print("=" * 75)

    all_passed = True
    repo_root = pathlib.Path(".")

    # ── Check 1: Mandatory 19 Files ───────────────────────────────────────────
    print("\n[CHECK 1/4] Verifying the 19 Mandatory Submission Files:")
    missing_or_empty = []
    for idx, rel_path in enumerate(MANDATORY_19_FILES, 1):
        p = repo_root / rel_path
        if not p.exists():
            print(f"  ❌ [{idx:02d}/19] MISSING: {rel_path}")
            missing_or_empty.append(rel_path)
            all_passed = False
        elif p.stat().st_size == 0:
            print(f"  ❌ [{idx:02d}/19] EMPTY: {rel_path}")
            missing_or_empty.append(rel_path)
            all_passed = False
        else:
            size_kb = p.stat().st_size / 1024
            print(f"  ✓  [{idx:02d}/19] {rel_path:<52} ({size_kb:8.1f} KB)")

    if missing_or_empty:
        print(f"\n❌ FAILED Check 1: {len(missing_or_empty)} files missing or empty.")
    else:
        print("\n✓ PASSED Check 1: All 19 mandatory submission files exist and are non-empty.")

    # ── Check 2: Mandatory Rubric Keywords in Documentation ───────────────────
    print("\n[CHECK 2/4] Verifying Mandatory Rubric Keywords in Documentation:")
    docs_to_check = ["report.md", "README.md"]
    doc_texts = {}
    for doc in docs_to_check:
        p = repo_root / doc
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                doc_texts[doc] = f.read()
        else:
            doc_texts[doc] = ""

    combined_docs = "\n".join(doc_texts.values())
    keyword_failures = []
    for kw in MANDATORY_KEYWORDS:
        # Check case-insensitively or exact
        count_report = doc_texts.get("report.md", "").count(kw)
        count_readme = doc_texts.get("README.md", "").count(kw)
        total_count = count_report + count_readme
        if total_count == 0:
            print(f"  ❌ Keyword '{kw}' NOT found in documentation!")
            keyword_failures.append(kw)
            all_passed = False
        else:
            print(f"  ✓  Keyword '{kw:<24}' found: {total_count} occurrences (report: {count_report}, README: {count_readme})")

    # Check for dedicated section title "Processor D Ideology"
    if "Processor D Ideology" in doc_texts.get("report.md", ""):
        print("  ✓  Dedicated section 'Processor D Ideology' confirmed in report.md")
    else:
        print("  ❌ Dedicated section 'Processor D Ideology' missing in report.md!")
        all_passed = False

    # Check for AI disclosure
    if "AI Disclosure" in combined_docs or "generative AI" in combined_docs:
        print("  ✓  Mandatory AI Disclosure confirmed in documentation")
    else:
        print("  ❌ Mandatory AI Disclosure missing in documentation!")
        all_passed = False

    if keyword_failures:
        print(f"\n❌ FAILED Check 2: Missing keywords: {keyword_failures}")
    else:
        print("\n✓ PASSED Check 2: All mandatory rubric keywords and disclosures verified.")

    # ── Check 3: main.ipynb Clean Execution & Populated Outputs ───────────────
    print("\n[CHECK 3/4] Verifying main.ipynb Clean Execution & Output Cells:")
    nb_path = repo_root / "main.ipynb"
    if nb_path.exists():
        try:
            with open(nb_path, "r", encoding="utf-8") as f:
                nb = json.load(f)
            cells = nb.get("cells", [])
            code_cells = [c for c in cells if c.get("cell_type") == "code"]
            cells_with_errors = []
            cells_without_output = []

            for idx, c in enumerate(code_cells, 1):
                outputs = c.get("outputs", [])
                if not outputs:
                    cells_without_output.append(idx)
                for out in outputs:
                    if out.get("output_type") == "error":
                        cells_with_errors.append((idx, out.get("ename"), out.get("evalue")))

            print(f"  • Total cells in notebook: {len(cells)} ({len(code_cells)} code cells)")
            if cells_with_errors:
                print(f"  ❌ Errors detected in code cells: {cells_with_errors}")
                all_passed = False
            else:
                print(f"  ✓  Zero exceptions across all {len(code_cells)} code cells")

            if cells_without_output:
                print(f"  ❌ Cells without output: {cells_without_output}")
                all_passed = False
            else:
                print(f"  ✓  All {len(code_cells)} code cells have populated outputs & inline figures")

        except Exception as e:
            print(f"  ❌ Failed to parse main.ipynb: {e}")
            all_passed = False
    else:
        print("  ❌ main.ipynb does not exist.")
        all_passed = False

    # ── Check 4: Results Tables Validity ──────────────────────────────────────
    print("\n[CHECK 4/4] Verifying Exported Results Tables:")
    for tbl in ["results/tables/architecture_routing_comparison.csv", "results/tables/classification_metrics.csv"]:
        p = repo_root / tbl
        if p.exists():
            try:
                df = pd.read_csv(p)
                print(f"  ✓  {tbl:<52} ({len(df)} rows, {len(df.columns)} columns)")
            except Exception as e:
                print(f"  ❌ Could not read {tbl}: {e}")
                all_passed = False
        else:
            print(f"  ❌ Missing table: {tbl}")
            all_passed = False

    # ── Final Verdict ─────────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    if all_passed:
        print("  🎉 AUDIT RESULT: ALL CHECKS PASSED PERFECTLY (100% SUBMISSION READY)")
        print("=" * 75 + "\n")
        return True
    else:
        print("  ❌ AUDIT RESULT: ISSUES DETECTED. PLEASE FIX BEFORE SUBMISSION.")
        print("=" * 75 + "\n")
        return False


if __name__ == "__main__":
    success = audit_repository()
    sys.exit(0 if success else 1)
