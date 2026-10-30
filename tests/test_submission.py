"""Submission-level regression checks for the QFF2026 QML kit-v3 entry.

The suite intentionally uses only :mod:`unittest` and the dependencies pinned
by the official kit. Run every check from the repository root with::

    python -m unittest discover -s tests -v

The SHA-256 values below identify the unmodified organizer files bundled with
kit v3. Update them only when deliberately upgrading to a different official
kit release; never update a digest merely to make an unexpected change pass.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import importlib.util
import json
import math
import re
import unittest
from collections import Counter
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]
FEATURE_COLUMNS = [f"x{index:02d}" for index in range(1, 51)]
TRAIN_PATH = ROOT / "data" / "q2" / "hd_data_train.csv"
TEST_PATH = ROOT / "data" / "q2" / "hd_data_test_features.csv"
PREDICTIONS_PATH = ROOT / "predictions_q2.csv"

# Authoritative organizer inputs from QFF2026 QML kit v3. Submission-owned
# files (model code, predictions, configuration, and results) are not listed.
OFFICIAL_SHA256 = {
    "benchmark/qff_qml_benchmark.py": (
        "975892045c6fc75e6cf73760a35c5327607fc52611aa217eca8667587795fd75"
    ),
    "data/q2/hd_data_test_features.csv": (
        "3f96fcb5619fafc42218ca395f30730a6539d886043740927d4ca4ad40b19966"
    ),
    "data/q2/hd_data_train.csv": (
        "b7b6aa70e3c7b86148694833b686dd0c01c255c8781c6d8c34ea307dd958835b"
    ),
    "noise_model/benchmark_noise_model.py": (
        "65b6a7bf8f395754cf0d493df4c913fcc77804b403cfd2954f4b6f23e829dde6"
    ),
    "qml-requirements.txt": (
        "dd18cdbf3e4b1b8784ec28db11560f86beb508516d461bd0fcf5700fe304245e"
    ),
    "schemas/processor_d_schema.json": (
        "aefcd98d37c56c5756a72d325ad163ca28001afd600f92955958e92e4866ce26"
    ),
    "schemas/q2_feature_dictionary.md": (
        "c0cbe0e2ea0013c140df8b29e7df4ebbbbb9545f5f6e98b0c14a74b5feeed650"
    ),
    "submission_templates/processor_d_example.json": (
        "03c95f761a732971a74f059209b35355e7b7ce3ea2fdde5d67f7bf1ce1ce0421"
    ),
    "submission_templates/results_template.md": (
        "ffbccf7c5f7546ec2c1d58487dab7f6de8d5ed434a29975d55a243586986d013"
    ),
    "tools/validate_predictions.py": (
        "d1947b859590916470022ca2daaa2a7725a7f7f267c2683a5a26cfcc867ce940"
    ),
}


def read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    """Read a CSV exactly as text so checks do not coerce IDs or labels."""

    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as error:
            raise AssertionError(f"{path.relative_to(ROOT)} is empty") from error
        return header, list(reader)


def assert_finite_features(
    testcase: unittest.TestCase,
    rows: list[list[str]],
    *,
    first_feature_index: int,
    expected_width: int,
    source: str,
) -> None:
    """Assert that all 50 feature fields are present, numeric, and finite."""

    feature_stop = first_feature_index + len(FEATURE_COLUMNS)
    for row_number, row in enumerate(rows, start=2):
        testcase.assertEqual(
            len(row),
            expected_width,
            f"{source} row {row_number} has {len(row)} fields",
        )
        for column, raw_value in zip(
            FEATURE_COLUMNS,
            row[first_feature_index:feature_stop],
            strict=True,
        ):
            try:
                value = float(raw_value)
            except ValueError as error:
                raise AssertionError(
                    f"{source} row {row_number} {column} is not numeric"
                ) from error
            testcase.assertTrue(
                math.isfinite(value),
                f"{source} row {row_number} {column} is not finite",
            )


def load_module(name: str, path: Path) -> ModuleType:
    """Load an official script by path without requiring package markers."""

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def function_node(tree: ast.AST, name: str) -> ast.FunctionDef:
    """Return one top-level function from a parsed Python module."""

    matches = [
        node
        for node in getattr(tree, "body", [])
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(matches) != 1:
        raise AssertionError(f"expected one function named {name}, found {len(matches)}")
    return matches[0]


def calls_named(tree: ast.AST, name: str) -> list[ast.Call]:
    """Return direct calls to a function with the supplied identifier."""

    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == name
    ]


class OfficialKitIntegrityTests(unittest.TestCase):
    def test_official_files_match_kit_v3(self) -> None:
        for relative_path, expected_digest in OFFICIAL_SHA256.items():
            with self.subTest(path=relative_path):
                path = ROOT / relative_path
                self.assertTrue(path.is_file(), f"missing official file: {relative_path}")
                actual_digest = hashlib.sha256(path.read_bytes()).hexdigest()
                self.assertEqual(
                    actual_digest,
                    expected_digest,
                    f"official kit-v3 file was modified: {relative_path}",
                )

    def test_root_requirements_are_the_official_pins(self) -> None:
        self.assertEqual(
            (ROOT / "requirements.txt").read_bytes(),
            (ROOT / "qml-requirements.txt").read_bytes(),
            "requirements.txt must remain an exact copy of the official QML pins",
        )


class Q2ArtifactTests(unittest.TestCase):
    def test_training_data_shape_labels_and_values(self) -> None:
        header, rows = read_csv(TRAIN_PATH)
        self.assertEqual(header, [*FEATURE_COLUMNS, "label"])
        self.assertEqual(len(rows), 800)
        self.assertEqual({row[-1] for row in rows}, {"A", "B"})
        assert_finite_features(
            self,
            rows,
            first_feature_index=0,
            expected_width=len(header),
            source=str(TRAIN_PATH.relative_to(ROOT)),
        )

    def test_test_data_shape_ids_and_values(self) -> None:
        header, rows = read_csv(TEST_PATH)
        self.assertEqual(header, ["sample_id", *FEATURE_COLUMNS])
        self.assertNotIn("label", header)
        self.assertEqual(len(rows), 200)
        self.assertEqual(
            [row[0] for row in rows],
            [f"s{index}" for index in range(200)],
        )
        assert_finite_features(
            self,
            rows,
            first_feature_index=1,
            expected_width=len(header),
            source=str(TEST_PATH.relative_to(ROOT)),
        )

    def test_prediction_ids_and_labels(self) -> None:
        prediction_header, predictions = read_csv(PREDICTIONS_PATH)
        _, test_rows = read_csv(TEST_PATH)
        expected_ids = [row[0] for row in test_rows]

        self.assertEqual(prediction_header, ["sample_id", "label"])
        self.assertEqual(len(predictions), len(expected_ids))
        self.assertTrue(
            all(len(row) == 2 for row in predictions),
            "every prediction row must have exactly two fields",
        )
        prediction_ids = [row[0] for row in predictions]
        self.assertEqual(len(prediction_ids), len(set(prediction_ids)))
        self.assertEqual(set(prediction_ids), set(expected_ids))
        self.assertLessEqual({row[1] for row in predictions}, {"A", "B"})
        self.assertTrue({row[1] for row in predictions})


class ProcessorDTests(unittest.TestCase):
    def test_schema_saved_results_and_official_benchmark_agree(self) -> None:
        benchmark = load_module(
            "qff_qml_benchmark_submission_test",
            ROOT / "benchmark" / "qff_qml_benchmark.py",
        )
        config = json.loads(
            (ROOT / "processor_d_config.json").read_text(encoding="utf-8")
        )

        self.assertEqual(benchmark.validate_processor_d(config), [])
        measured_results = benchmark.task_b_qft3(config)
        self.assertEqual(config["benchmark_results"], measured_results)

        transcript = (ROOT / "results" / "phase2" / "processor_d_benchmark.txt").read_text(
            encoding="utf-8"
        )
        json_start = transcript.find("{")
        self.assertNotEqual(json_start, -1, "saved Processor D benchmark has no JSON")
        self.assertEqual(json.loads(transcript[json_start:]), measured_results)


class NotebookTests(unittest.TestCase):
    @staticmethod
    def saved_errors(notebook: dict) -> list[str]:
        errors = []
        for cell_index, cell in enumerate(notebook.get("cells", []), start=1):
            for output in cell.get("outputs", []):
                if output.get("output_type") == "error":
                    errors.append(
                        f"cell {cell_index}: {output.get('ename', '?')}: "
                        f"{output.get('evalue', '')}"
                    )
        return errors

    def test_detector_recognizes_a_saved_error(self) -> None:
        notebook = {
            "cells": [
                {
                    "cell_type": "code",
                    "outputs": [
                        {
                            "output_type": "error",
                            "ename": "ValueError",
                            "evalue": "example",
                        }
                    ],
                }
            ]
        }
        self.assertEqual(
            self.saved_errors(notebook),
            ["cell 1: ValueError: example"],
        )

    def test_committed_notebooks_have_no_saved_errors(self) -> None:
        notebooks = sorted(ROOT.glob("*.ipynb"))
        self.assertIn(ROOT / "solution.ipynb", notebooks)
        for path in notebooks:
            with self.subTest(path=path.name):
                notebook = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(
                    self.saved_errors(notebook),
                    [],
                    f"{path.name} contains a saved exception output",
                )


class StaticLeakageGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.path = ROOT / "q2_solution.py"
        cls.source = cls.path.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source, filename=str(cls.path))

    def test_no_hidden_label_file_or_network_access(self) -> None:
        forbidden_modules = {"http", "requests", "socket", "urllib"}
        imported_roots = set()
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".", 1)[0])
        self.assertFalse(
            imported_roots & forbidden_modules,
            "the submission model must not fetch external labels or data",
        )

        forbidden_file_pattern = re.compile(
            r"(?:test[_-]?(?:labels?|targets?|ground[_-]?truth)|"
            r"y[_-]?test)\.(?:csv|json|npy|npz|parquet|txt)",
            re.IGNORECASE,
        )
        suspicious_literals = [
            node.value
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and forbidden_file_pattern.search(node.value)
        ]
        self.assertEqual(suspicious_literals, [])

        forbidden_names = {"y_test", "test_labels", "test_targets", "test_ground_truth"}
        used_names = {
            node.id for node in ast.walk(self.tree) if isinstance(node, ast.Name)
        }
        self.assertFalse(used_names & forbidden_names)

    def test_only_declared_inputs_are_read(self) -> None:
        opened_paths: Counter[str] = Counter()
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name) and node.func.id == "open":
                self.fail("use of an undeclared built-in open() input")
            if not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr in {"read_bytes", "read_text"}:
                self.fail(f"undeclared file read through .{node.func.attr}()")
            if node.func.attr != "open":
                continue
            self.assertIsInstance(
                node.func.value,
                ast.Name,
                "all opened paths must be declared module constants",
            )
            path_name = node.func.value.id
            self.assertIn(path_name, {"TRAIN_PATH", "TEST_PATH", "PREDICTIONS_PATH"})
            opened_paths[path_name] += 1
            if path_name == "PREDICTIONS_PATH":
                self.assertTrue(node.args, "predictions must be opened explicitly for writing")
                self.assertIsInstance(node.args[0], ast.Constant)
                self.assertEqual(node.args[0].value, "w")
            else:
                self.assertFalse(node.args, f"{path_name} must only be opened for reading")
        self.assertEqual(
            opened_paths,
            Counter({"TRAIN_PATH": 1, "TEST_PATH": 1, "PREDICTIONS_PATH": 1}),
        )

    def test_preprocessing_statistics_are_fit_on_training_data(self) -> None:
        standardized = function_node(self.tree, "_fit_standardized")
        statistic_receivers = []
        for node in ast.walk(standardized):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"mean", "std"}
            ):
                statistic_receivers.append(ast.unparse(node.func.value))
        self.assertEqual(statistic_receivers, ["train_signal", "train_signal"])

        transformed = function_node(self.tree, "_fit_transform")
        decompositions = [
            node
            for node in ast.walk(transformed)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "svd"
        ]
        self.assertEqual(len(decompositions), 1)
        self.assertEqual(ast.unparse(decompositions[0].args[0]), "centered_train")

    def test_model_selection_and_fitting_use_training_labels_only(self) -> None:
        experiment = function_node(self.tree, "run_experiment")
        cross_validation_calls = calls_named(experiment, "_cross_validate")
        self.assertEqual(len(cross_validation_calls), 1)
        self.assertEqual(
            [ast.unparse(argument) for argument in cross_validation_calls[0].args],
            ["x_train", "y_train"],
        )

        fit_calls = calls_named(self.tree, "_fit_kernel_svm")
        self.assertTrue(fit_calls)
        self.assertEqual(
            {ast.unparse(call.args[1]) for call in fit_calls},
            {"y_train"},
            "all SVM fits must use fold/final training labels, never test labels",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
