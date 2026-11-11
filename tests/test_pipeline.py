"""Integration checks against the archived R analysis, not fitted toy data."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def run_stage(filename, *arguments, expect_success=True):
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / filename),
                             *map(str, arguments)],
                            cwd=tempfile.gettempdir(), capture_output=True, text=True)
    if expect_success and result.returncode:
        raise AssertionError(result.stdout + result.stderr)
    return result


class ArchivedReplication(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="e4331-python-tests-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.work = Path(cls.temporary.name)
        cls.processed = cls.work / "processed"
        cls.diagnostics = cls.work / "diagnostics"
        cls.results = cls.work / "regression"
        run_stage("01_clean_panel.py", "--output-dir", cls.processed,
                  "--diagnostics-dir", cls.diagnostics)
        run_stage("02_construct_variables.py", "--input", cls.processed / "clean_panel.csv",
                  "--output-dir", cls.processed, "--diagnostics-dir", cls.diagnostics)
        run_stage("03_run_regression.py", "--input", cls.processed / "analysis_panel.csv",
                  "--output-dir", cls.results)

    def assert_results_match(self, output, archived):
        for filename in ["regression_coefficients.csv", "cluster_covariance.csv",
                         "college_premia.csv", "model_summary.csv", "completion_by_background.csv"]:
            with self.subTest(file=filename):
                actual = pd.read_csv(output / filename)
                expected = pd.read_csv(archived / filename)
                pd.testing.assert_frame_equal(actual, expected, check_dtype=False,
                                              check_exact=False, rtol=1e-9, atol=1e-5)

    def test_panel_matches_archived_people_and_measurements(self):
        actual = pd.read_csv(self.processed / "analysis_panel.csv")
        expected = pd.read_csv(ROOT / "archive/data/balanced_panel_60.csv")
        pd.testing.assert_frame_equal(actual[expected.columns], expected, check_dtype=False,
                                      check_exact=False, rtol=1e-12, atol=1e-7)
        self.assertEqual(actual.id.nunique(), 2123)
        self.assertEqual(len(actual), 6369)
        self.assertEqual(int(actual.wage_nominal.eq(0).sum()), 23)
        self.assertEqual(int(actual.drop_duplicates("id").household_net_worth_1997_nominal.lt(0).sum()), 121)
        flow = pd.read_csv(self.diagnostics / "sample_flow.csv")
        self.assertEqual(flow.respondents.tolist(), [8984, 5833, 3915, 2123, 2123])

    def test_regression_and_full_covariance_match_r(self):
        self.assert_results_match(self.results, ROOT / "archive/results_60")

    def test_changed_cutoff_reproduces_archived_alternative(self):
        processed = self.work / "p50"
        results = self.work / "regression50"
        run_stage("02_construct_variables.py", "--input", self.processed / "clean_panel.csv",
                  "--output-dir", processed, "--diagnostics-dir", self.work / "diagnostics50",
                  "--background-percentile", "0.50")
        run_stage("03_run_regression.py", "--input", processed / "analysis_panel.csv",
                  "--output-dir", results)
        expected = pd.read_csv(ROOT / "archive/data/balanced_panel_50.csv")
        actual = pd.read_csv(processed / "analysis_panel.csv")
        np.testing.assert_array_equal(actual.low_background, expected.low_background)
        self.assert_results_match(results, ROOT / "archive/results_50")

    def test_weight_mismatch_is_rejected_without_silent_selection(self):
        weights = pd.read_csv(ROOT / "data/raw/custom_weights.dat", sep=r"\s+", header=None)
        invalid = self.work / "missing_weight.dat"
        weights.iloc[1:].to_csv(invalid, sep=" ", header=False, index=False)
        result = run_stage("01_clean_panel.py", "--weights", invalid,
                           "--output-dir", self.work / "bad_weights",
                           "--diagnostics-dir", self.work / "bad_weight_diagnostics",
                           expect_success=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("do not match the selected sample", result.stderr)
        self.assertFalse((self.work / "bad_weights/clean_panel.csv").exists())

    def test_modified_panel_requires_updated_metadata(self):
        altered = self.work / "altered.csv"
        data = pd.read_csv(self.processed / "analysis_panel.csv")
        data.loc[0, "wage_2025"] += 100
        data.to_csv(altered, index=False)
        result = run_stage("03_run_regression.py", "--input", altered,
                           "--metadata", self.processed / "variable_metadata.json",
                           "--output-dir", self.work / "stale_results", expect_success=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("does not match its variable metadata", result.stderr)
        self.assertFalse((self.work / "stale_results").exists())


if __name__ == "__main__":
    unittest.main()
