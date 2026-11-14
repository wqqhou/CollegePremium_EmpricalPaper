"""Checks for period restrictions that preserve the original respondent sample."""

from pathlib import Path
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/03_run_regression.py"
PANEL = ROOT / "data/processed/analysis_panel.csv"
METADATA = ROOT / "data/processed/variable_metadata.json"
spec = importlib.util.spec_from_file_location("year_selection_regression", SCRIPT)
regression = importlib.util.module_from_spec(spec)
spec.loader.exec_module(regression)


class RegressionYearSelection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="e4331-years-tests-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.work = Path(cls.temporary.name)
        cls.output = cls.work / "exclude_2020"
        cls.panel, _ = regression.read_panel(PANEL, METADATA)
        result = cls.run_cli("--earnings-years", "2016", "2018",
                             "--output-dir", cls.output)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)

    @staticmethod
    def run_cli(*arguments):
        return subprocess.run([sys.executable, str(SCRIPT), *map(str, arguments)],
                              cwd=tempfile.gettempdir(), capture_output=True, text=True)

    def test_subset_preserves_people_weights_and_only_requested_rows(self):
        actual = regression.select_earnings_years(self.panel, [2018, 2016])
        expected = self.panel.loc[self.panel.year.ne(2020)].reset_index(drop=True)
        pd.testing.assert_frame_equal(actual, expected)
        self.assertEqual(actual.id.nunique(), 3171)
        self.assertEqual(len(actual), 6342)
        self.assertTrue(actual.groupby("id").size().eq(2).all())
        people = self.panel.drop_duplicates("id").set_index("id")
        retained = actual.drop_duplicates("id").set_index("id")
        pd.testing.assert_series_equal(retained.regression_weight, people.regression_weight)
        pd.testing.assert_series_equal(retained.low_background, people.low_background)
        pd.testing.assert_frame_equal(regression.select_earnings_years(self.panel), self.panel)

    def test_exported_metadata_and_counts_describe_actual_estimation_years(self):
        metadata = json.loads((self.output / "regression_metadata.json").read_text())
        self.assertEqual(metadata["configuration"]["source_earnings_years"], [2016, 2018, 2020])
        self.assertEqual(metadata["configuration"]["earnings_years"], [2016, 2018])
        self.assertEqual(metadata["year_selection"]["omitted_earnings_years"], [2020])
        self.assertTrue(metadata["year_selection"]["respondent_selection_unchanged"])
        self.assertTrue(metadata["year_selection"]["weights_and_baseline_variables_unchanged"])
        self.assertEqual(metadata["year_selection"]["source_observations"], 9513)
        self.assertEqual(metadata["sample"]["respondents"], 3171)
        self.assertEqual(metadata["sample"]["observations"], 6342)
        self.assertEqual(metadata["sample"]["periods"], 2)
        self.assertEqual(metadata["sample"]["parameters"], 17)
        self.assertEqual(metadata["sample"]["cluster_df"], 3170)
        self.assertEqual(metadata["reference_categories"]["year"], 2016)
        coefficients = pd.read_csv(self.output / "regression_coefficients.csv")
        self.assertIn("factor(year)2018", set(coefficients.term))
        self.assertNotIn("factor(year)2020", set(coefficients.term))

    def test_unavailable_or_duplicate_years_fail_before_outputs(self):
        for years in [(2016, 2022), (2016, 2016)]:
            with self.subTest(years=years):
                output = self.work / ("invalid_" + "_".join(map(str, years)))
                result = self.run_cli("--earnings-years", *years, "--output-dir", output)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(output.exists())
                self.assertTrue("not available" in result.stderr or "distinct" in result.stderr)

    def test_excluded_period_cannot_hide_changed_source_data(self):
        altered = self.panel.copy()
        altered.loc[altered.year.eq(2020), "wage_2025"] += 1
        path = self.work / "changed_source.csv"
        altered.to_csv(path, index=False)
        output = self.work / "stale_source"
        result = self.run_cli("--input", path, "--metadata", METADATA,
                              "--earnings-years", "2016", "2018", "--output-dir", output)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("does not match its variable metadata", result.stderr)
        self.assertFalse(output.exists())

    def test_subset_contrasts_match_independent_cell_intercept_fit(self):
        # Parameterize ten group/degree cell intercepts directly rather than
        # reproducing the stage-3 treatment-interaction design matrix.
        panel = self.panel.loc[self.panel.year.ne(2020)].copy()
        cells = [(group, degree) for group in (0, 1) for degree in range(5)]
        columns = [((panel.low_background == group) & (panel.education_category == degree)).astype(float)
                   for group, degree in cells]
        columns += [panel.asvab_percentile, panel.female, panel.age]
        columns += [panel.race_ethnicity.eq(code).astype(float) for code in (1, 2, 3)]
        columns += [panel.year.eq(2018).astype(float)]
        x = np.column_stack(columns)
        y = panel.wage_2025.to_numpy()
        w = panel.regression_weight.to_numpy()
        n, k = x.shape
        ids, inverse = np.unique(panel.id, return_inverse=True)
        g = len(ids)
        beta = np.linalg.lstsq(x * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
        residuals = y - x @ beta
        scores = np.zeros((g, k))
        np.add.at(scores, inverse, x * (w * residuals)[:, None])
        bread = np.linalg.inv(x.T @ (w[:, None] * x))
        covariance = (g / (g - 1)) * ((n - 1) / (n - k)) * bread @ scores.T @ scores @ bread
        critical = student_t.ppf(0.975, g - 1)
        exported = pd.read_csv(self.output / "education_premia.csv")
        for degree, label in [(0, "no_diploma_or_ged"), (2, "associate"),
                              (3, "bachelor"), (4, "graduate")]:
            vectors = []
            for group in (0, 1):
                vector = np.zeros(k)
                vector[cells.index((group, degree))] = 1
                vector[cells.index((group, 1))] = -1
                vectors.append(vector)
            vectors.append(vectors[1] - vectors[0])
            observed = exported.loc[exported.education.eq(label)]
            for vector, row in zip(vectors, observed.itertuples()):
                estimate = float(vector @ beta)
                se = float(np.sqrt(vector @ covariance @ vector))
                expected = [estimate, se, 2 * student_t.sf(abs(estimate / se), g - 1),
                            estimate - critical * se, estimate + critical * se]
                actual = [row.estimate, row.cluster_se, row.p_value, row.ci_low, row.ci_high]
                np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
