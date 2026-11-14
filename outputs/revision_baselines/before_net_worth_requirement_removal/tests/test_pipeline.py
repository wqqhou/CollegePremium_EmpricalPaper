"""Integration checks against the archived R analysis, not fitted toy data."""

from pathlib import Path
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy.stats import f as fisher_f


ROOT = Path(__file__).resolve().parents[1]
ARCHIVED_RAW = ROOT / "archive/data/default/default.csv"
ARCHIVED_WEIGHTS = ROOT / "archive/data/customweight_nlsy97_6ad2cf591a14779f5b2/customweight_nlsy97_6ad2cf591a14779f5b2.dat"


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
        run_stage("01_clean_panel.py", "--raw-data", ARCHIVED_RAW, "--weights", ARCHIVED_WEIGHTS,
                  "--earnings-policy", "observed-only", "--output-dir", cls.processed,
                  "--diagnostics-dir", cls.diagnostics)
        run_stage("02_construct_variables.py", "--input", cls.processed / "clean_panel.csv",
                  "--background-definition", "joint-legacy",
                  "--output-dir", cls.processed, "--diagnostics-dir", cls.diagnostics)
        run_stage("03_run_regression.py", "--input", cls.processed / "analysis_panel.csv",
                  "--education-model", "binary",
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
                  "--background-definition", "joint-legacy",
                  "--output-dir", processed, "--diagnostics-dir", self.work / "diagnostics50",
                  "--background-percentile", "0.50")
        run_stage("03_run_regression.py", "--input", processed / "analysis_panel.csv",
                  "--education-model", "binary",
                  "--output-dir", results)
        expected = pd.read_csv(ROOT / "archive/data/balanced_panel_50.csv")
        actual = pd.read_csv(processed / "analysis_panel.csv")
        np.testing.assert_array_equal(actual.low_background, expected.low_background)
        self.assert_results_match(results, ROOT / "archive/results_50")

    def test_weight_mismatch_is_rejected_without_silent_selection(self):
        weights = pd.read_csv(ARCHIVED_WEIGHTS, sep=r"\s+", header=None)
        invalid = self.work / "missing_weight.dat"
        weights.iloc[1:].to_csv(invalid, sep=" ", header=False, index=False)
        result = run_stage("01_clean_panel.py", "--raw-data", ARCHIVED_RAW,
                           "--earnings-policy", "observed-only", "--weights", invalid,
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


class IncomeQuintileEducationModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="e4331-education-tests-")
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
        cls.panel = pd.read_csv(cls.processed / "analysis_panel.csv")
        cls.people = cls.panel.drop_duplicates("id")
        cls.premia = pd.read_csv(cls.results / "education_premia.csv")
        cls.coefficients = pd.read_csv(cls.results / "regression_coefficients.csv").set_index("term")
        cls.covariance = pd.read_csv(cls.results / "cluster_covariance.csv", index_col=0)

    def test_income_cutoff_and_degree_cells_use_people_once(self):
        metadata = json.loads((self.processed / "variable_metadata.json").read_text())
        self.assertEqual(metadata["configuration"]["background_definition"], "income")
        self.assertEqual(metadata["configuration"]["background_percentile"], 0.20)
        self.assertEqual(metadata["background_definition"]["income_cutoff_1996_nominal"], 21360)
        self.assertIsNone(metadata["background_definition"]["net_worth_cutoff_1997_nominal"])
        self.assertIn("qjaf050", metadata["background_definition"]["literature_motivation"]["doi"])
        people = self.people
        np.testing.assert_array_equal(people.low_background, people.household_income_1996_nominal.le(21360))
        self.assertEqual(people.low_background.value_counts().to_dict(), {0: 2084, 1: 765})
        weights = people.regression_weight
        self.assertLess(np.average(people.household_income_1996_nominal.lt(21360), weights=weights), 0.20)
        self.assertAlmostEqual(np.average(people.low_background, weights=weights), 0.20039039968995173)
        self.assertTrue(people.loc[people.household_income_1996_nominal.eq(21360), "low_background"].eq(1).all())
        expected = np.array([[231, 236], [802, 354], [187, 70], [583, 80], [281, 25]])
        cells = pd.crosstab(people.education_category, people.low_background)
        np.testing.assert_array_equal(cells.to_numpy(), expected)
        exported = pd.read_csv(self.results / "education_by_background.csv")
        np.testing.assert_array_equal(exported.pivot(index="education_category", columns="low_background", values="respondents").to_numpy(), expected)
        self.assertEqual(exported.respondents.sum(), 2849)
        np.testing.assert_allclose(exported.groupby("low_background").weighted_share_within_background.sum(), [1, 1])

    def test_contrasts_match_independent_cell_intercept_parameterization(self):
        # Ten cell intercepts are an equivalent basis with no omitted education
        # category. Their contrasts independently check that HS, not all no-BA
        # respondents, is the exported reference in each background group.
        p = self.panel
        cell_keys = [(education, background) for education in range(5) for background in (0, 1)]
        cells = [(p.education_category.eq(e) & p.low_background.eq(b)).to_numpy(dtype=float)
                 for e, b in cell_keys]
        controls = [p.asvab_percentile.to_numpy(), p.female.to_numpy(), p.age.to_numpy()]
        controls += [p.race_ethnicity.eq(r).to_numpy(dtype=float) for r in (1, 2, 3)]
        controls += [p.year.eq(y).to_numpy(dtype=float) for y in (2018, 2020)]
        design = np.column_stack(cells + controls)
        root_weight = np.sqrt(p.regression_weight.to_numpy())
        coefficients, _, rank, _ = np.linalg.lstsq(design * root_weight[:, None],
                                                 p.wage_2025.to_numpy() * root_weight, rcond=None)
        self.assertEqual(rank, 18)
        self.assertEqual(len(self.premia), 12)
        self.assertTrue(self.premia.reference_education.eq("high_school").all())
        labels = {0: "no_diploma_or_ged", 2: "associate", 3: "bachelor", 4: "graduate"}
        for code, label in labels.items():
            with self.subTest(education=label):
                other = coefficients[cell_keys.index((code, 0))] - coefficients[cell_keys.index((1, 0))]
                lower = coefficients[cell_keys.index((code, 1))] - coefficients[cell_keys.index((1, 1))]
                rows = self.premia.loc[self.premia.education.eq(label)].set_index("contrast")
                np.testing.assert_allclose(rows.loc[["Other backgrounds (L=0)", "Lower background (L=1)",
                                                    "Difference (L=1 minus L=0)"], "estimate"],
                                           [other, lower, lower - other], rtol=1e-9, atol=1e-6)

    def test_premium_uncertainty_and_joint_test_use_full_covariance(self):
        covariance = self.covariance
        interactions = []
        for label in ["no_diploma_or_ged", "associate", "bachelor", "graduate"]:
            main = f"education[{label}]"
            interaction = f"{main}:low_background"
            interactions.append(interaction)
            rows = self.premia.loc[self.premia.education.eq(label)].set_index("contrast")
            variances = [covariance.loc[main, main],
                         covariance.loc[main, main] + covariance.loc[interaction, interaction]
                         + 2 * covariance.loc[main, interaction],
                         covariance.loc[interaction, interaction]]
            np.testing.assert_allclose(rows.loc[["Other backgrounds (L=0)", "Lower background (L=1)",
                                                "Difference (L=1 minus L=0)"], "cluster_se"] ** 2,
                                       variances, rtol=1e-10)
        beta = self.coefficients.loc[interactions, "estimate"].to_numpy()
        wald = beta @ np.linalg.solve(covariance.loc[interactions, interactions].to_numpy(), beta)
        joint = pd.read_csv(self.results / "joint_interaction_test.csv").iloc[0]
        self.assertEqual(joint.df_numerator, 4)
        self.assertEqual(joint.df_denominator, 2848)
        self.assertAlmostEqual(joint.wald_statistic, wald)
        self.assertAlmostEqual(joint.f_statistic, wald / 4)
        self.assertAlmostEqual(joint.p_value, fisher_f.sf(wald / 4, 4, 2848))
        self.assertFalse((self.results / "college_premia.csv").exists())


class ReceiptConfirmedSample(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="e4331-receipt-tests-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.work = Path(cls.temporary.name)
        spec = importlib.util.spec_from_file_location("cleaning", ROOT / "scripts/01_clean_panel.py")
        cls.cleaning = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.cleaning)

    def test_new_extract_keeps_all_original_fields(self):
        raw = pd.read_csv(ROOT / "data/raw/nlsy97.csv").sort_values("R0000100").reset_index(drop=True)
        original = pd.read_csv(ARCHIVED_RAW).sort_values("R0000100").reset_index(drop=True)
        self.assertEqual(raw.shape, (8984, 18))
        self.assertTrue(raw.R0000100.is_unique)
        pd.testing.assert_frame_equal(raw[original.columns], original)
        for name in ["U2857000", "U2857100", "U4282100", "U4282200", "U5753300", "U5753400"]:
            self.assertTrue(raw[name].isin([-5, -4, -3, -2, -1, 0, 1]).all())

    def test_only_documented_no_income_routes_recover_zero(self):
        amount = pd.Series([100, 0, -4, -4, -4, -4, -4, -4, -1, -2, -3, -5])
        receipt = pd.Series([1, 1, 0, -1, -2, -1, -2, -4, 1, 1, 1, -5])
        fallback = pd.Series([-4, -4, -4, 0, 0, -1, -2, -4, -4, -4, -4, -5])
        values, sources = self.cleaning.resolve_earnings(amount, receipt, fallback)
        np.testing.assert_allclose(values.to_numpy(), [100, 0, 0, 0, 0,
                                  np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan], equal_nan=True)
        self.assertEqual(sources.str.startswith("confirmed_zero").sum(), 3)
        with self.assertRaisesRegex(ValueError, "positive wage amount"):
            self.cleaning.resolve_earnings(pd.Series([10]), pd.Series([0]), pd.Series([-4]))

    def test_prepare_ids_preserves_weighted_panel_and_recovers_expected_people(self):
        output = self.work / "prepared"
        output.mkdir()
        sentinel = output / "clean_panel.csv"
        sentinel.write_text("Existing weighted panel must remain unchanged.\n")
        run_stage("01_clean_panel.py", "--prepare-weights", "--output-dir", output,
                  "--diagnostics-dir", self.work / "prepare_diagnostics")
        self.assertEqual(sentinel.read_text(), "Existing weighted panel must remain unchanged.\n")
        request = output / "weight_request"
        ids = set(map(int, (request / "respondent_ids.txt").read_text().split()))
        previous = pd.read_csv(ROOT / "archive/data/balanced_panel_60.csv")
        old_ids = set(previous.id)
        self.assertEqual(len(ids), 2849)
        self.assertEqual(len(ids - old_ids), 726)
        self.assertTrue(old_ids.issubset(ids))
        panel = pd.read_csv(request / "selected_panel_unweighted.csv")
        self.assertEqual(len(panel), 8547)
        self.assertEqual(set(panel.id), ids)
        self.assertEqual(panel.wage_nominal.eq(0).groupby(panel.year).sum().tolist(), [486, 475, 476])
        metadata = json.loads((request / "request_metadata.json").read_text())
        self.assertEqual(metadata["counts"]["recovered_zero_amounts_selected"], 1402)

    def test_new_sample_cannot_use_old_sample_weights(self):
        output = self.work / "old_weights"
        result = run_stage("01_clean_panel.py", "--weights", ARCHIVED_WEIGHTS,
                           "--output-dir", output, "--diagnostics-dir", self.work / "old_diagnostics",
                           expect_success=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("do not match the selected sample", result.stderr)
        self.assertFalse((output / "clean_panel.csv").exists())

    def test_official_weights_attach_and_construct_compatible_panel(self):
        output = self.work / "weighted"
        diagnostics = self.work / "weighted_diagnostics"
        run_stage("01_clean_panel.py", "--output-dir", output, "--diagnostics-dir", diagnostics)
        clean = pd.read_csv(output / "clean_panel.csv")
        self.assertEqual(clean.id.nunique(), 2849)
        self.assertTrue(clean.custom_weight_raw.gt(0).all())
        self.assertTrue(clean.groupby("id").custom_weight_raw.nunique().eq(1).all())
        run_stage("02_construct_variables.py", "--input", output / "clean_panel.csv",
                  "--output-dir", output, "--diagnostics-dir", diagnostics)
        variables = json.loads((output / "variable_metadata.json").read_text())
        self.assertEqual(variables["configuration"]["earnings_policy"], "receipt-confirmed")
        self.assertEqual(variables["sample"]["person_years"], 8547)

    def test_changed_clean_panel_requires_variable_rebuild(self):
        output = self.work / "changed_upstream"
        run_stage("01_clean_panel.py", "--output-dir", output,
                  "--diagnostics-dir", self.work / "upstream_diagnostics")
        run_stage("02_construct_variables.py", "--input", output / "clean_panel.csv",
                  "--output-dir", output, "--diagnostics-dir", self.work / "upstream_diagnostics")
        clean = output / "clean_panel.csv"
        clean.write_text(clean.read_text() + "\n")
        result = run_stage("03_run_regression.py", "--input", output / "analysis_panel.csv",
                           "--output-dir", self.work / "upstream_results", expect_success=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Clean panel has changed", result.stderr)


if __name__ == "__main__":
    unittest.main()
