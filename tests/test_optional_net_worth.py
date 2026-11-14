"""Checks that default eligibility ignores wealth without changing other inputs."""

from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
WEALTH = "household_net_worth_1997_nominal"


class OptionalNetWorthSelection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="e4331-optional-wealth-tests-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.work = Path(cls.temporary.name)
        cls.optional_path = cls.work / "optional"
        cls.run_clean(cls.optional_path)
        cls.optional = pd.read_csv(cls.optional_path / "weight_request/selected_panel_unweighted.csv")

    @classmethod
    def run_clean(cls, output, *arguments, expect_success=True):
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts/01_clean_panel.py"),
             "--prepare-weights", "--output-dir", str(output),
             "--diagnostics-dir", str(output / "diagnostics"), *map(str, arguments)],
            cwd=tempfile.gettempdir(), capture_output=True, text=True)
        if expect_success and result.returncode:
            raise AssertionError(result.stdout + result.stderr)
        return result

    def test_expansion_preserves_original_people_and_measurements(self):
        legacy_path = self.work / "legacy_required"
        self.run_clean(legacy_path, "--net-worth-policy", "required")
        original = pd.read_csv(legacy_path / "weight_request/selected_panel_unweighted.csv")
        self.assertEqual(original.id.nunique(), 2849)
        self.assertEqual(self.optional.id.nunique(), 3171)
        self.assertEqual(len(self.optional), 9513)
        retained = self.optional.loc[self.optional.id.isin(original.id)].reset_index(drop=True)
        pd.testing.assert_frame_equal(retained, original)
        added = self.optional.loc[~self.optional.id.isin(original.id)]
        self.assertEqual(added.id.nunique(), 322)
        self.assertTrue(added[WEALTH].isna().all())
        self.assertFalse(self.optional.drop(columns=WEALTH).isna().any().any())
        self.assertTrue(self.optional.groupby("id").size().eq(3).all())
        metadata = json.loads((self.optional_path / "weight_request/request_metadata.json").read_text())
        self.assertEqual(metadata["net_worth_policy"], "optional")
        self.assertEqual(metadata["counts"]["selected_respondents_missing_net_worth"], 322)

    def test_optional_selection_is_invariant_to_removing_all_wealth_values(self):
        altered = pd.read_csv(ROOT / "data/raw/nlsy97.csv")
        altered["R1204700"] = -1
        source = self.work / "all_wealth_missing.csv"
        altered.to_csv(source, index=False)
        output = self.work / "all_wealth_missing"
        self.run_clean(output, "--raw-data", source, "--net-worth-policy", "optional")
        actual = pd.read_csv(output / "weight_request/selected_panel_unweighted.csv")
        self.assertTrue(actual[WEALTH].isna().all())
        pd.testing.assert_frame_equal(actual.drop(columns=WEALTH), self.optional.drop(columns=WEALTH))
        result = self.run_clean(self.work / "required_all_missing", "--raw-data", source,
                                "--net-worth-policy", "required",
                                expect_success=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("No respondents satisfy", result.stderr)

    def test_default_accepts_extract_without_unused_wealth_column(self):
        source = self.work / "no_wealth_column.csv"
        pd.read_csv(ROOT / "data/raw/nlsy97.csv").drop(columns="R1204700").to_csv(source, index=False)
        output = self.work / "no_wealth_column"
        self.run_clean(output, "--raw-data", source)
        actual = pd.read_csv(output / "weight_request/selected_panel_unweighted.csv")
        self.assertTrue(actual[WEALTH].isna().all())
        pd.testing.assert_frame_equal(actual.drop(columns=WEALTH), self.optional.drop(columns=WEALTH))
        rejected = self.run_clean(self.work / "required_absent_column", "--raw-data", source,
                                  "--net-worth-policy", "required", expect_success=False)
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("missing required columns: R1204700", rejected.stderr)

    def test_optional_wealth_does_not_relax_required_income(self):
        excluded_id = int(self.optional.id.iloc[0])
        altered = pd.read_csv(ROOT / "data/raw/nlsy97.csv")
        altered.loc[altered.R0000100.eq(excluded_id), "R1204500"] = -2
        source = self.work / "one_income_missing.csv"
        altered.to_csv(source, index=False)
        output = self.work / "one_income_missing"
        self.run_clean(output, "--raw-data", source, "--net-worth-policy", "optional")
        actual = pd.read_csv(output / "weight_request/selected_panel_unweighted.csv")
        self.assertEqual(set(actual.id), set(self.optional.id) - {excluded_id})


if __name__ == "__main__":
    unittest.main()
