"""Stage-2 checks for retaining missing wealth without relaxing model inputs."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
WEALTH = "household_net_worth_1997_nominal"


class OptionalNetWorthVariables(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="e4331-optional-wealth-vars-")
        self.addCleanup(self.temporary.cleanup)
        self.work = Path(self.temporary.name)
        self.input = self.work / "clean_panel.csv"
        self.output = self.work / "constructed"
        self.original = pd.read_csv(ROOT / "data/processed/clean_panel.csv")
        self.frame = self.original.copy()
        self.missing_id = int(self.frame.id.iloc[0])
        self.frame.loc[self.frame.id.eq(self.missing_id), WEALTH] = np.nan
        self.metadata = json.loads((ROOT / "data/processed/cleaning_metadata.json").read_text())
        self.metadata["net_worth_policy"] = "optional"
        self.write_input()

    def write_input(self):
        self.frame.to_csv(self.input, index=False)
        self.metadata["output"]["path"] = str(self.input)
        self.metadata["output"]["sha256"] = hashlib.sha256(self.input.read_bytes()).hexdigest()
        (self.work / "cleaning_metadata.json").write_text(json.dumps(self.metadata))

    def run_stage(self, *arguments):
        return subprocess.run(
            [sys.executable, str(ROOT / "scripts/02_construct_variables.py"),
             "--input", str(self.input), "--output-dir", str(self.output),
             "--diagnostics-dir", str(self.work / "diagnostics"), *arguments],
            capture_output=True, text=True, cwd=tempfile.gettempdir(),
        )

    def test_income_only_keeps_missing_wealth_without_changing_model_variables(self):
        result = self.run_stage()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        actual = pd.read_csv(self.output / "analysis_panel.csv")
        expected = pd.read_csv(ROOT / "data/processed/analysis_panel.csv")
        self.assertEqual(set(actual.id), set(self.original.id))
        self.assertEqual(len(actual), len(self.original))
        for column in (WEALTH, "household_net_worth_2025"):
            pd.testing.assert_series_equal(
                actual[column].isna(), self.original[WEALTH].isna() | actual.id.eq(self.missing_id), check_names=False,
            )
        other_columns = [c for c in expected if c not in (WEALTH, "household_net_worth_2025")]
        pd.testing.assert_frame_equal(actual[other_columns], expected[other_columns],
                                      check_dtype=False, check_exact=False,
                                      rtol=1e-12, atol=1e-7)
        metadata = json.loads((self.output / "variable_metadata.json").read_text())
        self.assertEqual(metadata["configuration"]["net_worth_policy"], "optional")

    def test_joint_background_definition_rejects_missing_wealth(self):
        result = self.run_stage("--background-definition", "joint-legacy")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires observed net worth for every respondent", result.stderr)
        self.assertFalse((self.output / "analysis_panel.csv").exists())

    def test_optional_wealth_does_not_allow_missing_model_inputs(self):
        for column in ("household_income_1996_nominal", "asvab_stored", "wage_nominal"):
            with self.subTest(column=column):
                self.frame = self.original.copy()
                self.frame.loc[self.frame.id.eq(self.missing_id), [WEALTH, column]] = np.nan
                self.write_input()
                result = self.run_stage()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("fields must be observed and finite", result.stderr)
                self.assertFalse((self.output / "analysis_panel.csv").exists())

    def test_wealth_missingness_requires_explicit_upstream_policy(self):
        self.metadata.pop("net_worth_policy")
        self.write_input()
        result = self.run_stage()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("fields must be observed and finite", result.stderr)


if __name__ == "__main__":
    unittest.main()
