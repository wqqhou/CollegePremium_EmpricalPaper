#!/usr/bin/env python3
"""Construct analysis variables from the cleaned NLSY97 panel.

Run after 01_clean_panel.py. The default specification reproduces the archived
exploratory 60th-percentile definition; this script never selects a cutoff by
its regression result. Outputs are data and diagnostics, not presentation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EARNINGS_YEARS = (2016, 2018, 2020)
BASE_YEAR = 2025
REQUIRED_COLUMNS = [
    "id", "birth_year", "sex", "race_ethnicity", "asvab_stored",
    "highest_degree_2015", "household_income_1996_nominal",
    "household_net_worth_1997_nominal", "custom_weight_raw", "year",
    "survey_year", "wage_nominal",
]
BASELINE_COLUMNS = [c for c in REQUIRED_COLUMNS if c not in ("year", "survey_year", "wage_nominal")]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_record(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def read_panel(path: Path) -> pd.DataFrame:
    panel = pd.read_csv(path)
    missing = sorted(set(REQUIRED_COLUMNS) - set(panel.columns))
    require(not missing, f"Clean panel lacks columns {missing}. Run 01_clean_panel.py first.")
    require(not panel.empty, "The cleaned panel is empty.")
    for column in REQUIRED_COLUMNS:
        panel[column] = pd.to_numeric(panel[column], errors="raise")
    require(np.isfinite(panel[REQUIRED_COLUMNS].to_numpy(dtype=float)).all(),
            "Clean-panel fields must be observed and finite; do not bypass stage 1 filtering.")
    require(not panel.duplicated(["id", "year"]).any(), "Duplicate respondent/year rows in clean panel.")
    for column in ("id", "birth_year", "sex", "race_ethnicity", "highest_degree_2015", "year", "survey_year"):
        require((panel[column] == np.floor(panel[column])).all(), f"{column} must contain integers.")
        panel[column] = panel[column].astype("int64")
    require((panel["id"] > 0).all(), "Respondent identifiers must be positive.")
    require(set(panel["year"]) == set(EARNINGS_YEARS),
            f"Expected earnings years {EARNINGS_YEARS}; found {sorted(panel['year'].unique())}.")
    require((panel.groupby("id")["year"].nunique() == len(EARNINGS_YEARS)).all(),
            "Every respondent must have one observation for each earnings year.")
    require((panel["survey_year"] == panel["year"] + 1).all(),
            "Survey year must be the earnings year plus one for this extract.")
    stability = panel.groupby("id")[BASELINE_COLUMNS].nunique(dropna=False)
    unstable = stability.columns[(stability > 1).any()].tolist()
    require(not unstable, f"Baseline fields/weights change within respondent: {unstable}.")
    require(panel["sex"].isin([1, 2]).all(), "Sex codes must be 1 or 2.")
    require(panel["race_ethnicity"].isin([1, 2, 3, 4]).all(), "Race/ethnicity codes must be 1 through 4.")
    require(panel["highest_degree_2015"].isin(range(8)).all(), "Highest-degree codes must be 0 through 7.")
    require(panel["asvab_stored"].between(0, 100000).all(),
            "Stored ASVAB percentile must be between 0 and 100000 before division by 1000.")
    require((panel["custom_weight_raw"] > 0).all(), "Custom weights must be strictly positive.")
    require((panel["wage_nominal"] >= 0).all(), "Annual wage-and-salary amounts cannot be negative.")
    require((panel["birth_year"] < panel["year"]).all(), "Birth years must precede earnings years.")
    for column in ("household_income_1996_nominal", "household_net_worth_1997_nominal"):
        require(not panel[column].isin([-1, -2, -3, -4, -5]).any(),
                f"Uncleaned NLS missing codes remain in {column}.")
    return panel.sort_values(["id", "year"]).reset_index(drop=True)


def annual_cpi(path: Path) -> tuple[pd.DataFrame, list[str]]:
    source = pd.read_csv(path)
    require({"observation_date", "CPIAUCNS"}.issubset(source.columns),
            "CPI input must contain observation_date and CPIAUCNS columns.")
    source["date"] = pd.to_datetime(source["observation_date"], errors="raise")
    source["CPIAUCNS"] = pd.to_numeric(source["CPIAUCNS"], errors="raise")
    require(not source["date"].isna().any(), "CPI dates cannot be missing.")
    require(not source["date"].dt.to_period("M").duplicated().any(), "CPI has duplicate monthly observations.")
    observed = source["CPIAUCNS"].dropna()
    require(np.isfinite(observed).all() and (observed > 0).all(), "Observed CPI values must be finite and positive.")
    rows, warnings = [], []
    for year in sorted({1996, 1997, *EARNINGS_YEARS, BASE_YEAR}):
        subset = source.loc[source["date"].dt.year == year]
        valid = subset.loc[subset["CPIAUCNS"].notna()]
        absent = sorted(set(range(1, 13)) - set(valid["date"].dt.month))
        if absent:
            require(year == 2025 and absent == [10],
                    f"CPI coverage is incomplete for {year}; missing months {absent}. Only the documented October 2025 gap is allowed.")
            warnings.append("The 2025 CPI mean uses 11 supplied months: October is absent; no month is imputed.")
        rows.append({"year": year, "months_observed": len(valid), "cpi": float(valid["CPIAUCNS"].mean())})
    annual = pd.DataFrame(rows)
    base_cpi = float(annual.loc[annual["year"] == BASE_YEAR, "cpi"].iloc[0])
    annual["factor_to_2025"] = base_cpi / annual["cpi"]
    return annual, warnings


def weighted_quantile(values: np.ndarray, weights: np.ndarray, percentile: float) -> float:
    """Inverse weighted empirical CDF; no interpolation between observed values."""
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    index = int(np.searchsorted(cumulative, percentile * weights.sum(), side="left"))
    require(index < len(values), "Weighted quantile could not be located.")
    return float(values[order][index])


def construct(args: argparse.Namespace) -> None:
    require(0 < args.background_percentile < 1,
            "--background-percentile must be strictly between 0 and 1 (for example, 0.60).")
    panel = read_panel(args.input)
    annual, cpi_warnings = annual_cpi(args.cpi)
    factors = annual.set_index("year")["factor_to_2025"]
    # The percentile was already normalized within birth-quarter groups by NLS.
    # Division restores three implied decimal places, not an additional age correction.
    panel["asvab_percentile"] = panel["asvab_stored"] / 1000
    panel["female"] = panel["sex"] - 1
    panel["college"] = (panel["highest_degree_2015"] >= 4).astype("int64")
    panel["age"] = panel["year"] - panel["birth_year"]
    panel["custom_weight"] = panel["custom_weight_raw"] / 100
    people = panel.drop_duplicates("id").copy()
    panel["regression_weight"] = panel["custom_weight"] / people["custom_weight"].mean()
    panel["household_income_2025"] = panel["household_income_1996_nominal"] * factors.loc[1996]
    panel["household_net_worth_2025"] = panel["household_net_worth_1997_nominal"] * factors.loc[1997]
    panel["wage_2025"] = panel["wage_nominal"] * panel["year"].map(factors)
    people = panel.drop_duplicates("id").copy()
    weights = people["regression_weight"].to_numpy()
    cutoff_rows = []
    for variable, column, year in [
        ("household_income_1996", "household_income_1996_nominal", 1996),
        ("household_net_worth_1997", "household_net_worth_1997_nominal", 1997),
    ]:
        values = people[column].to_numpy()
        cutoff = weighted_quantile(values, weights, args.background_percentile)
        below = float(np.average(values < cutoff, weights=weights))
        at_or_below = float(np.average(values <= cutoff, weights=weights))
        require(below <= args.background_percentile + 1e-12 and
                at_or_below >= args.background_percentile - 1e-12,
                "Weighted quantile failed its empirical-CDF check.")
        cutoff_rows.append({
            "variable": variable, "observed_respondents": len(people),
            "percentile": args.background_percentile,
            "method": "inverse weighted empirical CDF within selected balanced panel",
            "nominal_cutoff": cutoff, "cutoff_2025": cutoff * factors.loc[year],
            "weighted_share_strictly_below": below,
            "weighted_share_at_or_below": at_or_below,
        })
    income_cut, wealth_cut = (row["nominal_cutoff"] for row in cutoff_rows)
    panel["low_background"] = (
        (panel["household_income_1996_nominal"] < income_cut) &
        (panel["household_net_worth_1997_nominal"] < wealth_cut)
    ).astype("int64")
    require(np.isfinite(panel.select_dtypes(include="number").to_numpy()).all(),
            "Construction produced a missing or non-finite numeric variable.")
    people = panel.drop_duplicates("id")
    require(np.isclose(people["regression_weight"].mean(), 1, rtol=0, atol=1e-12),
            "Person-level regression weights do not average to one.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.diagnostics_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "analysis_panel.csv"
    panel.to_csv(output, index=False)
    annual.to_csv(args.diagnostics_dir / "cpi_conversion.csv", index=False)
    pd.DataFrame(cutoff_rows).to_csv(args.diagnostics_dir / "background_thresholds.csv", index=False)
    pd.DataFrame([{
        "matched_respondents": len(people),
        "raw_weight_min": people["custom_weight_raw"].min(),
        "raw_weight_max": people["custom_weight_raw"].max(),
        "raw_weight_mean": people["custom_weight_raw"].mean(),
        "normalized_weight_mean": people["regression_weight"].mean(),
    }]).to_csv(args.diagnostics_dir / "weight_validation.csv", index=False)
    caveats = [
        "The default 60th-percentile definition is inherited from an exploratory cutoff search; it is not a preregistered definition.",
        "Both resources must be strictly below their separate cutoffs. The comparison group is all other backgrounds, not uniformly wealthy respondents.",
        "The cleaned sample excludes valid skips and requires three observed earnings amounts. Receipt filters are absent, so skipped amounts have not been recoded as zeros.",
        "Custom weights belong to the selected sample. New sample membership requires compatible weights and renewed sample validation.",
        "ASVAB_MATH_VERBAL_SCORE_PCT already normalizes within three-month birth groups; division by 1000 restores stored units.",
        "Degree status is fixed at the 2015 interview; no BA does not mean never attended college.",
    ] + cpi_warnings
    metadata = {
        "stage": "02_construct_variables",
        "configuration": {"background_percentile": args.background_percentile,
                          "base_dollar_year": BASE_YEAR, "earnings_years": list(EARNINGS_YEARS)},
        "inputs": {"clean_panel": input_record(args.input), "cpi": input_record(args.cpi)},
        "script": input_record(Path(__file__)),
        "outputs": {"analysis_panel": input_record(output)},
        "sample": {"respondents": len(people), "person_years": len(panel)},
        "background_definition": {
            "income_cutoff_1996_nominal": income_cut,
            "net_worth_cutoff_1997_nominal": wealth_cut,
            "rule": "Both nominal baseline measures strictly below separate weighted cutoffs",
            "reference_population": "Selected panel, each respondent counted once",
            "weighted_lower_background_share": float(np.average(people["low_background"], weights=people["regression_weight"])),
        },
        "variables": {
            "female": "1 for sex code 2; 0 for sex code 1",
            "asvab_percentile": "R9829600 / 1000; age-group-normalized percentile points",
            "college": "1 for highest-degree codes 4-7 by the 2015 interview; otherwise 0",
            "age": "Earnings calendar year minus birth year (year-end age)",
            "custom_weight": "Stored custom weight / 100 (two implied decimal places)",
            "regression_weight": "Custom weight divided by the person-level sample mean",
            "household_income_2025": "1996 household income times CPI_2025 / CPI_1996",
            "household_net_worth_2025": "1997 household net worth times CPI_2025 / CPI_1997",
            "wage_2025": "Annual wages/salary/commissions/tips times CPI_2025 / CPI_earnings_year",
            "low_background": "Indicator for both baseline resources strictly below their weighted cutoffs",
        },
        "caveats": caveats,
    }
    (args.output_dir / "variable_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Constructed variables: {len(people):,} respondents; {len(panel):,} person-years.")
    print(f"Background percentile: {args.background_percentile:.1%}; nominal cutoffs: income ${income_cut:,.0f}, net worth ${wealth_cut:,.0f}.")
    print("ASVAB is already age-group normalized; /1000 only restores its stored units.")
    print("Note: inherited cutoff definition is exploratory; excluded earnings valid skips remain unresolved.")
    for message in cpi_warnings:
        print(f"Note: {message}")
    print(f"Analysis panel: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=ROOT / "data/processed/clean_panel.csv", help="Cleaned long panel from stage 1.")
    parser.add_argument("--cpi", type=Path, default=ROOT / "data/raw/CPIAUCNS.csv", help="Monthly CPIAUCNS CSV.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed", help="Directory for analysis_panel.csv and variable_metadata.json.")
    parser.add_argument("--diagnostics-dir", type=Path, default=ROOT / "outputs/diagnostics", help="Directory for CPI, cutoff, and weight diagnostics.")
    parser.add_argument("--background-percentile", type=float, default=0.60, help="Prespecified marginal weighted percentile (0-1); default 0.60 reproduces the archived exploratory definition.")
    args = parser.parse_args()
    try:
        construct(args)
    except (ValueError, FileNotFoundError, KeyError) as error:
        parser.exit(1, f"Variable construction failed: {error}\n")


if __name__ == "__main__":
    main()
