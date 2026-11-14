#!/usr/bin/env python3
"""Construct analysis variables from the cleaned NLSY97 panel.

Run after 01_clean_panel.py. The default defines lower income by the weighted
bottom quintile of 1996 household income and creates five education categories.
This is an adaptation of the literature's parental-income quintiles, not an
exact replication of national tax-data ranks. Outputs are data and diagnostics.
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
EDUCATION_CODES = {0: 0, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 4, 7: 4}
EDUCATION_LABELS = {
    0: "No high-school diploma or GED only", 1: "High-school diploma",
    2: "Associate degree", 3: "Bachelor's degree only",
    4: "Graduate or professional degree",
}
BACKGROUND_SOURCE = {
    "citation": "Chetty, Raj, David J. Deming, and John N. Friedman (2026). Diversifying Society's Leaders? The Determinants and Causal Effects of Admission to Highly Selective Private Colleges. Quarterly Journal of Economics 141(1): 51-145.",
    "doi": "https://doi.org/10.1093/qje/qjaf050",
    "author_version": "https://opportunityinsights.org/wp-content/uploads/2023/07/CollegeAdmissions_Paper.pdf",
    "location": "Section 6.1, p. 43, and Table V, row 7, in the August 2025 author version",
    "use_in_source": "The need-affirmative admissions counterfactual gives the bottom parental-income quintile the largest boost, following Chetty et al. (2020). This is not the source paper's earnings-regression specification.",
    "source_income_measure": "Parental household income averaged over child ages 12-17, ranked against parents of children in the same national birth cohort; negative parental income is excluded.",
    "adaptation": "Single-year 1996 household income in the selected NLSY97 panel, pooling birth years 1980-1984 with custom person weights. Genuine negative baseline income is retained. These are sample-relative ranks, not national birth-cohort ranks.",
    "foundational_reference": "Chetty, Friedman, Saez, Turner, and Yagan (2020). Income Segregation and Intergenerational Mobility Across Colleges in the United States. Quarterly Journal of Economics 135(3): 1567-1633. https://doi.org/10.1093/qje/qjaa005",
}
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


def read_panel(path: Path, allow_missing_net_worth: bool = False) -> pd.DataFrame:
    panel = pd.read_csv(path)
    missing = sorted(set(REQUIRED_COLUMNS) - set(panel.columns))
    require(not missing, f"Clean panel lacks columns {missing}. Run 01_clean_panel.py first.")
    require(not panel.empty, "The cleaned panel is empty.")
    for column in REQUIRED_COLUMNS:
        panel[column] = pd.to_numeric(panel[column], errors="raise")
    complete_columns = [column for column in REQUIRED_COLUMNS
                        if not (allow_missing_net_worth and
                                column == "household_net_worth_1997_nominal")]
    require(np.isfinite(panel[complete_columns].to_numpy(dtype=float)).all(),
            "Clean-panel fields must be observed and finite; do not bypass stage 1 filtering.")
    require(np.isfinite(panel["household_net_worth_1997_nominal"].dropna()).all(),
            "Observed net worth must be finite.")
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
    if args.background_percentile is None:
        args.background_percentile = 0.20 if args.background_definition == "income" else 0.60
    require(0 < args.background_percentile < 1,
            "--background-percentile must be strictly between 0 and 1 (for example, 0.60).")
    cleaning_metadata_path = args.input.parent / "cleaning_metadata.json"
    cleaning_metadata = {}
    if cleaning_metadata_path.is_file():
        cleaning_metadata = json.loads(cleaning_metadata_path.read_text())
        require(cleaning_metadata.get("output", {}).get("sha256") == sha256(args.input),
                "Clean panel does not match its cleaning metadata. Rerun stage 1.")
        require(not cleaning_metadata.get("weights_pending", False),
                "Custom weights are still pending. Finish stage 1 with the official weights.")
    earnings_policy = cleaning_metadata.get("earnings_policy", "unspecified")
    net_worth_policy = cleaning_metadata.get("net_worth_policy", "required")
    require(net_worth_policy in ("required", "optional"),
            "Unknown net-worth selection policy in cleaning metadata.")
    panel = read_panel(args.input, allow_missing_net_worth=net_worth_policy == "optional")
    if args.background_definition == "joint-legacy":
        require(panel["household_net_worth_1997_nominal"].notna().all(),
                "The joint income/wealth definition requires observed net worth for every respondent. Use the income-only definition for the expanded sample.")
    annual, cpi_warnings = annual_cpi(args.cpi)
    factors = annual.set_index("year")["factor_to_2025"]
    # The percentile was already normalized within birth-quarter groups by NLS.
    # Division restores three implied decimal places, not an additional age correction.
    panel["asvab_percentile"] = panel["asvab_stored"] / 1000
    panel["female"] = panel["sex"] - 1
    panel["college"] = (panel["highest_degree_2015"] >= 4).astype("int64")
    panel["education_category"] = panel["highest_degree_2015"].map(EDUCATION_CODES).astype("int64")
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
    resources = [
        ("household_income_1996", "household_income_1996_nominal", 1996),
        ("household_net_worth_1997", "household_net_worth_1997_nominal", 1997),
    ]
    if args.background_definition == "income":
        resources = resources[:1]
    for variable, column, year in resources:
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
    income_cut = cutoff_rows[0]["nominal_cutoff"]
    wealth_cut = None
    if args.background_definition == "income":
        # Include all ties at the cutoff: do not split identical family incomes.
        panel["low_background"] = (panel["household_income_1996_nominal"] <= income_cut).astype("int64")
        background_rule = "1996 household income at or below its weighted percentile cutoff; ties kept together"
        definition_caveats = [
            "The 20th-percentile income-only default is motivated by the literature, selected before estimating this revised model. It is not preregistered; earlier joint income/wealth cutoffs were explored.",
            "The cutoff is sample-relative, based on one year of household income; it does not reproduce national birth-cohort ranks of multiyear parental income.",
            "The comparison group is all households above the income cutoff, not a uniformly wealthy group. Net worth does not determine the indicator.",
            ("Observed baseline net worth is not required for sample eligibility. Missing wealth is retained as missing and does not enter the background indicator or regression."
             if net_worth_policy == "optional" else
             "The existing panel still requires observed baseline net worth for comparability with earlier analyses; this revision does not broaden the sample."),
        ]
    else:
        wealth_cut = cutoff_rows[1]["nominal_cutoff"]
        panel["low_background"] = (
            (panel["household_income_1996_nominal"] < income_cut) &
            (panel["household_net_worth_1997_nominal"] < wealth_cut)
        ).astype("int64")
        background_rule = "Both nominal baseline measures strictly below separate weighted cutoffs"
        definition_caveats = [
            "The legacy joint 60th-percentile definition was selected after an exploratory cutoff search; it is not a preregistered definition.",
            "Both resources must be strictly below separate cutoffs. The comparison group is all other backgrounds, not uniformly wealthy respondents.",
        ]
    numeric = panel.select_dtypes(include="number")
    if net_worth_policy == "optional":
        numeric = numeric.drop(columns=["household_net_worth_1997_nominal", "household_net_worth_2025"])
        require(np.isfinite(panel["household_net_worth_2025"].dropna()).all(),
                "Inflation adjustment produced non-finite observed net worth.")
    require(np.isfinite(numeric.to_numpy()).all(),
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
    caveats = definition_caveats + [
        ("The sample includes screening-confirmed zero wage amounts and requires usable wage amounts in all three years. Unresolved skips and other missing amounts remain excluded. Zero wages do not imply zero self-employment income or nonemployment."
         if earnings_policy == "receipt-confirmed" else
         "Check the cleaning-stage policy for zero and missing wage amounts; the observed-only replication excludes all skipped amounts."),
        "Custom weights belong to the selected sample. New sample membership requires compatible weights and renewed sample validation.",
        "ASVAB_MATH_VERBAL_SCORE_PCT already normalizes within three-month birth groups; division by 1000 restores stored units.",
        "Highest degree is fixed at the 2015 interview. High-school diploma includes some college without a higher completed degree; it does not identify never-attenders.",
        "Codes 0-1 combine no high-school diploma and GED only; codes 5-7 combine master's, doctoral and professional degrees. These within-category differences remain pooled.",
    ] + cpi_warnings
    metadata = {
        "stage": "02_construct_variables",
        "configuration": {"background_definition": args.background_definition,
                          "background_percentile": args.background_percentile,
                          "base_dollar_year": BASE_YEAR, "earnings_years": list(EARNINGS_YEARS),
                          "earnings_policy": earnings_policy,
                          "net_worth_policy": net_worth_policy},
        "inputs": {"clean_panel": input_record(args.input), "cpi": input_record(args.cpi)},
        "script": input_record(Path(__file__)),
        "outputs": {"analysis_panel": input_record(output)},
        "sample": {"respondents": len(people), "person_years": len(panel)},
        "cleaning_counts": cleaning_metadata.get("counts", {}),
        "background_definition": {
            "income_cutoff_1996_nominal": income_cut,
            "net_worth_cutoff_1997_nominal": wealth_cut,
            "rule": background_rule,
            "reference_population": "Selected panel, birth years 1980-1984 pooled, each respondent counted once",
            "weighted_lower_background_share": float(np.average(people["low_background"], weights=people["regression_weight"])),
            "lower_background_respondents": int(people["low_background"].sum()),
            "negative_baseline_income_respondents_retained": int((people["household_income_1996_nominal"] < 0).sum()),
            "literature_motivation": BACKGROUND_SOURCE if args.background_definition == "income" and np.isclose(args.background_percentile, 0.20) else None,
        },
        "education_definition": {"source": "U0009400: highest degree by 2015 interview",
                                 "source_code_mapping": EDUCATION_CODES,
                                 "labels": EDUCATION_LABELS,
                                 "reference_category": 1},
        "variables": {
            "female": "1 for sex code 2; 0 for sex code 1",
            "asvab_percentile": "R9829600 / 1000; age-group-normalized percentile points",
            "college": "1 for highest-degree codes 4-7 by the 2015 interview; otherwise 0",
            "education_category": "0=no diploma/GED; 1=high-school diploma; 2=associate; 3=bachelor only; 4=graduate/professional",
            "age": "Earnings calendar year minus birth year (year-end age)",
            "custom_weight": "Stored custom weight / 100 (two implied decimal places)",
            "regression_weight": "Custom weight divided by the person-level sample mean",
            "household_income_2025": "1996 household income times CPI_2025 / CPI_1996",
            "household_net_worth_2025": "1997 household net worth times CPI_2025 / CPI_1997",
            "wage_2025": "Annual wages/salary/commissions/tips times CPI_2025 / CPI_earnings_year",
            "low_background": background_rule,
        },
        "caveats": caveats,
    }
    (args.output_dir / "variable_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Constructed variables: {len(people):,} respondents; {len(panel):,} person-years.")
    print(f"Background definition: {args.background_definition}; percentile: {args.background_percentile:.1%}; nominal income cutoff ${income_cut:,.0f}.")
    if wealth_cut is not None:
        print(f"Legacy nominal net-worth cutoff: ${wealth_cut:,.0f}.")
    print("ASVAB is already age-group normalized; /1000 only restores its stored units.")
    print(f"Earnings policy: {earnings_policy}. Background ranks are relative to the selected sample.")
    for message in cpi_warnings:
        print(f"Note: {message}")
    print(f"Analysis panel: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=ROOT / "data/processed/clean_panel.csv", help="Cleaned long panel from stage 1.")
    parser.add_argument("--cpi", type=Path, default=ROOT / "data/raw/CPIAUCNS.csv", help="Monthly CPIAUCNS CSV.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed", help="Directory for analysis_panel.csv and variable_metadata.json.")
    parser.add_argument("--diagnostics-dir", type=Path, default=ROOT / "outputs/diagnostics", help="Directory for CPI, cutoff, and weight diagnostics.")
    parser.add_argument("--background-definition", choices=["income", "joint-legacy"], default="income", help="Default: income-only. joint-legacy preserves the archived income AND wealth definition.")
    parser.add_argument("--background-percentile", type=float, default=None, help="Weighted percentile (0-1). Defaults to 0.20 for income-only, or 0.60 for joint-legacy; never selected by regression results.")
    args = parser.parse_args()
    try:
        construct(args)
    except (ValueError, FileNotFoundError, KeyError) as error:
        parser.exit(1, f"Variable construction failed: {error}\n")


if __name__ == "__main__":
    main()
