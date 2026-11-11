"""Clean the NLSY97 extract and select the archived balanced panel.

Run from any directory:
    python /path/to/E4331_ER/scripts/01_clean_panel.py

This first stage keeps source measurements in their original units. Economic
variables, inflation adjustment, and weight normalization belong to stage 2.
The default selection deliberately reproduces the previous report: all three
earnings amounts must be observed. Valid skips are not assumed to mean zero.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MISSING_CODES = {-1: "Refusal", -2: "Do not know", -3: "Invalid skip",
                 -4: "Valid skip", -5: "Non-interview"}
SOURCE_COLUMNS = {
    "R0000100": "id",
    "R0536402": "birth_year",
    "R0536300": "sex",
    "R1482600": "race_ethnicity",
    "R9829600": "asvab_stored",
    "U0009400": "highest_degree_2015",
    "R1204500": "household_income_1996_nominal",
    "R1204700": "household_net_worth_1997_nominal",
}
EARNINGS = {2016: "U2857200", 2018: "U4282300", 2020: "U5753500"}
BACKGROUND = ["R1204500", "R1204700"]
COVARIATES = [column for column in SOURCE_COLUMNS if column != "R0000100"]
SKIP_WARNING = (
    "The current sample excludes earnings valid skips. The amount question can "
    "be skipped when a respondent has no wage income, but the extract lacks "
    "YINC-1400/YINC-1600 receipt screens needed to distinguish confirmed zero "
    "income from unresolved answers. Do not recode every skip to zero. "
    "Changing the selected people also requires matching custom weights."
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_codes(frame, column, allowed):
    values = frame[column].dropna()
    require(values.isin(allowed).all(),
            f"Unexpected observed codes in {column}; check the extract codebook.")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-data", type=Path, default=ROOT / "data/raw/nlsy97.csv",
                        help="Original NLSY97 CSV extract.")
    parser.add_argument("--weights", type=Path, default=ROOT / "data/raw/custom_weights.dat",
                        help="Whitespace-delimited respondent ID and custom weight file.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed",
                        help="Directory for clean_panel.csv and cleaning_metadata.json.")
    parser.add_argument("--diagnostics-dir", type=Path, default=ROOT / "outputs/diagnostics",
                        help="Directory for sample_flow.csv and earnings_missingness.csv.")
    return parser.parse_args()


def main():
    args = parse_args()
    for path in (args.raw_data, args.weights):
        require(path.is_file(), f"Input does not exist: {path}")

    raw = pd.read_csv(args.raw_data)
    required_columns = list(SOURCE_COLUMNS) + list(EARNINGS.values())
    missing = sorted(set(required_columns) - set(raw.columns))
    require(not missing, f"Raw extract is missing required columns: {', '.join(missing)}")
    raw = raw[required_columns].apply(pd.to_numeric, errors="raise")
    require(len(raw) > 0, "The raw extract is empty.")
    observed_values = raw.to_numpy(dtype=float)
    require(not np.isinf(observed_values).any(), "Raw extract contains infinite values.")
    ids = raw["R0000100"]
    require(ids.notna().all() and (ids > 0).all() and (ids % 1 == 0).all(),
            "Respondent IDs must be positive observed integers.")
    require(not ids.duplicated().any(), "Raw extract contains duplicate respondent IDs.")

    # Only these exact codes mean missing. Other negative income/wealth values
    # are economically possible and are retained.
    clean = raw.copy()
    measurements = [column for column in clean if column != "R0000100"]
    clean[measurements] = clean[measurements].mask(clean[measurements].isin(list(MISSING_CODES)))
    check_codes(clean, "R0536300", (1, 2))
    check_codes(clean, "R1482600", (1, 2, 3, 4))
    check_codes(clean, "U0009400", range(8))
    check_codes(clean, "R0536402", range(1980, 1985))
    asvab = clean["R9829600"].dropna()
    require(asvab.between(0, 100000).all(),
            "Stored ASVAB percentile must be between 0 and 100000 before rescaling.")
    for column in EARNINGS.values():
        require((clean[column].dropna() >= 0).all(),
                f"Unexpected negative earnings in {column} after removing missing codes.")

    background_ok = clean[BACKGROUND].notna().all(axis=1)
    covariates_ok = clean[COVARIATES].notna().all(axis=1)
    keep = covariates_ok & clean[list(EARNINGS.values())].notna().all(axis=1)
    require(keep.any(), "No respondents satisfy the balanced complete-case selection.")

    # These weights were generated for the specific selected sample. A changed
    # selection must not silently inherit only the overlapping old weights.
    weights = pd.read_csv(args.weights, sep=r"\s+", header=None)
    require(weights.shape[1] == 2, "Custom weights must have exactly two columns: ID and weight.")
    weights.columns = ["id", "custom_weight_raw"]
    weights = weights.apply(pd.to_numeric, errors="raise")
    require(np.isfinite(weights.to_numpy(dtype=float)).all(),
            "Custom weights contain missing or non-finite values.")
    require((weights["id"] > 0).all() and (weights["id"] % 1 == 0).all(),
            "Custom-weight IDs must be positive integers.")
    require(not weights["id"].duplicated().any(), "Custom weights contain duplicate IDs.")
    require((weights["custom_weight_raw"] > 0).all(), "Custom weights must be positive.")
    selected_ids = set(clean.loc[keep, "R0000100"].astype(int))
    weight_ids = set(weights["id"].astype(int))
    require(selected_ids == weight_ids,
            "Custom weights do not match the selected sample: "
            f"{len(selected_ids - weight_ids)} selected people lack weights and "
            f"{len(weight_ids - selected_ids)} weighted people are not selected. "
            "Obtain weights for the revised sample or restore the intended selection.")

    people = clean.loc[keep, list(SOURCE_COLUMNS)].rename(columns=SOURCE_COLUMNS)
    people = people.merge(weights, on="id", how="left", validate="one_to_one")
    integer_columns = ["id", "birth_year", "sex", "race_ethnicity", "highest_degree_2015"]
    people[integer_columns] = people[integer_columns].astype("int64")
    wage_lookup = clean.set_index("R0000100")
    years = []
    for year, column in EARNINGS.items():
        block = people.copy()
        block["year"] = year
        block["survey_year"] = year + 1
        block["wage_nominal"] = block["id"].map(wage_lookup[column])
        years.append(block)
    panel = pd.concat(years, ignore_index=True).sort_values(["id", "year"]).reset_index(drop=True)
    require(not panel.isna().any().any(), "Unexpected missing values in the selected panel.")
    require(not panel.duplicated(["id", "year"]).any(), "Duplicate person-year in panel.")
    require(panel.groupby("id").size().eq(len(EARNINGS)).all(), "Panel is not balanced.")

    counts = [len(raw), int(background_ok.sum()), int(covariates_ok.sum()),
              int(keep.sum()), len(people)]
    flow = pd.DataFrame({
        "stage": ["Original extract", "Both baseline background measures observed",
                  "All background measures and regression covariates observed",
                  "All three earnings years observed", "Matched positive custom weight"],
        "respondents": counts,
        "excluded_at_stage": [0] + [left - right for left, right in zip(counts, counts[1:])],
    })
    missingness = []
    populations = {"Original extract": raw,
                   "Covariate-complete": raw.loc[covariates_ok],
                   "Selected balanced panel": raw.loc[keep]}
    for population, frame in populations.items():
        for year, column in EARNINGS.items():
            for code, meaning in {**MISSING_CODES, 0: "Reported zero amount"}.items():
                missingness.append({"population": population, "respondents": len(frame),
                                    "earnings_year": year, "extract_column": column,
                                    "code": code, "meaning": meaning,
                                    "count": int(frame[column].eq(code).sum())})
    skipped_people = int(raw.loc[covariates_ok, list(EARNINGS.values())].eq(-4).any(axis=1).sum())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.diagnostics_dir.mkdir(parents=True, exist_ok=True)
    panel_path = args.output_dir / "clean_panel.csv"
    panel.to_csv(panel_path, index=False)
    flow.to_csv(args.diagnostics_dir / "sample_flow.csv", index=False)
    pd.DataFrame(missingness).to_csv(args.diagnostics_dir / "earnings_missingness.csv", index=False)
    metadata = {
        "stage": "01_clean_panel",
        "inputs": {"raw_data": {"path": str(args.raw_data.resolve()), "sha256": sha256(args.raw_data)},
                   "weights": {"path": str(args.weights.resolve()), "sha256": sha256(args.weights)}},
        "script_sha256": sha256(Path(__file__)),
        "missing_codes": MISSING_CODES,
        "selection": "All required background measures and covariates, and all three earnings amounts, observed; exact custom-weight ID match.",
        "valid_skip_policy": SKIP_WARNING,
        "reported_zero_earnings_retained": True,
        "other_negative_background_amounts_retained": True,
        "earnings_mapping": {str(year): {"source": column, "survey_year": year + 1}
                             for year, column in EARNINGS.items()},
        "counts": {"raw_respondents": len(raw), "background_observed": counts[1],
                   "covariate_complete": counts[2], "selected_respondents": len(people),
                   "person_years": len(panel), "covariate_complete_with_earnings_valid_skip": skipped_people},
        "output": {"path": str(panel_path.resolve()), "sha256": sha256(panel_path),
                   "columns": list(panel.columns)},
    }
    (args.output_dir / "cleaning_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(flow.to_string(index=False))
    print(f"Saved {len(people):,} respondents / {len(panel):,} person-years to {panel_path}")
    print(f"WARNING: {skipped_people:,} otherwise covariate-complete respondents have at least one earnings valid skip.")
    print(SKIP_WARNING)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, pd.errors.ParserError) as error:
        print(f"Data cleaning failed: {error}", file=sys.stderr)
        sys.exit(1)
