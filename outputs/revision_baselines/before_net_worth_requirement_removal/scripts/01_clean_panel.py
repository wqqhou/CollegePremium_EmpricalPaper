"""Clean the NLSY97 extract and select the analysis panel.

Run from any directory:
    python /path/to/E4331_ER/scripts/01_clean_panel.py

This first stage keeps source measurements in their original units. Economic
variables, inflation adjustment, and weight normalization belong to stage 2.
The default uses receipt screens to recover confirmed zero wage earnings.
Use --prepare-weights to export the selected IDs before obtaining new weights.
Use --earnings-policy observed-only to reproduce the previous report's sample.
Use --net-worth-policy optional to retain people whose baseline wealth is missing.
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
RECEIPT_SCREENS = {2016: ("U2857000", "U2857100"),
                   2018: ("U4282100", "U4282200"),
                   2020: ("U5753300", "U5753400")}
BACKGROUND = ["R1204500", "R1204700"]
COVARIATES = [column for column in SOURCE_COLUMNS if column != "R0000100"]
WEIGHT_SERVICE = "https://www.nlsinfo.org/content/weights/nlsy97"
RECEIPT_POLICY = (
    "Preserve reported nonnegative wage amounts. Recover an amount coded -4 as "
    "zero only when YINC-1400=0, or when YINC-1400 is -1/-2 and YINC-1600=0. "
    "Other missing amounts and unresolved screens remain missing. This recovers "
    "wage-and-salary zeros, not necessarily zero total income or nonemployment."
)
OBSERVED_POLICY = (
    "Observed-only replication: all amount codes -1 through -5 remain missing; "
    "reported zero wages are retained. Receipt screens do not change selection."
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


def resolve_earnings(amount, receipt, fallback):
    """Return resolved wage amounts and provenance without guessing missing pay.

    YINC-1600 is asked after refusal/don't-know on YINC-1400. Its broader
    self-employment wording means a Yes is not evidence of positive wage pay.
    Only explicit No responses support recovering a skipped wage amount as 0.
    """
    for name, screen in [("YINC-1400", receipt), ("YINC-1600", fallback)]:
        require(screen.dropna().isin([*MISSING_CODES, 0, 1]).all(),
                f"Unexpected response codes in {name}.")
    require(not ((amount < 0) & ~amount.isin(list(MISSING_CODES))).any(),
            "Unexpected negative wage amounts after accounting for NLS missing codes.")
    fallback_asked = fallback.isin([-1, -2, 0, 1])
    require(not (fallback_asked & ~receipt.isin([-1, -2])).any(),
            "Receipt-screen routing conflict: observed fallback answer without an initial refusal/don't-know.")
    initial_no = receipt.eq(0)
    fallback_no = receipt.isin([-1, -2]) & fallback.eq(0)
    require(not ((amount > 0) & (initial_no | fallback_no)).any(),
            "Receipt-screen conflict: positive wage amount paired with an explicit No-income answer.")
    require(not (amount.ge(0) & receipt.eq(-5)).any(),
            "Receipt-screen conflict: observed wage amount paired with a non-interview screen.")

    resolved = amount.astype(float).mask(amount.isin(list(MISSING_CODES)))
    source = pd.Series("missing_unclassified", index=amount.index, dtype="str")
    source.loc[amount.gt(0)] = "reported_positive"
    source.loc[amount.eq(0)] = "reported_zero"
    for code, label in MISSING_CODES.items():
        source.loc[amount.eq(code)] = "missing_" + label.lower().replace(" ", "_")
    initial_zero = amount.eq(-4) & initial_no
    fallback_zero = amount.eq(-4) & fallback_no
    resolved.loc[initial_zero | fallback_zero] = 0.0
    source.loc[initial_zero] = "confirmed_zero_initial_screen"
    source.loc[fallback_zero] = "confirmed_zero_fallback_screen"
    return resolved, source


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
    parser.add_argument("--earnings-policy", choices=["receipt-confirmed", "observed-only"],
                        default="receipt-confirmed", help="Recover only receipt-confirmed zeros, or reproduce the old observed-only sample.")
    parser.add_argument("--net-worth-policy", choices=["required", "optional"],
                        default="required", help="Require observed baseline net worth (default), or retain missing wealth without using it for sample eligibility.")
    parser.add_argument("--prepare-weights", action="store_true",
                        help="Export selected IDs and an unweighted panel under output-dir/weight_request; do not load weights or overwrite clean_panel.csv.")
    return parser.parse_args()


def main():
    args = parse_args()
    for path in ([args.raw_data] if args.prepare_weights else [args.raw_data, args.weights]):
        require(path.is_file(), f"Input does not exist: {path}")

    raw = pd.read_csv(args.raw_data)
    required_columns = list(SOURCE_COLUMNS) + list(EARNINGS.values())
    if args.earnings_policy == "receipt-confirmed":
        required_columns += [column for pair in RECEIPT_SCREENS.values() for column in pair]
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

    # Inspect the original missing/receipt codes before replacing skipped wage
    # amounts. Keep provenance separate from the numeric downstream panel.
    provenance = []
    for year, amount_column in EARNINGS.items():
        amount = raw[amount_column]
        if args.earnings_policy == "receipt-confirmed":
            receipt_column, fallback_column = RECEIPT_SCREENS[year]
            receipt, fallback = raw[receipt_column], raw[fallback_column]
            clean[amount_column], source = resolve_earnings(amount, receipt, fallback)
        else:
            source = pd.Series("missing_unclassified", index=raw.index, dtype="str")
            source.loc[amount.gt(0)] = "reported_positive"
            source.loc[amount.eq(0)] = "reported_zero"
            for code, label in MISSING_CODES.items():
                source.loc[amount.eq(code)] = "missing_" + label.lower().replace(" ", "_")
            receipt = pd.Series(np.nan, index=raw.index)
            fallback = pd.Series(np.nan, index=raw.index)
        provenance.append(pd.DataFrame({
            "id": ids.astype("int64"), "year": year, "raw_wage_amount": amount,
            "receipt_screen_raw": receipt, "fallback_screen_raw": fallback,
            "wage_nominal": clean[amount_column], "wage_source": source,
        }))

    required_background = BACKGROUND if args.net_worth_policy == "required" else ["R1204500"]
    required_covariates = [column for column in COVARIATES
                           if args.net_worth_policy == "required" or column != "R1204700"]
    background_ok = clean[required_background].notna().all(axis=1)
    covariates_ok = clean[required_covariates].notna().all(axis=1)
    keep = covariates_ok & clean[list(EARNINGS.values())].notna().all(axis=1)
    require(keep.any(), "No respondents satisfy the balanced complete-case selection.")

    selected_ids = set(clean.loc[keep, "R0000100"].astype(int))
    people = clean.loc[keep, list(SOURCE_COLUMNS)].rename(columns=SOURCE_COLUMNS)
    zero_weight_rows = 0
    if not args.prepare_weights:
        # Official downloads may also contain zero-weight rows for respondents
        # outside the requested sample. Only those extra zeros can be ignored.
        weights = pd.read_csv(args.weights, sep=r"\s+", header=None)
        require(weights.shape[1] == 2, "Custom weights must have exactly two columns: ID and weight.")
        weights.columns = ["id", "custom_weight_raw"]
        weights = weights.apply(pd.to_numeric, errors="raise")
        require(np.isfinite(weights.to_numpy(dtype=float)).all(),
                "Custom weights contain missing or non-finite values.")
        require((weights["id"] > 0).all() and (weights["id"] % 1 == 0).all(),
                "Custom-weight IDs must be positive integers.")
        require(not weights["id"].duplicated().any(), "Custom weights contain duplicate IDs.")
        require((weights["custom_weight_raw"] >= 0).all(), "Custom weights cannot be negative.")
        zero_selected = weights["id"].isin(selected_ids) & weights["custom_weight_raw"].eq(0)
        require(not zero_selected.any(),
                f"{int(zero_selected.sum())} selected respondents have zero custom weight. Check the official weight request; do not drop them silently.")
        zero_weight_rows = int(weights["custom_weight_raw"].eq(0).sum())
        weights = weights.loc[weights["custom_weight_raw"] > 0].copy()
        weight_ids = set(weights["id"].astype(int))
        require(selected_ids == weight_ids,
                "Custom weights do not match the selected sample: "
                f"{len(selected_ids - weight_ids)} selected people lack weights and "
                f"{len(weight_ids - selected_ids)} positively weighted people are not selected. "
                "Run --prepare-weights and obtain weights for its respondent_ids.txt file.")
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
    required_panel_columns = [column for column in panel
                              if args.net_worth_policy == "required"
                              or column != "household_net_worth_1997_nominal"]
    require(not panel[required_panel_columns].isna().any().any(),
            "Unexpected missing values in the selected panel.")
    require(not panel.duplicated(["id", "year"]).any(), "Duplicate person-year in panel.")
    require(panel.groupby("id").size().eq(len(EARNINGS)).all(), "Panel is not balanced.")

    counts = [len(raw), int(background_ok.sum()), int(covariates_ok.sum()), int(keep.sum())]
    stages = ["Original extract",
              ("Both baseline background measures observed" if args.net_worth_policy == "required"
               else "Baseline household income observed; net worth not required"),
              ("All background measures and regression covariates observed" if args.net_worth_policy == "required"
               else "Baseline income and regression covariates observed; net worth not required"),
              ("All three earnings years observed" if args.earnings_policy == "observed-only"
               else "All three earnings years observed or confirmed zero")]
    if not args.prepare_weights:
        counts.append(len(people))
        stages.append("Matched positive custom weight")
    flow = pd.DataFrame({
        "stage": stages,
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

    sources = pd.concat(provenance, ignore_index=True).sort_values(["id", "year"]).reset_index(drop=True)
    sources["covariate_complete"] = sources["id"].isin(set(ids[covariates_ok]))
    sources["selected"] = sources["id"].isin(selected_ids)
    reconstruction = []
    for population, subset in [("Original extract", sources),
                               ("Covariate-complete", sources.loc[sources.covariate_complete]),
                               ("Selected balanced panel", sources.loc[sources.selected])]:
        counts_by_source = subset.groupby(["year", "wage_source"]).size()
        for (year, source), count in counts_by_source.items():
            reconstruction.append({"population": population, "earnings_year": int(year),
                                   "wage_source": source, "count": int(count)})
    recovered = sources["wage_source"].str.startswith("confirmed_zero_")
    recovered_selected = recovered & sources.selected

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.diagnostics_dir.mkdir(parents=True, exist_ok=True)
    flow.to_csv(args.diagnostics_dir / "sample_flow.csv", index=False)
    pd.DataFrame(missingness).to_csv(args.diagnostics_dir / "earnings_missingness.csv", index=False)
    pd.DataFrame(reconstruction).to_csv(args.diagnostics_dir / "earnings_reconstruction.csv", index=False)
    sources.to_csv(args.diagnostics_dir / "earnings_sources.csv", index=False)
    inputs = {"raw_data": {"path": str(args.raw_data.resolve()), "sha256": sha256(args.raw_data)}}
    if not args.prepare_weights:
        inputs["weights"] = {"path": str(args.weights.resolve()), "sha256": sha256(args.weights)}
    metadata = {
        "stage": "01_clean_panel",
        "inputs": inputs,
        "script_sha256": sha256(Path(__file__)),
        "earnings_policy": args.earnings_policy,
        "net_worth_policy": args.net_worth_policy,
        "weights_pending": args.prepare_weights,
        "missing_codes": MISSING_CODES,
        "selection": (
            ("Baseline income and net worth" if args.net_worth_policy == "required" else "Baseline income")
            + " and regression covariates and all three resolved wage amounts observed; "
            + "exact positive custom-weight ID match when weights are attached."
            + (" Missing net worth is retained without imputation." if args.net_worth_policy == "optional" else "")
        ),
        "valid_skip_policy": RECEIPT_POLICY if args.earnings_policy == "receipt-confirmed" else OBSERVED_POLICY,
        "reported_zero_earnings_retained": True,
        "other_negative_background_amounts_retained": True,
        "earnings_mapping": {str(year): {"source": column, "survey_year": year + 1}
                             for year, column in EARNINGS.items()},
        "receipt_mapping": {str(year): {"initial": pair[0], "fallback": pair[1]}
                            for year, pair in RECEIPT_SCREENS.items()},
        "counts": {"raw_respondents": len(raw), "background_observed": counts[1],
                   "covariate_complete": counts[2], "selected_respondents": len(people),
                   "person_years": len(panel), "covariate_complete_with_earnings_valid_skip": skipped_people,
                   "selected_respondents_missing_net_worth": int(people["household_net_worth_1997_nominal"].isna().sum()),
                   "recovered_zero_amounts_all": int(recovered.sum()),
                   "recovered_zero_amounts_selected": int(recovered_selected.sum()),
                   "selected_people_with_recovered_zero": int(sources.loc[recovered_selected, "id"].nunique()),
                   "ignored_zero_weights_outside_sample": zero_weight_rows},
    }
    if args.prepare_weights:
        destination = args.output_dir / "weight_request"
        destination.mkdir(parents=True, exist_ok=True)
        ids_path = destination / "respondent_ids.txt"
        ids_path.write_text("".join(f"{identifier}\n" for identifier in sorted(selected_ids)))
        panel_path = destination / "selected_panel_unweighted.csv"
        panel.to_csv(panel_path, index=False)
        metadata["weight_request"] = {
            "service": WEIGHT_SERVICE, "method": "Weight IDs; one respondent ID per line",
            "respondent_ids": {"path": str(ids_path.resolve()), "sha256": sha256(ids_path)},
            "requested_respondents": len(selected_ids),
        }
        metadata_path = destination / "request_metadata.json"
    else:
        panel_path = args.output_dir / "clean_panel.csv"
        panel.to_csv(panel_path, index=False)
        metadata_path = args.output_dir / "cleaning_metadata.json"
    metadata["output"] = {"path": str(panel_path.resolve()), "sha256": sha256(panel_path),
                          "columns": list(panel.columns)}
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(flow.to_string(index=False))
    print(f"Saved {len(people):,} respondents / {len(panel):,} person-years to {panel_path}")
    print(f"Recovered {int(recovered_selected.sum()):,} zero wage amounts in the selected panel "
          f"for {metadata['counts']['selected_people_with_recovered_zero']:,} respondents.")
    if args.prepare_weights:
        print(f"Weight request IDs: {ids_path}")
        print("Weighted clean_panel.csv was not changed. Obtain official weights, then rerun without --prepare-weights.")
    print(metadata["valid_skip_policy"])


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, pd.errors.ParserError) as error:
        print(f"Data cleaning failed: {error}", file=sys.stderr)
        sys.exit(1)
