"""Audit the previous empirical analysis without modifying the archive.

Run: python3 scripts/audit_archive_inputs.py
Requires Python's standard library and Rscript on PATH. Reproduction runs in a
temporary directory; only aggregate CSV audit outputs are retained.
"""

import csv
import math
from pathlib import Path
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "archive"
OUTPUT = ROOT / "data" / "derived"
OUTPUT.mkdir(parents=True, exist_ok=True)


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(filename, rows):
    with (OUTPUT / filename).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# Definitions checked against archive/data/default/default.cdb.
definitions = [
    ("R0000100", "PUBID", "Respondent identifier", "1997", "Link repeated observations; do not report individual identifiers.", 1),
    ("R0536300", "KEY!SEX", "Sex", "1997", "Female indicator: code 2; male reference: code 1.", 31),
    ("R0536401", "KEY!BDATE_M", "Birth month", "Birth", "Present in extract; unused by archived model.", 53),
    ("R0536402", "KEY!BDATE_Y", "Birth year", "Birth", "Year-end age = earnings calendar year minus birth year.", 87),
    ("R1204500", "CV_INCOME_GROSS_YR", "Gross household income", "1996; collected 1997", "Remove exact missing codes -1 through -5; retain other negative values. Top 2% mean-replacement topcoding. Background cutoff uses nominal dollars; summary levels also available in 2025 dollars.", 106),
    ("R1204700", "CV_HH_NET_WORTH_P", "Parent-reported household net worth", "1997", "Remove exact missing codes -1 through -5; retain other negative values. Values above $600,000 capped at $600,000. Background cutoff uses nominal dollars.", 152),
    ("R1482600", "KEY!RACE_ETHNICITY", "Combined race and ethnicity", "1997", "1 Black; 2 Hispanic; 3 mixed race non-Hispanic; 4 non-Black/non-Hispanic (regression reference).", 203),
    ("R9829600", "ASVAB_MATH_VERBAL_SCORE_PCT", "Age-normalized ASVAB math-verbal percentile", "Round-1 test; cross-round release", "Stored percentile / 1000 restores three implied decimal places. Already normalized within three-month birth groups; not a raw subtest score. Remove missing codes before rescaling.", 226),
    ("U0009400", "CV_HIGHEST_DEGREE_EVER_EDT", "Highest degree received by interview", "2015", "0 none; 1 GED; 2 high school; 3 associate; 4 bachelor; 5 master; 6 PhD; 7 professional. BA+ indicator uses 4-7. Does not identify college attendance among people without BA.", 277),
    ("U2857200", "YINC-1700", "Gross wages, salary, commissions and tips", "2016; collected 2017", "Annual amount, not hourly wage or total income. Top 2% mean-replacement topcoding. Valid skips need receipt-filter review; current analysis excludes them.", 305),
    ("U4282300", "YINC-1700", "Gross wages, salary, commissions and tips", "2018; collected 2019", "Annual amount, not hourly wage or total income. Top 2% mean-replacement topcoding. Valid skips need receipt-filter review; current analysis excludes them.", 351),
    ("U5753500", "YINC-1700", "Gross wages, salary, commissions and tips", "2020; collected 2021", "Annual amount, not hourly wage or total income. Top 2% mean-replacement topcoding. Valid skips need receipt-filter review; current analysis excludes them.", 396),
]
write_csv("archive_variable_dictionary.csv", [
    dict(extract_column=column, question_name=name, description=description,
         reference_period=period, construction_and_limits=notes,
         local_codebook=f"archive/data/default/default.cdb:{line}")
    for column, name, description, period, notes, line in definitions
])

# Question names identify the requested screens even where the public HTML
# leaves reference IDs blank. Only the 2019 IDs are independently verified.
receipt_filters = []
questionnaires = {
    2017: "https://www.nlsinfo.org/sites/default/files/attachments/18109/8.Income.html",
    2019: "https://www.nlsinfo.org/sites/default/files/attachments/21113/8.Income.html",
    2021: "https://www.nlsinfo.org/sites/default/files/attachments/230112/8.Income.html",
}
for survey_year, source in questionnaires.items():
    for name, purpose in [
        ("YINC-1400", "Initial receipt of wage/salary/commission/tip income"),
        ("YINC-1600", "Fallback receipt screen after uncertain/refused initial answer; also mentions self-employment"),
    ]:
        reference = {"YINC-1400": "U42821.00", "YINC-1600": "U42822.00"}.get(name) if survey_year == 2019 else ""
        receipt_filters.append(dict(survey_year=survey_year, earnings_year=survey_year - 1,
            question_name=name, verified_reference_id=reference, purpose=purpose,
            status="Missing from current extract; request by question name and survey year",
            questionnaire=source))
write_csv("needed_receipt_filters.csv", receipt_filters)

raw = [{key: float(value) for key, value in row.items()}
       for row in read_csv(ARCHIVE / "data/default/default.csv")]
missing_codes = {-1, -2, -3, -4, -5}
background_vars = ["R1204500", "R1204700"]
covariate_vars = background_vars + ["R0536402", "R0536300", "R1482600", "R9829600", "U0009400"]
earnings_vars = ["U2857200", "U4282300", "U5753500"]


def observed(row, variables):
    return all(math.isfinite(row[v]) and row[v] not in missing_codes for v in variables)


background = [row for row in raw if observed(row, background_vars)]
covariates = [row for row in raw if observed(row, covariate_vars)
              and row["R0536300"] in (1, 2) and row["R1482600"] in (1, 2, 3, 4)]
selected = [row for row in covariates if observed(row, earnings_vars)]
assert (len(raw), len(background), len(covariates), len(selected)) == (8984, 5833, 3915, 2123)
stages = [("Original extract", raw), ("Both baseline resources observed", background),
          ("All background and regression covariates observed", covariates),
          ("All three earnings amounts observed", selected)]
write_csv("archive_sample_flow.csv", [
    dict(stage=label, respondents=len(rows), share_of_original=len(rows) / len(raw))
    for label, rows in stages
])

missingness = []
labels = {-1: "Refusal", -2: "Do not know", -3: "Invalid skip", -4: "Valid skip",
          -5: "Non-interview", 0: "Reported zero amount"}
for population, rows in [("Original extract", raw), ("Covariate-complete", covariates),
                         ("Selected balanced panel", selected)]:
    for variable, year in zip(earnings_vars, (2016, 2018, 2020)):
        for code, label in labels.items():
            missingness.append(dict(population=population, respondents=len(rows),
                earnings_year=year, extract_column=variable, code=code, meaning=label,
                count=sum(row[variable] == code for row in rows)))
write_csv("archive_earnings_missingness.csv", missingness)
valid_skip_people = sum(any(row[v] == -4 for v in earnings_vars) for row in covariates)
assert valid_skip_people == 987
print(f"Covariate-complete respondents with at least one earnings valid skip: {valid_skip_people}")

# Execute the original script on isolated copies, then compare its numerical
# outputs with the archived results. This checks reproduction, not validity of
# the sample definition or the causal interpretation.
inputs = ["analysis_60.R", "data/default/default.csv", "data/default/default.cdb",
          "data/CPIAUCNS.csv",
          "data/customweight_nlsy97_6ad2cf591a14779f5b2/customweight_nlsy97_6ad2cf591a14779f5b2.dat"]
comparisons = [f"results_60/{filename}.csv" for filename in
               ["regression_coefficients", "college_premia", "sample_flow", "model_summary",
                "background_thresholds", "completion_by_background", "cpi_conversion"]]
comparisons.append("data/balanced_panel_60.csv")
checks = []
with tempfile.TemporaryDirectory(prefix="e4331-reproduce-") as temporary:
    staging = Path(temporary)
    for relative in inputs:
        destination = staging / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ARCHIVE / relative, destination)
    result = subprocess.run(["Rscript", str(staging / "analysis_60.R")],
                            cwd=staging, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    for relative in comparisons:
        expected, actual = read_csv(ARCHIVE / relative), read_csv(staging / relative)
        assert len(expected) == len(actual), relative
        for left, right in zip(expected, actual):
            assert left.keys() == right.keys(), relative
            for column in left:
                if left[column] == right[column]:
                    continue
                try:
                    equal = math.isclose(float(left[column]), float(right[column]),
                                         rel_tol=1e-10, abs_tol=1e-6)
                except ValueError:
                    equal = False
                assert equal, f"Reproduction mismatch: {relative}, {column}"
        checks.append(dict(file=relative, status="matches archived output", rows=len(actual)))
write_csv("archive_reproduction_check.csv", checks)
print("All 8 archived output checks pass. Aggregate audit CSVs: data/derived/")
