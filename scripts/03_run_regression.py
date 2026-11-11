"""Estimate the archived college/background interaction model using Python.

Run after 01_clean_panel.py and 02_construct_variables.py. Outputs are numeric
CSV files and a JSON record of the specification; no figures or LaTeX tables
are generated. Sampling weights and respondent-clustered CR1 standard errors
match the previous R analysis.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
import scipy
from scipy.stats import t as student_t


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_panel(path, metadata_path):
    if not path.is_file() or not metadata_path.is_file():
        raise ValueError("Analysis panel or variable metadata is missing. Run stage 2 first.")
    metadata = json.loads(metadata_path.read_text())
    expected_hash = metadata.get("outputs", {}).get("analysis_panel", {}).get("sha256")
    if expected_hash != sha256(path):
        raise ValueError("Analysis panel does not match its variable metadata. Rerun stage 2.")
    panel = pd.read_csv(path)
    required = ["id", "year", "wage_2025", "college", "low_background",
                "asvab_percentile", "female", "age", "race_ethnicity", "regression_weight"]
    missing = sorted(set(required) - set(panel.columns))
    if missing:
        raise ValueError(f"Analysis panel is missing columns: {missing}")
    if panel.empty or not np.isfinite(panel[required].to_numpy(dtype=float)).all():
        raise ValueError("Model variables must be finite and the panel must not be empty.")
    if panel.duplicated(["id", "year"]).any():
        raise ValueError("Duplicate respondent-year observations.")
    if (panel.regression_weight <= 0).any() or (panel.wage_2025 < 0).any():
        raise ValueError("Weights must be positive and annual wage amounts nonnegative.")
    for variable in ["college", "low_background", "female"]:
        if not panel[variable].isin([0, 1]).all():
            raise ValueError(f"{variable} must be coded 0/1.")
    if not panel.race_ethnicity.isin([1, 2, 3, 4]).all():
        raise ValueError("Unexpected race/ethnicity code.")
    groups = panel.groupby("id", sort=True)
    stable = ["college", "low_background", "asvab_percentile", "female",
              "race_ethnicity", "regression_weight"]
    if (groups[stable].nunique() != 1).any().any():
        raise ValueError("Baseline variables and respondent weights must be constant within person.")
    years = sorted(panel.year.unique().tolist())
    if not (groups.year.nunique() == len(years)).all():
        raise ValueError("This specification requires the balanced panel created by stage 1.")
    expected_years = metadata.get("configuration", {}).get("earnings_years")
    if expected_years is not None and years != sorted(expected_years):
        raise ValueError("Earnings years do not match variable metadata.")
    people = panel.drop_duplicates("id")
    if len(people.groupby(["low_background", "college"])) != 4:
        raise ValueError("All four background-by-degree groups are needed for the interaction.")
    return panel.sort_values(["id", "year"]).reset_index(drop=True), metadata


def design_matrix(panel):
    # R's archived formula orders the interaction last. Preserve its term names
    # so coefficients can be compared directly with the archived CSV outputs.
    columns = {
        "(Intercept)": np.ones(len(panel)),
        "college": panel.college.to_numpy(),
        "low_background": panel.low_background.to_numpy(),
        "asvab_percentile": panel.asvab_percentile.to_numpy(),
        "female": panel.female.to_numpy(),
        "age": panel.age.to_numpy(),
    }
    if 4 not in panel.race_ethnicity.unique():
        raise ValueError("The specified race reference category (code 4) is absent.")
    for code, label in [(1, "raceBlack"), (2, "raceHispanic"), (3, "raceMixedNonHispanic")]:
        if code in panel.race_ethnicity.unique():
            columns[label] = (panel.race_ethnicity == code).to_numpy(dtype=float)
    years = sorted(panel.year.unique())
    for year in years[1:]:
        columns[f"factor(year){int(year)}"] = (panel.year == year).to_numpy(dtype=float)
    columns["college:low_background"] = (panel.college * panel.low_background).to_numpy()
    return np.column_stack(list(columns.values())).astype(float), list(columns)


def estimate(panel):
    x, terms = design_matrix(panel)
    y = panel.wage_2025.to_numpy(dtype=float)
    w = panel.regression_weight.to_numpy(dtype=float)
    n, k = x.shape
    ids, inverse = np.unique(panel.id.to_numpy(), return_inverse=True)
    g = len(ids)
    if n <= k or g <= 1:
        raise ValueError("Insufficient observations or respondent clusters for inference.")
    x_weighted = x * np.sqrt(w)[:, None]
    coefficients, _, rank, _ = np.linalg.lstsq(x_weighted, y * np.sqrt(w), rcond=None)
    if rank != k:
        raise ValueError("Regressors are not full rank; inspect sample composition and controls.")
    residuals = y - x @ coefficients

    # Bread = inverse(X' W X), obtained through QR for numerical stability.
    # Probability-weighted scores include w_i, not sqrt(w_i).
    _, triangular = np.linalg.qr(x_weighted, mode="reduced")
    triangular_inverse = np.linalg.solve(triangular, np.eye(k))
    bread = triangular_inverse @ triangular_inverse.T
    cluster_scores = np.zeros((g, k))
    np.add.at(cluster_scores, inverse, x * (w * residuals)[:, None])
    correction = (g / (g - 1)) * ((n - 1) / (n - k))
    covariance = correction * bread @ (cluster_scores.T @ cluster_scores) @ bread
    covariance = (covariance + covariance.T) / 2
    if not np.isfinite(covariance).all() or (np.diag(covariance) <= 0).any():
        raise ValueError("Cluster covariance has invalid diagonal elements.")
    standard_errors = np.sqrt(np.diag(covariance))
    df = g - 1
    critical = student_t.ppf(0.975, df)
    statistics = coefficients / standard_errors
    coefficient_table = pd.DataFrame({
        "term": terms, "estimate": coefficients, "cluster_se": standard_errors,
        "t": statistics, "p_value": 2 * student_t.sf(np.abs(statistics), df),
        "ci_low": coefficients - critical * standard_errors,
        "ci_high": coefficients + critical * standard_errors,
    })
    contrasts = []
    for label, selected in [
        ("Other backgrounds (L=0)", ["college"]),
        ("Lower background (L=1)", ["college", "college:low_background"]),
        ("Difference (L=1 minus L=0)", ["college:low_background"]),
    ]:
        vector = np.array([float(term in selected) for term in terms])
        value = float(vector @ coefficients)
        se = float(np.sqrt(vector @ covariance @ vector))
        contrasts.append(dict(contrast=label, estimate=value, cluster_se=se,
            p_value=float(2 * student_t.sf(abs(value / se), df)),
            ci_low=value - critical * se, ci_high=value + critical * se))
    total_ss = np.sum(w * (y - np.average(y, weights=w)) ** 2)
    if total_ss <= 0:
        raise ValueError("Earnings must have nonzero weighted variance.")
    r_squared = float(1 - np.sum(w * residuals ** 2) / total_ss)
    summary = dict(observations=n, respondents=g, periods=int(panel.year.nunique()),
        parameters=k, weighting="custom sampling weights normalized to mean 1",
        r_squared=r_squared, adjusted_r_squared=1 - (1 - r_squared) * (n - 1) / (n - k),
        cluster_df=df)
    return coefficient_table, pd.DataFrame(covariance, index=terms, columns=terms), pd.DataFrame(contrasts), summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=ROOT / "data/processed/analysis_panel.csv")
    parser.add_argument("--metadata", type=Path, help="Defaults to variable_metadata.json beside the input panel.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/regression")
    args = parser.parse_args()
    metadata_path = args.metadata or args.input.parent / "variable_metadata.json"
    panel, variable_metadata = read_panel(args.input, metadata_path)
    coefficients, covariance, premia, summary = estimate(panel)
    people = panel.drop_duplicates("id")
    total_weight = people.regression_weight.sum()
    completion = []
    for group, members in people.groupby("low_background", sort=True):
        completion.append(dict(low_background=int(group), respondents=len(members),
            college_graduates=int(members.college.sum()),
            weighted_group_share=float(members.regression_weight.sum() / total_weight),
            weighted_college_share=float(np.average(members.college, weights=members.regression_weight))))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "regression_coefficients.csv": coefficients,
        "college_premia.csv": premia,
        "model_summary.csv": pd.DataFrame([summary]),
        "completion_by_background.csv": pd.DataFrame(completion),
    }
    for filename, frame in outputs.items():
        frame.to_csv(args.output_dir / filename, index=False)
    covariance.to_csv(args.output_dir / "cluster_covariance.csv")
    metadata = {
        "input": {"path": str(args.input.resolve()), "sha256": sha256(args.input)},
        "variable_metadata": {"path": str(metadata_path.resolve()), "sha256": sha256(metadata_path)},
        "configuration": variable_metadata.get("configuration", {}),
        "background_definition": variable_metadata.get("background_definition", {}),
        "formula": "wage_2025 ~ college * low_background + asvab_percentile + female + age + race + factor(year)",
        "reference_categories": {"college": 0, "low_background": 0, "female": 0,
                                 "race_ethnicity": 4, "year": int(panel.year.min())},
        "estimation": "Weighted pooled OLS; same respondent sampling weight in every earnings year",
        "covariance": "Respondent-clustered sandwich with weighted scores and CR1 factor G/(G-1)*(N-1)/(N-K)",
        "inference": "Two-sided Student-t tests and 95% pointwise confidence intervals with G-1 degrees of freedom",
        "sample": summary,
        "limitations": [
            "Associations, not a causal difference-in-differences design.",
            "Valid-skip earnings amounts remain excluded pending receipt-filter verification.",
            "The inherited 60th-percentile background definition was selected after exploratory comparisons.",
            "Inference does not adjust for cutoff selection, estimated-cutoff uncertainty, or the full survey design.",
            "BA+ is degree status at the 2015 interview, not college attendance or updated degree status.",
        ],
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scipy": scipy.__version__},
        "outputs": {filename: {"sha256": sha256(args.output_dir / filename)}
                    for filename in [*outputs, "cluster_covariance.csv"]},
    }
    (args.output_dir / "regression_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    difference = premia.iloc[2]
    print(f"Estimated weighted pooled OLS: {summary['respondents']:,} respondents, {summary['observations']:,} person-years.")
    print(f"Difference in college premiums (lower minus other): ${difference.estimate:,.2f}; "
          f"clustered SE ${difference.cluster_se:,.2f}; p={difference.p_value:.6f}.")
    print(f"Numeric results saved to {args.output_dir.resolve()}")
    print("Exploratory associations; no charts or formatted regression tables generated.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, FileNotFoundError) as error:
        print(f"Regression error: {error}", file=sys.stderr)
        sys.exit(1)
