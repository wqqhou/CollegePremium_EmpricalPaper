"""Estimate education-by-family-background earnings associations using Python.

Run after 01_clean_panel.py and 02_construct_variables.py. Outputs are numeric
CSV files and a JSON record of the specification; no figures or LaTeX tables
are generated. Sampling weights and respondent-clustered CR1 standard errors
match the previous R analysis. The default uses five education categories,
with high-school diploma as the reference. Use --education-model binary to
replicate the archived BA+ versus no-BA specification.
Use --earnings-years 2016 2018 to omit 2020 while retaining the selected
respondents, their official weights, and their constructed baseline variables.
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
from scipy.stats import f as fisher_f


ROOT = Path(__file__).resolve().parents[1]
EDUCATION_LABELS = {
    0: "no_diploma_or_ged",
    1: "high_school",
    2: "associate",
    3: "bachelor",
    4: "graduate",
}
DEGREE_TO_CATEGORY = {0: 0, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 4, 7: 4}
NONREFERENCE_EDUCATION = {code: label for code, label in EDUCATION_LABELS.items() if code != 1}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_panel(path, metadata_path, education_model="categorical"):
    if education_model not in ("categorical", "binary"):
        raise ValueError("Education model must be categorical or binary.")
    if not path.is_file() or not metadata_path.is_file():
        raise ValueError("Analysis panel or variable metadata is missing. Run stage 2 first.")
    metadata = json.loads(metadata_path.read_text())
    expected_hash = metadata.get("outputs", {}).get("analysis_panel", {}).get("sha256")
    if expected_hash != sha256(path):
        raise ValueError("Analysis panel does not match its variable metadata. Rerun stage 2.")
    clean_input = metadata.get("inputs", {}).get("clean_panel", {})
    if clean_input.get("path"):
        upstream = Path(clean_input["path"])
        if upstream.is_file() and clean_input.get("sha256") != sha256(upstream):
            raise ValueError("Clean panel has changed since variable construction. Rerun stage 2 before regression.")
    panel = pd.read_csv(path)
    required = ["id", "year", "wage_2025", "college", "low_background",
                "asvab_percentile", "female", "age", "race_ethnicity", "regression_weight"]
    if education_model == "categorical":
        required += ["education_category", "highest_degree_2015"]
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
    if education_model == "categorical":
        if not panel.education_category.isin(EDUCATION_LABELS).all():
            raise ValueError("Education category must use codes 0 through 4.")
        expected_category = panel.highest_degree_2015.map(DEGREE_TO_CATEGORY)
        if not panel.education_category.eq(expected_category).all():
            raise ValueError("Education categories do not match highest degree by 2015. Rerun stage 2.")
        if not panel.college.eq(panel.education_category.ge(3).astype(int)).all():
            raise ValueError("BA+ indicator and education categories disagree. Rerun stage 2.")
        stable += ["education_category", "highest_degree_2015"]
    if (groups[stable].nunique() != 1).any().any():
        raise ValueError("Baseline variables and respondent weights must be constant within person.")
    years = sorted(panel.year.unique().tolist())
    if not (groups.year.nunique() == len(years)).all():
        raise ValueError("This specification requires the balanced panel created by stage 1.")
    expected_years = metadata.get("configuration", {}).get("earnings_years")
    if expected_years is not None and years != sorted(expected_years):
        raise ValueError("Earnings years do not match variable metadata.")
    people = panel.drop_duplicates("id")
    degree_variable = "education_category" if education_model == "categorical" else "college"
    expected_cells = 10 if education_model == "categorical" else 4
    if len(people.groupby(["low_background", degree_variable])) != expected_cells:
        raise ValueError(f"All {expected_cells} background-by-education groups are needed for the interaction.")
    return panel.sort_values(["id", "year"]).reset_index(drop=True), metadata


def select_earnings_years(panel, requested_years=None):
    """Restrict earnings periods after validating the entire source panel.

    This does not redefine eligibility or recalculate person weights: the source
    panel's selected respondents remain the analysis population.
    """
    available = sorted(map(int, panel.year.unique()))
    selected = available if requested_years is None else list(requested_years)
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("Requested earnings years must be nonempty and distinct.")
    unavailable = sorted(set(selected) - set(available))
    if unavailable:
        raise ValueError(f"Requested earnings years {unavailable} are not available in the input panel {available}.")
    selected = sorted(selected)
    restricted = panel.loc[panel.year.isin(selected)].copy().reset_index(drop=True)
    if set(restricted.id) != set(panel.id):
        raise ValueError("Year selection must retain every respondent in the source panel.")
    if not restricted.groupby("id").size().eq(len(selected)).all():
        raise ValueError("Selected earnings years must form a balanced panel for the original respondents.")
    return restricted


def design_matrix(panel, education_model="categorical"):
    # R's archived formula orders the interaction last. Preserve its term names
    # so coefficients can be compared directly with the archived CSV outputs.
    columns = {"(Intercept)": np.ones(len(panel))}
    if education_model == "binary":
        degree_columns = {"college": panel.college.to_numpy()}
    elif education_model == "categorical":
        degree_columns = {
            f"education[{label}]": panel.education_category.eq(code).to_numpy(dtype=float)
            for code, label in NONREFERENCE_EDUCATION.items()
        }
    else:
        raise ValueError("Education model must be categorical or binary.")
    columns.update(degree_columns)
    columns.update({
        "low_background": panel.low_background.to_numpy(),
        "asvab_percentile": panel.asvab_percentile.to_numpy(),
        "female": panel.female.to_numpy(),
        "age": panel.age.to_numpy(),
    })
    if 4 not in panel.race_ethnicity.unique():
        raise ValueError("The specified race reference category (code 4) is absent.")
    for code, label in [(1, "raceBlack"), (2, "raceHispanic"), (3, "raceMixedNonHispanic")]:
        if code in panel.race_ethnicity.unique():
            columns[label] = (panel.race_ethnicity == code).to_numpy(dtype=float)
    years = sorted(panel.year.unique())
    for year in years[1:]:
        columns[f"factor(year){int(year)}"] = (panel.year == year).to_numpy(dtype=float)
    for term, values in degree_columns.items():
        columns[f"{term}:low_background"] = values * panel.low_background.to_numpy()
    return np.column_stack(list(columns.values())).astype(float), list(columns)


def estimate(panel, education_model="categorical"):
    x, terms = design_matrix(panel, education_model)
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
    degree_terms = {"college": "college"} if education_model == "binary" else {
        label: f"education[{label}]" for label in NONREFERENCE_EDUCATION.values()
    }
    for degree, degree_term in degree_terms.items():
        interaction_term = f"{degree_term}:low_background"
        for label, selected in [
            ("Other backgrounds (L=0)", [degree_term]),
            ("Lower background (L=1)", [degree_term, interaction_term]),
            ("Difference (L=1 minus L=0)", [interaction_term]),
        ]:
            vector = np.array([float(term in selected) for term in terms])
            value = float(vector @ coefficients)
            variance = float(vector @ covariance @ vector)
            if variance <= 0 or not np.isfinite(variance):
                raise ValueError(f"Invalid contrast variance for {degree}: {label}.")
            se = float(np.sqrt(variance))
            row = {} if education_model == "binary" else {
                "education": degree, "reference_education": "high_school",
            }
            row.update(contrast=label, estimate=value, cluster_se=se,
                p_value=float(2 * student_t.sf(abs(value / se), df)),
                ci_low=value - critical * se, ci_high=value + critical * se)
            contrasts.append(row)
    total_ss = np.sum(w * (y - np.average(y, weights=w)) ** 2)
    if total_ss <= 0:
        raise ValueError("Earnings must have nonzero weighted variance.")
    r_squared = float(1 - np.sum(w * residuals ** 2) / total_ss)
    summary = dict(observations=n, respondents=g, periods=int(panel.year.nunique()),
        parameters=k, weighting="custom sampling weights normalized to mean 1",
        r_squared=r_squared, adjusted_r_squared=1 - (1 - r_squared) * (n - 1) / (n - k),
        cluster_df=df)
    return coefficient_table, pd.DataFrame(covariance, index=terms, columns=terms), pd.DataFrame(contrasts), summary


def joint_interaction_test(coefficients, covariance, cluster_df):
    """CR1 Wald F test that every education/background interaction is zero."""
    terms = [f"education[{label}]:low_background" for label in NONREFERENCE_EDUCATION.values()]
    values = coefficients.set_index("term").loc[terms, "estimate"].to_numpy()
    restriction_covariance = covariance.loc[terms, terms].to_numpy()
    restrictions = len(terms)
    if np.linalg.matrix_rank(restriction_covariance) != restrictions:
        raise ValueError("Joint interaction covariance is singular; the joint test is not identified.")
    wald = float(values @ np.linalg.solve(restriction_covariance, values))
    if not np.isfinite(wald) or wald < 0:
        raise ValueError("Invalid joint interaction Wald statistic.")
    statistic = wald / restrictions
    return pd.DataFrame([{
        "null": "All four education-by-background interactions equal zero",
        "wald_statistic": wald, "f_statistic": statistic,
        "df_numerator": restrictions, "df_denominator": cluster_df,
        "p_value": float(fisher_f.sf(statistic, restrictions, cluster_df)),
    }])


def education_by_background(people):
    """Person-level cell counts and weighted degree shares within each group."""
    rows = []
    for group, members in people.groupby("low_background", sort=True):
        group_weight = members.regression_weight.sum()
        for code, label in EDUCATION_LABELS.items():
            cell = members.loc[members.education_category.eq(code)]
            rows.append({
                "low_background": int(group), "education_category": code,
                "education": label, "respondents": len(cell),
                "weighted_share_within_background": float(cell.regression_weight.sum() / group_weight),
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=ROOT / "data/processed/analysis_panel.csv")
    parser.add_argument("--metadata", type=Path, help="Defaults to variable_metadata.json beside the input panel.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/regression")
    parser.add_argument("--education-model", choices=["categorical", "binary"], default="categorical",
                        help="Default: five education categories, high-school diploma reference. Binary reproduces the archived BA+ model.")
    parser.add_argument("--earnings-years", type=int, nargs="+",
                        help="Optional subset of available earnings years, e.g. 2016 2018. Retains the source panel's people, weights, and baseline variables.")
    args = parser.parse_args()
    metadata_path = args.metadata or args.input.parent / "variable_metadata.json"
    panel, variable_metadata = read_panel(args.input, metadata_path, args.education_model)
    source_years = sorted(map(int, panel.year.unique()))
    source_observations = len(panel)
    source_respondents = int(panel.id.nunique())
    panel = select_earnings_years(panel, args.earnings_years)
    selected_years = sorted(map(int, panel.year.unique()))
    coefficients, covariance, premia, summary = estimate(panel, args.education_model)
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
        "model_summary.csv": pd.DataFrame([summary]),
        "completion_by_background.csv": pd.DataFrame(completion),
    }
    if args.education_model == "categorical":
        outputs["education_premia.csv"] = premia
        outputs["education_by_background.csv"] = education_by_background(people)
        outputs["joint_interaction_test.csv"] = joint_interaction_test(coefficients, covariance, summary["cluster_df"])
        obsolete_outputs = ["college_premia.csv"]
    else:
        outputs["college_premia.csv"] = premia
        obsolete_outputs = ["education_premia.csv", "education_by_background.csv", "joint_interaction_test.csv"]
    # Remove only known generated files belonging to the other model, so they
    # cannot be mistaken for results of this run. All checks above have passed.
    for filename in obsolete_outputs:
        (args.output_dir / filename).unlink(missing_ok=True)
    for filename, frame in outputs.items():
        frame.to_csv(args.output_dir / filename, index=False)
    covariance.to_csv(args.output_dir / "cluster_covariance.csv")
    metadata = {
        "input": {"path": str(args.input.resolve()), "sha256": sha256(args.input)},
        "variable_metadata": {"path": str(metadata_path.resolve()), "sha256": sha256(metadata_path)},
        "configuration": {**variable_metadata.get("configuration", {}),
                          "education_model": args.education_model,
                          "source_earnings_years": source_years,
                          "earnings_years": selected_years},
        "year_selection": {
            "source_earnings_years": source_years,
            "selected_earnings_years": selected_years,
            "omitted_earnings_years": sorted(set(source_years) - set(selected_years)),
            "source_observations": source_observations,
            "source_respondents": source_respondents,
            "respondent_selection_unchanged": True,
            "weights_and_baseline_variables_unchanged": True,
            "rule": "Validate the complete source panel, then retain only the selected earnings years. Eligibility remains based on the source panel; no new respondents or weights are introduced.",
        },
        "background_definition": variable_metadata.get("background_definition", {}),
        "education_definition": {
            "source": "Highest degree earned by the 2015 interview",
            "categories": EDUCATION_LABELS,
            "highest_degree_code_to_category": DEGREE_TO_CATEGORY,
            "pooled_groups": "Codes 0/1 pool no degree and GED; codes 5/6/7 pool master's, doctoral, and professional degrees.",
        },
        "formula": ("wage_2025 ~ education_category(reference=high_school) * low_background + asvab_percentile + female + age + race + factor(year)"
                    if args.education_model == "categorical" else
                    "wage_2025 ~ college * low_background + asvab_percentile + female + age + race + factor(year)"),
        "reference_categories": {("education_category" if args.education_model == "categorical" else "college"):
                                 (1 if args.education_model == "categorical" else 0), "low_background": 0, "female": 0,
                                 "race_ethnicity": 4, "year": int(panel.year.min())},
        "estimation": "Weighted pooled OLS; same respondent sampling weight in every earnings year",
        "covariance": "Respondent-clustered sandwich with weighted scores and CR1 factor G/(G-1)*(N-1)/(N-K)",
        "inference": ("Two-sided Student-t tests and 95% pointwise confidence intervals with G-1 degrees of freedom; categorical model additionally reports the joint CR1 Wald F(4,G-1) test of interactions"),
        "sample": summary,
        "limitations": [
            "Associations, not a causal difference-in-differences design.",
            ("Screening-confirmed zero wage amounts are included; unresolved skips and other missing amounts remain excluded."
             if variable_metadata.get("configuration", {}).get("earnings_policy") == "receipt-confirmed" else
             "The earnings-skip policy is inherited from the cleaning stage; inspect its metadata before interpreting the sample."),
            "The definition and reference population of family background are inherited from stage 2; see background_definition and its source/adaptation notes.",
            "Inference conditions on the chosen background definition and does not adjust for estimated-cutoff uncertainty, multiple pointwise contrasts, or the full survey design.",
            "Education is degree status at the 2015 interview, not college attendance or updated degree status; the high-school reference can include people who attended college without earning a postsecondary degree.",
            "Earlier exploratory specifications were examined before this revision; this is not a preregistered analysis.",
        ],
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scipy": scipy.__version__},
        "outputs": {filename: {"sha256": sha256(args.output_dir / filename)}
                    for filename in [*outputs, "cluster_covariance.csv"]},
    }
    (args.output_dir / "regression_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Estimated weighted pooled OLS: {summary['respondents']:,} respondents, {summary['observations']:,} person-years.")
    if selected_years != source_years:
        print(f"Earnings years: {selected_years}; all {source_respondents:,} source-panel respondents and their weights retained.")
    if args.education_model == "categorical":
        bachelor = premia.loc[premia.education.eq("bachelor")]
        print(f"BA-only versus high-school premium: other backgrounds ${bachelor.iloc[0].estimate:,.2f}; "
              f"lower background ${bachelor.iloc[1].estimate:,.2f}.")
        difference = bachelor.iloc[2]
    else:
        difference = premia.iloc[2]
    print(f"Difference in {'BA-only versus high-school' if args.education_model == 'categorical' else 'college'} premiums (lower minus other): ${difference.estimate:,.2f}; "
          f"clustered SE ${difference.cluster_se:,.2f}; p={difference.p_value:.6f}.")
    print(f"Numeric results saved to {args.output_dir.resolve()}")
    print("Exploratory associations; no charts or formatted regression tables generated.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, FileNotFoundError) as error:
        print(f"Regression error: {error}", file=sys.stderr)
        sys.exit(1)
