EMPIRICAL ANALYSIS IN PYTHON

Run these three main scripts in order from the project directory:

    .venv/bin/python scripts/01_clean_panel.py
    .venv/bin/python scripts/02_construct_variables.py
    .venv/bin/python scripts/03_run_regression.py

The project virtual environment is already configured. For a fresh setup:

    python3 -m venv .venv
    .venv/bin/python -m pip install -r requirements.txt

All three scripts also work when invoked by absolute path from another directory.
Use --help to inspect input and output path options.

1. 01_clean_panel.py
   Reads data/raw/nlsy97.csv and custom_weights.dat. Removes exact NLS missing
   codes -1 through -5, preserves other negative baseline resource amounts and
   reported zero wages, applies the previous complete-case selection, checks the
   custom-weight ID match, and reshapes the selected people into a long panel.
   Output: data/processed/clean_panel.csv (source units, no economic indicators).
   Also writes cleaning metadata, sample flow, and earnings-missingness counts.

2. 02_construct_variables.py
   Reads clean_panel.csv and data/raw/CPIAUCNS.csv. Constructs BA+, female,
   age, ASVAB percentile, normalized weights, real monetary variables, and the
   lower-background indicator. Cutoffs count each person once and use the first
   observed value where cumulative weight reaches the chosen percentile.
   Classification requires BOTH resources to be strictly below their cutoffs.
   Output: data/processed/analysis_panel.csv and variable_metadata.json.
   Diagnostics: outputs/diagnostics/ (CPI factors, cutoffs, and weight checks).

3. 03_run_regression.py
   Fits weighted pooled OLS of real annual earnings on BA+, lower background,
   their interaction, ASVAB, female, age, race/ethnicity, and earnings-year effects.
   Inference uses respondent-clustered CR1 covariance and t(G-1) p-values/CIs.
   Reference groups: no BA+, other background, male, non-Black/non-Hispanic,
   and earnings year 2016. The lower-background premium uses the full covariance
   of the BA+ and interaction coefficients.
   Outputs: outputs/regression/regression_coefficients.csv,
   cluster_covariance.csv, college_premia.csv, completion_by_background.csv,
   model_summary.csv, and regression_metadata.json.
   There are no plotting or report-table-generation steps in this pipeline.

STARTING SPECIFICATION AND REVISION POINTS

- Defaults reproduce the prior exploratory 60th-percentile analysis: 2,123
  respondents and 6,369 observations for earnings in 2016, 2018, and 2020.
- Raw files are byte-for-byte copies of archive/data/ inputs. The original
  archive is preserved. nlsy97_codebook.cdb documents the source variables.
- ASVAB R9829600 is ALREADY normalized within three-month birth groups.
  Dividing by 1000 restores the three implied decimals, not an age correction.
- BA+ means a bachelor's or higher degree by the 2015 interview. No BA+ does
  not mean never attended college. The classification stays fixed across years.
- The current input lacks YINC-1400/YINC-1600 receipt questions. Earnings valid
  skips remain excluded; do not convert every skip into zero. Add and inspect
  those questions before revising the zero-income/nonemployment treatment.
- If changing the selected respondents, obtain compatible custom weights.
  Stage 1 fails on an ID mismatch rather than silently dropping people.
- The inherited 60% cutoff was selected after exploratory comparisons. Do not
  choose a new definition by its p-value. To examine another substantively
  motivated definition, set --background-percentile in stage 2 and rerun stage 3.
  Separate output directories keep versions from overwriting one another:

    .venv/bin/python scripts/02_construct_variables.py --background-percentile 0.50 --output-dir data/processed/p50 --diagnostics-dir outputs/diagnostics/p50
    .venv/bin/python scripts/03_run_regression.py --input data/processed/p50/analysis_panel.csv --output-dir outputs/regression/p50

- The 2025 CPI mean uses 11 supplied months; October is missing. No month is
  imputed. Annual earnings include wages, salaries, commissions, and tips.
- Estimates are adjusted associations, not a causal DiD. The clustered
  covariance does not account for full survey strata/PSUs or cutoff selection.
- Input hashes and configuration are saved in JSON metadata. Stage 3 refuses
  a panel that does not match its stage-2 metadata. Rerun downstream stages
  after changing an input or specification.

VALIDATION

    .venv/bin/python -m unittest discover -s tests -v

The integration checks compare the Python panel, regression, covariance,
contrasts, and sample counts against the archived R outputs. They also check
an alternative cutoff and rejection of mismatched weights/stale metadata.

The older audit_archive_inputs.py and build_empirical_summary.R files are
supplemental audit/presentation utilities, not dependencies of this pipeline.
