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
   reported zero wages, and recovers a skipped wage amount only when the new
   income-receipt screens explicitly confirm zero wage income. Requires observed
   baseline income, degree status, controls, and all three resolved earnings
   amounts, checks the custom-weight ID match, and creates a long panel.
   Net worth is optional and has no role in current sample eligibility or the
   regression. The default also accepts an extract without the wealth column.
   Output: data/processed/clean_panel.csv (source units, no economic indicators).
   Also writes cleaning metadata, sample flow, raw missingness counts, and an
   audit identifying the source of each resolved wage amount.

2. 02_construct_variables.py
   Reads clean_panel.csv and data/raw/CPIAUCNS.csv. Constructs five education
   categories (plus descriptive BA+), female, age, ASVAB percentile, normalized
   weights, real monetary variables, and the lower-income indicator. The default
   cutoff is the weighted 20th percentile of 1996 household income. Each person
   is counted once; the cutoff is the first observed value where cumulative
   weight reaches 20%. All income ties at the cutoff are included (<=).
   Wealth does not determine this indicator. --background-definition joint-legacy
   preserves the earlier strict income AND wealth rule, defaulting to 60%.
   Output: data/processed/analysis_panel.csv and variable_metadata.json.
   Diagnostics: outputs/diagnostics/ (CPI factors, cutoffs, and weight checks).

3. 03_run_regression.py
   Fits weighted pooled OLS of real annual earnings on education categories,
   lower income, all four category interactions, ASVAB, female, age,
   race/ethnicity, and earnings-year effects.
   Inference uses respondent-clustered CR1 covariance and t(G-1) p-values/CIs.
   Reference groups: high-school diploma, income above the cutoff, male,
   non-Black/non-Hispanic, and earnings year 2016. Lower-income degree premiums
   use the full covariance of each degree coefficient and its interaction.
   A joint Wald F(4,G-1) tests all four interactions. --education-model binary
   retains the earlier BA+ versus no-BA specification for replication.
   Outputs: outputs/regression/regression_coefficients.csv,
   cluster_covariance.csv, education_premia.csv, education_by_background.csv,
   joint_interaction_test.csv, completion_by_background.csv, model_summary.csv,
   and regression_metadata.json. Binary mode writes college_premia.csv instead
   of the three education/joint-test outputs. Switching models removes only the
   obsolete generated files from the other model to avoid mixing results.
   There are no plotting or report-table-generation steps in this pipeline.

CURRENT SPECIFICATION AND REVISION RECORD

- Defaults include confirmed zero wage earnings: 3,171 respondents and
  9,513 observations for earnings in 2016, 2018, and 2020. The income-only
  20th-percentile cutoff is $21,613 in nominal 1996 dollars: 837 respondents,
  20.029% of weighted people (ties are not split).
- The cutoff adapts the bottom parental-income quintile used in the
  need-affirmative admissions exercise of Chetty, Deming, and Friedman (2026),
  Quarterly Journal of Economics 141(1):51-145, Section VI.A and Table V:
  https://doi.org/10.1093/qje/qjaf050
  Author version (August 2025), Section 6.1, p.43 and Table V, p.82:
  https://opportunityinsights.org/wp-content/uploads/2023/07/CollegeAdmissions_Paper.pdf
  Their policy follows Chetty et al. (2020), QJE135(3):1567-1633:
  https://doi.org/10.1093/qje/qjaa005
  This is not a replication of their income measure or causal design: they use
  multiyear parental income at child ages 12-17, national birth-cohort ranks,
  and exclude negative parental income. We use one year's household income,
  pool the selected 1980-84 cohort, and retain one genuine negative income.
  Our comparison group is all incomes above the cutoff, not uniformly wealthy.
  The current sample requires income and the model inputs, with matching official
  weights for all 3,171 selected respondents. A change in sample membership
  requires new custom weights; changing only degree categories or cutoffs does not.
- The updated raw CSV contains 18 fields. Its original 12 fields match the
  archive; the six new fields are YINC-1400 and YINC-1600 in 2017/2019/2021.
  The original archive is preserved. The current nlsy97_codebook.cdb now also
  documents all six receipt-screen variables in the revised extract.
- ASVAB R9829600 is ALREADY normalized within three-month birth groups.
  Dividing by 1000 restores the three implied decimals, not an age correction.
  Its four subtests are Arithmetic Reasoning, Mathematics Knowledge, Word
  Knowledge, and Paragraph Comprehension. The NLS-created percentile is retained;
  it is not reranked using the selected panel's custom regression weights.
  Construction: https://nlsinfo.org/content/cohorts/nlsy97/other-documentation/codebook-supplement/appendix-10-cat-asvab-scores
- Education categories (highest-degree source code -> constructed code):
    0/1 -> 0: no high-school diploma or GED only
      2 -> 1: high-school diploma (reference)
      3 -> 2: associate degree
      4 -> 3: bachelor's degree only
  5/6/7 -> 4: graduate or professional degree
  All categories are fixed at the 2015 interview. High-school diploma includes
  some college without a completed higher degree; it is not never-attendance.
  The lower-income graduate cell has 30 respondents; report its imprecision.
- The receipt-confirmed policy preserves observed nonnegative wage amounts.
  A skipped amount (-4) becomes zero only when YINC-1400=0, or YINC-1400 is
  -1/-2 and the fallback YINC-1600=0. Other missing amounts stay missing.
  This is zero wage-and-salary income, not necessarily nonemployment or zero
  self-employment income. Use --earnings-policy observed-only for old replication.
- If changing the selected respondents, obtain compatible custom weights.
  Stage 1 fails on an ID mismatch rather than silently dropping people.
- The old joint 60% cutoff was selected after exploratory comparisons. The new
  20% income-only definition was chosen from the literature before fitting the
  revised model; it is not preregistered. A citation cannot retrospectively
  justify the earlier search. --background-percentile permits explicitly
  motivated sensitivity checks, not automatic searches for significance.
  Prior draft and numeric outputs are preserved in
  outputs/revision_baselines/before_income_quintile/; the original archive is
  unchanged. reports/empirical_summary.* is still the old exploratory summary.

- The 2025 CPI mean uses 11 supplied months; October is missing. No month is
  imputed. Annual earnings include wages, salaries, commissions, and tips.
- Estimates are adjusted associations, not a causal DiD. The clustered
  covariance does not account for full survey strata/PSUs or estimated-cutoff
  uncertainty. Individual intervals are pointwise, without multiplicity adjustment.
- Input hashes and configuration are saved in JSON metadata. Stage 2 checks
  cleaning metadata; stage 3 rejects a changed analysis panel or a changed
  upstream clean panel. Rerun downstream stages after changing inputs.

OFFICIAL CUSTOM WEIGHTS

The current installed weights cover all 3,171 receipt-confirmed respondents.
They were downloaded from the NLSY97 official Weight IDs service, not estimated
locally. The exact ZIP, README, requested IDs, and validation manifest are in
outputs/sensitivity/no_net_worth/weights/official/. The previous primary report,
analysis, weights, and scripts are preserved under
outputs/revision_baselines/before_net_worth_requirement_removal/. Other historical
weights remain under data/raw/weights_previous/ and the original archive.

To obtain weights after a future change in sample membership:

    .venv/bin/python scripts/01_clean_panel.py --prepare-weights
    .venv/bin/python scripts/fetch_custom_weights.py
    .venv/bin/python scripts/01_clean_panel.py

The first command writes data/processed/weight_request/respondent_ids.txt and
an unweighted selected panel; it does not overwrite clean_panel.csv. The helper
submits only public respondent IDs to the official NLSY97 service, validates
that positive weights match the request exactly, and installs custom_weights.dat.
It needs internet access. Alternatively paste the ID file into the Weight IDs
tab at https://www.nlsinfo.org/content/weights/nlsy97 and download the package.
Do not combine the ID mode with year selections. Stored weights have two
implied decimals; stage 2 restores the units and normalizes once per respondent.

ARCHIVAL REPLICATION

For exact replication of the original observed-only sample, use its original
raw input and weights and separate output directories:

    .venv/bin/python scripts/01_clean_panel.py --earnings-policy observed-only --net-worth-policy required --raw-data archive/data/default/default.csv --weights archive/data/customweight_nlsy97_6ad2cf591a14779f5b2/customweight_nlsy97_6ad2cf591a14779f5b2.dat --output-dir data/processed/observed_only --diagnostics-dir outputs/diagnostics/observed_only

Then use explicit legacy flags in stages 2 and 3:

    .venv/bin/python scripts/02_construct_variables.py --input data/processed/observed_only/clean_panel.csv --background-definition joint-legacy --output-dir data/processed/observed_only --diagnostics-dir outputs/diagnostics/observed_only
    .venv/bin/python scripts/03_run_regression.py --input data/processed/observed_only/analysis_panel.csv --education-model binary --output-dir outputs/regression/observed_only

VALIDATION

The current sample-selection policy is --net-worth-policy optional. Stage 2
inherits it from cleaning metadata and retains any unused missing wealth fields
without imputation. --net-worth-policy required is available for archival
replication; the legacy joint income/wealth definition requires observed wealth.
The earlier comparison under outputs/sensitivity/no_net_worth/ is retained as
a revision record; its expanded sample is now the primary analysis.

    .venv/bin/python -m unittest discover -s tests -v

The integration checks compare the Python panel, regression, covariance,
contrasts, and sample counts against the archived R outputs in observed-only
mode. They also check an alternative cutoff, conservative zero reconstruction,
the new ID list, official-weight attachment, stale-input rejection, the new
income cutoff/ties, all ten education/background cells, contrasts against an
independent cell-intercept parameterization, and the joint interaction test.
They also verify that missing or absent wealth leaves current model eligibility
unchanged, while missing income, controls, or unresolved earnings remain excluded.

The older audit_archive_inputs.py and build_empirical_summary.R files are
supplemental audit/presentation utilities, not dependencies of this pipeline.

PAPER RESULTS

reports/tables/education_premia.tex contains the twelve degree/background
contrasts from outputs/regression/education_premia.csv and is included by
reports/main.tex. It records the source CSV hash in a comment. This presentation
table is a snapshot, not an automatic output of the three analysis scripts;
refresh it and the numerical prose after changing or rerunning the model.

reports/tables/summary_statistics_current.tex contains current weighted summary
statistics by income group and is included in the Data section. It records the
analysis-panel hash; refresh this snapshot after changing the panel. Earnings,
zero-earnings shares, and age use person-years; baseline measures count each
respondent once. Monetary figures are in thousands of 2025 dollars. The older
summary_statistics.tex belongs to the previous empirical_summary report.

APPENDIX ROBUSTNESS CHECKS

The appendix compares the main estimate with two exploratory checks chosen
before estimating their results: omit 2020 earnings, and use the weighted
25th-percentile income cutoff. Each retains the same 3,171 people and official
weights. The first retains the original three-year sample eligibility and
omits only 2020 outcome rows; it does not reconstruct a larger two-year sample.
The second uses all three years and changes only the lower-income indicator.

    .venv/bin/python scripts/03_run_regression.py --earnings-years 2016 2018 --output-dir outputs/sensitivity/exclude_2020/regression
    .venv/bin/python scripts/02_construct_variables.py --background-percentile 0.25 --output-dir outputs/sensitivity/income_p25/processed --diagnostics-dir outputs/sensitivity/income_p25/diagnostics
    .venv/bin/python scripts/03_run_regression.py --input outputs/sensitivity/income_p25/processed/analysis_panel.csv --output-dir outputs/sensitivity/income_p25/regression

Stage 3 validates the complete source panel before applying --earnings-years.
Its metadata records the source and estimation years and unchanged respondent
selection. The main output directories are not overwritten by these commands.
reports/tables/robustness_checks.tex is the presentation snapshot used by the
appendix; refresh it and the prose after changing either check. The analysis
plan and independent numerical audit are in outputs/sensitivity/robustness_plan.json
and outputs/sensitivity/robustness_independent_audit.json.

RESULTS FIGURE (SUPPLEMENTARY PRESENTATION STEP)

    .venv/bin/python scripts/plot_education_premia.py

Reads the saved categorical-model contrasts and their 95% confidence intervals;
does not refit the regression. It verifies the input provenance and renders
reports/figures/education_premia.pdf (embedded in main.tex), a 300-dpi PNG preview,
and education_premia_figure_metadata.json with all 12 plotted points and hashes.
Panel A shows degree premiums relative to high-school diploma within each income
group. Panel B shows lower-income minus above-cutoff premiums. Units are thousands
of 2025 dollars; intervals use the full saved clustered covariance. Regenerate
the figure after changing the analysis, and update its caption and interpretation
if the specification changes. Matplotlib is included in requirements.txt.
