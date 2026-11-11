# Descriptive audit of the archived exploratory 60th-percentile specification.
# Run from any directory: Rscript /path/to/E4331_ER/scripts/build_empirical_summary.R
# Base R only. Reads archived inputs; never modifies archive/ or reports/main.tex.
args <- commandArgs(trailingOnly = FALSE)
script <- sub("^--file=", "", args[grepl("^--file=", args)])
if (length(script) != 1L) stop("Run this file with Rscript.")
root <- dirname(dirname(normalizePath(script)))
out <- file.path(root, "data", "derived")
tables <- file.path(root, "reports", "tables")
dir.create(out, recursive = TRUE, showWarnings = FALSE)
dir.create(tables, recursive = TRUE, showWarnings = FALSE)
options(stringsAsFactors = FALSE)

panel <- read.csv(file.path(root, "archive/data/balanced_panel_60.csv"))
thresholds <- read.csv(file.path(root, "archive/results_60/background_thresholds.csv"))
stopifnot(nrow(panel) == 6369L, length(unique(panel$id)) == 2123L,
          !anyDuplicated(panel[c("id", "year")]), !anyNA(panel),
          all(table(panel$id) == 3L),
          identical(sort(unique(panel$year)), c(2016L, 2018L, 2020L)),
          all(panel$regression_weight > 0),
          all(panel$college == as.integer(panel$highest_degree_2015 >= 4)),
          all(thresholds$percentile == 0.60))
people <- panel[panel$year == 2016L, ]
people <- people[order(people$id), ]
income_cut <- thresholds$nominal_cutoff[thresholds$variable == "household_income_1996"]
wealth_cut <- thresholds$nominal_cutoff[thresholds$variable == "household_net_worth_1997"]
stopifnot(length(income_cut) == 1L, length(wealth_cut) == 1L,
          all(people$low_background == as.integer(
            people$household_income_1996_nominal < income_cut &
            people$household_net_worth_1997_nominal < wealth_cut)),
          all(people$asvab_percentile >= 0 & people$asvab_percentile <= 100))
for (year in c(2018L, 2020L)) {
  d <- panel[panel$year == year, ]
  d <- d[order(d$id), ]
  fixed <- c("id", "low_background", "college", "regression_weight",
             "asvab_percentile", "highest_degree_2015", "female", "race_ethnicity",
             "household_income_2025", "household_net_worth_2025")
  stopifnot(identical(people[fixed], d[fixed]) ||
            isTRUE(all.equal(unname(as.matrix(people[fixed])),
                             unname(as.matrix(d[fixed])))))
}

group_names <- c("All respondents", "Lower resources, no BA+",
                 "Lower resources, BA+", "Other backgrounds, no BA+",
                 "Other backgrounds, BA+")
groups <- list(rep(TRUE, nrow(people)),
               people$low_background == 1 & people$college == 0,
               people$low_background == 1 & people$college == 1,
               people$low_background == 0 & people$college == 0,
               people$low_background == 0 & people$college == 1)
names(groups) <- group_names
stopifnot(all(Reduce(`+`, groups[-1]) == 1L))
total_weight <- sum(people$regression_weight)
group_n <- vapply(groups, sum, numeric(1))
group_weight <- vapply(groups, function(g) sum(people$regression_weight[g]), numeric(1))
group_share <- 100 * group_weight / total_weight
stopifnot(sum(group_n[-1]) == nrow(people), abs(sum(group_share[-1]) - 100) < 1e-10)

weighted_stats <- function(x, w) {
  stopifnot(length(x) == length(w), !anyNA(x), all(is.finite(x)),
            all(is.finite(w)), all(w > 0))
  m <- weighted.mean(x, w)
  o <- order(x)
  c(mean = m, sd = sqrt(sum(w * (x - m)^2) / sum(w)),
    median = x[o][which(cumsum(w[o]) >= 0.5 * sum(w))[1]],
    min = min(x), max = max(x))
}
number <- function(x, digits = 1) formatC(x, format = "f", digits = digits, big.mark = ",")
row <- function(label, cells) paste0(label, " & ", paste(cells, collapse = " & "), " \\\\")
line <- function(label) paste0("\\addlinespace[4pt]\\multicolumn{6}{l}{\\textit{", label, "}} \\\\")
summary_records <- list()
summary_rows <- character()

continuous_row <- function(variable, label, x, units, display_scale = 1, include_median = FALSE) {
  stats <- vapply(groups, function(g) weighted_stats(x[g], people$regression_weight[g]), numeric(5))
  records <- do.call(rbind, lapply(seq_along(groups), function(j) data.frame(
    variable = variable, units = units, group = group_names[j], respondents = group_n[j],
    weighted_mean = stats["mean", j], weighted_descriptive_sd = stats["sd", j],
    weighted_median = stats["median", j], observed_min = stats["min", j],
    observed_max = stats["max", j])))
  summary_records[[length(summary_records) + 1L]] <<- records
  values <- paste0(number(stats["mean", ] / display_scale), " (",
                   number(stats["sd", ] / display_scale), ")")
  rows <- row(label, values)
  if (include_median) rows <- c(rows, row("\\quad Median", number(stats["median", ] / display_scale)))
  rows
}

summary_rows <- c(
  row("Respondents (unweighted)", number(group_n, 0)),
  row("Person-years (unweighted)", number(3 * group_n, 0)),
  row("Weighted sample share (\\%)", number(group_share)),
  line("Annual wage-and-salary earnings, thousands of 2025 dollars"))
for (year in c(2016L, 2018L, 2020L)) {
  d <- panel[panel$year == year, ]
  x <- d$wage_2025[match(people$id, d$id)]
  summary_rows <- c(summary_rows, continuous_row(paste0("earnings_", year),
    paste0(year, " earnings"), x, "2025 US dollars", 1000, TRUE))
}
summary_rows <- c(summary_rows, line("Childhood resources, thousands of 2025 dollars"),
  continuous_row("baseline_household_income", "1996 household income",
    people$household_income_2025, "2025 US dollars", 1000, TRUE),
  continuous_row("baseline_household_net_worth", "1997 household net worth",
    people$household_net_worth_2025, "2025 US dollars", 1000, TRUE),
  line("Academic skills and age"),
  continuous_row("asvab_percentile", "ASVAB percentile", people$asvab_percentile, "percentile points"),
  continuous_row("age_2016", "Age at end of 2016", people$age, "years"))
write.csv(do.call(rbind, summary_records), file.path(out, "summary_statistics.csv"), row.names = FALSE)
write.csv(data.frame(group = group_names, respondents = group_n, person_years = 3 * group_n,
                     weighted_sample_share_percent = group_share),
          file.path(out, "summary_group_counts.csv"), row.names = FALSE)

proportion_records <- list()
proportion_row <- function(variable, label, indicator) {
  stopifnot(length(indicator) == nrow(people), !anyNA(indicator))
  n <- vapply(groups, function(g) sum(indicator[g]), numeric(1))
  percent <- vapply(groups, function(g)
    100 * weighted.mean(as.numeric(indicator[g]), people$regression_weight[g]), numeric(1))
  proportion_records[[length(proportion_records) + 1L]] <<- data.frame(
    variable = variable, group = group_names, respondents = group_n,
    category_respondents = n, weighted_percent = percent)
  row(label, number(percent))
}
composition_rows <- c(line("Demographics (weighted percentages)"),
  proportion_row("female", "Female", people$female == 1),
  proportion_row("race_black", "Black", people$race_ethnicity == 1),
  proportion_row("race_hispanic", "Hispanic", people$race_ethnicity == 2),
  proportion_row("race_mixed_non_hispanic", "Mixed race, non-Hispanic", people$race_ethnicity == 3),
  proportion_row("race_non_black_non_hispanic", "Non-Black, non-Hispanic", people$race_ethnicity == 4),
  line("Highest degree by the 2015 interview (weighted percentages)"))
degree_labels <- c("No degree", "GED", "High school diploma", "Associate degree",
                   "Bachelor's degree", "Master's degree", "PhD", "Professional degree")
for (degree in 0:7) {
  composition_rows <- c(composition_rows, proportion_row(paste0("degree_", degree),
    degree_labels[degree + 1L], people$highest_degree_2015 == degree))
}
composition_rows <- c(composition_rows, line("Reported zero annual earnings (weighted percentages)"))
for (year in c(2016L, 2018L, 2020L)) {
  d <- panel[panel$year == year, ]
  x <- d$wage_2025[match(people$id, d$id)]
  composition_rows <- c(composition_rows, proportion_row(paste0("zero_earnings_", year),
    paste0(year, " earnings = 0"), x == 0))
}
write.csv(do.call(rbind, proportion_records), file.path(out, "summary_composition.csv"), row.names = FALSE)

header <- c("\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}lrrrrr@{}}", "\\toprule",
  " & All & \\multicolumn{2}{c}{Lower resources} & \\multicolumn{2}{c}{Other backgrounds} \\\\",
  "\\cmidrule(lr){3-4}\\cmidrule(l){5-6}",
  " & respondents & No BA+ & BA+ & No BA+ & BA+ \\\\", "\\midrule")
writeLines(c("% Generated by scripts/build_empirical_summary.R; do not edit by hand.",
  "\\begingroup\\fontsize{9}{11}\\selectfont", "\\setlength{\\tabcolsep}{3pt}",
  "\\captionof{table}{Summary statistics for the archived analysis sample}",
  "\\label{tab:empirical-summary}", header, summary_rows,
  "\\bottomrule", "\\end{tabular*}", "\\endgroup", "\\par\\smallskip",
  "{\\footnotesize Notes: Continuous variables are weighted means with descriptive standard deviations in parentheses; medians are shown separately. Standard deviations use $\\sum_i w_i(x_i-\\bar{x}_w)^2/\\sum_i w_i$ under the square root. Monetary entries are in thousands of 2025 dollars. Counts are unweighted; sample shares use custom respondent weights. Each respondent is counted once per earnings year and once for baseline characteristics. Ages in 2018 and 2020 are two and four years higher than in 2016, respectively.\\par}"),
  file.path(tables, "summary_statistics.tex"))
writeLines(c("% Generated by scripts/build_empirical_summary.R; do not edit by hand.",
  "\\begingroup\\fontsize{9}{11}\\selectfont", "\\setlength{\\tabcolsep}{3pt}",
  "\\captionof{table}{Demographics, education composition, and reported zero earnings}",
  "\\label{tab:empirical-composition}", header, composition_rows,
  "\\bottomrule", "\\end{tabular*}", "\\endgroup", "\\par\\smallskip",
  "{\\footnotesize Notes: Entries are weighted percentages within each column. BA+ means a bachelor's, master's, doctoral, or professional degree by the 2015 interview. No BA+ includes no degree, GED, high school diploma, and associate degree. Highest degree alone cannot distinguish high school graduates who never attended college from those who attended without earning a further degree. Degree status remains fixed at 2015. Race/ethnicity labels follow the archived NLSY97 coding. Reported zeros refer only to retained observations; excluded valid skips have not been recoded to zero.\\par}"),
  file.path(tables, "summary_demographics_education.tex"))

writeLines(c(
  "\\documentclass[11pt,letterpaper]{article}",
  "\\usepackage[margin=0.8in]{geometry}", "\\usepackage[T1]{fontenc}",
  "\\usepackage{newtxtext,newtxmath}", "\\usepackage{booktabs}",
  "\\usepackage{caption}", "\\usepackage[hidelinks]{hyperref}",
  "\\captionsetup{font=small,labelfont=bf,justification=raggedright,singlelinecheck=false,hypcap=false}",
  "\\setlength{\\parindent}{0pt}", "\\setlength{\\parskip}{6pt}",
  "\\begin{document}",
  "{\\Large\\bfseries Empirical sample: descriptive tables}\\par",
  "\\textbf{Exploratory archived specification.} NLSY97 balanced complete-case sample: 2,123 respondents observed for earnings in 2016, 2018, and 2020 (6,369 person-years). These tables describe the existing 60th-percentile analysis; they do not change the selected sample or background definition.\\par",
  paste0("Lower resources means both 1996 household income below \\$", number(income_cut, 0),
    " and 1997 household net worth below \\$", number(wealth_cut, 0),
    " in their original nominal dollars. These are separate custom-weighted 60th-percentile cutoffs within this sample. Other backgrounds means at least one measure meets or exceeds its cutoff; it does not identify an exclusively wealthy group.\\par"),
  "\\input{tables/summary_statistics.tex}",
  "\\newpage",
  "{\\large\\bfseries Composition and measurement notes}\\par",
  "\\input{tables/summary_demographics_education.tex}",
  "\\textbf{ASVAB.} The archived code uses \\texttt{R9829600}/1,000, restoring the three implied decimal places of the NLSY97-created math--verbal percentile. The score is already normalized within three-month birth groups; this division is a units conversion, not age adjustment. See the local extract codebook and \\href{https://nlsinfo.org/content/cohorts/nlsy97/other-documentation/codebook-supplement/appendix-10-cat-asvab-scores}{NLSY97 Appendix 10}.\\par",
  "\\textbf{Scope.} Annual wage-and-salary earnings include variation in pay and time worked. The archived sample excludes missing responses and valid skips and requires observed earnings amounts in all three years. Because the amount question is asked of respondents reporting job income, this requirement may exclude non-earners. The extract lacks the receipt-filter variables needed to distinguish such skips from other missingness. Audit those filters before finalizing the sample. The retained reported zeros do not measure nonemployment prevalence in the original cohort; custom weights do not remove every source of complete-case selection.\\par",
  "\\textbf{Reproduction.} Run \\texttt{Rscript scripts/build\\_empirical\\_summary.R} from the project root (or pass its absolute path), then compile this file from \\texttt{reports/}. Inputs are the archived \\texttt{balanced\\_panel\\_60.csv} and \\texttt{background\\_thresholds.csv}. Machine-readable aggregate statistics, including minima and maxima, are saved in \\texttt{data/derived/summary\\_statistics.csv}; cell counts and composition are in the accompanying summary CSVs. No respondent identifiers are written to these outputs.",
  "\\end{document}"), file.path(root, "reports", "empirical_summary.tex"))

cat("Generated descriptive tables for", nrow(people), "respondents and", nrow(panel), "person-years.\n")
cat("Cell counts (lower/no BA+, lower/BA+, other/no BA+, other/BA+):",
    paste(group_n[-1], collapse = ", "), "\n")
cat("Outputs: reports/empirical_summary.tex; reports/tables/summary_*.tex; data/derived/summary_*.csv\n")
