#!/usr/bin/env python3
"""Render the paper's two-panel figure from the saved regression contrasts.

This is a supplementary presentation step, not a fourth analysis stage. It
does not estimate a model or alter the data. Run after the three main scripts:
    .venv/bin/python scripts/plot_education_premia.py
"""

from pathlib import Path
import argparse
import hashlib
import json
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "e4331_matplotlib"))

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEGREES = ["no_diploma_or_ged", "associate", "bachelor", "graduate"]
LABELS = ["No diploma/GED", "Associate degree", "Bachelor's only", "Graduate /\nprofessional"]
ABOVE = "Other backgrounds (L=0)"
LOWER = "Lower background (L=1)"
DIFFERENCE = "Difference (L=1 minus L=0)"
CONTRASTS = [ABOVE, LOWER, DIFFERENCE]
TITLE = "Adjusted education earnings premiums by family income"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_record(record, description):
    path = Path(record["path"])
    if not path.is_file() or sha256(path) != record["sha256"]:
        raise ValueError(f"{description} differs from the regression's provenance record. Rebuild downstream analysis first.")


def read_results(folder):
    metadata_path = folder / "regression_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    source = folder / "education_premia.csv"
    if sha256(source) != metadata["outputs"][source.name]["sha256"]:
        raise ValueError("Education contrasts do not match regression metadata. Rerun stage 3.")
    if metadata["configuration"]["education_model"] != "categorical":
        raise ValueError("This figure requires the categorical education model.")
    check_record(metadata["input"], "Analysis panel")
    check_record(metadata["variable_metadata"], "Variable metadata")
    variables = json.loads(Path(metadata["variable_metadata"]["path"]).read_text())
    check_record(variables["inputs"]["clean_panel"], "Upstream clean panel")
    frame = pd.read_csv(source)
    keys = set(zip(frame.education, frame.contrast))
    if len(frame) != 12 or keys != {(degree, contrast) for degree in DEGREES for contrast in CONTRASTS}:
        raise ValueError("Expected exactly four education categories and three contrasts per category.")
    if not frame.reference_education.eq("high_school").all():
        raise ValueError("The figure labels require high-school diploma as the reference.")
    numeric = frame[["estimate", "cluster_se", "ci_low", "ci_high"]]
    if not np.isfinite(numeric.to_numpy()).all() or not frame.cluster_se.gt(0).all():
        raise ValueError("Estimates and intervals must be finite with positive standard errors.")
    if not (frame.ci_low.le(frame.estimate) & frame.ci_high.ge(frame.estimate)).all():
        raise ValueError("A saved confidence interval does not contain its estimate.")
    return frame, metadata, source, metadata_path


def render(folder, output_dir):
    frame, metadata, source, metadata_path = read_results(folder)
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "font.size": 10, "axes.titlesize": 10, "xtick.labelsize": 9,
        "ytick.labelsize": 10, "legend.fontsize": 9,
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.unicode_minus": True,
    })
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 4.4), sharey=True,
                             gridspec_kw={"width_ratios": [1.2, 1.0]})
    fig.subplots_adjust(left=0.235, right=0.985, bottom=0.17, top=0.78, wspace=0.16)
    y = np.arange(len(DEGREES))
    blue, orange, charcoal = "#0072B2", "#D55E00", "#3A3A3A"
    for ax in axes:
        ax.axhspan(1.63, 2.37, color="#EEF2F5", zorder=0)
        ax.grid(axis="x", color="#E3E3E3", linewidth=0.55, zorder=0)
        ax.axvline(0, color="#777777", linestyle="--", linewidth=0.8, zorder=1)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#777777")
        ax.tick_params(axis="y", length=0, pad=8)
        ax.tick_params(axis="x", length=3, color="#777777")
        ax.set_ylim(3.55, -0.55)
        ax.xaxis.set_major_locator(MultipleLocator(20))

    plotted = []
    for contrast, color, marker, offset in [(ABOVE, blue, "o", -0.14), (LOWER, orange, "s", 0.14)]:
        rows = frame.loc[frame.contrast.eq(contrast)].set_index("education").loc[DEGREES]
        estimates = rows.estimate.to_numpy() / 1000
        lows, highs = rows.ci_low.to_numpy() / 1000, rows.ci_high.to_numpy() / 1000
        axes[0].errorbar(estimates, y + offset,
                         xerr=np.vstack([estimates - lows, highs - estimates]),
                         fmt=marker, color=color, markersize=5.0, capsize=3,
                         elinewidth=1.25, markeredgewidth=0.9, zorder=3)
        for degree, value, low, high in zip(DEGREES, estimates, lows, highs):
            plotted.append({"panel": "A", "education": degree, "contrast": contrast,
                            "estimate_thousands": float(value), "ci_low_thousands": float(low),
                            "ci_high_thousands": float(high)})

    rows = frame.loc[frame.contrast.eq(DIFFERENCE)].set_index("education").loc[DEGREES]
    for i, degree in enumerate(DEGREES):
        row = rows.loc[degree]
        value, low, high = row[["estimate", "ci_low", "ci_high"]].to_numpy(dtype=float) / 1000
        axes[1].errorbar(value, i, xerr=[[value - low], [high - value]],
                         fmt="D", color=charcoal, markersize=5.5 if degree == "bachelor" else 4.5,
                         capsize=3, elinewidth=1.6 if degree == "bachelor" else 1.1, zorder=3)
        plotted.append({"panel": "B", "education": degree, "contrast": DIFFERENCE,
                        "estimate_thousands": float(value), "ci_low_thousands": float(low),
                        "ci_high_thousands": float(high)})

    for panel_name, ax in zip(["A", "B"], axes):
        points = [row for row in plotted if row["panel"] == panel_name]
        low = min(row["ci_low_thousands"] for row in points)
        high = max(row["ci_high_thousands"] for row in points)
        padding = (high - low) * 0.06
        ax.set_xlim(low - padding, high + padding)
    axes[0].set_yticks(y, LABELS)
    axes[0].get_yticklabels()[2].set_fontweight("bold")
    axes[1].tick_params(axis="y", labelleft=False)
    axes[0].set_title("A. Adjusted premium\nrelative to high school", loc="left", pad=13)
    axes[1].set_title("B. Difference in premiums\nlower income minus above", loc="left", pad=13)
    handles = [Line2D([], [], color=blue, marker="o", linestyle="none", markersize=5,
                      label="Above income cutoff"),
               Line2D([], [], color=orange, marker="s", linestyle="none", markersize=5,
                      label="Lower income")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.61, 0.985),
               frameon=False, ncol=2, columnspacing=1.2, handletextpad=0.35)
    fig.supxlabel("Annual earnings difference (thousands of 2025 US dollars)", fontsize=10, y=0.045)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = output_dir / "education_premia.pdf"
    png = output_dir / "education_premia.png"
    fig.savefig(pdf, metadata={"Title": TITLE, "CreationDate": None, "ModDate": None})
    fig.savefig(png, dpi=300, metadata={"Title": TITLE})
    manifest = {
        "title": TITLE,
        "source": {"path": str(source.resolve()), "sha256": sha256(source)},
        "regression_metadata": {"path": str(metadata_path.resolve()), "sha256": sha256(metadata_path)},
        "script": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))},
        "units": "Thousands of 2025 US dollars",
        "intervals": "Saved pointwise 95% Student-t intervals from respondent-clustered CR1 covariance; no recomputation or independent-coefficient approximation",
        "sample": metadata["sample"],
        "plotted_points": plotted,
        "axis_limits": {panel: list(ax.get_xlim()) for panel, ax in zip(["A", "B"], axes)},
        "outputs": {path.name: {"sha256": sha256(path)} for path in [pdf, png]},
        "matplotlib_version": matplotlib.__version__,
    }
    (output_dir / "education_premia_figure_metadata.json").write_text(json.dumps(manifest, indent=2) + "\n")
    plt.close(fig)
    print(f"Rendered 12 estimates with saved confidence intervals: {pdf}")
    print(f"Preview: {png}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "outputs/regression")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/figures")
    args = parser.parse_args()
    try:
        render(args.results_dir, args.output_dir)
    except (ValueError, FileNotFoundError, KeyError) as error:
        parser.exit(1, f"Figure rendering failed: {error}\n")


if __name__ == "__main__":
    main()
