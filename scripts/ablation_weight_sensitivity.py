# ============================================================================
# scripts/ablation_weight_sensitivity.py
# ============================================================================
"""
CRS Weight Sensitivity Analysis (Ablation Study)
=================================================

PURPOSE
-------
Reviewers raised concerns about the three empirically-chosen weights in:

    CRS = (w_err * ERR + w_tfp * TFP) * (1 - w_rp_pen * RP)

This script re-computes CRS over a fine grid of weight combinations using the
raw component scores (ERR, RP, TFP) already stored in results.csv — no LLM
calls are made.  It then correlates each CRS variant against the ordinal
human-proxy score (Overall_Score, derived from judge verdicts) and produces:

  1. Three 2-D Pearson-r heatmaps (one free parameter fixed at default each time)
  2. A 3-D scatter plot of the full grid (correlation vs the three weights)
  3. A ranked table of the top-20 weight configurations
  4. A text report with interpretation

USAGE
-----
    # After a batch run:
    python scripts/ablation_weight_sensitivity.py \
        --results-dir batch_results \
        --output-dir ablation_results

    # With custom grid resolution:
    python scripts/ablation_weight_sensitivity.py \
        --results-dir batch_results \
        --grid-step 0.05          # default 0.10
        --output-dir ablation_results

DESIGN NOTES
------------
* ERR and TFP weights must sum to 1.0 (they are the numerator of a convex
  combination), so the free parameters are really:
      w_err   ∈ [0.1, 0.9]   (w_tfp = 1 - w_err implicitly)
      w_rp_pen ∈ [0.0, 0.8]

  This means the grid is 2-D, which makes every visualisation exact (no
  free-parameter marginalisation needed).

* Correlation metric: Spearman ρ is used as the primary metric because
  Overall_Score is ordinal (1-5), not interval.  Pearson r is reported
  alongside for completeness.

* The script is self-contained — it only imports from the standard library,
  numpy, pandas, scipy, matplotlib, and seaborn.  No pipeline code is touched.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Plotting imports — deferred so the script is usable in headless envs
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False
    print("⚠  matplotlib / seaborn not available — skipping plots.")


# ============================================================================
# CRS recomputation (mirrors correction_metrics.py exactly, no import needed)
# ============================================================================

def recompute_crs(
    err: float,
    rp: float,
    tfp: float,
    w_err: float,
    w_rp_pen: float,
) -> float:
    """
    Pure recomputation — matches CorrectionMetricsCalculator.compute_crs().

        w_tfp  = 1 - w_err   (enforced by the convex-combination constraint)
        quality = w_err * ERR + (1 - w_err) * TFP
        CRS     = clip(quality * (1 - w_rp_pen * RP), 0, 1)
    """
    w_tfp = 1.0 - w_err
    quality = w_err * err + w_tfp * tfp
    crs_raw = quality * (1.0 - w_rp_pen * rp)
    return float(np.clip(crs_raw, 0.0, 1.0))


# ============================================================================
# Grid construction
# ============================================================================

def build_weight_grid(step: float = 0.10) -> list[dict[str, float]]:
    """
    Enumerate all (w_err, w_rp_pen) combinations on a regular grid.

    Constraints:
      * w_err   ∈ {step, 2*step, …, 1-step}   (excludes 0 and 1 extremes)
      * w_rp_pen ∈ {0, step, 2*step, …, 0.8}  (penalty capped at 0.8)
      * w_tfp = 1 - w_err  (not a free variable)
    """
    w_err_vals = np.round(np.arange(step, 1.0, step), 6)
    w_rp_vals  = np.round(np.arange(0.0, 0.85, step), 6)

    configs = []
    for w_err, w_rp in product(w_err_vals, w_rp_vals):
        w_tfp = round(1.0 - w_err, 6)
        configs.append(
            dict(w_err=float(w_err), w_tfp=float(w_tfp), w_rp_pen=float(w_rp))
        )
    return configs


# ============================================================================
# Data loading & validation
# ============================================================================

REQUIRED_COLS = {"ERR_Final", "RP_Final", "TFP_Final", "Overall_Score", "Processing_Status"}


def load_results(results_dir: Path) -> pd.DataFrame:
    csv_path = results_dir / "results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"results.csv not found in {results_dir}")

    df = pd.read_csv(csv_path)
    
    gt_path = results_dir / "ground_truth.csv"
    if gt_path.exists():
        gt = pd.read_csv(gt_path)[["Problem_Statement", "human_correctness"]]
        df = df.merge(gt, on="Problem_Statement", how="left")
        print("  ✓  ground_truth.csv found — using human_correctness as correlation target")
    else:
        print("  ⚠  No ground_truth.csv found — using judge-derived Overall_Score as proxy")

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"results.csv is missing required columns: {missing}\n"
            f"Make sure the batch run used run_batch.py v3 which populates ERR_Final, "
            f"RP_Final, TFP_Final, and Overall_Score."
        )

    completed = df[df["Processing_Status"] == "COMPLETED"].copy()
    if len(completed) == 0:
        raise ValueError("No COMPLETED rows in results.csv — nothing to analyse.")

    # Drop rows where component scores are all zero (first-iteration-only runs
    # that never entered correction may have 0/0/0; they carry no signal).
    has_signal = (completed["ERR_Final"] + completed["TFP_Final"] + completed["RP_Final"]) > 0
    usable = completed[has_signal].copy()

    skipped = len(completed) - len(usable)
    if skipped > 0:
        print(f"  ⚠  Skipped {skipped} row(s) where all component scores are 0 "
              f"(first-iteration runs with no correction phase).")

    if len(usable) < 5:
        raise ValueError(
            f"Only {len(usable)} usable rows — need at least 5 for meaningful correlation."
        )

    print(f"  ✓  {len(usable)} usable rows loaded from {csv_path}")
    return usable


# ============================================================================
# Core analysis
# ============================================================================

def run_sensitivity_analysis(
    df: pd.DataFrame,
    weight_configs: list[dict],
) -> pd.DataFrame:
    """
    For each weight config, recompute CRS for every row then measure
    Spearman ρ and Pearson r against the target metric.
    """
    correlation_target = "human_correctness" if "human_correctness" in df.columns else "Overall_Score"
    
    # Filter out rows where correlation target is missing
    df_valid = df.dropna(subset=[correlation_target])
    
    if len(df_valid) == 0:
        raise ValueError(f"No valid rows with '{correlation_target}' found.")
        
    err  = df_valid["ERR_Final"].values.astype(float)
    rp   = df_valid["RP_Final"].values.astype(float)
    tfp  = df_valid["TFP_Final"].values.astype(float)
    human = df_valid[correlation_target].values.astype(float)

    records = []
    for cfg in weight_configs:
        crs_vec = np.array([
            recompute_crs(e, r, t, cfg["w_err"], cfg["w_rp_pen"])
            for e, r, t in zip(err, rp, tfp)
        ])

        spearman_r, spearman_p = stats.spearmanr(crs_vec, human)
        pearson_r,  pearson_p  = stats.pearsonr(crs_vec, human)
        mae = float(np.mean(np.abs(crs_vec - (human / 5.0))))  # scale human to [0,1]

        records.append({
            "w_err":    cfg["w_err"],
            "w_tfp":    cfg["w_tfp"],
            "w_rp_pen": cfg["w_rp_pen"],
            "spearman_r": round(spearman_r, 6),
            "spearman_p": round(spearman_p, 6),
            "pearson_r":  round(pearson_r,  6),
            "pearson_p":  round(pearson_p,  6),
            "mae_vs_human": round(mae, 6),
        })

    return pd.DataFrame(records)


# ============================================================================
# Visualisation helpers
# ============================================================================

def _pivot(results_df: pd.DataFrame, metric: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Pivot the flat results table into a 2-D matrix for heatmap plotting."""
    w_err_vals  = sorted(results_df["w_err"].unique())
    w_rp_vals   = sorted(results_df["w_rp_pen"].unique())

    matrix = (
        results_df
        .pivot(index="w_rp_pen", columns="w_err", values=metric)
        .reindex(index=w_rp_vals, columns=w_err_vals)
    )
    return matrix, np.array(w_err_vals), np.array(w_rp_vals)


def plot_heatmaps(results_df: pd.DataFrame, output_dir: Path, correlation_target: str, default_w_err: float = 0.5, default_w_rp: float = 0.4):
    """
    2-D heatmap of Spearman ρ over the (w_err, w_rp_pen) grid.
    The default operating point is annotated with a red star.
    """
    if not HAS_PLOT:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "CRS Weight Sensitivity Analysis\n"
        f"Correlation of CRS with {correlation_target}",
        fontsize=14, fontweight="bold"
    )

    for ax, metric, title, fmt in [
        (axes[0], "spearman_r", "Spearman ρ  (primary — ordinal score)", ".3f"),
        (axes[1], "pearson_r",  "Pearson r   (secondary)",                ".3f"),
    ]:
        matrix, w_err_vals, w_rp_vals = _pivot(results_df, metric)

        # Round axis labels to 2 dp for readability
        x_labels = [f"{v:.2f}" for v in w_err_vals]
        y_labels = [f"{v:.2f}" for v in w_rp_vals]

        sns.heatmap(
            matrix,
            ax=ax,
            annot=True,
            fmt=fmt,
            cmap="RdYlGn",
            vmin=results_df[metric].min(),
            vmax=results_df[metric].max(),
            linewidths=0.3,
            linecolor="white",
            xticklabels=x_labels,
            yticklabels=y_labels,
            annot_kws={"size": 7},
        )

        ax.set_xlabel("w_err  (w_tfp = 1 − w_err)", fontsize=11)
        ax.set_ylabel("w_rp_pen  (regression penalty weight)", fontsize=11)
        ax.set_title(title, fontsize=11)

        # Annotate default operating point
        try:
            xi = list(w_err_vals).index(min(w_err_vals, key=lambda v: abs(v - default_w_err)))
            yi = list(w_rp_vals).index(min(w_rp_vals,  key=lambda v: abs(v - default_w_rp)))
            ax.add_patch(plt.Rectangle((xi, yi), 1, 1, fill=False, edgecolor="red", lw=2.5))
            ax.plot(xi + 0.5, yi + 0.5, "r*", markersize=14, zorder=5,
                    label=f"Default ({default_w_err:.1f}, {default_w_rp:.1f})")
            ax.legend(loc="upper right", fontsize=9)
        except (ValueError, IndexError):
            pass  # default not on grid — skip annotation

        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.tick_params(axis="y", rotation=0,  labelsize=8)

    plt.tight_layout()
    out = output_dir / "heatmap_weight_sensitivity.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  📊  Heatmap saved → {out}")


def plot_correlation_vs_w_err(results_df: pd.DataFrame, output_dir: Path):
    """
    Line plot: Spearman ρ vs w_err, one line per w_rp_pen value.
    Helps reviewers see the ERR/TFP balance effect at a glance.
    """
    if not HAS_PLOT:
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    rp_vals = sorted(results_df["w_rp_pen"].unique())
    palette = plt.cm.plasma(np.linspace(0.1, 0.9, len(rp_vals)))  # type: ignore[attr-defined]

    for color, rp_val in zip(palette, rp_vals):
        sub = results_df[np.isclose(results_df["w_rp_pen"], rp_val)].sort_values("w_err")
        ls = "-" if np.isclose(rp_val, 0.4) else "--"
        lw = 2.2 if np.isclose(rp_val, 0.4) else 1.0
        ax.plot(sub["w_err"], sub["spearman_r"], color=color,
                linestyle=ls, linewidth=lw, label=f"w_rp={rp_val:.2f}")

    ax.axvline(0.5, color="red", linestyle=":", linewidth=1.5, label="Default w_err=0.5")
    ax.set_xlabel("w_err  (w_tfp = 1 − w_err)", fontsize=12)
    ax.set_ylabel("Spearman ρ with Overall_Score", fontsize=12)
    ax.set_title("CRS–Human Correlation vs ERR/TFP Weight Balance\n"
                 "(each line = fixed regression-penalty weight)", fontsize=12)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out = output_dir / "lineplot_werr_vs_correlation.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊  Line plot saved → {out}")


def plot_top_configs(results_df: pd.DataFrame, output_dir: Path, top_n: int = 20):
    """Horizontal bar chart of top-N configs by Spearman ρ."""
    if not HAS_PLOT:
        return

    top = results_df.nlargest(top_n, "spearman_r").copy()
    top["label"] = top.apply(
        lambda r: f"w_err={r.w_err:.2f}  w_rp={r.w_rp_pen:.2f}", axis=1
    )
    top = top.sort_values("spearman_r")

    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.38)))
    bars = ax.barh(top["label"], top["spearman_r"], color="steelblue", edgecolor="white")

    # Highlight default config
    default_label = "w_err=0.50  w_rp=0.40"
    for bar, label in zip(bars, top["label"]):
        if label == default_label:
            bar.set_color("tomato")
            bar.set_label("Default config")

    ax.set_xlabel("Spearman ρ", fontsize=11)
    ax.set_title(f"Top {top_n} CRS Weight Configurations\n(red = current default)", fontsize=11)
    ax.set_xlim([top["spearman_r"].min() * 0.97, min(1.0, top["spearman_r"].max() * 1.03)])
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()

    out = output_dir / "barplot_top_configs.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊  Bar chart saved → {out}")


# ============================================================================
# Report generation
# ============================================================================

def generate_text_report(
    results_df: pd.DataFrame,
    n_rows: int,
    default_cfg: dict,
    output_dir: Path,
) -> str:
    best = results_df.loc[results_df["spearman_r"].idxmax()]
    default_row = results_df[
        np.isclose(results_df["w_err"], default_cfg["w_err"]) &
        np.isclose(results_df["w_rp_pen"], default_cfg["w_rp_pen"])
    ]

    lines = [
        "=" * 80,
        "CRS WEIGHT SENSITIVITY ANALYSIS — RESULTS REPORT",
        "=" * 80,
        "",
        "STUDY DESIGN",
        "-" * 40,
        f"  Dataset rows (completed, with correction phase) : {n_rows}",
        f"  Weight configurations tested                    : {len(results_df)}",
        f"  Grid step size                                  : inferred from data",
        f"  Correlation targets                             : Overall_Score (ordinal 1-5)",
        f"  Primary metric                                  : Spearman ρ (rank correlation)",
        f"  Secondary metric                                : Pearson r",
        "",
        "FORMULA EVALUATED",
        "-" * 40,
        "  CRS = (w_err * ERR + (1-w_err) * TFP) * (1 - w_rp_pen * RP)",
        "  Constraint: w_err + w_tfp = 1.0  →  only 2 free parameters",
        "  Free parameters: w_err ∈ (0,1),  w_rp_pen ∈ [0, 0.8]",
        "",
        "DEFAULT CONFIGURATION",
        "-" * 40,
        f"  w_err    = {default_cfg['w_err']:.2f}",
        f"  w_tfp    = {1 - default_cfg['w_err']:.2f}   (implicit)",
        f"  w_rp_pen = {default_cfg['w_rp_pen']:.2f}",
    ]

    if len(default_row) > 0:
        dr = default_row.iloc[0]
        lines += [
            f"  Spearman ρ @ default  : {dr['spearman_r']:+.4f}  (p={dr['spearman_p']:.4f})",
            f"  Pearson  r @ default  : {dr['pearson_r']:+.4f}  (p={dr['pearson_p']:.4f})",
        ]
    else:
        lines.append("  (default config not on grid — use exact grid step to include it)")

    lines += [
        "",
        "BEST CONFIGURATION (by Spearman ρ)",
        "-" * 40,
        f"  w_err    = {best['w_err']:.2f}",
        f"  w_tfp    = {best['w_tfp']:.2f}",
        f"  w_rp_pen = {best['w_rp_pen']:.2f}",
        f"  Spearman ρ : {best['spearman_r']:+.4f}  (p={best['spearman_p']:.4f})",
        f"  Pearson  r : {best['pearson_r']:+.4f}  (p={best['pearson_p']:.4f})",
        "",
        "TOP-10 CONFIGURATIONS",
        "-" * 40,
    ]

    top10 = results_df.nlargest(10, "spearman_r")
    lines.append(f"  {'Rank':>4}  {'w_err':>6}  {'w_tfp':>6}  {'w_rp_pen':>8}  "
                 f"{'Spearman ρ':>10}  {'Pearson r':>9}  {'MAE':>7}  {'Note'}")
    lines.append("  " + "-" * 74)
    for rank, (_, row) in enumerate(top10.iterrows(), 1):
        is_default = (
            np.isclose(row["w_err"], default_cfg["w_err"]) and
            np.isclose(row["w_rp_pen"], default_cfg["w_rp_pen"])
        )
        note = "← CURRENT DEFAULT" if is_default else ""
        lines.append(
            f"  {rank:>4}  {row['w_err']:>6.2f}  {row['w_tfp']:>6.2f}  "
            f"{row['w_rp_pen']:>8.2f}  {row['spearman_r']:>+10.4f}  "
            f"{row['pearson_r']:>+9.4f}  {row['mae_vs_human']:>7.4f}  {note}"
        )

    # ── Sensitivity summary ───────────────────────────────────────────────────
    spearman_range = results_df["spearman_r"].max() - results_df["spearman_r"].min()
    spearman_std   = results_df["spearman_r"].std()

    lines += [
        "",
        "SENSITIVITY SUMMARY",
        "-" * 40,
        f"  Spearman ρ range across all configs : {spearman_range:.4f}",
        f"  Spearman ρ std  across all configs  : {spearman_std:.4f}",
    ]

    if spearman_range < 0.05:
        lines.append(
            "  FINDING: Metric is ROBUST — correlation varies < 0.05 across all tested "
            "weight combinations. The exact weights have minimal impact on ranking quality."
        )
    elif spearman_range < 0.15:
        lines.append(
            "  FINDING: Metric has LOW sensitivity — moderate variation suggests the "
            "default weights are a reasonable choice within a broad basin."
        )
    else:
        lines.append(
            "  FINDING: Metric has NOTABLE sensitivity — weight choice meaningfully "
            "affects correlation. Review the heatmap to confirm the default sits in a "
            "high-correlation region."
        )

    # ── ERR/TFP balance analysis ──────────────────────────────────────────────
    by_w_err = results_df.groupby("w_err")["spearman_r"].mean()
    best_w_err = by_w_err.idxmax()

    lines += [
        "",
        "ERR vs TFP BALANCE ANALYSIS",
        "-" * 40,
        f"  Mean Spearman ρ by w_err (averaged over all w_rp_pen values):",
    ]
    for w, corr in by_w_err.items():
        marker = "  ◄ best" if np.isclose(w, best_w_err) else ""
        lines.append(f"    w_err={w:.2f}  →  mean ρ = {corr:+.4f}{marker}")

    diff_from_balanced = abs(best_w_err - 0.5)
    lines += [
        "",
        f"  Optimal w_err (by mean ρ)  : {best_w_err:.2f}  "
        f"({'balanced' if diff_from_balanced < 0.1 else 'imbalanced'})",
        f"  Default  w_err             : {default_cfg['w_err']:.2f}",
    ]

    if diff_from_balanced < 0.15:
        lines.append(
            "  CONCLUSION: CRS performs best — or near-best — when ERR and TFP are "
            "balanced (w_err ≈ 0.5). This empirically validates the 50/50 split in "
            "the default configuration and provides scientific grounding for the choice."
        )
    else:
        lines.append(
            f"  CONCLUSION: Optimal balance is w_err={best_w_err:.2f}. Consider updating "
            f"the default from 0.50 to {best_w_err:.2f} to maximise correlation with "
            f"human judgement."
        )

    # ── RP penalty analysis ───────────────────────────────────────────────────
    by_rp = results_df.groupby("w_rp_pen")["spearman_r"].mean()
    best_rp = by_rp.idxmax()

    lines += [
        "",
        "REGRESSION PENALTY (w_rp_pen) ANALYSIS",
        "-" * 40,
        f"  Mean Spearman ρ by w_rp_pen (averaged over all w_err values):",
    ]
    for rp_val, corr in by_rp.items():
        marker = "  ◄ best" if np.isclose(rp_val, best_rp) else ""
        lines.append(f"    w_rp_pen={rp_val:.2f}  →  mean ρ = {corr:+.4f}{marker}")

    lines += [
        "",
        f"  Optimal w_rp_pen (by mean ρ) : {best_rp:.2f}",
        f"  Default  w_rp_pen            : {default_cfg['w_rp_pen']:.2f}",
    ]

    rp_diff = abs(best_rp - default_cfg["w_rp_pen"])
    if rp_diff < 0.15:
        lines.append(
            "  CONCLUSION: Default regression penalty (0.40) is close to empirically "
            "optimal. The aggressive penalty for introducing new errors is validated "
            "by the data."
        )
    else:
        lines.append(
            f"  CONCLUSION: Data suggests w_rp_pen={best_rp:.2f} maximises correlation. "
            f"Current default (0.40) differs by {rp_diff:.2f}. Consider updating."
        )

    lines += [
        "",
        "=" * 80,
        "END OF REPORT",
        "=" * 80,
    ]

    report_text = "\n".join(lines)
    out = output_dir / "ablation_report.txt"
    out.write_text(report_text, encoding="utf-8")
    print(f"  📄  Text report saved → {out}")
    return report_text


# ============================================================================
# JSON summary (machine-readable for downstream use)
# ============================================================================

def save_json_summary(
    results_df: pd.DataFrame,
    default_cfg: dict,
    output_dir: Path,
):
    best = results_df.loc[results_df["spearman_r"].idxmax()]

    default_row = results_df[
        np.isclose(results_df["w_err"], default_cfg["w_err"]) &
        np.isclose(results_df["w_rp_pen"], default_cfg["w_rp_pen"])
    ]

    summary: dict[str, Any] = {
        "study_metadata": {
            "total_configs_tested": len(results_df),
            "metric": "spearman_r vs Overall_Score",
        },
        "default_config": {
            "w_err":    default_cfg["w_err"],
            "w_tfp":    1.0 - default_cfg["w_err"],
            "w_rp_pen": default_cfg["w_rp_pen"],
            "spearman_r": float(default_row.iloc[0]["spearman_r"]) if len(default_row) > 0 else None,
            "pearson_r":  float(default_row.iloc[0]["pearson_r"])  if len(default_row) > 0 else None,
        },
        "best_config": {
            "w_err":       float(best["w_err"]),
            "w_tfp":       float(best["w_tfp"]),
            "w_rp_pen":    float(best["w_rp_pen"]),
            "spearman_r":  float(best["spearman_r"]),
            "pearson_r":   float(best["pearson_r"]),
            "mae_vs_human": float(best["mae_vs_human"]),
        },
        "sensitivity": {
            "spearman_range": float(results_df["spearman_r"].max() - results_df["spearman_r"].min()),
            "spearman_std":   float(results_df["spearman_r"].std()),
        },
        "top_10_configs": results_df.nlargest(10, "spearman_r")[
            ["w_err", "w_tfp", "w_rp_pen", "spearman_r", "pearson_r", "mae_vs_human"]
        ].to_dict(orient="records"),
    }

    out = output_dir / "ablation_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  📦  JSON summary saved → {out}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="CRS weight sensitivity / ablation study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--results-dir", required=True,
        help="Directory containing results.csv from a completed batch run"
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Where to save plots/reports (default: <results-dir>/ablation)"
    )
    parser.add_argument(
        "--grid-step", type=float, default=0.10,
        help="Step size for weight grid (default: 0.10; use 0.05 for finer resolution)"
    )
    parser.add_argument(
        "--default-w-err", type=float, default=0.50,
        help="Default w_err to annotate on plots (default: 0.50)"
    )
    parser.add_argument(
        "--default-w-rp", type=float, default=0.40,
        help="Default w_rp_pen to annotate on plots (default: 0.40)"
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir  = Path(args.output_dir) if args.output_dir else results_dir / "ablation"
    output_dir.mkdir(parents=True, exist_ok=True)

    default_cfg = {"w_err": args.default_w_err, "w_rp_pen": args.default_w_rp}

    print(f"\n{'='*70}")
    print("CRS WEIGHT SENSITIVITY ANALYSIS")
    print(f"{'='*70}")
    print(f"  Results dir  : {results_dir}")
    print(f"  Output dir   : {output_dir}")
    print(f"  Grid step    : {args.grid_step}")
    print(f"  Default cfg  : w_err={default_cfg['w_err']}, w_rp_pen={default_cfg['w_rp_pen']}")
    print(f"{'='*70}\n")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Step 1/5  Loading results …")
    df = load_results(results_dir)

    # ── Build grid ────────────────────────────────────────────────────────────
    print(f"Step 2/5  Building weight grid (step={args.grid_step}) …")
    weight_configs = build_weight_grid(step=args.grid_step)
    print(f"  ✓  {len(weight_configs)} weight configurations to test")

    # ── Run analysis ──────────────────────────────────────────────────────────
    print("Step 3/5  Computing correlations …")
    results_df = run_sensitivity_analysis(df, weight_configs)
    csv_out = output_dir / "ablation_full_results.csv"
    results_df.to_csv(csv_out, index=False)
    print(f"  ✓  Full results saved → {csv_out}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("Step 4/5  Generating visualisations …")
    if HAS_PLOT:
        sns.set_style("whitegrid")
        correlation_target = "human_correctness" if "human_correctness" in df.columns else "Overall_Score"
        plot_heatmaps(
            results_df, output_dir,
            correlation_target=correlation_target,
            default_w_err=default_cfg["w_err"],
            default_w_rp=default_cfg["w_rp_pen"],
        )
        plot_correlation_vs_w_err(results_df, output_dir)
        plot_top_configs(results_df, output_dir)
    else:
        print("  ⚠  Skipping plots (matplotlib not available)")

    # ── Report ────────────────────────────────────────────────────────────────
    print("Step 5/5  Writing report …")
    report = generate_text_report(results_df, len(df), default_cfg, output_dir)
    save_json_summary(results_df, default_cfg, output_dir)

    # ── Console summary ───────────────────────────────────────────────────────
    best = results_df.loc[results_df["spearman_r"].idxmax()]
    default_row = results_df[
        np.isclose(results_df["w_err"], default_cfg["w_err"]) &
        np.isclose(results_df["w_rp_pen"], default_cfg["w_rp_pen"])
    ]

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Configurations tested  : {len(results_df)}")
    if len(default_row) > 0:
        dr = default_row.iloc[0]
        print(f"  Default  ρ             : {dr['spearman_r']:+.4f}  "
              f"(w_err={dr['w_err']:.2f}, w_rp_pen={dr['w_rp_pen']:.2f})")
    print(f"  Best     ρ             : {best['spearman_r']:+.4f}  "
          f"(w_err={best['w_err']:.2f}, w_rp_pen={best['w_rp_pen']:.2f})")
    print(f"  ρ range (all configs)  : "
          f"{results_df['spearman_r'].min():+.4f} → {results_df['spearman_r'].max():+.4f}")
    print(f"  Output directory       : {output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()