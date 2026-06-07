#!/usr/bin/env python3
# ============================================================================
# scripts/run_ablation.py
# ============================================================================
"""
End-to-End CRS Weight Ablation Study
======================================

WHAT THIS DOES
--------------
Runs the full proof-generation + evaluation pipeline multiple times, once per
weight configuration, on the same input CSV.  For each run the CRS formula

    CRS = (w_err * ERR + w_tfp * TFP) * (1 - w_rp_pen * RP)

uses different weights so the routing decision ("should the prover correct
again?") and all CRS values are genuinely computed under those weights — not
just post-hoc re-calculations.

After all runs finish, a full analysis is produced:
  • heatmap_spearman.png   — 2-D Spearman ρ heat-map over (w_err, w_rp_pen)
  • heatmap_pearson.png    — same for Pearson r
  • lineplot_balance.png   — ρ vs w_err for each w_rp level (ERR/TFP balance)
  • barplot_top20.png      — top-20 configs ranked by ρ
  • convergence_rate.png   — convergence % per config
  • crs_mean_heatmap.png   — mean CRS achieved per config
  • ablation_results.csv   — full numeric table (one row per config)
  • ablation_report.txt    — human-readable interpretation
  • ablation_summary.json  — machine-readable summary

SETUP
-----
Two small files in src/graph/ must be replaced with the patched versions
that accept `crs_weights`.  The script auto-detects whether the patch is
already applied and exits with instructions if not.

USAGE
-----
    python scripts/run_ablation.py \\
        --input  data/algos.csv \\
        --config config/default.yaml \\
        --output-dir ablation_out

    # Finer grid (more LLM calls):
    python scripts/run_ablation.py \\
        --input data/algos.csv \\
        --config config/default.yaml \\
        --grid-step 0.1 \\
        --output-dir ablation_out

    # Dry-run: print configs that WOULD run, no LLM calls:
    python scripts/run_ablation.py \\
        --input data/algos.csv --config config/default.yaml \\
        --dry-run

    # Resume a partial run (skips configs already done):
    python scripts/run_ablation.py \\
        --input data/algos.csv --config config/default.yaml \\
        --output-dir ablation_out --resume

GRID
----
Constraint: w_err + w_tfp = 1.0.  Free parameters:
  w_err    ∈ {0.1, 0.2, …, 0.9}   (default step 0.2 → 5 values)
  w_rp_pen ∈ {0.0, 0.2, 0.4, 0.6, 0.8}  (default step 0.2 → 5 values)
  → 25 configurations × N algorithms = total LLM calls.

Always uses multi-judge as configured in default.yaml.
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from scipy import stats

# ── plotting (optional) ───────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

# ── project root on path ─────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

logging.basicConfig(
    level=logging.WARNING,          # suppress most pipeline noise
    format="%(levelname)s  %(name)s  %(message)s",
)
logger = logging.getLogger("ablation")


# ============================================================================
# Patch detection
# ============================================================================

def _check_patch_applied() -> bool:
    """
    Returns True iff WorkflowNodes.__init__ accepts a crs_weights parameter.
    This means the patched nodes.py and workflow.py are in place.
    """
    try:
        from src.graph.nodes import WorkflowNodes
        sig = inspect.signature(WorkflowNodes.__init__)
        return "crs_weights" in sig.parameters
    except Exception:
        return False


# ============================================================================
# Grid construction
# ============================================================================

def build_weight_grid(step: float = 0.2) -> List[Dict[str, float]]:
    """
    All (w_err, w_rp_pen) combos on a regular grid.
    w_tfp = 1 - w_err is derived automatically.
    """
    w_err_vals = np.round(np.arange(step, 1.0, step), 8).tolist()
    w_rp_vals  = np.round(np.arange(0.0, 0.85, step), 8).tolist()
    configs = []
    for w_err, w_rp in product(w_err_vals, w_rp_vals):
        configs.append({
            "w_err":    round(w_err, 4),
            "w_tfp":    round(1.0 - w_err, 4),
            "w_rp_pen": round(w_rp, 4),
        })
    return configs


def config_label(cfg: Dict[str, float]) -> str:
    return f"werr{cfg['w_err']:.2f}_wrp{cfg['w_rp_pen']:.2f}"


# ============================================================================
# YAML / CLI helpers  (mirrors run_batch.py logic)
# ============================================================================

def load_yaml(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        print(f"⚠  Config not found: {path}")
        return {}
    with open(p) as f:
        return yaml.safe_load(f) or {}


def _get(cfg: dict, *keys, default=None):
    node = cfg
    for k in keys:
        if not isinstance(node, dict):
            return default
        node = node.get(k, default)
        if node is default:
            return default
    return node


# ============================================================================
# Single weight-config batch run
# ============================================================================

def run_one_config(
    input_csv: str,
    output_dir: Path,
    cfg: Dict[str, float],
    yaml_cfg: Dict,
    api_key: Optional[str],
    max_iter: int,
    timeout: int,
    max_retries: int,
) -> Optional[pd.DataFrame]:
    """
    Runs the full BatchProcessor for ONE weight configuration.
    Returns a DataFrame of results, or None on total failure.
    """
    # Deferred import so the patch check can run before importing
    from src.graph.workflow import ProofWorkflow
    from src.config.temperatures import TemperatureConfig

    label   = config_label(cfg)
    run_dir = output_dir / "runs" / label
    run_dir.mkdir(parents=True, exist_ok=True)

    provider       = _get(yaml_cfg, "models", "prover", "provider") or "nvidia"
    prover_model   = _get(yaml_cfg, "models", "prover", "name") or "openai/gpt-oss-120b"
    judge_models   = _get(yaml_cfg, "models", "evaluators", "models") or [prover_model]
    use_multi_judge= bool(_get(yaml_cfg, "models", "evaluators", "use_multi_judge", default=True))

    temp_config = TemperatureConfig(
        generation  = float(_get(yaml_cfg, "temperatures", "generation",   default=0.7)),
        correction  = float(_get(yaml_cfg, "temperatures", "correction",   default=0.4)),
        evaluation  = float(_get(yaml_cfg, "temperatures", "evaluation",   default=0.05)),
        verification= float(_get(yaml_cfg, "temperatures", "verification", default=0.05)),
    )

    # ── inline batch processor (avoids subprocess overhead) ──────────────────
    try:
        df_in = pd.read_csv(input_csv)
    except Exception as exc:
        logger.error("Cannot read %s: %s", input_csv, exc)
        return None

    df_in = df_in[df_in["Problem Statement"].notna()].reset_index(drop=True)

    # Prepare results frame
    result_rows = []

    for idx, row in df_in.iterrows():
        algo_text  = str(row["Problem Statement"])
        assump_text= str(row.get("Assumptions Used", ""))
        algo_dir   = run_dir / f"algo_{idx:03d}"

        start = time.time()
        try:
            workflow = ProofWorkflow(
                provider_name   = provider,
                prover_model    = prover_model,
                evaluator_models= judge_models,
                api_key         = api_key,
                use_multi_judge = use_multi_judge,
                max_iterations  = max_iter,
                timeout         = timeout,
                max_retries     = max_retries,
                output_dir      = str(algo_dir),
                temp_config     = temp_config,
                crs_weights     = cfg,          # ← INJECT WEIGHTS
            )
            final_state, tracker = workflow.run(algo_text, assump_text)
            elapsed = time.time() - start
            exec_log = tracker.get_log()

            # Extract key metrics from execution log
            iterations = exec_log.get("iterations", [])
            final_results = exec_log.get("final_results", {})

            # Get final CRS components
            crs_final = err_final = rp_final = tfp_final = 0.0
            errors_fixed = errors_introduced = 0
            v1_verdict = v2_verdict = ""
            overall_score = 0
            total_errors_v1 = total_errors_final = 0

            if iterations:
                # V1
                ev1 = iterations[0].get("nodes", {}).get("evaluator", {})
                v1_verdict = ev1.get("verdict", "")
                es1 = ev1.get("error_set", {})
                total_errors_v1 = sum(len(v) for v in es1.values() if isinstance(v, list))

                # Final
                evf = iterations[-1].get("nodes", {}).get("evaluator", {})
                v2_verdict = evf.get("verdict", "")
                esf = evf.get("error_set", {})
                total_errors_final = sum(len(v) for v in esf.values() if isinstance(v, list))

                verdict_scores = {"PASS": 5, "PASS_MINOR": 4, "CONDITIONAL": 3, "FAIL": 2, "REJECT": 1}
                overall_score = verdict_scores.get(v2_verdict, 0)

                cm = evf.get("correction_metrics") or {}
                if cm:
                    crs_final       = cm.get("correction_reasoning_score", 0.0)
                    err_final       = cm.get("error_resolution_rate", 0.0)
                    rp_final        = cm.get("regression_penalty", 0.0)
                    tfp_final       = cm.get("targeted_fix_precision", 0.0)
                    errors_fixed    = cm.get("errors_fixed", 0)
                    errors_introduced = cm.get("errors_introduced", 0)

            result_rows.append({
                "weight_config":    label,
                "w_err":            cfg["w_err"],
                "w_tfp":            cfg["w_tfp"],
                "w_rp_pen":         cfg["w_rp_pen"],
                "algo_index":       idx,
                "problem_statement":algo_text[:80],
                "processing_status":"COMPLETED",
                "processing_time":  round(elapsed, 2),
                "total_iterations": len(iterations),
                "converged":        final_results.get("convergence_achieved", False),
                "v1_verdict":       v1_verdict,
                "v2_verdict":       v2_verdict,
                "overall_score":    overall_score,
                "crs_final":        round(crs_final, 4),
                "err_final":        round(err_final, 4),
                "rp_final":         round(rp_final, 4),
                "tfp_final":        round(tfp_final, 4),
                "errors_fixed":     errors_fixed,
                "errors_introduced":errors_introduced,
                "net_improvement":  errors_fixed - errors_introduced,
                "total_errors_v1":  total_errors_v1,
                "total_errors_final":total_errors_final,
            })

        except Exception as exc:
            elapsed = time.time() - start
            logger.error("[%s] algo %d failed: %s", label, idx, exc)
            result_rows.append({
                "weight_config":    label,
                "w_err":            cfg["w_err"],
                "w_tfp":            cfg["w_tfp"],
                "w_rp_pen":         cfg["w_rp_pen"],
                "algo_index":       idx,
                "problem_statement":str(row["Problem Statement"])[:80],
                "processing_status":"FAILED",
                "processing_time":  round(elapsed, 2),
                "total_iterations": 0,
                "converged":        False,
                "v1_verdict":       "",
                "v2_verdict":       "",
                "overall_score":    0,
                "crs_final":        0.0,
                "err_final":        0.0,
                "rp_final":         0.0,
                "tfp_final":        0.0,
                "errors_fixed":     0,
                "errors_introduced":0,
                "net_improvement":  0,
                "total_errors_v1":  0,
                "total_errors_final":0,
            })

    return pd.DataFrame(result_rows) if result_rows else None


# ============================================================================
# Per-config aggregation
# ============================================================================

def aggregate_config(df_run: pd.DataFrame) -> Dict[str, Any]:
    """Summarise one weight config's results into a single metrics row."""
    completed = df_run[df_run["processing_status"] == "COMPLETED"]
    if len(completed) == 0:
        return {}

    human = completed["overall_score"].values.astype(float)
    crs   = completed["crs_final"].values.astype(float)

    # Need ≥3 points for meaningful correlation
    if len(completed) >= 3 and crs.std() > 1e-9 and human.std() > 1e-9:
        sp_r, sp_p = stats.spearmanr(crs, human)
        pe_r, pe_p = stats.pearsonr(crs, human)
    else:
        sp_r = sp_p = pe_r = pe_p = float("nan")

    return {
        "w_err":          completed["w_err"].iloc[0],
        "w_tfp":          completed["w_tfp"].iloc[0],
        "w_rp_pen":       completed["w_rp_pen"].iloc[0],
        "n_completed":    len(completed),
        "n_failed":       (df_run["processing_status"] == "FAILED").sum(),
        "spearman_r":     round(sp_r, 6) if not np.isnan(sp_r) else None,
        "spearman_p":     round(sp_p, 6) if not np.isnan(sp_p) else None,
        "pearson_r":      round(pe_r, 6) if not np.isnan(pe_r) else None,
        "pearson_p":      round(pe_p, 6) if not np.isnan(pe_p) else None,
        "mean_crs":       round(completed["crs_final"].mean(), 4),
        "std_crs":        round(completed["crs_final"].std(), 4),
        "mean_err":       round(completed["err_final"].mean(), 4),
        "mean_rp":        round(completed["rp_final"].mean(), 4),
        "mean_tfp":       round(completed["tfp_final"].mean(), 4),
        "convergence_rate":round(completed["converged"].mean(), 4),
        "mean_iterations":round(completed["total_iterations"].mean(), 2),
        "mean_errors_fixed":round(completed["errors_fixed"].mean(), 2),
        "mean_errors_introduced":round(completed["errors_introduced"].mean(), 2),
        "mean_net_improvement":round(completed["net_improvement"].mean(), 2),
    }

# ============================================================================
# Visualisations
# ============================================================================

def _pivot(agg_df: pd.DataFrame, metric: str):
    w_err_vals = sorted(agg_df["w_err"].unique())
    w_rp_vals  = sorted(agg_df["w_rp_pen"].unique())
    matrix = (
        agg_df.pivot(index="w_rp_pen", columns="w_err", values=metric)
        .reindex(index=w_rp_vals, columns=w_err_vals)
    )
    return matrix, w_err_vals, w_rp_vals


def _annotate_default(ax, agg_df, default_w_err, default_w_rp):
    """Draw a red rectangle around the default operating point if on grid."""
    w_err_vals = sorted(agg_df["w_err"].unique())
    w_rp_vals  = sorted(agg_df["w_rp_pen"].unique())
    try:
        xi = min(range(len(w_err_vals)), key=lambda i: abs(w_err_vals[i] - default_w_err))
        yi = min(range(len(w_rp_vals)),  key=lambda i: abs(w_rp_vals[i]  - default_w_rp))
        ax.add_patch(plt.Rectangle((xi, yi), 1, 1, fill=False, edgecolor="red", lw=2.5))
        ax.plot(xi + 0.5, yi + 0.5, "r*", markersize=14, zorder=5,
                label=f"Default ({default_w_err:.1f}, {default_w_rp:.1f})")
        ax.legend(loc="upper right", fontsize=9)
    except Exception:
        pass


def plot_heatmaps(agg_df: pd.DataFrame, output_dir: Path, default_w_err=0.5, default_w_rp=0.4):
    if not HAS_PLOT:
        return
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(
        "CRS Weight Ablation — Correlation with Judge-Derived Human Score\n"
        "(each cell = full pipeline run with those weights)",
        fontsize=13, fontweight="bold"
    )
    for ax, metric, title, fmt in [
        (axes[0], "spearman_r", "Spearman ρ  (primary — ordinal)", ".3f"),
        (axes[1], "pearson_r",  "Pearson r   (secondary)",          ".3f"),
    ]:
        valid = agg_df[agg_df[metric].notna()]
        if valid.empty:
            ax.set_title(f"{title}\n(no data)")
            continue
        matrix, w_err_vals, w_rp_vals = _pivot(valid, metric)
        x_labels = [f"{v:.2f}" for v in w_err_vals]
        y_labels = [f"{v:.2f}" for v in w_rp_vals]
        vmin = valid[metric].min()
        vmax = valid[metric].max()
        sns.heatmap(
            matrix, ax=ax, annot=True, fmt=fmt,
            cmap="RdYlGn", vmin=vmin, vmax=vmax,
            linewidths=0.4, linecolor="white",
            xticklabels=x_labels, yticklabels=y_labels,
            annot_kws={"size": 9},
        )
        ax.set_xlabel("w_err  (w_tfp = 1 − w_err)", fontsize=11)
        ax.set_ylabel("w_rp_pen  (regression penalty)", fontsize=11)
        ax.set_title(title, fontsize=11)
        _annotate_default(ax, valid, default_w_err, default_w_rp)
        ax.tick_params(axis="x", rotation=45, labelsize=9)
        ax.tick_params(axis="y", rotation=0,  labelsize=9)
    plt.tight_layout()
    p = output_dir / "heatmap_correlation.png"
    plt.savefig(p, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  📊  Correlation heatmap  → {p}")


def plot_crs_mean_heatmap(agg_df: pd.DataFrame, output_dir: Path, default_w_err=0.5, default_w_rp=0.4):
    if not HAS_PLOT or agg_df["mean_crs"].isna().all():
        return
    fig, ax = plt.subplots(figsize=(10, 7))
    valid = agg_df[agg_df["mean_crs"].notna()]
    matrix, w_err_vals, w_rp_vals = _pivot(valid, "mean_crs")
    x_labels = [f"{v:.2f}" for v in w_err_vals]
    y_labels = [f"{v:.2f}" for v in w_rp_vals]
    sns.heatmap(
        matrix, ax=ax, annot=True, fmt=".3f",
        cmap="Blues", vmin=0, vmax=1,
        linewidths=0.4, linecolor="white",
        xticklabels=x_labels, yticklabels=y_labels,
        annot_kws={"size": 9},
    )
    ax.set_xlabel("w_err  (w_tfp = 1 − w_err)", fontsize=11)
    ax.set_ylabel("w_rp_pen", fontsize=11)
    ax.set_title("Mean CRS Achieved per Weight Configuration\n"
                 "(higher ≠ better — must correlate with human score)", fontsize=11)
    _annotate_default(ax, valid, default_w_err, default_w_rp)
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.tick_params(axis="y", rotation=0,  labelsize=9)
    plt.tight_layout()
    p = output_dir / "heatmap_mean_crs.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊  Mean-CRS heatmap     → {p}")


def plot_convergence_heatmap(agg_df: pd.DataFrame, output_dir: Path, default_w_err=0.5, default_w_rp=0.4):
    if not HAS_PLOT or agg_df["convergence_rate"].isna().all():
        return
    fig, ax = plt.subplots(figsize=(10, 7))
    valid = agg_df[agg_df["convergence_rate"].notna()]
    matrix, w_err_vals, w_rp_vals = _pivot(valid, "convergence_rate")
    x_labels = [f"{v:.2f}" for v in w_err_vals]
    y_labels = [f"{v:.2f}" for v in w_rp_vals]
    sns.heatmap(
        matrix, ax=ax, annot=True, fmt=".2f",
        cmap="YlGn", vmin=0, vmax=1,
        linewidths=0.4, linecolor="white",
        xticklabels=x_labels, yticklabels=y_labels,
        annot_kws={"size": 9},
    )
    ax.set_xlabel("w_err  (w_tfp = 1 − w_err)", fontsize=11)
    ax.set_ylabel("w_rp_pen", fontsize=11)
    ax.set_title("Convergence Rate per Weight Configuration\n"
                 "(fraction of algorithms reaching PASS/PASS_MINOR)", fontsize=11)
    _annotate_default(ax, valid, default_w_err, default_w_rp)
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.tick_params(axis="y", rotation=0,  labelsize=9)
    plt.tight_layout()
    p = output_dir / "heatmap_convergence.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊  Convergence heatmap  → {p}")


def plot_lineplot_balance(agg_df: pd.DataFrame, output_dir: Path):
    if not HAS_PLOT:
        return
    valid = agg_df[agg_df["spearman_r"].notna()]
    if valid.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    rp_vals = sorted(valid["w_rp_pen"].unique())
    palette = plt.cm.plasma(np.linspace(0.1, 0.9, len(rp_vals)))
    for color, rp_val in zip(palette, rp_vals):
        sub = valid[np.isclose(valid["w_rp_pen"], rp_val)].sort_values("w_err")
        is_default = np.isclose(rp_val, 0.4)
        ax.plot(sub["w_err"], sub["spearman_r"],
                color=color,
                linestyle="-" if is_default else "--",
                linewidth=2.5 if is_default else 1.0,
                marker="o", markersize=5,
                label=f"w_rp={rp_val:.2f}{'  ← default' if is_default else ''}")
    ax.axvline(0.5, color="red", linestyle=":", linewidth=1.5, label="Default w_err=0.5")
    ax.set_xlabel("w_err  (w_tfp = 1 − w_err)", fontsize=12)
    ax.set_ylabel("Spearman ρ with Overall_Score", fontsize=12)
    ax.set_title("CRS–Human Correlation vs ERR/TFP Balance\n"
                 "Each line = fixed regression-penalty weight; solid = default (0.40)", fontsize=11)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8, ncol=1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = output_dir / "lineplot_balance.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊  Balance line plot    → {p}")


def plot_top_configs(agg_df: pd.DataFrame, output_dir: Path, top_n: int = 20):
    if not HAS_PLOT:
        return
    valid = agg_df[agg_df["spearman_r"].notna()]
    if valid.empty:
        return
    n = min(top_n, len(valid))
    top = valid.nlargest(n, "spearman_r").copy()
    top["label"] = top.apply(
        lambda r: f"w_err={r.w_err:.2f}  w_rp={r.w_rp_pen:.2f}", axis=1
    )
    top = top.sort_values("spearman_r")

    fig, ax = plt.subplots(figsize=(9, max(4, n * 0.42)))
    bars = ax.barh(top["label"], top["spearman_r"], color="steelblue", edgecolor="white")
    default_label = "w_err=0.50  w_rp=0.40"
    for bar, lbl in zip(bars, top["label"]):
        if lbl == default_label:
            bar.set_color("tomato")
    ax.set_xlabel("Spearman ρ  (CRS vs Overall_Score)", fontsize=11)
    ax.set_title(f"Top {n} CRS Weight Configurations\n(red = current default 0.50 / 0.40)",
                 fontsize=11)
    lo = top["spearman_r"].min()
    hi = top["spearman_r"].max()
    ax.set_xlim([lo - 0.02, min(1.0, hi + 0.02)])
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    p = output_dir / "barplot_top_configs.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊  Top-configs bar plot → {p}")


# ============================================================================
# Text report
# ============================================================================

def generate_report(
    agg_df: pd.DataFrame,
    n_configs_run: int,
    n_algorithms: int,
    default_cfg: Dict,
    output_dir: Path,
) -> str:
    valid = agg_df[agg_df["spearman_r"].notna()]
    if valid.empty:
        report = "No valid correlation data — all runs may have failed or had insufficient variance.\n"
        (output_dir / "ablation_report.txt").write_text(report)
        return report

    best = valid.loc[valid["spearman_r"].idxmax()]
    default_row = valid[
        np.isclose(valid["w_err"], default_cfg["w_err"]) &
        np.isclose(valid["w_rp_pen"], default_cfg["w_rp_pen"])
    ]

    lines = [
        "=" * 80,
        "END-TO-END CRS WEIGHT ABLATION STUDY — RESULTS REPORT",
        "=" * 80,
        "",
        "STUDY DESIGN",
        "-" * 40,
        f"  Algorithms processed per config : {n_algorithms}",
        f"  Weight configurations tested    : {n_configs_run}",
        f"  Total pipeline runs             : {n_configs_run * n_algorithms}",
        f"  Human-proxy target              : Overall_Score (ordinal 1=REJECT … 5=PASS)",
        f"  Primary correlation metric      : Spearman ρ (rank correlation, ordinal-safe)",
        f"  Secondary metric                : Pearson r",
        "",
        "CRS FORMULA",
        "-" * 40,
        "  CRS = (w_err * ERR + w_tfp * TFP) * (1 - w_rp_pen * RP)",
        "  Constraint: w_err + w_tfp = 1.0  →  2 free parameters",
        "",
        "DEFAULT CONFIGURATION (from config/default.yaml)",
        "-" * 40,
        f"  w_err    = {default_cfg['w_err']:.2f}",
        f"  w_tfp    = {1 - default_cfg['w_err']:.2f}  (implicit)",
        f"  w_rp_pen = {default_cfg['w_rp_pen']:.2f}",
    ]

    if len(default_row) > 0:
        dr = default_row.iloc[0]
        lines += [
            f"  Spearman ρ  : {dr['spearman_r']:+.4f}  (p = {dr['spearman_p']:.4f})",
            f"  Pearson  r  : {dr['pearson_r']:+.4f}  (p = {dr['pearson_p']:.4f})",
            f"  Mean CRS    : {dr['mean_crs']:.4f}",
            f"  Conv. rate  : {dr['convergence_rate']*100:.1f}%",
        ]
    else:
        lines.append("  (default config not on grid — use --grid-step 0.1 to include it)")

    lines += [
        "",
        "BEST CONFIGURATION (highest Spearman ρ)",
        "-" * 40,
        f"  w_err    = {best['w_err']:.2f}",
        f"  w_tfp    = {best['w_tfp']:.2f}",
        f"  w_rp_pen = {best['w_rp_pen']:.2f}",
        f"  Spearman ρ  : {best['spearman_r']:+.4f}  (p = {best['spearman_p']:.4f})",
        f"  Pearson  r  : {best['pearson_r']:+.4f}  (p = {best['pearson_p']:.4f})",
        f"  Mean CRS    : {best['mean_crs']:.4f}",
        f"  Conv. rate  : {best['convergence_rate']*100:.1f}%",
        "",
        "TOP-10 CONFIGURATIONS",
        "-" * 40,
        f"  {'Rk':>2}  {'w_err':>6}  {'w_tfp':>6}  {'w_rp':>6}  "
        f"{'Spear.ρ':>8}  {'Pearson':>8}  {'mCRS':>6}  {'Conv%':>6}  Note",
        "  " + "-" * 76,
    ]

    for rk, (_, row) in enumerate(valid.nlargest(10, "spearman_r").iterrows(), 1):
        is_def = (
            np.isclose(row["w_err"], default_cfg["w_err"]) and
            np.isclose(row["w_rp_pen"], default_cfg["w_rp_pen"])
        )
        note = " ← DEFAULT" if is_def else ""
        lines.append(
            f"  {rk:>2}  {row['w_err']:>6.2f}  {row['w_tfp']:>6.2f}  {row['w_rp_pen']:>6.2f}  "
            f"{row['spearman_r']:>+8.4f}  {row['pearson_r']:>+8.4f}  "
            f"{row['mean_crs']:>6.4f}  {row['convergence_rate']*100:>5.1f}%{note}"
        )

    # Sensitivity
    sp_range = valid["spearman_r"].max() - valid["spearman_r"].min()
    sp_std   = valid["spearman_r"].std()

    lines += [
        "",
        "SENSITIVITY ANALYSIS",
        "-" * 40,
        f"  Spearman ρ range : {valid['spearman_r'].min():+.4f} → {valid['spearman_r'].max():+.4f}  "
        f"(span {sp_range:.4f})",
        f"  Spearman ρ std   : {sp_std:.4f}",
    ]
    if sp_range < 0.05:
        lines.append("  VERDICT: ROBUST — correlation varies < 0.05. Weights have minimal "
                     "impact on ranking quality; default is well-justified.")
    elif sp_range < 0.15:
        lines.append("  VERDICT: LOW SENSITIVITY — moderate variation. Default weights sit "
                     "in a reasonably flat basin; choice is defensible.")
    else:
        lines.append("  VERDICT: NOTABLE SENSITIVITY — weight choice meaningfully changes "
                     "correlation. Heatmap shows where optimal region lies.")

    # ERR/TFP balance section
    by_w_err = valid.groupby("w_err")["spearman_r"].mean()
    best_w_err = by_w_err.idxmax()
    lines += [
        "",
        "ERR vs TFP BALANCE (averaged over all w_rp_pen levels)",
        "-" * 40,
    ]
    for w, corr in by_w_err.items():
        mark = "  ◄ best" if np.isclose(w, best_w_err) else ""
        d_mark = "  ← default" if np.isclose(w, default_cfg["w_err"]) else ""
        lines.append(f"  w_err={w:.2f} (w_tfp={1-w:.2f})  →  mean ρ = {corr:+.4f}{mark}{d_mark}")

    diff = abs(best_w_err - 0.5)
    if diff < 0.12:
        lines += [
            "",
            "  CONCLUSION: CRS performs best when ERR and TFP are balanced (w_err ≈ 0.5).",
            "  This empirically validates the 50/50 split in the default configuration.",
        ]
    else:
        lines += [
            "",
            f"  CONCLUSION: Optimal w_err = {best_w_err:.2f}. Current default (0.50) differs by "
            f"{diff:.2f}. Consider updating weights to improve metric validity.",
        ]

    # RP penalty section
    by_rp = valid.groupby("w_rp_pen")["spearman_r"].mean()
    best_rp = by_rp.idxmax()
    lines += [
        "",
        "REGRESSION PENALTY (averaged over all w_err levels)",
        "-" * 40,
    ]
    for rp_val, corr in by_rp.items():
        mark = "  ◄ best" if np.isclose(rp_val, best_rp) else ""
        d_mark = "  ← default" if np.isclose(rp_val, default_cfg["w_rp_pen"]) else ""
        lines.append(f"  w_rp_pen={rp_val:.2f}  →  mean ρ = {corr:+.4f}{mark}{d_mark}")

    rp_diff = abs(best_rp - default_cfg["w_rp_pen"])
    if rp_diff < 0.15:
        lines += [
            "",
            f"  CONCLUSION: Default regression penalty ({default_cfg['w_rp_pen']:.2f}) is "
            "close to empirically optimal. Aggressive penalty for introducing new errors "
            "is supported by the data.",
        ]
    else:
        lines += [
            "",
            f"  CONCLUSION: Data suggests w_rp_pen = {best_rp:.2f} maximises correlation. "
            f"Default (0.40) differs by {rp_diff:.2f}.",
        ]

    lines += ["", "=" * 80, "END OF REPORT", "=" * 80]
    text = "\n".join(lines)
    (output_dir / "ablation_report.txt").write_text(text, encoding="utf-8")
    print(f"  📄  Report              → {output_dir / 'ablation_report.txt'}")
    return text


# ============================================================================
# JSON summary
# ============================================================================

def save_json_summary(agg_df: pd.DataFrame, default_cfg: Dict, output_dir: Path):
    valid = agg_df[agg_df["spearman_r"].notna()]
    if valid.empty:
        return
    best = valid.loc[valid["spearman_r"].idxmax()]
    default_row = valid[
        np.isclose(valid["w_err"], default_cfg["w_err"]) &
        np.isclose(valid["w_rp_pen"], default_cfg["w_rp_pen"])
    ]

    summary = {
        "generated_at": datetime.now().isoformat(),
        "configs_tested": len(agg_df),
        "default_config": {
            "w_err": default_cfg["w_err"],
            "w_tfp": round(1 - default_cfg["w_err"], 4),
            "w_rp_pen": default_cfg["w_rp_pen"],
            "spearman_r": float(default_row.iloc[0]["spearman_r"]) if len(default_row) > 0 else None,
            "pearson_r":  float(default_row.iloc[0]["pearson_r"])  if len(default_row) > 0 else None,
        },
        "best_config": {
            "w_err": float(best["w_err"]),
            "w_tfp": float(best["w_tfp"]),
            "w_rp_pen": float(best["w_rp_pen"]),
            "spearman_r": float(best["spearman_r"]),
            "pearson_r":  float(best["pearson_r"]),
            "mean_crs":   float(best["mean_crs"]),
            "convergence_rate": float(best["convergence_rate"]),
        },
        "sensitivity": {
            "spearman_range": float(valid["spearman_r"].max() - valid["spearman_r"].min()),
            "spearman_std":   float(valid["spearman_r"].std()),
        },
        "top_10": valid.nlargest(10, "spearman_r")[
            ["w_err", "w_tfp", "w_rp_pen", "spearman_r", "pearson_r",
             "mean_crs", "convergence_rate"]
        ].to_dict(orient="records"),
    }
    p = output_dir / "ablation_summary.json"
    with open(p, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  📦  JSON summary        → {p}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="End-to-end CRS weight ablation study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input",      required=True,
                        help="Input CSV (same format as run_batch.py)")
    parser.add_argument("--config",     required=True,
                        help="YAML config (e.g. config/default.yaml)")
    parser.add_argument("--output-dir", default="ablation_out",
                        help="Where to store all run outputs + analysis")
    parser.add_argument("--api-key",    default=None,
                        help="API key (overrides .env)")
    parser.add_argument("--grid-step",  type=float, default=0.2,
                        help="Weight grid step (default 0.2 → 25 configs; "
                             "0.1 → 81 configs)")
    parser.add_argument("--max-iter",   type=int,   default=None,
                        help="Override max iterations (default: from YAML)")
    parser.add_argument("--timeout",    type=int,   default=300)
    parser.add_argument("--max-retries",type=int,   default=3)
    parser.add_argument("--default-w-err", type=float, default=0.5,
                        help="Default w_err to annotate on plots")
    parser.add_argument("--default-w-rp",  type=float, default=0.4,
                        help="Default w_rp_pen to annotate on plots")
    parser.add_argument("--dry-run",    action="store_true",
                        help="Print configs that would run, then exit")
    parser.add_argument("--resume",     action="store_true",
                        help="Skip configs whose result CSV already exists")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Skip running — only re-generate analysis from existing CSVs")
    args = parser.parse_args()

    load_dotenv()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Patch check ────────────────────────────────────────────────────────────
    if not args.analyze_only:
        if not _check_patch_applied():
            print("\n" + "=" * 70)
            print("ERROR: Patched src/graph/nodes.py and workflow.py not found.")
            print()
            print("Replace the two files with the patched versions that accept")
            print("the `crs_weights` parameter.  Both files are provided alongside")
            print("this script.  Then re-run.")
            print("=" * 70 + "\n")
            sys.exit(1)

    # ── Load YAML ──────────────────────────────────────────────────────────────
    yaml_cfg = load_yaml(args.config)
    max_iter = args.max_iter or int(_get(yaml_cfg, "workflow", "max_iterations", default=3))

    # ── Build grid ─────────────────────────────────────────────────────────────
    weight_configs = build_weight_grid(step=args.grid_step)
    n_algos = len(pd.read_csv(args.input).dropna(subset=["Problem Statement"]))

    print(f"\n{'='*70}")
    print("END-TO-END CRS WEIGHT ABLATION STUDY")
    print(f"{'='*70}")
    print(f"  Input CSV       : {args.input}  ({n_algos} algorithms)")
    print(f"  Config          : {args.config}")
    print(f"  Output dir      : {output_dir}")
    print(f"  Grid step       : {args.grid_step}")
    print(f"  Configs to run  : {len(weight_configs)}")
    print(f"  Total LLM runs  : {len(weight_configs) * n_algos}")
    print(f"  Max iterations  : {max_iter}")
    print(f"{'='*70}\n")

    if args.dry_run:
        print("DRY RUN — weight configurations that would be tested:")
        for i, cfg in enumerate(weight_configs, 1):
            print(f"  {i:>3}.  w_err={cfg['w_err']:.2f}  w_tfp={cfg['w_tfp']:.2f}  "
                  f"w_rp_pen={cfg['w_rp_pen']:.2f}  [{config_label(cfg)}]")
        print(f"\nTotal: {len(weight_configs)} configs × {n_algos} algorithms "
              f"= {len(weight_configs)*n_algos} runs")
        sys.exit(0)

    # ── Run loop ───────────────────────────────────────────────────────────────
    agg_records = []
    raw_csv_path = output_dir / "ablation_raw_all.csv"
    all_raw = []

    if not args.analyze_only:
        for run_idx, cfg in enumerate(weight_configs, 1):
            label = config_label(cfg)
            run_csv = output_dir / "runs" / label / "results.csv"

            if args.resume and run_csv.exists():
                print(f"[{run_idx}/{len(weight_configs)}]  SKIP (resume)  {label}")
                df_run = pd.read_csv(run_csv)
            else:
                print(f"\n[{run_idx}/{len(weight_configs)}]  Running  {label}  "
                      f"(w_err={cfg['w_err']:.2f}  w_tfp={cfg['w_tfp']:.2f}  "
                      f"w_rp_pen={cfg['w_rp_pen']:.2f})")
                t0 = time.time()
                df_run = run_one_config(
                    input_csv   = args.input,
                    output_dir  = output_dir,
                    cfg         = cfg,
                    yaml_cfg    = yaml_cfg,
                    api_key     = args.api_key,
                    max_iter    = max_iter,
                    timeout     = args.timeout,
                    max_retries = args.max_retries,
                )
                elapsed = time.time() - t0
                print(f"  ✓  Finished in {elapsed:.1f}s")

                if df_run is not None:
                    (output_dir / "runs" / label).mkdir(parents=True, exist_ok=True)
                    df_run.to_csv(run_csv, index=False)

            if df_run is not None:
                all_raw.append(df_run)
                rec = aggregate_config(df_run)
                if rec:
                    agg_records.append(rec)

        if all_raw:
            pd.concat(all_raw, ignore_index=True).to_csv(raw_csv_path, index=False)
            print(f"\n  💾  Raw data saved → {raw_csv_path}")
    else:
        # --analyze-only: load existing CSVs
        print("ANALYZE-ONLY mode — loading existing run CSVs …")
        for cfg in weight_configs:
            label = config_label(cfg)
            run_csv = output_dir / "runs" / label / "results.csv"
            if run_csv.exists():
                df_run = pd.read_csv(run_csv)
                all_raw.append(df_run)
                rec = aggregate_config(df_run)
                if rec:
                    agg_records.append(rec)
        if all_raw:
            pd.concat(all_raw, ignore_index=True).to_csv(raw_csv_path, index=False)

    if not agg_records:
        print("\n❌  No aggregation data — all runs may have failed. Exiting.")
        sys.exit(1)

    # ── Aggregate results ──────────────────────────────────────────────────────
    agg_df = pd.DataFrame(agg_records)
    agg_csv = output_dir / "ablation_results.csv"
    agg_df.to_csv(agg_csv, index=False)
    print(f"\n  💾  Aggregated results  → {agg_csv}")

    # ── Plots ──────────────────────────────────────────────────────────────────
    print("\nGenerating visualisations …")
    if HAS_PLOT:
        sns.set_style("whitegrid")
    default_cfg = {"w_err": args.default_w_err, "w_rp_pen": args.default_w_rp}
    plot_heatmaps(agg_df, output_dir, args.default_w_err, args.default_w_rp)
    plot_crs_mean_heatmap(agg_df, output_dir, args.default_w_err, args.default_w_rp)
    plot_convergence_heatmap(agg_df, output_dir, args.default_w_err, args.default_w_rp)
    plot_lineplot_balance(agg_df, output_dir)
    plot_top_configs(agg_df, output_dir)

    # ── Report ─────────────────────────────────────────────────────────────────
    print("\nGenerating report …")
    report = generate_report(agg_df, len(weight_configs), n_algos, default_cfg, output_dir)
    save_json_summary(agg_df, default_cfg, output_dir)

    # ── Console summary ────────────────────────────────────────────────────────
    valid = agg_df[agg_df["spearman_r"].notna()]
    print(f"\n{'='*70}")
    print("ABLATION COMPLETE")
    print(f"{'='*70}")
    if not valid.empty:
        best = valid.loc[valid["spearman_r"].idxmax()]
        default_row = valid[
            np.isclose(valid["w_err"], args.default_w_err) &
            np.isclose(valid["w_rp_pen"], args.default_w_rp)
        ]
        if len(default_row) > 0:
            dr = default_row.iloc[0]
            print(f"  Default config ρ  : {dr['spearman_r']:+.4f}  "
                  f"(w_err={dr['w_err']:.2f}, w_rp={dr['w_rp_pen']:.2f})")
        print(f"  Best config ρ     : {best['spearman_r']:+.4f}  "
              f"(w_err={best['w_err']:.2f}, w_rp={best['w_rp_pen']:.2f})")
        print(f"  ρ range           : {valid['spearman_r'].min():+.4f} → "
              f"{valid['spearman_r'].max():+.4f}")
    print(f"  Output directory  : {output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()