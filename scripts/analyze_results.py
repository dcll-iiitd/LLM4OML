# ============================================================================
# File: scripts/analyze_results.py
# ============================================================================
"""Comprehensive result analysis with statistics and detailed reports."""

import pandas as pd
import numpy as np
import json
import argparse
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))


class ResultsAnalyzer:
    """Comprehensive analysis of batch processing results."""

    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.csv_path = self.results_dir / "results.csv"
        self.json_path = self.results_dir / "batch_execution_log.json"
        self.analysis_path = self.results_dir / "detailed_analysis.txt"

        if not self.csv_path.exists():
            raise FileNotFoundError(f"Results CSV not found: {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)
        self.batch_log = {}
        if self.json_path.exists():
            with open(self.json_path) as f:
                self.batch_log = json.load(f)

        self.completed = self.df[self.df["Processing_Status"] == "COMPLETED"].copy()

    def generate_full_report(self):
        print(f"\n{'='*80}")
        print("DETAILED ANALYSIS OF BATCH RESULTS")
        print(f"{'='*80}\n")

        if len(self.completed) == 0:
            print("❌ No completed algorithms to analyse.")
            return

        lines = []
        lines += self._header()
        lines += self._basic_stats()
        lines += self._verdict_distribution()
        lines += self._convergence_analysis()
        lines += self._crs_analysis()
        lines += self._error_analysis()
        if self.completed.get("Num_Judges", pd.Series([0])).max() > 1:
            lines += self._judge_analysis()
        lines += self._correlation_analysis()
        lines += self._performers()

        report = "\n".join(lines)
        self.analysis_path.write_text(report, encoding="utf-8")
        print(report)
        print(f"\n✅ Detailed analysis saved to: {self.analysis_path}")
        self._generate_visualizations()

    # ------------------------------------------------------------------

    def _header(self):
        return [
            "=" * 80,
            "COMPREHENSIVE ANALYSIS REPORT",
            "=" * 80,
            f"Generated    : {datetime.now():%Y-%m-%d %H:%M:%S}",
            f"Directory    : {self.results_dir}",
            f"Total        : {len(self.df)}",
            f"Completed    : {len(self.completed)}",
            f"Failed       : {(self.df['Processing_Status'] == 'FAILED').sum()}",
        ]

    def _basic_stats(self):
        df = self.completed
        return [
            f"\n{'='*80}", "1. PROCESSING STATISTICS", f"{'='*80}",
            f"  Total time   : {df['Processing_Time_Seconds'].sum():.1f}s "
            f"({df['Processing_Time_Seconds'].sum()/3600:.2f}h)",
            f"  Mean time    : {df['Processing_Time_Seconds'].mean():.1f}s",
            f"  Mean iter    : {df['Total_Iterations'].mean():.2f}",
            f"  Iter range   : {df['Total_Iterations'].min()}-{df['Total_Iterations'].max()}",
        ]

    def _verdict_distribution(self):
        df = self.completed
        lines = [f"\n{'='*80}", "2. VERDICT DISTRIBUTION", f"{'='*80}"]
        for label, col in [("V1 (Initial)", "V1_Verdict"), ("V2 (Final)", "V2_Verdict")]:
            lines.append(f"\n  {label}:")
            for v, c in df[col].value_counts().items():
                lines.append(f"    {v:15s}: {c:3d} ({c/len(df)*100:.1f}%)")
        return lines

    def _convergence_analysis(self):
        df = self.completed
        conv = df["Converged"].sum()
        lines = [
            f"\n{'='*80}", "3. CONVERGENCE ANALYSIS", f"{'='*80}",
            f"  Convergence rate: {conv}/{len(df)} ({conv/len(df)*100:.1f}%)",
        ]
        for n in sorted(df["Total_Iterations"].unique()):
            sub = df[df["Total_Iterations"] == n]
            c = sub["Converged"].sum()
            lines.append(f"  {n} iteration(s): {c}/{len(sub)} converged")
        return lines

    def _crs_analysis(self):
        df = self.completed
        lines = [
            f"\n{'='*80}", "4. CORRECTION REASONING SCORE (CRS)", f"{'='*80}",
            f"\n  CRS Statistics:",
            f"    Mean   : {df['CRS_Final'].mean():.3f}",
            f"    Median : {df['CRS_Final'].median():.3f}",
            f"    Std    : {df['CRS_Final'].std():.3f}",
            f"    Min    : {df['CRS_Final'].min():.3f}",
            f"    Max    : {df['CRS_Final'].max():.3f}",
            f"\n  Component Means:",
            f"    ERR (Error Resolution Rate) : {df['ERR_Final'].mean():.3f}",
            f"    RP  (Regression Penalty)    : {df['RP_Final'].mean():.3f}",
            f"    TFP (Targeted Fix Precision): {df['TFP_Final'].mean():.3f}",
            f"\n  CRS Distribution:",
            f"    ≥ 0.80 Strong   : {(df['CRS_Final']>=0.80).sum():3d} ({(df['CRS_Final']>=0.80).sum()/len(df)*100:.1f}%)",
            f"    ≥ 0.60 Good     : {((df['CRS_Final']>=0.60)&(df['CRS_Final']<0.80)).sum():3d} ({((df['CRS_Final']>=0.60)&(df['CRS_Final']<0.80)).sum()/len(df)*100:.1f}%)",
            f"    ≥ 0.40 Moderate : {((df['CRS_Final']>=0.40)&(df['CRS_Final']<0.60)).sum():3d} ({((df['CRS_Final']>=0.40)&(df['CRS_Final']<0.60)).sum()/len(df)*100:.1f}%)",
            f"    < 0.40 Weak     : {(df['CRS_Final']<0.40).sum():3d} ({(df['CRS_Final']<0.40).sum()/len(df)*100:.1f}%)",
        ]

        # Per-iteration CRS
        lines.append("\n  CRS Progression (mean over algorithms with data):")
        for col in ["CRS_Iter_1", "CRS_Iter_2", "CRS_Iter_3"]:
            nz = df[df[col] > 0][col]
            if len(nz) > 0:
                lines.append(f"    {col}: {nz.mean():.3f} (n={len(nz)})")
        return lines

    def _error_analysis(self):
        df = self.completed
        return [
            f"\n{'='*80}", "5. ERROR RESOLUTION", f"{'='*80}",
            f"  Total errors V1   : {df['Total_Errors_V1'].sum()}",
            f"  Total errors Final: {df['Total_Errors_Final'].sum()}",
            f"  Total fixed       : {df['Errors_Fixed'].sum()}",
            f"  Total introduced  : {df['Errors_Introduced'].sum()}",
            f"  Net improvement   : {df['Net_Improvement'].sum()}",
            f"\n  Per-algorithm averages:",
            f"    Avg errors V1   : {df['Total_Errors_V1'].mean():.2f}",
            f"    Avg errors final: {df['Total_Errors_Final'].mean():.2f}",
            f"    Avg fixed       : {df['Errors_Fixed'].mean():.2f}",
            f"    Avg introduced  : {df['Errors_Introduced'].mean():.2f}",
        ]

    def _judge_analysis(self):
        df = self.completed
        jdf = df[df["Num_Judges"] > 1]
        if len(jdf) == 0:
            return []
        return [
            f"\n{'='*80}", "6. MULTI-JUDGE ANALYSIS", f"{'='*80}",
            f"  Judges used       : {int(jdf['Num_Judges'].max())}",
            f"  Multi-judge algos : {len(jdf)}",
            f"  Mean JRS          : {jdf['Mean_JRS'].mean():.3f}",
            f"  Mean Weighted CFRS: {jdf['Weighted_CFRS'].mean():.3f}",
        ]

    def _correlation_analysis(self):
        df = self.completed
        pairs = [
            ("CRS_Final", "Overall_Score", "CRS vs Overall Score"),
            ("Total_Iterations", "CRS_Final", "Iterations vs CRS"),
            ("Total_Errors_V1", "CRS_Final", "Initial Errors vs CRS"),
            ("ERR_Final", "CRS_Final", "ERR vs CRS"),
        ]
        lines = [f"\n{'='*80}", "7. CORRELATIONS", f"{'='*80}"]
        for c1, c2, desc in pairs:
            if c1 in df and c2 in df:
                lines.append(f"  {desc:40s}: {df[c1].corr(df[c2]):+.3f}")
        return lines

    def _performers(self):
        df = self.completed
        lines = [f"\n{'='*80}", "8. TOP / BOTTOM PERFORMERS (by CRS)", f"{'='*80}"]
        for label, fn in [("Top 5", df.nlargest), ("Bottom 5", df.nsmallest)]:
            lines.append(f"\n  {label}:")
            for i, (_, row) in enumerate(fn(min(5, len(df)), "CRS_Final").iterrows(), 1):
                algo = str(row["Problem Statement"])[:55] + "..."
                lines.append(
                    f"  {i}. CRS={row['CRS_Final']:.3f} | "
                    f"ERR={row['ERR_Final']:.3f} RP={row['RP_Final']:.3f} "
                    f"TFP={row['TFP_Final']:.3f} | Score={row['Overall_Score']}"
                )
                lines.append(f"     {algo}")
        return lines

    def _generate_visualizations(self):
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            sns.set_style("whitegrid")
            df = self.completed
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle("Batch Processing Analysis (v3)", fontsize=16, fontweight="bold")

            # CRS distribution
            axes[0, 0].hist(df["CRS_Final"], bins=20, edgecolor="black")
            axes[0, 0].axvline(0.8, color="g", linestyle="--", label="Strong (≥0.8)")
            axes[0, 0].axvline(0.6, color="orange", linestyle="--", label="Good (≥0.6)")
            axes[0, 0].set_title("CRS Distribution")
            axes[0, 0].legend(fontsize=8)

            # Verdict distribution
            vc = df["V2_Verdict"].value_counts()
            axes[0, 1].bar(vc.index, vc.values)
            axes[0, 1].set_title("Final Verdict Distribution")
            axes[0, 1].tick_params(axis="x", rotation=45)

            # CRS vs Overall Score
            axes[0, 2].scatter(df["CRS_Final"], df["Overall_Score"], alpha=0.6)
            axes[0, 2].set_title("CRS vs Overall Score")

            # CRS components
            comps = {"ERR": df["ERR_Final"].mean(), "RP": df["RP_Final"].mean(), "TFP": df["TFP_Final"].mean()}
            axes[1, 0].bar(comps.keys(), comps.values(), color=["steelblue", "tomato", "seagreen"])
            axes[1, 0].set_title("Mean CRS Components")
            axes[1, 0].set_ylim([0, 1])

            # CRS progression per iteration
            iter_data = {}
            for col in ["CRS_Iter_1", "CRS_Iter_2", "CRS_Iter_3"]:
                nz = df[df[col] > 0][col]
                if len(nz) > 0:
                    iter_data[col.replace("CRS_Iter_", "Iter ")] = nz.mean()
            if iter_data:
                axes[1, 1].plot(list(iter_data.keys()), list(iter_data.values()), "o-")
                axes[1, 1].set_title("CRS Progression (mean)")
                axes[1, 1].set_ylim([0, 1])

            # Error resolution
            ed = {
                "V1 Errors": df["Total_Errors_V1"].sum(),
                "Final Errors": df["Total_Errors_Final"].sum(),
                "Fixed": df["Errors_Fixed"].sum(),
                "Introduced": df["Errors_Introduced"].sum(),
            }
            axes[1, 2].bar(ed.keys(), ed.values())
            axes[1, 2].set_title("Error Resolution Summary")
            axes[1, 2].tick_params(axis="x", rotation=30)

            plt.tight_layout()
            plot_path = self.results_dir / "analysis_plots.png"
            plt.savefig(plot_path, dpi=200, bbox_inches="tight")
            print(f"📊 Visualizations saved to: {plot_path}")
        except ImportError:
            print("⚠️  matplotlib/seaborn not available. Skipping visualizations.")
        except Exception as e:
            print(f"⚠️  Could not generate visualizations: {e}")


def main():
    parser = argparse.ArgumentParser(description="Analyse batch results")
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()

    try:
        ResultsAnalyzer(args.results_dir).generate_full_report()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()