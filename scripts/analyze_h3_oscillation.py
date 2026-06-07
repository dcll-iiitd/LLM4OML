#!/usr/bin/env python3
import argparse
import ast
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def analyze_h3_from_csvs(runs_dir: Path, output_dir: Path):
    """Fallback H3 Analysis using results.csv summary stats."""
    output_dir.mkdir(parents=True, exist_ok=True)
    all_dfs = []
    
    for run_dir in runs_dir.glob("werr*"):
        if not run_dir.is_dir(): continue
        csv_file = run_dir / "results.csv"
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            all_dfs.append(df)
            
    if not all_dfs:
        logger.error(f"No results.csv files found in {runs_dir}")
        return
        
    full_df = pd.concat(all_dfs, ignore_index=True)
    penalties = sorted(full_df['w_rp_pen'].unique())
    logger.info(f"Found {len(penalties)} penalty configurations: {penalties}")
    
    stats_rows = []
    for w in penalties:
        sub = full_df[full_df['w_rp_pen'] == w]
        stats_rows.append({
            'w_rp_pen': w,
            'mean_iterations': sub['total_iterations'].mean(),
            'mean_introduced_errors': sub['errors_introduced'].mean(), 
            'convergence_rate': sub['converged'].mean(),
            'samples': len(sub)
        })
        
    stats_df = pd.DataFrame(stats_rows)
    
    r_iter, p_iter = spearmanr(stats_df['w_rp_pen'], stats_df['mean_iterations'])
    r_err, p_err = spearmanr(stats_df['w_rp_pen'], stats_df['mean_introduced_errors'])
    
    with open(output_dir / "h3_report.txt", "w") as f:
        f.write("H3 HYPOTHESIS TEST (Using CSV Summaries)\n")
        f.write("=" * 80 + "\n")
        f.write("HYPOTHESIS:\n")
        f.write("  Higher regression penalty compresses oscillation and accelerates convergence.\n\n")
        f.write("NOTE ON DATASET:\n")
        f.write("  This dataset converges very quickly (mean ~1.1 iterations). Because most proofs\n")
        f.write("  don't require feedback loops, oscillation is exceedingly rare in this sample.\n")
        f.write("  True oscillation measurement requires deep trajectory logs (JSONs), but we can\n")
        f.write("  approximate it here using 'introduced errors' and 'total iterations'.\n\n")
        
        f.write(f"1. Convergence Speed Effect (Total Iterations):\n")
        f.write(f"   Spearman r: {r_iter:.4f}\n")
        f.write(f"   p-value:    {p_iter:.4f}\n")
        f.write(f"   (Negative r supports H3 = higher penalty reduces iterations)\n\n")
        
        f.write(f"2. Oscillation/Regression Proxy (Introduced Errors):\n")
        f.write(f"   Spearman r: {r_err:.4f}\n")
        f.write(f"   p-value:    {p_err:.4f}\n")
        f.write(f"   (Negative r supports H3 = higher penalty reduces regression)\n\n")
        
        f.write("Summary Statistics:\n")
        f.write("-" * 80 + "\n")
        f.write(stats_df.to_string(index=False) + "\n")
        
    logger.info(f"Report saved to {output_dir}/h3_report.txt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-results", type=Path, default=Path("batch_results"))
    parser.add_argument("--ablation-runs", type=Path, default=Path("ablation_out/runs"))
    parser.add_argument("--output-dir", type=Path, default=Path("h3_results"))
    args = parser.parse_args()
    
    # Try fully detailed JSOns if available with >1 penalty
    # But since user asked to look at ablation_runs, let's use that data source here.
    analyze_h3_from_csvs(args.ablation_runs, args.output_dir)
