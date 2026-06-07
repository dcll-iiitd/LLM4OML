import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

def analyze_h3_from_csv(ablation_runs_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    all_data = []

    for run_dir in ablation_runs_dir.glob("*"):
        if not run_dir.is_dir(): continue
        csv_file = run_dir / "results.csv"
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            all_data.append(df)

    if not all_data:
        print("No results.csv found in ablation runs directories.")
        return

    full_df = pd.concat(all_data, ignore_index=True)
    
    # Analyze by regression penalty weight
    penalties = sorted(full_df['w_rp_pen'].unique())
    print(f"Found {len(penalties)} penalty configurations: {penalties}")
    
    if len(penalties) < 3:
        print("Need at least 3 penalty configurations to accurately test monotonic correlation (H3).")
        return
        
    stats_by_penalty = []
    
    for w_rp_pen in penalties:
        sub_df = full_df[full_df['w_rp_pen'] == w_rp_pen]
        stats_by_penalty.append({
            'w_rp_pen': w_rp_pen,
            'mean_iterations': sub_df['total_iterations'].mean(),
            'mean_introduced_errors': sub_df['errors_introduced'].mean(),
            'conv_rate': sub_df['converged'].mean(),
            'n_samples': len(sub_df)
        })

    stats_df = pd.DataFrame(stats_by_penalty)
    
    # H3 Tests
    # 1. Does higher penalty reduce iterations?
    r_iter, p_iter = spearmanr(stats_df['w_rp_pen'], stats_df['mean_iterations'])
    
    # 2. Does higher penalty reduce introduced errors (regressions/oscillation proxy)?
    r_err, p_err = spearmanr(stats_df['w_rp_pen'], stats_df['mean_introduced_errors'])
    
    # Plots
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(stats_df['w_rp_pen'], stats_df['mean_iterations'], marker='o')
    plt.title('H3: Convergence Speed')
    plt.xlabel('Regression Penalty (w_rp_pen)')
    plt.ylabel('Mean Total Iterations')

    plt.subplot(1, 2, 2)
    plt.plot(stats_df['w_rp_pen'], stats_df['mean_introduced_errors'], marker='o', color='orange')
    plt.title('H3: Oscillation/Regression Proxy')
    plt.xlabel('Regression Penalty (w_rp_pen)')
    plt.ylabel('Mean Introduced Errors')

    plt.tight_layout()
    plot_path = output_dir / "h3_csv_analysis.png"
    plt.savefig(plot_path)
    
    # Report
    with open(output_dir / "h3_csv_report.txt", "w") as f:
        f.write("H3 HYPOTHESIS TEST (Using CSV Summaries)\n")
        f.write("="*40 + "\n")
        f.write(f"Analyzed {len(full_df)} proofs across {len(penalties)} penalty settings.\n\n")
        
        f.write("1. Convergence Speed Effect:\n")
        f.write(f"   Spearman r = {r_iter:.4f}, p-value = {p_iter:.4f}\n")
        f.write("   (Negative r means higher penalty means fewer iterations)\n\n")
        
        f.write("2. Oscillation/Regression Effect (Proxy: Introduced Errors):\n")
        f.write(f"   Spearman r = {r_err:.4f}, p-value = {p_err:.4f}\n")
        f.write("   (Negative r means higher penalty means fewer regressions)\n\n")
        
        f.write("Averages by Penalty:\n")
        f.write(stats_df.to_string(index=False))

    print(f"Done. Check {output_dir}/h3_csv_report.txt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="ablation_out/runs")
    parser.add_argument("--out-dir", default="h3_csv_results")
    args = parser.parse_args()
    analyze_h3_from_csv(Path(args.runs_dir), Path(args.out_dir))
