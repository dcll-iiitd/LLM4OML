"""CLI tool to label ground truth correctness for a random sample of the results."""

import pandas as pd
import argparse
from pathlib import Path
import json

def label_ground_truth(results_dir: str):
    base_dir = Path(results_dir)
    results_csv = base_dir / "results.csv"
    batch_log = base_dir / "batch_execution_log.json"
    gt_path = base_dir / "ground_truth.csv"

    if not results_csv.exists():
        print(f"File not found: {results_csv}")
        return

    df = pd.read_csv(results_csv)
    
    # Exclude failed rows
    df = df[df['Processing_Status'] == "COMPLETED"]

    if gt_path.exists():
        mode = input("ground_truth.csv already exists. Append new unlabelled rows? (y/n): ")
        if mode.lower() != 'y':
            return
        gt_df = pd.read_csv(gt_path)
        done_probs = set(gt_df['Problem_Statement'])
        df = df[~df['Problem_Statement'].isin(done_probs)]
    else:
        gt_df = pd.DataFrame(columns=["Problem_Statement", "human_correctness"])

    if len(df) == 0:
        print("No new proofs to label.")
        return

    # Load proof content if possible
    logs = {}
    if batch_log.exists():
        try:
            with open(batch_log, "r") as f:
                logs = json.load(f)
        except Exception:
            pass

    print("\n--- Human Evaluation Ground Truth Labelling ---")
    print("For each proof, input:")
    print("  0: Incorrect")
    print("  1: Probably Correct (minor issues)")
    print("  2: Correct (sound math)")
    print("Press Ctrl+C to stop and save progress.\n")

    new_rows = []
    
    try:
        for idx, row in df.iterrows():
            prob = row["Problem_Statement"]
            algo_id = row.get("Algorithm_ID", f"idx_{idx}")
            
            print(f"\n{'='*80}")
            print(f"Algorithm ID: {algo_id}")
            print(f"Problem: {prob[:300]}...")
            
            # Fetch proof from logs if exists
            proof = "Proof not available in logs."
            feedback = row.get("Final_Feedback_Preview", "")
            
            try:
                # Find log entry
                log_entry = next((item for item in logs.values() if item.get("problem_statement") == prob), None)
                if log_entry and "iterations" in log_entry:
                    last_iter = log_entry["iterations"][-1]
                    proof = last_iter.get("nodes", {}).get("prover", {}).get("proof", proof)
            except Exception:
                pass
                
            print(f"\nVerdict from Judges: {row.get('Overall_Score', 'N/A')}")
            print("\nFinal Feedback Preview:")
            print(feedback)
            print(f"\nProof Preview (first 1000 chars):\n{proof[:1000]}...")
            
            while True:
                val = input("\nEnter label (0/1/2) or 's' to skip: ").strip()
                if val.lower() == 's':
                    break
                if val in ('0', '1', '2'):
                    new_rows.append({"Problem_Statement": prob, "human_correctness": int(val)})
                    break
                print("Invalid input. 0, 1, 2, or s.")

    except KeyboardInterrupt:
        print("\nStopping...")

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        # concat and save
        if len(gt_df) > 0:
            final_gt = pd.concat([gt_df, new_df], ignore_index=True)
        else:
            final_gt = new_df
        final_gt.to_csv(gt_path, index=False)
        print(f"\nSaved {len(final_gt)} labeled rows to {gt_path}")
    else:
        print("\nNo labels added.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="batch_results")
    args = parser.parse_args()
    label_ground_truth(args.results_dir)
