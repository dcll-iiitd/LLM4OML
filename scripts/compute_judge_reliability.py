"""
Reproduce the judge-reliability statistics reported in Table I and Section VI
of the paper, from the paired human / LLM-judge evaluations.

Inputs (relative to repo root):
    judge_baseline/human_evaluations.csv      one row per task, human expert
    judge_baseline/llm_judge_evaluations.csv  one row per task, 3-judge ensemble

Tiering follows the paper's task taxonomy: B0xx = textbook-tier,
V2_xx = research-sourced tier.

Usage:  python scripts/compute_judge_reliability.py
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HUMAN = os.path.join(HERE, "judge_baseline", "human_evaluations.csv")
LLM = os.path.join(HERE, "judge_baseline", "llm_judge_evaluations.csv")

CATS = [("HA", "Hallucination_Error", "V2_Hallucination_Error"),
        ("MS", "Missing_Step", "V2_Missing_Step"),
        ("OP", "Operator_Error", "V2_Operator_Error")]


def yn(series):
    """Normalise yes/no flags to boolean."""
    return series.astype(str).str.strip().str.lower().isin({"yes", "y", "1", "true"})


def load():
    human = pd.read_csv(HUMAN)
    llm = pd.read_csv(LLM)
    merged = pd.merge(human, llm, on=["Task_ID", "Problem Statement"],
                      suffixes=("_human", "_llm"))
    if len(merged) != len(human):
        raise SystemExit(f"merge lost rows: {len(human)} human -> {len(merged)} merged")
    merged["Tier"] = np.where(merged["Task_ID"].str.startswith("B"),
                              "textbook", "research")
    return merged


def report(df, label):
    n = len(df)
    h = df["Overall_Score_human"].astype(float).values
    l = df["Overall_Score_llm"].astype(float).values
    agree = int((df["Verdict"] == df["V2_Verdict"]).sum())

    print(f"--- {label} (n={n}) ---")
    print(f"  Mean score : human {h.mean():.2f}   LLM {l.mean():.2f}   MAE {np.abs(l-h).mean():.2f}")
    if n > 2:
        r = stats.pearsonr(h, l).statistic
        rho = stats.spearmanr(h, l).statistic
        print(f"  Correlation: Pearson r = {r:+.2f}   Spearman rho = {rho:+.2f}")
    print(f"  Verdict agreement: {agree}/{n} = {100*agree/n:.0f}%")
    print(f"  Mean JRS   : {df['Mean_JRS'].astype(float).mean():+.3f}")

    for name, hcol, lcol in CATS:
        hv, lv = yn(df[hcol]).values, yn(df[lcol]).values
        tp = int((hv & lv).sum())
        fn = int((hv & ~lv).sum())
        fp = int((~hv & lv).sum())
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        print(f"  {name}: agreement {100*np.mean(hv == lv):3.0f}%  "
              f"recall {recall:.2f}  precision {prec:.2f}  (TP={tp} FN={fn} FP={fp})")
    print()


def main():
    merged = load()
    report(merged[merged["Tier"] == "textbook"], "Textbook-tier")
    report(merged[merged["Tier"] == "research"], "Research-sourced tier")
    report(merged, "Pooled")

    print("Per-task score deltas (LLM - human), ranked by magnitude:")
    d = merged.assign(delta=merged["Overall_Score_llm"].astype(float)
                      - merged["Overall_Score_human"].astype(float))
    for _, row in d.reindex(d["delta"].abs().sort_values(ascending=False).index).iterrows():
        flag = "  <- negative JRS" if float(row["Mean_JRS"]) < 0 else ""
        print(f"  {row['Task_ID']:<7} delta {row['delta']:+.1f}   "
              f"JRS {float(row['Mean_JRS']):+.3f}{flag}")


if __name__ == "__main__":
    main()
