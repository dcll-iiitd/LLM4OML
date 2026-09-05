"""
Updated 3-panel judge reliability figure.
Data: human_evaluations.csv + llm_judge_evaluations.csv (16 tasks:
6 textbook-tier [B0xx] + 10 research-tier [V2_xx]).
Panel (a): per-task scores with tier divider.
Panel (b): per-category (HA/MS/OP) agreement, split by tier.
Panel (c): 4x4 verdict confusion matrix (PASS/PASS_MINOR/CONDITIONAL/FAIL), pooled.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

human  = pd.read_csv(os.path.join(ROOT, 'judge_baseline', 'human_evaluations.csv'))
llm    = pd.read_csv(os.path.join(ROOT, 'judge_baseline', 'llm_judge_evaluations.csv'))
merged = pd.merge(human, llm, on=['Task_ID', 'Problem Statement'],
                  suffixes=('_human', '_llm'))

merged['Tier'] = np.where(merged['Task_ID'].str.startswith('B'), 'textbook', 'research')
# keep the original 5+B038 textbook block first, then research block, matching paper order
merged = pd.concat([
    merged[merged['Tier'] == 'textbook'],
    merged[merged['Tier'] == 'research'],
]).reset_index(drop=True)

n_textbook = int((merged['Tier'] == 'textbook').sum())
n_research = int((merged['Tier'] == 'research').sum())

# ── palette ────────────────────────────────────────────────────────────────
C_H  = '#2166AC'
C_L  = '#C0392B'
C_AG = '#27AE60'
C_DG = '#E67E22'
C_GR = '#E8E8E8'
C_TB = '#F0F4FA'   # textbook-tier background tint
C_RS = '#FBF2EC'   # research-tier background tint

plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         9,
    'axes.titlesize':    9,
    'axes.labelsize':    8.5,
    'xtick.labelsize':   8,
    'ytick.labelsize':   8,
    'legend.fontsize':   8,
    'figure.dpi':        150,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
    'axes.spines.top':   False,
    'axes.spines.right': False,
})

tasks       = [f'T{i+1}' for i in range(len(merged))]
h_scores    = merged['Overall_Score_human'].values.astype(float)
l_scores    = merged['Overall_Score_llm'].values.astype(float)
jrs_vals    = merged['Mean_JRS'].values.astype(float)

# ── layout ─────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(11.5, 3.8))
gs  = gridspec.GridSpec(
    1, 3,
    figure=fig,
    width_ratios=[3.4, 2.6, 2.2],
    wspace=0.50,
    left=0.06, right=0.98,
    top=0.82, bottom=0.20,
)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])

# ══════════════════════════════════════════════════════════════════════════
# Panel (a) - per-task scores with delta labels, tier divider
# ══════════════════════════════════════════════════════════════════════════
x  = np.arange(len(tasks))
w  = 0.32

ax1.axvspan(-0.5, n_textbook - 0.5, color=C_TB, zorder=0)
ax1.axvspan(n_textbook - 0.5, len(tasks) - 0.5, color=C_RS, zorder=0)
ax1.axvline(n_textbook - 0.5, color='#888888', linewidth=1.0, linestyle='--', zorder=1)

ax1.bar(x - w/2, h_scores, w, color=C_H, alpha=0.9,
        zorder=3, linewidth=0.4, edgecolor='white')
ax1.bar(x + w/2, l_scores, w, color=C_L, alpha=0.9,
        zorder=3, linewidth=0.4, edgecolor='white')

for i, (hv, lv) in enumerate(zip(h_scores, l_scores)):
    delta = lv - hv
    ymax  = max(hv, lv)
    col   = C_AG if abs(delta) < 0.05 else C_DG
    ax1.text(i, ymax + 0.32, f'{delta:+.1f}',
             ha='center', va='bottom',
             fontsize=7, fontweight='bold', color=col, rotation=90)

ax1.text(n_textbook/2 - 0.5, 7.0, 'Textbook', ha='center', va='top', fontsize=8, color='#555555', fontweight='bold')
ax1.text(n_textbook + n_research/2 - 0.5, 7.0, 'Research-sourced', ha='center', va='top', fontsize=8, color='#555555', fontweight='bold')

h_patch = mpatches.Patch(color=C_H, alpha=0.9, label='Human')
l_patch = mpatches.Patch(color=C_L, alpha=0.9, label='LLM Judge')
ax1.legend(handles=[h_patch, l_patch],
           loc='upper center',
           bbox_to_anchor=(0.5, -0.24),
           ncol=2, frameon=False,
           handlelength=1.2, columnspacing=1.0)

ax1.set_xticks(x)
ax1.set_xticklabels(tasks, fontsize=6.5, rotation=90)
ax1.set_xlim(-0.5, len(tasks) - 0.5)
ax1.set_ylim(0, 7.6)
ax1.set_yticks([1, 2, 3, 4, 5])
ax1.set_ylabel('Overall Score (1–5)')
ax1.yaxis.grid(True, color=C_GR, linewidth=0.7, zorder=0)
ax1.set_axisbelow(True)
ax1.set_title('(a) Scores & Δ(LLM − Human)', pad=6)

# ══════════════════════════════════════════════════════════════════════════
# Panel (b) - per-category agreement, split by tier (computed from data)
# ══════════════════════════════════════════════════════════════════════════
cat_cols = {'Halluc.': ('Hallucination_Error', 'V2_Hallucination_Error'),
            'Missing\nStep': ('Missing_Step', 'V2_Missing_Step'),
            'Op. Error': ('Operator_Error', 'V2_Operator_Error')}

def norm_yn(s):
    return s.astype(str).str.strip().str.lower().isin(['yes', 'y', '1', 'true'])

tb = merged[merged['Tier'] == 'textbook']
rs = merged[merged['Tier'] == 'research']

agree_tb, agree_rs = [], []
for cat, (hcol, lcol) in cat_cols.items():
    h_tb, l_tb = norm_yn(tb[hcol]), norm_yn(tb[lcol])
    h_rs, l_rs = norm_yn(rs[hcol]), norm_yn(rs[lcol])
    agree_tb.append(100.0 * np.mean(h_tb.values == l_tb.values))
    agree_rs.append(100.0 * np.mean(h_rs.values == l_rs.values))

cats = list(cat_cols.keys())
xc = np.arange(len(cats))
wc = 0.32

ax2.bar(xc - wc/2, agree_tb, wc, color='#5B8FC7', alpha=0.9,
        zorder=3, linewidth=0.4, edgecolor='white', label='Textbook')
ax2.bar(xc + wc/2, agree_rs, wc, color='#D08A4A', alpha=0.9,
        zorder=3, linewidth=0.4, edgecolor='white', label='Research')

for xi, (a, b) in enumerate(zip(agree_tb, agree_rs)):
    ax2.text(xi - wc/2, a + 2, f'{a:.0f}%', ha='center', va='bottom', fontsize=7.5, color='#2C5F91', fontweight='bold')
    ax2.text(xi + wc/2, b + 2, f'{b:.0f}%', ha='center', va='bottom', fontsize=7.5, color='#A85F1F', fontweight='bold')

ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.24), ncol=2, frameon=False,
           handlelength=1.2, columnspacing=1.0)

mean_jrs_tb = tb['Mean_JRS'].mean()
mean_jrs_rs = rs['Mean_JRS'].mean()
ax2.text(0.97, 0.06, f'Mean JRS: TB {mean_jrs_tb:+.2f} / Res {mean_jrs_rs:+.2f}',
         transform=ax2.transAxes, ha='right', va='bottom',
         fontsize=7, color='#555555',
         bbox=dict(boxstyle='round,pad=0.25', fc='#F8F8F8', ec='#BBBBBB', lw=0.8))

ax2.set_xticks(xc)
ax2.set_xticklabels(cats)
ax2.set_ylim(0, 115)
ax2.set_yticks([0, 20, 40, 60, 80, 100])
ax2.set_ylabel('Agreement (%)')
ax2.yaxis.grid(True, color=C_GR, linewidth=0.7, zorder=0)
ax2.set_axisbelow(True)
ax2.set_title('(b) Error-Category Agreement\nby Task Tier', pad=6)

# ══════════════════════════════════════════════════════════════════════════
# Panel (c) - 4x4 verdict confusion matrix: PASS / PASS_MINOR / CONDITIONAL / FAIL
# (pooled across all 16 tasks; per-tier breakdown is in Table I)
# ══════════════════════════════════════════════════════════════════════════
labels = ['PASS', 'PM', 'COND.', 'FAIL']
label_keys = {'PASS': 0, 'PASS_MINOR': 1, 'CONDITIONAL': 2, 'FAIL': 3}

conf = np.zeros((4, 4), dtype=int)
for _, row in merged.iterrows():
    ri = label_keys.get(row['Verdict'], 2)
    ci = label_keys.get(row['V2_Verdict'], 0)
    conf[ri, ci] += 1

im = ax3.imshow(conf, cmap='Blues', vmin=0, vmax=5, aspect='equal')
ax3.set_adjustable('box')

ax3.set_xticks([0, 1, 2, 3])
ax3.set_xticklabels(labels, fontsize=6.5)
ax3.set_yticks([0, 1, 2, 3])
ax3.set_yticklabels(labels, fontsize=6.5)
ax3.set_xlabel('LLM Judge', fontsize=8.5)
ax3.set_ylabel('Human', fontsize=8.5)
ax3.set_title('(c) Verdict Agreement\n(all 16 tasks)', pad=6)

for i in range(4):
    for j in range(4):
        col = 'white' if conf[i, j] >= 3 else 'black'
        if conf[i, j] > 0:
            ax3.text(j, i, str(conf[i, j]),
                     ha='center', va='center',
                     fontsize=12, fontweight='bold', color=col)

for k in range(4):
    ax3.add_patch(plt.Rectangle(
        (k - 0.5, k - 0.5), 1, 1,
        fill=False, edgecolor=C_AG, lw=2.0, zorder=5))

for sp in ax3.spines.values():
    sp.set_visible(True)
    sp.set_linewidth(0.6)

# ── save ───────────────────────────────────────────────────────────────────
os.makedirs(os.path.join(ROOT, 'figures'), exist_ok=True)
out_base = os.path.join(ROOT, 'figures', 'judge_reliability_panel')
fig.savefig(out_base + '.pdf')
fig.savefig(out_base + '.png')
print(f"Saved {out_base}.pdf / .png")
print(f"\nTextbook (n={n_textbook}) per-category agreement: "
      + ", ".join(f"{c}={a:.0f}%" for c, a in zip(cats, agree_tb)))
print(f"Research (n={n_research}) per-category agreement: "
      + ", ".join(f"{c}={a:.0f}%" for c, a in zip(cats, agree_rs)))
print("\n4x4 confusion matrix (rows=Human, cols=LLM), order PASS/PASS_MINOR/CONDITIONAL/FAIL:")
print(conf)
diag = np.trace(conf)
print(f"Overall verdict agreement: {diag}/{len(merged)} = {100*diag/len(merged):.1f}%")
