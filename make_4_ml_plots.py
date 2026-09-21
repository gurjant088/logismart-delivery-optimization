# -*- coding: utf-8 -*-
"""
LogiSmart Delivery Optimization - 4 ML Diagnostic Plots
============================================================
Generates just: ROC Curve, KS Plot, SHAP Plot, QQ Plot.

Trains one real classifier (predicts on-time vs late delivery from
legitimate pre-dispatch order features) to power the ROC, KS and SHAP
plots. The QQ plot checks whether baseline delivery delay is normally
distributed.

Inputs required (same folder, or set via env var CHARTS_INPUT_DIR):
    orders_with_optimized_delay.csv

Usage:
    python make_4_ml_plots.py [input_dir] [output_dir]
    (or paste into a Colab/Jupyter cell and run as-is)
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split

sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.bbox"] = "tight"

OPT = "#2E7D32"
ACCENT = "#1F4E79"
RED = "#C0392B"


# =============================================================================
# INPUT RESOLUTION (works from CLI, or pasted into a notebook cell)
# =============================================================================
def _clean_argv(argv):
    cleaned, skip_next = [], False
    for tok in argv:
        if skip_next:
            skip_next = False
            continue
        if tok.startswith("-"):
            skip_next = True
            continue
        cleaned.append(tok)
    return cleaned


in_notebook = "ipykernel" in sys.modules or "IPython" in sys.modules
positional = [] if in_notebook else _clean_argv(sys.argv[1:])

CANDIDATE_INPUT_DIRS = [
    os.environ.get("CHARTS_INPUT_DIR"),
    positional[0] if len(positional) > 0 else None,
    ".",
    "/content/sample_data",
    "/content",
]
INPUT_DIR = None
for cand in CANDIDATE_INPUT_DIRS:
    if cand and os.path.isfile(os.path.join(cand, "orders_with_optimized_delay.csv")):
        INPUT_DIR = cand
        break

if INPUT_DIR is None:
    raise FileNotFoundError(
        "\n\nCould not find 'orders_with_optimized_delay.csv'.\n"
        "This file is produced by running the main optimizer script first "
        "(run_project.py / complete_project.py) - run that first, or set its "
        "location explicitly before running this script:\n"
        "    import os; os.environ['CHARTS_INPUT_DIR'] = '/content/sample_data'\n"
        f"Checked: {[c for c in CANDIDATE_INPUT_DIRS if c]}"
    )

OUTPUT_DIR = os.environ.get("CHARTS_OUTPUT_DIR", positional[1] if len(positional) > 1 else "./ml_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"[OK] Loading orders from: {INPUT_DIR}")
orders = pd.read_csv(os.path.join(INPUT_DIR, "orders_with_optimized_delay.csv"))
print(f"[OK] Loaded {len(orders)} orders.\n")


def savefig(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  [saved] {path}")


# =============================================================================
# Train classifier: on-time vs late delivery, from legitimate pre-dispatch
# features (not outcome leakage) - powers the ROC, KS and SHAP plots.
# =============================================================================
print("Training classifier: predicts on-time vs late delivery from order features...")
feat_df = orders.copy()
feat_df["target"] = (feat_df["Delivery_Status"] == "On Time").astype(int)

priority_map = {"High": 0, "Medium": 1, "Low": 2}
feat_df["Priority_Rank"] = feat_df["Priority"].map(priority_map)

X = pd.get_dummies(
    feat_df[["Distance_Km", "Package_Weight_Kg", "Service_Time_Min", "Priority_Rank",
             "Required_Vehicle_Type", "Origin_Hub"]],
    columns=["Required_Vehicle_Type", "Origin_Hub"], drop_first=True
)
y = feat_df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
clf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
clf.fit(X_train, y_train)
y_proba = clf.predict_proba(X_test)[:, 1]
print(f"  Test accuracy: {clf.score(X_test, y_test):.3f}\n")


# =============================================================================
# 1. ROC CURVE
# =============================================================================
def plot_roc():
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(7, 6.5))
    ax.plot(fpr, tpr, color=ACCENT, linewidth=2, label=f"On-Time Classifier (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color=RED, linestyle="--", linewidth=1.5, label="Random Classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve \u2013 On-Time Delivery Classifier", fontweight="bold")
    ax.legend(loc="lower right")
    fig.tight_layout()
    savefig(fig, "01_roc_curve.png")


# =============================================================================
# 2. KS PLOT
# =============================================================================
def plot_ks():
    pos = np.sort(y_proba[y_test.values == 1])
    neg = np.sort(y_proba[y_test.values == 0])
    grid = np.linspace(0, 1, 200)
    cdf_pos = np.searchsorted(pos, grid, side="right") / len(pos)
    cdf_neg = np.searchsorted(neg, grid, side="right") / len(neg)
    ks_idx = np.argmax(np.abs(cdf_pos - cdf_neg))
    ks_stat, ks_at = np.abs(cdf_pos - cdf_neg)[ks_idx], grid[ks_idx]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(grid, cdf_pos, color=OPT, label="On-Time (positive class)", linewidth=2)
    ax.plot(grid, cdf_neg, color=RED, label="Late (negative class)", linewidth=2)
    ax.annotate("", xy=(ks_at, cdf_pos[ks_idx]), xytext=(ks_at, cdf_neg[ks_idx]),
                arrowprops=dict(arrowstyle="<->", color="black"))
    ax.text(ks_at + 0.02, 0.5, f"KS = {ks_stat:.3f}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted Probability of On-Time Delivery")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("KS Plot \u2013 On-Time Classifier Separation", fontweight="bold")
    ax.legend(loc="lower right")
    fig.tight_layout()
    savefig(fig, "02_ks_plot.png")


# =============================================================================
# 3. SHAP PLOT
# =============================================================================
def plot_shap():
    try:
        import shap
    except ImportError:
        print("  [skip] shap not installed. Run: !pip install shap   then re-run this script.")
        return
    explainer = shap.TreeExplainer(clf)
    sample = X_test.sample(min(300, len(X_test)), random_state=42)
    shap_values = explainer.shap_values(sample)
    # shap_values may be a list [class0, class1] (older API) or a 3D array (newer API)
    if isinstance(shap_values, list):
        sv = shap_values[1]
    elif shap_values.ndim == 3:
        sv = shap_values[:, :, 1]
    else:
        sv = shap_values
    plt.figure(figsize=(9, 6))
    shap.summary_plot(sv, sample, show=False)
    fig = plt.gcf()
    fig.suptitle("SHAP Plot \u2013 Feature Impact on On-Time Prediction", fontweight="bold", y=1.02)
    fig.tight_layout()
    savefig(fig, "03_shap_plot.png")


# =============================================================================
# 4. QQ PLOT
# =============================================================================
def plot_qq():
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    stats.probplot(orders["Delay_Min"], dist="norm", plot=ax)
    ax.get_lines()[0].set_markerfacecolor(ACCENT)
    ax.get_lines()[0].set_markeredgecolor(ACCENT)
    ax.get_lines()[0].set_alpha(0.5)
    ax.get_lines()[1].set_color(RED)
    ax.set_title("QQ Plot \u2013 Baseline Delay vs Normal Distribution", fontweight="bold")
    fig.tight_layout()
    savefig(fig, "04_qq_plot.png")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("Generating 4 plots:\n")
    plot_roc()
    plot_ks()
    plot_shap()
    plot_qq()
    print(f"\n[OK] All plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
