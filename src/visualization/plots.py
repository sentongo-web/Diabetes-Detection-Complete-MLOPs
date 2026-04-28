"""
src/visualization/plots.py
__________________________
Publication-quality plots for EDA, model evaluation, and SHAP explanations.
Every function saves a PNG to the specified output_dir so the results are
reproducible and can be embedded directly in the README / reports.
"""

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for server / CI environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

# Consistent colour palette
PALETTE   = {"0": "#4878CF", "1": "#D65F5F"}
FIG_DIR   = Path("models/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight"})


def _save(fig: plt.Figure, name: str, output_dir: Path = FIG_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# _____________________________________________________________________________
# 1. Target distribution
# _____________________________________________________________________________

def plot_target_distribution(df: pd.DataFrame, target: str = "hypertension",
                             output_dir: Path = FIG_DIR) -> Path:
    """Bar chart showing class balance."""
    counts = df[target].value_counts().sort_index()
    labels = ["Normotensive (0)", "Hypertensive (1)"]
    colors = [PALETTE["0"], PALETTE["1"]]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, counts.values, color=colors, edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 15,
                f"{val:,}\n({val / counts.sum():.1%})", ha="center", fontsize=10)
    ax.set_title("Target Class Distribution _ Nakaseke Cohort", fontweight="bold")
    ax.set_ylabel("Patient Count")
    ax.set_ylim(0, counts.max() * 1.2)
    return _save(fig, "target_distribution", output_dir)


# _____________________________________________________________________________
# 2. Age distribution by hypertension status
# _____________________________________________________________________________

def plot_age_distribution(df: pd.DataFrame, target: str = "hypertension",
                          output_dir: Path = FIG_DIR) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, color in PALETTE.items():
        subset = df[df[target] == int(label)]["age"]
        subset.plot.kde(ax=ax, label=f"{'Hypertensive' if label=='1' else 'Normotensive'} (n={len(subset):,})",
                        color=color, linewidth=2)
    ax.set_xlabel("Age (years)")
    ax.set_title("Age Distribution by Hypertension Status", fontweight="bold")
    ax.legend()
    return _save(fig, "age_distribution", output_dir)


# _____________________________________________________________________________
# 3. BMI distribution
# _____________________________________________________________________________

def plot_bmi_distribution(df: pd.DataFrame, target: str = "hypertension",
                          output_dir: Path = FIG_DIR) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, color in PALETTE.items():
        subset = df[df[target] == int(label)]["bmi"]
        subset.plot.kde(ax=ax, label=f"{'Hypertensive' if label=='1' else 'Normotensive'}",
                        color=color, linewidth=2)
    ax.axvline(25, color="gray", linestyle="--", linewidth=1, label="BMI 25")
    ax.axvline(30, color="gray", linestyle=":",  linewidth=1, label="BMI 30")
    ax.set_xlabel("BMI (kg/m_)")
    ax.set_title("BMI Distribution by Hypertension Status", fontweight="bold")
    ax.legend()
    return _save(fig, "bmi_distribution", output_dir)


# _____________________________________________________________________________
# 4. Correlation heatmap
# _____________________________________________________________________________

def plot_correlation_heatmap(df: pd.DataFrame, target: str = "hypertension",
                             output_dir: Path = FIG_DIR) -> Path:
    numeric = df.select_dtypes(include="number")
    corr    = numeric.corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=False, cmap="RdBu_r", center=0,
                linewidths=0.4, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Matrix", fontweight="bold", pad=12)
    return _save(fig, "correlation_heatmap", output_dir)


# _____________________________________________________________________________
# 5. Feature importance (SHAP summary bar)
# _____________________________________________________________________________

def plot_shap_summary(shap_values: np.ndarray, feature_names: list[str],
                      top_n: int = 15, output_dir: Path = FIG_DIR) -> Path:
    mean_abs  = np.abs(shap_values).mean(axis=0)
    sorted_idx = np.argsort(mean_abs)[::-1][:top_n]
    names  = [feature_names[i] for i in sorted_idx]
    values = mean_abs[sorted_idx]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(names[::-1], values[::-1], color="#5C85D6", edgecolor="white")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Top {top_n} Predictors of Hypertension (SHAP)", fontweight="bold")
    ax.grid(axis="x", alpha=0.4)
    return _save(fig, "shap_feature_importance", output_dir)


# _____________________________________________________________________________
# 6. ROC curves for all models
# _____________________________________________________________________________

def plot_roc_curves(results: dict, output_dir: Path = FIG_DIR) -> Path:
    """
    results : {model_name: {"fpr": array, "tpr": array, "auc": float}}
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, r in results.items():
        ax.plot(r["fpr"], r["tpr"], label=f"{name}  (AUC={r['auc']:.3f})", linewidth=1.8)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC=0.500)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves _ Hypertension Detection", fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    return _save(fig, "roc_curves", output_dir)


# _____________________________________________________________________________
# 7. Precision-Recall curves
# _____________________________________________________________________________

def plot_pr_curves(results: dict, output_dir: Path = FIG_DIR) -> Path:
    """
    results : {model_name: {"precision": array, "recall": array, "avg_precision": float}}
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, r in results.items():
        ax.plot(r["recall"], r["precision"],
                label=f"{name}  (AP={r['avg_precision']:.3f})", linewidth=1.8)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves _ Hypertension Detection", fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    return _save(fig, "pr_curves", output_dir)


# _____________________________________________________________________________
# 8. Missing-data profile
# _____________________________________________________________________________

def plot_missing_profile(df: pd.DataFrame, output_dir: Path = FIG_DIR) -> Path:
    miss = (df.isna().mean() * 100).sort_values(ascending=False)
    miss = miss[miss > 0]
    if miss.empty:
        return None

    fig, ax = plt.subplots(figsize=(10, max(4, len(miss) * 0.35)))
    miss.plot.barh(ax=ax, color="#E88A58", edgecolor="white")
    ax.set_xlabel("Missing (%)")
    ax.set_title("Missing-Data Profile (raw dataset)", fontweight="bold")
    return _save(fig, "missing_profile", output_dir)


# _____________________________________________________________________________
# 9. Confusion matrix
# _____________________________________________________________________________

def plot_confusion_matrix(cm: np.ndarray, model_name: str = "",
                          output_dir: Path = FIG_DIR) -> Path:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Pred 0", "Pred 1"],
                yticklabels=["True 0", "True 1"],
                linewidths=0.5)
    ax.set_title(f"Confusion Matrix _ {model_name}", fontweight="bold")
    return _save(fig, f"confusion_{model_name.replace(' ', '_').lower()}", output_dir)


# _____________________________________________________________________________
# 10. Run all EDA plots in one call
# _____________________________________________________________________________

def run_eda_plots(df: pd.DataFrame, target: str = "hypertension",
                  output_dir: Path = FIG_DIR) -> None:
    """Convenience wrapper that generates the full EDA suite at once."""
    output_dir = Path(output_dir)
    plot_target_distribution(df, target, output_dir)
    plot_age_distribution(df, target, output_dir)
    plot_bmi_distribution(df, target, output_dir)
    plot_correlation_heatmap(df, target, output_dir)
    plot_missing_profile(df, output_dir)
    print(f"EDA plots saved to {output_dir}")
