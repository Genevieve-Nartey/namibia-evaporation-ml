"""
eda.py
------
Exploratory Data Analysis: correlation, distributions, scatter plots.
All figures are saved to the reports/figures directory.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

FIGURES_DIR = Path("reports/figures")

VARIABLE_LABELS = {
    "d2m": "Dewpoint Temp (K)",
    "t2m": "Air Temp (K)",
    "mer": "Evaporation Rate (kg/m²/s)",
    "mtdwswrf": "Shortwave Radiation (W/m²)",
    "mtpr": "Precipitation Rate (kg/m²/s)",
    "stl1": "Soil Temp (K)",
    "swvl1": "Soil Moisture (m³/m³)",
}


def run_eda(df: pd.DataFrame, save: bool = True) -> dict:
    """
    Run full EDA suite: summary stats, correlation heatmap,
    distributions, and scatter plots vs. target.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset.
    save : bool
        Whether to save figures to disk.

    Returns
    -------
    dict
        Summary statistics and correlation matrix.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    summary = _summary_statistics(df)
    corr = _correlation_heatmap(df, save=save)
    _distribution_plots(df, save=save)
    _scatter_plots_vs_target(df, target="mer", save=save)

    logger.info("EDA complete. Figures saved to %s", FIGURES_DIR)
    return {"summary": summary, "correlations": corr}


def _summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Print and return summary statistics."""
    numeric = df.select_dtypes(include="number")
    stats = numeric.describe().T
    stats["skewness"] = numeric.skew()
    logger.info("Summary statistics:\n%s", stats.to_string())
    return stats


def _correlation_heatmap(df: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """Plot and optionally save a Pearson correlation heatmap."""
    numeric = df.select_dtypes(include="number")
    corr = numeric.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr,
        vmin=-1,
        vmax=1,
        square=True,
        cmap="coolwarm",
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Namibian Soil Data — Pearson Correlation", fontsize=14, pad=12)
    plt.tight_layout()

    if save:
        _save_fig(fig, "correlation_heatmap.png")
    plt.close(fig)
    return corr


def _distribution_plots(df: pd.DataFrame, save: bool = True) -> None:
    """Plot histograms with KDE for each numeric variable."""
    numeric_cols = [c for c in VARIABLE_LABELS if c in df.columns]
    n = len(numeric_cols)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        axes[i].hist(df[col], bins=20, edgecolor="white", alpha=0.8, color="#4C72B0")
        axes[i].set_title(VARIABLE_LABELS.get(col, col), fontsize=11)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Count")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Variable Distributions — Northern Namibia (1959–2022)", fontsize=13, y=1.02)
    plt.tight_layout()

    if save:
        _save_fig(fig, "distributions.png")
    plt.close(fig)


def _scatter_plots_vs_target(
    df: pd.DataFrame, target: str = "mer", save: bool = True
) -> None:
    """Plot scatter plots of each feature against the target variable."""
    feature_cols = [c for c in VARIABLE_LABELS if c != target and c in df.columns]
    n = len(feature_cols)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes = axes.flatten()

    corr_with_target = df[feature_cols + [target]].corr()[target]

    for i, col in enumerate(feature_cols):
        r = corr_with_target[col]
        axes[i].scatter(df[col], df[target], alpha=0.4, s=10, color="#4C72B0")
        axes[i].set_xlabel(VARIABLE_LABELS.get(col, col))
        axes[i].set_ylabel(VARIABLE_LABELS.get(target, target))
        axes[i].set_title(f"r = {r:.2f}", fontsize=10)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature vs. Evaporation Rate", fontsize=13, y=1.02)
    plt.tight_layout()

    if save:
        _save_fig(fig, "scatter_vs_target.png")
    plt.close(fig)


def _save_fig(fig: plt.Figure, filename: str) -> None:
    """Save a matplotlib figure to the figures directory."""
    out_path = FIGURES_DIR / filename
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    logger.info("Saved figure: %s", out_path)
