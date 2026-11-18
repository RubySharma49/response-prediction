import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import warnings
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from typing import Literal
from typing import Literal
from scipy import stats
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde
from sklearn.metrics import (
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
)
from sklearn.metrics import  roc_auc_score, confusion_matrix
import re

import matplotlib.lines as mlines   # was missing from original imports


 


def corr_heatmap(corr_mat):
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr_mat, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Correlation Heatmap")
    plt.show()

def missin_val_heatmap(df):
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
    plt.title("Missing Values Heatmap")
    plt.show()


def stacked_bar_chart(
    labels: list[str],
    categories: list[str],
    values: list[list[float]],
    title: str = "Stacked Bar Chart",
    xlabel: str = "",
    ylabel: str = "",
    figsize: tuple = (10, 6),
    colors: list[str] = None,
):
    """
    Create a stacked bar chart.
 
    Args:
        labels:     Bar labels along the x-axis (e.g. ["Training","Validation"]).
        categories: Stack segment names (e.g. ["Responders", "Non-responders]).
        values:     2-D list of shape [n_categories x n_labels] — one row per category.
        title:      Chart title.
        xlabel:     X-axis label.
        ylabel:     Y-axis label.
        figsize:    Figure size (width, height) in inches.
        colors:     Optional list of colors, one per category.
    """
    if len(categories) != len(values):
        raise ValueError("len(categories) must equal len(values)")
    if any(len(v) != len(labels) for v in values):
        raise ValueError("Each values row must have the same length as labels")
 
    data = np.array(values, dtype=float)          # shape: (n_categories, n_labels)
    totals = data.sum(axis=0)                     # shape: (n_labels,)
    x = np.arange(len(labels))
    bar_width = 0.5
 
    if colors is None:
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i) for i in range(len(categories))]
 
    fig, ax = plt.subplots(figsize=figsize)
 
    bottoms = np.zeros(len(labels))
    bars_per_category = []
 
    for i, (cat, row, color) in enumerate(zip(categories, data, colors)):
        bars = ax.bar(x, row, bar_width, bottom=bottoms, label=cat, color=color)
        bars_per_category.append((bars, row, bottoms.copy()))
        bottoms += row
 
    # Annotate each segment with its value
    for bars, row, bot in bars_per_category:
        for bar, val, b in zip(bars, row, bot):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    b + val / 2,
                    f"{val:g}",
                    ha="center", va="center",
                    fontsize=9, fontweight="bold", color="white",
                )
 
    # Annotate totals above each bar
    for xi, total in zip(x, totals):
        ax.text(
            xi, total + totals.max() * 0.01,
            f"{total:g}",
            ha="center", va="bottom",
            fontsize=10, fontweight="bold",
        )
 
    ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc="upper right")
    ax.set_ylim(0, totals.max() * 1.12)
    ax.spines[["top", "right"]].set_visible(False)
 
    plt.tight_layout()
    return fig, ax
 
def plot_hyperparam_search(
    df: pd.DataFrame,
    metrics: list[str] = ["auroc", "auprc"],
    col_C: str = "param_model__C",
    col_penalty: str = "param_model__penalty",
    col_solver: str = "param_model__solver",
    col_l1_ratio: str = "param_model__l1_ratio",
    penalties: list[str] = ["l1", "l2", "elasticnet"],
    figsize: tuple = (16, 8),
    alpha_band: float = 0.15,
):
    """
    Plot AUROC and AUPRC across a logistic regression hyperparameter grid.
 
    Layout: rows = metrics, cols = penalty types (l1 / l2 / elasticnet).
    - L1 / L2:      one line per solver
    - ElasticNet:   one line per (solver, l1_ratio)
 
    Args:
        df:          GridSearchCV cv_results_ as a DataFrame.
        metrics:     Metric suffixes matching mean_test_<metric> / std_test_<metric>.
        col_C:       Column name for C values.
        col_penalty: Column name for penalty type.
        col_solver:  Column name for solver.
        col_l1_ratio:Column name for l1_ratio (NaN for L1/L2).
        penalties:   Penalty types to plot (subset / order).
        figsize:     Figure size.
        alpha_band:  Transparency of ±std shading.
    """
    n_rows = len(metrics)
    n_cols = len(penalties)
    penalty_labels = {"l1": "L1", "l2": "L2", "elasticnet": "ElasticNet"}
 
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=figsize,
        sharey="row",       # share y within each metric row
        sharex="col",       # share x within each penalty column
    )
    # Normalise to 2-D array for uniform indexing
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]
 
    # Colour cycle — one colour per unique group key
    cmap = plt.get_cmap("tab10")
 
    for col_idx, penalty in enumerate(penalties):
        sub = df[df[col_penalty].str.lower() == penalty].copy()
        sub[col_C] = sub[col_C].astype(float)
        sub = sub.sort_values(col_C)
 
        # Build group keys
        if penalty == "elasticnet":
            group_keys = (
                sub[[col_solver, col_l1_ratio]]
                .dropna(subset=[col_l1_ratio])
                .drop_duplicates()
                .sort_values([col_solver, col_l1_ratio])
                .apply(lambda r: (r[col_solver], round(float(r[col_l1_ratio]), 4)), axis=1)
                .tolist()
            )
            def get_group(key):
                s, r = key
                return sub[(sub[col_solver] == s) &
                           (sub[col_l1_ratio].astype(float).round(4) == r)]
            def key_label(key):
                return f"{key[0]} | l1_ratio={key[1]}"
        else:
            group_keys = sorted(sub[col_solver].dropna().unique())
            def get_group(key):
                return sub[sub[col_solver] == key]
            def key_label(key):
                return str(key)
 
        colour_map = {k: cmap(i % 10) for i, k in enumerate(group_keys)}
 
        for row_idx, metric in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            mean_col = f"mean_test_{metric}"
            std_col  = f"std_test_{metric}"
 
            if mean_col not in df.columns:
                ax.text(0.5, 0.5, f"'{mean_col}' not found",
                        ha="center", va="center", transform=ax.transAxes, color="red")
                continue
 
            for key in group_keys:
                grp = get_group(key).sort_values(col_C)
                c = colour_map[key]
                ax.plot(grp[col_C], grp[mean_col],
                        marker="o", markersize=4, linewidth=1.6,
                        color=c, label=key_label(key))
 
                if std_col in df.columns:
                    ax.fill_between(
                        grp[col_C],
                        grp[mean_col] - grp[std_col],
                        grp[mean_col] + grp[std_col],
                        alpha=alpha_band, color=c,
                    )
 
            # Mark the best point
            best_row = sub.loc[sub[mean_col].idxmax()]
            ax.scatter(
                float(best_row[col_C]), best_row[mean_col],
                marker="*", s=160, color="gold", edgecolors="black",
                linewidths=0.6, zorder=5,
                label=f"best C={float(best_row[col_C]):.3g}",
            )
 
            ax.set_xscale("log")
            ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.5)
            ax.spines[["top", "right"]].set_visible(False)
 
            # Titles / labels
            if row_idx == 0:
                ax.set_title(penalty_labels.get(penalty, penalty),
                             fontsize=12, fontweight="bold", pad=8)
            if col_idx == 0:
                ax.set_ylabel(metric.upper(), fontsize=11)
            if row_idx == n_rows - 1:
                ax.set_xlabel("C (log scale)", fontsize=10)
 
            ax.legend(fontsize=7, loc="lower right", framealpha=0.7)
 
    fig.suptitle("Logistic Regression Hyperparameter Search", fontsize=14,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    return fig, axes
 

def plot_histograms(
    df: pd.DataFrame,
    categories: list[str],
    color: str | list[str] = "steelblue",
    bins: int = 20,
    figsize: tuple = None,
    title: str = "Feature Distributions",
    alpha: float = 0.85,
    edgecolor: str = "white"
) -> plt.Figure:
    """
    Plot histograms for specified numeric columns in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the columns to plot.
    categories : list[str]
        Column names to include in the histogram plot.
        Must be numeric columns present in df.
    color : str or list[str], optional
        Single color applied to all histograms, or a list of colors
        (one per category). Accepts any matplotlib color format:
        named colors ("steelblue"), hex codes ("#2E86AB"), or
        RGB tuples ((0.18, 0.52, 0.67)). Default is "steelblue".
    bins : int, optional
        Number of bins for each histogram. Default is 20.
    figsize : tuple, optional
        Figure size as (width, height) in inches.
        Auto-calculated from number of categories if not provided.
    title : str, optional
        Overall figure title. Default is "Feature Distributions".
    alpha : float, optional
        Transparency of histogram bars (0=transparent, 1=opaque).
        Default is 0.85.
    edgecolor : str, optional
        Color of bar edges. Default is "white".

    Returns
    -------
    plt.Figure
        The matplotlib Figure object (can be saved or further modified).

    Raises
    ------
    ValueError
        If any category is not found in df, or is non-numeric.


  
    """

    # ── Validation ──────────────────────────────────────────────────────────
    missing = [c for c in categories if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in DataFrame: {missing}")

    non_numeric = [c for c in categories if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(
            f"Non-numeric columns cannot be plotted as histograms: {non_numeric}"
        )

    # ── Colour resolution ────────────────────────────────────────────────────
    n = len(categories)

    if isinstance(color, str):
        # Single colour → apply to all
        colors = [color] * n
    elif isinstance(color, (list, tuple)):
        if len(color) == n:
            colors = list(color)
        elif len(color) == 1:
            colors = color * n
        else:
            raise ValueError(
                f"Length of color list ({len(color)}) must match "
                f"number of categories ({n}), or be a single color string."
            )
    else:
        raise ValueError("color must be a string or list of strings.")

    # Validate each colour entry
    for i, c in enumerate(colors):
        if not mcolors.is_color_like(c):
            raise ValueError(f"Invalid color at index {i}: '{c}'")

    # ── Layout ───────────────────────────────────────────────────────────────
    ncols = min(3, n)                        # max 3 columns
    nrows = (n + ncols - 1) // ncols        # ceiling division

    if figsize is None:
        figsize = (ncols * 4.5, nrows * 3.5)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    # Flatten axes for easy iteration (handles 1-row or 1-column grids)
    axes_flat = (
        [axes] if n == 1
        else axes.flatten() if hasattr(axes, "flatten")
        else list(axes)
    )

    # ── Plotting ─────────────────────────────────────────────────────────────
    for i, (col, ax, clr) in enumerate(zip(categories, axes_flat, colors)):
        data = df[col].dropna()

        ax.hist(data, bins=bins, color=clr, alpha=alpha, edgecolor=edgecolor)

        # Annotation: mean and median lines
        mean_val = data.mean()
        median_val = data.median()
        ax.axvline(mean_val,   color="red",    linestyle="--", linewidth=1.2,
                   label=f"Mean {mean_val:.2f}")
        ax.axvline(median_val, color="orange", linestyle=":",  linewidth=1.2,
                   label=f"Median {median_val:.2f}")

        ax.set_title(col.replace("_", " ").title(), fontsize=11)
        ax.set_xlabel("Value", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.legend(fontsize=8, framealpha=0.6)
        ax.spines[["top", "right"]].set_visible(False)

    # Hide any unused subplots
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    return fig



def plot_response_barplots(
    df: pd.DataFrame,
    categories: list[str],
    response_col: str,
    responders: list[str] = ["CR", "PR"],
    non_responders: list[str] = ["SD", "PD"],
    colors: list[str] = None,
    label_style: Literal["count", "pct", "both"] = "count",
    legend_position: Literal["responder", "below", "outside"] = "responder",
    figsize: tuple = None,
    title: str = "ICI Response by Clinical Category",
    min_bar_height_for_label: float = 0.04,
) -> plt.Figure:
    """
    Stacked bar plots comparing Responder vs Non-Responder groups
    across categorical clinical variables.

    Parameters
    ----------
    df : pd.DataFrame
    categories : list[str]
        Categorical columns to plot.
    response_col : str
        Column containing RECIST response values.
    responders : list[str]
        RECIST values treated as responders. Default ["CR", "PR"].
    non_responders : list[str]
        RECIST values treated as non-responders. Default ["SD", "PD"].
    colors : list[str], optional
        One color per unique category value. Uses built-in palette if None.
    label_style : {"count", "pct", "both"}
        Text shown inside each bar segment.
    legend_position : {"responder", "below", "outside"}
        Where to place the legend:
        - "responder" → compact box anchored above the Responder bar.
          Falls back to below x-axis if the responder bar is too tall.
          Best choice when responders are the minority (typical in ICI data).
        - "below"     → horizontal legend centred below the x-axis labels.
          Clean and predictable; works for any bar height distribution.
        - "outside"   → vertical legend in the right margin of each subplot.
          Best when category has many unique values with long names.
    figsize : tuple, optional
    title : str
    min_bar_height_for_label : float
        Minimum segment height fraction before printing a label. Default 0.04.

    Returns
    -------
    plt.Figure
    """

    # ── Validation ────────────────────────────────────────────────────────────
    if response_col not in df.columns:
        raise ValueError(f"response_col '{response_col}' not found in DataFrame.")
    missing = [c for c in categories if c not in df.columns]
    if missing:
        raise ValueError(f"Categories not found in DataFrame: {missing}")

    valid = set(responders) | set(non_responders)
    n_dropped = (~df[response_col].isin(valid)).sum()
    if n_dropped > 0:
        warnings.warn(
            f"{n_dropped} rows dropped: response values not in "
            f"responders or non_responders lists.", UserWarning
        )
    df = df[df[response_col].isin(valid)].copy()

    # ── Group labels ──────────────────────────────────────────────────────────
    df["_group"] = df[response_col].apply(
        lambda x: "Responder" if x in responders else "Non-Responder"
    )
    groups = ["Responder", "Non-Responder"]
    group_colors = {"Responder": "#059669", "Non-Responder": "#DC2626"}
    recist_subtitles = {
        "Responder":     f"({', '.join(responders)})",
        "Non-Responder": f"({', '.join(non_responders)})",
    }

    # ── Default palette ───────────────────────────────────────────────────────
    DEFAULT_COLORS = [
        "#2E86AB", "#A23B72", "#F18F01", "#C73E1D",
        "#44BBA4", "#6C4E8A", "#E8936A", "#5BA4CF",
        "#8B5E3C", "#2D6A4F", "#D4A5A5", "#457B9D",
    ]
    if colors is None:
        colors = DEFAULT_COLORS

    # ── Layout ────────────────────────────────────────────────────────────────
    n = len(categories)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    # Extra right margin per subplot when legend is outside
    right_margin = 1.6 if legend_position == "outside" else 0.0
    # Extra bottom space per subplot when legend is below
    bottom_extra = 0.55 if legend_position == "below" else 0.0

    if figsize is None:
        figsize = (ncols * (4.2 + right_margin), nrows * (4.2 + bottom_extra))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)

    axes_flat = (
        [axes] if n == 1
        else axes.flatten() if hasattr(axes, "flatten")
        else list(axes)
    )

    # ── Plot ──────────────────────────────────────────────────────────────────
    for idx, (col, ax) in enumerate(zip(categories, axes_flat)):

        unique_vals = sorted(df[col].dropna().unique().tolist(), key=str)
        val_colors = {v: colors[i % len(colors)] for i, v in enumerate(unique_vals)}

        counts = (
            df.groupby(["_group", col])
              .size()
              .unstack(fill_value=0)
              .reindex(index=groups, columns=unique_vals, fill_value=0)
        )
        totals  = counts.sum(axis=1)
        max_total = totals.max()

        bar_positions = [0, 1]
        bar_width = 0.55
        bottoms = {g: 0 for g in groups}

        # ── Draw stacked bars ─────────────────────────────────────────────────
        for val in unique_vals:
            heights = [counts.loc[g, val] for g in groups]
            bars = ax.bar(
                bar_positions, heights, bar_width,
                bottom=[bottoms[g] for g in groups],
                color=[val_colors[val]] * 2,
                edgecolor="white", linewidth=0.8, alpha=0.9,
                label=str(val),
            )
            for bar, grp, height in zip(bars, groups, heights):
                if height == 0:
                    continue
                total = totals[grp]
                frac  = height / max_total if max_total > 0 else 0
                if frac < min_bar_height_for_label:
                    bottoms[grp] += height
                    continue

                pct   = (height / total * 100) if total > 0 else 0
                mid_y = bottoms[grp] + height / 2

                if label_style == "count":
                    txt = str(int(height))
                elif label_style == "pct":
                    txt = f"{pct:.0f}%"
                else:
                    txt = f"{int(height)}\n{pct:.0f}%"

                ax.text(
                    bar.get_x() + bar.get_width() / 2, mid_y, txt,
                    ha="center", va="center",
                    fontsize=8, fontweight="bold", color="white",
                )
                bottoms[grp] += height

        # ── n= labels above bars ──────────────────────────────────────────────
        for pos, grp in zip(bar_positions, groups):
            ax.text(
                pos, totals[grp] + max_total * 0.02,
                f"n={int(totals[grp])}",
                ha="center", va="bottom",
                fontsize=8.5, fontweight="bold", color="#333333",
            )

        # ── X-axis labels ─────────────────────────────────────────────────────
        ax.set_xticks(bar_positions)
        ax.set_xticklabels(groups, fontsize=9.5, fontweight="bold")
        for tick, grp in zip(ax.get_xticklabels(), groups):
            tick.set_color(group_colors[grp])

        # RECIST sub-labels (CR/PR) / (SD/PD) below x-tick labels
        for pos, grp in zip(bar_positions, groups):
            ax.text(
                pos, -max_total * 0.09,
                recist_subtitles[grp],
                ha="center", va="top",
                fontsize=7.5, color="gray",
                transform=ax.transData,
            )

        # Shared legend handles for this subplot
        handles = [
            mpatches.Patch(facecolor=val_colors[v], label=str(v), alpha=0.9)
            for v in unique_vals
        ]

        # ── LEGEND PLACEMENT ──────────────────────────────────────────────────
        if legend_position == "below":
            # Horizontal legend centred below axis — never overlaps bars
            ax.legend(
                handles=handles,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.18),
                ncol=min(len(unique_vals), 3),
                fontsize=8,
                framealpha=0.0,
                title=col.replace("_", " ").title(),
                title_fontsize=7.5,
                handlelength=1.0,
                handleheight=0.9,
                columnspacing=0.8,
            )

        elif legend_position == "outside":
            # Vertical legend in the right margin — zero overlap guaranteed
            ax.legend(
                handles=handles,
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                fontsize=8,
                framealpha=0.7,
                title=col.replace("_", " ").title(),
                title_fontsize=7.5,
                handlelength=1.0,
                borderpad=0.6,
            )

        else:
            # "responder" — legend above the Responder bar (pos=0)
            # Measure where the top of the Responder bar sits
            responder_top = totals["Responder"]
            available_frac = 1.0 - (responder_top / max_total)

            # Estimate legend height: ~0.065 of y-range per item
            legend_frac = len(unique_vals) * 0.065

            if available_frac >= legend_frac + 0.05:
                # Enough whitespace above Responder bar → place legend there
                ax.legend(
                    handles=handles,
                    loc="upper left",
                    bbox_to_anchor=(-0.02, 0.98),
                    fontsize=8,
                    framealpha=0.75,
                    title=col.replace("_", " ").title(),
                    title_fontsize=7.5,
                    handlelength=1.0,
                    borderpad=0.5,
                )
            else:
                # Fallback: below x-axis (same as "below" mode)
                ax.legend(
                    handles=handles,
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.18),
                    ncol=min(len(unique_vals), 3),
                    fontsize=8,
                    framealpha=0.0,
                    title=col.replace("_", " ").title(),
                    title_fontsize=7.5,
                    handlelength=1.0,
                    columnspacing=0.8,
                )

        # ── Axis formatting ───────────────────────────────────────────────────
        ax.set_title(
            col.replace("_", " ").title(),
            fontsize=11, fontweight="bold", pad=8,
        )
        ax.set_ylabel("Count", fontsize=9)
        ax.set_ylim(0, max_total * 1.18)
        ax.set_xlim(-0.6, 1.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", labelsize=8)
        ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        ax.set_axisbelow(True)

    # Hide unused axes
    for j in range(idx + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    return fig




def plot_response_boxplots(
    df: pd.DataFrame,
    numerical_cols: list[str],
    response_col: str,
    responders: list[str] = ["CR", "PR"],
    non_responders: list[str] = ["SD", "PD"],
    colors: list[str] = None,
    show_points: Literal["jitter", "none"] = "jitter",
    show_pvalue: Literal["mannwhitney", "ttest", "none"] = "mannwhitney",
    figsize: tuple = None,
    title: str = "ICI Response — Numerical Feature Distributions",
) -> plt.Figure:
    """
    Boxplots comparing Responder vs Non-Responder groups across
    numerical clinical features.

    Each subplot = one numerical feature.
    Each subplot has two side-by-side boxplots:
        - Left  → Responder  (CR / PR)
        - Right → Non-Responder (SD / PD)

    Includes optional jitter overlay, median annotation,
    outlier markers, and statistical significance bracket.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    numerical_cols : list[str]
        Numeric column names to plot
        (e.g. ["age", "hemoglobin", "neutrophil_count"]).
    response_col : str
        Column containing RECIST response values (e.g. "recist").
    responders : list[str]
        RECIST values treated as Responders. Default ["CR", "PR"].
        Also accepts "CRPR" as a combined label — split automatically.
    non_responders : list[str]
        RECIST values treated as Non-Responders. Default ["SD", "PD"].
    colors : list[str], optional
        Exactly two colors: [responder_color, non_responder_color].
        Defaults to ["#2E86AB", "#C73E1D"] (blue / red).
    show_points : {"jitter", "none"}
        "jitter" overlays individual data points with random horizontal
        jitter so the underlying distribution is visible through the box.
        Default "jitter".
    show_pvalue : {"mannwhitney", "ttest", "none"}
        Statistical test displayed as a significance bracket above each plot:
        - "mannwhitney" → Mann-Whitney U test (non-parametric, recommended
          for clinical data which is rarely normally distributed).
        - "ttest"       → Welch's independent t-test (assumes normality).
        - "none"        → no bracket drawn.
        Default "mannwhitney".
    figsize : tuple, optional
        (width, height) in inches. Auto-calculated if None.
    title : str
        Overall figure title.

    Returns
    -------
    plt.Figure

    Raises
    ------
    ValueError
        If response_col or any numerical_col is missing from df,
        or if a column is non-numeric.

    Examples
    --------

    """

    # ── Validation ────────────────────────────────────────────────────────────
    if response_col not in df.columns:
        raise ValueError(
            f"response_col '{response_col}' not found in DataFrame."
        )
    missing = [c for c in numerical_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in DataFrame: {missing}")

    non_numeric = [
        c for c in numerical_cols
        if not pd.api.types.is_numeric_dtype(df[c])
    ]
    if non_numeric:
        raise ValueError(f"Non-numeric columns: {non_numeric}")

    # ── Expand "CRPR" shorthand ───────────────────────────────────────────────
    # Allow user to pass "CRPR" as a convenience label
    responders = list({
        r for raw in responders
        for r in (["CR", "PR"] if raw.upper() == "CRPR" else [raw])
    })

    # ── Filter to known response values ──────────────────────────────────────
    valid = set(responders) | set(non_responders)
    n_dropped = (~df[response_col].isin(valid)).sum()
    if n_dropped > 0:
        warnings.warn(
            f"{n_dropped} rows dropped: RECIST values not in "
            f"responders or non_responders lists.", UserWarning
        )
    df = df[df[response_col].isin(valid)].copy()

    # ── Group assignment ──────────────────────────────────────────────────────
    df["_group"] = df[response_col].apply(
        lambda x: "Responder" if x in responders else "Non-Responder"
    )
    groups      = ["Responder", "Non-Responder"]
    subtitles   = {
        "Responder":     f"({' / '.join(sorted(responders))})",
        "Non-Responder": f"({' / '.join(sorted(non_responders))})",
    }

    # ── Colors ────────────────────────────────────────────────────────────────
    if colors is None:
        colors = ["#2E86AB", "#C73E1D"]
    if len(colors) < 2:
        raise ValueError("colors must contain at least 2 entries.")
    grp_colors = dict(zip(groups, colors[:2]))

    # ── Layout ────────────────────────────────────────────────────────────────
    n     = len(numerical_cols)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    if figsize is None:
        figsize = (ncols * 4.0, nrows * 4.2)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)

    axes_flat = (
        [axes] if n == 1
        else axes.flatten() if hasattr(axes, "flatten")
        else list(axes)
    )

    rng = np.random.default_rng(seed=42)   # reproducible jitter

    # ── Plot each feature ─────────────────────────────────────────────────────
    for idx, (col, ax) in enumerate(zip(numerical_cols, axes_flat)):

        group_data = {
            g: df.loc[df["_group"] == g, col].dropna().values
            for g in groups
        }

        # ── Box plots ─────────────────────────────────────────────────────────
        positions = [1, 2]
        bp = ax.boxplot(
            [group_data[g] for g in groups],
            positions=positions,
            widths=0.50,
            patch_artist=True,         # filled boxes
            showfliers=True,
            showmeans=True,
            meanprops=dict(
                marker="D",
                markerfacecolor="white",
                markeredgecolor="white",
                markersize=4,
            ),
            medianprops=dict(color="white", linewidth=2.0),
            whiskerprops=dict(linestyle="--", linewidth=1.0),
            capprops=dict(linewidth=1.2),
            flierprops=dict(
                marker="o", markersize=4,
                markeredgewidth=0.8,
                linestyle="none",
            ),
        )

        for patch, grp in zip(bp["boxes"], groups):
            clr = grp_colors[grp]
            patch.set_facecolor(clr)
            patch.set_alpha(0.82)
            patch.set_edgecolor(clr)

        for element in ["whiskers", "caps"]:
            for line, grp in zip(
                bp[element][::2], groups
            ):   # pairs per group
                pass  # color set below

        # Color whiskers, caps, fliers per group
        for i, grp in enumerate(groups):
            clr = grp_colors[grp]
            for item in ["whiskers", "caps"]:
                for line in bp[item][i * 2: i * 2 + 2]:
                    line.set_color(clr)
                    line.set_alpha(0.75)
            for flier in bp["fliers"][i: i + 1]:
                flier.set_markerfacecolor("none")
                flier.set_markeredgecolor(clr)
                flier.set_alpha(0.6)

        # ── Jitter overlay ────────────────────────────────────────────────────
        if show_points == "jitter":
            for pos, grp in zip(positions, groups):
                data = group_data[grp]
                jitter = rng.uniform(-0.18, 0.18, size=len(data))
                ax.scatter(
                    pos + jitter, data,
                    color=grp_colors[grp],
                    alpha=0.25, s=12,
                    linewidths=0, zorder=2,
                )

        # ── Median annotation ─────────────────────────────────────────────────
        for pos, grp in zip(positions, groups):
            med = np.median(group_data[grp])
            ax.text(
                pos, med,
                f" {med:.2f}",
                va="center", ha="left",
                fontsize=7.5, color='black',
                fontweight="bold", zorder=5,
            )

        # ── Statistical significance bracket ──────────────────────────────────
        if show_pvalue != "none":
            a, b = group_data["Responder"], group_data["Non-Responder"]

            if len(a) >= 4 and len(b) >= 4:
                if show_pvalue == "mannwhitney":
                    _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
                    test_label = "MWU"
                else:
                    _, p = stats.ttest_ind(a, b, equal_var=False)
                    test_label = "t-test"

                # Significance annotation
                if p < 0.001:
                    sig_text = "***"
                    sig_color = "#059669"
                elif p < 0.01:
                    sig_text = "**"
                    sig_color = "#059669"
                elif p < 0.05:
                    sig_text = "*"
                    sig_color = "#2563EB"
                else:
                    sig_text = "ns"
                    sig_color = "#9CA3AF"

                y_max = ax.get_ylim()[1]
                y_bracket = y_max * 0.93
                y_text    = y_max * 0.95

                ax.plot(
                    [1, 1, 2, 2],
                    [y_bracket - y_max * 0.02, y_bracket,
                     y_bracket, y_bracket - y_max * 0.02],
                    color=sig_color, linewidth=1.2,
                )
                ax.text(
                    1.5, y_text,
                    f"{sig_text}  p={p:.3f} ({test_label})",
                    ha="center", va="bottom",
                    fontsize=7.5, color=sig_color, fontweight="bold",
                )

        # ── X-axis labels ─────────────────────────────────────────────────────
        ax.set_xticks(positions)
        ax.set_xticklabels(groups, fontsize=9.5, fontweight="bold")
        for tick, grp in zip(ax.get_xticklabels(), groups):
            tick.set_color(grp_colors[grp])

        # RECIST sub-labels and n= counts
        for pos, grp in zip(positions, groups):
            n_grp = len(group_data[grp])
            ax.text(
                pos, ax.get_ylim()[0] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.09,
                f"{subtitles[grp]}  n={n_grp}",
                ha="center", va="top",
                fontsize=7.5, color="gray",
                transform=ax.transData,
            )

        # ── Formatting ────────────────────────────────────────────────────────
        ax.set_title(
            col.replace("_", " ").title(),
            fontsize=11, fontweight="bold", pad=8,
        )
        ax.set_ylabel(col.replace("_", " ").title(), fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", labelsize=8)
        ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        ax.set_axisbelow(True)
        ax.set_xlim(0.4, 2.6)

    # Hide unused axes
    for j in range(idx + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    return fig


def plot_cv_metrics(
    scores: list[pd.DataFrame],
    model_names: list[str],
    metric_cols: list[str] = None,
    figsize: tuple = (12, 6),
    title: str = "Repeated 5-Fold CV — Metric Distributions",
    palette: list[str] = None,
    show_points: bool = True,
    save_path: str = None
):
    """
    Plot boxplots for cross-validation metrics across one or more models.

    Parameters
    ----------
    scores      : list of pd.DataFrame — one DataFrame per model.
                  Each row = one fold, columns = metric names.
    model_names : list of str — display names for each model (same order as scores).
    metric_cols : list of str — which columns to plot. If None, uses all columns.
    figsize     : tuple — figure size.
    title       : str — chart title.
    palette     : list of str — hex colors, one per model.
    show_points : bool — overlay individual fold scores as jittered dots.
    save_path   : str — if provided, saves the figure to this path.

    Example
    -------
    # Each dataframe has one row per CV fold (e.g. 50 rows = 10 repeats × 5 folds)
    lr_scores = pd.DataFrame({
        'AUROC':       [0.81, 0.79, 0.83, ...],
        'AUPRC':       [0.72, 0.70, 0.74, ...],
        'Sensitivity': [0.76, 0.74, 0.79, ...],
        'Specificity': [0.84, 0.82, 0.86, ...],
        'F1':          [0.75, 0.73, 0.77, ...],
    })
    rf_scores = pd.DataFrame({ ... })

    plot_cv_metrics(
        scores=[lr_scores, rf_scores],
        model_names=['Logistic Regression', 'Random Forest'],
        metric_cols=['AUROC', 'AUPRC', 'Sensitivity', 'Specificity', 'F1'],
    )
    """
    if palette is None:
        palette = ['#378ADD', '#1D9E75', '#D85A30', '#9F4BB5', '#BA7517']

    n_models  = len(scores)
    metrics   = metric_cols if metric_cols else scores[0].columns.tolist()
    n_metrics = len(metrics)

    fig, axes = plt.subplots(1, n_metrics, figsize=figsize, sharey=True)
    if n_metrics == 1:
        axes = [axes]

    fig.suptitle(title, fontsize=13, fontweight='500', y=1.02)

    for ax_idx, metric in enumerate(metrics):
        ax = axes[ax_idx]
        data_per_model = [df[metric].dropna().values for df in scores]

        bp = ax.boxplot(
            data_per_model,
            patch_artist=True,
            notch=False,
            vert=True,
            widths=0.45,
            medianprops=dict(linewidth=2),
            flierprops=dict(marker='o', markersize=4, linestyle='none', alpha=0.5),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
        )

        for patch, color in zip(bp['boxes'], [palette[i % len(palette)] for i in range(n_models)]):
            patch.set_facecolor(color + '40')   # 25% alpha fill
            patch.set_edgecolor(color)
            patch.set_linewidth(1.5)

        for median, color in zip(bp['medians'], [palette[i % len(palette)] for i in range(n_models)]):
            median.set_color(color)

        for flier, color in zip(bp['fliers'], [palette[i % len(palette)] for i in range(n_models)]):
            flier.set_markerfacecolor(color)
            flier.set_markeredgecolor(color)

        # Jittered individual fold points
        if show_points:
            for model_idx, vals in enumerate(data_per_model):
                x_pos  = model_idx + 1
                jitter = np.random.uniform(-0.12, 0.12, size=len(vals))
                color  = palette[model_idx % len(palette)]
                ax.scatter(
                    x_pos + jitter, vals,
                    color=color, alpha=0.35, s=18, zorder=3, linewidths=0
                )

        ax.set_title(metric, fontsize=11, pad=6)
        ax.set_xticks([])
        ax.set_ylim(0, 1)
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.yaxis.set_tick_params(labelsize=9)
        ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.spines[['top', 'right', 'bottom']].set_visible(False)

    # Shared legend
    legend_patches = [
        mpatches.Patch(facecolor=palette[i % len(palette)] + '60',
                       edgecolor=palette[i % len(palette)],
                       label=name)
        for i, name in enumerate(model_names)
    ]
    fig.legend(
        handles=legend_patches,
        loc='lower center',
        ncol=n_models,
        fontsize=10,
        frameon=False,
        bbox_to_anchor=(0.5, -0.06)
    )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return fig



"""
expression_qc.py
================
Comprehensive QC plotting for RNA-seq and microarray expression data.

Usage
-----
    from expression_qc import plot_expression_qc

    # RNA-seq  — rows = genes, columns = samples, values = raw counts
    plot_expression_qc(counts_df, data_type="rnaseq", output_dir="qc_plots")

    # Microarray — rows = probes/genes, columns = samples, values = raw or log2 intensities
    plot_expression_qc(array_df, data_type="microarray", output_dir="qc_plots")

Plots generated
---------------
1.  Library size distribution (histogram) + outlier boxplot     (both)
2.  Expression density overlay (log2, all samples)              (both)
3.  PCA scatter  — PC1 vs PC2  and  PC1 vs PC3                 (both)
4.  PCA scree    — per-PC bars + elbow line + % labels           (both)
5.  Mean–Variance (dispersion) plot                             (both)
6.  Zero-count fraction — single boxplot + labelled strip       (RNA-seq only)
7.  Top-N HVG heatmap (Z-scored)                               (both)
"""


warnings.filterwarnings("ignore")
plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "figure.dpi": 120,
})

PALETTE = sns.color_palette("tab20", 20)

def _sample_colours(n):
    return [PALETTE[i % len(PALETTE)] for i in range(n)]


# ── helpers ───────────────────────────────────────────────────────────────────

def _log2_transform(df: pd.DataFrame, data_type: str, pseudo: float = 1.0) -> pd.DataFrame:
    if data_type == "microarray":
        if df.median().median() > 20:          # raw intensities → log2
            return np.log2(df + pseudo)
        return df.copy()                        # already log-scale
    return np.log2(df + pseudo)                 # RNA-seq: log2(count + 1)


def _top_variable_genes(log_df: pd.DataFrame, n: int = 500) -> pd.DataFrame:
    stds = log_df.std(axis=1)
    top_idx = stds.nlargest(min(n, len(stds))).index
    return log_df.loc[top_idx]


def _pca_on_samples(log_df: pd.DataFrame, n_components: int = 10):
    """PCA where each sample is an observation."""
    X = log_df.T                                # samples × genes
    hvg = log_df.std(axis=1).nlargest(min(5000, len(log_df))).index
    X = X[hvg]
    X_scaled = StandardScaler().fit_transform(X)
    n_comp = min(n_components, X.shape[0], X.shape[1])
    pca = PCA(n_components=n_comp)
    coords = pca.fit_transform(X_scaled)
    return coords, pca.explained_variance_ratio_ * 100, list(X.index)


def _iqr_outliers(series: pd.Series, k: float = 1.5):
    """Return boolean mask of IQR-based outliers."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return (series < q1 - k * iqr) | (series > q3 + k * iqr)


# ── 1. Library-size distribution + outlier boxplot ───────────────────────────

def plot_library_sizes(df: pd.DataFrame, data_type: str,
                       ax_hist: plt.Axes, ax_box: plt.Axes):
    """
    Left:  histogram of total library sizes across all samples.
    Right: single horizontal boxplot — each dot is one sample;
           IQR outliers are labelled with their sample name.
    """
    totals = df.sum(axis=0)
    scale  = 1e6
    vals   = totals / scale
    label  = "Total counts (M)" if data_type == "rnaseq" else "Total intensity (×10⁶)"

    # ── histogram ────────────────────────────────────────────────────────────
    ax_hist.hist(vals, bins=min(20, len(vals)), color="#4C72B0",
                 edgecolor="white", linewidth=0.6, alpha=0.85)
    ax_hist.axvline(vals.median(), color="#C44E52", linestyle="--", linewidth=1.4,
                    label=f"Median: {vals.median():.1f}M")
    ax_hist.set_xlabel(label, fontsize=11)
    ax_hist.set_ylabel("Number of samples", fontsize=11)
    ax_hist.set_title("Library size distribution", fontsize=12, fontweight="bold")
    ax_hist.legend(fontsize=9)

    # ── single boxplot with labelled outliers ────────────────────────────────
    ax_box.boxplot(vals, vert=False, patch_artist=True, widths=0.4,
                   boxprops=dict(facecolor="#AEC6E8", alpha=0.7),
                   medianprops=dict(color="#C44E52", linewidth=2),
                   flierprops=dict(marker="o", markersize=0),   # hide default fliers
                   whiskerprops=dict(linewidth=1),
                   capprops=dict(linewidth=1))

    # Overlay individual sample points (jittered vertically)
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.12, 0.12, size=len(vals))
    colours = _sample_colours(len(vals))
    for x, y, c in zip(vals.values, jitter, colours):
        ax_box.scatter(x, 1 + y, color=c, s=30, zorder=4,
                       edgecolors="white", linewidth=0.4)

    # Label IQR outliers
    outlier_mask = _iqr_outliers(vals)
    for sname, x, y in zip(vals.index, vals.values, jitter):
        if outlier_mask[sname]:
            ax_box.annotate(
                sname, xy=(x, 1 + y),
                xytext=(6, 0), textcoords="offset points",
                fontsize=8, color="#C44E52",
                arrowprops=dict(arrowstyle="-", color="#C44E52",
                                lw=0.8, shrinkA=0, shrinkB=2),
            )

    ax_box.set_xlabel(label, fontsize=11)
    ax_box.set_yticks([])
    ax_box.set_title("Library size — outlier view\n(labelled = IQR outliers)",
                     fontsize=12, fontweight="bold")
    ax_box.grid(axis="x", alpha=0.3)
    ax_box.spines["left"].set_visible(False)


# ── 2. Expression density overlay ─────────────────────────────────────────────

def plot_density(log_df: pd.DataFrame, ax: plt.Axes):
    colours = _sample_colours(len(log_df.columns))
    for col, c in zip(log_df.columns, colours):
        vals = log_df[col].dropna().values
        kde  = gaussian_kde(vals, bw_method=0.3)
        xs   = np.linspace(vals.min(), vals.max(), 300)
        ax.plot(xs, kde(xs), color=c, alpha=0.6, linewidth=1.0, label=col)
    ax.set_xlabel("log₂ expression", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Expression density (all samples)", fontsize=12, fontweight="bold")
    if len(log_df.columns) <= 20:
        ax.legend(fontsize=7, ncol=2, framealpha=0.5)


# ── 3. PCA scatter ────────────────────────────────────────────────────────────

def plot_pca(log_df: pd.DataFrame, ax12: plt.Axes, ax13: plt.Axes,
             label_samples: bool = True):
    """
    PCA scatter plots (PC1 vs PC2, PC1 vs PC3).

    Parameters
    ----------
    label_samples : bool
        If True (default), annotate each point with its sample name.
        Set to False when you have many samples and labels would overlap.
    """
    coords, var_exp, samples = _pca_on_samples(log_df)
    colours = _sample_colours(len(samples))

    for ax, pc_y, label_y in [(ax12, 1, 2), (ax13, 2, 3)]:
        if coords.shape[1] <= pc_y:
            ax.set_visible(False)
            continue
        for i, (s, c) in enumerate(zip(samples, colours)):
            ax.scatter(coords[i, 0], coords[i, pc_y], color=c, s=60,
                       zorder=3, edgecolors="white", linewidth=0.5)
            if label_samples:
                ax.annotate(s, (coords[i, 0], coords[i, pc_y]),
                            fontsize=6, alpha=0.75,
                            xytext=(4, 3), textcoords="offset points")
        ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
        ax.axvline(0, color="grey", linewidth=0.5, linestyle="--")
        ax.set_xlabel(f"PC1  ({var_exp[0]:.1f}% var)", fontsize=10)
        ax.set_ylabel(f"PC{label_y}  ({var_exp[pc_y]:.1f}% var)", fontsize=10)
        ax.set_title(f"PCA — PC1 vs PC{label_y}", fontsize=12, fontweight="bold")


# ── 4. Scree plot ─────────────────────────────────────────────────────────────

def plot_scree(log_df: pd.DataFrame, ax: plt.Axes):
    """
    Scree plot matching the reference figure:
    - Blue bars showing per-PC % variance (strictly decreasing left → right).
    - Black connected line running through the TOP of each bar (elbow curve).
    - Percentage label printed above each bar.
    Single y-axis only — no cumulative line, no twin axis.
    """
    _, var_exp, _ = _pca_on_samples(log_df)
    # PCA already returns components sorted by descending variance
    pcs = np.arange(1, len(var_exp) + 1)

    # Bars
    ax.bar(pcs, var_exp, color="#4C72B0", alpha=0.85,
           edgecolor="white", linewidth=0.5, zorder=2)

    # Elbow line through bar tops
    ax.plot(pcs, var_exp, color="black", marker="o",
            markersize=4, linewidth=1.5, zorder=3)

    # Percentage labels above each bar
    for pc, v in zip(pcs, var_exp):
        ax.text(pc, v + var_exp[0] * 0.02, f"{v:.1f}%",
                ha="center", va="bottom", fontsize=7, color="black")

    ax.set_xlabel("Dimensions", fontsize=11)
    ax.set_ylabel("Percentage of explained variances", fontsize=11)
    ax.set_xticks(pcs)
    ax.set_ylim(0, var_exp[0] * 1.22)   # headroom for labels
    ax.set_title("PCA scree plot", fontsize=12, fontweight="bold")


# ── 5. Mean–Variance ─────────────────────────────────────────────────────────

def plot_mean_variance(log_df: pd.DataFrame, ax: plt.Axes):
    means = log_df.mean(axis=1)
    varis = log_df.var(axis=1)
    if len(means) > 5000:
        hb = ax.hexbin(means, varis, gridsize=60, cmap="YlOrRd",
                       mincnt=1, linewidths=0.2)
        plt.colorbar(hb, ax=ax, label="Gene count")
    else:
        ax.scatter(means, varis, alpha=0.3, s=6,
                   color="#4C72B0", rasterized=True)
    ax.set_xlabel("Mean log₂ expression", fontsize=11)
    ax.set_ylabel("Variance", fontsize=11)
    ax.set_title("Mean–Variance (dispersion) plot", fontsize=12, fontweight="bold")


# ── 6. Zero-count fraction — scalable boxplot + strip ────────────────────────

def plot_zero_fraction(df: pd.DataFrame, ax: plt.Axes):
    """
    Single horizontal boxplot of zero-count % across all samples.
    Each sample is one dot; IQR outliers (unusually high zero fraction)
    are annotated with their sample name.
    Works for any number of samples.
    """
    zero_pct = (df == 0).mean(axis=0) * 100   # % zero per sample

    ax.boxplot(zero_pct, vert=False, patch_artist=True, widths=0.4,
               boxprops=dict(facecolor="#FFD6A5", alpha=0.75),
               medianprops=dict(color="#C44E52", linewidth=2),
               flierprops=dict(marker="o", markersize=0),
               whiskerprops=dict(linewidth=1),
               capprops=dict(linewidth=1))

    rng    = np.random.default_rng(1)
    jitter = rng.uniform(-0.12, 0.12, size=len(zero_pct))
    colours = _sample_colours(len(zero_pct))

    for x, y, c in zip(zero_pct.values, jitter, colours):
        ax.scatter(x, 1 + y, color=c, s=30, zorder=4,
                   edgecolors="white", linewidth=0.4)

    # Label outliers
    outlier_mask = _iqr_outliers(zero_pct)
    for sname, x, y in zip(zero_pct.index, zero_pct.values, jitter):
        if outlier_mask[sname]:
            ax.annotate(
                sname, xy=(x, 1 + y),
                xytext=(6, 0), textcoords="offset points",
                fontsize=8, color="#C44E52",
                arrowprops=dict(arrowstyle="-", color="#C44E52",
                                lw=0.8, shrinkA=0, shrinkB=2),
            )

    ax.set_xlabel("Zero-count genes (%)", fontsize=11)
    ax.set_yticks([])
    ax.set_title("Zero-count fraction per sample\n(labelled = IQR outliers)",
                 fontsize=12, fontweight="bold")
    ax.spines["left"].set_visible(False)


# ── 7. HVG heatmap ────────────────────────────────────────────────────────────

def plot_hvg_heatmap(log_df: pd.DataFrame, ax: plt.Axes, n: int = 50):
    top  = _top_variable_genes(log_df, n=n)
    z    = top.subtract(top.mean(axis=1), axis=0).divide(
               top.std(axis=1).replace(0, 1), axis=0)
    sns.heatmap(z, ax=ax, cmap="RdBu_r", center=0, vmin=-3, vmax=3,
                xticklabels=True, yticklabels=(len(top) <= 60),
                linewidths=0,
                cbar_kws={"shrink": 0.6, "label": "Z-score"})
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=6)
    ax.set_title(f"Top {n} variable genes/probes  (Z-scored)",
                 fontsize=12, fontweight="bold")


# ── main public function ──────────────────────────────────────────────────────

def plot_expression_qc(
    df: pd.DataFrame,
    data_type: str = "rnaseq",
    output_dir: str = "qc_plots",
    sample_labels: list = None,
    prefix: str = "qc",
    n_hvg_heatmap: int = 50,
    pca_label_samples: bool = True,
    save_pdf: bool = True,
    save_png: bool = True,
    show: bool = True,
) -> dict:
    """
    Generate a streamlined, scalable QC report for expression data.

    Parameters
    ----------
    df : pd.DataFrame
        Expression matrix.  Rows = genes/probes, columns = samples.
        RNA-seq  : raw integer counts.
        Microarray: raw or log2 probe intensities (auto-detected).
    data_type : str
        "rnaseq" or "microarray"
    output_dir : str
        Directory for output files (created if absent).
    sample_labels : list, optional
    prefix : str
        Filename prefix for saved files.
    n_hvg_heatmap : int
        Number of top variable genes in the heatmap (default 50).
    pca_label_samples : bool
        If True (default), annotate each PCA point with its sample name.
        Set to False when you have many samples and labels would overlap.
    save_pdf : bool
        Bundle all plots into one PDF (default True).
    save_png : bool
        Save each plot as a PNG (default True).
    show : bool
        Call plt.show() interactively (default True).

    Returns
    -------
    dict  {plot_name: Figure}
    """
    data_type = data_type.lower().strip()
    assert data_type in ("rnaseq", "microarray"), \
        "data_type must be 'rnaseq' or 'microarray'"

    if sample_labels is not None:
        assert len(sample_labels) == df.shape[1], \
            "sample_labels length must match column count"
        df = df.copy()
        df.columns = sample_labels

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    log_df      = _log2_transform(df, data_type)
    n_samples   = df.shape[1]
    dtype_label = ("RNA-seq (counts)" if data_type == "rnaseq"
                   else "Microarray (log2 expression)")

    figs = {}

    def _save(fig, name):
        figs[name] = fig
        if save_png:
            fig.savefig(out / f"{prefix}_{name}.png",
                        bbox_inches="tight", dpi=150)

    # ── 1. Library size ───────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle(f"{dtype_label} — Library sizes",
                 fontsize=13, fontweight="bold", y=1.01)
    plot_library_sizes(df, data_type, ax1, ax2)
    fig.tight_layout()
    _save(fig, "01_library_sizes")

    # ── 2. Expression density ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_density(log_df, ax)
    fig.suptitle(f"{dtype_label}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "02_density")

    # ── 3 & 4. PCA scatter + scree ────────────────────────────────────────────
    #fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    #fig.suptitle(f"{dtype_label} — PCA", fontsize=13, fontweight="bold")
    #plot_pca(log_df, axes[0], axes[1], label_samples=pca_label_samples)
    #plot_scree(log_df, axes[2])
    #fig.tight_layout()
    #_save(fig, "03_pca")

    # ── 5. Mean–Variance ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    plot_mean_variance(log_df, ax)
    fig.suptitle(f"{dtype_label}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "03_mean_variance")

    # ── 6. Zero-count fraction (RNA-seq only) ─────────────────────────────────
    if data_type == "rnaseq":
        fig, ax = plt.subplots(figsize=(8, 3.5))
        plot_zero_fraction(df, ax)
        fig.suptitle(f"{dtype_label}", fontsize=13, fontweight="bold")
        fig.tight_layout()
        _save(fig, "04_zero_fraction")

    # ── 7. HVG heatmap ────────────────────────────────────────────────────────
    #fig, ax = plt.subplots(figsize=(max(8, n_samples * 0.5), 10))
    #plot_hvg_heatmap(log_df, ax, n=n_hvg_heatmap)
    #fig.suptitle(f"{dtype_label}", fontsize=13, fontweight="bold")
    #fig.tight_layout()
    #save(fig, "05_hvg_heatmap")

    # ── PDF bundle ────────────────────────────────────────────────────────────
    if save_pdf:
        pdf_path = out / f"{prefix}_full_report.pdf"
        with PdfPages(pdf_path) as pdf:
            for fig in figs.values():
                pdf.savefig(fig, bbox_inches="tight")
        print(f"[QC] PDF saved → {pdf_path}")

    print(f"[QC] {len(figs)} plots saved to: {out.resolve()}")

    if show:
        plt.show()

    return figs


# ── demo ─────────────────────────────────────────────────────────────────────

def _make_demo_rnaseq(n_genes=8000, n_samples=12, seed=42):
    rng  = np.random.default_rng(seed)
    base = rng.negative_binomial(10, 0.3, size=(n_genes, n_samples)).astype(float)
    base[:1000, 6:] *= rng.uniform(2, 5, size=(1000, 6))
    base[:, 3]  *= 0.15   # outlier: very low depth
    base[:, 9]  *= 3.5    # outlier: very high depth
    cols = [f"S{i+1:02d}" for i in range(n_samples)]
    idx  = [f"Gene_{i+1}" for i in range(n_genes)]
    return pd.DataFrame(base.astype(int), index=idx, columns=cols)


def _make_demo_microarray(n_probes=20000, n_samples=10, seed=7):
    rng  = np.random.default_rng(seed)
    base = np.clip(rng.normal(8, 2, size=(n_probes, n_samples)), 3, 16)
    base[:500, 5:] += rng.uniform(1, 3, size=(500, 5))
    cols = [f"Array_{i+1}" for i in range(n_samples)]
    idx  = [f"Probe_{i+1}" for i in range(n_probes)]
    return pd.DataFrame(base, index=idx, columns=cols)

'''
if __name__ == "__main__":
    print("=" * 55)
    print("Demo 1: RNA-seq")
    print("=" * 55)
    plot_expression_qc(_make_demo_rnaseq(), data_type="rnaseq",
                       output_dir="qc_demo_rnaseq", prefix="rnaseq", show=False)

    print("\n" + "=" * 55)
    print("Demo 2: Microarray")
    print("=" * 55)
    plot_expression_qc(_make_demo_microarray(), data_type="microarray",
                       output_dir="qc_demo_microarray", prefix="array", show=False)

    print("\nAll done.")
'''


"""
pca_clinical.py
===============
Standalone PCA plotting for gene expression data coloured and shaped
by clinical variables.

Usage
-----
    from pca_clinical import plot_clinical_pca

    plot_clinical_pca(
        expr_df    = counts_df,          # genes × samples
        clinical_df= clinical_df,        # samples × clinical columns
        data_type  = "rnaseq",           # or "microarray"
        color_cols = ["drug_response", "grade", "sex"],
        shape_col  = "sex",              # optional — one column for marker shape
        pcs        = [(1,2), (1,3)],     # which PC pairs to plot
        label_samples = False,
        output_dir = "pca_plots",
    )

What it produces
----------------
One figure per (color_col × PC-pair) combination, each saved as a PNG.
A single PDF bundles every figure at the end.

Input requirements
------------------
- expr_df   : rows = genes/probes, columns = sample IDs
- clinical_df: rows = sample IDs (index), columns = clinical variables
  Sample IDs must overlap between the two dataframes.
  Columns can be categorical (str / int codes) or continuous (float).
"""

import warnings
from pathlib import Path
from itertools import cycle

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "figure.dpi": 120,
})

# ── constants ─────────────────────────────────────────────────────────────────

# Colour palette for categorical variables (up to 20 categories)
_CAT_PALETTE = [
    "#4C72B0", "#C44E52", "#55A868", "#DD8452", "#8172B2",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
    "#1B6CA8", "#A32020", "#207A3C", "#A05C1E", "#5A4A8A",
    "#6B5040", "#A0537A", "#4D4D4D", "#8C7A30", "#3A7F9A",
]

# Marker shapes for the shape_col variable (up to 8 categories)
_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


# ── helpers ───────────────────────────────────────────────────────────────────

def _log2_transform(df: pd.DataFrame, data_type: str) -> pd.DataFrame:
    """Auto-detect scale and return log2-transformed matrix."""
    if data_type == "microarray":
        if df.median().median() > 20:
            return np.log2(df + 1)
        return df.copy()
    return np.log2(df + 1)   # RNA-seq: log2(count + 1)


def _run_pca(log_df: pd.DataFrame, n_hvg: int = 5000,
             n_components: int = 10):
    """
    Run PCA on samples (transpose first).
    Returns
        coords       : ndarray  (n_samples × n_components)
        var_exp      : ndarray  % variance per PC
        sample_order : list of sample IDs in PCA row order
    """
    # Subset to top HVGs for speed & signal
    hvg = log_df.std(axis=1).nlargest(min(n_hvg, len(log_df))).index
    X = log_df.loc[hvg].T                     # samples × HVGs
    X_scaled = StandardScaler().fit_transform(X)
    n_comp = min(n_components, X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=n_comp)
    coords = pca.fit_transform(X_scaled)
    var_exp = pca.explained_variance_ratio_ * 100
    return coords, var_exp, list(X.index)


def _is_continuous(series: pd.Series, max_unique: int = 10) -> bool:
    """Heuristic: treat as continuous if numeric with many unique values."""
    return pd.api.types.is_float_dtype(series) and series.nunique() > max_unique


def _align_samples(expr_df: pd.DataFrame,
                   clinical_df: pd.DataFrame) -> tuple:
    """
    Keep only samples present in both dataframes.
    Returns aligned (expr_df, clinical_df) with a shared sample order.
    """
    common = expr_df.columns.intersection(clinical_df.index)
    if len(common) == 0:
        raise ValueError(
            "No overlapping sample IDs between expr_df columns "
            "and clinical_df index. Check that IDs match exactly."
        )
    missing_expr = set(expr_df.columns) - set(common)
    missing_clin = set(clinical_df.index) - set(common)
    if missing_expr:
        print(f"[PCA] Warning: {len(missing_expr)} expression samples "
              f"not in clinical_df — excluded: {sorted(missing_expr)[:5]}{'…' if len(missing_expr)>5 else ''}")
    if missing_clin:
        print(f"[PCA] Warning: {len(missing_clin)} clinical rows "
              f"not in expr_df — excluded: {sorted(missing_clin)[:5]}{'…' if len(missing_clin)>5 else ''}")
    common = list(common)
    return expr_df[common], clinical_df.loc[common]


# ── single-panel draw ─────────────────────────────────────────────────────────

def _draw_pca_panel(
    ax: plt.Axes,
    coords: np.ndarray,
    var_exp: np.ndarray,
    samples: list,
    clinical_col: pd.Series,
    pc_pair: tuple,
    shape_col: pd.Series = None,
    label_samples: bool = False,
    point_size: int = 70,
    draw_legend: bool = True,          # NEW — only the last panel passes True
):
    """
    Draw one PCA scatter panel on *ax*.

    When draw_legend=True the function returns a dict of legend handles for
    the caller to place at figure level.  For continuous colour variables the
    colorbar is drawn directly on the axis (it is self-contained) and None
    is returned.  When draw_legend=False nothing legend-related happens and
    None is returned.
    """
    pc_x, pc_y = pc_pair[0] - 1, pc_pair[1] - 1

    # ── colour mapping ────────────────────────────────────────────────────────
    continuous = _is_continuous(clinical_col)
    if continuous:
        norm_vals = (clinical_col - clinical_col.min()) / (
            clinical_col.max() - clinical_col.min() + 1e-9)
        cmap = plt.cm.viridis
        colour_map = {s: cmap(norm_vals[s]) for s in samples}
    else:
        categories = sorted(clinical_col.dropna().unique().tolist(), key=str)
        cat_colour = {cat: _CAT_PALETTE[i % len(_CAT_PALETTE)]
                      for i, cat in enumerate(categories)}
        colour_map = {s: cat_colour.get(clinical_col.get(s), "#cccccc")
                      for s in samples}

    # ── shape mapping ─────────────────────────────────────────────────────────
    if shape_col is not None:
        shape_cats = sorted(shape_col.dropna().unique().tolist(), key=str)
        shape_map  = {cat: _MARKERS[i % len(_MARKERS)]
                      for i, cat in enumerate(shape_cats)}
        marker_per_sample = {s: shape_map.get(shape_col.get(s), "o")
                             for s in samples}
    else:
        marker_per_sample = {s: "o" for s in samples}

    # ── scatter ───────────────────────────────────────────────────────────────
    for i, s in enumerate(samples):
        ax.scatter(
            coords[i, pc_x], coords[i, pc_y],
            color=colour_map[s],
            marker=marker_per_sample[s],
            s=point_size, zorder=3,
            edgecolors="white", linewidth=0.5, alpha=0.85,
        )
        if label_samples:
            ax.annotate(s, (coords[i, pc_x], coords[i, pc_y]),
                        fontsize=6, alpha=0.70,
                        xytext=(4, 3), textcoords="offset points")

    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--", zorder=1)
    ax.axvline(0, color="grey", linewidth=0.5, linestyle="--", zorder=1)
    ax.set_xlabel(f"PC{pc_pair[0]}  ({var_exp[pc_x]:.1f}% var)", fontsize=10)
    ax.set_ylabel(f"PC{pc_pair[1]}  ({var_exp[pc_y]:.1f}% var)", fontsize=10)

    # ── legend (only on the designated panel) ─────────────────────────────────
    if not draw_legend:
        return None

    if continuous:
        # Colorbar is axis-local — draw it here, return nothing for fig.legend
        sm = plt.cm.ScalarMappable(
            cmap=cmap,
            norm=plt.Normalize(vmin=clinical_col.min(), vmax=clinical_col.max()),
        )
        sm.set_array([])
        ax.figure.colorbar(sm, ax=ax, shrink=0.7, pad=0.02,
                           label=clinical_col.name)
        return None

    # Categorical: build handles and return them for figure-level placement
    colour_handles = [
        mpatches.Patch(facecolor=cat_colour[cat], edgecolor="white",
                       label=str(cat))
        for cat in categories
    ]
    shape_handles = []
    if shape_col is not None:
        shape_handles = [
            mlines.Line2D([], [], color="grey",
                          marker=shape_map[cat], linestyle="None",
                          markersize=7, label=str(cat))
            for cat in shape_cats
        ]

    return {
        "colour_handles": colour_handles,
        "colour_title":   str(clinical_col.name),
        "shape_handles":  shape_handles,
        "shape_title":    str(shape_col.name) if shape_col is not None else "",
    }


def _place_figure_legends(fig, last_ax: plt.Axes, legend_data: dict) -> list:
    """
    Add up to two figure-level legends anchored just to the right of
    the last (or only) subplot — no gap from using figure coordinates.
    """
    extra = []

    if legend_data.get("colour_handles"):
        leg_c = fig.legend(
            handles=legend_data["colour_handles"],
            title=legend_data["colour_title"],
            title_fontsize=8,
            fontsize=7,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            bbox_transform=last_ax.transAxes,   # ← relative to last subplot
            borderaxespad=0,
            framealpha=0.9,
        )
        extra.append(leg_c)

    if legend_data.get("shape_handles"):
        y_anchor = 0.45 if legend_data.get("colour_handles") else 1.0
        leg_s = fig.legend(
            handles=legend_data["shape_handles"],
            title=legend_data["shape_title"],
            title_fontsize=8,
            fontsize=7,
            loc="upper left",
            bbox_to_anchor=(1.02, y_anchor),
            bbox_transform=last_ax.transAxes,   # ← relative to last subplot
            borderaxespad=0,
            framealpha=0.9,
        )
        extra.append(leg_s)

    return extra


def _fit_legends(fig, extra_artists: list) -> None:
    """
    After tight_layout(), shrink the right subplot margin so that
    figure-level legends are fully inside the saved canvas.
    Works by measuring the actual rendered pixel widths.
    """
    if not extra_artists:
        return
    fig.canvas.draw()                                   # force renderer
    renderer  = fig.canvas.get_renderer()
    fig_right = fig.get_window_extent(renderer).x1      # figure right edge
    leg_right = max(a.get_window_extent(renderer).x1
                    for a in extra_artists)             # rightmost legend edge

    if leg_right > fig_right * 0.98:                    # legend overflows
        shrink = (fig_right / leg_right) - 0.02         # small extra pad
        fig.subplots_adjust(right=shrink)


def plot_clinical_pca(
    expr_df: pd.DataFrame,
    clinical_df: pd.DataFrame,
    data_type: str = "rnaseq",
    color_cols: list = None,
    shape_col: str = None,
    pcs: list = None,
    n_hvg: int = 5000,
    label_samples: bool = False,
    point_size: int = 70,
    output_dir: str = "pca_clinical",
    prefix: str = "pca",
    save_pdf: bool = True,
    save_png: bool = True,
    show: bool = True,
) -> dict:
    """
    Generate PCA plots coloured by clinical variables.
    (docstring unchanged — see original)
    """
    data_type = data_type.lower().strip()
    assert data_type in ("rnaseq", "microarray"), \
        "data_type must be 'rnaseq' or 'microarray'"

    if pcs is None:
        pcs = [(1, 2), (1, 3)]

    expr_df, clinical_df = _align_samples(expr_df, clinical_df)
    n_samples = expr_df.shape[1]
    print(f"[PCA] {n_samples} samples × {expr_df.shape[0]} features after alignment.")

    if color_cols is None:
        color_cols = list(clinical_df.columns)

    missing = [c for c in color_cols if c not in clinical_df.columns]
    if missing:
        raise ValueError(f"color_cols not found in clinical_df: {missing}")

    shape_series = None
    if shape_col is not None:
        if shape_col not in clinical_df.columns:
            raise ValueError(f"shape_col '{shape_col}' not found in clinical_df.")
        shape_series = clinical_df[shape_col]
        n_shapes = shape_series.nunique()
        if n_shapes > len(_MARKERS):
            print(f"[PCA] Warning: shape_col '{shape_col}' has {n_shapes} "
                  f"categories but only {len(_MARKERS)} marker shapes — "
                  f"shapes will repeat.")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    log_df = _log2_transform(expr_df, data_type)
    coords, var_exp, sample_order = _run_pca(log_df, n_hvg=n_hvg)

    max_pc    = coords.shape[1]
    valid_pcs = [(a, b) for (a, b) in pcs if a <= max_pc and b <= max_pc]
    if len(valid_pcs) < len(pcs):
        dropped = set(map(tuple, pcs)) - set(map(tuple, valid_pcs))
        print(f"[PCA] Warning: not enough components for {dropped} — skipped.")
    if not valid_pcs:
        raise ValueError("No valid PC pairs to plot.")

    n_pairs = len(valid_pcs)
    figs    = {}

    # ── one figure per color column ───────────────────────────────────────────
    for col in color_cols:
        color_series = clinical_df[col]

        fig, axes = plt.subplots(
            1, n_pairs,
            figsize=(6.0 * n_pairs, 5.0),
            squeeze=False,
        )
        axes = axes[0]

        dtype_label = "RNA-seq" if data_type == "rnaseq" else "Microarray"
        shape_label = f"  |  shape = {shape_col}" if shape_col else ""
        fig.suptitle(
            f"{dtype_label} PCA:  colour = {col}{shape_label}",
            fontsize=13, fontweight="bold",
        )

        legend_data   = None
        for i, (ax, pc_pair) in enumerate(zip(axes, valid_pcs)):
            result = _draw_pca_panel(
                ax=ax,
                coords=coords,
                var_exp=var_exp,
                samples=sample_order,
                clinical_col=color_series,
                pc_pair=pc_pair,
                shape_col=shape_series,
                label_samples=label_samples,
                point_size=point_size,
                draw_legend=(i == n_pairs - 1),    # only the last panel
            )
            if result is not None:
                legend_data = result

            ax.set_title(f"PC{pc_pair[0]} vs PC{pc_pair[1]}",
                         fontsize=11, fontweight="bold")

        # ── place ONE shared legend at figure level ───────────────────────────
        extra_artists = []
        if legend_data is not None:
            extra_artists = _place_figure_legends(fig, axes[-1],legend_data)

        # ── layout: tight first, then shrink right to fit the legend ──────────
        fig.tight_layout()
        _fit_legends(fig, extra_artists)

        safe_col = col.replace(" ", "_").replace("/", "_")
        name     = f"colour_{safe_col}"
        figs[name] = fig

        if save_png:
            path = out / f"{prefix}_{name}.png"
            # bbox_inches="tight" expands the canvas to capture legend overflow
            fig.savefig(path, bbox_inches="tight", dpi=150)
        print(f"[PCA] Figure saved: colour={col}")

    # ── PDF bundle ────────────────────────────────────────────────────────────
    if save_pdf and figs:
        pdf_path = out / f"{prefix}_clinical_pca.pdf"
        with PdfPages(pdf_path) as pdf:
            for fig in figs.values():
                pdf.savefig(fig, bbox_inches="tight")
        print(f"[PCA] PDF saved → {pdf_path}")

    if show:
        plt.show()

    return figs

# ── demo ─────────────────────────────────────────────────────────────────────

def _make_demo_data(n_genes=8000, n_samples=30, seed=42):
    """Simulate RNA-seq counts + clinical metadata."""
    rng = np.random.default_rng(seed)

    # Expression: two biological groups with distinct signatures
    counts = rng.negative_binomial(10, 0.3,
                                   size=(n_genes, n_samples)).astype(float)
    counts[:800, 15:] *= rng.uniform(2, 6, size=(800, 15))
    counts[:, 2]  *= 0.1   # low-depth outlier
    counts[:, 22] *= 4.0   # high-depth outlier

    sample_ids = [f"S{i+1:03d}" for i in range(n_samples)]
    expr = pd.DataFrame(counts.astype(int),
                        index=[f"Gene_{i}" for i in range(n_genes)],
                        columns=sample_ids)

    # Clinical metadata
    rng2 = np.random.default_rng(seed + 1)
    clinical = pd.DataFrame({
        "drug_response": (["Responder"] * 15 + ["Non-responder"] * 15),
        "grade":         (["I"] * 10 + ["II"] * 10 + ["III"] * 10),
        "sex":           rng2.choice(["Male", "Female"], size=n_samples).tolist(),
        "age":           rng2.integers(30, 80, size=n_samples).astype(float),
        "batch":         (["Batch_A"] * 10 + ["Batch_B"] * 10
                          + ["Batch_C"] * 10),
    }, index=sample_ids)

    return expr, clinical



"""
boxplot_categorical.py
======================
Flexible boxplot function for comparing a numerical variable across
categorical groups.

Usage
-----
    from boxplot_categorical import plot_categorical_boxplot

    plot_categorical_boxplot(
        df          = clinical_df,
        cat_col     = "drug_response",
        num_col     = "expression_score",
    )
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "figure.dpi": 120,
})

# 20-colour palette — categorical
_PALETTE = [
    "#4C72B0", "#C44E52", "#55A868", "#DD8452", "#8172B2",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
    "#1B6CA8", "#A32020", "#207A3C", "#A05C1E", "#5A4A8A",
    "#6B5040", "#A0537A", "#4D4D4D", "#8C7A30", "#3A7F9A",
]


def plot_categorical_boxplot(
    df: pd.DataFrame,
    cat_col: str,
    num_col: str,
    # ── ordering & filtering ──────────────────────────────────────────────────
    order: list = None,
    exclude_categories: list = None,
    # ── overlay points ────────────────────────────────────────────────────────
    show_points: bool = True,
    point_size: int = 40,
    point_alpha: float = 0.55,
    jitter: float = 0.15,
    title: str = None,
    # ── labels & title ────────────────────────────────────────────────────────
    plot_clinical_pca: str = None,
    xlabel: str = None,
    ylabel: str = None,
    # ── appearance ────────────────────────────────────────────────────────────
    palette: list = None,
    figsize: tuple = None,
    box_width: float = 0.45,
    box_alpha: float = 0.75,
    show_n: bool = True,           # add sample-size annotation under each box
    rotate_xlabels: int = 0,       # degrees to rotate x-tick labels
    # ── output ────────────────────────────────────────────────────────────────
    output_path: str = None,       # e.g. "plots/my_boxplot.png"
    show: bool = True,
) -> plt.Figure:
    """
    Boxplot of a numerical variable split by a categorical variable.
    Individual data points are overlaid as a strip plot (jittered).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    cat_col : str
        Column name for the categorical variable (x-axis).
    num_col : str
        Column name for the numerical variable (y-axis).
    order : list, optional
        Explicit left-to-right category order.
        Defaults to sorted unique values of cat_col.
    exclude_categories : list, optional
        Categories to drop before plotting (e.g. ["Unknown", "NA"]).
    show_points : bool
        Overlay individual data points as a strip plot (default True).
    point_size : int
        Marker size for strip points (default 40).
    point_alpha : float
        Transparency of strip points (default 0.55).
    jitter : float
        Horizontal jitter width for strip points (default 0.15).
    title : str, optional
        Figure title. Defaults to "{num_col} by {cat_col}".
    xlabel : str, optional
        X-axis label. Defaults to cat_col.
    ylabel : str, optional
        Y-axis label. Defaults to num_col.
    palette : list, optional
        List of hex colour strings, one per category.
        Defaults to built-in 20-colour palette.
    figsize : tuple, optional
        (width, height) in inches.
        Auto-calculated from number of categories if not provided.
    box_width : float
        Width of each box (default 0.45).
    box_alpha : float
        Transparency of box fill (default 0.75).
    show_n : bool
        Print "n=X" below each category label (default True).
    rotate_xlabels : int
        Degrees to rotate x-axis tick labels (default 0).
        Use 45 or 90 for long category names.
    output_path : str, optional
        If provided, save the figure to this path (PNG/PDF auto-detected
        from extension). Directory is created if it does not exist.
    show : bool
        Call plt.show() (default True).

    Returns
    -------
    matplotlib.figure.Figure
    """
    # ── validate inputs ───────────────────────────────────────────────────────
    for col in (cat_col, num_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataframe. "
                             f"Available: {list(df.columns)}")

    # ── prepare data ──────────────────────────────────────────────────────────
    plot_df = df[[cat_col, num_col]].dropna().copy()
    plot_df[cat_col] = plot_df[cat_col].astype(str)

    if exclude_categories:
        plot_df = plot_df[~plot_df[cat_col].isin(
            [str(c) for c in exclude_categories])]

    if order is None:
        # Try natural sort; fall back to lexicographic
        cats = sorted(plot_df[cat_col].unique().tolist(), key=str)
    else:
        cats = [str(c) for c in order]
        # Keep only cats that actually exist in data
        cats = [c for c in cats if c in plot_df[cat_col].unique()]

    n_cats = len(cats)
    if n_cats == 0:
        raise ValueError("No categories left to plot after filtering.")

    # Per-category sample counts
    counts = plot_df.groupby(cat_col)[num_col].count()

    # ── colours ───────────────────────────────────────────────────────────────
    pal = palette if palette is not None else _PALETTE
    colours = {cat: pal[i % len(pal)] for i, cat in enumerate(cats)}

    # ── figure size ───────────────────────────────────────────────────────────
    if figsize is None:
        w = max(5, n_cats * 1.4)
        figsize = (w, 5)

    fig, ax = plt.subplots(figsize=figsize)

    # ── draw boxes ────────────────────────────────────────────────────────────
    for i, cat in enumerate(cats):
        vals = plot_df.loc[plot_df[cat_col] == cat, num_col].values
        c    = colours[cat]

        bp = ax.boxplot(
            vals,
            positions=[i],
            widths=box_width,
            patch_artist=True,
            notch=False,
            boxprops    = dict(facecolor=c, alpha=box_alpha, linewidth=1.0),
            medianprops = dict(color="black", linewidth=2.0),
            whiskerprops= dict(color="black", linewidth=1.0),
            capprops    = dict(color="black", linewidth=1.0),
            flierprops  = dict(marker="o", markersize=0),  # hide — shown by strip
            zorder=2,
        )

        # ── strip (individual points) ─────────────────────────────────────────
        if show_points:
            rng     = np.random.default_rng(i)
            x_jit   = rng.uniform(-jitter, jitter, size=len(vals))
            ax.scatter(
                i + x_jit, vals,
                color=c, s=point_size, alpha=point_alpha,
                edgecolors="white", linewidth=0.4,
                zorder=3,
            )

    # ── axes labels & ticks ───────────────────────────────────────────────────
    tick_labels = cats
    if show_n:
        tick_labels = [
            f"{cat}\n(n={counts.get(cat, 0)})" for cat in cats
        ]

    ax.set_xticks(range(n_cats))
    ax.set_xticklabels(
        tick_labels,
        rotation=rotate_xlabels,
        ha="right" if rotate_xlabels > 0 else "center",
        fontsize=10,
    )

    ax.set_xlabel(xlabel or cat_col, fontsize=12)
    ax.set_ylabel(ylabel or num_col, fontsize=12)
    ax.set_title(title or f"{num_col}  by  {cat_col}",
                 fontsize=13, fontweight="bold", pad=10)

    ax.set_xlim(-0.6, n_cats - 0.4)

    fig.tight_layout()

    # ── save ──────────────────────────────────────────────────────────────────
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight", dpi=150)
        print(f"[boxplot] Saved → {out.resolve()}")

    if show:
        plt.show()

    return fig
def violin_plot_predictions(
    y_true,
    y_score,
    class_names: list[str] = None,
    colors: list[str] = None,
    title: str = "Predicted Scores by True Class",
    ylabel: str = "Predicted Probability",
    jitter: float = 0.05,
    point_alpha: float = 0.7,
    point_size: float = 12,
    figsize: tuple = (7, 5),
    threshold: float = None,
    ax=None,
):
    """
    Violin plot comparing predicted probabilities (or scores) across true class labels.

    Args:
        y_true:       True binary class labels (array-like of 0/1 or any two unique values).
        y_score:      Model predicted probabilities or scores (same length as y_true).
        class_names:  Display names for the two classes, e.g. ["Negative", "Positive"].
                      Defaults to the unique sorted values in y_true.
        colors:       Two colours for class 0 and class 1. Defaults to ["#4C72B0", "#DD8452"].
        title:        Plot title.
        ylabel:       Y-axis label.
        jitter:       Horizontal spread of the individual points (0 = no jitter).
        point_alpha:  Opacity of the scatter points.
        point_size:   Size of the scatter points.
        figsize:      Figure size (used only when ax is None).
        threshold:    Youden's J decision threshold. If provided, a dashed horizontal line
                      is drawn at this value to show the class separation boundary.
                      Points above threshold are classified as positive, below as negative.
        ax:           Existing matplotlib Axes to draw on. If None, a new figure is created.

    Returns:
        (fig, ax)  —  fig is None when an external ax was supplied.
    """
    y_true  = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)

    classes = np.sort(np.unique(y_true))
    if len(classes) != 2:
        raise ValueError(f"Expected exactly 2 classes, found {len(classes)}: {classes}")

    if class_names is None:
        class_names = [str(c) for c in classes]
    if len(class_names) != 2:
        raise ValueError("class_names must have exactly 2 entries.")

    if colors is None:
        colors = ["#4C72B0", "#DD8452"]

    # If threshold supplied, use it to derive predicted labels
    if threshold is not None:
        y_pred = (y_score >= threshold).astype(int)
        # Remap classes to 0/1 if needed
        class_map = {c: i for i, c in enumerate(classes)}
        y_true_mapped = np.array([class_map[c] for c in y_true])
    else:
        y_true_mapped = y_true

    # Split scores by true class
    groups = [y_score[y_true_mapped == i] for i in range(len(classes))]

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    positions = [1, 2]

    # ── Violin ────────────────────────────────────────────────────────────────
    parts = ax.violinplot(
        groups,
        positions=positions,
        widths=0.5,
        showmedians=False,
        showextrema=False,
    )
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("white")
        body.set_alpha(0.25)

    # ── Box (IQR + median) inside violin ──────────────────────────────────────
    for pos, grp, color in zip(positions, groups, colors):
        q25, median, q75 = np.percentile(grp, [25, 50, 75])
        ax.vlines(pos, q25, q75, color=color, linewidth=5, alpha=0.8, zorder=3)
        ax.scatter(pos, median, color="white", s=40, zorder=4, edgecolors=color, linewidths=1.5)

    # ── Jittered points ───────────────────────────────────────────────────────
    rng = np.random.default_rng(0)
    for pos, grp, color in zip(positions, groups, colors):
        x_jitter = pos + rng.uniform(-jitter, jitter, size=len(grp))
        ax.scatter(
            x_jitter, grp,
            color=color, alpha=point_alpha, s=point_size,
            zorder=2, edgecolors="none",
        )

    # ── Youden threshold line ─────────────────────────────────────────────────
    if threshold is not None:
        ax.axhline(
            y=threshold,
            color="gray",
            linestyle="--",
            linewidth=1.2,
            alpha=0.75,
            zorder=1,
            label=f"Threshold = {threshold:.2f}",
        )
        ax.legend(fontsize=9, frameon=False, loc="upper left")

    # ── Styling ───────────────────────────────────────────────────────────────
    ax.set_xticks(positions)
    ax.set_xticklabels(class_names, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlim(0.4, 2.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

    # Annotate n per class
    for pos, grp in zip(positions, groups):
        ax.text(
            pos, -0.08,
            f"n={len(grp)}",
            ha="center", va="top",
            fontsize=9, color="gray",
            transform=ax.get_xaxis_transform(),
        )

    return fig, ax


 
def _strip_prefix(name: str) -> str:
    """Remove sklearn ColumnTransformer prefix (e.g. 'num__', 'binary__', 'cat__')."""
    # Matches one word followed by __ at the start, e.g. "num__age" → "age"
    return re.sub(r"^\w+__", "", name)
 
 
def forest_plot_coefficients(
    feature_names=None,
    coefficients=None,
    std=None,
    cv_df: pd.DataFrame = None,
    col_fold: str = "fold",
    col_feature: str = "feature",
    col_coef: str = "coefficient",
    feature_types=None,
    mode: str = "odds_ratio",
    title: str = None,
    xlabel: str = None,
    strip_prefixes: bool = True,
    decimal_places: int = 2,
    color_positive: str = "#DD8452",
    color_negative: str = "#4C72B0",
    figsize: tuple = None,
    ax=None,
):
    """
    Horizontal forest plot of logistic regression coefficients or odds ratios.
 
    Accepts either:
      (a) pre-computed arrays via `feature_names` + `coefficients` (+ optional `std`), or
      (b) a CV results DataFrame via `cv_df` with one row per (fold, feature) —
          the function computes the mean and std of coefficients across folds.
 
    Changes vs previous version:
      - `xlabel`:         Override the x-axis label.
      - `strip_prefixes`: Remove sklearn ColumnTransformer prefixes
                          (e.g. "num__age" → "age", "binary__smoking" → "smoking").
      - `decimal_places`: Control rounding of the value annotations on the right.
      - `cv_df`:          DataFrame with per-fold coefficients. Columns controlled by
                          `col_fold`, `col_feature`, `col_coef`.
                          Mean ± std is shown as the annotation.
 
    Args:
        feature_names:  List/array of feature names  (required if cv_df is None).
        coefficients:   Raw log-scale coefficients   (required if cv_df is None).
        std:            Optional per-feature std on the coefficient scale.
        cv_df:          DataFrame with columns [fold, feature, coefficient].
                        If supplied, feature_names / coefficients / std are ignored.
        col_fold:       Column name for fold ID in cv_df.
        col_feature:    Column name for feature name in cv_df.
        col_coef:       Column name for coefficient value in cv_df.
        feature_types:  Optional list of strings per feature — "continuous" or "binary".
        mode:           "odds_ratio"  → exp(coef) on log x-axis, reference OR = 1.
                        "coefficient" → raw coefs, reference = 0.
        title:          Plot title. Auto-set from mode if None.
        xlabel:         X-axis label. Auto-set from mode if None.
        strip_prefixes: If True, remove transformer prefixes (e.g. "num__", "binary__").
        decimal_places: Decimal places for the value annotations (default 2).
        color_positive: Colour for OR > 1 / coef > 0.
        color_negative: Colour for OR < 1 / coef < 0.
        figsize:        Figure size. Auto-scales with n_features if None.
        ax:             Existing Axes to draw on. If None a new figure is created.
 
    Returns:
        (fig, ax)
 
    CV DataFrame example:
        cv_df columns: fold | feature          | coefficient
                       0    | num__age         | 0.42
                       0    | binary__smoking  | 0.81
                       1    | num__age         | 0.38
                       ...
    """
    # ── Input: CV DataFrame → compute mean & std ──────────────────────────────
    if cv_df is not None:
        agg = (
            cv_df.groupby(col_feature)[col_coef]
            .agg(["mean", "std"])
            .reset_index()
        )
        feature_names = agg[col_feature].to_numpy(dtype=object)
        coefficients  = agg["mean"].to_numpy(dtype=float)
        std           = agg["std"].to_numpy(dtype=float)
 
        # Align feature_types if provided (match by cleaned name)
        if feature_types is not None:
            # Assume feature_types is a dict {cleaned_name: type} or a matching list
            if isinstance(feature_types, dict):
                feature_types = np.array(
                    [feature_types.get(_strip_prefix(n), "continuous") for n in feature_names],
                    dtype=object,
                )
 
    feature_names = np.asarray(feature_names, dtype=object)
    coefficients  = np.asarray(coefficients, dtype=float)
    if std is not None:
        std = np.asarray(std, dtype=float)
    if feature_types is not None:
        feature_types = np.asarray(feature_types, dtype=object)
 
    # ── Strip prefixes ────────────────────────────────────────────────────────
    if strip_prefixes:
        feature_names = np.array([_strip_prefix(n) for n in feature_names], dtype=object)
 
    # ── Sort: strongest positive → strongest negative ─────────────────────────
    order         = np.argsort(coefficients)[::-1]
    feature_names = feature_names[order]
    coefficients  = coefficients[order]
    if std is not None:
        std = std[order]
    if feature_types is not None:
        feature_types = feature_types[order]
 
    n = len(feature_names)
    y = np.arange(n)
 
    # ── Convert to odds ratios if requested ──────────────────────────────────
    if mode == "odds_ratio":
        plot_values  = np.exp(coefficients)
        ref_line     = 1.0
        auto_xlabel  = "Odds Ratio (log scale)  ·  OR > 1 = increases risk"
        if std is not None:
            err_lo = plot_values - np.exp(coefficients - std)
            err_hi = np.exp(coefficients + std) - plot_values
        log_x = True
    else:
        plot_values  = coefficients
        ref_line     = 0.0
        auto_xlabel  = "Coefficient  ·  positive = increases log-odds"
        if std is not None:
            err_lo = err_hi = std
        log_x = False
 
    if title is None:
        title = "Odds Ratio Forest Plot" if mode == "odds_ratio" else "Coefficient Forest Plot"
    if xlabel is None:
        xlabel = auto_xlabel
 
    colors = [color_positive if v >= ref_line else color_negative for v in plot_values]
 
    if figsize is None:
        figsize = (9, max(4, n * 0.42))
 
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
 
    # ── Alternating row shading ───────────────────────────────────────────────
    for i in range(n):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color="gray", alpha=0.06, zorder=0)
 
    # ── Reference line ────────────────────────────────────────────────────────
    ax.axvline(ref_line, color="black", linewidth=0.9, linestyle="--", alpha=0.5, zorder=1)
 
    # ── Horizontal bars ───────────────────────────────────────────────────────
    if log_x:
        bar_left  = np.minimum(plot_values, ref_line)
        bar_width = np.abs(plot_values - ref_line)
        ax.barh(y, bar_width, left=bar_left, height=0.5,
                color=colors, alpha=0.30, zorder=2)
    else:
        ax.barh(y, plot_values, height=0.5, color=colors, alpha=0.30, zorder=2)
 
    # ── Error bars ────────────────────────────────────────────────────────────
    if std is not None:
        for yi, val, lo, hi, color in zip(y, plot_values, err_lo, err_hi, colors):
            ax.errorbar(
                val, yi,
                xerr=[[lo], [hi]],
                fmt="none",
                ecolor=color,
                elinewidth=1.5,
                capsize=3.5,
                capthick=1.5,
                alpha=0.9,
                zorder=3,
            )
 
    # ── Centre dots ───────────────────────────────────────────────────────────
    ax.scatter(plot_values, y, color=colors, s=50,
               edgecolors="white", linewidths=0.7, zorder=4)
 
    # ── Value annotations: "mean ± std" to the right ─────────────────────────
    fmt = f".{decimal_places}f"
    if log_x:
        x_annot = max(plot_values) * (1.3 if std is None else np.exp(max(err_hi)) * 1.4)
    else:
        x_annot = max(np.abs(plot_values)) * 1.15 + (max(err_hi) if std is not None else 0)
 
    for yi, val, color in zip(y, plot_values, colors):
        if std is not None:
            # std on the OR scale (mean of lo/hi error) for annotation
            i = list(y).index(yi)
            std_display = (err_lo[i] + err_hi[i]) / 2 if log_x else std[i]
            label = f"{val:{fmt}} ± {std_display:{fmt}}"
        else:
            sign = "+" if not log_x and val > 0 else ""
            label = f"{sign}{val:{fmt}}"
        ax.text(
            x_annot, yi, label,
            va="center", ha="left", fontsize=8, color=color,
        )
 
    # ── Y-axis labels ─────────────────────────────────────────────────────────
    if feature_types is not None:
        type_tag = {"continuous": " [cont]", "binary": " [bin]"}
        ylabels = [f"{nm}{type_tag.get(t, '')}"
                   for nm, t in zip(feature_names, feature_types)]
    else:
        ylabels = feature_names
 
    ax.set_yticks(y)
    ax.set_yticklabels(ylabels, fontsize=9)
    ax.invert_yaxis()
 
    # ── Log scale ticks ───────────────────────────────────────────────────────
    if log_x:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
 
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", linestyle="--", linewidth=0.4, alpha=0.45, zorder=0)
 
    plt.tight_layout()
    return fig, ax
 

 
def plot_roc_pr_curves(
    y_true,
    y_prob,
    cv_results: list[dict] = None,
    label: str = "Final model (test set)",
    title: str = "ROC & Precision-Recall Curves",
    color: str = "#DD8452",
    figsize: tuple = (13, 6),
):
    """
    Side-by-side ROC AUC and Precision-Recall curves for a final model,
    with optional per-fold CV curves overlaid.
 
    Args:
        y_true:      True binary labels for the held-out / test set.
        y_prob:      Predicted probabilities from the final model.
        cv_results:  Optional list of dicts, one per CV fold:
                       [{"y_true": ..., "y_prob": ...}, ...]
                     Plots faint per-fold lines + interpolated mean ± std band.
        label:       Legend label for the main test-set curve.
        title:       Overall figure title.
        color:       Colour for all curves.
        figsize:     Figure size.
 
    Returns:
        (fig, (ax_roc, ax_pr))
    """
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
 
    # ── CV fold curves ────────────────────────────────────────────────────────
    if cv_results:
        common_fpr      = np.linspace(0, 1, 200)
        common_recall   = np.linspace(0, 1, 200)
        tprs, precs     = [], []
        fold_aucs, fold_aps = [], []
 
        for fold in cv_results:
            # ROC
            fpr, tpr, _ = roc_curve(fold["y_true"], fold["y_prob"])
            fold_aucs.append(roc_auc_score(fold["y_true"], fold["y_prob"]))
            ax_roc.plot(fpr, tpr, color=color, alpha=0.15, linewidth=1, zorder=2)
            interp_tpr = np.interp(common_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
 
            # PR  (note: precision_recall_curve returns decreasing recall)
            prec, rec, _ = precision_recall_curve(fold["y_true"], fold["y_prob"])
            fold_aps.append(average_precision_score(fold["y_true"], fold["y_prob"]))
            ax_pr.plot(rec, prec, color=color, alpha=0.15, linewidth=1, zorder=2)
            interp_prec = np.interp(common_recall, rec[::-1], prec[::-1])
            precs.append(interp_prec)
 
        # Mean ROC
        mean_tpr, std_tpr = np.mean(tprs, axis=0), np.std(tprs, axis=0)
        mean_tpr[-1] = 1.0
        ax_roc.plot(common_fpr, mean_tpr, color=color, linewidth=2,
                    linestyle="--", alpha=0.75,
                    label=f"CV mean  AUC = {np.mean(fold_aucs):.3f} ± {np.std(fold_aucs):.3f}",
                    zorder=3)
        ax_roc.fill_between(common_fpr,
                            mean_tpr - std_tpr, mean_tpr + std_tpr,
                            color=color, alpha=0.10, zorder=1)
 
        # Mean PR
        mean_prec, std_prec = np.mean(precs, axis=0), np.std(precs, axis=0)
        ax_pr.plot(common_recall, mean_prec, color=color, linewidth=2,
                   linestyle="--", alpha=0.75,
                   label=f"CV mean  AP = {np.mean(fold_aps):.3f} ± {np.std(fold_aps):.3f}",
                   zorder=3)
        ax_pr.fill_between(common_recall,
                           np.clip(mean_prec - std_prec, 0, 1),
                           np.clip(mean_prec + std_prec, 0, 1),
                           color=color, alpha=0.10, zorder=1)
 
    # ── Final model — ROC ─────────────────────────────────────────────────────
    fpr, tpr, thresholds_roc = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    ax_roc.plot(fpr, tpr, color=color, linewidth=2.2, zorder=4,
                label=f"{label}  AUC = {auc:.3f}")
 
    best_idx = np.argmin(np.sqrt(fpr**2 + (1 - tpr)**2))
    ax_roc.scatter(fpr[best_idx], tpr[best_idx], marker="o", s=80,
                   color=color, edgecolors="white", linewidths=1.2, zorder=5,
                   label=f"Best threshold = {thresholds_roc[best_idx]:.3f}")
 
    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray",
                linewidth=1, alpha=0.6, label="Random (AUC = 0.500)")
    ax_roc.set_xlim(-0.02, 1.02)
    ax_roc.set_ylim(-0.02, 1.02)
    ax_roc.set_xlabel("False Positive Rate (1 − Specificity)", fontsize=11)
    ax_roc.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    ax_roc.set_title("ROC Curve", fontsize=12, fontweight="bold")
    ax_roc.legend(loc="lower right", fontsize=8.5, framealpha=0.8)
    ax_roc.spines[["top", "right"]].set_visible(False)
    ax_roc.grid(linestyle="--", linewidth=0.4, alpha=0.5)
 
    # ── Final model — PR ──────────────────────────────────────────────────────
    prec, rec, thresholds_pr = precision_recall_curve(y_true, y_prob)
    ap   = average_precision_score(y_true, y_prob)
    baseline = y_true.mean()          # prevalence = random classifier AP
 
    ax_pr.plot(rec, prec, color=color, linewidth=2.2, zorder=4,
               label=f"{label}  AP = {ap:.3f}")
 
    # Best threshold: maximise F1
    f1_scores = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
    best_pr   = np.argmax(f1_scores)
    ax_pr.scatter(rec[best_pr], prec[best_pr], marker="o", s=80,
                  color=color, edgecolors="white", linewidths=1.2, zorder=5,
                  label=f"Best F1 threshold = {thresholds_pr[best_pr]:.3f}")
 
    ax_pr.axhline(baseline, linestyle="--", color="gray",
                  linewidth=1, alpha=0.6, label=f"Random (AP = {baseline:.3f})")
    ax_pr.set_xlim(-0.02, 1.02)
    ax_pr.set_ylim(-0.02, 1.02)
    ax_pr.set_xlabel("Recall (Sensitivity)", fontsize=11)
    ax_pr.set_ylabel("Precision (PPV)", fontsize=11)
    ax_pr.set_title("Precision-Recall Curve", fontsize=12, fontweight="bold")
    ax_pr.legend(loc="lower left", fontsize=8.5, framealpha=0.8)
    ax_pr.spines[["top", "right"]].set_visible(False)
    ax_pr.grid(linestyle="--", linewidth=0.4, alpha=0.5)
 
    plt.tight_layout()
    return fig, (ax_roc, ax_pr)



 
def plot_bootstrap_ci_forest(
    df,
    col_metric: str = "metric",
    col_estimate: str = "estimate",
    col_lower: str = "ci_lower",
    col_upper: str = "ci_upper",
    col_model: str = None,
    model_name: str = "Model",
    title: str = "Bootstrap 95% CI — Model Performance",
    color: str = "#4C72B0",
    ref_line: float = None,
    figsize: tuple = (9, 5),
    ax=None,
):
    """
    Horizontal forest plot of bootstrap CIs for classification metrics.
 
    Args:
        df:            DataFrame with one row per metric (and optionally per model).
                       Expected columns: metric, estimate, ci_lower, ci_upper.
                       Pass col_model to compare multiple models.
        col_metric:    Column name for metric labels.
        col_estimate:  Column name for point estimates.
        col_lower:     Column name for CI lower bounds.
        col_upper:     Column name for CI upper bounds.
        col_model:     Column name for model labels (optional).
                       If provided, one line per model is drawn per metric.
        model_name:    Label used when col_model is not provided.
        title:         Plot title.
        color:         Dot/line colour for a single model. Ignored for multi-model.
        ref_line:      Optional vertical reference line (e.g. 0.5).
        figsize:       Figure size.
        ax:            Existing Axes; if None a new figure is created.
 
    Returns:
        (fig, ax)
 
    Example DataFrame:
        metric       estimate  ci_lower  ci_upper
        auroc        0.82      0.78      0.86
        auprc        0.74      0.69      0.79
        sensitivity  0.71      0.65      0.77
        specificity  0.88      0.83      0.92
        f1           0.76      0.70      0.81
    """

 
    METRIC_LABELS = {
        "auroc":       "AUROC",
        "auprc":       "AUPRC",
        "sensitivity": "Sensitivity",
        "specificity": "Specificity",
        "f1":          "F1 Score",
    }
    PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
 
    df = df.copy()
 
    # ── Resolve models ────────────────────────────────────────────────────────
    if col_model and col_model in df.columns:
        model_names = df[col_model].unique().tolist()
    else:
        col_model = "_model"
        df[col_model] = model_name
        model_names = [model_name]
 
    n_models = len(model_names)
    colors   = PALETTE[:n_models] if n_models > 1 else [color]
 
    # ── Metric order: reverse for bottom-to-top display ───────────────────────
    metric_order = df[col_metric].unique().tolist()[::-1]
    n_metrics    = len(metric_order)
    y_base       = np.arange(n_metrics)
    y_map        = {m: i for i, m in enumerate(metric_order)}
 
    offsets = np.linspace(-0.2, 0.2, n_models) if n_models > 1 else [0.0]
 
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
 
    # ── Alternating row shading ───────────────────────────────────────────────
    for i in range(n_metrics):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color="gray", alpha=0.06, zorder=0)
 
    # ── Reference line ────────────────────────────────────────────────────────
    if ref_line is not None:
        ax.axvline(ref_line, color="gray", linewidth=0.9,
                   linestyle="--", alpha=0.6, zorder=1)
 
    # ── Plot each model ───────────────────────────────────────────────────────
    for m_idx, (mname, mcolor, offset) in enumerate(zip(model_names, colors, offsets)):
        mdf = df[df[col_model] == mname]
        for row in mdf.itertuples(index=False):
            metric = getattr(row, col_metric)
            est    = getattr(row, col_estimate)
            lower  = getattr(row, col_lower)
            upper  = getattr(row, col_upper)
 
            if metric not in y_map:
                continue
            yi = y_map[metric] + offset
 
            # CI line + caps
            ax.plot([lower, upper], [yi, yi],
                    color=mcolor, linewidth=2, alpha=0.8, zorder=2)
            for x_cap in [lower, upper]:
                ax.plot([x_cap, x_cap], [yi - 0.07, yi + 0.07],
                        color=mcolor, linewidth=2, alpha=0.8, zorder=2)
 
            # Estimate dot
            ax.scatter(est, yi, color=mcolor, s=70,
                       edgecolors="white", linewidths=1, zorder=3,
                       label=mname if metric == mdf[col_metric].iloc[0] else "_nolegend_")
 
            # Annotation
            ax.text(
                1.01, yi,
                f"{est:.3f}  ({lower:.3f} – {upper:.3f})",
                va="center", ha="left", fontsize=8.5,
                color=mcolor,
                transform=ax.get_yaxis_transform(),
            )
 
    # ── Axes & styling ────────────────────────────────────────────────────────
    ylabels = [METRIC_LABELS.get(m, m) for m in metric_order]
    ax.set_yticks(y_base)
    ax.set_yticklabels(ylabels, fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Metric value", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", linestyle="--", linewidth=0.4, alpha=0.45, zorder=0)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
 
    if n_models > 1:
        ax.legend(loc="upper left", fontsize=9, framealpha=0.8)
 
    plt.tight_layout()
    return fig, ax

 
# ── Plot function ─────────────────────────────────────────────────────────────
 
 
 
# ── Plot function ─────────────────────────────────────────────────────────────
 
def plot_feature_significance(
    results: pd.DataFrame,
    top_n: int = None,
    title: str = "Univariate Feature Significance",
    figsize: tuple = None,
):
    """
    Two-panel plot: -log10(p_value) and mutual information per feature.
 
    Args:
        results:  Output of feature_significance().
        top_n:    Show only top N features by p-value. None = all.
        title:    Plot title.
        figsize:  Figure size.
 
    Returns:
        (fig, (ax_p, ax_mi))
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
 
    df = results.copy()
 
    def parse_p(p):
        if p == "<0.001":
            return 0.0005
        try:
            return float(p)
        except Exception:
            return np.nan
 
    df["_p_f"]      = df["p_value"].apply(parse_p)
    df["_neg_logp"] = -np.log10(df["_p_f"].clip(1e-10))
    df = df.sort_values("_neg_logp", ascending=True)
 
    if top_n:
        df = df.tail(top_n)
 
    n = len(df)
    if figsize is None:
        figsize = (13, max(4, n * 0.38))
 
    fig, (ax_p, ax_mi) = plt.subplots(1, 2, figsize=figsize,
                                       gridspec_kw={"width_ratios": [2, 1]})
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
 
    ypos   = np.arange(n)
    colors = ["#DD8452" if sig else "#4C72B0" for sig in df["significant"]]
 
    for ax in (ax_p, ax_mi):
        for i in range(n):
            if i % 2 == 0:
                ax.axhspan(i - 0.5, i + 0.5, color="gray", alpha=0.06, zorder=0)
 
    # ── Left: -log10(p) ───────────────────────────────────────────────────────
    ax_p.barh(ypos, df["_neg_logp"], height=0.5, color=colors, alpha=0.35, zorder=2)
    ax_p.scatter(df["_neg_logp"], ypos, color=colors, s=50,
                 edgecolors="white", linewidths=0.7, zorder=3)
    ax_p.axvline(-np.log10(0.05), color="red", linewidth=1,
                 linestyle="--", alpha=0.7, label="p = 0.05")
    ax_p.set_yticks(ypos)
    ax_p.set_yticklabels(df["feature"], fontsize=9)
    ax_p.set_xlabel("−log₁₀(p value)", fontsize=10)
    ax_p.set_title("Univariate p-value\n(Mann-Whitney U / Fisher's exact)", fontsize=11, fontweight="bold")
    ax_p.legend(fontsize=9, loc="lower right")
    ax_p.spines[["top", "right"]].set_visible(False)
    ax_p.grid(axis="x", linestyle="--", linewidth=0.4, alpha=0.4)
 
    # Annotate p + test type
    x_ann = df["_neg_logp"].max() * 1.02
    for yi, (_, row) in zip(ypos, df.iterrows()):
        ax_p.text(x_ann, yi, f"p={row['p_value']}",
                  va="center", fontsize=7.5, color="gray")
 
    # ── Right: mutual information ─────────────────────────────────────────────
    ax_mi.barh(ypos, df["mutual_info"], height=0.5, color=colors, alpha=0.35, zorder=2)
    ax_mi.scatter(df["mutual_info"], ypos, color=colors, s=50,
                  edgecolors="white", linewidths=0.7, zorder=3)
    ax_mi.set_yticks(ypos)
    ax_mi.set_yticklabels([], fontsize=9)
    ax_mi.set_xlabel("Mutual Information", fontsize=10)
    ax_mi.set_title("Mutual Information", fontsize=11, fontweight="bold")
    ax_mi.spines[["top", "right"]].set_visible(False)
    ax_mi.grid(axis="x", linestyle="--", linewidth=0.4, alpha=0.4)
 
    handles = [Patch(color="#DD8452", alpha=0.7, label=f"Significant (p < 0.05)"),
               Patch(color="#4C72B0", alpha=0.7, label="Not significant")]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               fontsize=9, framealpha=0.8, bbox_to_anchor=(0.5, -0.04))
 
    plt.tight_layout()
    return fig, (ax_p, ax_mi)
 

 
def plot_rf_heatmap(
    df: pd.DataFrame,
    metric: str = "auroc",
    col_n_estimators: str = "param_model__n_estimators",
    col_max_depth: str = "param_model__max_depth",
    col_max_features: str = "param_model__max_features",
    col_min_samples_split: str = "param_model__min_samples_split",
    col_min_samples_leaf: str = "param_model__min_samples_leaf",
    figsize: tuple = None,
    cmap: str = "YlOrRd",
):
    """
    Heatmap grid for Random Forest hyperparameter search.
 
    Layout:
      rows of panels    → n_estimators
      columns of panels → max_features
      heatmap y-axis    → max_depth
      heatmap x-axis    → min_samples_split
      cell value        → mean metric, averaged over min_samples_leaf
 
    Each cell is annotated with the numeric score so you can read exact values.
 
    Args:
        df:      cv_results_ DataFrame from GridSearchCV.
        metric:  Metric suffix, e.g. "auroc" → uses mean_test_auroc.
        col_*:   Column names (default = sklearn pipeline naming).
        figsize: Figure size (auto if None).
        cmap:    Matplotlib colormap name.
 
    Returns:
        (fig, axes)
    """
    mean_col = f"mean_test_{metric}"
    if mean_col not in df.columns:
        raise ValueError(f"Column '{mean_col}' not found in DataFrame.")
 
    df = df.copy()
 
    # Normalise max_depth: None → "None", others → int string
    df["_depth"] = df[col_max_depth].apply(
        lambda x: "None" if x is None or (isinstance(x, float) and np.isnan(x))
                  else str(int(float(x)))
    )
 
    n_est_vals  = sorted(df[col_n_estimators].astype(float).unique())
    mf_vals     = sorted(df[col_max_features].astype(str).unique())
    mss_vals    = sorted(df[col_min_samples_split].astype(float).unique())
    depth_order = sorted(
        [v for v in df["_depth"].unique() if v != "None"],
        key=int
    ) + (["None"] if "None" in df["_depth"].unique() else [])
 
    n_rows = len(n_est_vals)
    n_cols = len(mf_vals)
 
    if figsize is None:
        figsize = (4.5 * n_cols + 1, 3.5 * n_rows + 1)
 
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1: axes = axes[np.newaxis, :]
    if n_cols == 1: axes = axes[:, np.newaxis]
 
    # Global colour scale across all panels
    vmin = df[mean_col].min()
    vmax = df[mean_col].max()
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm   = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
 
    for row_idx, ne in enumerate(n_est_vals):
        for col_idx, mf in enumerate(mf_vals):
            ax = axes[row_idx, col_idx]
 
            sub = df[
                (df[col_n_estimators].astype(float) == ne) &
                (df[col_max_features].astype(str) == mf)
            ]
 
            # Best score across min_samples_split → pivot: depth (y) × min_samples_leaf (x)
            pivot = (
                sub.groupby(["_depth", col_min_samples_leaf])[mean_col]
                .max()
                .reset_index()
                .pivot(index="_depth", columns=col_min_samples_leaf, values=mean_col)
                .reindex(index=depth_order)
            )
 
            # Draw heatmap manually for full control
            data   = pivot.values                  # shape: (n_depths, n_mss)
            n_y, n_x = data.shape
 
            for yi in range(n_y):
                for xi in range(n_x):
                    val   = data[yi, xi]
                    color = sm.to_rgba(val)
                    ax.add_patch(plt.Rectangle(
                        (xi - 0.5, yi - 0.5), 1, 1,
                        facecolor=color, edgecolor="white", linewidth=1.5,
                    ))
                    # Text color: white on dark cells, black on light
                    brightness = 0.299*color[0] + 0.587*color[1] + 0.114*color[2]
                    txt_color  = "white" if brightness < 0.55 else "black"
                    ax.text(xi, yi, f"{val:.3f}",
                            ha="center", va="center", fontsize=8,
                            fontweight="bold", color=txt_color)
 
            # Mark best cell
            best_flat = np.nanargmax(data)
            best_yi, best_xi = np.unravel_index(best_flat, data.shape)
            ax.add_patch(plt.Rectangle(
                (best_xi - 0.5, best_yi - 0.5), 1, 1,
                facecolor="none", edgecolor="gold",
                linewidth=2.5, zorder=5,
            ))
 
            ax.set_xlim(-0.5, n_x - 0.5)
            ax.set_ylim(-0.5, n_y - 0.5)
            ax.set_xticks(range(n_x))
            ax.set_xticklabels([int(v) for v in pivot.columns], fontsize=9)
            ax.set_yticks(range(n_y))
            ax.set_yticklabels(pivot.index, fontsize=9)
            ax.invert_yaxis()
 
            if row_idx == 0:
                ax.set_title(f"max_features = {mf}", fontsize=11,
                             fontweight="bold", pad=8)
            if col_idx == 0:
                ax.set_ylabel(f"n_estimators = {int(ne)}\n\nmax_depth",
                              fontsize=9, labelpad=8)
            if row_idx == n_rows - 1:
                ax.set_xlabel("min_samples_leaf", fontsize=9)
 
            ax.tick_params(length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
 
    # ── Colourbar in its own axes, clearly outside the matrix ────────────────
    plt.tight_layout(rect=[0, 0, 0.88, 1])   # leave 12% on the right for the bar
    cbar_ax = fig.add_axes([0.91, 0.15, 0.025, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="vertical")
    cbar.set_label(metric.upper(), fontsize=11, labelpad=10)
    cbar.ax.tick_params(labelsize=9)
 
    fig.suptitle(
        f"Random Forest Hyperparameter Search — {metric.upper()}\n"
        f"(cell = best score across min_samples_split  ·  gold border = overall best)\n",
        fontsize=12, fontweight="bold", y=1.04,
    )
    return fig, axes

from collections import defaultdict
from sklearn.metrics import roc_curve, roc_auc_score, f1_score
 

def select_threshold_from_cv(
    folds: list[dict],
    method: str = "youden",
    plot: bool = True,
    figsize: tuple = (14, 5),
    title: str = None,
    title_left: str = None,
    title_right: str = None,
):
    """
    Choose a decision threshold from already-computed CV out-of-fold predictions
    by averaging each patient's probabilities across folds.
 
    Steps:
      1. Average each patient's out-of-fold probabilities across all folds/repeats.
      2. Build one ROC curve from those patient-level averaged probabilities.
      3. Pick and freeze one threshold from that curve.
 
    Args:
        folds:    List of dicts, one per fold. Each dict must contain:
                    "indices"  — array of patient indices (position in original dataset)
                    "y_true"   — true labels for those patients
                    "y_prob"   — predicted probabilities for those patients
                  Optionally: "fold" for the fold number (used only for display).
        method:      Threshold criterion:
                       "youden" — maximises Youden's J = TPR − FPR
                                  (balances sensitivity and specificity)
                       "f1"     — maximises F1 score
                                  (better for imbalanced classes)
        plot:        Whether to plot the ROC + criterion curves.
        figsize:     Figure size.
        title:       Main figure title. Auto-generated if None.
        title_left:  Left panel title. Auto-generated if None.
        title_right: Right panel title. Auto-generated if None.
 
    Returns:
        dict with keys:
          "threshold"    — the frozen threshold value
          "method"       — criterion used
          "auc"          — AUC of the patient-averaged ROC curve
          "y_true"       — true labels, one per patient
          "y_prob_avg"   — averaged probabilities, one per patient
          "fig"          — matplotlib Figure (None if plot=False)
 
    Example folds structure:
        folds = []
        for fold_num, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            model.fit(X[train_idx], y[train_idx])
            folds.append({
                "fold":    fold_num,
                "indices": val_idx,
                "y_true":  y[val_idx],
                "y_prob":  model.predict_proba(X[val_idx])[:, 1],
            })
    """
    # ── Average probabilities per patient ─────────────────────────────────────
    prob_accum  = defaultdict(list)
    label_store = {}
 
    for fold in folds:
        indices = np.asarray(fold["indices"])
        y_true  = np.asarray(fold["y_true"])
        y_prob  = np.asarray(fold["y_prob"])
 
        for idx, label, prob in zip(indices, y_true, y_prob):
            prob_accum[idx].append(prob)
            label_store[idx] = int(label)   # label is fixed per patient
 
    patient_ids  = np.array(sorted(prob_accum.keys()))
    y_prob_avg   = np.array([np.mean(prob_accum[i]) for i in patient_ids])
    y_true_all   = np.array([label_store[i]          for i in patient_ids])
 
    n_folds   = len(folds)
    n_repeats = int(np.round(np.mean([len(v) for v in prob_accum.values()])))
 
    # ── Build one ROC curve ───────────────────────────────────────────────────
    fpr, tpr, thresholds = roc_curve(y_true_all, y_prob_avg)
    auc = roc_auc_score(y_true_all, y_prob_avg)
 
    # ── Pick threshold ────────────────────────────────────────────────────────
    if method == "youden":
        criterion_vals  = tpr - fpr
        criterion_label = "Youden's J  (TPR − FPR)"
        legend_loc_crit = "lower left"
    elif method == "f1":
        criterion_vals  = np.array([
            f1_score(y_true_all, (y_prob_avg >= t).astype(int), zero_division=0)
            for t in thresholds
        ])
        criterion_label = "F1 Score"
        legend_loc_crit = "upper right"
    else:
        raise ValueError(f"method must be 'youden' or 'f1', got '{method}'")
 
    best_idx       = np.argmax(criterion_vals)
    best_threshold = float(thresholds[best_idx])
 
    print(f"Patients: {len(patient_ids)}  |  Folds: {n_folds}  |  "
          f"Avg repeats per patient: {n_repeats}")
    print(f"Frozen threshold ({method}): {best_threshold:.4f}  |  AUC: {auc:.4f}")
 
    # ── Plot ──────────────────────────────────────────────────────────────────
    fig = None
    if plot:
        fig, (ax_roc, ax_crit) = plt.subplots(1, 2, figsize=figsize)
 
        main_title = title if title is not None else (
            f"Threshold Selection — Patient-averaged OOF Probabilities  "
            f"({n_folds} folds  ·  method: {method})"
        )
        fig.suptitle(main_title, fontsize=13, fontweight="bold", y=1.01)
 
        # Left: ROC curve
        ax_roc.plot(fpr, tpr, color="#4C72B0", linewidth=2.2,
                    label=f"Patient-avg ROC  (AUC = {auc:.3f})")
        ax_roc.scatter(fpr[best_idx], tpr[best_idx],
                       color="red", s=90, zorder=5,
                       edgecolors="white", linewidths=1.2,
                       label=f"Chosen threshold = {best_threshold:.3f}\n"
                             f"TPR = {tpr[best_idx]:.3f}  |  FPR = {fpr[best_idx]:.3f}")
        ax_roc.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, alpha=0.5)
        ax_roc.set_xlabel("False Positive Rate", fontsize=11)
        ax_roc.set_ylabel("True Positive Rate", fontsize=11)
        ax_roc.set_title(
            title_left if title_left is not None else "Patient-averaged ROC Curve",
            fontsize=11, fontweight="bold",
        )
        ax_roc.legend(fontsize=9, loc="lower right")
        ax_roc.set_xlim(-0.02, 1.02)
        ax_roc.set_ylim(-0.02, 1.02)
        ax_roc.spines[["top", "right"]].set_visible(False)
        ax_roc.grid(linestyle="--", linewidth=0.4, alpha=0.5)
 
        # Right: criterion vs threshold
        ax_crit.plot(thresholds, criterion_vals, color="#DD8452", linewidth=2.2)
        ax_crit.axvline(best_threshold, color="red", linewidth=1.4, linestyle="--",
                        label=f"Best = {best_threshold:.3f}  "
                              f"({criterion_label[:3]}... = {criterion_vals[best_idx]:.3f})")
        ax_crit.scatter(best_threshold, criterion_vals[best_idx],
                        color="red", s=90, zorder=5,
                        edgecolors="white", linewidths=1.2)
        ax_crit.set_xlabel("Threshold", fontsize=11)
        ax_crit.set_ylabel(criterion_label, fontsize=11)
        ax_crit.set_title(
            title_right if title_right is not None else f"{criterion_label} vs Threshold",
            fontsize=11, fontweight="bold",
        )
        ax_crit.legend(fontsize=9, loc=legend_loc_crit)
        ax_crit.spines[["top", "right"]].set_visible(False)
        ax_crit.grid(linestyle="--", linewidth=0.4, alpha=0.5)
 
        for ax in (ax_roc, ax_crit):
            ax.text(0.98, 0.05,
                    f"Frozen threshold: {best_threshold:.4f}",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                              edgecolor="gray", alpha=0.85))
        plt.tight_layout()
 
    return {
        "threshold":  best_threshold,
        "method":     method,
        "auc":        auc,
        "y_true":     y_true_all,
        "y_prob_avg": y_prob_avg,
        "fig":        fig,
    }
 

  
def plot_svm_hyperparam_search(
    df: pd.DataFrame,
    metrics: list[str] = ["auroc", "auprc"],
    col_C: str = "param_model__C",
    col_kernel: str = "param_model__kernel",
    col_gamma: str = "param_model__gamma",
    col_class_weight: str = "param_model__class_weight",
    kernels: list[str] = ["linear", "rbf"],
    figsize: tuple = (14, 8),
    alpha_band: float = 0.12,
):
    """
    Plot hyperparameter search results for SVM (sklearn GridSearchCV cv_results_).
 
    Layout: rows = metrics, cols = kernels (linear / rbf).
 
    - Linear kernel: one line per class_weight value.
    - RBF kernel:    one line per gamma value;
                     line style encodes class_weight
                     (solid = None/default, dashed = "balanced").
 
    Args:
        df:               cv_results_ DataFrame from GridSearchCV.
        metrics:          Metric suffixes matching mean_test_<metric> / std_test_<metric>.
        col_C:            Column name for C.
        col_kernel:       Column name for kernel.
        col_gamma:        Column name for gamma.
        col_class_weight: Column name for class_weight.
        kernels:          Kernel types to plot and their column order.
        figsize:          Figure size.
        alpha_band:       Opacity of ±std shading.
 
    Returns:
        (fig, axes)  — axes shape: (n_metrics, n_kernels)
    """
    n_rows = len(metrics)
    n_cols = len(kernels)
    cmap   = plt.get_cmap("tab10")
 
    # Linestyle mapping for class_weight
    cw_styles = {None: "-", "none": "-", "balanced": "--"}
 
    def cw_style(val):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "-", "None"
        return cw_styles.get(str(val).lower(), "-"), str(val)
 
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize,
                              sharey="row", sharex="col")
    if n_rows == 1: axes = axes[np.newaxis, :]
    if n_cols == 1: axes = axes[:, np.newaxis]
 
    for col_idx, kernel in enumerate(kernels):
        sub = df[df[col_kernel].str.lower() == kernel.lower()].copy()
        sub[col_C] = sub[col_C].astype(float)
 
        # ── Determine line grouping ───────────────────────────────────────────
        if kernel.lower() == "linear":
            # Group by class_weight only
            cw_vals = sorted(
                sub[col_class_weight].fillna("None").unique(),
                key=lambda x: str(x)
            )
            groups = []
            for cw in cw_vals:
                mask = sub[col_class_weight].fillna("None").astype(str) == str(cw)
                groups.append({
                    "data":    sub[mask].sort_values(col_C),
                    "color":   cmap(cw_vals.index(cw) % 10),
                    "ls":      "-",
                    "label":   f"class_weight={cw}",
                })
        else:
            # Group by gamma; linestyle = class_weight
            gamma_vals = sub[col_gamma].astype(str).unique()
            # Sort: string gammas ('scale', 'auto') first, then numeric ascending
            def gamma_sort_key(g):
                try:    return (1, float(g))
                except: return (0, g)
            gamma_vals = sorted(gamma_vals, key=gamma_sort_key)
 
            cw_vals = sorted(
                sub[col_class_weight].fillna("None").unique(),
                key=lambda x: str(x)
            )
            groups = []
            for gi, gv in enumerate(gamma_vals):
                for cw in cw_vals:
                    gmask = sub[col_gamma].astype(str) == str(gv)
                    cmask = sub[col_class_weight].fillna("None").astype(str) == str(cw)
                    grp_data = sub[gmask & cmask].sort_values(col_C)
                    if grp_data.empty:
                        continue
                    ls, cw_label = cw_style(cw if cw != "None" else None)
                    groups.append({
                        "data":  grp_data,
                        "color": cmap(gi % 10),
                        "ls":    ls,
                        "gamma": gv,
                        "cw":    cw_label,
                        "label": f"γ={gv}  cw={cw_label}",
                    })
 
        for row_idx, metric in enumerate(metrics):
            ax       = axes[row_idx, col_idx]
            mean_col = f"mean_test_{metric}"
            std_col  = f"std_test_{metric}"
 
            if mean_col not in df.columns:
                ax.text(0.5, 0.5, f"'{mean_col}' not found",
                        ha="center", va="center",
                        transform=ax.transAxes, color="red")
                continue
 
            for g in groups:
                grp = g["data"]
                if grp.empty:
                    continue
                ax.plot(grp[col_C], grp[mean_col],
                        marker="o", markersize=4, linewidth=1.6,
                        color=g["color"], linestyle=g["ls"],
                        label=g["label"])
                if std_col in df.columns:
                    ax.fill_between(
                        grp[col_C],
                        grp[mean_col] - grp[std_col],
                        grp[mean_col] + grp[std_col],
                        alpha=alpha_band, color=g["color"],
                    )
 
            # Gold star: overall best C in this kernel × metric panel
            best_row = sub.loc[sub[mean_col].idxmax()]
            ax.scatter(
                float(best_row[col_C]), best_row[mean_col],
                marker="*", s=200, color="gold", edgecolors="black",
                linewidths=0.7, zorder=6,
                label=f"best C={float(best_row[col_C]):.3g}",
            )
 
            ax.set_xscale("log")
            ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.5)
            ax.spines[["top", "right"]].set_visible(False)
 
            if row_idx == 0:
                ax.set_title(f"kernel = {kernel}", fontsize=12,
                             fontweight="bold", pad=8)
            if col_idx == 0:
                ax.set_ylabel(metric.upper(), fontsize=11)
            if row_idx == n_rows - 1:
                ax.set_xlabel("C  (log scale)", fontsize=10)
 
            # Legend: for rbf, put gamma legend outside to avoid clutter
            if kernel.lower() == "rbf":
                ax.legend(fontsize=6.5, loc="lower right", framealpha=0.75,
                          ncol=2, title="gamma  ·  cw", title_fontsize=7)
            else:
                ax.legend(fontsize=8.5, loc="lower right", framealpha=0.75)
 
    # ── Shared linestyle legend for RBF class_weight ──────────────────────────
    if "rbf" in [k.lower() for k in kernels]:
        ls_handles = [
            mlines.Line2D([], [], color="gray", linestyle="-",
                          linewidth=1.5, label="class_weight = None"),
            mlines.Line2D([], [], color="gray", linestyle="--",
                          linewidth=1.5, label="class_weight = balanced"),
        ]
        fig.legend(handles=ls_handles, loc="lower center",
                   ncol=2, fontsize=9, framealpha=0.85,
                   bbox_to_anchor=(0.5, -0.04))
 
    fig.suptitle("SVM Hyperparameter Search", fontsize=14,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    return fig, axes


 
# Default colour cycle for comparison models
_DEFAULT_COLORS = [
    "#DD8452", "#4C72B0", "#55A868", "#C44E52",
    "#8172B2", "#937860", "#DA8BC3", "#8C8C8C",
]
 
 
def _plot_single_roc(ax, y_true, y_prob, model_name, color, threshold, linestyle="-"):
    """Plot one ROC curve + optional threshold marker onto ax."""
    auroc          = roc_auc_score(y_true, y_prob)
    fpr, tpr, thr  = roc_curve(y_true, y_prob)
    ax.plot(fpr, tpr, color=color, linewidth=2.2, linestyle=linestyle,
            label=f"{model_name}  (AUROC = {auroc:.3f})")
    if threshold is not None:
        idx   = np.argmin(np.abs(thr - threshold))
        t_fpr, t_tpr = fpr[idx], tpr[idx]
        ax.scatter(t_fpr, t_tpr, color=color, s=90, zorder=5,
                   edgecolors="white", linewidths=1.2)
        ax.annotate(
            f"{model_name}\nt={threshold:.2f}\nSens={t_tpr:.2f}  Spec={1-t_fpr:.2f}",
            xy=(t_fpr, t_tpr),
            xytext=(t_fpr + 0.06, t_tpr - 0.13),
            fontsize=7.5, color=color,
            arrowprops=dict(arrowstyle="->", color=color, lw=0.8),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=color, alpha=0.85),
        )
    return auroc
 
 
def _plot_single_pr(ax, y_true, y_prob, model_name, color, threshold, linestyle="-"):
    """Plot one PR curve + optional threshold marker onto ax."""
    auprc             = average_precision_score(y_true, y_prob)
    prec, rec, thr_pr = precision_recall_curve(y_true, y_prob)
    ax.plot(rec, prec, color=color, linewidth=2.2, linestyle=linestyle,
            label=f"{model_name}  (AUPRC = {auprc:.3f})")
    if threshold is not None:
        idx    = np.argmin(np.abs(thr_pr - threshold))
        t_prec, t_rec = prec[idx], rec[idx]
        y_pred = (y_prob >= threshold).astype(int)
        t_f1   = f1_score(y_true, y_pred, zero_division=0)
        ax.scatter(t_rec, t_prec, color=color, s=90, zorder=5,
                   edgecolors="white", linewidths=1.2)
        ax.annotate(
            f"{model_name}\nt={threshold:.2f}\nPrec={t_prec:.2f}  Rec={t_rec:.2f}\nF1={t_f1:.2f}",
            xy=(t_rec, t_prec),
            xytext=(max(t_rec - 0.32, 0.02), t_prec - 0.16),
            fontsize=7.5, color=color,
            arrowprops=dict(arrowstyle="->", color=color, lw=0.8),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=color, alpha=0.85),
        )
    return auprc
 
 
def plot_external_validation_curves(
    y_true,
    y_prob,
    threshold: float = None,
    model_name: str = "Model",
    color: str = None,
    comparisons: list[dict] = None,
    title: str = "External Validation — ROC & Precision-Recall Curves",
    title_left: str = "ROC Curve",
    title_right: str = "Precision-Recall Curve",
    figsize: tuple = (13, 6),
):
    """
    ROC and Precision-Recall curves for an external validation dataset,
    with optional overlay of additional models for comparison.
 
    Args:
        y_true:      True binary labels.
        y_prob:      Predicted probabilities from the primary model.
        threshold:   Optional frozen threshold for the primary model.
                     Marks the operating point and annotates Sens/Spec/Prec/F1.
        model_name:  Display name for the primary model.
        color:       Curve colour for the primary model. Defaults to first
                     colour in the cycle.
        comparisons: Optional list of dicts for additional models. Each dict:
                       {
                         "model_name": str,          # required
                         "y_prob":     array-like,   # required
                         "threshold":  float,         # optional
                         "color":      str,           # optional
                         "linestyle":  str,           # optional, e.g. "--"
                       }
        title:       Main figure title.
        title_left:  Left panel (ROC) title.
        title_right: Right panel (PR) title.
        figsize:     Figure size.
 
    Returns:
        (fig, (ax_roc, ax_pr))
 
    Example with comparisons:
        fig, axes = plot_external_validation_curves(
            y_true, y_prob_lr,
            model_name="Logistic Regression",
            threshold=0.42,
            comparisons=[
                {"model_name": "SVM",            "y_prob": y_prob_svm, "threshold": 0.50},
                {"model_name": "Random Forest",  "y_prob": y_prob_rf,  "linestyle": "--"},
            ],
            title="External Validation — Model Comparison",
        )
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    comparisons = comparisons or []
 
    # Assign colours
    primary_color = color if color is not None else _DEFAULT_COLORS[0]
    for i, m in enumerate(comparisons):
        if "color" not in m:
            m["color"] = _DEFAULT_COLORS[(i + 1) % len(_DEFAULT_COLORS)]
 
    prevalence = y_true.mean()
 
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
 
    # ── Primary model ─────────────────────────────────────────────────────────
    _plot_single_roc(ax_roc, y_true, y_prob, model_name, primary_color, threshold)
    _plot_single_pr(ax_pr,  y_true, y_prob, model_name, primary_color, threshold)
 
    # ── Comparison models ─────────────────────────────────────────────────────
    for m in comparisons:
        yp  = np.asarray(m["y_prob"])
        _plot_single_roc(
            ax_roc, y_true, yp,
            m["model_name"], m["color"],
            m.get("threshold"), m.get("linestyle", "-"),
        )
        _plot_single_pr(
            ax_pr, y_true, yp,
            m["model_name"], m["color"],
            m.get("threshold"), m.get("linestyle", "-"),
        )
 
    # ── ROC decorations ───────────────────────────────────────────────────────
    ax_roc.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, alpha=0.5,
                label="Random classifier")
    ax_roc.set_xlabel("False Positive Rate  (1 − Specificity)", fontsize=11)
    ax_roc.set_ylabel("True Positive Rate  (Sensitivity)", fontsize=11)
    ax_roc.set_title(title_left, fontsize=12, fontweight="bold")
    ax_roc.legend(fontsize=8.5, loc="lower right", framealpha=0.85)
    ax_roc.set_xlim(-0.02, 1.02)
    ax_roc.set_ylim(-0.02, 1.02)
    ax_roc.spines[["top", "right"]].set_visible(False)
    ax_roc.grid(linestyle="--", linewidth=0.4, alpha=0.5)
 
    # ── PR decorations ────────────────────────────────────────────────────────
    ax_pr.axhline(prevalence, color="gray", linewidth=1, linestyle="--", alpha=0.6,
                  label=f"No-skill baseline (prevalence = {prevalence:.2f})")
    ax_pr.set_xlabel("Recall  (Sensitivity)", fontsize=11)
    ax_pr.set_ylabel("Precision  (PPV)", fontsize=11)
    ax_pr.set_title(title_right, fontsize=12, fontweight="bold")
    ax_pr.legend(fontsize=8.5, loc="lower left", framealpha=0.85)
    ax_pr.set_xlim(-0.02, 1.02)
    ax_pr.set_ylim(-0.02, 1.05)
    ax_pr.spines[["top", "right"]].set_visible(False)
    ax_pr.grid(linestyle="--", linewidth=0.4, alpha=0.5)
 
    plt.tight_layout()
    return fig, (ax_roc, ax_pr)
 



 
_DEFAULT_COLORS = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B2", "#937860", "#DA8BC3", "#8C8C8C",
]
 
_METRIC_LABELS = {
    "auroc":       "AUROC",
    "auprc":       "AUPRC",
    "sensitivity": "Sensitivity",
    "specificity": "Specificity",
    "f1":          "F1 Score",
    "f1score":     "F1 Score",
    "f1_score":    "F1 Score",
}
 
 
def plot_cv_metric_comparison(
    models: list[dict],
    metrics: list[str] = None,
    col_fold: str = "fold",
    colors: list[str] = None,
    title: str = "Cross-Validation Metric Comparison",
    jitter: float = 0.06,
    point_alpha: float = 0.6,
    point_size: float = 20,
    show_median_label: bool = True,
    figsize: tuple = None,
):
    """
    Side-by-side boxplots comparing cross-validation metrics across models.
 
    One panel per metric; within each panel one box per model.
    Individual fold values are overlaid as jittered points.
 
    Args:
        models:             List of dicts, one per model:
                              {
                                "model_name": str,       # display name
                                "data":       DataFrame, # per-fold metrics
                              }
                            Each DataFrame must have a fold column and one
                            column per metric (auroc, auprc, sensitivity,
                            specificity, f1, etc.).
        metrics:            List of metric column names to plot.
                            If None, auto-detects from known metric names
                            in the first model's DataFrame.
        col_fold:           Column name for the fold identifier.
        colors:             List of colours, one per model. Auto-assigned if None.
        title:              Figure-level title.
        jitter:             Horizontal jitter width for fold points.
        point_alpha:        Opacity of fold-level scatter points.
        point_size:         Size of fold-level scatter points.
        show_median_label:  If True, annotates the median value above each box.
        figsize:            Figure size. Auto-scaled if None.
 
    Returns:
        (fig, axes)
 
    Example:
        fig, axes = plot_cv_metric_comparison(
            models=[
                {"model_name": "Logistic Regression", "data": df_lr},
                {"model_name": "SVM",                 "data": df_svm},
                {"model_name": "Random Forest",        "data": df_rf},
            ],
            metrics=["auroc", "auprc", "sensitivity", "specificity", "f1"],
        )
    """
    model_names = [m["model_name"] for m in models]
    n_models    = len(models)
 
    # ── Auto-detect metrics from first model's DataFrame ─────────────────────
    if metrics is None:
        known   = set(_METRIC_LABELS.keys())
        metrics = [c for c in models[0]["data"].columns
                   if c.lower() in known]
        if not metrics:
            raise ValueError(
                "No metric columns detected. Pass metrics= explicitly, e.g. "
                "metrics=['auroc', 'auprc', 'sensitivity', 'specificity', 'f1']"
            )
 
    n_metrics = len(metrics)
 
    # ── Colours ───────────────────────────────────────────────────────────────
    if colors is None:
        colors = [_DEFAULT_COLORS[i % len(_DEFAULT_COLORS)] for i in range(n_models)]
    color_map = dict(zip(model_names, colors))
 
    # ── Figure layout ─────────────────────────────────────────────────────────
    if figsize is None:
        figsize = (max(4 * n_metrics, 10), 5)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize, sharey=False)
    if n_metrics == 1:
        axes = [axes]
 
    rng = np.random.default_rng(0)
 
    for ax, metric in zip(axes, metrics):
        metric_label = _METRIC_LABELS.get(metric.lower(), metric.upper())
 
        # Gather per-model fold values
        model_data = []
        for m in models:
            vals = m["data"][metric].dropna().to_numpy(float)
            model_data.append(vals)
 
        # ── Boxplots ──────────────────────────────────────────────────────────
        positions = np.arange(n_models)
        bp = ax.boxplot(
            model_data,
            positions=positions,
            widths=0.45,
            patch_artist=True,
            medianprops=dict(color="white", linewidth=2),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
            flierprops=dict(marker="", markersize=0),  # hide default outlier dots
            zorder=2,
        )
        for patch, model in zip(bp["boxes"], model_names):
            patch.set_facecolor(color_map[model])
            patch.set_alpha(0.55)
            patch.set_edgecolor(color_map[model])
 
        for whisker, cap in zip(
            zip(bp["whiskers"][::2], bp["whiskers"][1::2]),
            zip(bp["caps"][::2],     bp["caps"][1::2]),
        ):
            pass  # colours already fine from default black
 
        # ── Jittered fold points ──────────────────────────────────────────────
        for pos, vals, name in zip(positions, model_data, model_names):
            x_jit = pos + rng.uniform(-jitter, jitter, size=len(vals))
            ax.scatter(x_jit, vals,
                       color=color_map[name], s=point_size,
                       alpha=point_alpha, edgecolors="none", zorder=3)
 
        # ── Median annotations ────────────────────────────────────────────────
        if show_median_label:
            for pos, vals in zip(positions, model_data):
                med = np.median(vals)
                ax.text(pos, med + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02,
                        f"{med:.2f}",
                        ha="center", va="bottom", fontsize=8,
                        fontweight="bold", color="black", zorder=4)
 
        # ── Styling ───────────────────────────────────────────────────────────
        ax.set_xticks(positions)
        ax.set_xticklabels(model_names, fontsize=9, rotation=15, ha="right")
        ax.set_ylabel(metric_label, fontsize=10)
        ax.set_title(metric_label, fontsize=11, fontweight="bold", pad=8)
        ax.set_xlim(-0.6, n_models - 0.4)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.5)
        ax.tick_params(axis="x", length=0)
 
    # ── Shared legend ─────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(facecolor=color_map[name], edgecolor=color_map[name],
                       alpha=0.7, label=name)
        for name in model_names
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=min(n_models, 5), fontsize=9, framealpha=0.85,
               bbox_to_anchor=(0.5, -0.08))
 
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    return fig, axes
 



_DEFAULT_COLORS = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B2", "#937860", "#DA8BC3", "#8C8C8C",
]
_DEFAULT_STYLES = ["-", "--", "-.", ":", "-", "--", "-.", ":"]
 
 
def plot_roc_pr_comparison(
    models: list[dict],
    y_true=None,
    col_true: str = "true_label",
    col_score: str = "predicted_score",
    title: str = "ROC & Precision-Recall Curve Comparison",
    title_left: str = "ROC Curve",
    title_right: str = "Precision-Recall Curve",
    colors: list[str] = None,
    linestyles: list[str] = None,
    figsize: tuple = (13, 6),
):
    """
    Plot ROC and Precision-Recall curves for multiple models on the same axes.
 
    Accepts either:
      (a) A shared y_true array + per-model "y_prob" arrays, or
      (b) Per-model DataFrames containing true labels and predicted scores.
 
    Args:
        models:      List of dicts, one per model:
                       {
                         "model_name": str,         # legend label
                         "y_prob":     array-like,  # predicted scores (option a)
                         "data":       DataFrame,   # option b — has col_true & col_score
                         "color":      str,          # optional
                         "linestyle":  str,          # optional, e.g. "--"
                       }
        y_true:      Shared true labels (only needed for option a).
        col_true:    Column name for true labels in each DataFrame (option b).
        col_score:   Column name for predicted scores in each DataFrame (option b).
        title:       Main figure title.
        title_left:  Left panel title.
        title_right: Right panel title.
        colors:      Colour cycle.
        linestyles:  Linestyle cycle.
        figsize:     Figure size.
 
    Returns:
        (fig, (ax_roc, ax_pr))
 
    Example — DataFrames (one per model, each with patient_id, true_label,
                           predicted_score, predicted_label):
        fig, axes = plot_roc_pr_comparison(
            models=[
                {"model_name": "Logistic Regression", "data": df_lr},
                {"model_name": "SVM",                  "data": df_svm, "linestyle": "--"},
                {"model_name": "Random Forest",         "data": df_rf,  "linestyle": "-."},
            ],
            col_true="true_label",
            col_score="predicted_score",
            title="Model Comparison",
        )
    """
    if colors     is None: colors     = _DEFAULT_COLORS
    if linestyles is None: linestyles = _DEFAULT_STYLES
 
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
 
    # Use first model's true labels for the PR baseline if DataFrames provided
    first = models[0]
    if "data" in first:
        prevalence = first["data"][col_true].mean()
    else:
        prevalence = np.asarray(y_true).mean()
 
    for i, m in enumerate(models):
        # Resolve y_true and y_prob for this model
        if "data" in m:
            df_m    = m["data"]
            yt      = df_m[col_true].to_numpy()
            y_prob  = df_m[col_score].to_numpy()
        else:
            yt      = np.asarray(y_true)
            y_prob  = np.asarray(m["y_prob"])
 
        name  = m["model_name"]
        color = m.get("color",     colors[i % len(colors)])
        ls    = m.get("linestyle", linestyles[i % len(linestyles)])
 
        # ── ROC ──────────────────────────────────────────────────────────────
        auroc        = roc_auc_score(yt, y_prob)
        fpr, tpr, _  = roc_curve(yt, y_prob)
        ax_roc.plot(fpr, tpr, color=color, linewidth=2, linestyle=ls,
                    label=f"{name}  (AUROC = {auroc:.3f})")
 
        # ── PR ───────────────────────────────────────────────────────────────
        auprc        = average_precision_score(yt, y_prob)
        prec, rec, _ = precision_recall_curve(yt, y_prob)
        ax_pr.plot(rec, prec, color=color, linewidth=2, linestyle=ls,
                   label=f"{name}  (AUPRC = {auprc:.3f})")
 
    # ── ROC decorations ───────────────────────────────────────────────────────
    ax_roc.plot([0, 1], [0, 1], "--", color="gray", linewidth=1,
                alpha=0.5, label="Random classifier")
    ax_roc.set_xlabel("False Positive Rate  (1 − Specificity)", fontsize=11)
    ax_roc.set_ylabel("True Positive Rate  (Sensitivity)", fontsize=11)
    ax_roc.set_title(title_left, fontsize=12, fontweight="bold")
    ax_roc.legend(fontsize=9, loc="lower right", framealpha=0.85)
    ax_roc.set_xlim(-0.02, 1.02)
    ax_roc.set_ylim(-0.02, 1.02)
    ax_roc.spines[["top", "right"]].set_visible(False)
    ax_roc.grid(linestyle="--", linewidth=0.4, alpha=0.5)
 
    # ── PR decorations ────────────────────────────────────────────────────────
    ax_pr.axhline(prevalence, color="gray", linewidth=1, linestyle="--",
                  alpha=0.6, label=f"No-skill baseline (prevalence = {prevalence:.2f})")
    ax_pr.set_xlabel("Recall  (Sensitivity)", fontsize=11)
    ax_pr.set_ylabel("Precision  (PPV)", fontsize=11)
    ax_pr.set_title(title_right, fontsize=12, fontweight="bold")
    ax_pr.legend(fontsize=9, loc="lower left", framealpha=0.85)
    ax_pr.set_xlim(-0.02, 1.02)
    ax_pr.set_ylim(-0.02, 1.05)
    ax_pr.spines[["top", "right"]].set_visible(False)
    ax_pr.grid(linestyle="--", linewidth=0.4, alpha=0.5)
 
    plt.tight_layout()
    return fig, (ax_roc, ax_pr)




def performance_summary_table(
    data: pd.DataFrame,
    col_model: str = "model",
    metrics: list[str] = None,
    decimal_places: int = 3,
    highlight_best: bool = True,
):
    """
    Display a pre-computed performance metrics DataFrame as a styled table
    in a Jupyter notebook, with the best value per metric highlighted in green.
 
    Args:
        data:            DataFrame with one row per model and one column per
                         metric. Must include a model name column.
                         Example columns: model, AUROC, AUPRC, sensitivity,
                                          specificity, F1
        col_model:       Column name containing model names (used as row index).
        metrics:         Metric columns to display. If None, all numeric columns
                         except col_model are shown.
        decimal_places:  Rounding applied to all numeric values.
        highlight_best:  Highlight the best value per metric column in green.
 
    Returns:
        pandas Styler — use display() in a notebook or leave as last cell.
 
    Example:
        summary = performance_summary_table(df, col_model="model")
        display(summary)
    """
    df = data.copy()
 
    # Set model column as index
    if col_model in df.columns:
        df = df.set_index(col_model)
 
    # Select metric columns
    if metrics is not None:
        df = df[[c for c in metrics if c in df.columns]]
    else:
        df = df.select_dtypes(include="number")
 
    # Round
    df = df.round(decimal_places)
 
    # Rename columns for display (capitalise common names)
    rename_map = {
        "auroc": "AUROC", "auprc": "AUPRC",
        "sensitivity": "Sensitivity", "specificity": "Specificity",
        "f1": "F1", "f1_score": "F1", "f1score": "F1",
        "precision": "Precision", "n": "N", "n_positive": "N Positive",
    }
    df.columns = [rename_map.get(c.lower(), c) for c in df.columns]
 
    if not highlight_best:
        return (
            df.style
            .set_properties(**{"text-align": "center", "padding": "6px 14px"})
            .set_table_styles([
                {"selector": "th", "props": [("text-align", "center"),
                                             ("font-weight", "bold"),
                                             ("background-color", "#f0f0f0"),
                                             ("padding", "6px 14px")]},
            ])
        )
 
    info_cols = {"N", "N Positive"}
 
    def highlight_col(col):
        styles = [""] * len(col)
        if col.name in info_cols:
            return styles
        try:
            numeric = pd.to_numeric(col, errors="coerce")
            if numeric.isna().all():
                return styles
            best_idx = numeric.idxmax()
            styles[col.index.get_loc(best_idx)] = (
                "background-color: #d4edda; font-weight: bold; color: #155724;"
            )
        except Exception:
            pass
        return styles
 
    styled = (
        df.style
        .apply(highlight_col, axis=0)
        .set_properties(**{"text-align": "center", "padding": "6px 14px"})
        .set_table_styles([
            {"selector": "th.col_heading",
             "props": [("text-align", "center"), ("font-weight", "bold"),
                       ("background-color", "#f0f0f0"), ("padding", "6px 14px")]},
            {"selector": "th.row_heading",
             "props": [("font-weight", "bold"), ("padding", "6px 14px")]},
            {"selector": "td",
             "props": [("border", "1px solid #dee2e6")]},
            {"selector": "tr:hover td",
             "props": [("background-color", "#f5f5f5")]},
        ])
        .set_caption(
            "<b>Performance Summary</b>"
            " — best value per metric highlighted in green"
        )
    )
    return styled
 


def plot_confusion_matrices(
    models: list[dict],
    col_true: str = "true_label",
    col_pred: str = "predicted_label",
    class_names: list[str] = None,
    title: str = "Confusion Matrices",
    cmap: str = "Blues",
    decimal_places: int = 1,
    figsize: tuple = None,
):
    """
    Plot confusion matrices for multiple models in a single row.
 
    Each cell shows the count and row-normalised percentage.
 
    Args:
        models:          List of dicts, one per model:
                           {
                             "model_name": str,
                             "data":       DataFrame with col_true & col_pred
                           }
        col_true:        Column name for true labels.
        col_pred:        Column name for predicted labels.
        class_names:     Display names for classes, e.g. ["Negative", "Positive"].
        title:           Figure-level title.
        cmap:            Colormap (default "Blues").
        decimal_places:  Decimal places for percentage annotations.
        figsize:         Figure size. Auto-scaled if None.
 
    Returns:
        (fig, axes)
 
    Example:
        fig, axes = plot_confusion_matrices(
            models=[
                {"model_name": "Logistic Regression", "data": df_lr},
                {"model_name": "SVM",                  "data": df_svm},
                {"model_name": "Random Forest",         "data": df_rf},
            ],
            class_names=["Negative", "Positive"],
        )
    """
    n = len(models)
    if figsize is None:
        figsize = (4.5 * n, 4)
 
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
 
    for ax, m in zip(axes, models):
        df    = m["data"]
        name  = m["model_name"]
        yt    = df[col_true].to_numpy()
        yp    = df[col_pred].to_numpy()
 
        classes = class_names or [str(c) for c in sorted(np.unique(yt))]
        cm      = confusion_matrix(yt, yp)
        cm_pct  = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
 
        im = ax.imshow(cm_pct, cmap=cmap, vmin=0, vmax=100)
 
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                brightness = cm_pct[i, j] / 100
                color      = "white" if brightness > 0.55 else "black"
                ax.text(j, i, f"{cm[i, j]}",
                        ha="center", va="center",
                        fontsize=13, fontweight="bold", color=color)
 
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, fontsize=10)
        ax.set_yticklabels(classes, fontsize=10)
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("Actual", fontsize=10)
        ax.set_title(name, fontsize=11, fontweight="bold", pad=10)
 
 
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig, axes






_DEFAULT_COLORS = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B2", "#937860", "#DA8BC3", "#8C8C8C",
]
 
 
def _strip_prefix(name: str) -> str:
    return re.sub(r"^\w+__", "", name)
 
 
def _aggregate(df, col_feature, col_coef):
    """Return DataFrame with columns [feature, mean, std]."""
    if df.groupby(col_feature).size().max() > 1:
        agg = (
            df.groupby(col_feature)[col_coef]
            .agg(["mean", "std"])
            .reset_index()
            .rename(columns={col_feature: "feature"})
        )
    else:
        agg = df[[col_feature, col_coef]].copy()
        agg.columns = ["feature", "mean"]
        agg["std"] = df["std"].values if "std" in df.columns else np.nan
    return agg.set_index("feature")
 
 
def plot_coefficient_comparison(
    models: list[dict],
    sort_by: int = 0,
    top_n: int = None,
    strip_prefixes: bool = True,
    colors: list[str] = None,
    title: str = "Feature Coefficient Comparison Across Models",
    decimal_places: int = 2,
    figsize: tuple = None,
    panel_width: float = 3.5,
):
    """
    Multi-panel horizontal bar chart comparing feature coefficients across models.
 
    Each model gets its own panel with its own x-axis and scale — so models
    with different magnitude types (odds ratios, feature importances, raw
    coefficients) are all clearly readable side by side.
 
    The y-axis (feature names) is shared, sorted by the sort_by model.
 
    Args:
        models:          List of dicts, one per model:
                           {
                             "model_name": str,           # display name
                             "data":       DataFrame,     # fold-level or aggregated
                             "mode":       str,           # "coefficient", "odds_ratio",
                                                          # or "importance" (all positive)
                             "xlabel":     str,           # optional x-axis label override
                             "col_feature":str,           # default "feature"
                             "col_coef":   str,           # default "coefficient"
                           }
        sort_by:         Model index used to sort features (highest → lowest).
        top_n:           Limit to top_n features by absolute mean of sort_by model.
        strip_prefixes:  Remove sklearn prefixes (num__, binary__, etc.).
        colors:          One colour per model.
        title:           Figure title.
        decimal_places:  Rounding for annotations.
        figsize:         Figure size. Auto-scaled if None.
        panel_width:     Width of each model panel (used when figsize is None).
 
    Returns:
        (fig, axes)
 
    Example:
        fig, axes = plot_coefficient_comparison(
            models=[
                {"model_name": "Logistic Reg.", "data": df_lr,
                 "mode": "odds_ratio"},
                {"model_name": "SVM",           "data": df_svm,
                 "mode": "coefficient"},
                {"model_name": "Random Forest", "data": df_rf,
                 "mode": "importance"},
            ],
        )
    """
    n_models = len(models)
    if colors is None:
        colors = [_DEFAULT_COLORS[i % len(_DEFAULT_COLORS)] for i in range(n_models)]
 
    # ── Aggregate each model ──────────────────────────────────────────────────
    agg_list = []
    for m in models:
        col_f = m.get("col_feature", "Feature")
        col_c = m.get("col_coef", "Coefficient")
        agg   = _aggregate(m["data"], col_f, col_c)
        if strip_prefixes:
            agg.index = [_strip_prefix(f) for f in agg.index]
        agg_list.append(agg)
 
    # ── Common features ───────────────────────────────────────────────────────
    common = set(agg_list[0].index)
    for agg in agg_list[1:]:
        common &= set(agg.index)
    common = sorted(common)
 
    # Sort by sort_by model
    ref   = agg_list[sort_by].loc[common, "mean"]
    if top_n is not None:
        common = ref.abs().nlargest(top_n).index.tolist()
        ref    = ref.loc[common]
    order      = ref.sort_values(ascending=False).index.tolist()
    n_features = len(order)
 
    # ── Figure: shared y-axis via sharey ──────────────────────────────────────
    if figsize is None:
        figsize = (panel_width * n_models + 1.5, max(5, n_features * 0.42 + 1.5))
 
    fig, axes = plt.subplots(
        1, n_models, figsize=figsize,
        sharey=True,
        gridspec_kw={"wspace": 0.08},
    )
    if n_models == 1:
        axes = [axes]
 
    fmt = f".{decimal_places}f"
    y   = np.arange(n_features)
 
    for ax, m, color, agg in zip(axes, models, colors, agg_list):
        mode = m.get("mode", "coefficient")
 
        # Convert to plot values
        means = agg.loc[order, "mean"].to_numpy(float)
        stds  = agg.loc[order, "std"].to_numpy(float)
 
        if mode == "odds_ratio":
            plot_vals = np.exp(means)
            lo_err    = plot_vals - np.exp(means - stds)
            hi_err    = np.exp(means + stds) - plot_vals
            ref_line  = 1.0
            log_x     = True
            auto_xl   = "Odds Ratio (log scale)"
        elif mode == "importance":
            plot_vals = means          # importances already positive
            lo_err    = stds
            hi_err    = stds
            ref_line  = 0.0
            log_x     = False
            auto_xl   = "Feature Importance"
        else:  # raw coefficient
            plot_vals = means
            lo_err    = stds
            hi_err    = stds
            ref_line  = 0.0
            log_x     = False
            auto_xl   = "Coefficient"
 
        xlabel = m.get("xlabel", auto_xl)
        nan_err = np.isnan(lo_err)
 
        # Alternating row shading (first panel only — shared y means it spans all)
        if ax is axes[0]:
            for i in range(n_features):
                if i % 2 == 0:
                    ax.axhspan(i - 0.5, i + 0.5, color="gray",
                               alpha=0.06, zorder=0)
 
        ax.axvline(ref_line, color="black", linewidth=0.9,
                   linestyle="--", alpha=0.5, zorder=1)
 
        # Bars
        if log_x:
            bar_left  = np.minimum(plot_vals, ref_line)
            bar_width = np.abs(plot_vals - ref_line)
        else:
            bar_left  = np.minimum(plot_vals, ref_line)
            bar_width = np.abs(plot_vals - ref_line)
 
        ax.barh(y, bar_width, left=bar_left, height=0.55,
                color=color, alpha=0.35, zorder=2)
 
        # Error bars
        for i, (yi, val, lo, hi) in enumerate(zip(y, plot_vals, lo_err, hi_err)):
            if not nan_err[i]:
                ax.errorbar(val, yi, xerr=[[lo], [hi]],
                            fmt="none", ecolor=color,
                            elinewidth=1.4, capsize=3, capthick=1.4,
                            alpha=0.9, zorder=3)
 
        # Dots
        ax.scatter(plot_vals, y, color=color, s=40,
                   edgecolors="white", linewidths=0.7, zorder=4)
 
        # Value labels
        for yi, val in zip(y, plot_vals):
            offset = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.01 if not log_x else 0
            ax.text(val + offset if val >= ref_line else val - offset,
                    yi, f"{val:{fmt}}",
                    ha="left" if val >= ref_line else "right",
                    va="center", fontsize=7.5, color=color)
 
        if log_x:
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(
                ticker.FuncFormatter(lambda v, _: f"{v:g}")
            )
            ax.xaxis.set_minor_formatter(ticker.NullFormatter())
 
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_title(m["model_name"], fontsize=11, fontweight="bold",
                     color=color, pad=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", linestyle="--", linewidth=0.4, alpha=0.45)
        ax.tick_params(axis="y", length=0)
 
    # Y-tick labels only on first panel
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(order, fontsize=9)
    axes[0].invert_yaxis()
 
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    return fig, axes




 
_DEFAULT_COLORS = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B2", "#937860", "#DA8BC3", "#8C8C8C",
]
 
 
def _strip_prefix(name: str) -> str:
    return re.sub(r"^\w+__", "", name)
 
 
def _aggregate(df, col_feature, col_coef):
    """Return DataFrame indexed by feature with columns [mean, std]."""
    if df.groupby(col_feature).size().max() > 1:
        agg = (
            df.groupby(col_feature)[col_coef]
            .agg(["mean", "std"])
            .reset_index()
            .rename(columns={col_feature: "feature"})
        )
    else:
        agg = df[[col_feature, col_coef]].copy()
        agg.columns = ["feature", "mean"]
        agg["std"] = df["std"].values if "std" in df.columns else np.nan
    return agg.set_index("feature")
 
 
def _draw_panel(ax, plot_vals, order, lo_err, hi_err, nan_err,
                ref_line, log_x, color, fmt):
    """Draw bars, error bars, dots and value labels onto a single axes."""
    n = len(order)
    y = np.arange(n)
 
    # Alternating row shading
    for i in range(n):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color="gray", alpha=0.06, zorder=0)
 
    ax.axvline(ref_line, color="black", linewidth=0.9,
               linestyle="--", alpha=0.5, zorder=1)
 
    bar_left  = np.minimum(plot_vals, ref_line)
    bar_width = np.abs(plot_vals - ref_line)
    ax.barh(y, bar_width, left=bar_left, height=0.55,
            color=color, alpha=0.35, zorder=2)
 
    for i, (yi, val, lo, hi) in enumerate(zip(y, plot_vals, lo_err, hi_err)):
        if not nan_err[i]:
            ax.errorbar(val, yi, xerr=[[lo], [hi]],
                        fmt="none", ecolor=color,
                        elinewidth=1.4, capsize=3, capthick=1.4,
                        alpha=0.9, zorder=3)
 
    ax.scatter(plot_vals, y, color=color, s=40,
               edgecolors="white", linewidths=0.7, zorder=4)
 
    if log_x:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
 
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=9)
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", linestyle="--", linewidth=0.4, alpha=0.45)
    ax.tick_params(axis="y", length=0)
 
    # ── Value labels just above each bar (in data coords) ────────────────────
    # x = outer tip of bar (accounting for error bar); y = slightly above bar
    for i, (yi, val) in enumerate(zip(y, plot_vals)):
        # x position: tip of error bar if available, else tip of bar
        if not nan_err[i]:
            x_tip = val + hi_err[i] if val >= ref_line else val - lo_err[i]
        else:
            x_tip = val
        ax.text(x_tip, yi - 0.32, f"{val:{fmt}}",
                ha="center", va="bottom", fontsize=8,
                color=color, fontweight="bold",
                zorder=6, clip_on=False)
 
 
def plot_coefficient_comparison(
    models: list[dict],
    top_n: int = None,
    strip_prefixes: bool = True,
    colors: list[str] = None,
    title: str = "Feature Coefficient Comparison Across Models",
    decimal_places: int = 2,
    figsize: tuple = None,
    panel_width: float = 5.0,
):
    """
    One independent panel per model — each with its own x-axis, y-axis,
    and feature ordering (sorted by that model's own coefficient magnitude,
    highest to lowest).
 
    Args:
        models:          List of dicts, one per model:
                           {
                             "model_name": str,
                             "data":       DataFrame  (fold-level or aggregated),
                             "mode":       str,        "coefficient" | "odds_ratio"
                                                       | "importance"
                             "xlabel":     str,        optional x-axis label
                             "col_feature":str,        default "feature"
                             "col_coef":   str,        default "coefficient"
                           }
        top_n:           Show only top_n features per panel by absolute mean.
        strip_prefixes:  Remove sklearn prefixes (num__, binary__, etc.).
        colors:          One colour per model.
        title:           Figure-level title.
        decimal_places:  Rounding for value annotations.
        figsize:         Figure size. Auto-scaled if None.
        panel_width:     Width per panel when figsize is None.
 
    Returns:
        (fig, axes)
 
    Example:
        fig, axes = plot_coefficient_comparison(
            models=[
                {"model_name": "Logistic Regression", "data": df_lr,
                 "mode": "odds_ratio"},
                {"model_name": "SVM",                 "data": df_svm,
                 "mode": "coefficient"},
                {"model_name": "Random Forest",        "data": df_rf,
                 "mode": "importance"},
            ],
        )
    """
    n_models = len(models)
    if colors is None:
        colors = [_DEFAULT_COLORS[i % len(_DEFAULT_COLORS)] for i in range(n_models)]
 
    fmt = f".{decimal_places}f"
 
    # ── Aggregate each model independently ───────────────────────────────────
    agg_list = []
    for m in models:
        col_f = m.get("col_feature", "Feature")
        col_c = m.get("col_coef", "Coefficient")
        agg   = _aggregate(m["data"], col_f, col_c)
        if strip_prefixes:
            agg.index = [_strip_prefix(f) for f in agg.index]
        agg_list.append(agg)
 
    # ── Figure ────────────────────────────────────────────────────────────────
    # Height driven by the largest feature set
    max_features = max(len(a) if top_n is None else min(top_n, len(a))
                       for a in agg_list)
    if figsize is None:
        figsize = (panel_width * n_models, max(5, max_features * 0.42 + 1.5))
 
    fig, axes = plt.subplots(
        1, n_models, figsize=figsize,
        gridspec_kw={"wspace": 0.70},   # space between independent panels
    )
    if n_models == 1:
        axes = [axes]
 
    for ax, m, color, agg in zip(axes, models, colors, agg_list):
        mode = m.get("mode", "coefficient")
 
        # Each panel orders features independently
        means = agg["mean"].to_numpy(float)
        stds  = agg["std"].to_numpy(float)
        mode  = m.get("mode", "coefficient")
 
        if mode == "importance":
            # Importance: sort by absolute magnitude (all positive anyway)
            sort_vals = np.abs(means)
        else:
            # LR / SVM: sort by actual value — positive first, negative last
            sort_vals = means
 
        if top_n is not None:
            top_idx    = np.argsort(np.abs(means))[::-1][:top_n]
            feat_order = agg.index[top_idx].tolist()
            # Re-sort those top_n by sign-aware value
            feat_order = sorted(feat_order,
                                key=lambda f: agg.loc[f, "mean"]
                                if mode != "importance"
                                else abs(agg.loc[f, "mean"]),
                                reverse=True)
        else:
            feat_order = agg.index[np.argsort(sort_vals)[::-1]].tolist()
 
        means = agg.loc[feat_order, "mean"].to_numpy(float)
        stds  = agg.loc[feat_order, "std"].to_numpy(float)
 
        # Convert to plot scale
        if mode == "odds_ratio":
            plot_vals = np.exp(means)
            lo_err    = plot_vals - np.exp(means - stds)
            hi_err    = np.exp(means + stds) - plot_vals
            ref_line  = 1.0
            log_x     = True
            auto_xl   = "Odds Ratio (log scale)"
        elif mode == "importance":
            plot_vals = means
            lo_err    = stds
            hi_err    = stds
            ref_line  = 0.0
            log_x     = False
            auto_xl   = "Feature Importance"
        else:
            plot_vals = means
            lo_err    = stds
            hi_err    = stds
            ref_line  = 0.0
            log_x     = False
            auto_xl   = "Coefficient"
 
        nan_err = np.isnan(lo_err)
        xlabel  = m.get("xlabel", auto_xl)
 
        _draw_panel(ax, plot_vals, feat_order, lo_err, hi_err,
                    nan_err, ref_line, log_x, color, fmt)
 
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_title(m["model_name"], fontsize=11, fontweight="bold",
                     color=color, pad=8)
 
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    return fig, axes
 