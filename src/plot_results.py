from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURE_DIR = PROJECT_ROOT / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Unified figure style
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 15,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 120,
    "savefig.dpi": 300,
})


def clean_axes(ax):
    """Apply a clean research-style layout."""

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.25,
    )

    ax.set_axisbelow(True)


def add_value_labels(ax, bars, percentage=False):
    """Add values above bars."""

    for bar in bars:

        value = bar.get_height()

        if percentage:
            label = f"{value:.2%}"
        else:
            label = f"{value:.4f}"

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.025,
            label,
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )


# ============================================================
# Figure 1
# Overall Performance
# ============================================================

metrics = [
    "Accuracy",
    "Macro F1",
]

values = [
    0.7358,
    0.6465,
]

fig, ax = plt.subplots(
    figsize=(7.5, 5),
)

bars = ax.bar(
    metrics,
    values,
    width=0.55,
)

ax.set_ylim(0, 1)

ax.set_ylabel("Score")

ax.set_title(
    "Laptop ABSA: Overall Performance",
    pad=15,
    fontweight="bold",
)

ax.text(
    0.5,
    1.01,
    "DistilBERT · SemEval Laptop · Sentence-grouped evaluation",
    transform=ax.transAxes,
    ha="center",
    fontsize=9,
    alpha=0.65,
)

clean_axes(ax)

add_value_labels(
    ax,
    bars,
    percentage=True,
)

fig.tight_layout()

fig.savefig(
    FIGURE_DIR / "overall_metrics.png",
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# Figure 2
# Per-Class F1
# ============================================================

classes = [
    "Negative",
    "Neutral",
    "Positive",
]

f1_scores = [
    0.7892,
    0.3359,
    0.8145,
]

fig, ax = plt.subplots(
    figsize=(7.5, 5),
)

bars = ax.bar(
    classes,
    f1_scores,
    width=0.58,
)

ax.set_ylim(0, 1)

ax.set_ylabel("F1 Score")

ax.set_title(
    "Laptop ABSA: Per-Class F1",
    pad=15,
    fontweight="bold",
)

ax.text(
    0.5,
    1.01,
    "Neutral sentiment remains the most difficult class",
    transform=ax.transAxes,
    ha="center",
    fontsize=9,
    alpha=0.65,
)

clean_axes(ax)

add_value_labels(
    ax,
    bars,
)

fig.tight_layout()

fig.savefig(
    FIGURE_DIR / "class_f1_scores.png",
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# Figure 3
# Test Label Distribution
# ============================================================

labels = [
    "Negative",
    "Neutral",
    "Positive",
]

counts = [
    189,
    68,
    201,
]

fig, ax = plt.subplots(
    figsize=(7.5, 5),
)

bars = ax.bar(
    labels,
    counts,
    width=0.58,
)

ax.set_ylabel(
    "Number of Samples"
)

ax.set_title(
    "Laptop ABSA: Test Label Distribution",
    pad=15,
    fontweight="bold",
)

ax.text(
    0.5,
    1.01,
    "Held-out evaluation set · 458 aspect instances",
    transform=ax.transAxes,
    ha="center",
    fontsize=9,
    alpha=0.65,
)

clean_axes(ax)

for bar in bars:

    value = int(
        bar.get_height()
    )

    ax.text(
        bar.get_x()
        + bar.get_width() / 2,
        value + 4,
        str(value),
        ha="center",
        fontweight="bold",
    )

ax.set_ylim(
    0,
    max(counts) * 1.20,
)

fig.tight_layout()

fig.savefig(
    FIGURE_DIR / "label_distribution.png",
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# Figure 4
# In-Domain vs Cross-Domain
# ============================================================

experiments = [
    "Laptop → Laptop\n(In-Domain)",
    "Laptop → Restaurant\n(Cross-Domain)",
]

accuracy_scores = [
    0.7358,
    0.6911,
]

macro_f1_scores = [
    0.6465,
    0.5217,
]

x = np.arange(
    len(experiments)
)

width = 0.30

fig, ax = plt.subplots(
    figsize=(9, 5.5),
)

accuracy_bars = ax.bar(
    x - width / 2,
    accuracy_scores,
    width,
    label="Accuracy",
)

f1_bars = ax.bar(
    x + width / 2,
    macro_f1_scores,
    width,
    label="Macro F1",
)

ax.set_ylim(
    0,
    1,
)

ax.set_ylabel(
    "Score"
)

ax.set_xticks(
    x
)

ax.set_xticklabels(
    experiments
)

ax.set_title(
    "In-Domain vs Cross-Domain ABSA Performance",
    pad=18,
    fontweight="bold",
)

ax.text(
    0.5,
    1.01,
    "DistilBERT trained on Laptop domain",
    transform=ax.transAxes,
    ha="center",
    fontsize=10,
    alpha=0.65,
)

clean_axes(ax)

add_value_labels(
    ax,
    accuracy_bars,
    percentage=True,
)

add_value_labels(
    ax,
    f1_bars,
    percentage=True,
)

ax.legend(
    frameon=False,
    loc="upper right",
)

# ------------------------------------------------------------
# Performance change annotations
# ------------------------------------------------------------

accuracy_change = (
    accuracy_scores[1]
    - accuracy_scores[0]
)

f1_change = (
    macro_f1_scores[1]
    - macro_f1_scores[0]
)

ax.text(
    0.5,
    0.88,
    f"Accuracy change: {accuracy_change:+.2%}",
    transform=ax.transAxes,
    ha="center",
    fontsize=10,
)

ax.text(
    0.5,
    0.82,
    f"Macro F1 change: {f1_change:+.2%}",
    transform=ax.transAxes,
    ha="center",
    fontsize=10,
    fontweight="bold",
)

fig.tight_layout()

fig.savefig(
    FIGURE_DIR / "domain_comparison.png",
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# Finished
# ============================================================

print("=" * 60)

print(
    "Figures generated successfully."
)

print("=" * 60)

print(
    FIGURE_DIR
    / "overall_metrics.png"
)

print(
    FIGURE_DIR
    / "class_f1_scores.png"
)

print(
    FIGURE_DIR
    / "label_distribution.png"
)

print(
    FIGURE_DIR
    / "domain_comparison.png"
)

print("=" * 60)