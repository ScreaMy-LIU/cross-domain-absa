from pathlib import Path

import matplotlib.pyplot as plt


# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURE_DIR = PROJECT_ROOT / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Figure 1: Overall Performance
# ============================================================

metrics = ["Accuracy", "Macro F1"]
values = [0.7358, 0.6465]

plt.figure(figsize=(7, 5))

bars = plt.bar(metrics, values)

plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Laptop ABSA - Overall Performance")

for bar, value in zip(bars, values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.02,
        f"{value:.2%}",
        ha="center",
        fontsize=11,
    )

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "overall_metrics.png",
    dpi=300,
)

plt.close()


# ============================================================
# Figure 2: Per-Class F1
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

plt.figure(figsize=(7, 5))

bars = plt.bar(
    classes,
    f1_scores,
)

plt.ylim(0, 1)
plt.ylabel("F1 Score")
plt.title("Laptop ABSA - Per-Class F1")

for bar, value in zip(bars, f1_scores):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.02,
        f"{value:.4f}",
        ha="center",
        fontsize=11,
    )

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "class_f1_scores.png",
    dpi=300,
)

plt.close()


# ============================================================
# Figure 3: Test Label Distribution
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

plt.figure(figsize=(7, 5))

bars = plt.bar(
    labels,
    counts,
)

plt.ylabel("Number of Samples")
plt.title("Laptop ABSA - Test Label Distribution")

for bar, value in zip(bars, counts):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 3,
        str(value),
        ha="center",
        fontsize=11,
    )

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "label_distribution.png",
    dpi=300,
)

plt.close()


print("Figures generated successfully.")

print(
    FIGURE_DIR / "overall_metrics.png"
)

print(
    FIGURE_DIR / "class_f1_scores.png"
)

print(
    FIGURE_DIR / "label_distribution.png"
)