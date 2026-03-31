import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "DejaVu Sans"

experiments = [
    "MLP+kNN\nbaseline",
    "Extended\nfeatures (16)",
    "ParticleNeXt\nbackbone",
    "ISAB\n(global attn)",
    "Score\npropagation",
    "Pairwise LV\n(concat)",
    "Pairwise LV\n(attn bias)",
    "ASL\nloss",
    "Triplet\nscorer",
]

recall_at_200 = [0.62, 0.625, 0.62, 0.62, 0.58, 0.24, 0.39, 0.41, None]
colors = ["#2C3E50", "#7F8C8D", "#E74C3C", "#E74C3C", "#E74C3C", "#E74C3C", "#E74C3C", "#E74C3C", "#95A5A6"]
labels = ["Baseline", "No gain", "No gain", "No gain", "Degraded", "Collapsed", "Collapsed", "Degraded", "OOM"]

figure, axis = plt.subplots(figsize=(12, 5))

# Plot bars (skip None values)
bar_positions = range(len(experiments))
bar_values = [value if value is not None else 0 for value in recall_at_200]
bars = axis.bar(bar_positions, bar_values, color=colors, width=0.6, edgecolor="white", linewidth=1.5)

# Add value labels on bars
for index, (bar, value, label) in enumerate(zip(bars, recall_at_200, labels)):
    if value is not None:
        axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                  f"{value:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold", color="#333333")
        axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                  label, ha="center", va="center", fontsize=8, color="white", fontweight="bold")
    else:
        axis.text(bar.get_x() + bar.get_width() / 2, 0.05,
                  "OOM\n(96GB)", ha="center", va="bottom", fontsize=9, color="#95A5A6", fontweight="bold")

# Baseline reference line
axis.axhline(y=0.62, color="#2C3E50", linestyle="--", linewidth=1, alpha=0.5)
axis.text(len(experiments) - 0.5, 0.625, "baseline = 0.62", ha="right", va="bottom",
          fontsize=9, color="#2C3E50", alpha=0.7)

axis.set_xticks(bar_positions)
axis.set_xticklabels(experiments, fontsize=9, color="#333333")
axis.set_ylabel("Recall@200", fontsize=12, color="#333333")
axis.set_ylim(0, 0.75)
axis.spines["top"].set_visible(False)
axis.spines["right"].set_visible(False)
axis.spines["left"].set_color("#CCCCCC")
axis.spines["bottom"].set_color("#CCCCCC")
axis.tick_params(colors="#666666")

plt.tight_layout()
output_path = "/Users/oleh/Projects/masters/part/reports/graphics/experiment_results.png"
plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved to {output_path}")
