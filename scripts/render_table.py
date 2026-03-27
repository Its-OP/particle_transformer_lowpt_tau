import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "DejaVu Sans"

row_labels = [
    "Input",
    "Output",
    "Task",
    "Architecture",
    "Complexity",
    "Metric",
    "Status",
]

col_labels = [
    "Stage 1: Pre-Filter",
    "Stage 2: Refinement",
    "Stage 3: Triplet Selection",
]

cell_text = [
    ["All ~1130 tracks", "Top-600 candidates", "~50 candidate tracks"],
    ["600 tracks", "~50 tracks", "Best \u03c0\u03c0\u03c0 triplet"],
    [
        "Rank tracks by\nsignal likelihood",
        "Re-score with\nricher context",
        "Enumerate & score triplets\nusing model confidence\n+ physics",
    ],
    [
        "MLP + local kNN\nmessage passing",
        "Cross-attention /\nDETR-like",
        "Combinatorial search\n+ physics constraints",
    ],
    [
        "O(P\u00b7k) \u2014 fast",
        "O(K\u00b2) \u2014 affordable\nat K=600",
        "C(50,3) \u2248 19K triplets\n\u2014 tractable",
    ],
    [
        "Recall@600 > 0.9",
        "Recall / precision\non reduced set",
        "Exact match of\n\u03c4 decay products",
    ],
    ["Implemented", "In progress", "Planned"],
]

dark_red = "#8B1A1A"
border_color = "#CCCCCC"

figure, axis = plt.subplots(figsize=(14, 7))
axis.set_axis_off()

number_of_rows = len(row_labels)

table = axis.table(
    cellText=cell_text,
    rowLabels=row_labels,
    colLabels=col_labels,
    cellLoc="center",
    rowLoc="center",
    loc="center",
)

table.auto_set_font_size(False)
table.set_fontsize(11)

row_label_width = 0.14
stage_column_width = (1.0 - row_label_width) / len(col_labels)

for (row_index, col_index), cell in table.get_celld().items():
    # Column widths
    if col_index == -1:
        cell.set_width(row_label_width)
    else:
        cell.set_width(stage_column_width)

    # All cells: white background, thin border
    cell.set_facecolor("white")
    cell.set_edgecolor(border_color)
    cell.set_linewidth(0.8)
    cell.PAD = 0.05

    # Header row
    if row_index == 0:
        cell.set_text_props(color=dark_red, fontweight="bold", fontsize=12)
        cell.set_height(0.08)
    # Row labels
    elif col_index == -1:
        cell.set_text_props(color="#333333", fontweight="bold", fontsize=11)
        cell.set_height(0.1)
    # Data cells
    else:
        cell.set_text_props(color="#333333", fontsize=10.5)
        cell.set_height(0.1)

plt.tight_layout()
output_path = "/Users/oleh/Projects/masters/part/reports/graphics/approach_table.png"
plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved to {output_path}")
