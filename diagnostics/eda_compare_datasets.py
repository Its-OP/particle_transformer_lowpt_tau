"""
Exploratory Data Analysis: Compare train, val, and extended subset val datasets.

Goal: Determine if the extended subset has fundamentally different characteristics
that would explain a model performance drop from R@200=0.62 to R@200=0.40.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent / "data" / "low-pt"

DATASET_PATHS = {
    "train": BASE_DIR / "train",
    "val": BASE_DIR / "val",
    "ext_subset_val": BASE_DIR / "extended" / "subset" / "val",
}

# Features whose distributions we compare across datasets
KEY_FEATURES = [
    "track_pt",
    "track_eta",
    "track_phi",
    "track_dxy_significance",
    "track_dz_significance",
    "track_charge",
]

LABEL_COLUMN = "track_label_from_tau"
TRACK_COUNT_COLUMN = "event_n_tracks"


# ──────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_dataset(directory: Path) -> pd.DataFrame:
    """Load all parquet files from *directory* into a single DataFrame."""
    parquet_files = sorted(directory.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {directory}")
    dataframe = pd.concat(
        [pq.read_table(file_path).to_pandas() for file_path in parquet_files],
        ignore_index=True,
    )
    return dataframe


def compute_per_event_list_lengths(series: pd.Series) -> np.ndarray:
    """Return an array of list lengths for a column that stores lists."""
    return np.array([len(row) for row in series])


def compute_per_event_signal_count(label_series: pd.Series) -> np.ndarray:
    """Count how many entries equal 1 (ground-truth signal tracks) per event."""
    return np.array([int(np.sum(np.array(row) == 1)) for row in label_series])


def flatten_list_column(series: pd.Series) -> np.ndarray:
    """Flatten a column of lists into a single 1-D numpy array."""
    return np.concatenate([np.array(row, dtype=np.float64) for row in series])


# ──────────────────────────────────────────────────────────────────────────────
# Reporting helpers
# ──────────────────────────────────────────────────────────────────────────────

def distribution_summary(values: np.ndarray) -> dict:
    """Compute a compact statistical summary of *values*."""
    return {
        "count": len(values),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p5": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def print_comparison_table(title: str, summaries: dict[str, dict]):
    """Pretty-print a comparison table for several datasets."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")

    # Collect all stat keys from the first summary
    stat_keys = list(next(iter(summaries.values())).keys())
    dataset_names = list(summaries.keys())

    # Header
    header = f"{'stat':<10}" + "".join(f"{name:>18}" for name in dataset_names)
    print(header)
    print("-" * len(header))

    for stat_key in stat_keys:
        row = f"{stat_key:<10}"
        for dataset_name in dataset_names:
            value = summaries[dataset_name][stat_key]
            if isinstance(value, int) or (isinstance(value, float) and value == int(value) and abs(value) < 1e9):
                row += f"{int(value):>18}"
            else:
                row += f"{value:>18.4f}"
        print(row)


# ──────────────────────────────────────────────────────────────────────────────
# Main analysis
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # ── 1. Load datasets ─────────────────────────────────────────────────────
    datasets: dict[str, pd.DataFrame] = {}
    for dataset_name, dataset_path in DATASET_PATHS.items():
        print(f"Loading {dataset_name} from {dataset_path} ...")
        datasets[dataset_name] = load_dataset(dataset_path)
        print(f"  -> {len(datasets[dataset_name])} events")

    # ── 2. Column comparison ─────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("  Column comparison")
    print(f"{'=' * 80}")

    all_column_sets = {name: set(dataframe.columns) for name, dataframe in datasets.items()}
    all_columns_union = set().union(*all_column_sets.values())

    for column in sorted(all_columns_union):
        present_in = [name for name, columns in all_column_sets.items() if column in columns]
        missing_from = [name for name, columns in all_column_sets.items() if column not in columns]
        if missing_from:
            print(f"  {column}: PRESENT in {present_in}, MISSING from {missing_from}")
    if all(columns == next(iter(all_column_sets.values())) for columns in all_column_sets.values()):
        print("  All datasets have IDENTICAL columns.")

    # ── 3. Event count ───────────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("  Event counts")
    print(f"{'=' * 80}")
    for dataset_name, dataframe in datasets.items():
        print(f"  {dataset_name}: {len(dataframe)} events")

    # ── 4. Per-event track count distribution ────────────────────────────────
    track_count_summaries = {}
    for dataset_name, dataframe in datasets.items():
        # Use the stored event_n_tracks if available; otherwise count from list length
        if TRACK_COUNT_COLUMN in dataframe.columns:
            track_counts = dataframe[TRACK_COUNT_COLUMN].values.astype(np.float64)
        else:
            track_counts = compute_per_event_list_lengths(dataframe["track_pt"]).astype(np.float64)
        track_count_summaries[dataset_name] = distribution_summary(track_counts)

    print_comparison_table("Per-event track count (event_n_tracks)", track_count_summaries)

    # Also verify that list lengths match event_n_tracks
    print("\n  Verifying list lengths match event_n_tracks ...")
    for dataset_name, dataframe in datasets.items():
        actual_lengths = compute_per_event_list_lengths(dataframe["track_pt"])
        stored_counts = dataframe[TRACK_COUNT_COLUMN].values
        mismatches = int(np.sum(actual_lengths != stored_counts))
        print(f"    {dataset_name}: {mismatches} mismatches out of {len(dataframe)} events")

    # ── 5. Ground-truth track count per event ────────────────────────────────
    ground_truth_count_summaries = {}
    for dataset_name, dataframe in datasets.items():
        ground_truth_counts = compute_per_event_signal_count(dataframe[LABEL_COLUMN]).astype(np.float64)
        ground_truth_count_summaries[dataset_name] = distribution_summary(ground_truth_counts)

    print_comparison_table("Per-event GT track count (track_label_from_tau == 1)", ground_truth_count_summaries)

    # ── 6. Signal-to-noise ratio (GT tracks / total tracks) ──────────────────
    signal_ratio_summaries = {}
    for dataset_name, dataframe in datasets.items():
        ground_truth_counts = compute_per_event_signal_count(dataframe[LABEL_COLUMN]).astype(np.float64)
        total_counts = dataframe[TRACK_COUNT_COLUMN].values.astype(np.float64)
        # Avoid division by zero for events with zero tracks
        safe_total = np.where(total_counts > 0, total_counts, 1.0)
        signal_ratio = ground_truth_counts / safe_total
        signal_ratio_summaries[dataset_name] = distribution_summary(signal_ratio)

    print_comparison_table("Signal-to-noise ratio (GT / total tracks per event)", signal_ratio_summaries)

    # ── 7. Events with zero GT tracks ────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("  Events with zero GT tracks")
    print(f"{'=' * 80}")
    for dataset_name, dataframe in datasets.items():
        ground_truth_counts = compute_per_event_signal_count(dataframe[LABEL_COLUMN])
        zero_gt_count = int(np.sum(ground_truth_counts == 0))
        fraction = zero_gt_count / len(dataframe) * 100.0
        print(f"  {dataset_name}: {zero_gt_count} / {len(dataframe)} ({fraction:.2f}%)")

    # ── 8. Feature distributions ─────────────────────────────────────────────
    for feature_name in KEY_FEATURES:
        feature_summaries = {}
        for dataset_name, dataframe in datasets.items():
            if feature_name not in dataframe.columns:
                continue
            flat_values = flatten_list_column(dataframe[feature_name])
            feature_summaries[dataset_name] = distribution_summary(flat_values)
        print_comparison_table(f"Feature distribution: {feature_name} (all tracks, flattened)", feature_summaries)

    # ── 9. Feature distributions for GT tracks only ──────────────────────────
    print(f"\n{'=' * 80}")
    print("  Feature distributions for GT tracks only (track_label_from_tau == 1)")
    print(f"{'=' * 80}")
    for feature_name in KEY_FEATURES:
        feature_summaries = {}
        for dataset_name, dataframe in datasets.items():
            if feature_name not in dataframe.columns:
                continue
            # Extract only signal-track values
            signal_values = []
            for feature_row, label_row in zip(dataframe[feature_name], dataframe[LABEL_COLUMN]):
                feature_array = np.array(feature_row, dtype=np.float64)
                label_array = np.array(label_row)
                signal_mask = label_array == 1
                if signal_mask.any():
                    signal_values.append(feature_array[signal_mask])
            if signal_values:
                all_signal = np.concatenate(signal_values)
                feature_summaries[dataset_name] = distribution_summary(all_signal)
        if feature_summaries:
            print_comparison_table(
                f"GT-only distribution: {feature_name}",
                feature_summaries,
            )

    # ── 10. Track count histogram buckets ────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("  Track count histogram (% of events in each bucket)")
    print(f"{'=' * 80}")

    bucket_edges = [0, 100, 500, 1000, 1500, 2000, 2500, 3000, np.inf]
    bucket_labels = []
    for i in range(len(bucket_edges) - 1):
        low = int(bucket_edges[i])
        high = int(bucket_edges[i + 1]) if bucket_edges[i + 1] != np.inf else "inf"
        bucket_labels.append(f"{low}-{high}")

    dataset_names = list(datasets.keys())
    header = f"{'bucket':<12}" + "".join(f"{name:>18}" for name in dataset_names)
    print(header)
    print("-" * len(header))

    for i in range(len(bucket_edges) - 1):
        row = f"{bucket_labels[i]:<12}"
        for dataset_name, dataframe in datasets.items():
            track_counts = dataframe[TRACK_COUNT_COLUMN].values
            in_bucket = np.sum((track_counts >= bucket_edges[i]) & (track_counts < bucket_edges[i + 1]))
            fraction = in_bucket / len(dataframe) * 100.0
            row += f"{fraction:>17.2f}%"
        print(row)

    # ── 11. GT track count histogram buckets ─────────────────────────────────
    print(f"\n{'=' * 80}")
    print("  GT track count histogram (% of events in each bucket)")
    print(f"{'=' * 80}")

    gt_bucket_edges = [0, 1, 5, 10, 20, 30, 50, 100, np.inf]
    gt_bucket_labels = []
    for i in range(len(gt_bucket_edges) - 1):
        low = int(gt_bucket_edges[i])
        high = int(gt_bucket_edges[i + 1]) if gt_bucket_edges[i + 1] != np.inf else "inf"
        gt_bucket_labels.append(f"{low}-{high}")

    header = f"{'bucket':<12}" + "".join(f"{name:>18}" for name in dataset_names)
    print(header)
    print("-" * len(header))

    for i in range(len(gt_bucket_edges) - 1):
        row = f"{gt_bucket_labels[i]:<12}"
        for dataset_name, dataframe in datasets.items():
            ground_truth_counts = compute_per_event_signal_count(dataframe[LABEL_COLUMN])
            in_bucket = np.sum(
                (ground_truth_counts >= gt_bucket_edges[i]) & (ground_truth_counts < gt_bucket_edges[i + 1])
            )
            fraction = in_bucket / len(dataframe) * 100.0
            row += f"{fraction:>17.2f}%"
        print(row)

    print(f"\n{'=' * 80}")
    print("  Analysis complete.")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
