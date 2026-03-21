"""BDT ceiling test: per-track classification without kNN context.

Answers two questions:
1. What's the ceiling for per-track features alone? (BDT vs NN comparison)
2. What's R@K for K=200,300,400,500? (cascade viability diagnostic)

Uses sklearn HistGradientBoostingClassifier on flattened per-track features.
Evaluates per-event R@K by ranking tracks within each event by BDT score.
"""

import sys
import numpy as np
import pyarrow.parquet as pq
from collections import defaultdict

# ---- Load data ----
print("Loading data...")
train_files = [f"data/low-pt/val/val_0{i}.parquet" for i in range(5)]
test_files = [f"data/low-pt/val/val_0{i}.parquet" for i in range(5, 10)]

feature_columns = [
    "track_pt", "track_eta", "track_phi", "track_charge",
    "track_dxy_significance", "track_dz_significance",
    "track_pt_error", "track_n_valid_pixel_hits",
    "track_dca_significance", "track_covariance_phi_phi",
    "track_covariance_lambda_lambda",
]

# Check if norm_chi2 exists
schema = pq.read_schema(train_files[0])
if "track_norm_chi2" in schema.names:
    feature_columns.append("track_norm_chi2")
    print(f"  Using {len(feature_columns)} features (including norm_chi2)")
else:
    print(f"  Using {len(feature_columns)} features (no norm_chi2)")

all_columns = feature_columns + ["track_label_from_tau"]


def load_events(file_list):
    """Load parquet files, return list of per-event dicts."""
    events = []
    for filepath in file_list:
        table = pq.read_table(filepath, columns=all_columns)
        num_events = len(table)
        for event_index in range(num_events):
            features = {}
            for column_name in feature_columns:
                values = table[column_name][event_index].as_py()
                features[column_name] = np.array(values, dtype=np.float32)
            labels = np.array(
                table["track_label_from_tau"][event_index].as_py(),
                dtype=np.float32,
            )
            # Filter padding (pt == 0)
            valid = features["track_pt"] > 0
            if valid.sum() == 0:
                continue
            event = {
                "features": np.column_stack(
                    [features[column_name][valid] for column_name in feature_columns]
                ),
                "labels": labels[valid],
                "num_tracks": int(valid.sum()),
                "num_gt": int(labels[valid].sum()),
            }
            if event["num_gt"] > 0:
                events.append(event)
    return events


print("Loading train events...")
train_events = load_events(train_files)
print(f"  {len(train_events)} events with GT tracks")

print("Loading test events...")
test_events = load_events(test_files)
print(f"  {len(test_events)} events with GT tracks")

# ---- Flatten for BDT training ----
print("\nFlattening tracks for BDT...")
train_features = np.concatenate([event["features"] for event in train_events])
train_labels = np.concatenate([event["labels"] for event in train_events])

num_positive = train_labels.sum()
num_negative = len(train_labels) - num_positive
imbalance_ratio = num_negative / max(num_positive, 1)
print(f"  {len(train_labels):,} tracks ({int(num_positive):,} positive, "
      f"ratio 1:{imbalance_ratio:.0f})")

# Sample weights for class imbalance
sample_weights = np.where(train_labels == 1, imbalance_ratio, 1.0)

# ---- Train BDT ----
print("\nTraining HistGradientBoostingClassifier...")
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

classifier = HistGradientBoostingClassifier(
    max_iter=500,
    max_depth=6,
    learning_rate=0.05,
    min_samples_leaf=50,
    random_state=42,
    verbose=0,
)
classifier.fit(train_features, train_labels, sample_weight=sample_weights)
print("  Done.")

# ---- Evaluate per-event R@K ----
print("\nEvaluating per-event R@K on test set...")
k_values = [10, 20, 30, 100, 200, 300, 400, 500, 600]
recall_sums = {k: 0.0 for k in k_values}
perfect_sums = {k: 0 for k in k_values}
all_gt_ranks = []
num_test_events = 0

for event in test_events:
    scores = classifier.predict_proba(event["features"])[:, 1]
    labels = event["labels"]
    num_tracks = event["num_tracks"]

    # Rank by score descending
    ranked_indices = np.argsort(-scores)
    gt_positions = np.where(labels == 1)[0]
    num_gt = len(gt_positions)
    if num_gt == 0:
        continue

    num_test_events += 1

    # GT ranks
    rank_lookup = np.argsort(np.argsort(-scores))
    gt_ranks = rank_lookup[gt_positions]
    all_gt_ranks.extend(gt_ranks.tolist())

    for k in k_values:
        actual_k = min(k, num_tracks)
        top_k_set = set(ranked_indices[:actual_k])
        found = sum(1 for gt_pos in gt_positions if gt_pos in top_k_set)
        recall_sums[k] += found / num_gt
        if found == num_gt:
            perfect_sums[k] += 1

# ---- Results ----
print(f"\n{'='*60}")
print(f"BDT CEILING TEST ({num_test_events} test events)")
print(f"{'='*60}")
print(f"{'K':<8} {'R@K':>8} {'P@K':>8}")
print(f"{'-'*28}")
for k in k_values:
    recall = recall_sums[k] / num_test_events
    perfect = perfect_sums[k] / num_test_events
    print(f"{k:<8} {recall:>8.4f} {perfect:>8.4f}")

gt_ranks_array = np.array(all_gt_ranks)
print(f"\nGT rank distribution:")
percentiles = np.percentile(gt_ranks_array, [5, 10, 25, 50, 75, 90, 95])
print(f"  p5={percentiles[0]:.0f}  p10={percentiles[1]:.0f}  "
      f"p25={percentiles[2]:.0f}  p50={percentiles[3]:.0f}  "
      f"p75={percentiles[4]:.0f}  p90={percentiles[5]:.0f}  "
      f"p95={percentiles[6]:.0f}")
print(f"  mean={gt_ranks_array.mean():.1f}")

# AUC on flattened test
test_features_flat = np.concatenate([e["features"] for e in test_events])
test_labels_flat = np.concatenate([e["labels"] for e in test_events])
test_scores_flat = classifier.predict_proba(test_features_flat)[:, 1]
auc = roc_auc_score(test_labels_flat, test_scores_flat)
print(f"\nGlobal AUC: {auc:.4f}")

# Feature importances (built-in, not permutation — faster)
print(f"\nFeature importances (built-in):")
importances = classifier.feature_importances_
for feature_index in np.argsort(-importances):
    name = feature_columns[feature_index]
    print(f"  {name:<35} {importances[feature_index]:.4f}")

print(f"\n--- Comparison ---")
print(f"Phase-B NN (kNN):  R@200 = 0.623,  median rank = 111")
print(f"BDT (no kNN):      R@200 = {recall_sums[200]/num_test_events:.3f},  "
      f"median rank = {np.median(gt_ranks_array):.0f}")
