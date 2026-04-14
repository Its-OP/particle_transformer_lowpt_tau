"""Compute C@K and RC@K metrics from the eval parquet output joined
with input GT labels. No model loading — pure parquet post-processing.

Joins the eval output (couples + remaining_pions as track indices) with
the input validation parquet (track_label_from_tau) on the event key
``(event_run, event_id, event_luminosity_block)``. For each event,
finds the rank of the best GT couple in the reranker output, then
aggregates C@K and RC@K at configurable K cutoffs.

Usage::

    python diagnostics/compute_couple_metrics.py \\
        --eval-parquet data/low-pt/eval/couple_reranker_val.parquet \\
        --input-dir data/low-pt/val/ \\
        --k-values 1 3 5 10 20 30 50 75 100 150 200
"""
from __future__ import annotations

import argparse
import os
import sys
from itertools import combinations

import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Per-event metric computation
# ---------------------------------------------------------------------------

def compute_event_metrics(
    gt_pion_pt: list[float],
    couples: list[list[float]],
    remaining: list[float],
) -> dict:
    """Compute per-event metrics from GT labels and reranker output.

    Args:
        gt_pion_pt: pT values of GT pions in the event.
        couples: Reranker output — list of ``[pt_i, pt_j]`` pairs,
            sorted by score descending.
        remaining: Stage 1 top-K1 track pT values.

    Returns:
        Dict with ``best_rank`` (1-indexed, or None if no GT couple
        found), ``eligible`` (≥1 GT couple has both tracks in
        remaining), ``full_triplet`` (all GT pions in remaining).
    """
    gt_set = set(gt_pion_pt)
    remaining_set = set(remaining)

    # GT couples: all pairs of GT pion indices
    gt_couples = {
        frozenset(pair) for pair in combinations(gt_set, 2)
    }

    # Full triplet: all GT pions survived Stage 1
    # For events with <3 GT pions, full_triplet is always False
    full_triplet = len(gt_set) >= 3 and gt_set.issubset(remaining_set)

    # Eligible: at least one GT couple has both tracks in remaining
    eligible = any(
        couple.issubset(remaining_set) for couple in gt_couples
    )

    # Find the rank of the first (best) GT couple in the output
    best_rank = None
    if eligible:
        for rank_zero_indexed, (idx_i, idx_j) in enumerate(couples):
            if frozenset({idx_i, idx_j}) in gt_couples:
                best_rank = rank_zero_indexed + 1  # 1-indexed
                break

    return {
        'best_rank': best_rank,
        'eligible': eligible,
        'full_triplet': full_triplet,
    }


# ---------------------------------------------------------------------------
# Multi-event aggregation
# ---------------------------------------------------------------------------

def aggregate_metrics(
    event_results: list[dict],
    k_values: list[int],
) -> dict:
    """Aggregate per-event results into C@K, RC@K, and bookkeeping.

    Args:
        event_results: List of dicts from ``compute_event_metrics``.
        k_values: K cutoff values for the table rows.

    Returns:
        Dict with ``c_at_K``, ``rc_at_K`` for each K, plus
        ``mean_rank``, ``eligible``, ``total``, ``full_triplet_count``,
        ``duplet_in_stage1``.
    """
    total = len(event_results)
    if total == 0:
        metrics: dict = {f'c_at_{k}': 0.0 for k in k_values}
        metrics.update({f'rc_at_{k}': 0.0 for k in k_values})
        metrics.update({
            'mean_rank': 0.0, 'eligible': 0, 'total': 0,
            'full_triplet_count': 0, 'duplet_in_stage1': 0,
        })
        return metrics

    eligible_count = sum(1 for r in event_results if r['eligible'])
    full_triplet_count = sum(1 for r in event_results if r['full_triplet'])
    # Duplet in Stage 1: events where ≥2 GT pions are in remaining
    # (this is the same as eligible — both tracks of a GT couple must
    # be in remaining for the event to be eligible)
    duplet_in_stage1 = eligible_count

    # Mean rank of best GT couple (over eligible events with a found rank)
    ranks = [r['best_rank'] for r in event_results if r['best_rank'] is not None]
    mean_rank = sum(ranks) / len(ranks) if ranks else 0.0

    metrics = {}
    for k in k_values:
        c_count = sum(
            1 for r in event_results
            if r['best_rank'] is not None and r['best_rank'] <= k
        )
        rc_count = sum(
            1 for r in event_results
            if r['best_rank'] is not None and r['best_rank'] <= k
            and r['full_triplet']
        )
        metrics[f'c_at_{k}'] = c_count / total
        metrics[f'rc_at_{k}'] = rc_count / total

    metrics['mean_rank'] = mean_rank
    metrics['eligible'] = eligible_count
    metrics['total'] = total
    metrics['full_triplet_count'] = full_triplet_count
    metrics['duplet_in_stage1'] = duplet_in_stage1

    return metrics


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

def format_metrics_table(
    metrics: dict,
    k_values: list[int],
) -> str:
    """Render C@K / RC@K as an ASCII table."""
    col_widths = (5, 8, 8)
    headers = ('K', 'C@K', 'RC@K')

    def _row(values: tuple) -> str:
        cells = [str(v).center(w) for v, w in zip(values, col_widths)]
        return '|' + '|'.join(cells) + '|'

    def _sep() -> str:
        return '+' + '+'.join('-' * w for w in col_widths) + '+'

    lines = [
        f'Couple Reranker Metrics ({metrics["total"]:,} events)',
        '',
        _sep(), _row(headers), _sep(),
    ]
    for k in k_values:
        c = metrics.get(f'c_at_{k}', 0.0)
        rc = metrics.get(f'rc_at_{k}', 0.0)
        lines.append(_row((str(k), f'{c:.4f}', f'{rc:.4f}')))
    lines.append(_sep())

    eligible = metrics['eligible']
    total = metrics['total']
    full_triplet = metrics['full_triplet_count']
    mean_rank = metrics['mean_rank']
    lines.append(
        f'eligible: {eligible} / {total} | '
        f'full_triplet: {full_triplet} | '
        f'mean_rank: {mean_rank:.1f}',
    )
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Data loading + joining
# ---------------------------------------------------------------------------

def load_eval_results(eval_parquet_path: str) -> list[dict]:
    """Load eval parquet into a list of per-event dicts.

    The eval parquet contains ``gt_pion_pt`` alongside the model
    output so no join with the input data is needed (the DataLoader
    reorders events, making key-based joins unreliable for datasets
    with non-unique ``(run, event_id, lumi_block)`` keys).
    """
    table = pq.read_table(eval_parquet_path)
    results = []
    has_gt = 'gt_pion_pt' in table.schema.names
    for i in range(table.num_rows):
        entry: dict = {
            'couples': table.column('couple_pt')[i].as_py(),
            'remaining_pions': table.column('remaining_pion_pt')[i].as_py(),
        }
        if has_gt:
            entry['gt_pion_pt'] = table.column('gt_pion_pt')[i].as_py()
        results.append(entry)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description='Compute C@K and RC@K from eval parquet + input GT labels.',
    )
    parser.add_argument(
        '--eval-parquet', type=str, required=True,
        help='Path to the eval output parquet (must contain gt_pion_pt).',
    )
    parser.add_argument(
        '--k-values', type=int, nargs='+',
        default=[1, 3, 5, 10, 20, 30, 50, 75, 100, 150, 200],
    )
    args = parser.parse_args(argv)

    print(f'Loading eval results from {args.eval_parquet}...')
    eval_results = load_eval_results(args.eval_parquet)
    print(f'  {len(eval_results)} events')

    if not eval_results or 'gt_pion_pt' not in eval_results[0]:
        print('ERROR: eval parquet missing gt_pion_pt column.')
        print('Re-run eval_couple_reranker.py to regenerate.')
        sys.exit(1)

    # Compute per-event metrics
    event_metrics_list = []
    for event in eval_results:
        result = compute_event_metrics(
            event['gt_pion_pt'],
            event['couples'],
            event['remaining_pions'],
        )
        event_metrics_list.append(result)

    # Aggregate
    metrics = aggregate_metrics(event_metrics_list, k_values=args.k_values)

    # Print
    print()
    print(format_metrics_table(metrics, args.k_values))


if __name__ == '__main__':
    main()
