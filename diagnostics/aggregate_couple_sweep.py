"""Aggregate one ``top_k2`` sweep into a JSON + Markdown summary.

The companion launcher (``sweep_topk2.sh``) creates a sweep root that
looks like::

    experiments/topk2_sweep_<timestamp>/
        topk2_20/
            topk2_20_CoupleReranker_<run_ts>/
                loss_history.json
                metrics/
                checkpoints/
        topk2_30/
            topk2_30_CoupleReranker_<run_ts>/
                ...
        ...

This script walks every ``topk2_<K>`` immediate subdirectory, finds the
trainer's experiment dir inside it, reads ``loss_history.json``, picks
the epoch with the best ``val_c_at_100_couples``, and produces:

- ``sweep_summary.json`` — machine-readable: full per-run metrics at
  the best epoch, sweep-level metadata, run statuses
- ``sweep_summary.md`` — human-readable headline table + detailed
  per-K table for the winning run + failed-run list

Usage::

    python diagnostics/aggregate_couple_sweep.py experiments/topk2_sweep_20260408_233000

Idempotent: safe to re-run while the sweep is still in progress to get
a partial summary.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# The trainer always picks the best checkpoint by C@100_couples, so we
# use the same key as the per-run sort criterion.
DEFAULT_CRITERION_KEY = 'val_c_at_100_couples'


# ---------------------------------------------------------------------------
# Subrun discovery + parsing
# ---------------------------------------------------------------------------

def parse_top_k2_from_subrun(subrun_dir: Path) -> int | None:
    """Extract the top_k2 value from a directory name like ``topk2_50``.

    Returns ``None`` for any directory that does not match the expected
    naming convention so the aggregator can safely walk a sweep root
    that contains other artifacts (logs/, sweep_summary/, etc.).
    """
    name = subrun_dir.name
    if not name.startswith('topk2_'):
        return None
    suffix = name[len('topk2_'):]
    if not suffix:
        return None
    try:
        return int(suffix)
    except ValueError:
        return None


def find_subrun_experiment_dir(subrun_dir: Path) -> Path | None:
    """Locate the trainer's experiment directory inside a subrun.

    The trainer creates ``<experiments_dir>/<model_name>_<timestamp>/``,
    so for each subrun directory we look at its child directories and
    pick the most recently modified one that actually contains a
    ``loss_history.json`` (the file the aggregator depends on).
    """
    if not subrun_dir.is_dir():
        return None
    candidates = [child for child in subrun_dir.iterdir() if child.is_dir()]
    if not candidates:
        return None
    # Most recently modified first → if the user re-ran the same K, the
    # latest run wins.
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    for candidate in candidates:
        if (candidate / 'loss_history.json').exists():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Loss history parsing (handles labelled-dict and bare-list formats)
# ---------------------------------------------------------------------------

def _extract_values(entry) -> list[float]:
    """Pull the per-epoch values out of a loss_history entry.

    Two on-disk formats:
        ``{'label': str, 'values': [...]}`` (current) or bare ``[...]``
        (older). Returns an empty list for anything else.
    """
    if isinstance(entry, dict) and 'values' in entry:
        values = entry['values']
        if isinstance(values, list):
            return values
        return []
    if isinstance(entry, list):
        return entry
    return []


def find_best_epoch(
    loss_history: dict,
    criterion_key: str = DEFAULT_CRITERION_KEY,
) -> dict | None:
    """Pick the epoch with the highest ``criterion_key`` value.

    Returns a dict with the best epoch index (0-indexed), the
    1-indexed epoch number, the criterion value, and every other metric
    sampled at that epoch. Returns ``None`` if the criterion key is
    missing or empty (e.g., crashed before first validation).
    """
    criterion_values = _extract_values(loss_history.get(criterion_key))
    if not criterion_values:
        return None
    best_index = max(
        range(len(criterion_values)),
        key=lambda index: criterion_values[index],
    )
    metrics_at_best: dict[str, float] = {}
    for key, entry in loss_history.items():
        values = _extract_values(entry)
        if values and best_index < len(values):
            metrics_at_best[key] = values[best_index]
    return {
        'best_epoch_index': best_index,
        'best_epoch_number': best_index + 1,
        'criterion_key': criterion_key,
        'criterion_value': criterion_values[best_index],
        'metrics_at_best_epoch': metrics_at_best,
    }


def load_loss_history(experiment_dir: Path) -> dict | None:
    """Load and parse loss_history.json. Returns None on missing/invalid."""
    path = experiment_dir / 'loss_history.json'
    if not path.exists():
        return None
    try:
        with open(path) as file_handle:
            return json.load(file_handle)
    except (OSError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------------------
# Sweep aggregation
# ---------------------------------------------------------------------------

def aggregate_sweep(
    sweep_root: Path,
    criterion_key: str = DEFAULT_CRITERION_KEY,
) -> dict:
    """Walk a sweep root, collect best results from every subrun.

    Args:
        sweep_root: Directory created by ``sweep_topk2.sh``.
        criterion_key: Loss-history key to argmax for selecting the
            best epoch in each subrun.

    Returns:
        Dict containing sweep metadata and a list of run results,
        sorted by ``top_k2`` ascending. Each run is one of:
            - ``status='OK'`` with all best-epoch metrics
            - ``status='FAILED'`` with an ``error`` field describing why
    """
    runs: list[dict] = []
    if sweep_root.is_dir():
        for subrun_dir in sorted(sweep_root.iterdir()):
            top_k2 = parse_top_k2_from_subrun(subrun_dir)
            if top_k2 is None:
                continue
            run_info: dict = {
                'top_k2': top_k2,
                'subrun_dir': subrun_dir.name,
            }
            experiment_dir = find_subrun_experiment_dir(subrun_dir)
            if experiment_dir is None:
                run_info['status'] = 'FAILED'
                run_info['error'] = 'no experiment directory found in subrun'
                runs.append(run_info)
                continue
            run_info['experiment_dir'] = str(
                experiment_dir.relative_to(sweep_root),
            )
            loss_history = load_loss_history(experiment_dir)
            if loss_history is None:
                run_info['status'] = 'FAILED'
                run_info['error'] = 'loss_history.json missing or invalid'
                runs.append(run_info)
                continue
            best = find_best_epoch(loss_history, criterion_key=criterion_key)
            if best is None:
                run_info['status'] = 'FAILED'
                run_info['error'] = (
                    f'criterion {criterion_key} not present in loss_history '
                    f'(no completed validation epochs?)'
                )
                runs.append(run_info)
                continue
            run_info['status'] = 'OK'
            run_info.update(best)
            runs.append(run_info)
    runs.sort(key=lambda r: r['top_k2'])
    return {
        'sweep_root': str(sweep_root),
        'aggregated_at': datetime.now().isoformat(timespec='seconds'),
        'criterion_key': criterion_key,
        'num_runs': len(runs),
        'num_ok': sum(1 for run in runs if run['status'] == 'OK'),
        'num_failed': sum(1 for run in runs if run['status'] == 'FAILED'),
        'runs': runs,
    }


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def _extract_k_couples_from_metrics(metrics: dict) -> list[int]:
    """Pull every K_couples value present in a metrics dict."""
    k_values: list[int] = []
    for key in metrics:
        if key.startswith('val_c_at_') and key.endswith('_couples'):
            middle = key[len('val_c_at_'):-len('_couples')]
            try:
                k_values.append(int(middle))
            except ValueError:
                continue
    return sorted(set(k_values))


def _extract_k_tracks_from_metrics(metrics: dict) -> list[int]:
    """Pull every K_tracks value present in a metrics dict."""
    k_values: list[int] = []
    for key in metrics:
        if key.startswith('val_d_at_') and key.endswith('_tracks'):
            middle = key[len('val_d_at_'):-len('_tracks')]
            try:
                k_values.append(int(middle))
            except ValueError:
                continue
    return sorted(set(k_values))


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a GitHub-flavored markdown table."""
    header_line = '| ' + ' | '.join(headers) + ' |'
    separator = '|' + '|'.join('---:' for _ in headers) + '|'
    body = '\n'.join('| ' + ' | '.join(row) + ' |' for row in rows)
    return '\n'.join([header_line, separator, body]) if rows else (
        '\n'.join([header_line, separator])
    )


def format_summary_markdown(sweep: dict) -> str:
    """Render the sweep summary as Markdown.

    Sections:
        1. Sweep metadata
        2. Headline table (one row per top_k2, key columns)
        3. Per-K detailed table for the winning run
        4. Failed-run list (if any)
    """
    lines: list[str] = []
    lines.append('# Couple Reranker top_k2 Sweep Summary')
    lines.append('')
    lines.append(f"**Sweep root:** `{sweep['sweep_root']}`")
    lines.append(f"**Aggregated:** {sweep['aggregated_at']}")
    lines.append(f"**Criterion:** `{sweep['criterion_key']}`")
    lines.append(
        f"**Runs:** {sweep['num_ok']} OK, {sweep['num_failed']} failed, "
        f"{sweep['num_runs']} total",
    )
    lines.append('')

    ok_runs = [run for run in sweep['runs'] if run['status'] == 'OK']
    failed_runs = [run for run in sweep['runs'] if run['status'] == 'FAILED']

    if ok_runs:
        # Identify the winning run for highlighting
        winning_run = max(ok_runs, key=lambda r: r['criterion_value'])
        winning_top_k2 = winning_run['top_k2']

        lines.append(
            f"**Best run:** `top_k2={winning_top_k2}` with "
            f"**C@100c = {winning_run['criterion_value']:.4f}** "
            f"at epoch {winning_run['best_epoch_number']}",
        )
        lines.append('')

        # ---- Headline table: per top_k2 ----
        lines.append('## Headline: best epoch per top_k2')
        lines.append('')
        headers = [
            'top_k2', 'best_epoch',
            'C@50c', 'C@100c', 'C@200c',
            'RC@100c', 'RC@200c',
            'mean_rank', 'eligible/total', 'full_triplet',
        ]
        rows: list[list[str]] = []
        for run in ok_runs:
            metrics = run['metrics_at_best_epoch']
            marker = ' ★' if run['top_k2'] == winning_top_k2 else ''
            eligible = int(metrics.get('val_eligible_events', 0))
            total = int(metrics.get('val_total_events', 0))
            full_triplet = int(metrics.get('val_events_with_full_triplet', 0))
            rows.append([
                f"**{run['top_k2']}**{marker}",
                str(run['best_epoch_number']),
                f"{metrics.get('val_c_at_50_couples', 0.0):.4f}",
                f"{metrics.get('val_c_at_100_couples', 0.0):.4f}",
                f"{metrics.get('val_c_at_200_couples', 0.0):.4f}",
                f"{metrics.get('val_rc_at_100_couples', 0.0):.4f}",
                f"{metrics.get('val_rc_at_200_couples', 0.0):.4f}",
                f"{metrics.get('val_mean_first_gt_rank_couples', 0.0):.1f}",
                f"{eligible} / {total}",
                str(full_triplet),
            ])
        lines.append(_markdown_table(headers, rows))
        lines.append('')

        # ---- Detailed per-K table for the winning run ----
        winning_metrics = winning_run['metrics_at_best_epoch']
        k_couples = _extract_k_couples_from_metrics(winning_metrics)
        k_tracks = _extract_k_tracks_from_metrics(winning_metrics)
        all_k = sorted(set(k_couples) | set(k_tracks))

        lines.append(f'## Detailed per-K table — best run (top_k2={winning_top_k2})')
        lines.append('')
        detail_headers = ['K', 'D@K_tracks', 'C@K_couples', 'RC@K_couples']
        detail_rows: list[list[str]] = []
        for k in all_k:
            d_value = winning_metrics.get(f'val_d_at_{k}_tracks')
            c_value = winning_metrics.get(f'val_c_at_{k}_couples')
            rc_value = winning_metrics.get(f'val_rc_at_{k}_couples')
            detail_rows.append([
                str(k),
                f'{d_value:.4f}' if d_value is not None else '—',
                f'{c_value:.4f}' if c_value is not None else '—',
                f'{rc_value:.4f}' if rc_value is not None else '—',
            ])
        lines.append(_markdown_table(detail_headers, detail_rows))
        lines.append('')

        # ---- C@K_couples curves across all top_k2 (one row per K) ----
        if k_couples:
            lines.append('## C@K_couples by top_k2 (one column per run)')
            lines.append('')
            curve_headers = ['K_couples'] + [
                f"top_k2={r['top_k2']}" for r in ok_runs
            ]
            curve_rows: list[list[str]] = []
            for k in k_couples:
                row = [str(k)]
                for run in ok_runs:
                    value = run['metrics_at_best_epoch'].get(
                        f'val_c_at_{k}_couples',
                    )
                    row.append(f'{value:.4f}' if value is not None else '—')
                curve_rows.append(row)
            lines.append(_markdown_table(curve_headers, curve_rows))
            lines.append('')

    if failed_runs:
        lines.append('## Failed runs')
        lines.append('')
        for run in failed_runs:
            lines.append(
                f"- `top_k2={run['top_k2']}` ({run['subrun_dir']}): "
                f"FAILED — {run.get('error', 'unknown error')}",
            )
        lines.append('')

    if not ok_runs and not failed_runs:
        lines.append('_(No runs found in this sweep root yet.)_')
        lines.append('')

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description='Aggregate a top_k2 sweep into JSON + Markdown summaries.',
    )
    parser.add_argument(
        'sweep_root',
        type=str,
        help='Path to the sweep root directory (contains topk2_<K>/ subdirs).',
    )
    parser.add_argument(
        '--criterion',
        type=str,
        default=DEFAULT_CRITERION_KEY,
        help=(
            'Loss-history key to argmax for the best epoch '
            f'(default: {DEFAULT_CRITERION_KEY}).'
        ),
    )
    parser.add_argument(
        '--output-prefix',
        type=str,
        default='sweep_summary',
        help='Prefix for the output files (default: sweep_summary).',
    )
    args = parser.parse_args(argv)

    sweep_root = Path(args.sweep_root)
    if not sweep_root.exists():
        print(f'ERROR: sweep root does not exist: {sweep_root}', file=sys.stderr)
        sys.exit(1)

    summary = aggregate_sweep(sweep_root, criterion_key=args.criterion)

    json_path = sweep_root / f'{args.output_prefix}.json'
    md_path = sweep_root / f'{args.output_prefix}.md'

    with open(json_path, 'w') as file_handle:
        json.dump(summary, file_handle, indent=2)
    with open(md_path, 'w') as file_handle:
        file_handle.write(format_summary_markdown(summary))

    print(f'Wrote: {json_path}')
    print(f'Wrote: {md_path}')
    print(
        f'Sweep: {summary["num_ok"]} OK, '
        f'{summary["num_failed"]} failed, '
        f'{summary["num_runs"]} total',
    )


if __name__ == '__main__':
    main()
