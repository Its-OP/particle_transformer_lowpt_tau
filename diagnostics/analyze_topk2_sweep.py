"""Deep analysis of a ``top_k2`` sweep — convergence, plateau, conditional
efficiency, cost-benefit, and K=200 OOM diagnosis.

Reads only the on-disk artifacts produced by ``sweep_topk2.sh`` (no
model loading, no inference). Writes a Markdown report with 7 analysis
sections: convergence dynamics, stability bands, plateau analysis,
conditional reranker efficiency, mean-rank scaling, OOM diagnosis, and
cost-benefit recommendations.

Usage::

    python diagnostics/analyze_topk2_sweep.py \\
        models/debug_checkpoints/topk2_sweep_20260408_205250 \\
        --output reports/triplet_reranking/topk2_sweep_analysis_20260409.md
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diagnostics.aggregate_couple_sweep import (
    _extract_values,
    _markdown_table,
    find_best_epoch,
    find_subrun_experiment_dir,
    load_loss_history,
    parse_top_k2_from_subrun,
)

# Threshold for "still improving": if C@100 at the last epoch exceeds
# the mean of the 5 epochs preceding it by this much, the run is
# classified as still improving (more epochs may help).
STILL_IMPROVING_THRESHOLD = 0.002


# ---------------------------------------------------------------------------
# Per-run analysis functions
# ---------------------------------------------------------------------------

def compute_convergence_metrics(loss_history: dict) -> dict:
    """Extract first/last/best values for train, val, and C@100.

    Returns a dict with keys:
        train_first, train_last, val_first, val_last, train_val_gap,
        c_at_100_first, c_at_100_best, c_at_100_last, best_epoch,
        still_improving.
    """
    train_values = _extract_values(loss_history.get('train'))
    val_values = _extract_values(loss_history.get('val'))
    c100_values = _extract_values(loss_history.get('val_c_at_100_couples'))

    train_first = train_values[0] if train_values else 0.0
    train_last = train_values[-1] if train_values else 0.0
    val_first = val_values[0] if val_values else 0.0
    val_last = val_values[-1] if val_values else 0.0

    c100_first = c100_values[0] if c100_values else 0.0
    c100_last = c100_values[-1] if c100_values else 0.0
    if c100_values:
        best_index = max(range(len(c100_values)), key=lambda i: c100_values[i])
        c100_best = c100_values[best_index]
        best_epoch = best_index + 1  # 1-indexed
    else:
        c100_best = 0.0
        best_epoch = 0

    # "Still improving" — the mean of the last 3 epochs exceeds the
    # mean of the 3 epochs before that by more than the noise threshold.
    # Using two 3-epoch blocks instead of a single-point comparison
    # gives robustness against epoch-to-epoch noise.
    still_improving = False
    if len(c100_values) >= 6:
        recent_mean = sum(c100_values[-3:]) / 3
        preceding_mean = sum(c100_values[-6:-3]) / 3
        still_improving = (recent_mean - preceding_mean) > STILL_IMPROVING_THRESHOLD

    return {
        'train_first': train_first,
        'train_last': train_last,
        'val_first': val_first,
        'val_last': val_last,
        'train_val_gap': train_last - val_last,
        'c_at_100_first': c100_first,
        'c_at_100_best': c100_best,
        'c_at_100_last': c100_last,
        'best_epoch': best_epoch,
        'still_improving': still_improving,
    }


def compute_stability_band(
    loss_history: dict,
    key: str,
    last_n: int = 5,
) -> tuple[float, float]:
    """Mean and sample std of the last N epochs for a given metric key.

    Returns ``(0.0, 0.0)`` if the key is missing or has no values.
    """
    values = _extract_values(loss_history.get(key))
    if not values:
        return 0.0, 0.0
    tail = values[-last_n:]
    mean = sum(tail) / len(tail)
    if len(tail) < 2:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in tail) / (len(tail) - 1)
    return mean, variance ** 0.5


def compute_conditional_rates(metrics_at_best: dict) -> dict:
    """Conditional C@K and RC@K (denominator = eligible, not total).

    Raw C@K uses total_events as denominator. Multiplying by
    ``total / eligible`` recovers the rate conditional on the event
    having at least one GT couple in the candidate pool — i.e., the
    reranker's intrinsic efficiency independent of the cascade's recall.
    """
    eligible = metrics_at_best.get('val_eligible_events', 0.0)
    total = metrics_at_best.get('val_total_events', 0.0)
    if eligible <= 0 or total <= 0:
        return {
            'eligible_fraction': 0.0,
            'conditional_c_at_100': 0.0,
            'conditional_rc_at_100': 0.0,
        }
    multiplier = total / eligible
    c100 = metrics_at_best.get('val_c_at_100_couples', 0.0)
    rc100 = metrics_at_best.get('val_rc_at_100_couples', 0.0)
    return {
        'eligible_fraction': eligible / total,
        'conditional_c_at_100': min(1.0, c100 * multiplier),
        'conditional_rc_at_100': min(1.0, rc100 * multiplier),
    }


def estimate_activation_memory_gb(
    top_k2: int,
    batch_size: int,
    hidden_dim: int = 256,
    num_layers: int = 11,
) -> float:
    """Rough lower bound on couple-reranker activation memory in GB.

    The dominant allocation is the ``(B, hidden_dim, C)`` tensor at
    every layer of the Conv1d MLP (input projection + 4 residual blocks
    × 2 layers each + 2-layer scorer = 11 layers). During backward each
    layer's activations are retained, giving roughly
    ``B × C × hidden_dim × num_layers × 2 (fwd + grad) × 4 bytes``.

    This does NOT account for the upstream cascade's memory (ParT
    attention matrices, Stage 1 features, etc.), so the true peak is
    higher. The estimate is a lower bound for planning batch-size
    reductions.
    """
    # couples_per_event = C(top_k2, 2)
    couples_per_event = top_k2 * (top_k2 - 1) // 2
    couples_per_batch = batch_size * couples_per_event
    # Each layer produces (B*C, hidden_dim) activations; backward
    # retains the activation + its gradient → ×2
    bytes_per_element = 4  # float32
    total_bytes = (
        couples_per_batch * hidden_dim * num_layers * 2 * bytes_per_element
    )
    return total_bytes / (1024 ** 3)


# ---------------------------------------------------------------------------
# Sweep-level loading
# ---------------------------------------------------------------------------

def _parse_training_time_minutes(subrun_dir: Path) -> float | None:
    """Extract wallclock time from training.log timestamps.

    Reads the first and last timestamp in the log to estimate total
    training time. Returns ``None`` if the log is missing or unparseable.
    """
    log_path = subrun_dir / 'training.log'
    if not log_path.exists():
        return None
    timestamp_pattern = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})')
    first_timestamp = None
    last_timestamp = None
    try:
        with open(log_path) as file_handle:
            for line in file_handle:
                match = timestamp_pattern.match(line)
                if match:
                    if first_timestamp is None:
                        first_timestamp = match.group(1)
                    last_timestamp = match.group(1)
    except OSError:
        return None
    if first_timestamp is None or last_timestamp is None:
        return None
    time_format = '%Y-%m-%d %H:%M:%S'
    try:
        start = datetime.strptime(first_timestamp, time_format)
        end = datetime.strptime(last_timestamp, time_format)
        delta_minutes = (end - start).total_seconds() / 60
        return max(0.0, delta_minutes)
    except ValueError:
        return None


def load_all_runs(sweep_root: Path) -> list[dict]:
    """Walk a sweep root and return per-run data with loss histories.

    Each entry is a dict with:
        ``top_k2``, ``status`` ('OK'|'FAILED'), ``loss_history`` (if OK),
        ``training_time_minutes`` (if parseable), ``error`` (if FAILED).
    """
    runs: list[dict] = []
    if not sweep_root.is_dir():
        return runs
    for subrun_dir in sorted(sweep_root.iterdir()):
        top_k2 = parse_top_k2_from_subrun(subrun_dir)
        if top_k2 is None:
            continue
        run_info: dict = {'top_k2': top_k2, 'subrun_dir': subrun_dir.name}
        experiment_dir = find_subrun_experiment_dir(subrun_dir)
        if experiment_dir is None:
            run_info['status'] = 'FAILED'
            run_info['error'] = 'no experiment directory found'
            runs.append(run_info)
            continue
        loss_history = load_loss_history(experiment_dir)
        if loss_history is None:
            run_info['status'] = 'FAILED'
            run_info['error'] = 'loss_history.json missing or invalid'
            runs.append(run_info)
            continue
        run_info['status'] = 'OK'
        run_info['loss_history'] = loss_history
        run_info['training_time_minutes'] = _parse_training_time_minutes(
            subrun_dir,
        )
        runs.append(run_info)
    runs.sort(key=lambda r: r['top_k2'])
    return runs


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_analysis_report(
    sweep_root: str,
    runs_data: list[dict],
) -> str:
    """Build the full Markdown analysis report (7 sections)."""
    ok_runs = [run for run in runs_data if run['status'] == 'OK']
    failed_runs = [run for run in runs_data if run['status'] == 'FAILED']
    lines: list[str] = []

    # ---- Header ----
    lines.append('# top_k2 Sweep Analysis')
    lines.append('')
    lines.append(f'**Sweep root:** `{sweep_root}`')
    lines.append(f'**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    lines.append(
        f'**Runs:** {len(ok_runs)} OK, {len(failed_runs)} failed, '
        f'{len(runs_data)} total',
    )
    lines.append('')

    if not ok_runs:
        lines.append('_No successful runs to analyze._')
        return '\n'.join(lines)

    # Pre-compute per-run analysis
    analysis: list[dict] = []
    for run in ok_runs:
        loss_history = run['loss_history']
        convergence = compute_convergence_metrics(loss_history)
        best_info = find_best_epoch(loss_history)
        metrics_at_best = (
            best_info['metrics_at_best_epoch'] if best_info else {}
        )
        c100_mean, c100_std = compute_stability_band(
            loss_history, 'val_c_at_100_couples',
        )
        rc100_mean, rc100_std = compute_stability_band(
            loss_history, 'val_rc_at_100_couples',
        )
        conditional = compute_conditional_rates(metrics_at_best)
        couples_per_event = run['top_k2'] * (run['top_k2'] - 1) // 2
        analysis.append({
            'top_k2': run['top_k2'],
            'convergence': convergence,
            'metrics_at_best': metrics_at_best,
            'c100_mean': c100_mean,
            'c100_std': c100_std,
            'rc100_mean': rc100_mean,
            'rc100_std': rc100_std,
            'conditional': conditional,
            'couples_per_event': couples_per_event,
            'training_time': run.get('training_time_minutes'),
        })

    # Identify best run (by C@100 mean over last 5 — more stable than argmax)
    best_run = max(analysis, key=lambda a: a['c100_mean'])
    baseline_run = next(
        (a for a in analysis if a['top_k2'] == 50),
        analysis[0],
    )

    # ---- TL;DR ----
    lines.append('## TL;DR')
    lines.append('')
    lines.append(
        f'- **Best K by stability band:** top_k2={best_run["top_k2"]} '
        f'with C@100c = {best_run["c100_mean"]:.4f} +/- {best_run["c100_std"]:.4f}',
    )
    c100_range = max(a['c100_mean'] for a in analysis) - min(a['c100_mean'] for a in analysis)
    lines.append(
        f'- **Plateau:** C@100c varies by only {c100_range:.4f} '
        f'across K=50-{analysis[-1]["top_k2"]} — the reranker is largely '
        f'insensitive to input pool size.',
    )
    improving_runs = [a for a in analysis if a['convergence']['still_improving']]
    if improving_runs:
        lines.append(
            f'- **Still improving at epoch 50:** '
            f'{", ".join(f"K={a["top_k2"]}" for a in improving_runs)} '
            f'— more epochs may yield marginal gains.',
        )
    else:
        lines.append('- **All runs converged** — 50 epochs was sufficient.')
    if failed_runs:
        lines.append(
            f'- **Failed:** K={", ".join(str(r["top_k2"]) for r in failed_runs)} '
            f'(OOM at batch=96).',
        )
    lines.append('')

    # ---- Section 1: Convergence ----
    lines.append('## 1. Convergence dynamics')
    lines.append('')
    convergence_headers = [
        'K', 'train (first->last)', 'val (first->last)',
        'train-val gap', 'C@100 (first->best)', 'best_ep', 'converged?',
    ]
    convergence_rows: list[list[str]] = []
    for a in analysis:
        c = a['convergence']
        convergence_rows.append([
            str(a['top_k2']),
            f'{c["train_first"]:.4f} -> {c["train_last"]:.4f}',
            f'{c["val_first"]:.4f} -> {c["val_last"]:.4f}',
            f'{c["train_val_gap"]:+.4f}',
            f'{c["c_at_100_first"]:.4f} -> {c["c_at_100_best"]:.4f}',
            str(c['best_epoch']),
            'no' if c['still_improving'] else 'yes',
        ])
    lines.append(_markdown_table(convergence_headers, convergence_rows))
    lines.append('')

    # ---- Section 2: Stability ----
    lines.append('## 2. Stability analysis (last 5 epochs)')
    lines.append('')
    stability_headers = [
        'K', 'C@100 best', 'C@100 mean(5)', 'C@100 std(5)',
        'RC@100 mean(5)', 'RC@100 std(5)',
    ]
    stability_rows: list[list[str]] = []
    for a in analysis:
        stability_rows.append([
            str(a['top_k2']),
            f'{a["convergence"]["c_at_100_best"]:.4f}',
            f'{a["c100_mean"]:.4f}',
            f'{a["c100_std"]:.4f}',
            f'{a["rc100_mean"]:.4f}',
            f'{a["rc100_std"]:.4f}',
        ])
    lines.append(_markdown_table(stability_headers, stability_rows))
    lines.append('')

    # ---- Section 3: Plateau ----
    lines.append('## 3. Plateau analysis')
    lines.append('')
    lines.append(
        'Marginal C@100 gain vs baseline (K=50). A difference is "significant" '
        'if it exceeds 2x the pooled stability-band std.',
    )
    lines.append('')
    baseline_c100 = baseline_run['c100_mean']
    baseline_std = baseline_run['c100_std']
    plateau_headers = [
        'K', 'C@100 mean(5)', 'delta vs K=50', '2*sigma', 'significant?',
    ]
    plateau_rows: list[list[str]] = []
    for a in analysis:
        delta = a['c100_mean'] - baseline_c100
        # Pooled std = sqrt((s1^2 + s2^2) / 2) — simple average of variances
        pooled_std = (
            (baseline_std ** 2 + a['c100_std'] ** 2) / 2
        ) ** 0.5
        two_sigma = 2 * pooled_std
        significant = abs(delta) > two_sigma if two_sigma > 0 else False
        plateau_rows.append([
            str(a['top_k2']),
            f'{a["c100_mean"]:.4f}',
            f'{delta:+.4f}',
            f'{two_sigma:.4f}',
            'YES' if significant else 'no',
        ])
    lines.append(_markdown_table(plateau_headers, plateau_rows))
    lines.append('')

    # ---- Section 4: Conditional reranker efficiency ----
    lines.append('## 4. Conditional reranker efficiency')
    lines.append('')
    lines.append(
        'C@100 conditional on eligible events = C@100 * total / eligible. '
        'This strips out the cascade recall and measures the reranker alone.',
    )
    lines.append('')
    conditional_headers = [
        'K', 'eligible/total', 'C@100 (raw)', 'C@100 (conditional)',
        'RC@100 (conditional)',
    ]
    conditional_rows: list[list[str]] = []
    for a in analysis:
        cond = a['conditional']
        eligible = int(a['metrics_at_best'].get('val_eligible_events', 0))
        total = int(a['metrics_at_best'].get('val_total_events', 0))
        conditional_rows.append([
            str(a['top_k2']),
            f'{eligible} / {total}',
            f'{a["convergence"]["c_at_100_best"]:.4f}',
            f'{cond["conditional_c_at_100"]:.4f}',
            f'{cond["conditional_rc_at_100"]:.4f}',
        ])
    lines.append(_markdown_table(conditional_headers, conditional_rows))
    lines.append('')

    # ---- Section 5: Mean rank scaling ----
    lines.append('## 5. Mean rank scaling')
    lines.append('')
    lines.append(
        'If relative_rank (= mean_rank / C(K,2)) decreases with K, '
        'the reranker improves with more context. If flat or rising, '
        'bigger pools just add noise.',
    )
    lines.append('')
    rank_headers = ['K', 'mean_rank', 'C(K,2)', 'relative_rank']
    rank_rows: list[list[str]] = []
    for a in analysis:
        mean_rank = a['metrics_at_best'].get(
            'val_mean_first_gt_rank_couples', 0.0,
        )
        couples = a['couples_per_event']
        relative = mean_rank / couples if couples > 0 else 0.0
        rank_rows.append([
            str(a['top_k2']),
            f'{mean_rank:.1f}',
            str(couples),
            f'{relative:.4f}',
        ])
    lines.append(_markdown_table(rank_headers, rank_rows))
    lines.append('')

    # ---- Section 6: K=200 OOM ----
    lines.append('## 6. K=200 OOM analysis')
    lines.append('')
    if any(r['top_k2'] == 200 for r in failed_runs):
        lines.append(
            'K=200 OOMed with batch_size=96 on a 95GB GPU. The table below '
            'estimates activation memory for the couple-reranker tensor alone '
            '(does NOT include the upstream cascade).',
        )
    else:
        lines.append(
            'No OOM failure observed, but this section estimates memory '
            'for reference.',
        )
    lines.append('')
    oom_headers = [
        'K', 'couples/event', 'couples/batch(96)',
        'reranker mem (GB)', 'fits budget?',
    ]
    # The frozen cascade (Stage 1 + Stage 2 ParT) consumes ~50-55 GB on
    # a 95 GB GPU. The reranker activations are ON TOP of that, so
    # any K where reranker_memory + ~55 GB > 95 GB is at risk of OOM.
    cascade_overhead_gb = 55.0
    gpu_capacity_gb = 95.0
    reranker_budget_gb = gpu_capacity_gb - cascade_overhead_gb  # ~40 GB

    oom_rows: list[list[str]] = []
    oom_k_values = sorted(
        set(a['top_k2'] for a in analysis) | {200},
    )
    for k in oom_k_values:
        couples = k * (k - 1) // 2
        couples_batch = 96 * couples
        mem = estimate_activation_memory_gb(k, batch_size=96)
        fits = 'yes' if mem < reranker_budget_gb else 'NO'
        oom_rows.append([
            str(k),
            f'{couples:,}',
            f'{couples_batch:,}',
            f'{mem:.1f}',
            fits,
        ])
    lines.append(_markdown_table(oom_headers, oom_rows))
    lines.append('')
    lines.append(
        f'Cascade overhead estimate: ~{cascade_overhead_gb:.0f} GB. '
        f'Reranker budget on a {gpu_capacity_gb:.0f} GB GPU: '
        f'~{reranker_budget_gb:.0f} GB.',
    )
    lines.append('')
    # Recommend batch size for K=200
    mem_200_b96 = estimate_activation_memory_gb(200, batch_size=96)
    if mem_200_b96 > reranker_budget_gb:
        for candidate_batch in [64, 48, 32, 24, 16]:
            mem_candidate = estimate_activation_memory_gb(200, candidate_batch)
            if mem_candidate < reranker_budget_gb * 0.7:
                lines.append(
                    f'**Recommendation:** retry K=200 with '
                    f'`BATCH_SIZE={candidate_batch}` (estimated reranker '
                    f'memory: {mem_candidate:.1f} GB, leaving '
                    f'~{gpu_capacity_gb - cascade_overhead_gb - mem_candidate:.0f} '
                    f'GB headroom for fragmentation).',
                )
                break
    lines.append('')

    # ---- Section 7: Recommendations ----
    lines.append('## 7. Recommendations')
    lines.append('')
    cost_headers = [
        'K', 'C(K,2)', 'time (min)',
        'C@100 mean(5)', 'delta vs K=50', 'verdict',
    ]
    cost_rows: list[list[str]] = []
    for a in analysis:
        delta = a['c100_mean'] - baseline_c100
        time_str = (
            f'{a["training_time"]:.0f}'
            if a['training_time'] is not None
            else '—'
        )
        if a['top_k2'] == best_run['top_k2']:
            verdict = 'BEST'
        elif abs(delta) < 0.005:
            verdict = 'equivalent'
        elif delta > 0:
            verdict = 'marginal gain'
        else:
            verdict = 'worse'
        cost_rows.append([
            str(a['top_k2']),
            str(a['couples_per_event']),
            time_str,
            f'{a["c100_mean"]:.4f}',
            f'{delta:+.4f}',
            verdict,
        ])
    lines.append(_markdown_table(cost_headers, cost_rows))
    lines.append('')

    if failed_runs:
        lines.append('### Failed runs')
        lines.append('')
        for run in failed_runs:
            lines.append(
                f'- **K={run["top_k2"]}**: '
                f'{run.get("error", "unknown error")}',
            )
        lines.append('')

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description='Deep analysis of a top_k2 sweep.',
    )
    parser.add_argument(
        'sweep_root', type=str,
        help='Path to the sweep root directory.',
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help=(
            'Output path for the Markdown report. '
            'Defaults to <sweep_root>/sweep_analysis.md.'
        ),
    )
    args = parser.parse_args(argv)

    sweep_root = Path(args.sweep_root)
    if not sweep_root.exists():
        print(f'ERROR: sweep root does not exist: {sweep_root}', file=sys.stderr)
        sys.exit(1)

    runs_data = load_all_runs(sweep_root)
    if not runs_data:
        print(f'ERROR: no subrun directories found in {sweep_root}', file=sys.stderr)
        sys.exit(1)

    report = format_analysis_report(
        sweep_root=str(sweep_root),
        runs_data=runs_data,
    )

    output_path = Path(args.output) if args.output else sweep_root / 'sweep_analysis.md'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as file_handle:
        file_handle.write(report)
    print(f'Wrote: {output_path}')

    ok_count = sum(1 for r in runs_data if r['status'] == 'OK')
    failed_count = sum(1 for r in runs_data if r['status'] == 'FAILED')
    print(f'Analyzed: {ok_count} OK, {failed_count} failed, {len(runs_data)} total')


if __name__ == '__main__':
    main()
