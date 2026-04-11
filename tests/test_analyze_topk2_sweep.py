"""Tests for diagnostics.analyze_topk2_sweep — deep analysis of a top_k2 sweep.

Reuses the synthetic-sweep fixture builders from test_aggregate_couple_sweep
so we don't duplicate the loss_history file setup.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

# Fixture builder reused from the aggregator tests.
from tests.test_aggregate_couple_sweep import _build_subrun, _write_loss_history

from diagnostics.analyze_topk2_sweep import (
    compute_conditional_rates,
    compute_convergence_metrics,
    compute_stability_band,
    estimate_activation_memory_gb,
    format_analysis_report,
    load_all_runs,
)


# ---------------------------------------------------------------------------
# Helpers for building richer synthetic loss histories
# ---------------------------------------------------------------------------

def _synthetic_loss_history(
    num_epochs: int = 10,
    c_at_100_trajectory: list[float] | None = None,
) -> dict:
    """Build a realistic loss_history dict in the labelled-dict format."""
    if c_at_100_trajectory is None:
        # Monotonically improving then saturating
        c_at_100_trajectory = [
            0.50 + 0.30 * (1 - math.exp(-0.3 * epoch))
            for epoch in range(num_epochs)
        ]
    num_epochs = len(c_at_100_trajectory)
    train_trajectory = [0.20 - 0.12 * (epoch / num_epochs) for epoch in range(num_epochs)]
    val_trajectory = [0.15 - 0.08 * (epoch / num_epochs) for epoch in range(num_epochs)]

    history: dict = {}
    for key, values in [
        ('train', train_trajectory),
        ('val', val_trajectory),
        ('lr', [5e-4 * (1 - epoch / num_epochs) for epoch in range(num_epochs)]),
        ('val_c_at_100_couples', c_at_100_trajectory),
        ('val_c_at_50_couples', [v * 0.92 for v in c_at_100_trajectory]),
        ('val_c_at_200_couples', [min(1.0, v * 1.05) for v in c_at_100_trajectory]),
        ('val_rc_at_100_couples', [v * 0.95 for v in c_at_100_trajectory]),
        ('val_rc_at_200_couples', [v * 0.97 for v in c_at_100_trajectory]),
        ('val_d_at_100_tracks', [0.91] * num_epochs),
        ('val_d_at_200_tracks', [0.96] * num_epochs),
        ('val_mean_first_gt_rank_couples', [50.0 - 2.0 * epoch for epoch in range(num_epochs)]),
        ('val_eligible_events', [1700.0] * num_epochs),
        ('val_total_events', [2000.0] * num_epochs),
        ('val_events_with_full_triplet', [1620.0] * num_epochs),
    ]:
        history[key] = {'label': key, 'values': values}
    return history


# ---------------------------------------------------------------------------
# compute_convergence_metrics
# ---------------------------------------------------------------------------

class TestComputeConvergenceMetrics:
    def test_basic_convergence(self):
        history = _synthetic_loss_history(num_epochs=10)
        result = compute_convergence_metrics(history)
        assert 'train_first' in result
        assert 'train_last' in result
        assert 'val_first' in result
        assert 'val_last' in result
        assert 'c_at_100_first' in result
        assert 'c_at_100_best' in result
        assert 'c_at_100_last' in result
        assert 'best_epoch' in result
        assert 'train_val_gap' in result
        assert 'still_improving' in result
        # Train should decrease
        assert result['train_last'] < result['train_first']
        # C@100 should increase
        assert result['c_at_100_best'] >= result['c_at_100_first']

    def test_still_improving_when_trajectory_rising(self):
        """Steadily improving trajectory → still_improving = True."""
        trajectory = [0.5 + 0.02 * i for i in range(10)]
        history = _synthetic_loss_history(c_at_100_trajectory=trajectory)
        result = compute_convergence_metrics(history)
        assert result['still_improving'] is True

    def test_converged_when_flat(self):
        """Flat trajectory for the last 6+ epochs → still_improving = False."""
        trajectory = [0.5, 0.6, 0.7, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
        history = _synthetic_loss_history(c_at_100_trajectory=trajectory)
        result = compute_convergence_metrics(history)
        assert result['still_improving'] is False

    def test_best_epoch_is_argmax(self):
        trajectory = [0.3, 0.5, 0.9, 0.4, 0.6]
        history = _synthetic_loss_history(c_at_100_trajectory=trajectory)
        result = compute_convergence_metrics(history)
        assert result['best_epoch'] == 3  # 1-indexed

    def test_train_val_gap_sign(self):
        history = _synthetic_loss_history(num_epochs=5)
        result = compute_convergence_metrics(history)
        # Gap = train_last - val_last; sign indicates overfit direction
        assert isinstance(result['train_val_gap'], float)


# ---------------------------------------------------------------------------
# compute_stability_band
# ---------------------------------------------------------------------------

class TestComputeStabilityBand:
    def test_mean_and_std_over_last_n(self):
        values = [0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.81, 0.82, 0.83, 0.84]
        history = _synthetic_loss_history(c_at_100_trajectory=values)
        mean, std = compute_stability_band(history, 'val_c_at_100_couples', last_n=5)
        # Last 5 values: 0.80, 0.81, 0.82, 0.83, 0.84
        assert abs(mean - 0.82) < 1e-6
        assert std > 0  # nonzero since values differ

    def test_last_n_larger_than_epochs_uses_all(self):
        values = [0.50, 0.60, 0.70]
        history = _synthetic_loss_history(c_at_100_trajectory=values)
        mean, std = compute_stability_band(history, 'val_c_at_100_couples', last_n=10)
        assert abs(mean - 0.60) < 1e-6

    def test_missing_key_returns_zeros(self):
        history = _synthetic_loss_history(num_epochs=5)
        mean, std = compute_stability_band(history, 'nonexistent_key', last_n=3)
        assert mean == 0.0
        assert std == 0.0


# ---------------------------------------------------------------------------
# compute_conditional_rates
# ---------------------------------------------------------------------------

class TestComputeConditionalRates:
    def test_conditional_equals_unconditional_when_all_eligible(self):
        metrics = {
            'val_c_at_100_couples': 0.80,
            'val_rc_at_100_couples': 0.75,
            'val_eligible_events': 2000.0,
            'val_total_events': 2000.0,
        }
        result = compute_conditional_rates(metrics)
        assert abs(result['conditional_c_at_100'] - 0.80) < 1e-6
        assert abs(result['conditional_rc_at_100'] - 0.75) < 1e-6

    def test_conditional_higher_when_some_ineligible(self):
        metrics = {
            'val_c_at_100_couples': 0.80,
            'val_rc_at_100_couples': 0.75,
            'val_eligible_events': 1600.0,
            'val_total_events': 2000.0,
        }
        result = compute_conditional_rates(metrics)
        # 0.80 * 2000 / 1600 = 1.00
        assert abs(result['conditional_c_at_100'] - 1.00) < 1e-6
        assert abs(result['conditional_rc_at_100'] - 0.9375) < 1e-6
        assert result['eligible_fraction'] == pytest.approx(0.80)


# ---------------------------------------------------------------------------
# estimate_activation_memory_gb
# ---------------------------------------------------------------------------

class TestEstimateActivationMemory:
    def test_scales_with_k_squared(self):
        mem_50 = estimate_activation_memory_gb(top_k2=50, batch_size=96)
        mem_200 = estimate_activation_memory_gb(top_k2=200, batch_size=96)
        # C(200,2) / C(50,2) = 19900/1225 ≈ 16.2x
        # Memory should scale proportionally
        ratio = mem_200 / mem_50
        assert 10 < ratio < 20

    def test_scales_linearly_with_batch(self):
        mem_32 = estimate_activation_memory_gb(top_k2=100, batch_size=32)
        mem_96 = estimate_activation_memory_gb(top_k2=100, batch_size=96)
        assert abs(mem_96 / mem_32 - 3.0) < 0.01

    def test_returns_positive_gb(self):
        mem = estimate_activation_memory_gb(top_k2=50, batch_size=16)
        assert mem > 0


# ---------------------------------------------------------------------------
# format_analysis_report
# ---------------------------------------------------------------------------

class TestFormatAnalysisReport:
    def _make_runs_data(self, tmp_path: Path) -> list:
        """Build synthetic runs_data matching the shape load_all_runs returns."""
        sweep = tmp_path / 'sweep'
        sweep.mkdir()
        runs_data = []
        for top_k2, c_values in [(50, [0.5, 0.7, 0.8]), (80, [0.6, 0.75, 0.85])]:
            subrun = _build_subrun(sweep, top_k2, c_values)
            inner = next(d for d in subrun.iterdir() if d.is_dir())
            with open(inner / 'loss_history.json') as f:
                loss_history = json.load(f)
            runs_data.append({
                'top_k2': top_k2,
                'status': 'OK',
                'loss_history': loss_history,
                'training_time_minutes': 25.0,
            })
        return runs_data

    def test_report_contains_all_section_headers(self, tmp_path):
        runs_data = self._make_runs_data(tmp_path)
        report = format_analysis_report(
            sweep_root=str(tmp_path / 'sweep'),
            runs_data=runs_data,
        )
        assert '## 1.' in report  # Convergence
        assert '## 2.' in report  # Stability
        assert '## 3.' in report  # Plateau
        assert '## 4.' in report  # Conditional
        assert '## 5.' in report  # Mean rank
        assert '## 6.' in report  # OOM
        assert '## 7.' in report  # Recommendations

    def test_report_contains_k_values(self, tmp_path):
        runs_data = self._make_runs_data(tmp_path)
        report = format_analysis_report(
            sweep_root=str(tmp_path / 'sweep'),
            runs_data=runs_data,
        )
        assert '50' in report
        assert '80' in report

    def test_report_is_nonempty_markdown(self, tmp_path):
        runs_data = self._make_runs_data(tmp_path)
        report = format_analysis_report(
            sweep_root=str(tmp_path / 'sweep'),
            runs_data=runs_data,
        )
        assert report.startswith('#')
        assert len(report) > 500


# ---------------------------------------------------------------------------
# E2E: load_all_runs + main writes file
# ---------------------------------------------------------------------------

class TestLoadAllRuns:
    def test_loads_ok_runs_with_loss_history(self, tmp_path):
        sweep = tmp_path / 'sweep'
        sweep.mkdir()
        _build_subrun(sweep, 50, [0.5, 0.7])
        _build_subrun(sweep, 80, [0.6, 0.8])
        (sweep / 'topk2_99').mkdir()  # failed subrun
        runs = load_all_runs(sweep)
        ok_runs = [r for r in runs if r['status'] == 'OK']
        failed_runs = [r for r in runs if r['status'] == 'FAILED']
        assert len(ok_runs) == 2
        assert len(failed_runs) == 1
        assert all('loss_history' in r for r in ok_runs)


class TestMainEntryPoint:
    def test_writes_report_file(self, tmp_path):
        from diagnostics.analyze_topk2_sweep import main
        sweep = tmp_path / 'sweep'
        sweep.mkdir()
        _build_subrun(sweep, 50, [0.5, 0.7, 0.8])
        _build_subrun(sweep, 80, [0.6, 0.75, 0.85])
        output_path = tmp_path / 'report.md'
        main([str(sweep), '--output', str(output_path)])
        assert output_path.exists()
        content = output_path.read_text()
        assert '## 1.' in content
        assert len(content) > 300
