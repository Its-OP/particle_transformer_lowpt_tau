"""Tests for diagnostics.aggregate_couple_sweep.

The aggregator walks a sweep root with one subdirectory per top_k2,
finds each subrun's loss_history.json, picks the epoch with the best
``val_c_at_100_couples``, and produces sweep_summary.{json,md}.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from diagnostics.aggregate_couple_sweep import (
    _extract_values,
    aggregate_sweep,
    find_best_epoch,
    find_subrun_experiment_dir,
    format_summary_markdown,
    parse_top_k2_from_subrun,
)


# ---------------------------------------------------------------------------
# Synthetic sweep filesystem builders
# ---------------------------------------------------------------------------

def _write_loss_history(
    experiment_dir: Path,
    *,
    c_at_100_per_epoch: list[float],
    extra_metrics: dict[str, list[float]] | None = None,
) -> None:
    """Write a labelled-metric loss_history.json under experiment_dir."""
    experiment_dir.mkdir(parents=True, exist_ok=True)
    history = {
        'val_c_at_100_couples': {
            'label': 'C@100_couples (per total events)',
            'values': c_at_100_per_epoch,
        },
        'val_c_at_50_couples': {
            'label': 'C@50_couples (per total events)',
            'values': [v * 0.9 for v in c_at_100_per_epoch],
        },
        'val_c_at_200_couples': {
            'label': 'C@200_couples (per total events)',
            'values': [min(1.0, v * 1.05) for v in c_at_100_per_epoch],
        },
        'val_rc_at_100_couples': {
            'label': 'RC@100_couples (per total events)',
            'values': [v * 0.95 for v in c_at_100_per_epoch],
        },
        'val_rc_at_200_couples': {
            'label': 'RC@200_couples (per total events)',
            'values': [v * 0.97 for v in c_at_100_per_epoch],
        },
        'val_d_at_100_tracks': {
            'label': 'D@100_tracks (cascade duplet rate)',
            'values': [0.91] * len(c_at_100_per_epoch),
        },
        'val_d_at_200_tracks': {
            'label': 'D@200_tracks (cascade duplet rate)',
            'values': [0.96] * len(c_at_100_per_epoch),
        },
        'val_mean_first_gt_rank_couples': {
            'label': 'Mean rank of best GT couple',
            'values': [50.0 - i * 2 for i in range(len(c_at_100_per_epoch))],
        },
        'val_eligible_events': {
            'label': 'Eligible events',
            'values': [1700] * len(c_at_100_per_epoch),
        },
        'val_total_events': {
            'label': 'Total events',
            'values': [1700] * len(c_at_100_per_epoch),
        },
        'val_events_with_full_triplet': {
            'label': 'Events with full triplet',
            'values': [1620] * len(c_at_100_per_epoch),
        },
    }
    if extra_metrics:
        for key, values in extra_metrics.items():
            history[key] = {'label': key, 'values': values}
    with open(experiment_dir / 'loss_history.json', 'w') as f:
        json.dump(history, f)


def _build_subrun(sweep_root: Path, top_k2: int, c_at_100: list[float]) -> Path:
    """Create a subrun directory mimicking the trainer's layout."""
    subrun_dir = sweep_root / f'topk2_{top_k2}'
    inner_dir = subrun_dir / f'topk2_{top_k2}_CoupleReranker_20260408_120000'
    _write_loss_history(inner_dir, c_at_100_per_epoch=c_at_100)
    return subrun_dir


# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------

class TestParseTopK2:
    def test_simple_case(self):
        assert parse_top_k2_from_subrun(Path('/x/topk2_50')) == 50
        assert parse_top_k2_from_subrun(Path('/x/topk2_200')) == 200

    def test_returns_none_on_unrelated_dir(self):
        assert parse_top_k2_from_subrun(Path('/x/sweep_summary')) is None
        assert parse_top_k2_from_subrun(Path('/x/something_else')) is None

    def test_returns_none_on_malformed(self):
        assert parse_top_k2_from_subrun(Path('/x/topk2_abc')) is None
        assert parse_top_k2_from_subrun(Path('/x/topk2_')) is None


class TestExtractValues:
    def test_labelled_dict_format(self):
        entry = {'label': 'val', 'values': [1.0, 2.0, 3.0]}
        assert _extract_values(entry) == [1.0, 2.0, 3.0]

    def test_bare_list_format(self):
        assert _extract_values([1.0, 2.0]) == [1.0, 2.0]

    def test_invalid_returns_empty(self):
        assert _extract_values(None) == []
        assert _extract_values(42) == []
        assert _extract_values({}) == []


class TestFindBestEpoch:
    def test_finds_argmax(self):
        loss_history = {
            'val_c_at_100_couples': {
                'label': 'C@100',
                'values': [0.1, 0.5, 0.9, 0.4, 0.7],
            },
            'val_c_at_50_couples': {
                'label': 'C@50',
                'values': [0.05, 0.25, 0.8, 0.3, 0.6],
            },
        }
        result = find_best_epoch(loss_history)
        assert result is not None
        assert result['best_epoch_index'] == 2
        assert result['best_epoch_number'] == 3  # 1-indexed
        assert result['criterion_value'] == 0.9
        assert result['metrics_at_best_epoch']['val_c_at_50_couples'] == 0.8

    def test_returns_none_when_criterion_missing(self):
        loss_history = {
            'val_c_at_50_couples': {
                'label': 'C@50',
                'values': [0.5, 0.8],
            },
        }
        assert find_best_epoch(loss_history) is None


class TestFindSubrunExperimentDir:
    def test_finds_dir_with_loss_history(self, tmp_path):
        subrun = tmp_path / 'topk2_50'
        inner = subrun / 'topk2_50_CoupleReranker_xyz'
        inner.mkdir(parents=True)
        (inner / 'loss_history.json').write_text('{}')
        assert find_subrun_experiment_dir(subrun) == inner

    def test_returns_none_when_no_loss_history(self, tmp_path):
        subrun = tmp_path / 'topk2_50'
        (subrun / 'topk2_50_CoupleReranker_abc').mkdir(parents=True)
        # No loss_history.json
        assert find_subrun_experiment_dir(subrun) is None

    def test_picks_most_recent_when_multiple(self, tmp_path):
        subrun = tmp_path / 'topk2_50'
        old = subrun / 'topk2_50_old'
        new = subrun / 'topk2_50_new'
        old.mkdir(parents=True)
        new.mkdir(parents=True)
        (old / 'loss_history.json').write_text('{}')
        (new / 'loss_history.json').write_text('{}')
        # Touch new to be newer
        import os
        os.utime(new, (10**10, 10**10))
        os.utime(new / 'loss_history.json', (10**10, 10**10))
        assert find_subrun_experiment_dir(subrun) == new


# ---------------------------------------------------------------------------
# End-to-end aggregation
# ---------------------------------------------------------------------------

class TestAggregateSweep:
    def test_three_subruns_picks_best_per_run(self, tmp_path):
        sweep = tmp_path / 'sweep'
        sweep.mkdir()
        _build_subrun(sweep, 20, [0.3, 0.5, 0.6, 0.55])
        _build_subrun(sweep, 50, [0.4, 0.7, 0.85, 0.82])
        _build_subrun(sweep, 100, [0.5, 0.65, 0.78, 0.80])
        result = aggregate_sweep(sweep)
        assert len(result['runs']) == 3
        # Sorted by top_k2
        assert [r['top_k2'] for r in result['runs']] == [20, 50, 100]
        assert all(r['status'] == 'OK' for r in result['runs'])
        # Best epoch per run picks the argmax of C@100
        run_50 = next(r for r in result['runs'] if r['top_k2'] == 50)
        assert run_50['best_epoch_number'] == 3
        assert run_50['criterion_value'] == pytest.approx(0.85)

    def test_failed_subrun_marked_failed(self, tmp_path):
        sweep = tmp_path / 'sweep'
        sweep.mkdir()
        _build_subrun(sweep, 20, [0.5])
        # Subrun with no inner dir at all
        (sweep / 'topk2_30').mkdir()
        # Subrun with inner dir but no loss_history.json
        (sweep / 'topk2_40' / 'topk2_40_CoupleReranker_x').mkdir(parents=True)
        result = aggregate_sweep(sweep)
        statuses = {r['top_k2']: r['status'] for r in result['runs']}
        assert statuses == {20: 'OK', 30: 'FAILED', 40: 'FAILED'}

    def test_skips_non_topk2_directories(self, tmp_path):
        sweep = tmp_path / 'sweep'
        sweep.mkdir()
        _build_subrun(sweep, 50, [0.7])
        (sweep / 'sweep_summary').mkdir()
        (sweep / 'logs').mkdir()
        result = aggregate_sweep(sweep)
        assert len(result['runs']) == 1
        assert result['runs'][0]['top_k2'] == 50


# ---------------------------------------------------------------------------
# Markdown formatter
# ---------------------------------------------------------------------------

class TestFormatSummaryMarkdown:
    def test_includes_per_topk2_table(self, tmp_path):
        sweep = tmp_path / 'sweep'
        sweep.mkdir()
        _build_subrun(sweep, 20, [0.3, 0.5])
        _build_subrun(sweep, 50, [0.4, 0.85])
        result = aggregate_sweep(sweep)
        markdown = format_summary_markdown(result)
        assert isinstance(markdown, str)
        assert '20' in markdown
        assert '50' in markdown
        assert '0.85' in markdown or '0.8500' in markdown
        assert 'top_k2' in markdown.lower()

    def test_marks_winning_run(self, tmp_path):
        sweep = tmp_path / 'sweep'
        sweep.mkdir()
        _build_subrun(sweep, 20, [0.5])
        _build_subrun(sweep, 50, [0.9])
        _build_subrun(sweep, 100, [0.7])
        result = aggregate_sweep(sweep)
        markdown = format_summary_markdown(result)
        # The best run is top_k2=50 with C@100=0.9
        assert 'best' in markdown.lower()
        # 0.9 is the criterion at the best run
        assert '0.9' in markdown

    def test_lists_failed_subruns(self, tmp_path):
        sweep = tmp_path / 'sweep'
        sweep.mkdir()
        _build_subrun(sweep, 20, [0.5])
        (sweep / 'topk2_99').mkdir()  # failed
        result = aggregate_sweep(sweep)
        markdown = format_summary_markdown(result)
        assert 'FAILED' in markdown or 'failed' in markdown.lower()
        assert '99' in markdown

    def test_no_runs_does_not_crash(self, tmp_path):
        sweep = tmp_path / 'sweep'
        sweep.mkdir()
        result = aggregate_sweep(sweep)
        markdown = format_summary_markdown(result)
        assert isinstance(markdown, str)


# ---------------------------------------------------------------------------
# CLI smoke test (writes both .json and .md)
# ---------------------------------------------------------------------------

class TestCliEntryPoint:
    def test_main_writes_summary_files(self, tmp_path):
        from diagnostics.aggregate_couple_sweep import main
        sweep = tmp_path / 'sweep'
        sweep.mkdir()
        _build_subrun(sweep, 50, [0.5, 0.8])
        _build_subrun(sweep, 100, [0.6, 0.75])
        main([str(sweep)])
        assert (sweep / 'sweep_summary.json').exists()
        assert (sweep / 'sweep_summary.md').exists()
        with open(sweep / 'sweep_summary.json') as f:
            data = json.load(f)
        assert 'runs' in data
        assert len(data['runs']) == 2
