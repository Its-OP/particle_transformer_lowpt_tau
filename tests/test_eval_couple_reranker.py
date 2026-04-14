"""Tests for diagnostics.eval_couple_reranker — evaluate the couple reranker
and export results as a parquet file with track indices.

Reuses the synthetic model + input fixtures from test_couple_cascade_model.
"""
from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import torch

from tests.test_couple_cascade_model import (
    BATCH_SIZE,
    NUM_PADDED,
    NUM_TRACKS,
    NUM_VALID,
    TOP_K1,
    TOP_K2,
    _make_couple_cascade,
    _make_inputs,
)

from diagnostics.eval_couple_reranker import evaluate_batch


# ---------------------------------------------------------------------------
# evaluate_batch output structure
# ---------------------------------------------------------------------------

class TestEvaluateBatchOutputStructure:
    def test_returns_list_of_dicts(self):
        model = _make_couple_cascade()
        points, features, lorentz_vectors, mask, track_labels = _make_inputs()
        results = evaluate_batch(
            model, (points, features, lorentz_vectors, mask),
        )
        assert isinstance(results, list)
        assert len(results) == BATCH_SIZE
        for result in results:
            assert isinstance(result, dict)
            assert 'couples' in result
            assert 'remaining_pions' in result

    def test_couples_are_index_pairs(self):
        model = _make_couple_cascade()
        inputs = _make_inputs()
        results = evaluate_batch(model, inputs[:4])
        for result in results:
            for couple in result['couples']:
                assert isinstance(couple, list)
                assert len(couple) == 2
                # Both entries are non-negative integers (track positions)
                assert all(isinstance(idx, int) for idx in couple)
                assert all(idx >= 0 for idx in couple)
                # Track index must be within the original event (< NUM_TRACKS)
                assert all(idx < NUM_TRACKS for idx in couple)

    def test_remaining_pions_are_indices(self):
        model = _make_couple_cascade()
        inputs = _make_inputs()
        results = evaluate_batch(model, inputs[:4])
        for result in results:
            remaining = result['remaining_pions']
            assert isinstance(remaining, list)
            assert len(remaining) <= TOP_K1
            assert len(remaining) > 0  # at least some valid tracks
            for idx in remaining:
                assert isinstance(idx, int)
                assert 0 <= idx < NUM_TRACKS

    def test_no_padded_tracks_in_remaining(self):
        """Padded positions (last NUM_PADDED) should NOT appear in
        remaining_pions — those tracks have mask=0."""
        model = _make_couple_cascade()
        inputs = _make_inputs()
        results = evaluate_batch(model, inputs[:4])
        padded_positions = set(range(NUM_VALID, NUM_TRACKS))
        for result in results:
            remaining_set = set(result['remaining_pions'])
            assert remaining_set.isdisjoint(padded_positions), (
                f'Padded positions leaked into remaining_pions: '
                f'{remaining_set & padded_positions}'
            )


# ---------------------------------------------------------------------------
# Couple output constraints
# ---------------------------------------------------------------------------

class TestCoupleOutputConstraints:
    def test_couples_at_most_200(self):
        model = _make_couple_cascade()
        inputs = _make_inputs()
        results = evaluate_batch(
            model, inputs[:4], top_k_output_couples=200,
        )
        for result in results:
            assert len(result['couples']) <= 200

    def test_couples_at_most_custom_k(self):
        model = _make_couple_cascade()
        inputs = _make_inputs()
        results = evaluate_batch(
            model, inputs[:4], top_k_output_couples=5,
        )
        for result in results:
            assert len(result['couples']) <= 5

    def test_couple_indices_are_subset_of_remaining(self):
        """Every track index appearing in a couple must also be present
        in remaining_pions, since K2 tracks are a subset of K1 tracks."""
        model = _make_couple_cascade()
        inputs = _make_inputs()
        results = evaluate_batch(model, inputs[:4])
        for result in results:
            remaining_set = set(result['remaining_pions'])
            for couple in result['couples']:
                for idx in couple:
                    assert idx in remaining_set, (
                        f'Couple track {idx} not in remaining_pions'
                    )

    def test_couple_indices_within_pair_are_distinct(self):
        """A couple cannot pair a track with itself."""
        model = _make_couple_cascade()
        inputs = _make_inputs()
        results = evaluate_batch(model, inputs[:4])
        for result in results:
            for couple in result['couples']:
                assert couple[0] != couple[1], (
                    f'Self-couple detected: {couple}'
                )


# ---------------------------------------------------------------------------
# Parquet schema (via a small helper that writes the output)
# ---------------------------------------------------------------------------

class TestParquetSchema:
    def test_written_parquet_has_correct_columns(self, tmp_path):
        from diagnostics.eval_couple_reranker import write_results_parquet

        # Minimal synthetic results — both index and pT forms
        results_rows = [
            {
                'event_run': 1,
                'event_id': 100,
                'event_luminosity_block': 10,
                'source_batch_id': 3,
                'source_microbatch_id': 42,
                'couple_indices': [[3, 7], [5, 12]],
                'couple_pt': [[0.5, 1.2], [0.8, 2.1]],
                'remaining_pion_indices': [0, 3, 5, 7, 12, 20],
                'remaining_pion_pt': [0.3, 0.5, 0.8, 1.2, 2.1, 3.0],
                'gt_pion_indices': [3, 7, 20],
                'gt_pion_pt': [0.5, 1.2, 3.0],
            },
            {
                'event_run': 1,
                'event_id': 101,
                'event_luminosity_block': 10,
                'source_batch_id': 3,
                'source_microbatch_id': 43,
                'couple_indices': [[1, 2]],
                'couple_pt': [[0.4, 0.9]],
                'remaining_pion_indices': [1, 2, 8],
                'remaining_pion_pt': [0.4, 0.9, 1.5],
                'gt_pion_indices': [1, 2, 8],
                'gt_pion_pt': [0.4, 0.9, 1.5],
            },
        ]
        output_path = tmp_path / 'test_output.parquet'
        write_results_parquet(results_rows, str(output_path))

        table = pq.read_table(output_path)
        assert table.num_rows == 2
        expected_columns = {
            'event_run', 'event_id', 'event_luminosity_block',
            'source_batch_id', 'source_microbatch_id',
            'couple_indices', 'couple_pt',
            'remaining_pion_indices', 'remaining_pion_pt',
            'gt_pion_indices', 'gt_pion_pt',
        }
        assert set(table.schema.names) == expected_columns
        # Index columns are int, pT columns are float
        assert table.column('couple_indices')[0].as_py() == [[3, 7], [5, 12]]
        assert len(table.column('couple_pt')[0].as_py()) == 2
        assert table.column('remaining_pion_indices')[0].as_py() == [0, 3, 5, 7, 12, 20]
        assert len(table.column('remaining_pion_pt')[0].as_py()) == 6
        assert table.column('gt_pion_indices')[0].as_py() == [3, 7, 20]
