"""
Validate data quality of low-pT tau parquet files.

Scans all .parquet files in a directory for simulation artefacts and data
corruption, prints a combined report across all batches, then interactively
asks whether to write cleaned copies (corrupt events removed) to a separate
output directory. Original files are never modified.

Checks performed:
  - Empty events (0 tracks)
  - NaN and Inf values in any column
  - Impossibly-large values (e.g. track_pt = 65504 ≈ float16 max,
    track_eta ~ 10^30 from CMSSW simulation bugs)
  - Unphysical values violating CMS detector acceptance:
      track_eta outside [-2.5, 2.5], track_phi outside [-π, π],
      track_pt <= 0 or > 500 GeV, track_charge not ±1
  - Tau-origin track counts (multi-tau events reported, not dropped)
  - Track count consistency between event_n_tracks and jagged arrays

Usage:
    # Validate all parquet files in a directory (interactive):
    python part/utils/validate_parquet_quality.py part/data/low-pt/

    # Non-interactive (auto-clean without prompting):
    python part/utils/validate_parquet_quality.py part/data/low-pt/ --yes

    # With JSON report:
    python part/utils/validate_parquet_quality.py part/data/low-pt/ --report report.json
"""

import os
import sys
import json
import argparse
import glob
from collections import defaultdict
from dataclasses import dataclass, field, asdict

import numpy as np
import awkward as ak


# ---------------------------------------------------------------------------
# Physical bounds for CMS pion tracks
# ---------------------------------------------------------------------------

# CMS tracking detector covers |η| < 2.5. Tracks beyond this come from
# simulation artefacts. We use 5.0 as a generous bound for forward tracks.
ETA_ABSOLUTE_MAX = 5.0

# Azimuthal angle: φ ∈ [-π, π] by definition.
PHI_ABSOLUTE_MAX = np.pi + 0.01  # small epsilon for floating-point

# Transverse momentum: pion tracks should be positive and within detector range.
# float16 max = 65504 — values at this boundary indicate upstream clipping.
PT_MIN = 0.0
PT_MAX = 500.0  # GeV; pions above this are simulation artefacts (pions typically < 10 GeV)
FLOAT16_MAX = 65504.0

# Valid charges for charged pions: ±1
VALID_CHARGES = {-1, 1}

# Maximum expected tau-origin tracks per event from a single tau decay.
# τ → πππν gives exactly 3 charged pions. However, rare events can contain
# multiple tau decays, yielding > 3 tau-origin tracks. These are physically
# valid and should NOT be filtered.
MAX_TAU_TRACKS_SINGLE_DECAY = 3

# Impact parameter significance: values beyond this are likely artefacts
DXY_SIGNIFICANCE_MAX = 1000.0


# ---------------------------------------------------------------------------
# Column classification
# ---------------------------------------------------------------------------

# Track-level columns: jagged arrays (variable-length per event)
TRACK_COLUMNS_CONTINUOUS = [
    'track_pt', 'track_eta', 'track_phi',
    'track_dxy_significance',
]

TRACK_COLUMNS_INTEGER = [
    'track_charge', 'track_label_from_tau',
]

# Event-level columns: scalars per event
EVENT_COLUMNS = [
    'event_n_tracks',
    'event_primary_vertex_x', 'event_primary_vertex_y', 'event_primary_vertex_z',
]


# ---------------------------------------------------------------------------
# Issue tracking
# ---------------------------------------------------------------------------

# Issue types that warrant event removal during cleaning.
# Other issue types (e.g. multi_tau_event, extreme_dxy_significance) are
# informational only and do not cause event removal.
#
# Rationale:
# - extreme_dxy_significance: auto-standardization uses median + IQR (16th/84th
#   percentiles), so outliers don't affect normalization. After standardization
#   the value is clipped to [-5, 5]. Safe to keep.
# - multi_tau_event: physically valid events with multiple tau decays.
# - nan / inf / invalid_charge / unphysical_* / high pT: corrupt data that
#   can propagate through raw 4-vectors (pf_vectors bypass standardization)
#   and break pairwise_lv_fts() or backbone computations.
DROP_ISSUE_TYPES = {
    'nan', 'inf',
    'empty_event',
    'unphysical_eta', 'unphysical_phi',
    'nonpositive_pt', 'float16_clipped_pt', 'very_high_pt',
    'invalid_charge',
    'negative_tau_count', 'non_binary_label',
    'track_count_mismatch',
}


@dataclass
class EventIssue:
    """A single data quality issue for one event."""
    event_index: int
    issue_type: str  # e.g. "nan", "inf", "unphysical_eta", "empty_event"
    column: str
    detail: str  # human-readable description
    should_drop: bool = False  # whether this issue warrants event removal

    def to_dict(self):
        return asdict(self)


@dataclass
class FileReport:
    """Aggregated report for a single parquet file."""
    filepath: str
    number_of_events: int = 0
    columns_found: list = field(default_factory=list)
    columns_missing: list = field(default_factory=list)
    issues: list = field(default_factory=list)
    issue_counts: dict = field(default_factory=lambda: defaultdict(int))
    flagged_event_indices: set = field(default_factory=set)
    drop_event_indices: set = field(default_factory=set)

    def add_issue(self, event_index, issue_type, column, detail):
        should_drop = issue_type in DROP_ISSUE_TYPES
        issue = EventIssue(event_index, issue_type, column, detail, should_drop)
        self.issues.append(issue)
        self.issue_counts[issue_type] += 1
        self.flagged_event_indices.add(event_index)
        if should_drop:
            self.drop_event_indices.add(event_index)

    def to_dict(self):
        return {
            'filepath': self.filepath,
            'number_of_events': self.number_of_events,
            'columns_found': self.columns_found,
            'columns_missing': self.columns_missing,
            'issue_counts': dict(self.issue_counts),
            'total_issues': len(self.issues),
            'flagged_events': len(self.flagged_event_indices),
            'flagged_event_indices': sorted(self.flagged_event_indices),
            'events_to_drop': len(self.drop_event_indices),
            'drop_event_indices': sorted(self.drop_event_indices),
            'issues': [issue.to_dict() for issue in self.issues],
        }


# ---------------------------------------------------------------------------
# Validation checks
# ---------------------------------------------------------------------------

def check_nan_inf(data, column_name, report, is_jagged):
    """Check for NaN and Inf values in a column.

    Uses vectorized ak.count_nonzero along axis=1 to compute per-event
    NaN/Inf counts without per-event random access into jagged arrays.

    Args:
        data: awkward array (jagged or flat) for this column.
        column_name: Name of the column being checked.
        report: FileReport to accumulate issues.
        is_jagged: Whether this is a variable-length (per-track) column.
    """
    if is_jagged:
        # Vectorized: compute per-event NaN/Inf counts in one pass
        nan_mask = np.isnan(data)
        inf_mask = np.isinf(data)
        nan_counts_per_event = ak.to_numpy(ak.count_nonzero(nan_mask, axis=1))
        inf_counts_per_event = ak.to_numpy(ak.count_nonzero(inf_mask, axis=1))

        nan_event_indices = np.where(nan_counts_per_event > 0)[0]
        inf_event_indices = np.where(inf_counts_per_event > 0)[0]

        for event_index in nan_event_indices:
            nan_count = int(nan_counts_per_event[event_index])
            report.add_issue(
                int(event_index), 'nan', column_name,
                f'{nan_count} NaN track value(s)'
            )

        for event_index in inf_event_indices:
            inf_count = int(inf_counts_per_event[event_index])
            report.add_issue(
                int(event_index), 'inf', column_name,
                f'{inf_count} Inf track value(s)'
            )
    else:
        values = ak.to_numpy(data)
        nan_indices = np.where(np.isnan(values))[0]
        inf_indices = np.where(np.isinf(values))[0]

        for event_index in nan_indices:
            report.add_issue(
                int(event_index), 'nan', column_name,
                f'NaN value: {values[event_index]}'
            )

        for event_index in inf_indices:
            report.add_issue(
                int(event_index), 'inf', column_name,
                f'Inf value: {values[event_index]}'
            )


def check_empty_events(data, report):
    """Flag events with zero tracks.

    These events cause NaN in FPS (Farthest Point Sampling), kNN
    (k-Nearest Neighbors), and max-pool operations in the backbone.

    Args:
        data: The full awkward record array.
        report: FileReport to accumulate issues.
    """
    if 'event_n_tracks' in ak.fields(data):
        track_counts = ak.to_numpy(data['event_n_tracks'])
        empty_indices = np.where(track_counts == 0)[0]
        for event_index in empty_indices:
            report.add_issue(
                int(event_index), 'empty_event', 'event_n_tracks',
                'Event has 0 tracks'
            )
    elif 'track_pt' in ak.fields(data):
        # Fallback: count tracks from jagged array
        track_counts = ak.to_numpy(ak.num(data['track_pt'], axis=1))
        empty_indices = np.where(track_counts == 0)[0]
        for event_index in empty_indices:
            report.add_issue(
                int(event_index), 'empty_event', 'track_pt',
                'Event has 0 tracks (inferred from track_pt length)'
            )


def check_unphysical_eta(data, report):
    """Flag tracks with |η| > 5.0 (outside CMS tracking acceptance).

    CMS tracking covers |η| < 2.5. Values beyond 5.0 are simulation
    artefacts — e.g. event 11960 in merged_noBKstar.root had tracks
    with η ~ 7×10³⁰.

    Uses vectorized ak.count_nonzero and ak.max along axis=1 to avoid
    per-event random access into jagged arrays.

    Args:
        data: The full awkward record array.
        report: FileReport to accumulate issues.
    """
    if 'track_eta' not in ak.fields(data):
        return

    eta = data['track_eta']
    abs_eta = np.abs(eta)
    # |η| > ETA_ABSOLUTE_MAX, excluding NaN (caught by check_nan_inf)
    bad_eta_mask = abs_eta > ETA_ABSOLUTE_MAX

    # Vectorized per-event statistics
    bad_count_per_event = ak.to_numpy(ak.count_nonzero(bad_eta_mask, axis=1))
    # ak.max on abs_eta gives the worst |η| per event (NaN-safe: ak.max skips None)
    worst_abs_eta_per_event = ak.to_numpy(ak.fill_none(ak.max(abs_eta, axis=1), 0.0))

    bad_event_indices = np.where(bad_count_per_event > 0)[0]

    for event_index in bad_event_indices:
        count = int(bad_count_per_event[event_index])
        worst_value = float(worst_abs_eta_per_event[event_index])
        report.add_issue(
            int(event_index), 'unphysical_eta', 'track_eta',
            f'{count} track(s) with |η| > {ETA_ABSOLUTE_MAX}, '
            f'worst: |η| = {worst_value:.6g}'
        )


def check_unphysical_phi(data, report):
    """Flag tracks with |φ| > π (mathematically impossible for azimuthal angle).

    Uses vectorized ak.count_nonzero and ak.max along axis=1.

    Args:
        data: The full awkward record array.
        report: FileReport to accumulate issues.
    """
    if 'track_phi' not in ak.fields(data):
        return

    phi = data['track_phi']
    abs_phi = np.abs(phi)
    bad_phi_mask = abs_phi > PHI_ABSOLUTE_MAX

    # Vectorized per-event statistics
    bad_count_per_event = ak.to_numpy(ak.count_nonzero(bad_phi_mask, axis=1))
    worst_abs_phi_per_event = ak.to_numpy(ak.fill_none(ak.max(abs_phi, axis=1), 0.0))

    bad_event_indices = np.where(bad_count_per_event > 0)[0]

    for event_index in bad_event_indices:
        count = int(bad_count_per_event[event_index])
        worst_value = float(worst_abs_phi_per_event[event_index])
        report.add_issue(
            int(event_index), 'unphysical_phi', 'track_phi',
            f'{count} track(s) with |φ| > π, '
            f'worst: |φ| = {worst_value:.6g}'
        )


def check_unphysical_pt(data, report):
    """Flag tracks with non-positive pT or impossibly large pT.

    Checks for:
      - pT ≤ 0 (unphysical: transverse momentum is non-negative by definition)
      - pT ≥ 65504 (float16 max: indicates upstream precision clipping)
      - pT > 500 GeV (unphysical for pion tracks, corrupts raw 4-vectors)

    Uses vectorized ak reductions along axis=1 to avoid per-event jagged access.

    Args:
        data: The full awkward record array.
        report: FileReport to accumulate issues.
    """
    if 'track_pt' not in ak.fields(data):
        return

    track_pt = data['track_pt']

    # --- pT ≤ 0 (non-positive transverse momentum) ---
    nonpositive_mask = track_pt <= PT_MIN
    nonpositive_count_per_event = ak.to_numpy(ak.count_nonzero(nonpositive_mask, axis=1))
    # For worst value: replace non-flagged tracks with +inf so ak.min finds the worst
    pt_for_min = ak.where(nonpositive_mask, track_pt, np.inf)
    worst_nonpositive_per_event = ak.to_numpy(ak.fill_none(ak.min(pt_for_min, axis=1), np.inf))

    nonpositive_event_indices = np.where(nonpositive_count_per_event > 0)[0]
    for event_index in nonpositive_event_indices:
        count = int(nonpositive_count_per_event[event_index])
        worst_value = float(worst_nonpositive_per_event[event_index])
        report.add_issue(
            int(event_index), 'nonpositive_pt', 'track_pt',
            f'{count} track(s) with pT ≤ 0, worst: pT = {worst_value:.6g} GeV'
        )

    # --- pT ≥ 65504 (float16 max boundary — upstream clipping artefact) ---
    float16_clipped_mask = track_pt >= FLOAT16_MAX
    clipped_count_per_event = ak.to_numpy(ak.count_nonzero(float16_clipped_mask, axis=1))

    clipped_event_indices = np.where(clipped_count_per_event > 0)[0]
    for event_index in clipped_event_indices:
        count = int(clipped_count_per_event[event_index])
        report.add_issue(
            int(event_index), 'float16_clipped_pt', 'track_pt',
            f'{count} track(s) with pT ≥ {FLOAT16_MAX} (float16 max — '
            f'simulation precision artefact)'
        )

    # --- pT > 500 GeV but below float16 max (corrupts raw 4-vectors) ---
    # These bypass standardization via pf_vectors and break pairwise_lv_fts().
    very_high_mask = (track_pt > PT_MAX) & (track_pt < FLOAT16_MAX)
    very_high_count_per_event = ak.to_numpy(ak.count_nonzero(very_high_mask, axis=1))
    # For worst value: replace non-flagged tracks with -inf so ak.max finds the worst
    pt_for_max = ak.where(very_high_mask, track_pt, -np.inf)
    worst_high_per_event = ak.to_numpy(ak.fill_none(ak.max(pt_for_max, axis=1), -np.inf))

    very_high_event_indices = np.where(very_high_count_per_event > 0)[0]
    for event_index in very_high_event_indices:
        count = int(very_high_count_per_event[event_index])
        worst_value = float(worst_high_per_event[event_index])
        report.add_issue(
            int(event_index), 'very_high_pt', 'track_pt',
            f'{count} track(s) with pT > {PT_MAX} GeV, '
            f'max: {worst_value:.1f} GeV'
        )


def check_invalid_charge(data, report):
    """Flag tracks with charge not equal to ±1.

    Charged pions have charge ±1 by definition (π⁺ = +1, π⁻ = −1).
    Other values indicate data corruption.

    Uses vectorized ak.count_nonzero along axis=1. For the rare invalid-charge
    events (~5 in 104K), per-event access to get unique bad values is acceptable.

    Args:
        data: The full awkward record array.
        report: FileReport to accumulate issues.
    """
    if 'track_charge' not in ak.fields(data):
        return

    charge = data['track_charge']
    # Check each event for tracks with invalid charge
    valid_mask = (charge == 1) | (charge == -1)
    invalid_mask = ~valid_mask
    invalid_count_per_event = ak.to_numpy(ak.count_nonzero(invalid_mask, axis=1))

    bad_event_indices = np.where(invalid_count_per_event > 0)[0]

    # Extract invalid charge values only for the (very few) flagged events.
    # charge[invalid_mask] keeps only invalid values (jagged), then flatten.
    if len(bad_event_indices) > 0:
        invalid_charges_flat = ak.to_numpy(ak.flatten(charge[invalid_mask]))
        global_unique_bad = np.unique(invalid_charges_flat)

    for event_index in bad_event_indices:
        count = int(invalid_count_per_event[event_index])
        # Show a limited sample of unique bad values to keep output readable
        shown_values = global_unique_bad[:10].tolist()
        suffix = f' ... ({len(global_unique_bad)} unique total)' if len(global_unique_bad) > 10 else ''
        report.add_issue(
            int(event_index), 'invalid_charge', 'track_charge',
            f'{count} track(s) with charge ∉ {{±1}}, '
            f'sample values: {shown_values}{suffix}'
        )


def check_tau_track_counts(data, report):
    """Report tau-origin track count distribution and flag corrupted labels.

    Events with > 3 tau-origin tracks are physically valid (rare multi-tau
    decays) and are NOT flagged for removal. Only corrupted labels (negative
    sums or non-binary values) are flagged.

    Fully vectorized: uses ak.sum and ak.count_nonzero along axis=1.

    Args:
        data: The full awkward record array.
        report: FileReport to accumulate issues.
    """
    if 'track_label_from_tau' not in ak.fields(data):
        return

    labels = data['track_label_from_tau']
    tau_tracks_per_event = ak.to_numpy(ak.sum(labels, axis=1))

    # Informational: events with > 3 tau tracks (multi-tau decays, not an error)
    multi_tau_indices = np.where(tau_tracks_per_event > MAX_TAU_TRACKS_SINGLE_DECAY)[0]
    for event_index in multi_tau_indices:
        count = int(tau_tracks_per_event[event_index])
        report.add_issue(
            int(event_index), 'multi_tau_event', 'track_label_from_tau',
            f'{count} tau-origin tracks (multi-tau decay, informational only)'
        )

    # Events with negative label sum (should be impossible — data corruption)
    negative_indices = np.where(tau_tracks_per_event < 0)[0]
    for event_index in negative_indices:
        count = int(tau_tracks_per_event[event_index])
        report.add_issue(
            int(event_index), 'negative_tau_count', 'track_label_from_tau',
            f'Negative tau-track sum: {count} (data corruption)'
        )

    # Check for non-binary label values (should be 0 or 1)
    non_binary_mask = (labels != 0) & (labels != 1)
    non_binary_count_per_event = ak.to_numpy(ak.count_nonzero(non_binary_mask, axis=1))
    bad_event_indices = np.where(non_binary_count_per_event > 0)[0]

    if len(bad_event_indices) > 0:
        # Get global unique bad values: labels[non_binary_mask] keeps only flagged values
        non_binary_flat = ak.to_numpy(ak.flatten(labels[non_binary_mask]))
        global_unique_bad = np.unique(non_binary_flat)

        for event_index in bad_event_indices:
            count = int(non_binary_count_per_event[event_index])
            report.add_issue(
                int(event_index), 'non_binary_label', 'track_label_from_tau',
                f'{count} track(s) with label ∉ {{0, 1}}, '
                f'found global unique values: {global_unique_bad.tolist()}'
            )


def check_dxy_significance_outliers(data, report):
    """Flag tracks with extremely large impact parameter significance.

    Large |dxy/σ(dxy)| values (> 1000) are likely simulation artefacts
    or poorly reconstructed tracks.

    Uses vectorized ak.count_nonzero and ak.max along axis=1.

    Args:
        data: The full awkward record array.
        report: FileReport to accumulate issues.
    """
    if 'track_dxy_significance' not in ak.fields(data):
        return

    dxy_significance = data['track_dxy_significance']
    abs_dxy_significance = np.abs(dxy_significance)
    outlier_mask = abs_dxy_significance > DXY_SIGNIFICANCE_MAX

    # Vectorized per-event statistics
    outlier_count_per_event = ak.to_numpy(ak.count_nonzero(outlier_mask, axis=1))
    worst_abs_dxy_per_event = ak.to_numpy(
        ak.fill_none(ak.max(abs_dxy_significance, axis=1), 0.0)
    )

    bad_event_indices = np.where(outlier_count_per_event > 0)[0]

    for event_index in bad_event_indices:
        count = int(outlier_count_per_event[event_index])
        worst_value = float(worst_abs_dxy_per_event[event_index])
        report.add_issue(
            int(event_index), 'extreme_dxy_significance', 'track_dxy_significance',
            f'{count} track(s) with |dxy_sig| > {DXY_SIGNIFICANCE_MAX}, '
            f'worst: {worst_value:.6g}'
        )


def check_track_count_consistency(data, report):
    """Verify event_n_tracks matches actual number of tracks in jagged arrays.

    If these disagree, the conversion pipeline has a bug.

    Args:
        data: The full awkward record array.
        report: FileReport to accumulate issues.
    """
    if 'event_n_tracks' not in ak.fields(data):
        return

    declared_counts = ak.to_numpy(data['event_n_tracks'])

    # Find a jagged track column to count actual tracks
    track_column = None
    for column_name in ['track_pt', 'track_eta', 'track_phi']:
        if column_name in ak.fields(data):
            track_column = column_name
            break

    if track_column is None:
        return

    actual_counts = ak.to_numpy(ak.num(data[track_column], axis=1))
    mismatch_indices = np.where(declared_counts != actual_counts)[0]

    for event_index in mismatch_indices:
        declared = int(declared_counts[event_index])
        actual = int(actual_counts[event_index])
        report.add_issue(
            int(event_index), 'track_count_mismatch', 'event_n_tracks',
            f'Declared {declared} tracks but jagged array has {actual} '
            f'(in column {track_column})'
        )


# ---------------------------------------------------------------------------
# Main validation
# ---------------------------------------------------------------------------

def validate_parquet_file(filepath, verbose=True):
    """Run all data quality checks on a single parquet file.

    Args:
        filepath: Path to the parquet file.
        verbose: Whether to print progress messages.

    Returns:
        Tuple of (FileReport, data) where data is the loaded awkward array,
        kept in memory so write_cleaned_parquet can reuse it without re-reading.
    """
    if verbose:
        print(f'\n{"=" * 70}')
        print(f'Validating: {filepath}')
        print(f'{"=" * 70}')

    report = FileReport(filepath=filepath)

    # Load the data
    data = ak.from_parquet(filepath)
    report.number_of_events = len(data)
    report.columns_found = sorted(ak.fields(data))

    if verbose:
        file_size_megabytes = os.path.getsize(filepath) / (1024 * 1024)
        print(f'  Events: {report.number_of_events:,}')
        print(f'  File size: {file_size_megabytes:.1f} MB')
        print(f'  Columns: {report.columns_found}')

    # Identify which columns are present
    all_expected = (TRACK_COLUMNS_CONTINUOUS + TRACK_COLUMNS_INTEGER + EVENT_COLUMNS)
    report.columns_missing = [
        column for column in all_expected
        if column not in report.columns_found
    ]
    if report.columns_missing and verbose:
        print(f'  ⚠ Missing expected columns: {report.columns_missing}')

    # --- Run all checks ---
    if verbose:
        print(f'\n  Running checks...')

    # 1. Empty events
    if verbose:
        print(f'    [1/8] Empty events...')
    check_empty_events(data, report)

    # 2. NaN/Inf in continuous track columns
    if verbose:
        print(f'    [2/8] NaN/Inf in track columns...')
    for column_name in TRACK_COLUMNS_CONTINUOUS:
        if column_name in ak.fields(data):
            check_nan_inf(data[column_name], column_name, report, is_jagged=True)

    # 3. NaN/Inf in event-level columns
    if verbose:
        print(f'    [3/8] NaN/Inf in event columns...')
    for column_name in EVENT_COLUMNS:
        if column_name in ak.fields(data):
            check_nan_inf(data[column_name], column_name, report, is_jagged=False)

    # 4. Unphysical η
    if verbose:
        print(f'    [4/8] Unphysical η values...')
    check_unphysical_eta(data, report)

    # 5. Unphysical φ
    if verbose:
        print(f'    [5/8] Unphysical φ values...')
    check_unphysical_phi(data, report)

    # 6. Unphysical pT (non-positive, float16 clipping, extremely high)
    if verbose:
        print(f'    [6/8] Unphysical pT values...')
    check_unphysical_pt(data, report)

    # 7. Invalid charge, tau-track counts, dxy significance outliers
    if verbose:
        print(f'    [7/8] Charge, labels, impact parameter...')
    check_invalid_charge(data, report)
    check_tau_track_counts(data, report)
    check_dxy_significance_outliers(data, report)

    # 8. Track count consistency
    if verbose:
        print(f'    [8/8] Track count consistency...')
    check_track_count_consistency(data, report)

    return report, data


def print_report_summary(report):
    """Print a human-readable summary of the validation report.

    Args:
        report: A FileReport instance.
    """
    print(f'\n  --- Summary for {os.path.basename(report.filepath)} ---')
    print(f'  Total events:       {report.number_of_events:,}')
    print(f'  Events to drop:     {len(report.drop_event_indices):,} '
          f'({100 * len(report.drop_event_indices) / max(report.number_of_events, 1):.2f}%)')
    print(f'  Events after clean: {report.number_of_events - len(report.drop_event_indices):,}')
    print(f'  Informational only: {len(report.flagged_event_indices) - len(report.drop_event_indices):,}')

    if report.issue_counts:
        print(f'\n  Issue breakdown:')
        # Sort by count (descending), mark drop vs info
        for issue_type, count in sorted(
            report.issue_counts.items(), key=lambda item: -item[1]
        ):
            marker = '✗ DROP' if issue_type in DROP_ISSUE_TYPES else '  info'
            print(f'    {marker}  {issue_type:30s} : {count:,}')
    else:
        print(f'\n  ✓ No issues found — data is clean.')

    # Print a few example issues for each type
    if report.issues:
        print(f'\n  Example issues (first 3 per type):')
        issues_by_type = defaultdict(list)
        for issue in report.issues:
            issues_by_type[issue.issue_type].append(issue)

        for issue_type, issues in sorted(issues_by_type.items()):
            marker = '[DROP]' if issue_type in DROP_ISSUE_TYPES else '[info]'
            print(f'\n    {marker} {issue_type}')
            for issue in issues[:3]:
                print(f'      Event {issue.event_index}: '
                      f'{issue.column} — {issue.detail}')
            if len(issues) > 3:
                print(f'      ... and {len(issues) - 3} more')


def write_cleaned_parquet(input_filepath, output_filepath, drop_indices, data=None):
    """Write a cleaned parquet file with corrupt events removed.

    Only events with DROP-level issues are removed. Informational-only
    issues (e.g. extreme_dxy_significance, multi_tau_event) are kept.

    Args:
        input_filepath: Path to the original parquet file.
        output_filepath: Path for the cleaned output file.
        drop_indices: Set of event indices to remove.
        data: Optional pre-loaded awkward array. If None, reads from
            input_filepath (slower — requires re-reading from disk).
    """
    if data is None:
        data = ak.from_parquet(input_filepath)
    number_of_events = len(data)

    # Create a boolean mask: True = keep, False = remove
    keep_mask = np.ones(number_of_events, dtype=bool)
    for index in drop_indices:
        keep_mask[index] = False

    cleaned_data = data[keep_mask]
    number_removed = number_of_events - len(cleaned_data)

    output_directory = os.path.dirname(output_filepath)
    if output_directory and not os.path.exists(output_directory):
        os.makedirs(output_directory)

    ak.to_parquet(cleaned_data, output_filepath, compression='LZ4', compression_level=4)

    print(f'\n  Cleaned file: {output_filepath}')
    print(f'    Removed {number_removed:,} events '
          f'({100 * number_removed / max(number_of_events, 1):.2f}%)')
    print(f'    Remaining: {len(cleaned_data):,} events')
    output_size_megabytes = os.path.getsize(output_filepath) / (1024 * 1024)
    print(f'    File size: {output_size_megabytes:.1f} MB')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def find_parquet_files(path):
    """Find all parquet files in a path (file or directory).

    Args:
        path: Path to a single parquet file or a directory containing them.

    Returns:
        Sorted list of parquet file paths.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If no parquet files are found.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f'Path not found: {path}')

    if os.path.isfile(path):
        if not path.endswith('.parquet'):
            raise ValueError(f'Not a parquet file: {path}')
        return [path]

    parquet_files = sorted(glob.glob(os.path.join(path, '*.parquet')))
    if not parquet_files:
        raise ValueError(f'No parquet files found in: {path}')

    return parquet_files


def default_clean_directory(input_path):
    """Derive a default output directory name for cleaned files.

    Appends '-clean' to the input directory name:
        part/data/low-pt/  →  part/data/low-pt-clean/
        part/data/low-pt   →  part/data/low-pt-clean

    Args:
        input_path: The original input directory path.

    Returns:
        A string path for the cleaned output directory.
    """
    input_path = input_path.rstrip(os.sep)
    return input_path + '-clean'


def print_overall_summary(all_reports):
    """Print a combined summary across all validated files.

    Args:
        all_reports: List of FileReport instances.

    Returns:
        Tuple of (total_events, total_to_drop, total_issues, aggregated_counts).
    """
    total_events = sum(report.number_of_events for report in all_reports)
    total_to_drop = sum(len(report.drop_event_indices) for report in all_reports)
    total_issues = sum(len(report.issues) for report in all_reports)

    # Aggregate issue counts across all files
    aggregated_counts = defaultdict(int)
    for report in all_reports:
        for issue_type, count in report.issue_counts.items():
            aggregated_counts[issue_type] += count

    print(f'\n{"=" * 70}')
    print(f'OVERALL SUMMARY ({len(all_reports)} files)')
    print(f'{"=" * 70}')
    print(f'  Total events:       {total_events:,}')
    print(f'  Events to drop:     {total_to_drop:,} '
          f'({100 * total_to_drop / max(total_events, 1):.2f}%)')
    print(f'  Events after clean: {total_events - total_to_drop:,}')

    if aggregated_counts:
        print(f'\n  Aggregated issue breakdown:')
        for issue_type, count in sorted(
            aggregated_counts.items(), key=lambda item: -item[1]
        ):
            marker = '✗ DROP' if issue_type in DROP_ISSUE_TYPES else '  info'
            print(f'    {marker}  {issue_type:30s} : {count:,}')

    return total_events, total_to_drop, total_issues, aggregated_counts


def main():
    parser = argparse.ArgumentParser(
        description=(
            'Validate data quality of low-pT tau parquet files. '
            'Scans all .parquet files in a directory, prints a combined report, '
            'then asks whether to write cleaned copies with corrupt events removed.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Examples:\n'
            '  # Interactive: validate and optionally clean:\n'
            '  python part/utils/validate_parquet_quality.py part/data/low-pt/\n'
            '\n'
            '  # Non-interactive: auto-clean without prompting:\n'
            '  python part/utils/validate_parquet_quality.py part/data/low-pt/ --yes\n'
            '\n'
            '  # Custom output directory:\n'
            '  python part/utils/validate_parquet_quality.py part/data/low-pt/ '
            '--output-dir /tmp/cleaned/\n'
            '\n'
            '  # With JSON report:\n'
            '  python part/utils/validate_parquet_quality.py part/data/low-pt/ '
            '--report report.json\n'
        ),
    )
    parser.add_argument(
        'input',
        help='Directory containing .parquet files (or a single .parquet file)',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help=(
            'Directory to write cleaned parquet files. '
            'Defaults to <input>-clean/ (e.g. low-pt/ → low-pt-clean/). '
            'Original files are never modified.'
        ),
    )
    parser.add_argument(
        '--yes', '-y',
        action='store_true',
        help='Skip confirmation prompt and write cleaned files automatically',
    )
    parser.add_argument(
        '--report',
        type=str,
        default=None,
        help='Path to write a JSON report with per-event details',
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress per-file progress messages (still prints summaries)',
    )

    arguments = parser.parse_args()

    # --- Phase 1: Find and validate all files ---

    parquet_files = find_parquet_files(arguments.input)
    print(f'Found {len(parquet_files)} parquet file(s) to validate')

    all_reports = []
    for filepath in parquet_files:
        report, data = validate_parquet_file(filepath, verbose=not arguments.quiet)
        print_report_summary(report)
        all_reports.append(report)
        # Free the loaded data immediately — keeping all files in memory
        # simultaneously causes OOM on memory-constrained systems (e.g. lxplus).
        # Files are re-read individually during the cleaning phase if needed.
        del data

    # --- Phase 2: Print combined report ---

    total_events, total_to_drop, total_issues, aggregated_counts = \
        print_overall_summary(all_reports)

    # Identify which files need cleaning (have events to drop)
    reports_needing_clean = [
        report for report in all_reports
        if len(report.drop_event_indices) > 0
    ]

    # Write JSON report if requested
    if arguments.report:
        report_data = {
            'total_files': len(parquet_files),
            'total_events': total_events,
            'total_issues': total_issues,
            'total_events_to_drop': total_to_drop,
            'total_events_after_clean': total_events - total_to_drop,
            'files': [report.to_dict() for report in all_reports],
        }

        report_directory = os.path.dirname(arguments.report)
        if report_directory and not os.path.exists(report_directory):
            os.makedirs(report_directory)

        with open(arguments.report, 'w') as report_file:
            json.dump(report_data, report_file, indent=2)

        print(f'\n  JSON report written to: {arguments.report}')

    # --- Phase 3: Ask user whether to write cleaned files ---

    if not reports_needing_clean:
        print('\n✓ All files are clean — no events to drop.')
        return

    # Determine output directory
    if arguments.output_dir:
        output_directory = arguments.output_dir
    else:
        input_path = arguments.input
        if os.path.isfile(input_path):
            input_path = os.path.dirname(input_path)
        output_directory = default_clean_directory(input_path)

    print(f'\n{len(reports_needing_clean)} file(s) need cleaning '
          f'({total_to_drop:,} events to drop).')
    print(f'Cleaned files will be written to: {output_directory}/')
    print(f'Original files will NOT be modified.')

    # Interactive confirmation (unless --yes)
    if not arguments.yes:
        try:
            answer = input('\nProceed with cleaning? [y/N] ').strip().lower()
        except (EOFError, KeyboardInterrupt):
            print('\nAborted.')
            sys.exit(1)

        if answer not in ('y', 'yes'):
            print('Aborted — no files written.')
            sys.exit(0)

    # --- Phase 4: Write cleaned files ---

    print(f'\nWriting cleaned files...')
    for report in reports_needing_clean:
        filename = os.path.basename(report.filepath)
        output_path = os.path.join(output_directory, filename)
        write_cleaned_parquet(
            report.filepath, output_path, report.drop_event_indices,
        )

    # Also copy files that had no issues (so the output directory is complete)
    reports_already_clean = [
        report for report in all_reports
        if len(report.drop_event_indices) == 0
        and report in all_reports  # exclude non-data files
    ]
    # Only copy clean files if the input was a directory (not a single file)
    if os.path.isdir(arguments.input) and reports_already_clean:
        import shutil
        for report in reports_already_clean:
            filename = os.path.basename(report.filepath)
            output_path = os.path.join(output_directory, filename)
            if not os.path.exists(output_path):
                if not os.path.exists(output_directory):
                    os.makedirs(output_directory)
                shutil.copy2(report.filepath, output_path)
                file_size_megabytes = os.path.getsize(output_path) / (1024 * 1024)
                print(f'\n  Copied (already clean): {output_path} '
                      f'({file_size_megabytes:.1f} MB)')

    print(f'\n✓ Done. Cleaned dataset: {output_directory}/')


if __name__ == '__main__':
    main()
