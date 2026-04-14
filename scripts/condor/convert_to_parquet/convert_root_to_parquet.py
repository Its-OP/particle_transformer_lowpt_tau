"""
Convert low-pT tau dataset from CMS NanoAOD ROOT format to sharded Parquet.

Reads multiple merged_noBKstar_batch*.root files sequentially, applies:
  1. Event quality cleaning (corrupt tracks, NaN/Inf)
  2. Pion track selection (|pdgId| == 211)
  3. pT cutoff: drops all pion tracks with pT below threshold
  4. GT pion filter: keeps only events with exactly N tau-origin pions
     surviving the pT cutoff (default: 3)

Output is split into train/ and val/ subdirectories, each containing
sharded Parquet files with jagged awkward arrays consumable by weaver.

After writing, all output files are validated against the pT cutoff
and GT pion count constraints.

Usage:
    python convert_root_to_parquet.py \\
        --input-dir ./root \\
        --output-dir ./parquet_low_pt_cutoff \\
        --train-events 300000 --val-events 75000 \\
        --pt-cutoff 0.5 --gt-pions 3

References:
  - Dataset description: part/data/low-pt/description.tex
  - YAML config: part/data/low-pt/lowpt_tau_trackfinder.yaml

Versions:
  - uproot >= 5.1
  - awkward >= 2.8
  - pyarrow >= 22.0
"""

import os
import re
import sys
import glob
import time
import argparse
from collections import Counter

import numpy as np
import awkward as ak
import uproot


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PION_PDG_ID = 211

ROOT_TREE_NAME = 'Events'

# Complete mapping from ROOT branch names to output column names for track features.
# Only raw features are stored — derived features (log pT, 4-vectors, tanh(dxy),
# vertex displacements, etc.) are computed by weaver at runtime via YAML new_variables.
TRACK_BRANCH_MAP_ALL = {
    'Track_pt': 'track_pt',
    'Track_eta': 'track_eta',
    'Track_phi': 'track_phi',
    'Track_mass': 'track_mass',
    'Track_charge': 'track_charge',
    'Track_dxy': 'track_dxy',
    'Track_dxyS': 'track_dxy_significance',
    'Track_dz': 'track_dz',
    'Track_dzS': 'track_dz_significance',
    'Track_dzTrg': 'track_dz_trigger',
    'Track_normChi2': 'track_norm_chi2',
    'Track_nValidHits': 'track_n_valid_hits',
    'Track_nValidPixelHits': 'track_n_valid_pixel_hits',
    'Track_ptErr': 'track_pt_error',
    'Track_DCASig': 'track_dca_significance',
    'Track_covQopQop': 'track_covariance_qop_qop',
    'Track_covQopLam': 'track_covariance_qop_lambda',
    'Track_covQopPhi': 'track_covariance_qop_phi',
    'Track_covLamLam': 'track_covariance_lambda_lambda',
    'Track_covLamPhi': 'track_covariance_lambda_phi',
    'Track_covPhiPhi': 'track_covariance_phi_phi',
    'Track_vx': 'track_vertex_x',
    'Track_vy': 'track_vertex_y',
    'Track_vz': 'track_vertex_z',
    'Track_trackFromTau': 'track_label_from_tau',
}

# Default output columns matching the YAML data config
# (part/data/low-pt/lowpt_tau_trackfinder.yaml).
DEFAULT_COLUMNS = [
    # Kinematic features
    'track_pt',
    'track_eta',
    'track_phi',
    'track_charge',
    # Displacement features
    'track_dxy_significance',
    'track_dz_significance',
    # Track quality / measurement precision
    'track_pt_error',
    'track_n_valid_pixel_hits',
    'track_dca_significance',
    # Helix fit precision (covariance matrix elements)
    'track_covariance_phi_phi',
    'track_covariance_lambda_lambda',
    'track_norm_chi2',
    # Per-track binary label: 1 = tau-origin pion, 0 = background
    'track_label_from_tau',
]

# Event-level branches (scalars per event).
# PV_x/y/z: primary vertex coordinates (float).
# run, event, luminosityBlock: CMS event identifiers for tracing back to ROOT source.
# source_batch_id, source_microbatch_id: added during merge_batches to uniquely
#   identify the tau candidate together with (run, event, luminosityBlock).
# source_batch_id, source_microbatch_id: added during merge_batches to
#   disambiguate tau candidates — (run, event, luminosityBlock) is NOT unique
#   because the same CMS event appears in multiple microbatch files.
EVENT_BRANCHES = [
    'PV_x', 'PV_y', 'PV_z',
    'run', 'event', 'luminosityBlock',
    'source_batch_id', 'source_microbatch_id',
]

# Integer columns for type casting (int32 instead of float32).
TRACK_INTEGER_COLUMNS = {
    'track_charge', 'track_n_valid_hits',
    'track_n_valid_pixel_hits', 'track_label_from_tau',
}


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def discover_root_files(input_dir, pattern='merged_noBKstar_batch*.root'):
    """Find and sort ROOT files by batch number.

    Args:
        input_dir: Directory containing ROOT files.
        pattern: Glob pattern for matching files.

    Returns:
        List of file paths sorted numerically by batch number.

    Raises:
        FileNotFoundError: If no matching files are found.
    """
    matches = glob.glob(os.path.join(input_dir, pattern))
    if not matches:
        raise FileNotFoundError(
            f'No ROOT files matching "{pattern}" found in {input_dir}'
        )

    def batch_number(filepath):
        match = re.search(r'batch(\d+)', os.path.basename(filepath))
        return int(match.group(1)) if match else 0

    matches.sort(key=batch_number)
    return matches


# ---------------------------------------------------------------------------
# Data reading
# ---------------------------------------------------------------------------

def load_root_tree(filepath):
    """Open a ROOT file and return the Events TTree.

    Args:
        filepath: Path to the ROOT file.

    Returns:
        An uproot TTree object for the Events tree.

    Raises:
        FileNotFoundError: If the ROOT file does not exist.
        KeyError: If the Events tree is not found in the file.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f'ROOT file not found: {filepath}')

    root_file = uproot.open(filepath)

    if ROOT_TREE_NAME not in root_file:
        available_keys = list(root_file.keys())
        raise KeyError(
            f'Tree "{ROOT_TREE_NAME}" not found in {filepath}. '
            f'Available keys: {available_keys}'
        )

    return root_file[ROOT_TREE_NAME]


def read_and_filter_pion_tracks(tree, track_branch_map, entry_start=None, entry_stop=None):
    """Read Track branches and filter to keep only charged pions (|pdgId| == 211).

    The filtering applies a boolean mask per event, so the output jagged arrays
    have variable length (only pion entries survive).

    Also drops events where any track has corrupt eta (|eta| > 5 or NaN).
    These are simulation artefacts (e.g. event 11960 in merged_noBKstar.root
    has 24 tracks with eta up to ~7e31 and 2 with NaN).

    Args:
        tree: An uproot TTree object.
        track_branch_map: Dict mapping ROOT branch names to output column names.
        entry_start: Optional first entry index to read.
        entry_stop: Optional last entry index (exclusive) to read.

    Returns:
        Tuple of (filtered_data, clean_event_mask) where:
          - filtered_data: dict mapping ROOT branch names to filtered awkward arrays.
          - clean_event_mask: boolean numpy array (True = event is clean).
    """
    # Read all track branches needed, plus pdgId and eta for filtering
    branches_to_read = list(track_branch_map.keys()) + ['Track_pdgId']
    if 'Track_eta' not in branches_to_read:
        branches_to_read.append('Track_eta')
    track_data = tree.arrays(
        expressions=branches_to_read,
        entry_start=entry_start,
        entry_stop=entry_stop,
    )

    # ---- Event cleaning ----
    # Drop events with data quality issues that would corrupt training.
    clean_event_mask = np.ones(len(track_data), dtype=bool)
    drop_reasons = {}

    # 1. Corrupt eta: |eta| > 5 or NaN (CMS tracking covers |eta| < 2.5)
    eta = track_data['Track_eta']
    corrupt_eta = ak.to_numpy(ak.any((np.abs(eta) > 5.0) | np.isnan(eta), axis=1))
    if corrupt_eta.any():
        drop_reasons['corrupt_eta'] = int(corrupt_eta.sum())
        clean_event_mask &= ~corrupt_eta

    # 2. Unphysical phi: |phi| > pi
    if 'Track_phi' in track_data.fields:
        phi = track_data['Track_phi']
        corrupt_phi = ak.to_numpy(ak.any(np.abs(phi) > (np.pi + 0.01), axis=1))
        if corrupt_phi.any():
            drop_reasons['corrupt_phi'] = int(corrupt_phi.sum())
            clean_event_mask &= ~corrupt_phi

    # 3. Unphysical pT: <= 0, > 500 GeV, or float16 max (65504)
    pt = track_data['Track_pt']
    bad_pt = ak.to_numpy(ak.any((pt <= 0) | (pt > 500) | (pt >= 65504), axis=1))
    if bad_pt.any():
        drop_reasons['corrupt_pt'] = int(bad_pt.sum())
        clean_event_mask &= ~bad_pt

    # 4. Invalid charge: not +/-1
    charge = track_data['Track_charge']
    bad_charge = ak.to_numpy(ak.any((charge != 1) & (charge != -1), axis=1))
    if bad_charge.any():
        drop_reasons['invalid_charge'] = int(bad_charge.sum())
        clean_event_mask &= ~bad_charge

    # 5. NaN, Inf, or extreme values (|val| > 1e10) in any requested column
    for root_branch_name in track_branch_map:
        branch_data = track_data[root_branch_name]
        has_nan = ak.any(np.isnan(branch_data), axis=1)
        has_inf = ak.any(np.isinf(branch_data), axis=1)
        has_extreme = ak.any(np.abs(branch_data) > 1e10, axis=1)
        artifact_mask = ak.to_numpy(has_nan | has_inf | has_extreme)
        new_drops = int(np.sum(artifact_mask & clean_event_mask))
        if new_drops > 0:
            drop_reasons[f'artifacts_{root_branch_name}'] = new_drops
            clean_event_mask &= ~artifact_mask

    num_dropped = int(np.sum(~clean_event_mask))
    if num_dropped > 0 and os.environ.get('CONVERT_VERBOSE'):
        print(f'    Dropping {num_dropped} event(s):')
        for reason, count in sorted(drop_reasons.items(), key=lambda x: -x[1]):
            print(f'      {reason}: {count}')

    track_data = track_data[clean_event_mask]

    # Filter: keep only charged pions where |pdgId| == 211
    pion_mask = abs(track_data['Track_pdgId']) == PION_PDG_ID

    filtered_data = {}
    for root_branch_name in track_branch_map:
        filtered_data[root_branch_name] = track_data[root_branch_name][pion_mask]

    return filtered_data, clean_event_mask


def read_event_info(tree, clean_event_mask, entry_start=None, entry_stop=None):
    """Read event-level scalar branches (primary vertex), filtered to clean events.

    Args:
        tree: An uproot TTree object.
        clean_event_mask: Boolean numpy array from read_and_filter_pion_tracks
            (True = event is clean, False = event dropped).
        entry_start: Optional first entry index to read.
        entry_stop: Optional last entry index (exclusive) to read.

    Returns:
        A dict mapping branch names to numpy arrays (one value per clean event).
    """
    available = set(tree.keys())
    branches_to_read = [b for b in EVENT_BRANCHES if b in available]
    event_data = tree.arrays(
        expressions=branches_to_read,
        entry_start=entry_start,
        entry_stop=entry_stop,
    )

    result = {}
    n_clean = int(clean_event_mask.sum())
    for branch in EVENT_BRANCHES:
        if branch in available:
            result[branch] = ak.to_numpy(event_data[branch])[clean_event_mask]
        else:
            # Fill missing source ID branches with -1 (old .root files
            # produced before merge_batches added these columns).
            result[branch] = np.full(n_clean, -1, dtype=np.int32)
    return result


# ---------------------------------------------------------------------------
# Track filtering
# ---------------------------------------------------------------------------

def filter_tracks_by_pt(track_data, pt_key, pt_cutoff):
    """Remove pion tracks with transverse momentum below cutoff.

    Applies a per-track boolean mask: tracks with pT < pt_cutoff are dropped.
    All branches in track_data are filtered consistently using the same mask.

    Args:
        track_data: Dict mapping ROOT branch names to jagged awkward arrays.
        pt_key: Key for the pT branch (e.g. 'Track_pt').
        pt_cutoff: Minimum pT threshold in GeV (inclusive: pT >= cutoff kept).

    Returns:
        Dict with same keys, low-pT tracks removed from each event.
    """
    # Boolean mask: True = track survives, False = track dropped
    pt_mask = track_data[pt_key] >= pt_cutoff
    return {key: array[pt_mask] for key, array in track_data.items()}


def filter_events_by_gt_pion_count(track_data, label_key, required_gt_pions):
    """Keep only events with exactly the required number of tau-origin (GT) pions.

    Counts tracks with label == 1 per event. Events where this count does not
    equal required_gt_pions are dropped.

    Args:
        track_data: Dict mapping branch names to jagged awkward arrays.
        label_key: Key for the track label branch (1 = GT pion, 0 = background).
        required_gt_pions: Exact number of GT pions required per event.

    Returns:
        Tuple of (filtered_track_data, event_mask) where:
          - filtered_track_data: dict with only qualifying events.
          - event_mask: boolean numpy array (True = event kept).
    """
    labels = track_data[label_key]
    # Sum of (label == 1) per event gives GT pion count
    gt_count_per_event = ak.sum(labels == 1, axis=1)
    event_mask = ak.to_numpy(gt_count_per_event == required_gt_pions)
    filtered = {key: array[event_mask] for key, array in track_data.items()}
    return filtered, event_mask


# ---------------------------------------------------------------------------
# Branch renaming and type casting
# ---------------------------------------------------------------------------

def rename_branches(data_dict, branch_map):
    """Rename dictionary keys from ROOT branch names to output column names.

    Args:
        data_dict: Dict mapping ROOT branch names to arrays.
        branch_map: Dict mapping ROOT names to desired output names.

    Returns:
        A new dict with renamed keys.
    """
    return {branch_map[root_name]: array for root_name, array in data_dict.items()}


def cast_to_float32(data_dict, integer_columns=None):
    """Cast all arrays to float32, except specified integer columns which use int32.

    Args:
        data_dict: Dict mapping column names to awkward or numpy arrays.
        integer_columns: Set of column names that should remain int32.

    Returns:
        A new dict with consistently typed arrays.
    """
    if integer_columns is None:
        integer_columns = set()

    casted = {}
    for column_name, array in data_dict.items():
        if column_name in integer_columns:
            if isinstance(array, ak.Array):
                casted[column_name] = ak.values_astype(array, np.int32)
            else:
                casted[column_name] = array.astype(np.int32)
        else:
            if isinstance(array, ak.Array):
                casted[column_name] = ak.values_astype(array, np.float32)
            else:
                casted[column_name] = array.astype(np.float32)

    return casted


# ---------------------------------------------------------------------------
# Output assembly
# ---------------------------------------------------------------------------

def build_output_table(track_features, event_info, pion_counts):
    """Combine track features and event-level info into output dict.

    Args:
        track_features: Dict of renamed, typed track columns (jagged arrays).
        event_info: Dict of event-level scalars (numpy arrays).
        pion_counts: Numpy array with number of pion tracks per event.

    Returns:
        A dict ready to be wrapped in ak.Array and written to Parquet.
    """
    output = {}

    # Event-level scalars
    output['event_primary_vertex_x'] = event_info['PV_x'].astype(np.float32)
    output['event_primary_vertex_y'] = event_info['PV_y'].astype(np.float32)
    output['event_primary_vertex_z'] = event_info['PV_z'].astype(np.float32)
    output['event_n_tracks'] = pion_counts.astype(np.int32)

    # CMS event identifiers — for tracing events back to ROOT source.
    # (run, luminosityBlock, event) is NOT unique per row because the
    # upstream CMSSW analysis writes one entry per tau candidate.
    # source_batch_id + source_microbatch_id (added during merge_batches)
    # disambiguate candidates within the same CMS event.
    output['event_run'] = event_info['run'].astype(np.int32)
    output['event_id'] = event_info['event'].astype(np.int64)
    output['event_luminosity_block'] = event_info['luminosityBlock'].astype(np.int32)
    output['source_batch_id'] = event_info['source_batch_id'].astype(np.int32)
    output['source_microbatch_id'] = event_info['source_microbatch_id'].astype(np.int32)

    # Track (pion) features and per-track label — jagged arrays
    output.update(track_features)

    return output


def build_track_branch_map(output_columns=None):
    """Build a ROOT-to-output branch map for the requested output columns.

    Args:
        output_columns: List of output column names to include, or None
            for DEFAULT_COLUMNS.

    Returns:
        Dict mapping ROOT branch names to output column names.

    Raises:
        ValueError: If a requested column is not in TRACK_BRANCH_MAP_ALL.
    """
    if output_columns is None:
        output_columns = DEFAULT_COLUMNS

    reverse_map = {v: k for k, v in TRACK_BRANCH_MAP_ALL.items()}

    track_branch_map = {}
    for output_name in output_columns:
        if output_name not in reverse_map:
            available = sorted(TRACK_BRANCH_MAP_ALL.values())
            raise ValueError(
                f'Unknown output column: "{output_name}". '
                f'Available columns: {available}'
            )
        track_branch_map[reverse_map[output_name]] = output_name

    return track_branch_map


# ---------------------------------------------------------------------------
# Sharded Parquet output
# ---------------------------------------------------------------------------

def write_parquet_shard(output_array, filepath):
    """Write an awkward array to a single Parquet file with LZ4 compression.

    Creates parent directories if they don't exist.

    Args:
        output_array: Awkward array to write.
        filepath: Output file path.
    """
    output_dir = os.path.dirname(filepath)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    ak.to_parquet(output_array, filepath, compression='LZ4', compression_level=4)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_parquet_output(output_dir, pt_cutoff, required_gt_pions):
    """Validate all Parquet files in a directory against quality constraints.

    Reads each file and checks:
      1. Every pion track has pT >= pt_cutoff.
      2. Every event has exactly required_gt_pions tracks with label == 1.

    Args:
        output_dir: Directory containing .parquet files.
        pt_cutoff: Minimum pT in GeV.
        required_gt_pions: Required GT pion count per event.

    Raises:
        FileNotFoundError: If no parquet files are found.
        ValueError: If any constraint is violated.
    """
    parquet_files = sorted(glob.glob(os.path.join(output_dir, '*.parquet')))
    if not parquet_files:
        raise FileNotFoundError(f'No parquet files found in {output_dir}')

    total_events = 0
    total_tracks = 0

    for filepath in parquet_files:
        data = ak.from_parquet(filepath)
        n_events = len(data)
        total_events += n_events

        # Validate pT >= cutoff for ALL pion tracks
        pt = data['track_pt']
        flat_pt = ak.to_numpy(ak.flatten(pt))
        n_tracks = len(flat_pt)
        total_tracks += n_tracks
        min_pt = float(flat_pt.min())

        if min_pt < pt_cutoff:
            raise ValueError(
                f'pT violation in {os.path.basename(filepath)}: '
                f'found track with pT={min_pt:.6f} < cutoff {pt_cutoff}'
            )

        # Validate exactly required_gt_pions GT pions per event
        labels = data['track_label_from_tau']
        gt_counts = ak.to_numpy(ak.sum(labels == 1, axis=1))
        bad_mask = gt_counts != required_gt_pions
        n_bad = int(bad_mask.sum())

        if n_bad > 0:
            bad_counts = sorted(set(gt_counts[bad_mask].tolist()))
            raise ValueError(
                f'GT pion count violation in {os.path.basename(filepath)}: '
                f'{n_bad} events have wrong count. '
                f'Found: {bad_counts}, required: {required_gt_pions}'
            )

        # Validate CMS event identifiers and source IDs exist with correct types
        for id_col, expected_kind in [
            ('event_run', 'i'), ('event_id', 'i'), ('event_luminosity_block', 'i'),
            ('source_batch_id', 'i'), ('source_microbatch_id', 'i'),
        ]:
            if id_col not in data.fields:
                raise ValueError(
                    f'Missing column {id_col} in {os.path.basename(filepath)}'
                )
            id_values = ak.to_numpy(data[id_col])
            if id_values.dtype.kind != expected_kind:
                raise ValueError(
                    f'Wrong dtype for {id_col} in {os.path.basename(filepath)}: '
                    f'{id_values.dtype} (expected integer)'
                )

        print(f'    OK {os.path.basename(filepath)}: '
              f'{n_events} events, {n_tracks} tracks, min pT={min_pt:.4f}')

    print(f'    Total: {total_events} events, {total_tracks} tracks — all valid')


# ---------------------------------------------------------------------------
# Progress display
# ---------------------------------------------------------------------------

def print_progress(total_scanned, total_qualifying, total_needed,
                   current_shard_index, total_shards, start_time):
    """Print a single-line progress indicator that updates in place.

    Shows: events scanned, qualifying collected / needed, acceptance rate,
    shards written, and elapsed time.
    """
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    if total_scanned > 0:
        acceptance_percent = 100.0 * total_qualifying / total_scanned
    else:
        acceptance_percent = 0.0

    line = (
        f'  [{minutes:02d}:{seconds:02d}] '
        f'Scanned {total_scanned:,} | '
        f'Qualifying {total_qualifying:,}/{total_needed:,} '
        f'({acceptance_percent:.1f}% accept) | '
        f'Shards {current_shard_index}/{total_shards}'
    )

    # Overwrite previous line; pad with spaces to clear leftover characters
    sys.stdout.write(f'\r{line:<100}')
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Main conversion pipeline
# ---------------------------------------------------------------------------

def convert(input_dir, output_dir, train_events, val_events,
            train_files, val_files, pt_cutoff, required_gt_pions,
            chunk_size, output_columns):
    """Convert ROOT files to sharded Parquet with pT cutoff and GT pion filter.

    Processes ROOT files sequentially in chunks, applying quality cleaning,
    pion selection, pT cutoff, and GT pion count filtering. Output is split
    into train/val sharded Parquet files. All output is validated after writing.

    Memory is bounded: events are accumulated in a buffer and flushed to disk
    as soon as a shard's worth of events is collected.

    Args:
        input_dir: Directory containing merged_noBKstar_batch*.root files.
        output_dir: Base output directory (creates train/ and val/ subdirs).
        train_events: Total number of training events to collect.
        val_events: Total number of validation events to collect.
        train_files: Number of train parquet shards.
        val_files: Number of val parquet shards.
        pt_cutoff: Minimum pT in GeV for all pion tracks.
        required_gt_pions: Exact number of GT pions required per event.
        chunk_size: Events per ROOT reading chunk.
        output_columns: List of output column names, None for defaults, 'all' for all.
    """
    # --- Validate configuration ---
    if train_events % train_files != 0:
        raise ValueError(
            f'train_events ({train_events}) must be evenly divisible by '
            f'train_files ({train_files})'
        )
    if val_events % val_files != 0:
        raise ValueError(
            f'val_events ({val_events}) must be evenly divisible by '
            f'val_files ({val_files})'
        )

    train_events_per_file = train_events // train_files
    val_events_per_file = val_events // val_files
    total_needed = train_events + val_events

    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')

    # Build shard plan: ordered list of (directory, filename, events_per_shard)
    # Train shards are filled first, then val shards.
    shard_plan = []
    for i in range(train_files):
        shard_plan.append((train_dir, f'train_{i:03d}.parquet', train_events_per_file))
    for i in range(val_files):
        shard_plan.append((val_dir, f'val_{i:03d}.parquet', val_events_per_file))

    # Build branch map
    if output_columns == 'all':
        track_branch_map = dict(TRACK_BRANCH_MAP_ALL)
    else:
        track_branch_map = build_track_branch_map(output_columns)

    # Discover input files
    root_files = discover_root_files(input_dir)

    print(f'Found {len(root_files)} ROOT files in {input_dir}')
    print(f'Extracting {len(track_branch_map)} track columns')
    print(f'pT cutoff: >= {pt_cutoff} GeV | Required GT pions: {required_gt_pions}')
    print(f'Target: {train_events} train + {val_events} val = {total_needed} events')
    print(f'Output: {train_files} x {train_events_per_file} train + '
          f'{val_files} x {val_events_per_file} val shards')
    print()

    # --- Process ROOT files ---
    current_shard_index = 0
    buffer = []
    buffer_count = 0
    total_scanned = 0
    total_qualifying = 0
    total_dropped_quality = 0
    total_dropped_filter = 0
    start_time = time.time()

    for file_index, root_filepath in enumerate(root_files):
        if current_shard_index >= len(shard_plan):
            break

        basename = os.path.basename(root_filepath)
        # Clear progress line before printing file header
        sys.stdout.write('\r' + ' ' * 100 + '\r')
        print(f'[{file_index + 1}/{len(root_files)}] {basename}')
        tree = load_root_tree(root_filepath)
        number_of_entries = tree.num_entries
        file_qualifying = 0

        for chunk_start in range(0, number_of_entries, chunk_size):
            if current_shard_index >= len(shard_plan):
                break

            chunk_end = min(chunk_start + chunk_size, number_of_entries)
            chunk_events = chunk_end - chunk_start
            total_scanned += chunk_events

            # Step 1: Read pion tracks + quality cleaning
            raw_track_data, clean_event_mask = read_and_filter_pion_tracks(
                tree, track_branch_map, chunk_start, chunk_end,
            )
            event_info = read_event_info(
                tree, clean_event_mask, chunk_start, chunk_end,
            )

            number_of_clean = int(clean_event_mask.sum())
            total_dropped_quality += (chunk_events - number_of_clean)

            if number_of_clean == 0:
                print_progress(total_scanned, total_qualifying, total_needed,
                               current_shard_index, len(shard_plan), start_time)
                continue

            # Step 2: Apply pT cutoff — drop all pion tracks with pT < threshold
            pt_filtered_data = filter_tracks_by_pt(
                raw_track_data, 'Track_pt', pt_cutoff,
            )

            # Step 3: Keep only events with exactly required_gt_pions GT pions
            # (counted after pT cutoff, so events where a GT pion was below
            # the cutoff are automatically excluded)
            gt_filtered_data, gt_event_mask = filter_events_by_gt_pion_count(
                pt_filtered_data, 'Track_trackFromTau', required_gt_pions,
            )

            number_after_filter = int(gt_event_mask.sum())
            total_dropped_filter += (number_of_clean - number_after_filter)

            if number_after_filter == 0:
                print_progress(total_scanned, total_qualifying, total_needed,
                               current_shard_index, len(shard_plan), start_time)
                continue

            total_qualifying += number_after_filter
            file_qualifying += number_after_filter

            # Filter event info to matching events
            event_info_filtered = {
                key: value[gt_event_mask] for key, value in event_info.items()
            }

            # Compute pion counts per event (after pT cutoff + GT filter)
            first_key = next(iter(gt_filtered_data))
            pion_counts_per_event = ak.to_numpy(
                ak.num(gt_filtered_data[first_key], axis=1)
            )

            # Rename ROOT branch names -> output column names, cast types
            track_features = rename_branches(gt_filtered_data, track_branch_map)
            track_features = cast_to_float32(
                track_features, integer_columns=TRACK_INTEGER_COLUMNS,
            )

            chunk_output = build_output_table(
                track_features, event_info_filtered, pion_counts_per_event,
            )
            buffer.append(ak.Array(chunk_output))
            buffer_count += number_after_filter

            # Flush full shards from buffer to disk
            while (current_shard_index < len(shard_plan)
                   and buffer_count >= shard_plan[current_shard_index][2]):
                shard_dir, shard_filename, shard_n_events = shard_plan[current_shard_index]

                all_buffered = ak.concatenate(buffer)
                shard_data = all_buffered[:shard_n_events]
                remainder = all_buffered[shard_n_events:]

                shard_path = os.path.join(shard_dir, shard_filename)
                write_parquet_shard(shard_data, shard_path)
                shard_size_mb = os.path.getsize(shard_path) / (1024 * 1024)
                # Clear progress line, print shard info, then resume progress
                sys.stdout.write('\r' + ' ' * 100 + '\r')
                print(f'  -> {shard_filename} '
                      f'({shard_n_events:,} events, {shard_size_mb:.1f} MB)')

                if len(remainder) > 0:
                    buffer = [remainder]
                    buffer_count = len(remainder)
                else:
                    buffer = []
                    buffer_count = 0

                current_shard_index += 1

            print_progress(total_scanned, total_qualifying, total_needed,
                           current_shard_index, len(shard_plan), start_time)

            # Free chunk memory
            del raw_track_data, pt_filtered_data, gt_filtered_data
            del event_info, event_info_filtered, track_features, chunk_output

        # Clear progress line for per-file summary
        sys.stdout.write('\r' + ' ' * 100 + '\r')
        print(f'  {basename}: {file_qualifying:,} qualifying / {number_of_entries:,} total '
              f'(cumulative: {total_qualifying:,})')

    # --- Summary ---
    elapsed = time.time() - start_time
    elapsed_minutes = int(elapsed // 60)
    elapsed_seconds = int(elapsed % 60)

    print(f'\n{"=" * 50}')
    print(f'Conversion Summary')
    print(f'{"=" * 50}')
    print(f'  Elapsed time:            {elapsed_minutes:02d}:{elapsed_seconds:02d}')
    print(f'  ROOT files processed:    {len(root_files)}')
    print(f'  Events scanned:          {total_scanned:,}')
    print(f'  Dropped (quality):       {total_dropped_quality:,}')
    print(f'  Dropped (pT cutoff/GT):  {total_dropped_filter:,}')
    print(f'  Qualifying events:       {total_qualifying:,}')
    print(f'  Shards written:          {current_shard_index}/{len(shard_plan)}')

    if current_shard_index < len(shard_plan):
        print(f'\n  WARNING: Not enough qualifying events!')
        print(f'  Collected {total_qualifying}, needed {total_needed}.')
        print(f'  Only {current_shard_index}/{len(shard_plan)} shards written.')
        return

    # --- Validate output ---
    print(f'\n{"=" * 50}')
    print(f'Validating Output')
    print(f'{"=" * 50}')

    print(f'  Train ({train_dir}):')
    validate_parquet_output(train_dir, pt_cutoff, required_gt_pions)

    print(f'  Val ({val_dir}):')
    validate_parquet_output(val_dir, pt_cutoff, required_gt_pions)

    print(f'\nDone. Output directory: {output_dir}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    available_columns = sorted(TRACK_BRANCH_MAP_ALL.values())

    parser = argparse.ArgumentParser(
        description='Convert low-pT tau ROOT dataset to sharded Parquet with pT cutoff.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f'Available track columns:\n  ' + '\n  '.join(available_columns),
    )

    # Input / output paths
    parser.add_argument(
        '--input-dir',
        default='/eos/user/o/oprostak/tau_data/root1',
        help='Directory containing merged_noBKstar_batch*.root files (default: %(default)s)',
    )
    parser.add_argument(
        '--output-dir',
        default='/eos/user/o/oprostak/tau_data/parquet_low_pt_cutoff',
        help='Base output directory (creates train/ and val/ subdirs) (default: %(default)s)',
    )

    # Event counts and sharding
    parser.add_argument(
        '--train-events', type=int, default=300000,
        help='Total number of training events (default: %(default)s)',
    )
    parser.add_argument(
        '--val-events', type=int, default=75000,
        help='Total number of validation events (default: %(default)s)',
    )
    parser.add_argument(
        '--train-files', type=int, default=10,
        help='Number of train parquet shards (default: %(default)s)',
    )
    parser.add_argument(
        '--val-files', type=int, default=10,
        help='Number of val parquet shards (default: %(default)s)',
    )

    # Filtering
    parser.add_argument(
        '--pt-cutoff', type=float, default=0.5,
        help='Minimum pT in GeV for all pion tracks (default: %(default)s)',
    )
    parser.add_argument(
        '--gt-pions', type=int, default=3,
        help='Required number of GT (tau-origin) pions per event (default: %(default)s)',
    )

    # Processing
    parser.add_argument(
        '--chunk-size', type=int, default=2500,
        help='Events per ROOT reading chunk (default: %(default)s)',
    )

    # Column selection
    column_group = parser.add_mutually_exclusive_group()
    column_group.add_argument(
        '--columns', type=str, default=None,
        help=f'Comma-separated output columns (default: {",".join(DEFAULT_COLUMNS)})',
    )
    column_group.add_argument(
        '--all-columns', action='store_true',
        help='Extract all available track columns',
    )

    arguments = parser.parse_args()

    # Resolve column selection
    if arguments.all_columns:
        output_columns = 'all'
    elif arguments.columns is not None:
        output_columns = [c.strip() for c in arguments.columns.split(',')]
    else:
        output_columns = None  # uses DEFAULT_COLUMNS

    convert(
        input_dir=arguments.input_dir,
        output_dir=arguments.output_dir,
        train_events=arguments.train_events,
        val_events=arguments.val_events,
        train_files=arguments.train_files,
        val_files=arguments.val_files,
        pt_cutoff=arguments.pt_cutoff,
        required_gt_pions=arguments.gt_pions,
        chunk_size=arguments.chunk_size,
        output_columns=output_columns,
    )


if __name__ == '__main__':
    main()
