"""
Convert low-pT tau dataset from CMS NanoAOD ROOT format to Parquet.

The dataset contains B-meson decays to tau-lepton pairs and resonances
(B⁰ → τ⁺τ⁻ K⁰* or Bₛ⁰ → τ⁺τ⁻ φ), generated with Pythia 8 and EvtGen,
with CMS detector simulation via CMSSW 14.

Each event is one B-meson decay. One τ is forced muonic (τ_μ), the other
decays hadronically (τ_h). The task is per-track binary classification:
identify which pion tracks originate from the tau decay.

This script extracts:
  - Pion tokens: all charged pion tracks (|pdgId| == 211) with kinematic,
    impact parameter, track quality, and covariance features.
  - Per-track label: Track_trackFromTau — 1 if the pion track originates
    from a tau decay, 0 otherwise (background / pile-up).
  - Event-level scalars: primary vertex coordinates and pion track count.

Output is a Parquet file with jagged awkward arrays, directly consumable
by the weaver framework.

References:
  - Dataset description: part/data/low-pt/description.tex
  - Conversion pattern: part/utils/convert_qg_datasets.py

Versions:
  - uproot >= 5.1
  - awkward >= 2.8
  - pyarrow >= 22.0
"""

import os
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

# Default subset of output columns to extract. These are the columns needed
# for the hierarchical graph backbone (7 input features + label).
# Use --columns to override with a comma-separated list of output column names,
# or --all-columns to extract all available columns.
DEFAULT_COLUMNS = [
    'track_pt',
    'track_eta',
    'track_phi',
    'track_charge',
    'track_dxy_significance',
    'track_label_from_tau',
]

# Event-level branches (scalars per event).
EVENT_BRANCHES = ['PV_x', 'PV_y', 'PV_z']


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

    Args:
        tree: An uproot TTree object.
        track_branch_map: Dict mapping ROOT branch names to output column names.
        entry_start: Optional first entry index to read.
        entry_stop: Optional last entry index (exclusive) to read.

    Returns:
        A dict mapping ROOT branch names to filtered awkward arrays.
    """
    # Read all track branches needed, plus pdgId for filtering
    branches_to_read = list(track_branch_map.keys()) + ['Track_pdgId']
    track_data = tree.arrays(
        expressions=branches_to_read,
        entry_start=entry_start,
        entry_stop=entry_stop,
    )

    # Filter: keep only charged pions where |pdgId| == 211
    pion_mask = abs(track_data['Track_pdgId']) == PION_PDG_ID

    filtered_data = {}
    for root_branch_name in track_branch_map:
        filtered_data[root_branch_name] = track_data[root_branch_name][pion_mask]

    return filtered_data


def read_event_info(tree, entry_start=None, entry_stop=None):
    """Read event-level scalar branches (primary vertex).

    Args:
        tree: An uproot TTree object.
        entry_start: Optional first entry index to read.
        entry_stop: Optional last entry index (exclusive) to read.

    Returns:
        A dict mapping branch names to numpy arrays (one value per event).
    """
    event_data = tree.arrays(
        expressions=EVENT_BRANCHES,
        entry_start=entry_start,
        entry_stop=entry_stop,
    )

    return {branch: ak.to_numpy(event_data[branch]) for branch in EVENT_BRANCHES}


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

    The per-track label (track_label_from_tau) is included in track_features,
    having been read from Track_trackFromTau and filtered to pion tracks.

    Args:
        track_features: Dict of renamed, typed track columns (jagged arrays).
            Includes track_label_from_tau as the per-track binary label.
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

    # Track (pion) features and per-track label — jagged arrays
    output.update(track_features)

    return output


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def build_track_branch_map(output_columns=None):
    """Build a ROOT→output branch map for the requested output columns.

    Args:
        output_columns: List of output column names to include, or None
            for DEFAULT_COLUMNS. Use list(TRACK_BRANCH_MAP_ALL.values())
            for all available columns.

    Returns:
        Dict mapping ROOT branch names to output column names.

    Raises:
        ValueError: If a requested column is not in TRACK_BRANCH_MAP_ALL.
    """
    if output_columns is None:
        output_columns = DEFAULT_COLUMNS

    # Build reverse map: output_name → root_name
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


def convert(input_filepath, output_filepath, entry_start=None, entry_stop=None,
            output_columns=None):
    """Convert a ROOT file to Parquet format.

    Reads the NanoAOD Events tree, filters tracks to charged pions,
    and writes a Parquet file with jagged awkward arrays. Per-track
    tau-origin labels (Track_trackFromTau) are included as a jagged
    int32 column alongside the track features.

    Args:
        input_filepath: Path to the input ROOT file.
        output_filepath: Path for the output Parquet file.
        entry_start: Optional first entry index to process.
        entry_stop: Optional last entry index (exclusive) to process.
        output_columns: List of output column names to extract, or None
            for DEFAULT_COLUMNS. Use 'all' for all available columns.
    """
    # Build branch map for the requested columns
    if output_columns == 'all':
        track_branch_map = dict(TRACK_BRANCH_MAP_ALL)
    else:
        track_branch_map = build_track_branch_map(output_columns)

    print(f'Reading ROOT file: {input_filepath}')
    print(f'  Extracting {len(track_branch_map)} track columns: '
          f'{sorted(track_branch_map.values())}')
    tree = load_root_tree(input_filepath)
    number_of_events = tree.num_entries
    print(f'  Total events in file: {number_of_events}')

    if entry_start is not None or entry_stop is not None:
        effective_start = entry_start or 0
        effective_stop = entry_stop or number_of_events
        print(f'  Processing entries [{effective_start}, {effective_stop})')
    else:
        effective_start = 0
        effective_stop = number_of_events

    # Read branches
    print('  Reading track branches and filtering to |pdgId| == 211 ...')
    raw_track_data = read_and_filter_pion_tracks(
        tree, track_branch_map, entry_start, entry_stop
    )

    print('  Reading event-level branches ...')
    event_info = read_event_info(tree, entry_start, entry_stop)

    # Compute pion counts per event (after filtering)
    # ak.num returns the number of elements in each inner list
    pion_counts_per_event = ak.to_numpy(
        ak.num(raw_track_data[list(track_branch_map.keys())[0]], axis=1)
    )

    # Rename track branches to output column names
    track_features = rename_branches(raw_track_data, track_branch_map)

    # Cast to consistent dtypes
    track_integer_columns = {'track_charge', 'track_n_valid_hits', 'track_n_valid_pixel_hits', 'track_label_from_tau'}
    track_features = cast_to_float32(track_features, integer_columns=track_integer_columns)

    # Build output table
    output = build_output_table(track_features, event_info, pion_counts_per_event)

    # Write to Parquet
    output_dir = os.path.dirname(output_filepath)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if os.path.exists(output_filepath):
        os.remove(output_filepath)

    print(f'  Writing Parquet: {output_filepath}')
    output_array = ak.Array(output)
    ak.to_parquet(output_array, output_filepath, compression='LZ4', compression_level=4)

    # Print summary statistics
    number_of_processed_events = effective_stop - effective_start

    print(f'\n=== Conversion Summary ===')
    print(f'  Events processed: {number_of_processed_events}')
    print(f'  Track columns: {sorted(track_features.keys())}')
    print(f'  Event-level columns: 4')

    print(f'\n  Pion tracks per event:')
    print(f'    min={pion_counts_per_event.min()}, '
          f'max={pion_counts_per_event.max()}, '
          f'mean={pion_counts_per_event.mean():.1f}')

    if 'track_label_from_tau' in track_features:
        # Per-track label distribution
        track_labels = track_features['track_label_from_tau']
        flat_labels = ak.to_numpy(ak.flatten(track_labels))
        total_tracks = len(flat_labels)
        tau_origin_tracks = int(np.sum(flat_labels == 1))
        background_tracks = total_tracks - tau_origin_tracks

        # Per-event tau-origin track counts
        tau_tracks_per_event = ak.to_numpy(ak.sum(track_labels, axis=1))
        tau_track_counts = Counter(tau_tracks_per_event.tolist())

        print(f'\n  Per-track label distribution (track_label_from_tau):')
        print(f'    tau-origin tracks: {tau_origin_tracks}')
        print(f'    background tracks: {background_tracks}')
        print(f'    tau-origin fraction: {tau_origin_tracks / max(total_tracks, 1):.4f}')

        print(f'\n  Tau-origin tracks per event:')
        for count in sorted(tau_track_counts):
            print(f'    {count}: {tau_track_counts[count]} events')

    print(f'\n  Output file: {output_filepath}')
    print(f'  Output size: {os.path.getsize(output_filepath) / 1024:.1f} KB')
    print('  Done.')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    available_columns = sorted(TRACK_BRANCH_MAP_ALL.values())

    parser = argparse.ArgumentParser(
        description='Convert low-pT tau ROOT dataset (NanoAOD) to Parquet format.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f'Available track columns:\n  ' + '\n  '.join(available_columns),
    )
    parser.add_argument(
        '-i', '--input',
        default='part/data/low-pt/merged_noBKstar.root',
        help='Path to the input ROOT file (default: %(default)s)',
    )
    parser.add_argument(
        '-o', '--output',
        default='part/data/low-pt/lowpt_tau_trackorigin.parquet',
        help='Path for the output Parquet file (default: %(default)s)',
    )
    parser.add_argument(
        '--entry-start',
        type=int,
        default=None,
        help='First entry index to process (optional, for chunked conversion)',
    )
    parser.add_argument(
        '--entry-stop',
        type=int,
        default=None,
        help='Last entry index to process, exclusive (optional, for chunked conversion)',
    )
    column_group = parser.add_mutually_exclusive_group()
    column_group.add_argument(
        '--columns',
        type=str,
        default=None,
        help=(
            'Comma-separated list of output column names to extract. '
            f'Default: {",".join(DEFAULT_COLUMNS)}'
        ),
    )
    column_group.add_argument(
        '--all-columns',
        action='store_true',
        help='Extract all available track columns (overrides --columns)',
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
        arguments.input, arguments.output,
        arguments.entry_start, arguments.entry_stop,
        output_columns=output_columns,
    )


if __name__ == '__main__':
    main()
