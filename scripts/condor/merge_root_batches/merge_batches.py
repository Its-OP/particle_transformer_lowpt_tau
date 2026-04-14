"""Merge per-job ROOT files into a single file per batch.

Keeps only the branches needed by convert_root_to_parquet.py:
  - CMS event identifiers: run, event, luminosityBlock
  - Source file identifiers: source_batch_id, source_microbatch_id
  - Primary vertex: PV_x, PV_y, PV_z
  - Track features: all branches from TRACK_BRANCH_MAP_ALL
  - Track_pdgId: used for pion filtering (|pdgId| == 211)

The source_batch_id and source_microbatch_id columns are derived from
the input file path. Together with ``(run, event, luminosityBlock)``
they form a composite key that uniquely identifies each tau candidate
across the full dataset.

source_batch_id: from the batch folder name (``batch_{N1}``)
source_microbatch_id: from the filename (``step_MINI_{N2}_...``)

This reduces output size ~8x compared to keeping all 272 NanoAOD branches.

Usage (via HTCondor):
    python merge_batches.py <batch_id>
"""

import os
import re
import sys
import glob

import numpy as np
import uproot
import awkward as ak

# All branches consumed by the parquet conversion pipeline.
KEEP_BRANCHES = [
    # CMS event identifiers
    'run', 'event', 'luminosityBlock',
    # Primary vertex
    'PV_x', 'PV_y', 'PV_z',
    # Track kinematics
    'Track_pt', 'Track_eta', 'Track_phi', 'Track_mass', 'Track_charge',
    # Track impact parameters
    'Track_dxy', 'Track_dxyS', 'Track_dz', 'Track_dzS', 'Track_dzTrg',
    # Track quality
    'Track_normChi2', 'Track_nValidHits', 'Track_nValidPixelHits',
    'Track_ptErr', 'Track_DCASig',
    # Track covariance matrix
    'Track_covQopQop', 'Track_covQopLam', 'Track_covQopPhi',
    'Track_covLamLam', 'Track_covLamPhi', 'Track_covPhiPhi',
    # Track vertex position
    'Track_vx', 'Track_vy', 'Track_vz',
    # Track labels and ID
    'Track_trackFromTau', 'Track_pdgId',
]

# Regex to extract microbatch id from filename:
#   step_MINI_{N2}_nano_ditaus_mc.root → N2
MICROBATCH_RE = re.compile(r'step_MINI_(\d+)_nano_ditaus_mc\.root$')


def parse_microbatch_id(filepath):
    """Extract microbatch id from a source file path.

    Args:
        filepath: Full path to a microbatch ROOT file.

    Returns:
        Integer microbatch id, or -1 if the filename doesn't match.
    """
    match = MICROBATCH_RE.search(filepath)
    return int(match.group(1)) if match else -1


batch_id = int(sys.argv[1])

BASE_DIR = "/eos/user/o/oprostak/tau_data"
output = os.path.join(BASE_DIR, "root1", f"merged_noBKstar_batch{batch_id}.root")

if os.path.exists(output):
    print(f"batch{batch_id}: already exists, skipping -> {output}")
    sys.exit(0)

files = sorted(glob.glob(
    f"/eos/cms/store/group/phys_bphys/valukash/mc_signal/"
    f"batch{batch_id}_2024/*.root"
))
files = [f for f in files if "merged_" not in f]

if not files:
    print(f"batch{batch_id}: no files found, skipping")
    sys.exit(0)

print(f"batch{batch_id}: merging {len(files)} files...")

os.makedirs(os.path.dirname(output), exist_ok=True)

with uproot.recreate(output) as out_file:
    writer = None

    for filepath in files:
        microbatch_id = parse_microbatch_id(filepath)
        tree = uproot.open(filepath)["Events"]

        # Keep only branches present in this file
        available = set(tree.keys())
        branches_to_read = [b for b in KEEP_BRANCHES if b in available]

        data = tree.arrays(branches_to_read, library="ak")
        n_entries = len(data)

        # Add source identifiers as scalar columns
        data["source_batch_id"] = np.full(n_entries, batch_id, dtype=np.int32)
        data["source_microbatch_id"] = np.full(n_entries, microbatch_id, dtype=np.int32)

        if writer is None:
            writer = out_file.mktree("Events", {
                field: data[field].type for field in data.fields
            })

        writer.extend({field: data[field] for field in data.fields})
        print(f"  {os.path.basename(filepath)}: {n_entries} entries "
              f"(microbatch_id={microbatch_id})")

print(f"batch{batch_id}: done -> {output}")
