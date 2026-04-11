"""Merge per-job ROOT files into a single file per batch.

Keeps only the branches needed by convert_root_to_parquet.py:
  - CMS event identifiers: run, event, luminosityBlock
  - Primary vertex: PV_x, PV_y, PV_z
  - Track features: all branches from TRACK_BRANCH_MAP_ALL
  - Track_pdgId: used for pion filtering (|pdgId| == 211)

This reduces output size ~8x compared to keeping all 272 NanoAOD branches.

Usage (via HTCondor):
    python merge_batches.py <batch_id>
"""

import os
import sys
import glob
import ROOT

# All branches consumed by the parquet conversion pipeline.
# Intersection with available columns is taken at runtime (some branches
# like Track_dzTrg may not be present in all production campaigns).
KEEP_BRANCHES = {
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
}

batch_id = int(sys.argv[1])

opts = ROOT.RDF.RSnapshotOptions()
opts.fCompressionAlgorithm = 1
opts.fCompressionLevel = 1

BASE_DIR = "/eos/user/o/oprostak/tau_data"
output = os.path.join(BASE_DIR, "root1", f"merged_noBKstar_batch{batch_id}.root")

if os.path.exists(output):
    print(f"batch{batch_id}: already exists, skipping -> {output}")
    sys.exit(0)

files = glob.glob(
    f"/eos/cms/store/group/phys_bphys/valukash/mc_signal/"
    f"batch{batch_id}_2024/*.root"
)
files = [f for f in files if "merged_" not in f]

if not files:
    print(f"batch{batch_id}: no files found, skipping")
    sys.exit(0)

print(f"batch{batch_id}: merging {len(files)} files...")
df = ROOT.RDataFrame("Events", files)

# Keep only branches that are both needed and present in this file.
available_columns = {str(c) for c in df.GetColumnNames()}
keep_cols = sorted(available_columns & KEEP_BRANCHES)
missing = KEEP_BRANCHES - available_columns
if missing:
    print(f"  note: {len(missing)} requested branches not in source: {sorted(missing)}")
print(f"  keeping {len(keep_cols)}/{len(available_columns)} branches")

ROOT.RDF.Experimental.AddProgressBar(df)
df.Snapshot("Events", output, keep_cols, opts)
print(f"batch{batch_id}: done -> {output}")
