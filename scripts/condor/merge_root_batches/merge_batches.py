import os
import sys
import glob
import ROOT

batch_id = int(sys.argv[1])

opts = ROOT.RDF.RSnapshotOptions()
opts.fCompressionAlgorithm = 1
opts.fCompressionLevel = 1

BASE_DIR = "/eos/user/o/oprostak/tau_data"
output = os.path.join(BASE_DIR, "root1", f"merged_noBKstar_batch{batch_id}.root")

if os.path.exists(output):
    print(f"batch{batch_id}: already exists, skipping -> {output}")
    sys.exit(0)

files = glob.glob(f"/eos/cms/store/group/phys_bphys/valukash/mc_signal/batch{batch_id}_2024/*.root")
files = [f for f in files if "merged_" not in f]

if not files:
    print(f"batch{batch_id}: no files found, skipping")
    sys.exit(0)

print(f"batch{batch_id}: merging {len(files)} files...")
df = ROOT.RDataFrame("Events", files)
cols = [str(c) for c in df.GetColumnNames()]
keep_cols = [c for c in cols
             if not (c.startswith("BToKstarTauTau_") or
                     c.startswith("nBToKstarTauTau"))]

# Built-in RDataFrame progress bar (prints % during event loop)
ROOT.RDF.Experimental.AddProgressBar(df)
df.Snapshot("Events", output, keep_cols, opts)
print(f"batch{batch_id}: done -> {output}")
