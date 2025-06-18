import tiledbsoma.io
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--h5ad", help="H5AD input file")
parser.add_argument("--out", help="Output soma folder")
args = parser.parse_args()

tiledbsoma.io.from_h5ad(args.out, args.h5ad, measurement_name = "RNA", ingest_mode = "write")


