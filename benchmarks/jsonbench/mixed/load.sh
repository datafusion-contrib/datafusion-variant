#!/usr/bin/env bash
#
# Writes Bluesky files 1 to NUM_FILES (default 10, 1M rows each) to
# data/jsonbench/mixed, one Parquet file per Bluesky file,
# with every fifth file (5, 10, 15, ...) shredded as in shredded/load.sh
# and the rest unshredded.
# Run from the repository root:
#
#   ./benchmarks/jsonbench/mixed/load.sh
#
# Files already written are skipped. See ../load_layout.sh for the DFV,
# NUM_FILES, and JOBS environment variables.

exec "$(dirname "$0")/../load_layout.sh" mixed "$@"
