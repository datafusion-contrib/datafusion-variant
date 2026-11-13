#!/usr/bin/env bash
#
# Writes Bluesky files 1 to NUM_FILES (default 10, 1M rows each) to
# data/jsonbench/unshredded, one Parquet file per Bluesky file,
# as unshredded Variant.
# Run from the repository root:
#
#   ./benchmarks/jsonbench/unshredded/load.sh
#
# Files already written are skipped. See ../load_layout.sh for the DFV,
# NUM_FILES, and JOBS environment variables.

exec "$(dirname "$0")/../load_layout.sh" unshredded "$@"
