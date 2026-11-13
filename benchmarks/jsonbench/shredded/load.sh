#!/usr/bin/env bash
#
# Writes Bluesky files 1 to NUM_FILES (default 10, 1M rows each) to
# data/jsonbench/shredded, one Parquet file per Bluesky file,
# with every field the queries read shredded.
# Run from the repository root:
#
#   ./benchmarks/jsonbench/shredded/load.sh
#
# Files already written are skipped. See ../load_layout.sh for the DFV,
# NUM_FILES, and JOBS environment variables.

exec "$(dirname "$0")/../load_layout.sh" shredded "$@"
