#!/usr/bin/env bash
#
# Writes Bluesky files 1 to 100 (100M rows) to data/jsonbench/unshredded, one Parquet
# file per Bluesky file, as unshredded Variant.
# Run from the repository root:
#
#   ./benchmarks/jsonbench/unshredded/load.sh
#
# Files already written are skipped. See ../load_layout.sh for the DFV, FILES,
# and JOBS environment variables.

exec "$(dirname "$0")/../load_layout.sh" unshredded "$@"
