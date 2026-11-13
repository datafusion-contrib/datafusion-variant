#!/usr/bin/env bash
#
# Writes Bluesky files 1 to 100 (100M rows) to data/jsonbench/mixed, one Parquet
# file per Bluesky file, with the first fifth of the files (1 to 20) unshredded and the rest shredded as in shredded/load.sh.
# Run from the repository root:
#
#   ./benchmarks/jsonbench/mixed/load.sh
#
# Files already written are skipped. See ../load_layout.sh for the DFV, FILES,
# and JOBS environment variables.

exec "$(dirname "$0")/../load_layout.sh" mixed "$@"
