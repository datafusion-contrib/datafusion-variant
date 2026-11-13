#!/usr/bin/env bash
#
# Writes the Bluesky files to data/jsonbench/<layout> as Parquet Variant, one
# Parquet file per Bluesky file. Run from the repository root:
#
#   ./benchmarks/jsonbench/load.sh mixed
#
# Files that are already written are skipped, so an interrupted load picks up
# where it left off.
#
# Usage: load.sh <unshredded|shredded|mixed> [file number]
#
# With a file number, converts only that file. Environment variables:
#
#   DFV            dfv binary to run (default: target/release/dfv)
#   NUM_FILES      convert Bluesky files 1 to NUM_FILES, (default: 10, max 100)
#   JOBS           files to download and convert at once (default: number of cores)
#   DFV_BSKY_DIR   where the Bluesky files are downloaded (default: data/bluesky)

set -euo pipefail

SHREDDING='{"data": "STRUCT<kind VARCHAR, did VARCHAR, time_us BIGINT, commit STRUCT<collection VARCHAR, operation VARCHAR>>"}'

usage() {
  sed -n '/^# Usage/,/^$/p' "$0" | sed 's/^# \{0,1\}//' >&2
  exit 2
}

LAYOUT=${1:-}
case "$LAYOUT" in
  unshredded | shredded | mixed) ;;
  *) usage ;;
esac

export DFV=${DFV:-target/release/dfv}
NUM_FILES=${NUM_FILES:-10}
JOBS=${JOBS:-$(getconf _NPROCESSORS_ONLN)}

OUT_DIR=data/jsonbench/$LAYOUT
# Files are written here and moved to OUT_DIR when complete, so OUT_DIR never
# holds a partly written file.
TMP_DIR=data/jsonbench/.tmp/$LAYOUT

# Succeeds if Bluesky file $1 is shredded in this layout. `mixed` shreds every
# fifth file: files 5, 10, 15, and so on.
is_shredded() {
  case "$LAYOUT" in
    unshredded) return 1 ;;
    shredded) return 0 ;;
    mixed) [ $(($1 % 5)) -eq 0 ] ;;
  esac
}

# Convert Bluesky file $1, downloading it first if needed.
load_file() {
  local name out tmp log options=""
  name=$(printf 'file_%04d' "$1")
  out=$OUT_DIR/$name.parquet
  tmp=$TMP_DIR/$name.parquet
  log=$TMP_DIR/$name.log
  if [ -f "$out" ]; then
    return 0
  fi
  if is_shredded "$1"; then
    options="OPTIONS ('variant.shredding' '$SHREDDING')"
  fi
  rm -f "$tmp"
  # One partition per dfv: the files are converted in parallel instead, and
  # the rows keep their order. dfv reports a failed statement but still exits
  # successfully, so also look for an error in its output.
  if ! "$DFV" \
    -c "SET datafusion.execution.target_partitions = 1" \
    -c "SET datafusion.execution.parquet.compression = 'snappy'" \
    -c "SET datafusion.execution.parquet.max_row_group_size = 65536" \
    -c "COPY (SELECT try_parse_json(json_data) AS data FROM load_bsky($1)) TO '$tmp' STORED AS PARQUET $options" \
    >"$log" 2>&1 || grep -q '^Error' "$log" || [ ! -f "$tmp" ]; then
    echo "FAILED $out:" >&2
    cat "$log" >&2
    return 1
  fi
  mv "$tmp" "$out"
  rm -f "$log"
  echo "Wrote $out"
}

case "$NUM_FILES" in
  '' | *[!0-9]*) NUM_FILES=0 ;;
esac
if [ "$NUM_FILES" -lt 1 ] || [ "$NUM_FILES" -gt 100 ]; then
  echo "NUM_FILES must be between 1 and 100" >&2
  exit 2
fi
if ! command -v "$DFV" >/dev/null; then
  echo "dfv not found at '$DFV'. Build it with 'cargo build --release --bin dfv' or set DFV." >&2
  exit 1
fi
mkdir -p "$OUT_DIR" "$TMP_DIR"

if [ $# -ge 2 ]; then
  load_file "$2"
  exit
fi

echo "Loading Bluesky files 1 to $NUM_FILES into $OUT_DIR, $JOBS at a time"
status=0
seq 1 "$NUM_FILES" | xargs -P "$JOBS" -n 1 "$0" "$LAYOUT" || status=$?
rm -rf "$TMP_DIR"
rmdir data/jsonbench/.tmp 2>/dev/null || true
if [ "$status" -ne 0 ]; then
  echo "Some files failed. Run this script again to retry them." >&2
  exit 1
fi
echo "Done: $(find "$OUT_DIR" -name '*.parquet' | wc -l | tr -d ' ') files in $OUT_DIR"
