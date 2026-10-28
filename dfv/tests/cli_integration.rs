//! End-to-end tests that run the `dfv` binary and check its output.

use std::io::Write;

use assert_cmd::cargo::cargo_bin_cmd;
use flate2::Compression;
use flate2::write::GzEncoder;
use predicates::prelude::PredicateBooleanExt;

/// Rows of a query result as printed by the CLI (table borders omitted).
const EVENTS_JSON: &str = r#"
    SELECT parse_json(json_data) AS event
    FROM (VALUES
        ('{"service":"api","status":200,"tags":["cached"]}'),
        ('{"service":"web","status":500,"error":null}')
    ) AS input(json_data)
"#;

/// `-V`/`--version` prints the `dfv` version; every run starts with a banner
/// that names the DataFusion CLI version it is built on.
#[test]
fn test_version_and_banner() {
    let expected = format!("dfv {}\n", env!("CARGO_PKG_VERSION"));
    for flag in ["-V", "--version"] {
        cargo_bin_cmd!("dfv")
            .arg(flag)
            .assert()
            .success()
            .stdout(expected.clone());
    }

    cargo_bin_cmd!("dfv")
        .args(["-c", "SELECT 1"])
        .assert()
        .success()
        .stdout(predicates::str::starts_with(format!(
            "dfv (DataFusion Variant CLI) v{} based on DataFusion CLI v",
            env!("CARGO_PKG_VERSION")
        )));
}

/// The interactive shell (no `-c`/`-f`, SQL piped on stdin) prints example
/// commands after the banner; `-c` does not.
#[test]
fn test_examples_banner_in_repl_only() {
    cargo_bin_cmd!("dfv")
        .write_stdin("SELECT 1 AS one;\n\\q\n")
        .assert()
        .success()
        .stdout(predicates::str::contains("\nTry:\n"))
        .stdout(predicates::str::contains(
            "-- Build a table of Variant values and query it",
        ))
        .stdout(predicates::str::contains("| one |"));

    cargo_bin_cmd!("dfv")
        .args(["-c", "SELECT 1"])
        .assert()
        .success()
        .stdout(predicates::str::contains("\nTry:\n").not())
        .stdout(predicates::str::contains("-- Build a table of Variant values").not());
}

/// Only `-c` and `-f` are supported; other `datafusion-cli` options are
/// rejected by clap before any SQL runs.
#[test]
fn test_help_and_unsupported_options() {
    cargo_bin_cmd!("dfv")
        .arg("--help")
        .assert()
        .success()
        .stdout(predicates::str::contains("Usage: dfv [OPTIONS]"))
        .stdout(predicates::str::contains("-c, --command"))
        .stdout(predicates::str::contains("-f, --file"))
        .stdout(predicates::str::contains("--format").not())
        .stdout(predicates::str::contains("--memory-limit").not());

    for option in ["--format", "--quiet", "--memory-limit", "--spark"] {
        cargo_bin_cmd!("dfv")
            .args([option, "-c", "SELECT 1"])
            .assert()
            .code(2)
            .stderr(predicates::str::contains(format!(
                "unexpected argument '{option}' found"
            )));
    }
}

/// `parse_json` converts JSON text to Variant and `variant_get` extracts
/// typed fields.
#[test]
fn test_parse_json_and_variant_get() {
    cargo_bin_cmd!("dfv")
        .args([
            "-c",
            r#"SELECT variant_get(parse_json('{"service":"api","status":500}'), 'service', 'VARCHAR') AS service,
                      variant_get(parse_json('{"service":"api","status":500}'), 'status', 'BIGINT') AS status"#,
        ])
        .assert()
        .success()
        .stdout(predicates::str::contains(
            "\
| service | status |
+---------+--------+
| api     | 500    |
",
        ))
        .stdout(predicates::str::contains("1 row(s) fetched."));
}

/// The `:` field access syntax, array indexing, and `to_json` work, and
/// missing fields print as `NULL`.
#[test]
fn test_field_access_syntax_and_to_json() {
    cargo_bin_cmd!("dfv")
        .args([
            "-c",
            &format!(
                r#"WITH events AS ({EVENTS_JSON})
                   SELECT event:service::VARCHAR AS service,
                          event:tags[0]::VARCHAR AS first_tag,
                          to_json(event:status) AS status_json
                   FROM events
                   ORDER BY event:status::BIGINT"#
            ),
        ])
        .assert()
        .success()
        .stdout(predicates::str::contains(
            "\
| service | first_tag | status_json |
+---------+-----------+-------------+
| api     | cached    | 200         |
| web     | NULL      | 500         |
",
        ));
}

/// The SQL `VARIANT` type name is understood in casts.
#[test]
fn test_sql_variant_type() {
    cargo_bin_cmd!("dfv")
        .args([
            "-c",
            r#"SELECT to_json(CAST(variant_get(parse_json('{"x":7}'), 'x') AS VARIANT)) AS x,
                      arrow_typeof(CAST(NULL AS VARIANT)) = arrow_typeof(parse_json('1')) AS same_type"#,
        ])
        .assert()
        .success()
        .stdout(predicates::str::contains(
            "\
| x | same_type |
+---+-----------+
| 7 | true      |
",
        ));
}

/// `-f` writes a Variant column to Parquet; a second process reads it back
/// with several `-c` commands, keeping the Variant extension metadata during
/// schema inference.
#[test]
fn test_write_and_read_variant_parquet() {
    let temp_dir = tempfile::tempdir().unwrap();
    let script = temp_dir.path().join("write.sql");
    std::fs::write(
        &script,
        format!("COPY ({EVENTS_JSON}) TO 'events.parquet' STORED AS PARQUET;\n"),
    )
    .unwrap();

    cargo_bin_cmd!("dfv")
        .current_dir(temp_dir.path())
        .arg("-f")
        .arg(&script)
        .assert()
        .success()
        .stdout(predicates::str::contains(
            "\
| count |
+-------+
| 2     |
",
        ));
    assert!(temp_dir.path().join("events.parquet").is_file());

    cargo_bin_cmd!("dfv")
        .current_dir(temp_dir.path())
        .args([
            "-c",
            "SET datafusion.execution.parquet.skip_metadata = false",
            "-c",
            "CREATE EXTERNAL TABLE saved_events STORED AS PARQUET LOCATION 'events.parquet'",
            "-c",
            r#"SELECT event:service::VARCHAR AS service,
                      variant_get(event, 'status', 'BIGINT') AS status
               FROM saved_events
               ORDER BY status"#,
        ])
        .assert()
        .success()
        .stdout(predicates::str::contains(
            "\
| service | status |
+---------+--------+
| api     | 200    |
| web     | 500    |
",
        ));
}

/// SQL errors are reported on stdout with a failing exit code; a missing
/// `-f` file is rejected by clap.
#[test]
fn test_error_reporting() {
    cargo_bin_cmd!("dfv")
        .args(["-c", "SELECT parse_json('not json')"])
        .assert()
        .failure()
        .stdout(predicates::str::contains("Error: Arrow error"))
        .stdout(predicates::str::contains("JSON format error"));

    cargo_bin_cmd!("dfv")
        .args(["-f", "does-not-exist.sql"])
        .assert()
        .code(2)
        .stderr(predicates::str::contains(
            "Invalid file 'does-not-exist.sql'",
        ));
}

/// Write `file_000<n>.json.gz` with the given JSON lines into `dir`, in the
/// layout `load_bsky` expects.
fn write_bsky_file(dir: &std::path::Path, n: u32, lines: &[&str]) {
    let file = std::fs::File::create(dir.join(format!("file_{n:04}.json.gz"))).unwrap();
    let mut encoder = GzEncoder::new(file, Compression::fast());
    for line in lines {
        writeln!(encoder, "{line}").unwrap();
    }
    encoder.finish().unwrap();
}

/// `load_bsky(n)` reads `file_000n.json.gz` from `DFV_BSKY_DIR` without
/// downloading when the file exists, and exposes it as `json_data` rows.
#[test]
fn test_load_bsky_reads_local_file() {
    let temp_dir = tempfile::tempdir().unwrap();
    write_bsky_file(
        temp_dir.path(),
        2,
        &[
            r#"{"kind":"commit","time_us":3}"#,
            r#"{"kind":"identity","time_us":1}"#,
            r#"{"kind":"commit","time_us":2}"#,
        ],
    );

    cargo_bin_cmd!("dfv")
        .env("DFV_BSKY_DIR", temp_dir.path())
        .args([
            "-c",
            "SELECT count(*) AS rows FROM load_bsky(2)",
            "-c",
            r#"SELECT json_data:kind::VARCHAR AS kind, count(*) AS n
               FROM (SELECT parse_json(json_data) AS json_data FROM load_bsky(2))
               GROUP BY kind ORDER BY n DESC"#,
        ])
        .assert()
        .success()
        .stdout(predicates::str::contains(
            "\
| rows |
+------+
| 3    |
",
        ))
        .stdout(predicates::str::contains(
            "\
| kind     | n |
+----------+---+
| commit   | 2 |
| identity | 1 |
",
        ))
        .stderr(predicates::str::contains("Loaded 3 rows"))
        .stderr(predicates::str::contains("Downloading").not());
}

/// `load_bsky` requires a single file number between 1 and 1000.
#[test]
fn test_load_bsky_rejects_bad_arguments() {
    for args in ["0", "1001", "'x'", "1, 2"] {
        cargo_bin_cmd!("dfv")
            .args(["-c", &format!("SELECT * FROM load_bsky({args})")])
            .assert()
            .failure()
            .stdout(predicates::str::contains(
                "load_bsky expects a single file number between 1 and 1000",
            ));
    }
}

/// Downloads the real 135 MB `file_0001.json.gz`; run with `--ignored`.
#[test]
#[ignore = "downloads 135 MB from the network"]
fn test_load_bsky_downloads_missing_file() {
    let temp_dir = tempfile::tempdir().unwrap();
    cargo_bin_cmd!("dfv")
        .env("DFV_BSKY_DIR", temp_dir.path())
        .args(["-c", "SELECT count(*) AS rows FROM load_bsky(1)"])
        .assert()
        .success()
        .stdout(predicates::str::contains("| 1000000 |"))
        .stderr(predicates::str::contains("Downloading https://"));
    assert!(temp_dir.path().join("file_0001.json.gz").is_file());
}
