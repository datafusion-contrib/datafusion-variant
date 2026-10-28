//! End-to-end tests that run the `dfv` binary and check its output.

use assert_cmd::cargo::cargo_bin_cmd;
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

/// The final output adapter applies to commands, files, and the REPL, while
/// CREATE TABLE / INSERT and subsequent expressions retain Variant values.
#[test]
fn test_variant_display() {
    let sql = r#"
CREATE TABLE displayed AS
SELECT parse_json(j) AS v FROM (VALUES ('{"a":1}'), ('null'), (NULL)) AS t(j);
INSERT INTO displayed SELECT parse_json('{"a":2}');
SET datafusion.format.null = '[NULL]';
SELECT v, v IS NULL AS sql_null FROM displayed;
SELECT variant_get(v, 'a', 'BIGINT') AS a FROM displayed;
"#;
    let temp_dir = tempfile::tempdir().unwrap();
    let script = temp_dir.path().join("display.sql");
    std::fs::write(&script, sql).unwrap();

    for mode in ["command", "file", "repl"] {
        let mut cmd = cargo_bin_cmd!("dfv");
        match mode {
            "command" => {
                cmd.args(["-c", sql]);
            }
            "file" => {
                cmd.arg("-f").arg(&script);
            }
            _ => {
                cmd.write_stdin(format!("{sql}\\q\n"));
            }
        }
        cmd.assert()
            .success()
            .stdout(predicates::str::contains("| {\"a\":1} | false    |"))
            .stdout(predicates::str::contains("| null    | false    |"))
            .stdout(predicates::str::contains("| [NULL]  | true     |"))
            .stdout(predicates::str::contains("| 1      |"));
    }
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
            "-c",
            "SELECT event FROM saved_events ORDER BY event:status::BIGINT",
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
        ))
        .stdout(predicates::str::contains(
            r#"{"service":"api","status":200,"tags":["cached"]}"#,
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
