//! End-to-end tests that run the `dfv` binary with `-c` commands and check
//! the printed output.

use std::path::Path;
use std::process::Command;

/// Run `dfv -q --format csv -c <cmd>...` in `dir` and return stdout.
///
/// Panics if the process fails or prints an error. The CLI reports SQL errors
/// on stdout with a zero exit code, so both are checked.
fn dfv(dir: &Path, commands: &[&str]) -> String {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_dfv"));
    cmd.current_dir(dir).args(["--quiet", "--format", "csv"]);
    for command in commands {
        cmd.args(["-c", command]);
    }
    let output = cmd.output().expect("failed to run dfv");
    let stdout = String::from_utf8(output.stdout).unwrap();
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(
        output.status.success() && !stdout.contains("Error:"),
        "dfv failed for {commands:?}\nstatus: {}\nstdout:\n{stdout}\nstderr:\n{stderr}",
        output.status
    );
    stdout
}

fn tempdir() -> tempfile::TempDir {
    tempfile::tempdir().expect("failed to create temp dir")
}

#[test]
fn prints_banner_with_version() {
    let output = Command::new(env!("CARGO_BIN_EXE_dfv"))
        .args(["-c", "SELECT 1"])
        .output()
        .expect("failed to run dfv");
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(output.status.success(), "stdout:\n{stdout}");
    assert!(
        stdout.starts_with("dfv (DataFusion Variant CLI) v"),
        "stdout:\n{stdout}"
    );
    assert!(stdout.contains("DataFusion CLI v"), "stdout:\n{stdout}");
}

#[test]
fn parse_json_and_extract_fields() {
    let dir = tempdir();
    let stdout = dfv(
        dir.path(),
        &[
            r#"SELECT variant_get(parse_json('{"service":"api","status":500}'), 'service', 'VARCHAR') AS service,
                    variant_get(parse_json('{"service":"api","status":500}'), 'status', 'BIGINT') AS status"#,
        ],
    );
    assert_eq!(stdout, "service,status\napi,500\n");
}

#[test]
fn field_access_syntax_and_to_json() {
    let dir = tempdir();
    let stdout = dfv(
        dir.path(),
        &[r#"WITH events AS (
                 SELECT parse_json(json_data) AS event
                 FROM (VALUES
                     ('{"service":"api","status":200,"tags":["cached"]}'),
                     ('{"service":"web","status":500,"error":null}')
                 ) AS input(json_data)
             )
             SELECT event:service::VARCHAR AS service,
                    event:tags[0]::VARCHAR AS first_tag,
                    to_json(event:status) AS status_json
             FROM events
             ORDER BY event:status::BIGINT"#],
    );
    assert_eq!(
        stdout,
        "service,first_tag,status_json\napi,cached,200\nweb,,500\n"
    );
}

#[test]
fn sql_variant_type() {
    let dir = tempdir();
    let stdout = dfv(
        dir.path(),
        &[
            r#"SELECT to_json(CAST(variant_get(parse_json('{"x":7}'), 'x') AS VARIANT)) AS x,
                    arrow_typeof(CAST(NULL AS VARIANT)) = arrow_typeof(parse_json('1')) AS same_type"#,
        ],
    );
    assert_eq!(stdout, "x,same_type\n7,true\n");
}

#[test]
fn write_and_read_variant_parquet() {
    let dir = tempdir();
    // Write a Variant column to Parquet in the temp dir ...
    let stdout = dfv(
        dir.path(),
        &[r#"COPY (
                 SELECT parse_json(json_data) AS event
                 FROM (VALUES
                     ('{"service":"api","status":200}'),
                     ('{"service":"web","status":500}')
                 ) AS input(json_data)
             ) TO 'events.parquet' STORED AS PARQUET"#],
    );
    assert_eq!(stdout, "count\n2\n");
    assert!(dir.path().join("events.parquet").is_file());

    // ... then read it back in a fresh process, keeping the Variant
    // extension metadata during schema inference.
    let stdout = dfv(
        dir.path(),
        &[
            "SET datafusion.execution.parquet.skip_metadata = false",
            "CREATE EXTERNAL TABLE saved_events STORED AS PARQUET LOCATION 'events.parquet'",
            r#"SELECT event:service::VARCHAR AS service,
                      variant_get(event, 'status', 'BIGINT') AS status
               FROM saved_events
               ORDER BY status"#,
        ],
    );
    assert_eq!(stdout, "service,status\napi,200\nweb,500\n");
}

#[test]
fn reports_sql_errors() {
    let dir = tempdir();
    let output = Command::new(env!("CARGO_BIN_EXE_dfv"))
        .current_dir(dir.path())
        .args(["-q", "-c", "SELECT parse_json('not json')"])
        .output()
        .expect("failed to run dfv");
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("Error"), "stdout:\n{stdout}");
}
