// SPDX-License-Identifier: Apache-2.0

//! End-to-end tests that run the `dfv` binary and check its output.

use std::io::{BufRead, BufReader, Write};
use std::process::Stdio;
use std::sync::mpsc;
use std::time::Duration;

use assert_cmd::cargo::cargo_bin_cmd;
use datafusion_cli::DATAFUSION_CLI_VERSION;
use flate2::Compression;
use flate2::write::GzEncoder;

/// A query returning two rows of a Variant `event` column.
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

    let assert = cargo_bin_cmd!("dfv")
        .args(["-c", "SELECT 1"])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&assert.get_output().stdout);
    let banner = format!(
        "dfv (DataFusion Variant CLI) v{} based on DataFusion CLI v{DATAFUSION_CLI_VERSION}",
        env!("CARGO_PKG_VERSION")
    );
    assert_eq!(stdout.lines().next(), Some(banner.as_str()), "{stdout}");
}

/// Insta settings that normalize the parts of `dfv` output that vary between
/// runs or releases: query timings, the temporary directory holding Bluesky
/// files, and the `dfv` and DataFusion versions.
fn dfv_snapshot_settings() -> insta::Settings {
    let mut settings = insta::Settings::clone_current();
    settings.add_filter(r"Elapsed [0-9.]+ seconds\.", "Elapsed [ELAPSED] seconds.");
    settings.add_filter(r"Loading .*/(file_\d+\.json\.gz)", "Loading [BSKY_DIR]/$1");
    // test_version_and_banner checks the versions
    settings.add_filter(
        r"v\S+ based on DataFusion CLI v\S+",
        "v[VERSION] based on DataFusion CLI v[DATAFUSION_VERSION]",
    );
    // The REPL prints the examples banner, which has its own tests
    settings.add_filter(r"(?s)\nTry:\n.*?\nEnter \\q to exit\.\n", "\n[EXAMPLES]\n");
    // insta compares lines exactly, and editors often strip trailing spaces
    settings.add_filter(r"(?m) +$", "");
    settings
}

/// Run the command and return its exit code, stdout, and stderr formatted for
/// snapshotting.
fn snapshot_output(cmd: &mut assert_cmd::Command) -> String {
    let output = cmd.output().unwrap();
    let stdout = String::from_utf8(output.stdout).unwrap();
    let stderr = String::from_utf8(output.stderr).unwrap();
    format!(
        "exit code: {}\nstdout:\n{stdout}\nstderr:\n{stderr}",
        output.status.code().unwrap()
    )
}

/// Extract the SQL examples in the CLI example output
///
/// ```text
/// dfv (DataFusion Variant CLI) v[VERSION] based on DataFusion CLI v[DATAFUSION_VERSION]
///
/// Try:
///   -- Build a table of Variant values by parsing JSON    <--- Captured
/// ...                                                     <--- Captured
/// See https://github.com/datafusion-contrib/datafusion-variant#functions for all functions.
/// ```
///
/// Panics if the output has no examples banner.
fn banner_examples(stdout: &str) -> String {
    let mut lines = stdout.lines().skip_while(|line| *line != "Try:");
    assert!(lines.next().is_some(), "no examples banner in:\n{stdout}");
    lines
        .take_while(|line| !line.starts_with("See https://"))
        .collect::<Vec<_>>()
        .join("\n")
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

    let mut context = String::new();
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
        context += &format!("=== {mode} ===\n{}\n", snapshot_output(&mut cmd));
    }

    dfv_snapshot_settings().bind(|| {
        insta::assert_snapshot!(context, @r#"
        === command ===
        exit code: 0
        stdout:
        dfv (DataFusion Variant CLI) v[VERSION] based on DataFusion CLI v[DATAFUSION_VERSION]
        0 row(s) fetched.
        Elapsed [ELAPSED] seconds.

        +-------+
        | count |
        +-------+
        | 1     |
        +-------+
        1 row(s) fetched.
        Elapsed [ELAPSED] seconds.

        0 row(s) fetched.
        Elapsed [ELAPSED] seconds.

        +---------+----------+
        | v       | sql_null |
        +---------+----------+
        | {"a":1} | false    |
        | null    | false    |
        | [NULL]  | true     |
        | {"a":2} | false    |
        +---------+----------+
        4 row(s) fetched.
        Elapsed [ELAPSED] seconds.

        +--------+
        | a      |
        +--------+
        | 1      |
        | [NULL] |
        | [NULL] |
        | 2      |
        +--------+
        4 row(s) fetched.
        Elapsed [ELAPSED] seconds.


        stderr:

        === file ===
        exit code: 0
        stdout:
        dfv (DataFusion Variant CLI) v[VERSION] based on DataFusion CLI v[DATAFUSION_VERSION]
        0 row(s) fetched.
        Elapsed [ELAPSED] seconds.

        +-------+
        | count |
        +-------+
        | 1     |
        +-------+
        1 row(s) fetched.
        Elapsed [ELAPSED] seconds.

        0 row(s) fetched.
        Elapsed [ELAPSED] seconds.

        +---------+----------+
        | v       | sql_null |
        +---------+----------+
        | {"a":1} | false    |
        | null    | false    |
        | [NULL]  | true     |
        | {"a":2} | false    |
        +---------+----------+
        4 row(s) fetched.
        Elapsed [ELAPSED] seconds.

        +--------+
        | a      |
        +--------+
        | 1      |
        | [NULL] |
        | [NULL] |
        | 2      |
        +--------+
        4 row(s) fetched.
        Elapsed [ELAPSED] seconds.


        stderr:

        === repl ===
        exit code: 0
        stdout:
        dfv (DataFusion Variant CLI) v[VERSION] based on DataFusion CLI v[DATAFUSION_VERSION]

        [EXAMPLES]

        0 row(s) fetched.
        Elapsed [ELAPSED] seconds.

        +-------+
        | count |
        +-------+
        | 1     |
        +-------+
        1 row(s) fetched.
        Elapsed [ELAPSED] seconds.

        0 row(s) fetched.
        Elapsed [ELAPSED] seconds.

        +---------+----------+
        | v       | sql_null |
        +---------+----------+
        | {"a":1} | false    |
        | null    | false    |
        | [NULL]  | true     |
        | {"a":2} | false    |
        +---------+----------+
        4 row(s) fetched.
        Elapsed [ELAPSED] seconds.

        +--------+
        | a      |
        +--------+
        | 1      |
        | [NULL] |
        | [NULL] |
        | 2      |
        +--------+
        4 row(s) fetched.
        Elapsed [ELAPSED] seconds.


        stderr:
        "#);
    });
}

/// Only `-c` and `-f` are supported; other `datafusion-cli` options are
/// rejected by clap before any SQL runs.
#[test]
fn test_help_and_unsupported_options() {
    let mut context = format!(
        "=== --help ===\n{}\n",
        snapshot_output(cargo_bin_cmd!("dfv").arg("--help"))
    );
    for option in ["--format", "--quiet", "--memory-limit", "--spark"] {
        context += &format!(
            "=== {option} -c 'SELECT 1' ===\n{}\n",
            snapshot_output(cargo_bin_cmd!("dfv").args([option, "-c", "SELECT 1"]))
        );
    }

    dfv_snapshot_settings().bind(|| {
        insta::assert_snapshot!(context, @r"
        === --help ===
        exit code: 0
        stdout:
        datafusion-cli with the DataFusion Variant functions preinstalled

        Usage: dfv [OPTIONS]

        Options:
          -c, --command [<COMMAND>...]  Execute the given command string(s), then exit. Commands are expected to be non empty.
          -f, --file [<FILE>...]        Execute commands from file(s), then exit
          -h, --help                    Print help
          -V, --version                 Print version

        stderr:

        === --format -c 'SELECT 1' ===
        exit code: 2
        stdout:

        stderr:
        error: unexpected argument '--format' found

        Usage: dfv [OPTIONS]

        For more information, try '--help'.

        === --quiet -c 'SELECT 1' ===
        exit code: 2
        stdout:

        stderr:
        error: unexpected argument '--quiet' found

        Usage: dfv [OPTIONS]

        For more information, try '--help'.

        === --memory-limit -c 'SELECT 1' ===
        exit code: 2
        stdout:

        stderr:
        error: unexpected argument '--memory-limit' found

        Usage: dfv [OPTIONS]

        For more information, try '--help'.

        === --spark -c 'SELECT 1' ===
        exit code: 2
        stdout:

        stderr:
        error: unexpected argument '--spark' found

        Usage: dfv [OPTIONS]

        For more information, try '--help'.
        ");
    });
}

/// The SQL `VARIANT` type name is understood in casts.
#[test]
fn test_sql_variant_type() {
    let context = snapshot_output(cargo_bin_cmd!("dfv").args([
        "-c",
        r#"SELECT to_json(CAST(variant_get(parse_json('{"x":7}'), 'x') AS VARIANT)) AS x,
                  arrow_typeof(CAST(NULL AS VARIANT)) = arrow_typeof(parse_json('1')) AS same_type"#,
    ]));

    dfv_snapshot_settings().bind(|| {
        insta::assert_snapshot!(context, @"
        exit code: 0
        stdout:
        dfv (DataFusion Variant CLI) v[VERSION] based on DataFusion CLI v[DATAFUSION_VERSION]
        +---+-----------+
        | x | same_type |
        +---+-----------+
        | 7 | true      |
        +---+-----------+
        1 row(s) fetched.
        Elapsed [ELAPSED] seconds.


        stderr:
        ");
    });
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

    let write = snapshot_output(
        cargo_bin_cmd!("dfv")
            .current_dir(temp_dir.path())
            .arg("-f")
            .arg(&script),
    );
    let read = snapshot_output(cargo_bin_cmd!("dfv").current_dir(temp_dir.path()).args([
        "-c",
        "CREATE EXTERNAL TABLE saved_events STORED AS PARQUET LOCATION 'events.parquet'",
        "-c",
        r#"SELECT event:service::VARCHAR AS service,
                  variant_get(event, 'status', 'BIGINT') AS status
           FROM saved_events
           ORDER BY status"#,
        "-c",
        "SELECT event FROM saved_events ORDER BY event:status::BIGINT",
    ]));
    let context = format!("=== write ===\n{write}\n=== read ===\n{read}");

    dfv_snapshot_settings().bind(|| {
        insta::assert_snapshot!(context, @r#"
        === write ===
        exit code: 0
        stdout:
        dfv (DataFusion Variant CLI) v[VERSION] based on DataFusion CLI v[DATAFUSION_VERSION]
        +-------+
        | count |
        +-------+
        | 2     |
        +-------+
        1 row(s) fetched.
        Elapsed [ELAPSED] seconds.


        stderr:

        === read ===
        exit code: 0
        stdout:
        dfv (DataFusion Variant CLI) v[VERSION] based on DataFusion CLI v[DATAFUSION_VERSION]
        0 row(s) fetched.
        Elapsed [ELAPSED] seconds.

        +---------+--------+
        | service | status |
        +---------+--------+
        | api     | 200    |
        | web     | 500    |
        +---------+--------+
        2 row(s) fetched.
        Elapsed [ELAPSED] seconds.

        +--------------------------------------------------+
        | event                                            |
        +--------------------------------------------------+
        | {"service":"api","status":200,"tags":["cached"]} |
        | {"error":null,"service":"web","status":500}      |
        +--------------------------------------------------+
        2 row(s) fetched.
        Elapsed [ELAPSED] seconds.


        stderr:
        "#);
    });
}

/// SQL errors are reported on stdout with a failing exit code; a missing
/// `-f` file is rejected by clap.
#[test]
fn test_error_reporting() {
    let context = format!(
        "=== invalid JSON ===\n{}\n=== missing file ===\n{}",
        snapshot_output(cargo_bin_cmd!("dfv").args(["-c", "SELECT parse_json('not json')"])),
        snapshot_output(cargo_bin_cmd!("dfv").args(["-f", "does-not-exist.sql"])),
    );

    dfv_snapshot_settings().bind(|| {
        insta::assert_snapshot!(context, @"
        === invalid JSON ===
        exit code: 1
        stdout:
        dfv (DataFusion Variant CLI) v[VERSION] based on DataFusion CLI v[DATAFUSION_VERSION]
        Error: Arrow error: Invalid argument error: JSON format error: expected ident at line 1 column 2

        stderr:

        === missing file ===
        exit code: 2
        stdout:

        stderr:
        error: invalid value 'does-not-exist.sql' for '--file [<FILE>...]': Invalid file 'does-not-exist.sql'

        For more information, try '--help'.
        ");
    });
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

/// Real rows from the Bluesky dataset (`file_0001.json.gz`): the first two
/// `commit` events and the first `identity` event (line 99).
const BSKY_ROWS: [&str; 3] = [
    r#"{"did":"did:plc:yj3sjq3blzpynh27cumnp5ks","time_us":1732206349000167,"kind":"commit","commit":{"rev":"3lbhtytnn2k2f","operation":"create","collection":"app.bsky.feed.post","rkey":"3lbhtyteurk2y","record":{"$type":"app.bsky.feed.post","createdAt":"2024-11-21T16:09:27.095Z","langs":["en"],"reply":{"parent":{"cid":"bafyreibfglofvqou2yiqvwzk4rcgkhhxrbunyemshdjledgwymimqkg24e","uri":"at://did:plc:6tr6tuzlx2db3rduzr2d6r24/app.bsky.feed.post/3lbhqo2rtys2z"},"root":{"cid":"bafyreibfglofvqou2yiqvwzk4rcgkhhxrbunyemshdjledgwymimqkg24e","uri":"at://did:plc:6tr6tuzlx2db3rduzr2d6r24/app.bsky.feed.post/3lbhqo2rtys2z"}},"text":"aaaaah.  LIght shines in a corner of WTF...."},"cid":"bafyreidblutgvj75o4q4akzyyejedjj6l3it6hgqwee6jpwv2wqph5fsgm"}}"#,
    r#"{"did":"did:plc:3i4xf2v4wcnyktgv6satke64","time_us":1732206349000644,"kind":"commit","commit":{"rev":"3lbhuvzds6d2a","operation":"create","collection":"app.bsky.feed.like","rkey":"3lbhuvzdked2a","record":{"$type":"app.bsky.feed.like","createdAt":"2024-11-21T16:25:46.221Z","subject":{"cid":"bafyreidjvrcmckkm765mct5fph36x7kupkfo35rjklbf2k76xkzwyiauge","uri":"at://did:plc:azrv4rcbws6kmcga4fsbphg2/app.bsky.feed.post/3lbgjdpbiec2l"}},"cid":"bafyreia5l5vrkh5oj4cjyhcqby2dprhyvcyofo2q5562tijlae2pzih23m"}}"#,
    r#"{"did":"did:plc:2dtvzdnwwhikc4i62eh47uir","time_us":1732206349164509,"kind":"identity","identity":{"did":"did:plc:2dtvzdnwwhikc4i62eh47uir","handle":"cjackson292.bsky.social","seq":4148295340,"time":"2024-11-21T16:25:48.604Z"}}"#,
];

/// Every example printed in the `Try:` banner runs successfully, in order,
/// through `dfv -f`. The script and the resulting exit code, stdout, and
/// stderr are compared against an inline snapshot (update with `cargo insta
/// test --accept -p dfv`). The Bluesky example reads a local file holding
/// [`BSKY_ROWS`] via `DFV_BSKY_DIR` instead of downloading.
#[test]
fn test_banner_examples_run_end_to_end() {
    let temp_dir = tempfile::tempdir().unwrap();
    write_bsky_file(temp_dir.path(), 1, &BSKY_ROWS);

    // Capture the examples from the interactive shell's banner. The REPL
    // writes `.history` to its working directory.
    let banner = cargo_bin_cmd!("dfv")
        .current_dir(temp_dir.path())
        .write_stdin("\\q\n")
        .assert()
        .success();
    let sql = banner_examples(&String::from_utf8_lossy(&banner.get_output().stdout));
    let script = temp_dir.path().join("examples.sql");
    std::fs::write(&script, &sql).unwrap();

    let output = snapshot_output(
        cargo_bin_cmd!("dfv")
            .current_dir(temp_dir.path())
            .env("DFV_BSKY_DIR", temp_dir.path())
            .arg("-f")
            .arg(&script),
    );
    let context = format!("script:\n{sql}\n{output}");

    dfv_snapshot_settings().bind(|| {
        insta::assert_snapshot!(context, @r#"
        script:
          -- Build a table of Variant values by parsing JSON
          CREATE TABLE events AS
          SELECT parse_json(json_data) AS event
          FROM (VALUES
              ('{"service":"api","status":200,"latency_ms":12,"tags":["cached"]}'),
              ('{"service":"api","status":500,"latency_ms":85,"error":null}'),
              ('{"service":"api","status":504,"error":"Somethings not working"}')
          ) AS input(json_data);

          -- Select fields from the Variant values using the `:` operator and filter by a field value
          SELECT event:service AS service
          FROM events
          WHERE event:status::BIGINT >= 400;

          -- Write the values to the events.parquet file
          COPY events TO 'events.parquet' STORED AS PARQUET;
          SELECT event FROM 'events.parquet';

          -- Write to shredded_events.parquet, shredding the `service` and `status` fields into typed columns
          COPY events TO 'shredded_events.parquet' STORED AS PARQUET
          OPTIONS ('variant.shredding' '{"event": "STRUCT<service VARCHAR, status BIGINT>"}');
          SELECT path_in_schema, type FROM parquet_metadata('shredded_events.parquet');
          SELECT * FROM 'shredded_events.parquet';

          -- Show the schema (fields and types in each variant value)
          SELECT schema_of_variant(event) FROM events;

          -- Load 1M real Bluesky events (downloads ~135MB to data/bluesky on first use)
          CREATE TABLE bsky AS SELECT parse_json(json_data) AS event FROM load_bsky(1);
          SELECT event:kind::VARCHAR AS kind, count(*) FROM bsky GROUP BY kind ORDER BY 2 DESC;

        exit code: 0
        stdout:
        dfv (DataFusion Variant CLI) v[VERSION] based on DataFusion CLI v[DATAFUSION_VERSION]
        0 row(s) fetched.
        Elapsed [ELAPSED] seconds.

        +---------+
        | service |
        +---------+
        | "api"   |
        | "api"   |
        +---------+
        2 row(s) fetched.
        Elapsed [ELAPSED] seconds.

        +-------+
        | count |
        +-------+
        | 3     |
        +-------+
        1 row(s) fetched.
        Elapsed [ELAPSED] seconds.

        +------------------------------------------------------------------+
        | event                                                            |
        +------------------------------------------------------------------+
        | {"latency_ms":12,"service":"api","status":200,"tags":["cached"]} |
        | {"error":null,"latency_ms":85,"service":"api","status":500}      |
        | {"error":"Somethings not working","service":"api","status":504}  |
        +------------------------------------------------------------------+
        3 row(s) fetched.
        Elapsed [ELAPSED] seconds.

        +-------+
        | count |
        +-------+
        | 3     |
        +-------+
        1 row(s) fetched.
        Elapsed [ELAPSED] seconds.

        +-----------------------------------------+------------+
        | path_in_schema                          | type       |
        +-----------------------------------------+------------+
        | "event.metadata"                        | BYTE_ARRAY |
        | "event.value"                           | BYTE_ARRAY |
        | "event.typed_value.service.value"       | BYTE_ARRAY |
        | "event.typed_value.service.typed_value" | BYTE_ARRAY |
        | "event.typed_value.status.value"        | BYTE_ARRAY |
        | "event.typed_value.status.typed_value"  | INT64      |
        +-----------------------------------------+------------+
        6 row(s) fetched.
        Elapsed [ELAPSED] seconds.

        +------------------------------------------------------------------+
        | event                                                            |
        +------------------------------------------------------------------+
        | {"latency_ms":12,"service":"api","status":200,"tags":["cached"]} |
        | {"error":null,"latency_ms":85,"service":"api","status":500}      |
        | {"error":"Somethings not working","service":"api","status":504}  |
        +------------------------------------------------------------------+
        3 row(s) fetched.
        Elapsed [ELAPSED] seconds.

        +----------------------------------------------------------------------------------+
        | schema_of_variant(events.event)                                                  |
        +----------------------------------------------------------------------------------+
        | OBJECT<latency_ms: BIGINT, service: STRING, status: BIGINT, tags: ARRAY<STRING>> |
        | OBJECT<error: VOID, latency_ms: BIGINT, service: STRING, status: BIGINT>         |
        | OBJECT<error: STRING, service: STRING, status: BIGINT>                           |
        +----------------------------------------------------------------------------------+
        3 row(s) fetched.
        Elapsed [ELAPSED] seconds.

        0 row(s) fetched.
        Elapsed [ELAPSED] seconds.

        +----------+----------+
        | kind     | count(*) |
        +----------+----------+
        | commit   | 2        |
        | identity | 1        |
        +----------+----------+
        2 row(s) fetched.
        Elapsed [ELAPSED] seconds.


        stderr:
        Loading [BSKY_DIR]/file_0001.json.gz
        Loaded 3 rows
        "#);
    });
}

/// `load_bsky(n)` reads `file_000n.json.gz` from `DFV_BSKY_DIR` without
/// downloading when the file exists, and exposes it as `json_data` rows.
#[test]
fn test_load_bsky_reads_local_file() {
    let temp_dir = tempfile::tempdir().unwrap();
    write_bsky_file(temp_dir.path(), 2, &BSKY_ROWS);

    let context = snapshot_output(
        cargo_bin_cmd!("dfv")
            .env("DFV_BSKY_DIR", temp_dir.path())
            .args([
                "-c",
                "SELECT count(*) AS rows FROM load_bsky(2)",
                "-c",
                r#"SELECT json_data:kind::VARCHAR AS kind, count(*) AS n
                   FROM (SELECT parse_json(json_data) AS json_data FROM load_bsky(2))
                   GROUP BY kind ORDER BY n DESC"#,
            ]),
    );

    dfv_snapshot_settings().bind(|| {
        insta::assert_snapshot!(context, @r"
        exit code: 0
        stdout:
        dfv (DataFusion Variant CLI) v[VERSION] based on DataFusion CLI v[DATAFUSION_VERSION]
        +------+
        | rows |
        +------+
        | 3    |
        +------+
        1 row(s) fetched.
        Elapsed [ELAPSED] seconds.

        +----------+---+
        | kind     | n |
        +----------+---+
        | commit   | 2 |
        | identity | 1 |
        +----------+---+
        2 row(s) fetched.
        Elapsed [ELAPSED] seconds.


        stderr:
        Loading [BSKY_DIR]/file_0002.json.gz
        Loaded 3 rows
        Loading [BSKY_DIR]/file_0002.json.gz
        Loaded 3 rows
        ");
    });
}

/// Wait for another process to publish the file, then reuse it without downloading.
#[test]
fn test_load_bsky_waits_for_download() {
    let temp_dir = tempfile::tempdir().unwrap();
    let lock = std::fs::File::create(temp_dir.path().join("file_0001.json.lock")).unwrap();
    lock.lock().unwrap();

    let mut child = std::process::Command::new(assert_cmd::cargo::cargo_bin!("dfv"))
        .env("DFV_BSKY_DIR", temp_dir.path())
        .args(["-c", "SELECT count(*) AS rows FROM load_bsky(1)"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let stderr = child.stderr.take().unwrap();
    let (sender, receiver) = mpsc::channel();
    let reader = std::thread::spawn(move || {
        let mut stderr = BufReader::new(stderr);
        let mut line = String::new();
        stderr.read_line(&mut line).unwrap();
        sender.send(line).unwrap();
        std::io::read_to_string(stderr).unwrap()
    });

    let waiting = receiver.recv_timeout(Duration::from_secs(10));
    if !matches!(&waiting, Ok(line) if line.contains("Waiting for download")) {
        child.kill().unwrap();
        child.wait().unwrap();
        panic!("expected the CLI to wait for the download, got {waiting:?}");
    }
    assert!(child.try_wait().unwrap().is_none());
    write_bsky_file(temp_dir.path(), 1, &[r#"{"kind":"commit"}"#]);
    drop(lock);

    let output = child.wait_with_output().unwrap();
    let stderr = reader.join().unwrap();
    assert!(output.status.success(), "{stderr}");
    assert!(stderr.contains("Loaded 1 rows"), "{stderr}");
    assert!(!stderr.contains("Downloading"), "{stderr}");
}

/// `load_bsky` requires a single file number between 1 and 1000.
#[test]
fn test_load_bsky_rejects_bad_arguments() {
    let mut context = String::new();
    for args in ["0", "1001", "'x'", "1, 2"] {
        let sql = format!("SELECT * FROM load_bsky({args})");
        context += &format!(
            "=== {sql} ===\n{}\n",
            snapshot_output(cargo_bin_cmd!("dfv").args(["-c", &sql]))
        );
    }

    dfv_snapshot_settings().bind(|| {
        insta::assert_snapshot!(context, @"
        === SELECT * FROM load_bsky(0) ===
        exit code: 1
        stdout:
        dfv (DataFusion Variant CLI) v[VERSION] based on DataFusion CLI v[DATAFUSION_VERSION]
        Error: Error during planning: load_bsky expects a single file number between 1 and 1000, e.g. load_bsky(1)

        stderr:

        === SELECT * FROM load_bsky(1001) ===
        exit code: 1
        stdout:
        dfv (DataFusion Variant CLI) v[VERSION] based on DataFusion CLI v[DATAFUSION_VERSION]
        Error: Error during planning: load_bsky expects a single file number between 1 and 1000, e.g. load_bsky(1)

        stderr:

        === SELECT * FROM load_bsky('x') ===
        exit code: 1
        stdout:
        dfv (DataFusion Variant CLI) v[VERSION] based on DataFusion CLI v[DATAFUSION_VERSION]
        Error: Error during planning: load_bsky expects a single file number between 1 and 1000, e.g. load_bsky(1)

        stderr:

        === SELECT * FROM load_bsky(1, 2) ===
        exit code: 1
        stdout:
        dfv (DataFusion Variant CLI) v[VERSION] based on DataFusion CLI v[DATAFUSION_VERSION]
        Error: Error during planning: load_bsky expects a single file number between 1 and 1000, e.g. load_bsky(1)

        stderr:
        ");
    });
}
