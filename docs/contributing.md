# Contributing

Contributions are welcome, including compatibility fixes, regression tests,
benchmarks, documentation, and integration examples.

Use a current stable Rust toolchain with `rustfmt` and `clippy`.

Run the development checks from the repository root:

```sh
cargo test
cargo fmt --all -- --check
cargo clippy -- -D warnings
```

`cargo test` also checks the Rust example in [Getting started](getting-started.md)
and the SQL results in the README and in the pages under `docs/`.

Regenerate the SQL reference after changing function documentation.
`cargo test` checks its examples, and CI fails if the committed file is stale:

```sh
cargo run --bin generate_function_docs
```

For Spark compatibility changes, reference the upstream behavior and add
regression coverage in the [SQL tests][sql-tests].

The [issue tracker][issue-tracker]
contains individual tasks and discussions. Overall direction is tracked
in [#69][roadmap].

## Documentation

This site is built with [Sphinx] from the pages under `docs/`. To build it
locally, run:

```sh
pip install -r docs/requirements.txt
sphinx-build -W --keep-going -b html docs target/docs
```

Or, if you have [uv] installed, build without setting up a Python environment:

```sh
uvx --with-requirements docs/requirements.txt --from sphinx sphinx-build -W --keep-going -b html docs target/docs
```

Then open `target/docs/index.html`. CI runs the same build, and warnings such as
broken links between pages fail it.

## Benchmarks

See [Benchmarks](benchmarks.md) for the function benchmarks and JSONBench.

[sql-tests]: https://github.com/datafusion-contrib/datafusion-variant/blob/main/tests/test_files
[issue-tracker]: https://github.com/datafusion-contrib/datafusion-variant/issues
[roadmap]: https://github.com/datafusion-contrib/datafusion-variant/issues/69
[Sphinx]: https://www.sphinx-doc.org/
[uv]: https://docs.astral.sh/uv/
