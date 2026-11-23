# Getting started

## Quick start

First, enable Variant in your project by adding the following to `Cargo.toml`:

```toml
[dependencies]
datafusion-variant = { git = "https://github.com/datafusion-contrib/datafusion-variant.git" }
datafusion = { git = "https://github.com/apache/datafusion.git", branch = "branch-56" } # match this crate
parquet = { version = "60", features = ["variant_experimental"] } # enable Parquet Variant support
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

Then, configure your DataFusion [`SessionContext`] to use the Variant functions,
SQL field access, and `VARIANT` type with `register_all`. See the
[Apache DataFusion documentation] for more details on creating a session and
executing queries.

```rust
use datafusion::prelude::*;
use datafusion_variant::register_all;

#[tokio::main]
async fn main() -> datafusion::error::Result<()> {
    let mut ctx = SessionContext::new();
    register_all(&mut ctx)?;
    Ok(())
}
```

> [!WARNING]
> Call `register_all` during session setup, before preparing statements. It replaces
> existing functions with the same names and the type planner, and clears prepared
> statements.

Congratulations, you can now query semi-structured data with the Variant type!

## Run SQL

You can run the SQL examples in this guide like any other DataFusion SQL
statement using the `SessionContext` (e.g.
[`ctx.sql(sql).await?.collect().await?`][session-context-sql]).

## Try it with `dfv`

This repository also includes a command-line tool `dfv` with the Variant
functionality preinstalled to try the examples interactively. Run it like this:

```shell
# Drop `--release` for faster compile times if you aren't measuring performance.
cargo run --release
```

The [cookbook](cookbook/load-and-query-json.md) has examples of common tasks to try.

[`SessionContext`]: https://docs.rs/datafusion/latest/datafusion/execution/context/struct.SessionContext.html
[Apache DataFusion documentation]: https://datafusion.apache.org/library-user-guide/using-the-sql-api.html
[session-context-sql]: https://docs.rs/datafusion/latest/datafusion/execution/context/struct.SessionContext.html#method.sql
