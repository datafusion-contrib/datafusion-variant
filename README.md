# datafusion-variant

This crate provides user-defined functions for efficient Variant type handling in Datafusion. Variant types enable semi-structured data storage and querying, supporting JSON-like nested structs with dynamic schemas.

This crate aims to achieve complete feature parity with Spark and Databricks Variant functions.

Contributers are welcomed! [Now check this out](https://www.youtube.com/watch?v=1dj1kCrUFCY)

# Status

`datafusion-variant` is still under development. Progress is tracked in https://github.com/datafusion-contrib/datafusion-variant/issues/2; once it's closed, the crate's output should be considered stable.

# Usage

```sh
# run all tests
cargo test

# run sqllogictests
cargo test --test sqllogictests
```
