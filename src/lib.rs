#![warn(clippy::all)]

mod shared;

mod cast_to_variant;
mod is_variant_null;
mod json_to_variant;
mod variant_get;
mod variant_pretty;
mod variant_to_json;

pub use cast_to_variant::*;
pub use is_variant_null::*;
pub use json_to_variant::*;
pub use variant_get::*;
pub use variant_pretty::*;
pub use variant_to_json::*;
