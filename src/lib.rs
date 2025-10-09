#![warn(clippy::all)]

mod shared;

mod json_to_variant;
mod variant_get;
mod variant_to_json;

pub use json_to_variant::*;
pub use variant_get::*;
pub use variant_to_json::*;
