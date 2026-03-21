#![warn(clippy::all)]

mod shared;

mod cast_to_variant;
mod is_variant_null;
mod json_to_variant;
mod variant_get;
mod variant_get_bool;
mod variant_get_float;
mod variant_get_int;
mod variant_get_str;
mod variant_list_construct;
mod variant_list_delete;
mod variant_list_insert;
mod variant_object_construct;
mod variant_object_delete;
mod variant_object_insert;
mod variant_object_keys;
mod variant_pretty;
mod variant_to_json;

pub use cast_to_variant::*;
pub use is_variant_null::*;
pub use json_to_variant::*;
pub use variant_get::*;
pub use variant_get_bool::*;
pub use variant_get_float::*;
pub use variant_get_int::*;
pub use variant_get_str::*;
pub use variant_list_construct::*;
pub use variant_list_delete::*;
pub use variant_list_insert::*;
pub use variant_object_construct::*;
pub use variant_object_delete::*;
pub use variant_object_insert::*;
pub use variant_object_keys::*;
pub use variant_pretty::*;
pub use variant_to_json::*;
