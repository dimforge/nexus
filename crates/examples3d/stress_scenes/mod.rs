//! Large rigid-body scenes for performance work, grouped under their own entry in the demo
//! picker.

mod box_columns;
mod box_pile;
mod brick_ring;
mod brick_walls;
mod builder;
mod common;
mod jointed_drop;
mod ragdolls_on_cloth;
mod registry;

pub use registry::{demo_list, run};
