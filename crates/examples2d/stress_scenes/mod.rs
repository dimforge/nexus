//! Large rigid-body scenes for performance work, grouped under their own entry in the demo
//! picker.

mod box_rain;
mod builder;
mod joint_lattice;
mod pyramid;
mod registry;
mod wrecking_ball;

pub use registry::{demo_list, run};
