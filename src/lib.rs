pub mod pipeline;
pub mod state;

#[cfg(all(feature = "dim2", feature = "rbd"))]
pub use nexus_rbd2d as rbd;
#[cfg(all(feature = "dim3", feature = "rbd"))]
pub use nexus_rbd3d as rbd;

pub use rbd::{parry, rapier};

pub mod prelude {
    pub use crate::pipeline::*;
    pub use crate::state::*;
}
