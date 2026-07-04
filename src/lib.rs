// The umbrella pipeline/state currently only drive the rbd subsystem, so they
// are gated on it: without `rbd` this crate must still compile (e.g. the
// `cargo publish` verification build runs with default features only).
#[cfg(feature = "rbd")]
pub mod pipeline;
#[cfg(feature = "rbd")]
pub mod state;

#[cfg(all(feature = "dim2", feature = "rbd"))]
pub use nexus_rbd2d as rbd;
#[cfg(all(feature = "dim3", feature = "rbd"))]
pub use nexus_rbd3d as rbd;

#[cfg(feature = "rbd")]
pub use rbd::{parry, rapier};

pub mod prelude {
    #[cfg(feature = "rbd")]
    pub use crate::pipeline::*;
    #[cfg(feature = "rbd")]
    pub use crate::state::*;
}
