use crate::Vector;

/// Parameters for the MPM simulation.
///
/// In 2D, a padding field is added after gravity to satisfy uniform size/alignment requirements.
#[derive(Clone, Copy, Default)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct SimulationParams {
    /// Gravity vector (Vec2 in 2D, Vec3 in 3D).
    pub gravity: Vector,
    /// Padding required in 2D due to uniform size limits.
    #[cfg(feature = "dim2")]
    pub padding: f32,
    /// The simulation timestep.
    pub dt: f32,
}
