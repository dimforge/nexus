//! Force and velocity integration.

use crate::dynamics::body::GpuBodySet;
use crate::dynamics::{GpuMassProperties, GpuVelocity};
use crate::math::GpuSim;
use slang_hal::backend::Backend;
use slang_hal::function::GpuFunction;
use slang_hal::Shader;
use slang_hal::ShaderArgs;
use stensor::tensor::GpuTensor;

#[derive(Shader)]
#[shader(module = "nexus::dynamics::integrate")]
/// Shaders exposing composable functions for force and velocity integration.
pub struct WgIntegrate<B: Backend> {
    /// Compute shader for integrating forces and velocities of every rigid-body.
    pub integrate: GpuFunction<B>,
}

#[derive(ShaderArgs)]
struct IntegrateArgs<'a, B: Backend> {
    mprops: &'a GpuTensor<GpuMassProperties, B>,
    local_mprops: &'a GpuTensor<GpuMassProperties, B>,
    poses: &'a GpuTensor<GpuSim, B>,
    vels: &'a GpuTensor<GpuVelocity, B>,
}

impl<B: Backend> WgIntegrate<B> {
    /// Dispatch an invocation of [`WgIntegrate::integrate`] for integrating forces and velocities
    /// of every rigid-body in the given [`GpuBodySet`]:
    pub fn launch(
        &self,
        backend: &B,
        pass: &mut B::Pass,
        bodies: &GpuBodySet<B>,
    ) -> Result<(), B::Error> {
        let args = IntegrateArgs {
            mprops: &bodies.mprops,
            local_mprops: &bodies.local_mprops,
            poses: &bodies.poses,
            vels: &bodies.vels,
        };
        self.integrate
            .launch(backend, pass, &args, [bodies.len(), 1, 1])
    }
}
