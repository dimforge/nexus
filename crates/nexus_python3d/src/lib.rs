//! Python bindings for the 3D `nexus` GPU physics engine and its viewer.
//!
//! The API mirrors the Rust one closely: build bodies/colliders with the
//! `rapier`-style builders, insert them into a `NexusState`, register their
//! shapes with the `NexusViewer`, then drive the same `while
//! viewer.render_frame(): ...` loop. All GPU work is async in Rust; the
//! bindings block on it so Python code stays synchronous.

use pyo3::prelude::*;

pub mod loaders;
pub mod math;
pub mod nexus;
pub mod rbd;
pub mod viewer;

/// The `nexus3d` Python module.
#[pymodule]
fn nexus3d(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Math
    m.add_class::<math::Vec3>()?;
    m.add_class::<math::Vec4>()?;
    m.add_class::<math::Quat>()?;
    m.add_class::<math::Pose>()?;
    m.add_function(wrap_pyfunction!(math::vec3, m)?)?;
    m.add_function(wrap_pyfunction!(math::vec4, m)?)?;

    // Rigid bodies, colliders, shapes, handles
    m.add_class::<rbd::RigidBodyHandle>()?;
    m.add_class::<rbd::ImpulseJointHandle>()?;
    m.add_class::<rbd::MultibodyJointHandle>()?;
    m.add_class::<rbd::SharedShape>()?;
    m.add_class::<rbd::RigidBody>()?;
    m.add_class::<rbd::Collider>()?;
    m.add_class::<rbd::RigidBodyBuilder>()?;
    m.add_class::<rbd::ColliderBuilder>()?;
    m.add_class::<rbd::InteractionGroups>()?;

    // Joints
    m.add_class::<rbd::JointAxis>()?;
    m.add_class::<rbd::MotorModel>()?;
    m.add_class::<rbd::GenericJoint>()?;
    m.add_class::<rbd::FixedJointBuilder>()?;
    m.add_class::<rbd::SphericalJointBuilder>()?;
    m.add_class::<rbd::RevoluteJointBuilder>()?;
    m.add_class::<rbd::PrismaticJointBuilder>()?;

    // Core simulation
    m.add_class::<nexus::NexusCounts>()?;
    m.add_class::<nexus::GpuTimestamps>()?;
    m.add_class::<nexus::NexusState>()?;
    m.add_class::<nexus::NexusPipeline>()?;

    // Viewer
    m.add_class::<viewer::NexusViewer>()?;

    // Robot loaders (URDF / MJCF)
    m.add_class::<loaders::UrdfLoaderOptions>()?;
    m.add_class::<loaders::UrdfRobotHandles>()?;
    m.add_class::<loaders::MjcfSceneInfo>()?;

    Ok(())
}
