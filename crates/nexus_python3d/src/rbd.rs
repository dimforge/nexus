//! Rigid-body building blocks mirroring the `rapier3d` prelude: body & collider
//! builders, shapes, handles, and joint builders.

use crate::math::Vec3;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rapier3d::prelude as rp;

/// Opaque handle to a rigid body, returned by `NexusState.insert_rigid_body`.
#[pyclass(name = "RigidBodyHandle", from_py_object)]
#[derive(Clone, Copy)]
pub struct RigidBodyHandle(pub rp::RigidBodyHandle);

#[pymethods]
impl RigidBodyHandle {
    fn __repr__(&self) -> String {
        format!("RigidBodyHandle({:?})", self.0)
    }
}

/// Opaque handle to an impulse joint.
#[pyclass(name = "ImpulseJointHandle", from_py_object)]
#[derive(Clone, Copy)]
pub struct ImpulseJointHandle(pub rp::ImpulseJointHandle);

/// Opaque handle to a multibody joint.
#[pyclass(name = "MultibodyJointHandle", from_py_object)]
#[derive(Clone, Copy)]
pub struct MultibodyJointHandle(pub rp::MultibodyJointHandle);

/// A collision shape (`rapier3d::prelude::SharedShape`).
#[pyclass(name = "SharedShape", from_py_object)]
#[derive(Clone)]
pub struct SharedShape(pub rp::SharedShape);

/// A built rigid body (`rapier3d::prelude::RigidBody`).
#[pyclass(name = "RigidBody", from_py_object)]
#[derive(Clone)]
pub struct RigidBody(pub rp::RigidBody);

/// A built collider (`rapier3d::prelude::Collider`).
#[pyclass(name = "Collider", from_py_object)]
#[derive(Clone)]
pub struct Collider(pub rp::Collider);

#[pymethods]
impl Collider {
    /// Returns a clone of the collider's shape (mirrors `collider.shared_shape().clone()`).
    fn shared_shape(&self) -> SharedShape {
        SharedShape(self.0.shared_shape().clone())
    }
}

/// Builder for a [`RigidBody`] (`rapier3d::prelude::RigidBodyBuilder`).
#[pyclass(name = "RigidBodyBuilder", from_py_object)]
#[derive(Clone)]
pub struct RigidBodyBuilder(pub rp::RigidBodyBuilder);

#[pymethods]
impl RigidBodyBuilder {
    #[staticmethod]
    fn dynamic() -> Self {
        Self(rp::RigidBodyBuilder::dynamic())
    }
    #[staticmethod]
    fn fixed() -> Self {
        Self(rp::RigidBodyBuilder::fixed())
    }
    #[staticmethod]
    fn kinematic_velocity_based() -> Self {
        Self(rp::RigidBodyBuilder::kinematic_velocity_based())
    }
    #[staticmethod]
    fn kinematic_position_based() -> Self {
        Self(rp::RigidBodyBuilder::kinematic_position_based())
    }

    fn translation(&self, t: Vec3) -> Self {
        Self(self.0.clone().translation(t.0))
    }
    /// Sets the orientation from an axis-angle (scaled-axis) vector.
    fn rotation(&self, axisangle: Vec3) -> Self {
        Self(self.0.clone().rotation(axisangle.0))
    }
    /// Sets the full initial pose (translation + rotation).
    fn pose(&self, pose: crate::math::Pose) -> Self {
        Self(self.0.clone().pose(pose.0))
    }
    fn linvel(&self, v: Vec3) -> Self {
        Self(self.0.clone().linvel(v.0))
    }
    fn angvel(&self, v: Vec3) -> Self {
        Self(self.0.clone().angvel(v.0))
    }
    fn gravity_scale(&self, scale: f32) -> Self {
        Self(self.0.clone().gravity_scale(scale))
    }
    fn additional_mass(&self, mass: f32) -> Self {
        Self(self.0.clone().additional_mass(mass))
    }
    fn ccd_enabled(&self, enabled: bool) -> Self {
        Self(self.0.clone().ccd_enabled(enabled))
    }
    fn can_sleep(&self, can_sleep: bool) -> Self {
        Self(self.0.clone().can_sleep(can_sleep))
    }

    fn build(&self) -> RigidBody {
        RigidBody(self.0.clone().build())
    }
}

/// Collision membership/filter bit masks (`rapier3d::prelude::InteractionGroups`).
///
/// Two colliders interact only if each one's `memberships` intersects the
/// other's `filter`. Use `none()` to make a collider ignore all contacts.
#[pyclass(name = "InteractionGroups", from_py_object)]
#[derive(Clone, Copy)]
pub struct InteractionGroups(pub rp::InteractionGroups);

#[pymethods]
impl InteractionGroups {
    /// A member of every group, and interacting with every group.
    #[staticmethod]
    fn all() -> Self {
        Self(rp::InteractionGroups::all())
    }
    /// A member of no group, interacting with nothing (ignores all contacts).
    #[staticmethod]
    fn none() -> Self {
        Self(rp::InteractionGroups::none())
    }
    /// Explicit `memberships` / `filter` bit masks (each a 32-bit group mask).
    #[staticmethod]
    fn new(memberships: u32, filter: u32) -> Self {
        Self(
            rp::InteractionGroups::all()
                .with_memberships(rp::Group::from_bits_truncate(memberships))
                .with_filter(rp::Group::from_bits_truncate(filter)),
        )
    }
}

/// Builder for a [`Collider`] (`rapier3d::prelude::ColliderBuilder`).
#[pyclass(name = "ColliderBuilder", from_py_object)]
#[derive(Clone)]
pub struct ColliderBuilder(pub rp::ColliderBuilder);

#[pymethods]
impl ColliderBuilder {
    #[staticmethod]
    fn cuboid(hx: f32, hy: f32, hz: f32) -> Self {
        Self(rp::ColliderBuilder::cuboid(hx, hy, hz))
    }
    #[staticmethod]
    fn ball(radius: f32) -> Self {
        Self(rp::ColliderBuilder::ball(radius))
    }
    #[staticmethod]
    fn capsule_x(half_height: f32, radius: f32) -> Self {
        Self(rp::ColliderBuilder::capsule_x(half_height, radius))
    }
    #[staticmethod]
    fn capsule_y(half_height: f32, radius: f32) -> Self {
        Self(rp::ColliderBuilder::capsule_y(half_height, radius))
    }
    #[staticmethod]
    fn capsule_z(half_height: f32, radius: f32) -> Self {
        Self(rp::ColliderBuilder::capsule_z(half_height, radius))
    }
    #[staticmethod]
    fn cylinder(half_height: f32, radius: f32) -> Self {
        Self(rp::ColliderBuilder::cylinder(half_height, radius))
    }
    #[staticmethod]
    fn cone(half_height: f32, radius: f32) -> Self {
        Self(rp::ColliderBuilder::cone(half_height, radius))
    }
    /// A triangle-mesh collider from `vertices` (list of `(x, y, z)`) and
    /// `indices` (list of `(i, j, k)`).
    ///
    /// The mesh is always built with parry's `ORIENTED` flag: the solver
    /// requires per-vertex/edge pseudo-normals for every trimesh collider (to
    /// resolve inside/outside), and only oriented meshes carry them. Provide a
    /// consistently-wound mesh so those normals point outward.
    #[staticmethod]
    fn trimesh(vertices: Vec<[f32; 3]>, indices: Vec<[u32; 3]>) -> PyResult<Self> {
        let verts: Vec<glamx::Vec3> = vertices
            .into_iter()
            .map(|[x, y, z]| glamx::Vec3::new(x, y, z))
            .collect();
        rp::ColliderBuilder::trimesh_with_flags(verts, indices, rp::TriMeshFlags::ORIENTED)
            .map(Self)
            .map_err(|e| PyValueError::new_err(format!("invalid trimesh: {e:?}")))
    }

    /// A convex-hull collider computed from a point cloud (list of `(x, y, z)`).
    #[staticmethod]
    fn convex_hull(points: Vec<[f32; 3]>) -> PyResult<Self> {
        let pts: Vec<glamx::Vec3> = points
            .into_iter()
            .map(|[x, y, z]| glamx::Vec3::new(x, y, z))
            .collect();
        rp::ColliderBuilder::convex_hull(&pts)
            .map(Self)
            .ok_or_else(|| PyValueError::new_err("convex hull computation failed"))
    }

    fn translation(&self, t: Vec3) -> Self {
        Self(self.0.clone().translation(t.0))
    }
    fn rotation(&self, axisangle: Vec3) -> Self {
        Self(self.0.clone().rotation(axisangle.0))
    }
    fn density(&self, density: f32) -> Self {
        Self(self.0.clone().density(density))
    }
    fn mass(&self, mass: f32) -> Self {
        Self(self.0.clone().mass(mass))
    }
    fn friction(&self, friction: f32) -> Self {
        Self(self.0.clone().friction(friction))
    }
    fn restitution(&self, restitution: f32) -> Self {
        Self(self.0.clone().restitution(restitution))
    }
    fn collision_groups(&self, groups: InteractionGroups) -> Self {
        Self(self.0.clone().collision_groups(groups.0))
    }
    fn solver_groups(&self, groups: InteractionGroups) -> Self {
        Self(self.0.clone().solver_groups(groups.0))
    }

    fn build(&self) -> Collider {
        Collider(self.0.clone().build())
    }
}

// ---------------------------------------------------------------------------
// Joints
// ---------------------------------------------------------------------------

/// Which joint degree of freedom a motor / limit acts on.
#[pyclass(name = "JointAxis", eq, eq_int, from_py_object)]
#[derive(Clone, Copy, PartialEq)]
pub enum JointAxis {
    LinX,
    LinY,
    LinZ,
    AngX,
    AngY,
    AngZ,
}

impl JointAxis {
    pub fn to_rapier(self) -> rp::JointAxis {
        match self {
            JointAxis::LinX => rp::JointAxis::LinX,
            JointAxis::LinY => rp::JointAxis::LinY,
            JointAxis::LinZ => rp::JointAxis::LinZ,
            JointAxis::AngX => rp::JointAxis::AngX,
            JointAxis::AngY => rp::JointAxis::AngY,
            JointAxis::AngZ => rp::JointAxis::AngZ,
        }
    }
}

/// Motor control model.
#[pyclass(name = "MotorModel", eq, eq_int, from_py_object)]
#[derive(Clone, Copy, PartialEq)]
pub enum MotorModel {
    AccelerationBased,
    ForceBased,
}

impl MotorModel {
    fn to_rapier(self) -> rp::MotorModel {
        match self {
            MotorModel::AccelerationBased => rp::MotorModel::AccelerationBased,
            MotorModel::ForceBased => rp::MotorModel::ForceBased,
        }
    }
}

/// A fully-specified joint (`rapier3d::prelude::GenericJoint`).
#[pyclass(name = "GenericJoint", from_py_object)]
#[derive(Clone)]
pub struct GenericJoint(pub rp::GenericJoint);

/// Builder for a fixed joint (welds two bodies together).
#[pyclass(name = "FixedJointBuilder", from_py_object)]
#[derive(Clone)]
pub struct FixedJointBuilder(pub rp::FixedJointBuilder);

#[pymethods]
impl FixedJointBuilder {
    #[staticmethod]
    fn new() -> Self {
        Self(rp::FixedJointBuilder::new())
    }
    fn local_anchor1(&self, anchor: Vec3) -> Self {
        Self(self.0.local_anchor1(anchor.0))
    }
    fn local_anchor2(&self, anchor: Vec3) -> Self {
        Self(self.0.local_anchor2(anchor.0))
    }
    fn contacts_enabled(&self, enabled: bool) -> Self {
        Self(self.0.contacts_enabled(enabled))
    }
    fn build(&self) -> GenericJoint {
        GenericJoint(self.0.into())
    }
}

/// Builder for a spherical (ball) joint. Motors and limits are per-axis.
#[pyclass(name = "SphericalJointBuilder", from_py_object)]
#[derive(Clone)]
pub struct SphericalJointBuilder(pub rp::SphericalJointBuilder);

#[pymethods]
impl SphericalJointBuilder {
    #[staticmethod]
    fn new() -> Self {
        Self(rp::SphericalJointBuilder::new())
    }
    fn local_anchor1(&self, anchor: Vec3) -> Self {
        Self(self.0.local_anchor1(anchor.0))
    }
    fn local_anchor2(&self, anchor: Vec3) -> Self {
        Self(self.0.local_anchor2(anchor.0))
    }
    fn contacts_enabled(&self, enabled: bool) -> Self {
        Self(self.0.contacts_enabled(enabled))
    }
    fn limits(&self, axis: JointAxis, min: f32, max: f32) -> Self {
        Self(self.0.limits(axis.to_rapier(), [min, max]))
    }
    fn motor_velocity(&self, axis: JointAxis, target_vel: f32, factor: f32) -> Self {
        Self(self.0.motor_velocity(axis.to_rapier(), target_vel, factor))
    }
    fn motor_position(&self, axis: JointAxis, target: f32, stiffness: f32, damping: f32) -> Self {
        Self(
            self.0
                .motor_position(axis.to_rapier(), target, stiffness, damping),
        )
    }
    fn motor_model(&self, axis: JointAxis, model: MotorModel) -> Self {
        Self(self.0.motor_model(axis.to_rapier(), model.to_rapier()))
    }
    fn motor_max_force(&self, axis: JointAxis, max_force: f32) -> Self {
        Self(self.0.motor_max_force(axis.to_rapier(), max_force))
    }
    fn build(&self) -> GenericJoint {
        GenericJoint(self.0.into())
    }
}

/// Builder for a revolute (hinge) joint.
#[pyclass(name = "RevoluteJointBuilder", from_py_object)]
#[derive(Clone)]
pub struct RevoluteJointBuilder(pub rp::RevoluteJointBuilder);

#[pymethods]
impl RevoluteJointBuilder {
    #[staticmethod]
    fn new(axis: Vec3) -> Self {
        Self(rp::RevoluteJointBuilder::new(axis.0))
    }
    fn local_anchor1(&self, anchor: Vec3) -> Self {
        Self(self.0.local_anchor1(anchor.0))
    }
    fn local_anchor2(&self, anchor: Vec3) -> Self {
        Self(self.0.local_anchor2(anchor.0))
    }
    fn limits(&self, min: f32, max: f32) -> Self {
        Self(self.0.limits([min, max]))
    }
    fn motor_velocity(&self, target_vel: f32, factor: f32) -> Self {
        Self(self.0.motor_velocity(target_vel, factor))
    }
    fn motor_position(&self, target_pos: f32, stiffness: f32, damping: f32) -> Self {
        Self(self.0.motor_position(target_pos, stiffness, damping))
    }
    fn motor_model(&self, model: MotorModel) -> Self {
        Self(self.0.motor_model(model.to_rapier()))
    }
    fn motor_max_force(&self, max_force: f32) -> Self {
        Self(self.0.motor_max_force(max_force))
    }
    fn contacts_enabled(&self, enabled: bool) -> Self {
        Self(self.0.contacts_enabled(enabled))
    }
    fn build(&self) -> GenericJoint {
        GenericJoint(self.0.into())
    }
}

/// Builder for a prismatic (slider) joint.
#[pyclass(name = "PrismaticJointBuilder", from_py_object)]
#[derive(Clone)]
pub struct PrismaticJointBuilder(pub rp::PrismaticJointBuilder);

#[pymethods]
impl PrismaticJointBuilder {
    #[staticmethod]
    fn new(axis: Vec3) -> Self {
        Self(rp::PrismaticJointBuilder::new(axis.0))
    }
    fn local_anchor1(&self, anchor: Vec3) -> Self {
        Self(self.0.local_anchor1(anchor.0))
    }
    fn local_anchor2(&self, anchor: Vec3) -> Self {
        Self(self.0.local_anchor2(anchor.0))
    }
    fn local_axis1(&self, axis: Vec3) -> Self {
        Self(self.0.local_axis1(axis.0))
    }
    fn local_axis2(&self, axis: Vec3) -> Self {
        Self(self.0.local_axis2(axis.0))
    }
    fn limits(&self, min: f32, max: f32) -> Self {
        Self(self.0.limits([min, max]))
    }
    fn motor_velocity(&self, target_vel: f32, factor: f32) -> Self {
        Self(self.0.motor_velocity(target_vel, factor))
    }
    fn motor_position(&self, target_pos: f32, stiffness: f32, damping: f32) -> Self {
        Self(self.0.motor_position(target_pos, stiffness, damping))
    }
    fn motor_model(&self, model: MotorModel) -> Self {
        Self(self.0.motor_model(model.to_rapier()))
    }
    fn motor_max_force(&self, max_force: f32) -> Self {
        Self(self.0.motor_max_force(max_force))
    }
    fn build(&self) -> GenericJoint {
        GenericJoint(self.0.into())
    }
}

/// Any joint argument accepted by `insert_*_joint`: a built [`GenericJoint`] or
/// any of the joint builders.
#[derive(FromPyObject)]
pub enum JointArg {
    Generic(GenericJoint),
    Fixed(FixedJointBuilder),
    Spherical(SphericalJointBuilder),
    Revolute(RevoluteJointBuilder),
    Prismatic(PrismaticJointBuilder),
}

impl JointArg {
    pub fn into_generic(self) -> rp::GenericJoint {
        match self {
            JointArg::Generic(j) => j.0,
            JointArg::Fixed(b) => b.0.into(),
            JointArg::Spherical(b) => b.0.into(),
            JointArg::Revolute(b) => b.0.into(),
            JointArg::Prismatic(b) => b.0.into(),
        }
    }
}
