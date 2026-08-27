//! Articulated robots loaded from URDF or MJCF: link and joint metadata in the
//! multibody's own order, CPU-side kinematics (forward and inverse), and the
//! per-DoF PD gains used by `NexusState.set_robot_targets`.

use crate::math::{Pose, Quat, Vec3};
use crate::rbd::{RigidBodyHandle, SharedShape};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rapier3d::prelude as rp;
use std::collections::HashMap;

/// A robot inserted as one multibody in one environment.
///
/// DoF order is the multibody's assembly order: links in multibody order, each
/// link's free linear axes then its free angular axes. Joint `k` is the joint
/// between link `joint_link[k]` and its parent; fixed joints are not listed.
#[pyclass(name = "Robot", from_py_object)]
#[derive(Clone)]
pub struct Robot {
    #[pyo3(get)]
    pub env: usize,
    /// The root link's body, used to look the multibody up.
    pub root: rp::RigidBodyHandle,
    #[pyo3(get)]
    pub link_names: Vec<String>,
    pub link_bodies: Vec<rp::RigidBodyHandle>,
    #[pyo3(get)]
    pub joint_names: Vec<String>,
    /// Index (into `link_names`) of each joint's child link.
    #[pyo3(get)]
    pub joint_links: Vec<usize>,
    #[pyo3(get)]
    pub joint_dof_offsets: Vec<usize>,
    #[pyo3(get)]
    pub joint_ndofs: Vec<usize>,
    /// rapier axis index (`0..3` linear, `3..6` angular) of each DoF.
    #[pyo3(get)]
    pub dof_axes: Vec<u8>,
    /// Index (into `link_names`) of the link each DoF belongs to.
    #[pyo3(get)]
    pub dof_links: Vec<usize>,
    #[pyo3(get)]
    pub dof_lower: Vec<f32>,
    #[pyo3(get)]
    pub dof_upper: Vec<f32>,
    /// Per-DoF PD gains and force limit applied by `set_robot_targets`.
    #[pyo3(get, set)]
    pub kp: Vec<f32>,
    #[pyo3(get, set)]
    pub kv: Vec<f32>,
    #[pyo3(get, set)]
    pub max_force: Vec<f32>,
    /// `(body, shape, local_pose)` render shapes (URDF only; MJCF visuals are
    /// registered with the viewer by the loader).
    #[pyo3(get)]
    pub render_shapes: Vec<(RigidBodyHandle, SharedShape, Pose)>,
}

#[pymethods]
impl Robot {
    #[getter]
    fn n_dofs(&self) -> usize {
        self.dof_axes.len()
    }

    #[getter]
    fn n_links(&self) -> usize {
        self.link_names.len()
    }

    #[getter]
    fn root_body(&self) -> RigidBodyHandle {
        RigidBodyHandle(self.root)
    }

    /// Index of the link named `name`.
    pub fn link_index(&self, name: &str) -> PyResult<usize> {
        self.link_names
            .iter()
            .position(|n| n == name)
            .ok_or_else(|| PyValueError::new_err(format!("unknown link {name:?}")))
    }

    /// Index of the (articulated) joint named `name`.
    pub fn joint_index(&self, name: &str) -> PyResult<usize> {
        self.joint_names
            .iter()
            .position(|n| n == name)
            .ok_or_else(|| PyValueError::new_err(format!("unknown joint {name:?}")))
    }

    /// The rigid body of the link named `name`.
    fn link_body(&self, name: &str) -> PyResult<RigidBodyHandle> {
        Ok(RigidBodyHandle(self.link_bodies[self.link_index(name)?]))
    }

    /// Rigid bodies of every link, in link order.
    #[getter]
    fn link_body_handles(&self) -> Vec<RigidBodyHandle> {
        self.link_bodies
            .iter()
            .map(|h| RigidBodyHandle(*h))
            .collect()
    }

    /// DoF indices of joint `name`, in DoF order.
    fn joint_dofs(&self, name: &str) -> PyResult<Vec<usize>> {
        let j = self.joint_index(name)?;
        Ok((self.joint_dof_offsets[j]..self.joint_dof_offsets[j] + self.joint_ndofs[j]).collect())
    }

    /// Sets the PD gains (and optionally the force limit) of the given DoFs.
    #[pyo3(signature = (dofs, kp=None, kv=None, max_force=None))]
    fn set_pd_gains(
        &mut self,
        dofs: Vec<usize>,
        kp: Option<Vec<f32>>,
        kv: Option<Vec<f32>>,
        max_force: Option<Vec<f32>>,
    ) -> PyResult<()> {
        for (name, values, target) in [
            ("kp", kp, &mut self.kp),
            ("kv", kv, &mut self.kv),
            ("max_force", max_force, &mut self.max_force),
        ] {
            let Some(values) = values else { continue };
            if values.len() != dofs.len() {
                return Err(PyValueError::new_err(format!(
                    "{name} has {} entries for {} dofs",
                    values.len(),
                    dofs.len()
                )));
            }
            for (d, v) in dofs.iter().zip(values) {
                let slot = target
                    .get_mut(*d)
                    .ok_or_else(|| PyValueError::new_err(format!("dof {d} out of range")))?;
                *slot = v;
            }
        }
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "Robot(env={}, links={}, dofs={})",
            self.env,
            self.link_names.len(),
            self.dof_axes.len()
        )
    }
}

/// Free axes of a joint in rapier's assembly order (linear, then angular).
pub fn free_axes(joint: &rp::GenericJoint) -> Vec<u8> {
    let locked = joint.locked_axes.bits();
    (0..6u8).filter(|axis| locked & (1 << axis) == 0).collect()
}

/// Limits of `axis`, `(-inf, inf)` when the joint has none on that axis.
fn axis_limits(joint: &rp::GenericJoint, axis: u8) -> (f32, f32) {
    let mask = rp::JointAxesMask::from_bits_truncate(1 << axis);
    if joint.limit_axes.contains(mask) {
        let l = &joint.limits[axis as usize];
        (l.min, l.max)
    } else {
        (f32::NEG_INFINITY, f32::INFINITY)
    }
}

/// Builds a [`Robot`] from the multibody containing `any_link` in `world`, using
/// the loader's names: `body_names` maps each link body to its name, and
/// `child_joint_names` maps each non-root link body to the name of the joint
/// connecting it to its parent. Gains default to the joints' current position
/// motors (zero when there is none).
pub fn build_robot(
    world: &rp::PhysicsWorld,
    env: usize,
    any_link: rp::RigidBodyHandle,
    body_names: &HashMap<rp::RigidBodyHandle, String>,
    child_joint_names: &HashMap<rp::RigidBodyHandle, String>,
    render_shapes: Vec<(RigidBodyHandle, SharedShape, Pose)>,
) -> PyResult<Robot> {
    let link_id = world
        .multibody_joints
        .rigid_body_link(any_link)
        .ok_or_else(|| PyRuntimeError::new_err("robot root is not a multibody link"))?;
    let mb = world
        .multibody_joints
        .get_multibody(link_id.multibody)
        .ok_or_else(|| PyRuntimeError::new_err("robot multibody not found"))?;
    let root_is_dynamic = world
        .bodies
        .get(mb.root().rigid_body_handle())
        .map(|rb| rb.is_dynamic())
        .unwrap_or(false);

    let mut robot = Robot {
        env,
        root: mb.root().rigid_body_handle(),
        link_names: Vec::new(),
        link_bodies: Vec::new(),
        joint_names: Vec::new(),
        joint_links: Vec::new(),
        joint_dof_offsets: Vec::new(),
        joint_ndofs: Vec::new(),
        dof_axes: Vec::new(),
        dof_links: Vec::new(),
        dof_lower: Vec::new(),
        dof_upper: Vec::new(),
        kp: Vec::new(),
        kv: Vec::new(),
        max_force: Vec::new(),
        render_shapes,
    };
    for (link_idx, link) in mb.links().enumerate() {
        let body = link.rigid_body_handle();
        robot.link_names.push(
            body_names
                .get(&body)
                .cloned()
                .unwrap_or_else(|| format!("link_{link_idx}")),
        );
        robot.link_bodies.push(body);
        // A fixed root has no DoFs on the GPU even though rapier models the
        // root joint as free; mirror the GPU build.
        let axes = if link_idx == 0 && !root_is_dynamic {
            Vec::new()
        } else {
            free_axes(&link.joint().data)
        };
        if axes.is_empty() {
            continue;
        }
        robot.joint_names.push(
            child_joint_names
                .get(&body)
                .cloned()
                .unwrap_or_else(|| format!("joint_{link_idx}")),
        );
        robot.joint_links.push(link_idx);
        robot.joint_dof_offsets.push(robot.dof_axes.len());
        robot.joint_ndofs.push(axes.len());
        for axis in axes {
            let (lo, hi) = axis_limits(&link.joint().data, axis);
            let motor = link.joint().data.motor(joint_axis(axis)).copied();
            robot.dof_axes.push(axis);
            robot.dof_links.push(link_idx);
            robot.dof_lower.push(lo);
            robot.dof_upper.push(hi);
            robot.kp.push(motor.map(|m| m.stiffness).unwrap_or(0.0));
            robot.kv.push(motor.map(|m| m.damping).unwrap_or(0.0));
            robot
                .max_force
                .push(motor.map(|m| m.max_force).unwrap_or(f32::INFINITY));
        }
    }
    Ok(robot)
}

/// The single-axis mask of rapier axis index `axis`.
pub fn axis_mask(axis: u8) -> rp::JointAxesMask {
    rp::JointAxesMask::from_bits_truncate(1 << axis)
}

/// The `JointAxis` of rapier axis index `axis`.
pub fn joint_axis(axis: u8) -> rp::JointAxis {
    match axis {
        0 => rp::JointAxis::LinX,
        1 => rp::JointAxis::LinY,
        2 => rp::JointAxis::LinZ,
        3 => rp::JointAxis::AngX,
        4 => rp::JointAxis::AngY,
        _ => rp::JointAxis::AngZ,
    }
}

/// Converts a `(w, x, y, z)` quaternion.
pub fn quat_wxyz(q: [f32; 4]) -> Quat {
    Quat(glamx::Quat::from_xyzw(q[1], q[2], q[3], q[0]).normalize())
}

/// `(w, x, y, z)` components of a quaternion.
pub fn to_wxyz(q: glamx::Quat) -> [f32; 4] {
    [q.w, q.x, q.y, q.z]
}

/// A pose from a position and a `(w, x, y, z)` quaternion.
pub fn pose_from_wxyz(pos: [f32; 3], quat: [f32; 4]) -> rp::Pose {
    rp::Pose::from_parts(Vec3(glamx::Vec3::from(pos)).0, quat_wxyz(quat).0)
}
