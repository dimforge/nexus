//! Vector / quaternion / pose types mirroring `glamx` (re-exported by rapier as
//! `Vec3`, `Vec4`, `Quat`, `Pose`).

use pyo3::prelude::*;

/// A 3D vector (`glamx::Vec3`).
#[pyclass(name = "Vec3", from_py_object)]
#[derive(Clone, Copy)]
pub struct Vec3(pub glamx::Vec3);

#[pymethods]
impl Vec3 {
    #[new]
    fn new(x: f32, y: f32, z: f32) -> Self {
        Self(glamx::Vec3::new(x, y, z))
    }

    #[classattr]
    const ZERO: Vec3 = Vec3(glamx::Vec3::ZERO);
    #[classattr]
    const X: Vec3 = Vec3(glamx::Vec3::X);
    #[classattr]
    const Y: Vec3 = Vec3(glamx::Vec3::Y);
    #[classattr]
    const Z: Vec3 = Vec3(glamx::Vec3::Z);

    #[getter]
    fn x(&self) -> f32 {
        self.0.x
    }
    #[setter]
    fn set_x(&mut self, v: f32) {
        self.0.x = v;
    }
    #[getter]
    fn y(&self) -> f32 {
        self.0.y
    }
    #[setter]
    fn set_y(&mut self, v: f32) {
        self.0.y = v;
    }
    #[getter]
    fn z(&self) -> f32 {
        self.0.z
    }
    #[setter]
    fn set_z(&mut self, v: f32) {
        self.0.z = v;
    }

    fn __add__(&self, rhs: Vec3) -> Vec3 {
        Vec3(self.0 + rhs.0)
    }
    fn __sub__(&self, rhs: Vec3) -> Vec3 {
        Vec3(self.0 - rhs.0)
    }
    fn __mul__(&self, rhs: f32) -> Vec3 {
        Vec3(self.0 * rhs)
    }
    fn __rmul__(&self, rhs: f32) -> Vec3 {
        Vec3(self.0 * rhs)
    }
    fn __neg__(&self) -> Vec3 {
        Vec3(-self.0)
    }

    fn __repr__(&self) -> String {
        format!("Vec3({}, {}, {})", self.0.x, self.0.y, self.0.z)
    }
}

/// Convenience constructor matching `glamx::vec3`.
#[pyfunction]
pub fn vec3(x: f32, y: f32, z: f32) -> Vec3 {
    Vec3(glamx::Vec3::new(x, y, z))
}

/// A 4D vector (`glamx::Vec4`), commonly an RGBA color.
#[pyclass(name = "Vec4", from_py_object)]
#[derive(Clone, Copy)]
pub struct Vec4(pub glamx::Vec4);

#[pymethods]
impl Vec4 {
    #[new]
    fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self(glamx::Vec4::new(x, y, z, w))
    }

    #[getter]
    fn x(&self) -> f32 {
        self.0.x
    }
    #[getter]
    fn y(&self) -> f32 {
        self.0.y
    }
    #[getter]
    fn z(&self) -> f32 {
        self.0.z
    }
    #[getter]
    fn w(&self) -> f32 {
        self.0.w
    }

    fn __repr__(&self) -> String {
        format!(
            "Vec4({}, {}, {}, {})",
            self.0.x, self.0.y, self.0.z, self.0.w
        )
    }
}

/// Convenience constructor matching `glamx::vec4`.
#[pyfunction]
pub fn vec4(x: f32, y: f32, z: f32, w: f32) -> Vec4 {
    Vec4(glamx::Vec4::new(x, y, z, w))
}

/// A unit quaternion rotation (`glamx::Quat`).
#[pyclass(name = "Quat", from_py_object)]
#[derive(Clone, Copy)]
pub struct Quat(pub glamx::Quat);

#[pymethods]
impl Quat {
    #[classattr]
    const IDENTITY: Quat = Quat(glamx::Quat::IDENTITY);

    #[staticmethod]
    fn from_axis_angle(axis: Vec3, angle: f32) -> Self {
        Self(glamx::Quat::from_axis_angle(axis.0, angle))
    }

    #[staticmethod]
    fn from_rotation_x(angle: f32) -> Self {
        Self(glamx::Quat::from_rotation_x(angle))
    }

    #[staticmethod]
    fn from_rotation_y(angle: f32) -> Self {
        Self(glamx::Quat::from_rotation_y(angle))
    }

    #[staticmethod]
    fn from_rotation_z(angle: f32) -> Self {
        Self(glamx::Quat::from_rotation_z(angle))
    }

    /// Rotation from a scaled-axis (axis-angle) vector.
    #[staticmethod]
    fn from_scaled_axis(axisangle: Vec3) -> Self {
        Self(glamx::Quat::from_scaled_axis(axisangle.0))
    }

    fn __mul__(&self, rhs: Quat) -> Quat {
        Quat(self.0 * rhs.0)
    }

    fn __repr__(&self) -> String {
        let v = self.0;
        format!("Quat({}, {}, {}, {})", v.x, v.y, v.z, v.w)
    }
}

/// A rigid-body transform: rotation + translation (`glamx::Pose3`, aliased
/// `Pose` in the rapier prelude).
#[pyclass(name = "Pose", from_py_object)]
#[derive(Clone, Copy)]
pub struct Pose(pub glamx::Pose3);

#[pymethods]
impl Pose {
    /// `Pose::new(translation, axisangle)` — translation + axis-angle rotation.
    #[new]
    fn new(translation: Vec3, axisangle: Vec3) -> Self {
        Self(glamx::Pose3::new(translation.0, axisangle.0))
    }

    #[classattr]
    const IDENTITY: Pose = Pose(glamx::Pose3::IDENTITY);

    #[staticmethod]
    fn from_translation(translation: Vec3) -> Self {
        Self(glamx::Pose3::from_translation(translation.0))
    }

    #[staticmethod]
    fn from_rotation(rotation: Quat) -> Self {
        Self(glamx::Pose3::from_rotation(rotation.0))
    }

    #[staticmethod]
    fn from_parts(translation: Vec3, rotation: Quat) -> Self {
        Self(glamx::Pose3::from_parts(translation.0, rotation.0))
    }

    #[getter]
    fn translation(&self) -> Vec3 {
        Vec3(self.0.translation)
    }

    #[getter]
    fn rotation(&self) -> Quat {
        Quat(self.0.rotation)
    }

    fn __repr__(&self) -> String {
        let t = self.0.translation;
        format!("Pose(translation=({}, {}, {}))", t.x, t.y, t.z)
    }
}
