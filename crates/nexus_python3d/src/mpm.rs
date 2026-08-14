//! Material-Point-Method types (`nexus3d::mpm::solver`).

use crate::math::Vec3;
use nexus3d::mpm::solver::{
    BoundaryCondition as RBoundaryCondition, Particle as RParticle,
    ParticleModel as RParticleModel, SimulationParams as RSimulationParams,
};
use pyo3::prelude::*;

/// MPM global simulation parameters.
#[pyclass(name = "SimulationParams", from_py_object)]
#[derive(Clone, Copy)]
pub struct SimulationParams(pub RSimulationParams);

#[pymethods]
impl SimulationParams {
    #[new]
    fn new(gravity: Vec3, dt: f32) -> Self {
        SimulationParams(RSimulationParams {
            gravity: gravity.0,
            dt,
        })
    }

    #[getter]
    fn dt(&self) -> f32 {
        self.0.dt
    }
    #[getter]
    fn gravity(&self) -> Vec3 {
        Vec3(self.0.gravity)
    }
}

/// A boundary condition applied to grid nodes at a coupled collider surface.
#[pyclass(name = "BoundaryCondition", from_py_object)]
#[derive(Clone, Copy)]
pub struct BoundaryCondition(pub RBoundaryCondition);

#[pymethods]
impl BoundaryCondition {
    /// No-slip: the material sticks to the surface (zero relative velocity).
    #[staticmethod]
    fn stick() -> Self {
        BoundaryCondition(RBoundaryCondition::stick())
    }
    /// Free-slip: the normal velocity component is removed, tangential kept.
    #[staticmethod]
    fn slip() -> Self {
        BoundaryCondition(RBoundaryCondition::slip())
    }
    /// Separating contact with Coulomb `friction` (`>= 0`): the material can
    /// pull away from the surface but is resisted tangentially.
    #[staticmethod]
    fn separate(friction: f32) -> Self {
        BoundaryCondition(RBoundaryCondition::separate(friction))
    }

    /// The Coulomb friction coefficient (only meaningful for `separate`).
    #[getter]
    fn friction(&self) -> f32 {
        self.0.friction
    }
}

/// A particle constitutive model.
#[pyclass(name = "ParticleModel", from_py_object)]
#[derive(Clone, Copy)]
pub struct ParticleModel(pub RParticleModel);

#[pymethods]
impl ParticleModel {
    #[staticmethod]
    fn elastic(young_modulus: f32, poisson_ratio: f32) -> Self {
        ParticleModel(RParticleModel::elastic(young_modulus, poisson_ratio))
    }
    #[staticmethod]
    fn elastic_neo_hookean(young_modulus: f32, poisson_ratio: f32) -> Self {
        ParticleModel(RParticleModel::elastic_neo_hookean(
            young_modulus,
            poisson_ratio,
        ))
    }
    #[staticmethod]
    fn sand(young_modulus: f32, poisson_ratio: f32) -> Self {
        ParticleModel(RParticleModel::sand(young_modulus, poisson_ratio))
    }
    #[staticmethod]
    fn sand_neo_hookean(young_modulus: f32, poisson_ratio: f32) -> Self {
        ParticleModel(RParticleModel::sand_neo_hookean(
            young_modulus,
            poisson_ratio,
        ))
    }
}

/// A single MPM particle.
#[pyclass(name = "Particle", from_py_object)]
#[derive(Clone, Copy)]
pub struct Particle(pub RParticle);

#[pymethods]
impl Particle {
    #[new]
    fn new(position: Vec3, radius: f32, density: f32, model: ParticleModel) -> Self {
        Particle(RParticle::new(position.0, radius, density, model.0))
    }

    /// The particle velocity (`particle.dynamics.velocity`).
    #[getter]
    fn velocity(&self) -> Vec3 {
        Vec3(self.0.dynamics.velocity)
    }
    #[setter]
    fn set_velocity(&mut self, v: Vec3) {
        self.0.dynamics.velocity = v.0;
    }

    #[getter]
    fn position(&self) -> Vec3 {
        Vec3(self.0.position)
    }
    #[setter]
    fn set_position(&mut self, p: Vec3) {
        self.0.position = p.0;
    }

    fn set_fixed(&mut self, fixed: bool) {
        self.0.dynamics.set_fixed(fixed);
    }
    fn set_damping(&mut self, damping: f32) {
        self.0.dynamics.set_damping(damping);
    }
    fn set_density(&mut self, density: f32) {
        self.0.dynamics.set_density(density);
    }
}
