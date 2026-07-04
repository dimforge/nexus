use super::SimulationBackend;
use crate::rbd::SimulationState;
use khal::backend::GpuBackend as KhalGpuBackend;
use nexus::rbd::math::Pose;
use nexus::rbd::pipeline::RunStats;
use rapier::dynamics::{CCDSolver, IntegrationParameters, IslandManager};
use rapier::geometry::{BroadPhaseBvh, ColliderSet, NarrowPhase};
use rapier::prelude::{
    ImpulseJointSet, JointAxis, MultibodyJointSet, PhysicsPipeline, RigidBodySet,
};

/// CPU-based physics backend using rapier
pub struct CpuBackend {
    pipeline: PhysicsPipeline,
    integration_parameters: IntegrationParameters,
    islands: IslandManager,
    broad_phase: BroadPhaseBvh,
    narrow_phase: NarrowPhase,
    bodies: RigidBodySet,
    colliders: ColliderSet,
    impulse_joints: ImpulseJointSet,
    multibody_joints: MultibodyJointSet,
    ccd_solver: CCDSolver,
    poses_cache: Vec<Pose>,
}

impl CpuBackend {
    /// See [`super::PhysicsBackend::set_multibody_motor_velocity`].
    pub fn set_multibody_motor_velocity(
        &mut self,
        _batch: u32,
        link_id: u32,
        axis: JointAxis,
        target_vel: f32,
    ) {
        // Walk every multibody and apply on the link whose body's local id
        // matches `link_id`. For a one-collider-per-body world the local id is
        // also the body id.
        let mut handle = None;
        for (h, _) in self.bodies.iter() {
            if h.into_raw_parts().0 as u32 == link_id {
                handle = Some(h);
                break;
            }
        }
        let Some(handle) = handle else { return };
        if let Some(link) = self.multibody_joints.rigid_body_link(handle) {
            let multibody_handle = link.multibody;
            let link_id_in_mb = link.id;
            if let Some(mb) = self.multibody_joints.get_multibody_mut(multibody_handle) {
                if let Some(link) = mb.link_mut(link_id_in_mb) {
                    link.joint.data.set_motor_velocity(axis, target_vel, 1.0);
                }
            }
        }
    }

    pub fn new(phys: &SimulationState) -> Self {
        let env = &phys.environments[0];
        let mut poses_cache = Vec::new();
        let mut shapes_cache = Vec::new();

        // Build initial poses and shapes from the first environment.
        for (_, co) in env.colliders.iter() {
            poses_cache.push(*co.position());
            shapes_cache.push(co.shared_shape().clone());
        }

        let mut params = IntegrationParameters::default();
        params.dt = env.sim_params.dt;
        Self {
            pipeline: PhysicsPipeline::new(),
            integration_parameters: params,
            islands: IslandManager::new(),
            broad_phase: BroadPhaseBvh::new(),
            narrow_phase: NarrowPhase::new(),
            bodies: env.bodies.clone(),
            colliders: env.colliders.clone(),
            impulse_joints: env.impulse_joints.clone(),
            multibody_joints: env.multibody_joints.clone(),
            ccd_solver: CCDSolver::new(),
            poses_cache,
        }
    }
}

impl SimulationBackend for CpuBackend {
    fn poses(&self) -> &[Pose] {
        &self.poses_cache
    }
    fn num_bodies(&self) -> usize {
        self.poses().len()
    }
    fn num_joints(&self) -> usize {
        self.impulse_joints.len()
    }

    fn num_batches(&self) -> usize {
        1
    }

    async fn step(&mut self, _gpu: Option<&KhalGpuBackend>) -> RunStats {
        let t0 = web_time::Instant::now();

        #[cfg(feature = "dim2")]
        let gravity = glamx::Vec2::Y * -9.81;
        #[cfg(feature = "dim3")]
        let gravity = glamx::Vec3::Y * -9.81;

        self.pipeline.step(
            gravity,
            &self.integration_parameters,
            &mut self.islands,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.bodies,
            &mut self.colliders,
            &mut self.impulse_joints,
            &mut self.multibody_joints,
            &mut self.ccd_solver,
            &(),
            &(),
        );
        let total_sim_time = t0.elapsed();

        // Update poses cache
        self.poses_cache.clear();
        for (_, co) in self.colliders.iter() {
            self.poses_cache.push(*co.position());
        }

        RunStats {
            total_simulation_time_with_readback: total_sim_time,
            ..Default::default()
        }
    }
}
