//! The debug renderer: turns the GPU state into debug segments.

use super::backend::{DebugLine, DebugPoint, LineCollector, hsla_to_rgba};
use super::mpm::{MpmDebugRenderMode, render_grid, render_particles};
use crate::rapier::pipeline::{DebugRenderMode, DebugRenderPipeline, DebugRenderStyle};
use crate::rapier::prelude::{ColliderSet, RigidBodyHandle, RigidBodySet};
use khal::backend::{Backend, GpuBackend};
use nexus::rbd::math::Pose;
use nexus::state::NexusState;

/// Debug-renderer settings, edited in the viewer UI.
#[derive(Clone, Debug, PartialEq)]
pub struct DebugRenderSettings {
    /// Main switch. When off, nothing is drawn or read back.
    pub enabled: bool,
    /// What to draw for the rigid-bodies (see [`DebugRenderMode`]).
    pub mode: DebugRenderMode,
    /// What to draw for MPM (see [`MpmDebugRenderMode`]).
    pub mpm_mode: MpmDebugRenderMode,
    /// Width of the debug segments, in pixels.
    pub line_width: f32,
    /// Size of the debug points, in pixels.
    pub point_size: f32,
    /// World-space length of the contact normals.
    pub contact_normal_length: f32,
    /// World-space length of the rigid-body axes.
    pub rigid_body_axes_length: f32,
    /// The time (in seconds) an MPM velocity segment stands for.
    pub mpm_velocity_scale: f32,
}

impl Default for DebugRenderSettings {
    fn default() -> Self {
        let style = DebugRenderStyle::default();
        Self {
            enabled: false,
            // Contacts are on by default (this is the main reason for a GPU debug renderer).
            // Collider shapes are off since the viewer already draws them.
            mode: DebugRenderMode::CONTACTS
                | DebugRenderMode::JOINTS
                | DebugRenderMode::RIGID_BODY_AXES,
            // The viewer already draws the particles, so only the grid is on by default.
            mpm_mode: MpmDebugRenderMode::GRID_BLOCKS,
            line_width: 2.0,
            point_size: 6.0,
            contact_normal_length: style.contact_normal_length,
            rigid_body_axes_length: style.rigid_body_axes_length,
            mpm_velocity_scale: 0.1,
        }
    }
}

/// A copy of the rapier sets of one environment, with poses updated from the GPU.
struct WorldMirror {
    bodies: RigidBodySet,
    colliders: ColliderSet,
}

/// Draws the scene as wireframes and contact markers.
#[derive(Default)]
pub struct DebugRenderer {
    pipeline: DebugRenderPipeline,
    mirrors: Vec<WorldMirror>,
    /// `(bodies, colliders)` counts of each environment when the mirrors were made.
    /// The mirrors are rebuilt when they change (bodies added or removed).
    mirrored_counts: Vec<(usize, usize)>,
    lines: Vec<DebugLine>,
    points: Vec<DebugPoint>,
}

impl DebugRenderer {
    /// The segments to draw this frame.
    pub fn lines(&self) -> &[DebugLine] {
        &self.lines
    }

    /// The points to draw this frame.
    pub fn points(&self) -> &[DebugPoint] {
        &self.points
    }

    /// Drops the pose mirrors (e.g. when switching demo). They are rebuilt by the next sync.
    pub fn clear_scene(&mut self) {
        self.mirrors.clear();
        self.mirrored_counts.clear();
        self.lines.clear();
        self.points.clear();
    }

    /// Rebuilds the debug geometry from the GPU state.
    /// Reads data back from the GPU, so only call it when enabled.
    pub async fn sync(
        &mut self,
        state: &NexusState,
        backend: &GpuBackend,
        settings: &DebugRenderSettings,
    ) {
        self.lines.clear();
        self.points.clear();

        if !settings.enabled {
            return;
        }

        if let Some(rbd) = state.rbd.as_ref()
            && !settings.mode.is_empty()
        {
            self.pipeline.mode = settings.mode;
            self.pipeline.style.contact_normal_length = settings.contact_normal_length;
            self.pipeline.style.rigid_body_axes_length = settings.rigid_body_axes_length;

            // Only the wireframes need the poses, so skip the readback for contacts alone.
            let wireframe = DebugRenderMode::COLLIDER_SHAPES
                | DebugRenderMode::COLLIDER_AABBS
                | DebugRenderMode::RIGID_BODY_AXES
                | DebugRenderMode::JOINTS;
            if settings.mode.intersects(wireframe) {
                let poses = backend
                    .slow_read_vec::<Pose>(rbd.body_poses().buffer())
                    .await
                    .unwrap_or_default();
                self.sync_mirrors(state, &poses);
                self.render_wireframes(state);
            }

            let contact_modes = DebugRenderMode::CONTACTS | DebugRenderMode::SOLVER_CONTACTS;
            if settings.mode.intersects(contact_modes) {
                self.render_contacts(rbd, backend, settings).await;
            }
        }

        if let Some(mpm) = state.mpm.as_ref()
            && !settings.mpm_mode.is_empty()
        {
            self.render_mpm(mpm, backend, settings).await;
        }
    }

    /// Draws the MPM particles and grid, read back from the GPU.
    async fn render_mpm(
        &mut self,
        mpm: &nexus::mpm::pipeline::MpmState,
        backend: &GpuBackend,
        settings: &DebugRenderSettings,
    ) {
        if settings.mpm_mode.needs_particles() {
            let particles = mpm.debug_particles(backend).await;
            render_particles(
                &particles,
                settings.mpm_mode,
                settings.mpm_velocity_scale,
                &mut self.lines,
                &mut self.points,
            );
        }

        if settings.mpm_mode.needs_grid() {
            let grid = mpm.debug_grid(backend).await;
            render_grid(
                &grid,
                settings.mpm_mode,
                settings.mpm_velocity_scale,
                &mut self.lines,
                &mut self.points,
            );
        }
    }

    /// Rebuilds the pose mirrors if the scene changed, then updates their poses.
    fn sync_mirrors(&mut self, state: &NexusState, poses: &[Pose]) {
        let counts: Vec<(usize, usize)> = (0..state.num_environments())
            .map(|env| {
                let world = state.rbd_world(env);
                (world.bodies.len(), world.colliders.len())
            })
            .collect();

        if counts != self.mirrored_counts {
            self.mirrors = (0..counts.len())
                .map(|env| {
                    let world = state.rbd_world(env);
                    WorldMirror {
                        bodies: world.bodies.clone(),
                        colliders: world.colliders.clone(),
                    }
                })
                .collect();
            self.mirrored_counts = counts;
        }

        for (env, mirror) in self.mirrors.iter_mut().enumerate() {
            let Some(map) = state.rbd2gpu.get(env) else {
                continue;
            };
            let pose_of = |handle: RigidBodyHandle| {
                map.get(handle.0)
                    .map(|r| r.gpu_id)
                    .filter(|id| *id != u32::MAX)
                    .and_then(|id| poses.get(id as usize))
                    .copied()
            };

            for (handle, rb) in mirror.bodies.iter_mut() {
                if let Some(pose) = pose_of(handle) {
                    rb.set_position(pose, false);
                }
            }
            // Collider poses are computed from the body poses (their offset is fixed),
            // which saves a readback.
            for (_, co) in mirror.colliders.iter_mut() {
                let Some(parent) = co.parent() else { continue };
                let local = co.position_wrt_parent().copied().unwrap_or_default();
                if let Some(pose) = pose_of(parent) {
                    co.set_position(pose * local);
                }
            }
        }
    }

    /// Runs rapier's debug-render passes on the pose mirrors.
    fn render_wireframes(&mut self, state: &NexusState) {
        let mut collector = LineCollector::default();

        for (env, mirror) in self.mirrors.iter().enumerate() {
            self.pipeline
                .render_rigid_bodies(&mut collector, &mirror.bodies);
            self.pipeline
                .render_colliders(&mut collector, &mirror.bodies, &mirror.colliders);
            // Joints only need the body poses, so the joint sets of the state are used directly.
            let world = state.rbd_world(env);
            self.pipeline.render_joints(
                &mut collector,
                &mirror.bodies,
                &world.impulse_joints,
                &world.multibody_joints,
            );
        }

        self.lines.append(&mut collector.lines);
    }

    /// Draws the GPU contacts like rapier's `render_contacts`: a segment between
    /// the two contact points, and the normal.
    async fn render_contacts(
        &mut self,
        rbd: &nexus::rbd::pipeline::RbdState,
        backend: &GpuBackend,
        settings: &DebugRenderSettings,
    ) {
        let contacts = rbd.debug_contacts(backend).await;
        let depth_color = hsla_to_rgba(self.pipeline.style.contact_depth_color);
        let normal_color = hsla_to_rgba(self.pipeline.style.contact_normal_color);
        let normal_len = settings.contact_normal_length;

        for contact in &contacts {
            if settings.mode.contains(DebugRenderMode::CONTACTS) {
                self.lines.push(DebugLine {
                    a: contact.point,
                    b: contact.point_b(),
                    color: depth_color,
                });
                self.lines.push(DebugLine {
                    a: contact.point,
                    b: contact.point + contact.normal * normal_len,
                    color: normal_color,
                });
                self.points.push(DebugPoint {
                    point: contact.point,
                    color: depth_color,
                });
            }

            if settings.mode.contains(DebugRenderMode::SOLVER_CONTACTS) {
                // The constraint acts on the middle of the two contact points.
                let point = contact.solver_point();
                self.lines.push(DebugLine {
                    a: point,
                    b: point + contact.normal * normal_len,
                    color: normal_color,
                });
                self.points.push(DebugPoint {
                    point,
                    color: normal_color,
                });
            }
        }
    }
}
