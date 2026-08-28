//! Per-environment reset primitives for batched RL envs.
//!
//! Two paths sit on top of the `gpu_mb_env_reset*` kernels:
//!
//! - [`GpuMultibodySnapshot`] plus [`GpuMultibodySet::reset_env_from_snapshot`]:
//!   one staging upload and one dispatch per reset, no GPU to CPU readback.
//! - [`GpuMultibodySet::publish_reset_templates`] plus
//!   [`GpuMultibodySet::encode_reset_envs_batch`]: the templates live on the
//!   GPU permanently and N resets ride a single dispatch, with the teleport
//!   offset applied in-kernel.
//!
//! Reset loops should prefer the second: it is what keeps a rollout free of
//! per-step host writes, and therefore capturable into a CUDA graph.

use super::multibody_set::GpuMultibodySet;
use crate::math::Vector;
use crate::shaders::dynamics::{
    GpuMbEnvReset, GpuMbEnvResetBatch, MULTIBODY_ROOT, MultibodyLinkStatic, MultibodyLinkWorkspace,
    WS_QUADS, ws_soa_from_structs, ws_soa_to_structs,
};
use glamx::{UVec4, Vec4};
use khal::BufferUsages;
use khal::Shader;
use khal::backend::{Backend, GpuBackend};
use vortx::tensor::Tensor;

/// CPU snapshot of one (single-batch) multibody template: the AoS per-link
/// workspace, the static link descriptors, and the generalized coordinates and
/// velocities of batch 0.
#[derive(Clone)]
pub struct GpuMultibodySnapshot {
    /// AoS per-link workspace of batch 0, `links_per_batch` entries including
    /// the padding slots. Converted to the SoA quad layout on upload.
    pub(super) links_workspace: Vec<MultibodyLinkWorkspace>,
    pub(super) links_static: Vec<MultibodyLinkStatic>,
    /// Generalized coordinates of batch 0 (`dofs_per_batch`).
    pub(super) dof_values: Vec<f32>,
    /// Generalized velocities of batch 0 (`dofs_per_batch`): the velocity
    /// section of `dof_state`, the sections after it being static config.
    pub(super) dof_vels: Vec<f32>,
}

impl GpuMultibodySnapshot {
    /// True for entries describing a real link. The buffers are padded to
    /// `links_per_batch` with zeroed slots (rb_id 0, parent 0, ndofs 0), a
    /// combination no real link can have: a chain's body-0 link is its root,
    /// and roots carry `parent_link_id == MULTIBODY_ROOT`.
    pub(super) fn link_is_valid(ls: &MultibodyLinkStatic) -> bool {
        ls.parent_link_id == MULTIBODY_ROOT || ls.ndofs > 0 || ls.rb_id != 0
    }

    /// Whether multibody `multibody_id` has an unlocked (floating) root, i.e.
    /// whether an offset reset may move it.
    pub(super) fn mb_root_is_free(&self, multibody_id: u32) -> bool {
        self.links_static.iter().any(|ls| {
            Self::link_is_valid(ls)
                && ls.multibody_id == multibody_id
                && ls.parent_link_id == MULTIBODY_ROOT
                && ls.data.locked_axes == 0
        })
    }

    /// Calls `f(rb_id)` for every rigid body backing a link of a free-rooted
    /// (floating-base) multibody: the set of bodies an offset reset moves.
    pub(crate) fn for_each_link_rb_id(&self, mut f: impl FnMut(u32)) {
        for ls in &self.links_static {
            if Self::link_is_valid(ls) && self.mb_root_is_free(ls.multibody_id) {
                f(ls.rb_id);
            }
        }
    }

    /// A copy with every floating-base multibody translated by `offset` (world
    /// frame). Rotations, joint coordinates past the free linear DoFs,
    /// velocities and `dof_values` are translation-invariant; the free root's
    /// world position lives in `coords[0..3]` and `local_to_parent` (a root's
    /// parent frame is the world), and each link's `local_to_world` carries its
    /// body pose. Fixed-base multibodies are untouched. `body_poses`, owned by
    /// the caller, must be translated for the same rb ids.
    pub(crate) fn translated(&self, offset: Vector) -> GpuMultibodySnapshot {
        let mut out = self.clone();
        for (ws, ls) in out.links_workspace.iter_mut().zip(&self.links_static) {
            if !Self::link_is_valid(ls) || !self.mb_root_is_free(ls.multibody_id) {
                continue;
            }
            ws.local_to_world.translation += offset;
            if ls.parent_link_id == MULTIBODY_ROOT {
                ws.local_to_parent.translation += offset;
                ws.coords[0] += offset.x;
                ws.coords[1] += offset.y;
                ws.coords[2] += offset.z;
            }
        }
        out
    }
}

/// `#[derive(Shader)]` supplies `from_backend`, loading the embedded entry.
#[derive(Shader)]
struct EnvResetShader {
    kernel: GpuMbEnvReset,
}

/// `#[derive(Shader)]` supplies `from_backend` for the batched entry.
#[derive(Shader)]
struct EnvResetBatchShader {
    kernel: GpuMbEnvResetBatch,
}

/// Shader bundle plus persistent staging buffers for the per-env reset
/// scatter. Created on first reset, so the allocations stay outside any
/// captured region.
pub(super) struct EnvResetBundle {
    shader: EnvResetShader,
    staging_ws: Tensor<Vec4>,
    staging_links: Tensor<MultibodyLinkStatic>,
    staging_dofs: Tensor<f32>,
    params: Tensor<UVec4>,
}

impl EnvResetBundle {
    fn new(backend: &GpuBackend, lpb: u32, dpb: u32) -> Self {
        let storage = BufferUsages::STORAGE | BufferUsages::COPY_DST;
        let uniform = BufferUsages::STORAGE | BufferUsages::UNIFORM | BufferUsages::COPY_DST;
        Self {
            shader: EnvResetShader::from_backend(backend).unwrap(),
            staging_ws: Tensor::vector(
                backend,
                &vec![Vec4::ZERO; (lpb * WS_QUADS).max(1) as usize],
                storage,
            )
            .unwrap(),
            staging_links: Tensor::vector(
                backend,
                &vec![<MultibodyLinkStatic as bytemuck::Zeroable>::zeroed(); lpb.max(1) as usize],
                storage,
            )
            .unwrap(),
            staging_dofs: Tensor::vector(
                backend,
                &vec![0.0f32; (2 * dpb).max(1) as usize],
                storage,
            )
            .unwrap(),
            params: Tensor::scalar(backend, UVec4::new(0, 0, lpb, dpb), uniform).unwrap(),
        }
    }
}

/// GPU-resident reset templates plus the batch-reset shader.
pub(super) struct ResetTemplatesMb {
    ws: Tensor<Vec4>,
    links: Tensor<MultibodyLinkStatic>,
    dofs: Tensor<f32>,
    flags: Tensor<u32>,
    shader: EnvResetBatchShader,
    /// Host copies, used to keep the `links_static` mirror in step.
    mirror_links: Vec<Vec<MultibodyLinkStatic>>,
}

impl GpuMultibodySet {
    /// Reads this set's batch-0 state off the GPU into a CPU snapshot. Call it
    /// once per template at setup (typically on a single-env set) and pass the
    /// result to [`Self::reset_env_from_snapshot`] for readback-free resets.
    pub async fn snapshot(&self, backend: &GpuBackend) -> GpuMultibodySnapshot {
        let nb = self.num_batches;
        let lpb = self.links_per_batch as usize;
        let dpb = self.dofs_per_batch as usize;

        let mut ws_soa: Vec<Vec4> = bytemuck::zeroed_vec(self.links_workspace.len() as usize);
        backend
            .slow_read_buffer(self.links_workspace.buffer(), &mut ws_soa)
            .await
            .unwrap();
        let mut ls_all: Vec<MultibodyLinkStatic> =
            bytemuck::zeroed_vec(self.links_static.len() as usize);
        backend
            .slow_read_buffer(self.links_static.buffer(), &mut ls_all)
            .await
            .unwrap();
        let mut dv_all: Vec<f32> = bytemuck::zeroed_vec(self.dof_values.len() as usize);
        backend
            .slow_read_buffer(self.dof_values.buffer(), &mut dv_all)
            .await
            .unwrap();
        let mut ds_all: Vec<f32> = bytemuck::zeroed_vec(self.dof_state.len() as usize);
        backend
            .slow_read_buffer(self.dof_state.buffer(), &mut ds_all)
            .await
            .unwrap();

        // Gather batch 0 out of the interleave; the workspace is de-SoA'd
        // through the shared layout accessors, so it stays one source of truth
        // with the kernels. `ws_soa_to_structs` lays batch `b` out at
        // `b * links_cap`, so batch 0 is the leading `lpb` entries.
        let mut links_workspace = ws_soa_to_structs(&ws_soa, lpb as u32, nb);
        links_workspace.truncate(lpb);
        GpuMultibodySnapshot {
            links_workspace,
            links_static: (0..lpb).map(|k| ls_all[k * nb as usize]).collect(),
            dof_values: (0..dpb).map(|d| dv_all[d * nb as usize]).collect(),
            dof_vels: (0..dpb).map(|d| ds_all[d * nb as usize]).collect(),
        }
    }

    /// Resets env `dst_env` from a CPU snapshot.
    pub fn reset_env_from_snapshot(
        &mut self,
        backend: &GpuBackend,
        dst_env: u32,
        snap: &GpuMultibodySnapshot,
    ) {
        if self.is_empty() {
            return;
        }
        let nb = self.num_batches;
        let lpb = self.links_per_batch;
        let dpb = self.dofs_per_batch;
        debug_assert_eq!(snap.links_static.len(), lpb as usize);
        debug_assert_eq!(snap.dof_values.len(), dpb as usize);

        // Keep the host mirror in lockstep: the motor setters read-modify-write
        // it.
        for k in 0..lpb as usize {
            self.links_static_mirror[k * nb as usize + dst_env as usize] = snap.links_static[k];
        }

        // Take the bundle out so the live buffers below can be borrowed
        // mutably at the same time.
        let mut bundle = match self.env_reset.take() {
            Some(b) => b,
            None => EnvResetBundle::new(backend, lpb, dpb),
        };

        let ws = ws_soa_from_structs(&snap.links_workspace, lpb, 1);
        backend
            .write_buffer(bundle.staging_ws.buffer_mut(), 0, &ws)
            .unwrap();
        backend
            .write_buffer(bundle.staging_links.buffer_mut(), 0, &snap.links_static)
            .unwrap();
        let mut dofs = snap.dof_values.clone();
        dofs.extend_from_slice(&snap.dof_vels);
        if !dofs.is_empty() {
            backend
                .write_buffer(bundle.staging_dofs.buffer_mut(), 0, &dofs)
                .unwrap();
        }
        bundle.params = Tensor::scalar(
            backend,
            UVec4::new(dst_env, nb, lpb, dpb),
            BufferUsages::STORAGE | BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        )
        .unwrap();

        let mut encoder = backend.begin_encoding();
        {
            use khal::backend::Encoder as _;
            let mut pass = encoder.begin_pass("[RBD] mb-env-reset", None);
            bundle
                .shader
                .kernel
                .call(
                    &mut pass,
                    lpb * WS_QUADS,
                    &bundle.staging_ws,
                    &bundle.staging_links,
                    &bundle.staging_dofs,
                    &mut self.links_workspace,
                    &mut self.links_static,
                    &mut self.dof_values,
                    &mut self.dof_state,
                    &bundle.params,
                )
                .unwrap();
        }
        backend.submit(encoder).unwrap();
        self.env_reset = Some(bundle);
    }

    /// Uploads the reset templates once as GPU-resident blobs (SoA workspace,
    /// links, coords and velocities) plus the per-link translate flags the
    /// batch kernel needs, enabling [`Self::encode_reset_envs_batch`]. A host
    /// copy of each template's `links_static` is kept so the batch reset can
    /// maintain the CPU mirror.
    pub fn publish_reset_templates(
        &mut self,
        backend: &GpuBackend,
        snaps: &[&GpuMultibodySnapshot],
    ) {
        if self.is_empty() || snaps.is_empty() {
            return;
        }
        let lpb = self.links_per_batch as usize;
        let dpb = self.dofs_per_batch as usize;
        let storage = BufferUsages::STORAGE | BufferUsages::COPY_DST;

        let mut ws = Vec::with_capacity(snaps.len() * lpb * WS_QUADS as usize);
        let mut links = Vec::with_capacity(snaps.len() * lpb);
        let mut dofs = Vec::with_capacity(snaps.len() * 2 * dpb);
        let mut mirror_links = Vec::with_capacity(snaps.len());
        for snap in snaps {
            debug_assert_eq!(snap.links_static.len(), lpb);
            debug_assert_eq!(snap.dof_values.len(), dpb);
            ws.extend_from_slice(&ws_soa_from_structs(&snap.links_workspace, lpb as u32, 1));
            links.extend_from_slice(&snap.links_static);
            dofs.extend_from_slice(&snap.dof_values);
            dofs.extend_from_slice(&snap.dof_vels);
            mirror_links.push(snap.links_static.clone());
        }
        // Per-link translate flags, constant per robot and identical across
        // templates: bit 0 = valid link of a free-root multibody, bit 1 = the
        // root link itself. Matches `GpuMultibodySnapshot::translated`.
        let flags: Vec<u32> = snaps[0]
            .links_static
            .iter()
            .map(|ls| {
                let movable = GpuMultibodySnapshot::link_is_valid(ls)
                    && snaps[0].mb_root_is_free(ls.multibody_id);
                (movable as u32) | (((movable && ls.parent_link_id == MULTIBODY_ROOT) as u32) << 1)
            })
            .collect();

        self.reset_templates = Some(ResetTemplatesMb {
            ws: Tensor::vector(backend, &ws, storage).unwrap(),
            links: Tensor::vector(backend, &links, storage).unwrap(),
            dofs: Tensor::vector(backend, &dofs, storage).unwrap(),
            flags: Tensor::vector(backend, &flags, storage).unwrap(),
            shader: EnvResetBatchShader::from_backend(backend).unwrap(),
            mirror_links,
        });
    }

    /// Encodes one dispatch resetting every `(dst_env, template)` in `resets`
    /// from the resident templates, translating each by its `offsets` entry and
    /// writing its `dof_vels` slice (`dofs_per_batch` floats per reset) into
    /// the velocity section. Only the compact reset list is uploaded. The host
    /// `links_static` mirror is refreshed for the reset envs.
    ///
    /// [`Self::publish_reset_templates`] must have run first.
    pub fn encode_reset_envs_batch(
        &mut self,
        backend: &GpuBackend,
        enc: &mut <GpuBackend as Backend>::Encoder,
        resets: &[UVec4],
        offsets: &[Vec4],
        dof_vels: &[f32],
    ) {
        use khal::backend::Encoder as _;
        let n = resets.len() as u32;
        if n == 0 || self.is_empty() {
            return;
        }
        let nb = self.num_batches;
        let lpb = self.links_per_batch;
        let dpb = self.dofs_per_batch;
        debug_assert_eq!(dof_vels.len(), (n * dpb) as usize);
        let tpl = self
            .reset_templates
            .take()
            .expect("publish_reset_templates must run first");

        // Host mirror lockstep: the motor setters read-modify-write it.
        for meta in resets {
            let (env, t) = (meta.x as usize, meta.y as usize);
            for (k, ls) in tpl.mirror_links[t].iter().enumerate() {
                self.links_static_mirror[k * nb as usize + env] = *ls;
            }
        }

        let storage = BufferUsages::STORAGE | BufferUsages::COPY_DST;
        let t_resets = Tensor::vector(backend, resets, storage).unwrap();
        let t_offs = Tensor::vector(backend, offsets, storage).unwrap();
        let t_vels = Tensor::vector(backend, dof_vels, storage).unwrap();
        let params = Tensor::scalar(
            backend,
            UVec4::new(nb, lpb, dpb, n),
            BufferUsages::STORAGE | BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        )
        .unwrap();
        {
            let mut pass = enc.begin_pass("[RBD] mb-env-reset-batch", None);
            tpl.shader
                .kernel
                .call(
                    &mut pass,
                    [lpb * WS_QUADS, n, 1],
                    &tpl.ws,
                    &tpl.links,
                    &tpl.dofs,
                    &tpl.flags,
                    &t_resets,
                    &t_offs,
                    &t_vels,
                    &mut self.links_workspace,
                    &mut self.links_static,
                    &mut self.dof_values,
                    &mut self.dof_state,
                    &params,
                )
                .unwrap();
        }
        self.reset_templates = Some(tpl);
    }
}
