//! Mass splitting of bodies with many contacts ("hubs").
//!
//! See the `mass_splitting` shader module: a hub body is split into one sub-body per contact,
//! each with its own solver velocity slot, so its contacts don't all need different colors.

use crate::shaders::broad_phase::ContactPlan;
use crate::shaders::dynamics::{
    GpuHubAssign, GpuHubAverage, GpuHubScatter, GpuHubSplitConstraints,
    GpuHubUpdateEffectiveMasses, GpuHubWarmstart, HUB_MIN_CONSTRAINTS, NOT_A_HUB,
    TwoBodyConstraint, Velocity, WorldMassProperties,
};
use crate::shaders::utils::BatchIndices;
use khal::backend::{GpuBackend, GpuBackendError, GpuPass};
use khal::{BufferUsages, Shader};
use vortx::tensor::Tensor;

/// Buffers splitting bodies with many contacts into one sub-body per contact.
pub struct HubState {
    /// Per body: its first sub-body slot, or `NOT_A_HUB`.
    pub(crate) first_slot: Tensor<u32>,
    /// The split bodies of the current step.
    list: Tensor<u32>,
    /// Per sub-body slot: the split body.
    slot_body: Tensor<u32>,
    /// Per sub-body slot: the constraint acting on it.
    slot_constraint: Tensor<u32>,
    /// `[hubs, slots, first sub-body slot index, slot capacity]`.
    pub(crate) counts: Tensor<u32>,
    /// Dispatch grid over the allocated sub-body slots.
    slots_indirect: Tensor<[u32; 3]>,
    /// Dispatch grid of the averaging kernel (one workgroup per hub).
    average_indirect: Tensor<[u32; 3]>,
}

impl HubState {
    /// Number of sub-body slots for a scene with `num_body_slots` body slots.
    fn pool(num_body_slots: usize) -> usize {
        num_body_slots.max(1024)
    }

    pub(crate) fn new(backend: &GpuBackend, num_body_slots: usize) -> Self {
        let storage = BufferUsages::STORAGE | BufferUsages::COPY_SRC;
        let indirect = BufferUsages::STORAGE | BufferUsages::INDIRECT;
        let pool = Self::pool(num_body_slots);
        let max_hubs = pool / HUB_MIN_CONSTRAINTS as usize + 1;
        Self {
            first_slot: Tensor::vector(backend, vec![NOT_A_HUB; num_body_slots], storage).unwrap(),
            list: Tensor::vector(backend, vec![0u32; max_hubs], storage).unwrap(),
            slot_body: Tensor::vector(backend, vec![0u32; pool], storage).unwrap(),
            slot_constraint: Tensor::vector(backend, vec![0u32; pool], storage).unwrap(),
            counts: Tensor::vector(
                backend,
                vec![0, 0, num_body_slots as u32, pool as u32],
                storage,
            )
            .unwrap(),
            slots_indirect: Tensor::scalar(backend, [0u32, 1, 1], indirect).unwrap(),
            average_indirect: Tensor::scalar(backend, [0u32, 1, 1], indirect).unwrap(),
        }
    }

    /// The solver velocity buffer: `vels` followed by the sub-body slots.
    pub(crate) fn solver_vels(
        backend: &GpuBackend,
        vels: &[Velocity],
        usage: BufferUsages,
    ) -> Tensor<Velocity> {
        let mut all = vels.to_vec();
        all.resize(vels.len() + Self::pool(vels.len()), Velocity::default());
        Tensor::vector(backend, &all, usage).unwrap()
    }
}

/// Kernels of the mass splitting.
#[derive(Shader)]
pub struct GpuMassSplitting {
    assign: GpuHubAssign,
    split_constraints: GpuHubSplitConstraints,
    update_effective_masses: GpuHubUpdateEffectiveMasses,
    scatter: GpuHubScatter,
    warmstart: GpuHubWarmstart,
    average: GpuHubAverage,
}

/// Inputs of [`GpuMassSplitting::split`].
pub struct SplitArgs<'a> {
    /// Number of body slots (all batches).
    pub num_body_slots: u32,
    /// Cumulative per-body constraint counts (each body's list end).
    pub body_constraint_counts: &'a Tensor<u32>,
    /// Per-body constraint lists.
    pub body_constraint_ids: &'a Tensor<u32>,
    /// The contact constraints of the step.
    pub constraints: &'a mut Tensor<TwoBodyConstraint>,
    /// World-space mass properties.
    pub mprops: &'a Tensor<WorldMassProperties>,
    /// Per-body graph-coloring group (multibody links are never split).
    pub body_group: &'a Tensor<u32>,
    /// The flat contact sweep bound.
    pub contact_plan: &'a Tensor<ContactPlan>,
    /// Dispatch grid over the contacts.
    pub contacts_len_indirect: &'a Tensor<[u32; 3]>,
    /// Shared per-batch indices.
    pub batch_indices: &'a Tensor<BatchIndices>,
}

impl GpuMassSplitting {
    /// Picks the hubs and points their constraints at the sub-bodies. Runs once per step,
    /// after the per-body constraint lists are built and before the coloring.
    pub fn split(
        &self,
        pass: &mut GpuPass,
        hubs: &mut HubState,
        args: SplitArgs<'_>,
    ) -> Result<(), GpuBackendError> {
        self.assign.call(
            pass,
            args.num_body_slots,
            args.body_constraint_counts,
            args.mprops,
            args.body_group,
            &mut hubs.first_slot,
            &mut hubs.list,
            &mut hubs.counts,
            args.batch_indices,
        )?;
        // Each constraint has two entries in the per-body lists; the kernel strides over them.
        self.split_constraints.call(
            pass,
            args.contacts_len_indirect,
            args.body_constraint_counts,
            args.body_constraint_ids,
            &hubs.first_slot,
            &mut *args.constraints,
            &mut hubs.slot_body,
            &mut hubs.slot_constraint,
            &hubs.counts,
            args.batch_indices,
        )?;
        self.update_effective_masses.call(
            pass,
            args.contacts_len_indirect,
            args.constraints,
            &hubs.counts,
            &mut hubs.slots_indirect,
            &mut hubs.average_indirect,
            args.contact_plan,
        )?;
        Ok(())
    }

    /// Copies each hub's velocity into its sub-bodies, before a contact sweep.
    pub fn scatter(
        &self,
        pass: &mut GpuPass,
        hubs: &HubState,
        solver_vels: &mut Tensor<Velocity>,
    ) -> Result<(), GpuBackendError> {
        self.scatter.call(
            pass,
            &hubs.slots_indirect,
            solver_vels,
            &hubs.slot_body,
            &hubs.counts,
        )
    }

    /// Applies each hub constraint's warmstart impulse to its sub-body.
    pub fn warmstart(
        &self,
        pass: &mut GpuPass,
        hubs: &HubState,
        solver_vels: &mut Tensor<Velocity>,
        constraints: &Tensor<TwoBodyConstraint>,
    ) -> Result<(), GpuBackendError> {
        self.warmstart.call(
            pass,
            &hubs.slots_indirect,
            solver_vels,
            constraints,
            &hubs.slot_constraint,
            &hubs.counts,
        )
    }

    /// Averages the sub-body velocities back into their hub, after a contact sweep.
    pub fn average(
        &self,
        pass: &mut GpuPass,
        hubs: &HubState,
        solver_vels: &mut Tensor<Velocity>,
        body_constraint_counts: &Tensor<u32>,
    ) -> Result<(), GpuBackendError> {
        self.average.call(
            pass,
            &hubs.average_indirect,
            solver_vels,
            &hubs.list,
            &hubs.first_slot,
            body_constraint_counts,
            &hubs.counts,
        )
    }
}
