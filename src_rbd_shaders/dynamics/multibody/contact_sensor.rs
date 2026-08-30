//! Contact "force sensor" readout for RL observations.

use super::types::{MB_CONTACT_KIND_NORMAL, MultibodyContactConstraint, MultibodyInfo};
use crate::utils::BatchIndices;
use khal_std::glamx::UVec3;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};

/// Maximum sensed links per multibody for the contact force-sensor readout.
pub const MAX_CONTACT_SENSORS: u32 = 4;

/// Per sensed link, sums the accumulated normal-constraint impulses. Dispatch
/// it once per step, after the last substep's stabilization sweep: the value is
/// then the step's total accumulated normal impulse (divide by the step `dt`
/// for an average force) when the constraints are built once per step, or the
/// last substep's impulse when they are rebuilt per substep.
///
/// Slots whose sensed link has no active normal rows read exactly 0.0: the
/// kernel zeroes its slots before accumulating, so no host-side clear pass is
/// needed and the dispatch is graph-capture safe.
///
/// `contact_sensor_links` holds `MAX_CONTACT_SENSORS` multibody link ids
/// (`u32::MAX` marks an unused slot); the same set is sensed for every
/// multibody in every batch. The output is interleaved like the other per-mb
/// buffers: `contact_sensor_out[batch_ids.mbi(batch, mb_idx) *
/// MAX_CONTACT_SENSORS + slot]`.
#[spirv_bindgen]
#[spirv(compute(threads(1)))]
pub fn gpu_mb_sense_contact_impulses(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    contact_constraints: &[MultibodyContactConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] contact_sensor_links: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] contact_sensor_out: &mut [f32],
    #[spirv(uniform, descriptor_set = 0, binding = 4)] batch_ids: &BatchIndices,
) {
    let batch_id = invocation_id.y;
    let mb_idx = invocation_id.x;
    let out_base = batch_ids.mbi(batch_id, mb_idx as usize) * (MAX_CONTACT_SENSORS as usize);
    for s in 0..MAX_CONTACT_SENSORS {
        contact_sensor_out.write(out_base + s as usize, 0.0);
    }
    if mb_idx >= batch_ids.multibodies_len {
        return;
    }

    let mb = multibody_info.read(batch_ids.mbi(batch_id, mb_idx as usize));
    let cons_base = mb.contact_constraint_start as usize;
    let count = mb.contact_constraint_count;

    for c in 0..count {
        let cons = contact_constraints.read(cons_base + c as usize);
        if cons.kind != MB_CONTACT_KIND_NORMAL {
            continue;
        }
        for s in 0..MAX_CONTACT_SENSORS {
            if contact_sensor_links.read(s as usize) == cons.link_id {
                let cur = contact_sensor_out.read(out_base + s as usize);
                contact_sensor_out.write(out_base + s as usize, cur + cons.impulse);
            }
        }
    }
}
