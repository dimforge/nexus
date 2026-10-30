//! Native layouts needed by the connected rigid-body scene oracle.
#[allow(dead_code)]
#[path = "collision_layout.rs"]
mod collision_layout;
use nexus_rbd_shaders3d::{Pose, dynamics::*, queries::IndexedManifold, utils::BatchIndices};
use serde_json::{Value, json};
use std::mem::{offset_of, size_of};
pub(crate) fn layout() -> Value {
    let mut l = collision_layout::layout();
    l["constants"] = json!({"max_flt_bits":nexus_rbd_shaders3d::MAX_FLT.to_bits(), "two_pi_bits":std::f32::consts::TAU.to_bits()});
    macro_rules! fields {($ty:ty,$($f:ident),+)=>{json!({$(stringify!($f):offset_of!($ty,$f)/4),+})}}
    l["hub"] = fields!(HubCounts, hubs, slots, base, pool);
    l["hub_min_constraints"] = json!(HUB_MIN_CONSTRAINTS);
    l["hub_average_workgroups"] = json!(HUB_AVERAGE_WORKGROUPS);
    l["not_a_hub"] = json!(NOT_A_HUB);
    l["rbd_params_words"] = json!(size_of::<RbdSimParams>() / 4);
    let additions = fields!(
        RbdSimParams,
        dt,
        contact_damping_ratio,
        contact_natural_frequency,
        static_contact_damping_ratio,
        static_contact_natural_frequency,
        normalized_max_corrective_velocity,
        length_unit
    );
    l["rbd_params"]
        .as_object_mut()
        .unwrap()
        .extend(additions.as_object().unwrap().clone());
    l["batch_words"] = json!(size_of::<BatchIndices>() / 4);
    let additions = fields!(
        BatchIndices,
        num_batches,
        bodies_len,
        solver_color_buckets_stride
    );
    l["batch"]
        .as_object_mut()
        .unwrap()
        .extend(additions.as_object().unwrap().clone());
    for (key, value) in [
        ("pose_padding", offset_of!(Pose, padding) / 4),
        ("local_words", size_of::<LocalMassProperties>() / 4),
        ("local_com", offset_of!(LocalMassProperties, com) / 4),
        (
            "local_inertia_frame",
            offset_of!(LocalMassProperties, inertia_ref_frame) / 4,
        ),
        (
            "local_inv_inertia",
            offset_of!(LocalMassProperties, inv_principal_inertia) / 4,
        ),
        (
            "local_inv_mass",
            offset_of!(LocalMassProperties, inv_mass) / 4,
        ),
        (
            "world_inertia",
            offset_of!(WorldMassProperties, inv_inertia) / 4,
        ),
        ("world_com", offset_of!(WorldMassProperties, com) / 4),
        (
            "world_padding0",
            offset_of!(WorldMassProperties, padding0) / 4,
        ),
        (
            "world_padding1",
            offset_of!(WorldMassProperties, padding1) / 4,
        ),
        ("world_words", size_of::<WorldMassProperties>() / 4),
        (
            "world_inv_mass",
            offset_of!(WorldMassProperties, inv_mass) / 4,
        ),
        ("hub_words", size_of::<HubCounts>() / 4),
        ("hub_hubs", offset_of!(HubCounts, hubs) / 4),
        ("hub_slots", offset_of!(HubCounts, slots) / 4),
    ] {
        l[key] = json!(value);
    }

    l["rbd_params"]["normalized_max_linear_velocity"] =
        json!(offset_of!(RbdSimParams, normalized_max_linear_velocity) / 4);
    l["rbd_params"]["num_solver_iterations"] =
        json!(offset_of!(RbdSimParams, num_solver_iterations) / 4);
    l["rbd_params"]["warmstart_coefficient"] =
        json!(offset_of!(RbdSimParams, warmstart_coefficient) / 4);
    l["constants"]["pi_over_four_bits"] = json!(std::f32::consts::FRAC_PI_4.to_bits());
    l["batch"]["colliders_batch_capacity"] =
        json!(offset_of!(BatchIndices, colliders_batch_capacity) / 4);
    l["rbd_params"]["num_internal_pgs_iterations"] =
        json!(offset_of!(RbdSimParams, num_internal_pgs_iterations) / 4);
    l["rbd_params"]["friction_in_bias_pass"] =
        json!(offset_of!(RbdSimParams, friction_in_bias_pass) / 4);
    l["params_words"] = json!(size_of::<RbdSimParams>() / 4);
    l["warmstart_coefficient"] = json!(offset_of!(RbdSimParams, warmstart_coefficient) / 4);
    macro_rules! add {($key:literal,$ty:ty,$($f:ident),+) => {$(l[$key][stringify!($f)]=json!(offset_of!($ty,$f)/4);)+};}
    add!(
        "link",
        ContactLink,
        warmstart_collider_a,
        warmstart_collider_b,
        warmstart_subshape,
        _padding
    );
    add!("contact", IndexedManifold, recycle_extent);
    l["state"] = json!({"pose_a":offset_of!(ContactRecycleState,pose_a)/4,
        "pose_b":offset_of!(ContactRecycleState,pose_b)/4,"colliders":offset_of!(ContactRecycleState,colliders)/4,
        "max_extent":offset_of!(ContactRecycleState,max_extent)/4,"max_drift":offset_of!(ContactRecycleState,max_drift)/4});
    l["params_words"] = json!(size_of::<RbdSimParams>() / 4);
    l["recycle_params"] = json!({"normalized_distance":offset_of!(RbdSimParams,normalized_contact_recycle_distance)/4,
        "length_unit":offset_of!(RbdSimParams,length_unit)/4});
    l["batch"]["deterministic"] = json!(offset_of!(BatchIndices, deterministic) / 4);
    l["batch"]["contact_sort_collider_shift"] =
        json!(offset_of!(BatchIndices, contact_sort_collider_shift) / 4);
    l["sort_uniforms_words"] =
        json!(size_of::<nexus_rbd_shaders3d::utils::radix_sort::SortUniforms>() / 4);
    l["sort_uniforms"] = json!({"shift":offset_of!(nexus_rbd_shaders3d::utils::radix_sort::SortUniforms,shift)/4,"max_keys_per_batch":offset_of!(nexus_rbd_shaders3d::utils::radix_sort::SortUniforms,max_keys_per_batch)/4,"has_aux":offset_of!(nexus_rbd_shaders3d::utils::radix_sort::SortUniforms,has_aux)/4});
    l["default_contact"] = json!(bytemuck::cast_slice::<_, u32>(
        &[IndexedManifold::default()]
    ));
    l["stats_words"] = json!(size_of::<ColorStats>() / 4);
    l["stats"] = json!({"sizes":offset_of!(ColorStats,sizes)/4,"max_bucket":offset_of!(ColorStats,max_bucket)/4,"highest_color":offset_of!(ColorStats,highest_color)/4});

    l
}
