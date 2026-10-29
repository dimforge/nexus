//! Shared native broad/narrow-phase layouts; every offset comes from live Rust.
#[allow(dead_code)]
#[path = "contacts.rs"]
pub(crate) mod contacts;
use nexus_rbd_shaders3d::{
    PaddedVector,
    bounding_volumes::Aabb,
    broad_phase::{CollisionPair, ContactPlan},
    dynamics::RbdSimParams,
    queries::{ColliderMaterial, IndexedManifold},
    rapier::geometry::InteractionGroups,
    shapes::*,
    utils::BatchIndices,
};
use serde_json::{Value, json};
use std::mem::{offset_of, size_of};
pub(crate) fn layout() -> Value {
    let mut l = contacts::layout();
    macro_rules! fields {($ty:ty,$($field:ident),+)=>{json!({$(stringify!($field):offset_of!($ty,$field)/4),+})};}
    l["contact"]["recycle_extent"] = json!(offset_of!(IndexedManifold, recycle_extent) / 4);
    l["plan_words"] = json!(size_of::<ContactPlan>() / 4);
    l["plan"] = fields!(ContactPlan, bound, pfm_base, pfm_len);
    l["material_words"] = json!(size_of::<ColliderMaterial>() / 4);
    l["material"] = fields!(
        ColliderMaterial,
        friction,
        restitution,
        friction_combine_rule,
        restitution_combine_rule
    );
    l["shape_words"] = json!(size_of::<Shape>() / 4);
    l["shape"] = fields!(Shape, a, b, c);
    l["shape_types"] = json!({"ball":SHAPE_TYPE_BALL,"cuboid":SHAPE_TYPE_CUBOID,"capsule":SHAPE_TYPE_CAPSULE,"cone":SHAPE_TYPE_CONE,"cylinder":SHAPE_TYPE_CYLINDER,"polyline":SHAPE_TYPE_POLYLINE,"trimesh":SHAPE_TYPE_TRIMESH,"convex_poly":SHAPE_TYPE_CONVEX_POLY,"triangle":SHAPE_TYPE_TRIANGLE});
    l["polyline_thickness_bits"] = json!(POLYLINE_THICKNESS.to_bits());
    l["aabb_words"] = json!(size_of::<Aabb>() / 4);
    l["aabb"] = fields!(Aabb, mins, maxs, padding0, padding1);
    l["vertex_words"] = json!(size_of::<PaddedVector>() / 4);
    l["vertex_vector"] = json!(offset_of!(PaddedVector, 0) / 4);
    l["group_words"] = json!(size_of::<InteractionGroups>() / 4);
    l["groups"] = fields!(InteractionGroups, memberships, filter, test_mode);
    l["pair_words"] = json!(size_of::<CollisionPair>() / 4);
    l["pair_colliders"] = json!(offset_of!(CollisionPair, colliders) / 4);
    l["batch_words"] = json!(size_of::<BatchIndices>() / 4);
    l["batch"] = fields!(
        BatchIndices,
        num_batches,
        colliders_batch_capacity,
        colliders_len,
        collision_pairs_capacity
    );
    l["rbd_params_words"] = json!(size_of::<RbdSimParams>() / 4);
    l["rbd_params"] = fields!(RbdSimParams, normalized_prediction_distance, length_unit);
    l["batch"]["contacts_capacity"] = json!(offset_of!(BatchIndices, contacts_capacity) / 4);
    l["batch"]["multibodies_batch_capacity"] =
        json!(offset_of!(BatchIndices, multibodies_batch_capacity) / 4);
    l
}
