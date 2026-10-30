//! Native compact contact ABI overlay for arithmetic and scene oracles.
use nexus_rbd_shaders3d::dynamics::twist_tiles::TwistTile;
use serde_json::{Value, json};
use std::mem::{offset_of, size_of};
pub(crate) fn add_layout(l: &mut Value) {
    l["friction_model"] = json!("twist");
    l["tile_words"] = json!(size_of::<TwistTile>() / 4);
    l["warmstart"] = json!(offset_of!(TwistTile, warmstart) / 4);
    l["twist"] = json!({
        "offset_b": offset_of!(TwistTile, offset_b)/4,
        "center_a": offset_of!(TwistTile, center_a)/4,
        "friction": offset_of!(TwistTile, friction)/4,
        "frame_a": offset_of!(TwistTile, frame_a)/4,
        "frame_b": offset_of!(TwistTile, frame_b)/4,
        "normal_impulse": offset_of!(TwistTile, normal_impulse)/4,
        "dist": offset_of!(TwistTile, dist)/4,
        "normal_vel": offset_of!(TwistTile, normal_vel)/4,
        "radius": offset_of!(TwistTile, radius)/4,
        "arm_a": offset_of!(TwistTile, arm_a)/4
    });
    l["twist_arm_words"] =
        json!(size_of::<[nexus_rbd_shaders3d::dynamics::contact_tiles::VecScalar; 32]>() / 4);
}
