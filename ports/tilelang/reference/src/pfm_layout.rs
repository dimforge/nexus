//! Explicit PFM diagnostic offsets; shader physics uses the native shape/manifold ABI.
use serde_json::{Value, json};
use std::mem::{offset_of, size_of};
#[repr(C)]
struct PfmInput {
    prediction: f32,
    padding: [u32; 7],
}
#[repr(C)]
struct PfmRadius {
    thickness: f32,
    valid: u32,
}
pub fn layout() -> Value {
    let mut l = crate::feature_layout::layout();
    l["pfm_input_words"] = json!(size_of::<PfmInput>() / 4);
    l["pfm_input"] = json!({"prediction":offset_of!(PfmInput,prediction)/4,"padding":offset_of!(PfmInput,padding)/4});
    l["pfm_radius_words"] = json!(size_of::<PfmRadius>() / 4);
    l["pfm_radius"] = json!({"thickness":offset_of!(PfmRadius,thickness)/4,"valid":offset_of!(PfmRadius,valid)/4});
    assert_eq!(size_of::<PfmInput>() / 4, 8);
    assert_eq!(size_of::<PfmRadius>() / 4, 2);
    l
}
