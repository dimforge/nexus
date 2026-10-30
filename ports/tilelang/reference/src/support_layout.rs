//! Native shape layouts plus explicit support/GJK diagnostic records.
#[allow(dead_code)]
#[path = "collision_layout.rs"]
mod collision_layout;
use serde_json::{Value, json};
use std::mem::{offset_of, size_of};
#[repr(C)]
struct SupportInput {
    direction: [f32; 3],
    max_distance: f32,
    exact: u32,
    padding: [u32; 3],
}
#[repr(C)]
struct SupportOutput {
    a: [f32; 3],
    b: [f32; 3],
    direction: [f32; 3],
    status: u32,
    padding: [u32; 6],
}
pub fn layout() -> Value {
    let mut l = collision_layout::layout();
    l.as_object_mut().unwrap().extend(
        crate::geometry_layout::layout()
            .as_object()
            .unwrap()
            .clone(),
    );
    l["support_input_words"] = json!(size_of::<SupportInput>() / 4);
    l["support_output_words"] = json!(size_of::<SupportOutput>() / 4);
    macro_rules! fields {($key:literal,$t:ty,$($f:ident),+)=>{l[$key]=json!({$(stringify!($f):offset_of!($t,$f)/4),+});};}
    fields!(
        "support_input",
        SupportInput,
        direction,
        max_distance,
        exact,
        padding
    );
    fields!(
        "support_output",
        SupportOutput,
        a,
        b,
        direction,
        status,
        padding
    );
    assert_eq!(size_of::<SupportInput>() / 4, 8);
    assert_eq!(size_of::<SupportOutput>() / 4, 16);
    l
}
