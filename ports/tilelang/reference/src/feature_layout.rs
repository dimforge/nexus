//! Explicit guarded feature diagnostics and native manifold field offsets.
use nexus_rbd_shaders3d::queries::{ContactManifold, ContactPoint};
use serde_json::{Value, json};
use std::mem::{offset_of, size_of};
#[repr(C)]
struct FeatureRecord {
    vertices: [[f32; 3]; 4],
    count: u32,
    padding: [u32; 3],
}
#[repr(C)]
struct FeatureInput {
    direction: [f32; 3],
    padding: [u32; 5],
}
#[repr(C)]
struct PolygonalInput {
    normal: [f32; 3],
    prediction: f32,
    flipped: u32,
    padding: [u32; 3],
}
#[repr(C)]
struct ManifoldOutput {
    value: ContactManifold,
    padding: [u32; 12],
}
pub fn layout() -> Value {
    let mut l = crate::support_layout::layout();
    macro_rules! fields {($key:literal,$t:ty,$($f:ident),+)=>{l[$key]=json!({$(stringify!($f):offset_of!($t,$f)/4),+});};}
    l["feature_words"] = json!(size_of::<FeatureRecord>() / 4);
    fields!("feature", FeatureRecord, vertices, count, padding);
    l["feature_input_words"] = json!(size_of::<FeatureInput>() / 4);
    fields!("feature_input", FeatureInput, direction, padding);
    l["polygonal_input_words"] = json!(size_of::<PolygonalInput>() / 4);
    fields!(
        "polygonal_input",
        PolygonalInput,
        normal,
        prediction,
        flipped,
        padding
    );
    l["feature_manifold_words"] = json!(size_of::<ManifoldOutput>() / 4);
    l["feature_manifold"] = json!({"points":(offset_of!(ManifoldOutput,value)+offset_of!(ContactManifold,points_a))/4,"normal":(offset_of!(ManifoldOutput,value)+offset_of!(ContactManifold,normal_a))/4,"count":(offset_of!(ManifoldOutput,value)+offset_of!(ContactManifold,len))/4,"padding":offset_of!(ManifoldOutput,padding)/4});
    l["feature_point_words"] = json!(size_of::<ContactPoint>() / 4);
    fields!("feature_point", ContactPoint, pt, dist);
    assert_eq!(size_of::<FeatureRecord>() / 4, 16);
    assert_eq!(size_of::<FeatureInput>() / 4, 8);
    assert_eq!(size_of::<PolygonalInput>() / 4, 8);
    assert_eq!(size_of::<ManifoldOutput>() / 4, 32);
    l
}
