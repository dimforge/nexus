//! Explicit diagnostic wire records, separate from native Rust object layout.
use serde_json::{Value, json};
use std::mem::{offset_of, size_of};
#[repr(C)]
struct ProjectionInput {
    a: [f32; 4],
    b: [f32; 4],
    c: [f32; 4],
    d: [f32; 4],
    query: [f32; 4],
    solid: u32,
    padding: [u32; 3],
}
#[repr(C)]
struct ProjectionOutput {
    point: [f32; 3],
    inside: u32,
    barycentric: [f32; 3],
    feature: u32,
    index: u32,
    padding: [u32; 3],
    expanded: [f32; 3],
    guard: u32,
}
#[repr(C)]
struct SimplexState {
    vertices: [[[f32; 3]; 3]; 4],
    projection: [f32; 3],
    previous_projection: [f32; 3],
    dim: u32,
    previous_dim: u32,
    previous_vertices: [u32; 4],
    new_point: [[f32; 3]; 3],
    projected: [f32; 3],
    guard: [u32; 4],
}
pub fn layout() -> Value {
    let mut l = json!({"wire_format":"explicit_geometry_diagnostics_v1","projection_input_words":size_of::<ProjectionInput>()/4,"projection_output_words":size_of::<ProjectionOutput>()/4,"simplex_words":size_of::<SimplexState>()/4,"cso_words":9,"vector_words":3});
    macro_rules! fields {($key:literal,$t:ty,$($f:ident),+)=>{l[$key]=json!({$(stringify!($f):offset_of!($t,$f)/4),+});};}
    fields!(
        "projection_input",
        ProjectionInput,
        a,
        b,
        c,
        d,
        query,
        solid,
        padding
    );
    fields!(
        "projection_output",
        ProjectionOutput,
        point,
        inside,
        barycentric,
        feature,
        index,
        padding,
        expanded,
        guard
    );
    fields!(
        "simplex",
        SimplexState,
        vertices,
        projection,
        previous_projection,
        dim,
        previous_dim,
        previous_vertices,
        new_point,
        projected,
        guard
    );
    assert_eq!(size_of::<ProjectionInput>() / 4, 24);
    assert_eq!(size_of::<ProjectionOutput>() / 4, 16);
    assert_eq!(size_of::<SimplexState>() / 4, 64);
    l
}
