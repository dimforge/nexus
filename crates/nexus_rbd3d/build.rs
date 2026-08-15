use khal_builder::KhalBuilder;
use std::path::PathBuf;

fn main() {
    let output_dir = PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR not set by cargo"))
        .join("shaders-spirv");

    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap();

    let mut builder = KhalBuilder::from_dependency("nexus_rbd_shaders3d", true)
        .feature("dim3")
        // Feature enabled unconditionally for the radix-sort device lost issue (see comment in the radix sort shader code).
        .feature("unsafe_remove_boundchecks");

    if target_arch == "wasm32" {
        builder = builder.feature("web-compat");
    }

    builder.build(output_dir);
}
