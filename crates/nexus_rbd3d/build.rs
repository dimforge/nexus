use khal_builder::KhalBuilder;
use std::path::PathBuf;

fn main() {
    let output_dir = PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR not set by cargo"))
        .join("shaders-spirv");

    KhalBuilder::from_dependency("nexus_rbd_shaders3d", true)
        .feature("dim3")
        // Feature enabled unconditionally for the radix-sort device lost issue (see comment in the radix sort shader code).
        .feature("unsafe_remove_boundchecks")
        .build(output_dir);
}
