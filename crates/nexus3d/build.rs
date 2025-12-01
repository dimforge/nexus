#[cfg(not(feature = "comptime"))]
pub fn main() {}

#[cfg(feature = "comptime")]
pub fn main() {
    use slang_hal_build::ShaderCompiler;
    use std::env;

    const SLANG_SRC_DIR: include_dir::Dir<'_> =
        include_dir::include_dir!("$CARGO_MANIFEST_DIR/../../shaders");

    let out_dir = env::var("OUT_DIR").expect("Couldn't determine output directory.");
    let mut compiler = ShaderCompiler::new(vec![], &out_dir);
    compiler.add_dir(stensor::SLANG_SRC_DIR);
    compiler.add_dir(SLANG_SRC_DIR);

    // Compile all shaders from examples/shaders directory.
    // Note: slang-hal-build will automatically detect which backends to compile for
    // based on the cargo features enabled during the build.
    compiler
        .compile_shaders_dir("shaders", &[])
        .expect("Failed to compile shaders");
}
