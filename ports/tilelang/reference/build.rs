fn between<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
    let offset = source
        .find(start)
        .expect("Native solver source marker changed");
    let tail = &source[offset..];
    &tail[..tail.find(end).expect("Native solver end marker changed")]
}
fn main() {
    println!("cargo:rerun-if-env-changed=CARGO_ENCODED_RUSTFLAGS");
    assert_eq!(
        std::env::var("CARGO_ENCODED_RUSTFLAGS").as_deref(),
        Ok("-Zmutable-noalias=no"),
        "The khal CPU oracle requires mutable-reference noalias disabled. Use python ports/tilelang/cpu_oracle.py <cargo subcommand> [arguments]."
    );
    let root = std::path::PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let path = root.join("../../../src_rbd_shaders/dynamics/solver_utils.rs");
    println!("cargo:rerun-if-changed={}", path.display());
    let source = std::fs::read_to_string(path).unwrap();
    // Copy the current native arithmetic, including point order and friction.
    // Only the CFM branch selector gains a diagnostic override. Marker changes
    // must be reviewed; a second handwritten solver cannot silently drift here.
    let helpers = between(
        &source,
        "/// Helper function for maybe inverse",
        "#[cfg(feature = \"dim3\")]\n/// Computes an orthonormal",
    );
    let loops = between(
        &source,
        "macro_rules! for_contact_point",
        "pub(crate) use for_contact_point;",
    );
    let solver = between(
        &source,
        "/// Contact geometry shared by all point rows",
        "#[cfg(all(test, not(target_arch_is_gpu)))]",
    );
    let signature = "        solve_friction: bool,\n    ) {";
    let branch = "let cfm = if dist <= 0.0 { cfm_factor } else { 1.0 };";
    assert_eq!(
        solver.matches(signature).count(),
        1,
        "Native solver signature changed"
    );
    assert_eq!(
        solver.matches(branch).count(),
        1,
        "Native CFM selector changed"
    );
    let solver=solver.replace(signature,"        solve_friction: bool,\n        force_soft: u32,\n        force_hard: u32,\n    ) {")
        .replace(branch,"let cfm = if force_soft & (1 << k) != 0 || (force_hard & (1 << k) == 0 && dist <= 0.0) { cfm_factor } else { 1.0 };");
    let prelude = r#"
use nexus_rbd_shaders3d::dynamics::{Velocity,RbdSimParams,ContactPoint,TwoBodyConstraint,SymInertia,MAX_CONSTRAINTS_PER_MANIFOLD,SUB_LEN};
use nexus_rbd_shaders3d::{Pose,Vector,gcross,gdot,utils::Slice};
use glamx::Vec2;
use khal_std::index::MaybeIndexUnchecked;
type SolverInertia = SymInertia;
"#;
    let generated = format!("{prelude}\n{helpers}\n{loops}\n{solver}");
    let output = std::path::PathBuf::from(std::env::var_os("OUT_DIR").unwrap());
    std::fs::write(output.join("boundary_solver.rs"), generated).unwrap();
}
