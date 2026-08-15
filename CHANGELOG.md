## v0.5.0 (16 August 2026)

### Breaking changes

- ⚠ `NexusState::insert_rigid_body`/`insert_rigid_body_in` take an extra `RbdCoupling` argument.
  Pass `RbdCoupling::None` for a rigid-body-only scene.
- ⚠ Default solver parameters changed: `contact_damping_ratio` `5.0` → `10.0`,
  `normalized_allowed_linear_error` `0.001` → `0.005`, `normalized_max_corrective_velocity`
  `10.0` → `3.0`, `normalized_prediction_distance` `0.002` → `0.02`.
- ⚠ The viewer's `BackendType::Rapier` (CPU rapier reference backend) was removed. The remaining
  backends all run the nexus pipeline: `Gpu`, `Cpu`, `Cuda`, `Metal`.
- ⚠ Examples are prefixed by the subsystem they exercise (`boxes3` → `rbd_boxes3`). The
  `bench_joints3`, `bench_multibody_pendulum3` and `bench_urdf3` benchmarks were removed.
- ⚠ Python: the PyPI distribution is now `dimforge-nexus3d` (the import name stays `nexus3d`).
- Built against rapier `0.35` and parry `0.30` (was rapier `0.34`/parry `0.29`).

### Added

- **`nexus_mpm`: a GPU Material Point Method solver, in 2D and 3D**, behind the `mpm` feature.
  Particles are added and removed by chunk (`NexusState::add_particles`, `extend_chunk`,
  `remove_chunk`), on a sparse sorted grid with substepping and a CFL timestep bound.
- MPM constitutive models: linear and Neo-Hookean elasticity, Drucker-Prager sand (with cohesion),
  a weakly-compressible fluid, and the Stomakhin snow model, all built through `ParticleModel`.
- `RbdCoupling::MpmOneWay`: colliders act as moving boundaries for the particles, with per-body
  `stick`/`slip`/`separate`/`non-reflecting` conditions and optional CPIC for thin obstacles.
- `RbdCoupling::MpmTwoWay` hands a body over to MPM entirely: the rigid-body pipeline treats it as
  static while MPM integrates it from the particle impulses. At most 16 coupled bodies (CPIC limit).
- Multibody self-contacts: two links of the same multibody now collide, unless the multibody
  disables self-contacts.
- Restitution on multibody contacts, applied as an end-of-step pass seeded from the approach
  velocity measured at the start of the step.
- Per-link external forces/torques and gravity scale on multibody links
  (`GpuMultibodySet::set_link_external_wrench`), and DOF couplings between two joint axes.
- Multibody motors can be read back and retargeted at runtime (`GpuMultibodySet::set_motor`,
  `set_motors`, `motor`), plus `set_num_internal_pgs_iterations` and `set_implicit_coriolis`.
- `RbdSimParams::static_contact_natural_frequency`/`static_contact_damping_ratio`: contacts
  touching a fixed body get their own, stiffer by default, softness coefficients.
- `RbdSimParams::normalized_max_linear_velocity` (default `400.0` m/s) caps the linear velocity
  after each substep so speculative contacts stay reliable. Set to `f32::MAX` to disable.
- A brute-force O(n²) broad-phase, used instead of the LBVH for environments with at most
  64 colliders.
- `NexusState::rbd_world_mut_untracked`: mutate a rapier world after `finalize` without marking
  the GPU state dirty.
- Viewer: `snap_rgb` frame capture, configurable resolution, a headless mode, a vsync toggle,
  pipelined capture (`render_async`/`render_flush`) and kiss3d's GPU path tracer
  (`raytrace_frame`), all exposed to Python (PRs #7, #8 and #11 by @haixuanTao).
- Python bindings for MPM (`set_mpm_params`, `add_particles`, `ParticleModel`, …).
- A `web-compat` feature on the shader crates, enabled automatically when targeting `wasm32`.

### Modified

- The contact solver follows rapier's TGS-soft relax pass: the unbiased normal rhs is refreshed
  from the post-integration poses, instead of stripping CFM and bias from the constraints in place.
- `NexusState::set_rbd_gravity` applies to free rigid-bodies and multibody links alike, and works
  in 2D (where the third component is ignored). It used to be 3D- and multibody-only.
- Extensive, mostly result-identical pipeline optimizations: frame-to-frame coloring, contacts
  bucket-sorted by color, fused colored sweeps, a shared-memory multibody PGS sweep, an SoA link
  workspace, LBVH subtree pruning, and skipping the pipelines that are provably inert.

### Fixed

- `RbdState::from_rapier` zero-filled the velocity buffer, dropping every body's initial linear
  and angular velocity (PR #10 by @haixuanTao).
- Multibody joint limits no longer emit a constraint row while the joint sits strictly inside its
  bounds, where it can never apply an impulse (PR #14 by @haixuanTao).
- Contact manifold reduction now matches rapier's, including its degenerate-selection guards.
- The fused multibody solver kernels no longer place barriers under non-uniform control flow,
  which WebGPU rejects; the loop bound is now a uniform holding the max over all multibodies.
- Out-of-bounds writes to the polygonal-feature pair buffer, and constraint counting over the
  padded capacity instead of the real contact count.
- The LBVH pair traversal used a `while` loop, which naga might miscompile on macos.

## v0.4.0 (04 July 2026)

Complete rewrite. Nexus is now a full GPU physics engine written in
[rust-gpu](https://github.com/Rust-GPU/rust-gpu), with everything from the broad-phase to the
constraint solver running on the device.

### Breaking changes

- ⚠ Shaders are written in Rust and compiled to SPIR-V with rust-gpu, replacing Slang.
  `slang-hal`/`stensor` are replaced by [khal](https://crates.io/crates/khal)/
  [vortx](https://crates.io/crates/vortx), and the `comptime`/`runtime` features are gone.
- ⚠ Backends are selected by the `webgpu` (default), `metal`, `cpu`, `cpu-parallel` and `cuda`
  features. Shader-facing math moved from `nalgebra` to [glamx](https://crates.io/crates/glamx).
- ⚠ `nexus2d`/`nexus3d` are now umbrella crates over `nexus_rbd2d`/`nexus_rbd3d` (behind the `rbd`
  feature) plus `NexusState`/`NexusPipeline`. The old `dynamics::{BodyDesc, GpuBodySet, …}` API and
  the `BodyCoupling`/`BodyCouplingEntry` types were removed.

### Added

- A GPU broad-phase: parallel LBVH construction over Morton codes (with a GPU radix sort and
  prefix sum) and a bounded, stackless pair traversal.
- A GPU narrow-phase: analytic contacts for primitive pairs, GJK/EPA with SAT-based feature
  clipping otherwise. Balls, cuboids, capsules, cones, cylinders, convex shapes, polylines,
  trimeshes.
- A rigid-body solver: TGS-soft contacts with graph-colored Gauss-Seidel sweeps, cross-frame
  warmstarting, Coulomb friction and speculative contacts, tuned by `RbdSimParams`.
- Impulse joints: ball, fixed, prismatic and revolute, with limits and motors.
- A reduced-coordinates multibody solver (3D): articulated-body dynamics with a per-multibody mass
  matrix and LU solve, joint limits/motors, and loop-closing impulse joints.
- `NexusState`/`NexusPipeline`: one rapier world per *environment*, baked into GPU buffers on
  `finalize` and stepped in parallel. Batched environments make nexus usable as an RL simulator.
- Incremental insertion and removal of rigid-bodies, plus capacity reservation and a resize policy
  for the collision buffers.
- A cross-platform viewer (`nexus_viewer2d`/`nexus_viewer3d`) built on kiss3d, with a demo picker,
  a backend selector, per-kernel GPU timings, and a full set of 2D and 3D demos.
- URDF and MJCF robot loading, including the MuJoCo Menagerie models.
- Python bindings for the 3D engine and viewer (`crates/nexus_python3d`), published on PyPI.
- A [website](https://nexus.dimforge.com) with the demos compiled to WebAssembly.

## v0.3.0 (20 January 2026)

### Added

- `comptime` and `runtime` features to select whether the Slang shaders are compiled by the
  crate's `build.rs` or at runtime.
- Backend selection features: `webgpu`, `vulkan`, `metal`, `cpu` and `cuda`.

### Modified

- Update to `slang-hal`/`stensor` 0.3 and rapier 0.31.

## v0.2.1 (27 October 2025)

### Fixed

- Fix the 2D build with slang-compiler 2025.19.1: the angular inertia is a scalar in 2D, so
  applying an impulse or integrating forces must not go through `mul`.

## v0.2.0 (27 October 2025)

### Modified

- Update to wgpu 27, `slang-hal`/`stensor` 0.2 and rapier 0.30.
- The crate is now fully documented (`#![warn(missing_docs)]`), and `BodyCoupling`/
  `BodyCouplingEntry` are re-exported from `nexus::dynamics`.

## v0.1.0 (20 September 2025)

Initial release: GPU rigid-body state (poses, velocities, forces, mass-properties) and Slang
shaders for shapes, geometric queries (ray-casting, point projection, contacts) and force/velocity
integration, with conversion from a rapier `RigidBodySet`/`ColliderSet`.
