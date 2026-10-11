## v0.7.0 (11 October 2026)

The Python package `dimforge-nexus3d` is released as 0.3.0 alongside the crates.

### Breaking changes

- ⚠ `RbdSimParams` gains `normalized_contact_recycle_distance` and, in 3D, `friction_model`; its
  `_padding*` fields are gone (2D keeps `_padding0`). Struct literals need `..Default::default()`.
- ⚠ `RbdState::rigid_contact_constraints` returns `&ContactConstraints` (contact links, constraint
  indices and the friction model's tiles) instead of `&Tensor<TwoBodyConstraint>`. Read contact
  impulses with the new `RbdState::read_rigid_contact_impulses`.
- ⚠ `GpuMultibodySet::links_static` returns `&Tensor<UVec4>`: static link records are interleaved
  across environments per 16-byte quad, quad `q` of link `l` of env `e` at
  `(l * LS_QUADS + q) * num_batches + e`. Decode it with `ls_soa_to_structs`. The
  `links_workspace` buffer keeps its type but uses the same per-quad interleaving
  (`ws_soa_to_structs` still decodes it).
- ⚠ Renamed `RbdPipeline::fused_color_sweeps` and `RbdGraphKey::fused_color_sweeps` to
  `fused_color_dispatches`, and `mb_sweep_indirect` to `mb_dispatch_indirect`.
- ⚠ Removed `WarmstartArgs`, `GpuWarmstart::transfer_warmstart_impulses` and
  `GpuColoring::dispatch_topo_gc`. `SolverArgs`, `ColoringArgs` and `ColorBucketsArgs` changed.
- ⚠ The rbd shader crates only compile their kernels with the new `entry-points` feature (on by
  default): crates depending on them with `default-features = false` must enable it to get the
  kernels.
- Shader crates: `TwoBodyConstraint` has a compact layout (its builder, infos, element and
  normal/tangent part types are removed), the multibody Delassus kernels and
  `MAX_MB_CONTACTS_PER_MB`/`MAX_MB_CONTACT_CONSTRAINTS_PER_MB` are removed,
  `MultibodyLinkStatic::_pad0` is now `ancestor_dofs` (3D), `IndexedManifold`'s padding is
  `recycle_extent`, `LbvhNode` has a new layout, and many kernel signatures changed.
- Viewer: `DemoKind` gains `Stress`, `SimSettings` gains `rbd_substeps` and (3D)
  `rbd_friction_model`, and `UiState` gains `wait_for_render`. Sensor rendering
  (`render_sensor_rgb`/`depth`/`segmentation`) is not available on wasm32.

### Added

- Rapier's simplified friction model (3D): `FrictionModel::Simplified` solves one sliding
  friction constraint at the manifold center plus one twist constraint, instead of friction at
  every point (`FrictionModel::Coulomb`, the default). Multibody contacts always use Coulomb. Set
  with `RbdSimParams::friction_model`, `NexusState::set_rbd_friction_model` or
  `RbdState::set_friction_model`; switching keeps the bodies but resets the contact warmstarts.
- Contact recycling: a pair whose relative pose drifted less than
  `RbdSimParams::normalized_contact_recycle_distance` (times `length_unit`) since its contacts
  were computed keeps them instead of running the narrow phase again. Also
  `RbdSimParams::contact_recycle_distance` and `RbdState::set_contact_recycle_distance`.
- `NexusState::set_rbd_substeps`/`rbd_substeps` and `RbdState::set_timestep` change the substep
  count of a running simulation without resetting it.
- `RbdState::read_rigid_contact_impulses` returns the last step's contact impulses as
  `ContactImpulseSnapshot`s.
- `NEXUS_NO_FUSED_SUBSTEPS` disables the fused multibody substep kernel (see below).
- Static link layout helpers: `LinkStatics`, `LinkStaticRef`, `ls_soa_from_structs`,
  `ls_soa_to_structs`, `LS_QUADS` and `LS_WORDS`.
- Viewer: substeps slider, 3D friction-model selector, "Wait for render before stepping"
  checkbox, and `NexusViewer::with_wait_for_render`, `with_rbd_friction_model` (3D) and
  `with_snapshot` (saves a frame and quits, native only).
- Demos: `--headless` and `--snapshot <frame> <path>` flags, and a "Rigid-bodies - stress tests"
  group. 3D: brick rings (28k, 110k), brick walls (27k), pyramids (500k), ragdolls on cloth (24k),
  box piles (4k, 32k), jointed drop (34k), box columns (100k). 2D: pyramids (1.3k, 5k, 20k), box
  rain (1k, 5k, 90k), joint lattices (4k, 100k, 262k), wrecking balls (4k, 40k).
- A user guide on nexus.dimforge.com, for Rust (2D and 3D) and Python, whose code snippets are
  compiled and run by CI.
- Python: `set_rbd_solver_params(friction_model="coulomb" | "simplified")` (an unknown name raises
  `ValueError`), a `"friction_model"` key in `rbd_solver_params()`, and a `twist_impulse` key in
  `debug_rigid_contact_impulses()`.
- Python (already in 0.2.1): `NexusViewer.set_deterministic`/`deterministic` (applied to the
  state by `sync`), and `NexusState.set_deterministic(viewer, enabled)`, `deterministic` and
  `steps` for states stepped without a viewer sync.
- Python (already in 0.2.1): `NexusViewer.with_backend(name)` selects `"webgpu"` (the default),
  `"cpu"`, `"metal"` or `"cuda"`, raising `ValueError` for a backend the wheel was built without;
  `NexusViewer.backend_name()` and `nexus3d.available_backends()` report the selection and what
  the wheel supports.
- Python wheels (already in 0.2.1): the macOS wheel is built with the `metal` feature (native
  Metal, alongside WebGPU), and Linux aarch64 wheels are published.

### Fixed

- Contacts against a fixed body were not warmstarted when the fixed body came second in the pair.
- The graph coloring could give a constraint a color already used by a neighbor, so coloring
  never converged and the color budget kept growing.
- Multibodies with up to 8 DOFs had wrong dynamics with 1024 or more multibodies: joint damping,
  armature, stiffness, kinematic DOFs and implicit Coriolis terms were ignored on all DOFs but
  the first.
- WebGPU: large batched scenes failed validation once the collision-pair buffers exceeded the
  storage binding size limit. The pair capacity is now capped by that limit.
- WebGPU: reading multibody state back no longer panics with a wgpu validation error. The
  multibody DoF state, static link data and (once joint friction is set) multibody info buffers
  lacked `COPY_SRC`, so `NexusState::multibody_joint_velocities` (Python: `robot_qvel`),
  `RbdState::snapshot` and `GpuMultibodySet::debug_cons_layout` failed. Native Metal ignores
  buffer usages, which hid the bug.
- `RbdState::rigid_contact_constraints` and Python's `debug_rigid_contact_impulses` returned the
  contacts of the step before last.
- States built with `reserve_rigid_bodies` ignored the scene's timestep and substep count.
- Large scenes ran their first steps on the slow single-workgroup solver path.
- `nexus_viewer3d` did not build for wasm32.
- The topological graph coloring did not build with cuda-oxide.
- The demos now release the GPU memory of a demo when switching to the next one on CUDA.

### Modified

- Contact recycling is on by default (`0.05` times `length_unit`; `0` disables it), so results
  differ from v0.6. Trimesh and polyline sub-shape contacts and multibody contacts are never
  recycled. It stays active in deterministic mode, which remains reproducible.
- Dynamic rigid bodies with more than 32 contact constraints are split into one sub-body per
  contact, solved independently then averaged. Heavy bodies resting on many others converge
  differently and no longer inflate the number of solver colors. Multibody links are not split.
- Adaptive resize policies, including the default `Grow`, now also shrink the solver color budget
  once the coloring converges, and large dense scenes (4096 colliders or more) start with 32
  colors (64 from 65536 colliders). Deterministic mode keeps a fixed budget.
- Native Metal runs small multibodies (up to 32 DOFs) with dedicated SIMD-group kernels. With one
  such multibody per environment, explicit Coriolis terms, no substep refresh, at most 32 links
  and no rigid-body contacts or impulse joints, the whole substep loop runs in a single dispatch.
  Results can differ from the other backends by floating-point rounding.
- `NexusViewer::sync` also applies the viewer's substep count and (3D) friction model to the
  state after the scene's first sync. `NexusState::add_environment` copies environment 0's
  friction model.
- The 3D demos run with the simplified friction model, and the demos wait for rendering before
  stepping so the GPU timings exclude it.
- Performance: the LBVH is refit in parallel chunks and traversed without a stack, contact
  constraints are compact, color-ordered tiles solved by specialized kernels, the coloring sizes
  its dispatches from the previous step, and multibody batches use per-quad interleaved layouts,
  right-sized workgroups and fused kernels, which mostly speeds up many environments with one
  robot each.
- The rbd kernels are no longer compiled into the MPM shaders, which cuts their build time and
  memory and shrinks the binaries.

## v0.6.0 (4 October 2026)

### Breaking changes

- ⚠ Per-body, per-collider and joint GPU buffers are batch-interleaved: entity `i` of env `e` is at
  `i * num_batches + e` (was `e * stride + i`). This affects `RbdState::body_poses` and the other
  per-body tensors, and `GpuRigidBodyRef::gpu_id`.
- ⚠ Collision pairs and contacts are one flat buffer shared by all envs: `collision_pairs_len` and
  `NexusCounts::collision_pairs` report the total, `RbdCapacities::collisions_capacity` is a
  per-env hint, and `RbdState::contacts_len` is removed.
- ⚠ `RbdSimParams` is a single uniform shared by all envs (finalizing panics if envs differ). New
  fields: `contact_merge_cos`, `num_internal_pgs_iterations`, `friction_in_bias_pass`.
- ⚠ The contact prediction distance is `RbdSimParams::prediction_distance()` instead of a
  hard-coded `0.02`.
- ⚠ `GpuMultibodySet::set_num_internal_pgs_iterations` no longer drives the solver; use
  `NexusState::set_rbd_num_internal_pgs_iterations` (or `RbdSimParams::num_internal_pgs_iterations`).
- ⚠ `GpuMultibodySet`: `dof_values`, `contact_constraints_per_batch`, `contact_constraint_jacs` and
  `contact_constraint_columns` are removed (see `contact_jac_cols`); `from_rapier` takes a
  contact-slot count.
- ⚠ Viewer: `UiSections` is replaced by the `UiSection` enum (`UiState::ui_section`); one panel tab
  is open at a time.
- ⚠ MPM shaders: `NBH_SHIFTS` is now a plain array; use `nbh_shift(i)` for a vector.
- Many low-level kernel argument structs and dispatch signatures changed (`SolverArgs`,
  `ColoringArgs`, `GpuNarrowPhase::dispatch`, `BatchIndices`, ...).
- Built against khal `0.4`, vortx `0.5`, kiss3d `0.47`, rapier `0.36` and parry `0.31`.

### Added

- `NexusState::set_deterministic`: two runs of the same scene give identical results on the same
  machine and build. Costs ~10-17% per step, up to ~45% with trimeshes or polylines, and fixes the
  rigid-body resize policies. Testbed: "Deterministic" checkbox, `--deterministic` flag and a step
  counter (`NexusState::steps`).
- `cuda-oxide` feature (CUDA kernels compiled by cuda-oxide), and compute graphs:
  `NexusState::set_compute_graphs_enabled` records the step once and replays it as a CUDA graph
  (testbed: `--compute-graphs`). Both need the GitHub version of khal/vortx.
- Viewer debug renderer ("Debug render" tab): collider shapes, AABBs, joints, contacts, LBVH
  nodes, MPM particles and grid. Readbacks: `RbdState::debug_contacts`, `debug_lbvh`,
  `MpmState::debug_particles`, `debug_grid`.
- Viewer sensor cameras (3D): `add_sensor_camera`, `attach_sensor_camera`, `render_sensor_rgb`,
  `render_sensor_depth`, `render_sensor_segmentation`, with MSAA, shadow and ambient controls.
- Viewer: `set_ambient`, `add_directional_light`, `set_body_color`, `set_body_casts_shadows`,
  textured visual meshes, `set_shadow_softness`/`range`/`resolution`.
- Env reset (3D): `RbdState::snapshot`, `reset_env_from_snapshot`, and the batched
  `publish_reset_templates`/`reset_envs_from_templates`.
- Multibodies: joint dry friction (MJCF `frictionloss`), per-link contact sensors
  (`set_contact_sensor_links`), actuator delay, motor targets read from a GPU tensor
  (`scatter_motor_targets`), and `set_substep_refresh`.
- `NexusState`: `control_multibody_motors`, `read_multibody_links`,
  `multibody_joint_positions`/`set_multibody_joint_positions`/`multibody_joint_velocities`,
  `read_rigid_body_poses`/`velocities` and `set_rigid_body_pose`/`velocity` (3D),
  `set_rbd_timestep`, `set_rbd_collisions_capacity`, `set_rbd_mb_contact_constraints_capacity`,
  `set_rbd_implicit_coriolis`, `set_rbd_substep_refresh`.
- `RbdSimParams::friction_in_bias_pass`: also solve friction during the biased pass (off by
  default), which keeps pinch grasps and resting stacks from drifting.
- Optional contact reduction to at most 4 points per collider pair (`RbdPipeline::contact_reduction`).
- `RbdPipeline::step_encoded` records a step into a caller-owned encoder.
- Python: `Robot` (URDF/MJCF loading, `robot_state`/`robot_qvel`, `set_robot_targets`, FK/IK, PD
  gains), MJCF actuators (`apply_actuator_controls`), body pose/velocity read/write,
  `set_rbd_solver_params`, sensor cameras, contact debug readbacks, `set_compute_graphs_enabled`.

### Fixed

- Bodies resting on trimeshes were ejected (unvalidated seeded colors, EPA picking a triangle back
  face, duplicate manifold points).
- Warmstarting matched contacts by body pair and to the first old point within 10 cm; it now
  matches by collider pair and sub-shape, to the nearest point. Also fixes polyline warmstart.
- Contacts between a multibody link and a rigid body were solved twice, cancelling a robot's grip.
- Multibody contacts beyond 128 points per multibody were silently dropped.
- The narrow phase used env 0's collider parents in every env.
- Polyline AABBs ignored the segment thickness.
- `GpuMultibodySet::read_dof_coords` returned wrong locked axes for envs other than 0.
- Buffer auto-resize could race in-flight steps on Metal and leave stale warmstart ranges.
- The crates did not build without the `mpm` feature.
- `RbdState::debug_constraint_colors` now reads the constraints and colors of the last step.

### Modified

- Performance: batched kernels run as flat dispatches over the interleaved layout, pair buffers
  are sized by total demand instead of the busiest env, and multibody contacts are indexed per
  multibody.
- Multibody contact friction is velocity-level only, and CFM softening applies only to normal rows.
- `num_internal_pgs_iterations` also drives the rigid-body sweeps of the biased pass, interleaved
  with the multibody sweeps.
- Viewer defaults: 4096 shadow map over 4 atlas layers, first cascade at 3 m, mipmapped textures
  with 16x anisotropic filtering. The viewer no longer steps before the first sync of a scene.

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
