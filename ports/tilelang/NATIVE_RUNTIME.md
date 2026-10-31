# Native TileLang runtime integration

Enable `runtime` on the `nexus-tilelang-native` crate under `ports/tilelang/native`.
The default feature also enables Metal; use `--no-default-features --features runtime`
for generated C on a CPU-only host. The runtime feature includes native CPU scene
conversion and therefore requires the pinned noalias-disabled compiler setup:

```sh
python3 ports/tilelang/cpu_oracle.py --native check --locked --features runtime
python3 ports/tilelang/cpu_oracle.py --native check --locked --no-default-features --features runtime
```

Create the initial typed inputs with `RbdState::read_scene_upload` and choose
`TileLangBackend::C` or `TileLangBackend::Metal`. Both execute generated TileLang
kernels. C uses a persistent Python control process plus compiled native C++; Metal
uses the Rust/khal executor directly after source specialization.

```rust,ignore
use nexus_tilelang_native::runtime::{RuntimeOptions, TileLangBackend, TileLangSimulation};

let upload = nexus_state.read_scene_upload(&source_backend).await?;
let options = RuntimeOptions {
    backend: TileLangBackend::Metal,
    python: compiler_python,
    port_directory: tilelang_port_directory,
    workspace: new_artifact_directory,
    kernel_cache: None, // Some(cache_directory) opts into immutable compiler reuse.
    initial_pair_capacity: 64,
    max_pair_capacity: 1_048_576,
    trace_steps: None,
    contact_reduction: false,
};
// The compiler, generated executable code and artifact workspace are trusted.
let mut simulation = unsafe { TileLangSimulation::from_upload(&upload, options)? };
let progress = simulation.step()?;
let poses = simulation.body_poses()?;
let velocities = simulation.velocities()?;
```

The artifact directory must not exist. Its parent must exist. It retains the native
upload, generated variants, exact source/archive fingerprints and compiler records.
The caller selects a Python environment with the pinned compiler dependencies and
a C++ compiler for C execution. No Rust reference physics runs during specialization.
The native typed constructors supply initial memory; translated kernels perform
all subsequent dynamics.

`kernel_cache: Some(path)` enables a persistent cache of lowered modules and native
C/Metal libraries. Keys include full input IR, target, conservative source inputs,
compiler packages, host/environment and native toolchain identities. Entries have
hash-checked payloads and atomic publication; corruption causes a rebuild. Cache
paths contain trusted executable code. Ordinary parity runners and API examples
leave this disabled so their evidence still exercises compilation. Initial native
allocations, physics state and all scene values are always exported afresh.
`preparation_cache()` exposes hit/miss counts for the initial specialization.

Metal library caching uses `xcrun --sdk macosx metal` when the optional offline
Metal toolchain is installed, with fast math and contraction disabled. Otherwise
lowered MSL still comes from the cache and the native device compiles it normally.
The device creates pipeline objects and validates dispatch limits on each new
runtime. A reset retains the initial loaded program, even after capacity growth.
No cross-device pipeline objects or simulation storage are persisted.

Each `step` advances exactly one frame. On pair/deferred-work overflow or insufficient contact coloring, the
runtime asks the compiler for a larger variant and retries before solver commit.
It checks that the compiler sources and original scene archive still match the
session. Failed compilation or insufficient application capacity leaves the prior
physical state available. `set_pair_capacity_limit` permits retry after raising a
resource limit. `FrameProgress.retries` counts capacity migrations since the prior
successful frame, including migrations preceding an error returned to the caller.

`trace_steps: None` allocates no trajectory and permits ongoing stepping. Explicit
state reads are the only application body-state readbacks. Internal capacity gates
read counters. Typed poses and velocities preserve native interleaved body order;
collider indices remain a separate space. `Some(N)` records at most N frames and
then rejects further stepping.

The integration remains experimental. Cold-start uploads do not preserve existing
contact caches. Topo coloring, Coulomb and simplified twist friction, ordinary joints and LBVH
form the current composed path, including native hub splitting. Metal small scenes use fused solve/warmstart; other fused/tail variants use ordered colors;
see the dispatch substitution below.
No Ascend execution backend is claimed. `runtime_compile.py --target ascend`
prepares source only for a future CANN execution adapter.

Run the explicit integration tests with current native fixtures:

```sh
NEXUS_TILELANG_RUNTIME_PYTHON=/absolute/path/to/compiler/python \
NEXUS_TILELANG_RUNTIME_EXPECTED=/absolute/path/to/runtime_scene/mesh-contact-joint-f8 \
NEXUS_TILELANG_RUNTIME_EXPECTED_UNREDUCED=/absolute/path/to/runtime_scene/mesh-overlap-unreduced-f8 \
NEXUS_TILELANG_RUNTIME_WORK=/absolute/path/to/existing/test-output \
python3 ports/tilelang/cpu_oracle.py --native test --locked --features runtime \
  --test runtime -- --ignored --nocapture
```

Those tests build a real two-batch Nexus scene, start with insufficient capacity,
step through eight frames, compare every pose and velocity to the fresh Rust trace,
check exact static-body state and exercise a ninth untraced frame. They are ignored
in ordinary Cargo tests because they explicitly require the compiler environment
and fresh reference artifacts. The registry and all remaining source-bound checks
are still required for full qualification.

`contact_reduction` selects native per-pair manifold merging. Set it to `false` to
match the default Nexus pipeline, or `true` to merge composite contacts. It is
part of the immutable compilation configuration and survives capacity growth.
The imported palette supplies the initial bounded Topo budget. Before any physics
commit, device counters report the highest active color and any uncolored contact.
The runtime grows the palette or iteration budget and retries. Palette values above
63 fail explicitly; the application pair limit also bounds the maximum round budget.
An unresolved graph returns a capacity error and leaves published physical state
unchanged. Checkpoints retain pair, deferred-work, color and round capacities as
separate named limits.

`set_gravity`, `set_body_poses`, and `set_velocities` update live device inputs between
frames. Body ranges follow the same native interleaved order as readback. Updates
validate the complete range and values before writing; invalid input leaves state
unchanged. Gravity and body state migrate through capacity retries. Teleports keep
the previous contact cache, which the next native matching pass revalidates.
Use scene reconstruction below to change body/collider membership.

Native hubs use the ordinary body allocation followed by `max(native_body_slots, 1024)`
split velocity slots. Assignment and constraint splitting run before coloring;
scatter and cooperative averaging bracket each ordered contact pass. The ordinary
body and collider ID spaces remain separate. Tests compare slot ownership by body
and contact identity because atomic reservation order can differ across devices.
Mass scaling, writer uniqueness, guards, physical state and counters remain checked.

The hub runtime integration tests use `NEXUS_TILELANG_HUB_EXPECTED` pointing to the
fresh `runtime_scene/split-hubs-f8` fixture, plus the compiler/workspace variables
above. Run `--test runtime_hubs -- --ignored --nocapture`; both C and Metal start
with capacity one, grow on demand, and compare eight frames plus untraced stepping.

`checkpoint()` returns a serializable `RuntimeCheckpoint` with contact caches,
poses, velocities, gravity, trace state (if enabled), and the completed step count.
`restore_checkpoint` accepts only matching source, original-upload and configuration
identity; it can prepare larger capacity before restoring. Restoring is unsafe
because serialized cached indices are trusted shader inputs: only use unmodified
checkpoints with trusted provenance. Shape/capacity checks are not a validator for
arbitrary externally edited solver state. C and Metal use the same word-based ABI.

`reset()` replaces all state and scratch with the original upload and resets the
step counter. Already compiled growth variants remain available. Checkpoints and
reset perform explicit state transfers; normal stepping still runs generated
kernels without a host physics fallback. A cold-start upload alone does not contain
the caches required for continuation.

`solver_parameters()` returns the native parameter block, whose `dt` is the substep
duration. `set_solver_parameters` compiles a replacement configuration, checks that
all non-parameter scene inputs and compiler sources still match, then transfers
persistent state before publishing it. A friction-model change resets contact
history as described below; other updates retain all cached words. Errors leave the live physical state intact.
The convenience setters `set_timestep` (full-frame duration and substep count),
`set_num_internal_pgs_iterations`, `set_contact_merge_cos` and
`set_contact_recycle_distance` match the corresponding native controls. Changed
parameters require compilation, so use these as configuration operations. Repeating
an identical setting does no work. Reset uses the original uploaded bodies and the
latest parameter configuration. Serialized checkpoints retain their configuration
identity and cannot be restored after parameters change.

Run `--test runtime_configuration -- --ignored --nocapture` with the compiler and
workspace variables above plus `NEXUS_DISABLE_BF=1` to compare live configuration
changes against the production CPU pipeline.

`read_scene_upload()` exports current typed body state, current gravity and solver
parameters with the original native geometry, materials and ordinary joints. It is
a cold-start upload; use checkpoints to preserve contact caches.

`replace_scene(&new_upload, &[(old_body_slot, new_body_slot), ...])` supports
insertion/removal and changed collider or joint topology. Slots are global native
batch-interleaved body indices, independent of collider indices. Each surviving
body can be mapped once; other bodies use the new upload's initial state. The new
upload supplies gravity and solver parameters. A new configuration is compiled and
validated before publication, so errors leave the old simulation available. This
reconstructs the scene: contacts and colors start empty and the frame counter resets.
It corresponds to rebuilding a native `RbdState`, and does not reproduce the cache
behavior of native incremental `append_bodies`/`remove_bodies`. Use the returned
new body ordering for future state reads and edits.

The resident API accepts `FrictionModel::Simplified` as well as the default
`Coulomb`. Simplified contacts use the native compact twist tile: normal impulses
per point, one central sliding constraint and one angular constraint per manifold.
The compiler selects the corresponding tile layout and shader implementations.
Unknown native model values retain Nexus's Coulomb fallback.

`friction_model()` reads the current choice. `set_friction_model(model)` and a
model change through `set_solver_parameters(params)` compile a replacement
configuration before publishing it. They preserve poses, velocities, live gravity
and the absolute frame count. As in `RbdState::set_friction_model`, old contact
adjacency counts become zero and the model-dependent tile storage is reallocated;
this disables old contact warmstarts and color seeds. Other parameter updates
retain all persistent caches. Reapplying the same parameters is a no-op, and an
old-model checkpoint cannot be restored into a new-model simulation.

Model selection is a control operation with compilation cost. It is not intended
for switching every frame. `native/tests/runtime_friction.rs` compares twelve
frames of two batches of spinning/sliding boxes against the production CPU
pipeline, with switches in both directions and initial capacity growth. These
checks require the noalias-disabled CPU profile even when TileLang runs on Metal.
The registered `runtime_twist_scene` matrix additionally covers native joints,
multiple colliders, mesh reduction, split hubs and large batch storage.

## Portable dispatch substitution

The runtime dispatches colors in ascending order. Metal scenes with allocated
contact capacity at most 512 times the batch count use mirrored
`gpu_solve_constraints_fused` and `gpu_warmstart_constraints_fused`: one 64-lane
workgroup per batch, with a `mem_device` workgroup barrier between colors.
Per-contact arithmetic, bias modes and read-only static velocities use the same
helpers as separate launches. This heuristic limits serialized work; it differs
from Rust's latest pair-count estimate because these manifests know capacities.
Larger scenes and other targets retain wider, separate color dispatches. Cached
fused and tail variants are still substituted by separate colors.

Native fused kernels and the tail beginning at color 24 execute
the same per-contact arithmetic and ascending order within one workgroup, using
barriers between colors. For a valid coloring, contacts within one color have
disjoint writable velocity slots; static slots are read-only. Different batches
have disjoint state. Therefore replacing a fused/tail dispatch with ordered color
dispatches preserves dependencies without reproducing its launch-cost optimization.
The native `schedule-variants` regression compares every storage word after three
passes, warmstart and cached output for both models, with occupied colors
24/25/62/63, empty colors, multiple batches and widths exceeding a workgroup.

The frame uses ordered scatter warmstart. Native rigid-only production normally
uses per-body gathering of tagged cached deltas. Both primitives are separately
ported and tested; their accumulation order can differ in floating point. Complete
native scene comparisons and identical-state restart contracts qualify that
substitution. GPU synchronization and performance on Ascend still need device tests.

`native/tests/runtime_coloring.rs` exercises pair and palette growth, high colors,
120 native production frames with sleeping disabled, live gravity and fresh-runtime
checkpoint restoration. It uses the standard compiler/workspace variables and
`NEXUS_DISABLE_BF=1`; no pre-exported expected fixture is needed.

The dense fixture does not require a round-budget increase. Exact standalone
`color_requirements` cases cover unfinished coloring and saturating round demand;
C/Metal frame-control tests cover multiple gates at one stage and preserved state
when a later gate rejects an attempt. Bounded-coloring fixtures retain deliberate
round/palette exhaustion cases. These checks do not imply a public scene requiring
round growth was observed.


Preparation performance: a configured kernel cache also retains state-free prepared
programs. Exact layout/case/buffer dimensions and the transitive local Python
builder sources key those entries; full-IR/compiler identities key lowerings, and
native source/ABI/compiler flags key libraries. Evidence provenance remains tied
to the full source envelope independently of reusable compiled code. Worker
processes isolate TileLang compiler globals, and the default concurrency is up to
four (`NEXUS_TILELANG_COMPILE_JOBS=1..16`). Compilation flags, dispatch order and
numerical tolerances are unchanged. `preparation_timings()` exposes measured
compiler/export/runtime setup phases; `specialization.json` also retains per-kernel
lowering times. A program-cache hit always receives fresh native scene buffers.

Metal execution batches each consecutive range between capacity gates in one command buffer with ordered compute passes. Metal resource tracking preserves dependencies between dispatches; explicit GPU fences cover untracked resources, including repeated kernel iterations. The host waits at gate/readback boundaries; it does not submit or wait per shader. Native allocation sizes and devices still validate before execution, while immutable manifest and binding checks are retained from construction.


Contact canonicalization preserves subshape, collider B, collider A ordering.
Subshape keys remain full-width. Collider keys use the known scene collider-count
bound, rounded up to an even number of radix digits, reserving the all-ones key
for inactive contacts. This removes high zero-byte passes while leaving the final
permutation in the same buffer. Standalone fixtures without that bound keep the
full-width workspace contract. Optimized scratch is checked independently; output
contacts and stable permutations must match fresh Rust results exactly.

## Device scheduling and frame ownership

Metal runtime scenes publish live pair/contact/color grids on the device and use
indirect dispatch for independent element kernels. Compiled capacity grids remain
upper bounds, and cooperative radix/scan/refit launches retain their required
shape. Empty color ranges dispatch zero workgroups; empty key generation still
runs one group to publish a zero sort count. No dispatch-size readback is added.

Frame schema 2 adds disjoint, equal-size logical buffer swaps at successful commit.
The six link/adjacency/color/tile caches exchange owners instead of copying their
capacity. Checkpoints, growth, reset and reads resolve logical buffer names after
the exchange. Contact states retain their two-generation allocation ABI and copy
only live rows. A failed capacity check never exchanges owners. Schema 1 remains
supported. Other target schedules currently retain static launch/copy behavior.

Native indirect launch is not an Ascend requirement. The fallback design follows
`khal-derive/src/spirv_bindgen/cuda.rs::virtual_workgroup_loop`: launch a bounded
physical grid, read the virtual grid from device memory, iterate virtual groups
by the physical grid stride, and place a uniform workgroup barrier between
iterations. This preserves workgroup-local synchronization without CPU readback.
The TileLang CUDA/Ascend wrapper for this fallback still needs implementation and
device qualification; the Metal tests do not establish those targets' behavior.

Frame schema 2 also permits commit-stage conditions based on an already completed
capacity gate. Metal omits colors above the highest color returned by the existing
palette check. Conditions cannot precede a gate or add a new readback. Counts are
refreshed every frame, so a previously absent color becomes executable again.

Metal LBVH refit uses a workgroup-uniform active vote to stop completed levels,
matching native non-web Rust. The vote has three shared-memory barriers and the
refit retains its device-memory fence; no lane-local early exit bypasses a barrier.
