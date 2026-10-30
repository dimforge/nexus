# Completion scope and outstanding work

The user requested continued implementation until the full port is complete and
tested. The initial scope is the **3D rigid-body simulation path, including ordinary
joints, collision detection, scene tests, and an offline Ascend handoff**. Articulated
multibodies, MPM and 2D are explicitly deferred. Never push; commit incrementally
in the isolated worktree. Do not stop at isolated operation milestones.

The port is not complete. Registered kernels passing does not establish completion
of the backend. `inventory.json`, feature contracts, classified `changes/` records,
and `parity.py report` describe coverage and exact-source validation. Historical
checkpoints under `evidence/` preserve earlier results; they do not validate later
source revisions or replace the final full-matrix reconciliation.

Implemented and connected:

- Body lifecycle, world inertia, collider transforms, gravity, integration and
  finalization; Coulomb contact preparation, matching/recycling, canonical ordering,
  adjacency, bounded Topo coloring, color buckets, cached warmstart and ordered TGS.
- Analytic and general collision, convex support/GJK/EPA/clipping, deferred
  mesh/polyline work and contact reduction; all native LBVH stages and resident
  build/refit/traversal. Ordinary joint motors, limits and colored solve.
- Fixture scenes through 120 frames and identical-state restart diagnostics.
  Native cold-start uploads preserve separate body/collider index spaces, multiple
  colliders, parentless geometry, native mass/materials and interleaved batches.
- Rust-owned C/Metal runtime with source-bound specialization, resident storage,
  capacity gates and lazy recompilation/migration before solver commit, unbounded
  stepping without trajectory allocation, and explicit typed readback.
- Native-sized hierarchical scans and an explicit contact-reduction option. Native
  scene oracles are independently compared with production `RbdPipeline::step`.
  Both reduced and unreduced overlapping-triangle cases are required.
- Validated live gravity, pose and velocity updates. Persistent-region migration
  includes changed gravity. Native production comparisons and C/Metal runtime
  integration tests are required for these controls.
- Native split-hub storage and frame scheduling, with exact ownership/scaled-mass
  validation and production comparisons for one/two 33-contact plates.
- Source/archive-bound checkpoints and reset preserve complete continuation state,
  including contact caches and gravity, with C/Metal control and scene regressions.

Also implemented: scene reconstruction for insertion/removal, mutable solver
parameters, model switching between Coulomb and compact twist friction, native
mesh/shape point queries and render-instance preparation. The public runtime now
checks contact palette and completion before committing and can grow both the
palette and deterministic round budget. Their current-source qualification remains
explicit in the evidence ledger.

Remaining qualification and audit:

1. Finish the 120-frame dense/shared-body runtime comparison and checkpoint
   continuation tests. Keep sleeping disabled, finite/unit states, exact counters,
   guards and independent physical bounds mandatory.
2. Complete the scheduling substitution audit and native fused/tail regression.
   Ordered launches preserve color dependencies; different warmstart accumulation
   orders must remain covered by native scene and restart contracts. Native CPU
   fused/tail kernels require cooperative dispatch as well as noalias disabled.
3. Refresh stale inventory descriptions against implemented active 3D paths.
   Distinguish executable helpers, ABI representations, namespace modules and
   user-deferred features; do not claim every Rust convenience API was ported merely
   because a simulation capability is covered.
4. Run the final complete C/Metal execution and Ascend source matrices, all required
   corruption/regression and public runtime tests, and immutable handoff/return
   checks. CPU CI has exhaustive registry shards and native runtime checks; remote
   CI has not run because no branch is pushed.
5. Reconcile every operation against the final source snapshot. Ascend vendor
   compilation, execution, synchronization and performance are not prerequisites
   for offline completion and must never be inferred from source lowering.

The CPU oracle requires `-Zmutable-noalias=no`; always use `cpu_oracle.py` and retain
its LLVM attribute probe. Default floating tolerances already permit roundoff.
The contact-sensitive triangle and joint-chain policies in `SCENE_NUMERICS.md` are
scoped by scene, frame and field, with required identical-state restart fixtures
and independent physical checks. Preserve every zero-distance stress case under
`BOUNDARY_NUMERICS.md`. Never fix a mismatch by deleting cases, changing Rust
arithmetic for convenience or widening global tolerances. Paired Rust/TileLang
bug fixes and intentional differences must remain distinguishable in `changes/`.
