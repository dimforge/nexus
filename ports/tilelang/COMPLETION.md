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

Remaining implementation and qualification:

1. Complete scene lifecycle: insertion/removal or explicit reconstruction preserving
   the intended state, continuation/reset semantics, mutable solver parameters and
   remaining public backend integration. Cold-start upload is not a continuation
   checkpoint. Do not substitute host physics for a missing generated stage.
2. Broaden dense hub scene and lifecycle qualification. Complete coloring
   convergence/palette behavior and the remaining
   solver/friction variants. Audit production fused/tail paths against the portable
   ordered schedule; any intentional substitution needs a classified semantic
   difference and production comparison, not an unearned module-complete flag.
3. Complete required query/shape/render inventory capabilities and other missing
   active 3D shader behavior. Preserve the mirrored Rust module organization.
4. Add broader production scene and lifecycle coverage, including dense/shared-body
   contacts, hubs, configuration variants and long-run physical stability. Sleeping
   stays disabled in scene/benchmark comparisons. Verify guards, exact identities,
   counters, contact geometry, response/cache consistency and finite/unit states.
5. Complete the current full C/Metal execution matrix, Ascend source matrix,
   corruption/regression tests and immutable handoff/return protocol. Update the
   CPU-only CI workflow with native runtime checks and a scalable execution layout;
   the old single-job timeout is not qualified for the expanded matrix. Recorded
   Metal/Ascend protocol tests are conditional only when recordings are absent;
   explicitly configured missing/stale evidence must fail.
6. Reconcile the full registry against the final source snapshot. Ascend vendor
   compilation, execution, synchronization and performance remain unavailable until
   hardware/toolkit access exists; they are not prerequisites for completing the
   offline work and must never be inferred from source lowering.

The CPU oracle requires `-Zmutable-noalias=no`; always use `cpu_oracle.py` and retain
its LLVM attribute probe. Default floating tolerances already permit roundoff.
The contact-sensitive triangle and joint-chain policies in `SCENE_NUMERICS.md` are
scoped by scene, frame and field, with required identical-state restart fixtures
and independent physical checks. Preserve every zero-distance stress case under
`BOUNDARY_NUMERICS.md`. Never fix a mismatch by deleting cases, changing Rust
arithmetic for convenience or widening global tolerances. Paired Rust/TileLang
bug fixes and intentional differences must remain distinguishable in `changes/`.
