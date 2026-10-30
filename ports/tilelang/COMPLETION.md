# Completion scope and outstanding work

The user requested continued implementation until the full port is complete and
tested. The agreed initial scope is the **3D rigid-body simulation path, including
rigid-body joints, collision detection, scene tests, and an offline Ascend handoff**.
Articulated multibodies, MPM, and 2D are explicitly deferred. Never push; commit
incrementally in this worktree. This file records scope, not evidence of completion.
`inventory.json`, feature contracts and current `parity.py report` remain authoritative
for implemented coverage and exact-source validation.

The port is not complete. Do not stop after isolated operation milestones or mark
the complete backend ready because all currently registered kernel cases pass.

Remaining implementation sequence:

1. Body/collider state, Coulomb preparation and contact-link persistence stages
   are ported, along with cached warmstart scaling, body gathering
   and split-slot application. The analytic ordinary-body scene program now connects
   changing contact topology and persistence through 120-frame checkpoints. Connect
   this to production frame stepping, general collision and the remaining variants.
2. Ordered native dispatch, hierarchical prefix scan and composed contact
   adjacency, bounded Topo coloring, and color-bucket construction are implemented.
   Native Coulomb color ranges, cached unbiased solve and ordered contact iterations
   are implemented with device-loaded coefficients. Connect these to the general production
   frame sequence. A composed native TGS frame now handles fresh supplied
   manifolds and two fixed colors, with body initialization, gravity, contact
   preparation, substeps and finalization. Collision generation and changing topology
   are now connected in the general fixture scene controller. Production scheduling
   and hubs still need integration. Unbounded Luby scheduling remains.
   Palette/iteration exhaustion routes
   unresolved contacts to the Rust unsolved bucket (retained for warmstart, not solved);
   preserve and explicitly assess that limitation in scene tests. Hub assignment, splitting, dispatch sizes, scatter and averaging are ported
   in isolation. Implement actual color schedules and the contact solver variants,
   connecting shared static bodies and hubs to the production iteration order. Wire native generated kernels into Rust simulation stepping;
   the existing reference collision path may be retained only for an explicitly
   labeled intermediate hybrid milestone.
3. Native world bounds for every 3D shape tag and brute-force candidate-pair
   kernels are implemented, including an ordered bounds/pairs program. All twelve 3D LBVH stages and the resident Morton/radix/build/refit/traversal program are translated;
   production scene integration and overflow retry remain. The complete native
   analytic first pass now handles ball/primitive and box/box contacts, including
   material combination and positional inert-slot writes. Native segment/triangle/tetrahedron projection and Voronoi simplex transitions are now ported; Cold GJK and all seven convex support maps are also ported; Bounded EPA and its GJK composition are translated; polygonal support faces and generic 3D clipping are also implemented. Composed convex PFM manifolds are translated with exact conversion, independent witness checks and paired Rust/TileLang secondary-witness corrections. Deferred convex/mesh/polyline dispatch, work-list keys and contact reduction are now translated as isolated stages. Resident deferred ordering and reduction are now connected to fifteen general collision scenes with all-frame trajectory checks. Runtime overflow retry and production composition remain. PFM counter reset and contact-plan publication are implemented;
   a resident five-stage bounds/pairs/plan/manifold program now covers ball/box
   collision detection. Native radix stages and contact canonicalization are now
   implemented as resident programs. Canonicalization is connected to analytic scene solving. Generic awake fixture scene integration is implemented; public runtime integration remains.
   Port collision detection and shapes, contact creation/removal, broad-phase
   schedules, joint frame integration, queries, deterministic ordering and other required
   inventory capabilities. Do not introduce CPU physics fallbacks and call them
   completed TileLang features.
4. Ordinary rigid-joint stages (motor coefficients, COM-centered initialization,
   row construction, orthogonalization and colored solve) are ported. They are connected to analytic frames, with sixteen joint scenes and all-frame
   device trajectory checks. General runtime scene upload remains. Analytic awake ball/box scenes now cover free motion, impacts, stacks, friction,
   restitution, empty inputs and interleaved batches through connected steps and
   1/8/120-frame checkpoints with independent physical assertions. Ordinary joints and
   general collision have all-frame traces; extend analytic traces and broader scene
   coverage. Compare intermediate stages and short-run checkpoints to Rust;
   test long-run bounded penetration, energy/momentum behavior and stability.
   Keep sleeping disabled for benchmarks and stability comparisons.
5. Complete offline Ascend source coverage, immutable scene/kernel handoff bundles,
   return validation and runtime contracts. Ascend vendor binary compilation,
   execution, memory-model qualification and performance remain unavailable until
   a real device/toolkit is provided; they are not prerequisites for offline work.

The user explicitly accepts that floating-point outputs need not be bitwise equal.
Current float comparisons already use absolute/relative tolerances; integer fields,
identities, counters, copied bits and untouched storage remain exact. The retained
zero-distance contact case can select different soft-CFM branches on Rust CPU,
Rust Metal and TileLang due to cancellation. Keep the stress cases and classify
this using branch sensitivity plus scene-level physical/stability evidence.
A justified scoped numerical policy may accept harmless divergence without changing
the Rust physics. Do not clear it solely by increasing a global tolerance, deleting
cases, or declaring every floating-point mismatch harmless.

The two contact-sensitive 120-frame triangle and joint-chain fixtures now have
explicit per-case, per-field numeric policies backed by required identical-state
single-frame restart fixtures and independent geometry, effective-response and
all-frame physical checks. Their final registered matrix remains required after
source changes. The isolated zero-distance CFM stress case remains unresolved. A paired Rust/TileLang
clipping fix now rejects tangentially separated edge witnesses; its registered
geometry and scene matrix remains required. No global float tolerance or
physics arithmetic is changed by this classification.
