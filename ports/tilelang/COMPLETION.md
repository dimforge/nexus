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
   are ported in isolation, along with cached warmstart scaling, body gathering
   and split-slot application. Connect these to changing scene topology and
   production frame stepping.
2. Implement actual color schedules, shared static bodies, hubs and the contact
   solver variants. Wire native generated kernels into Rust simulation stepping;
   the existing reference collision path may be retained only for an explicitly
   labeled intermediate hybrid milestone.
3. Port collision detection and shapes, contact creation/removal, broad-phase
   schedules, rigid-body joints, queries, deterministic ordering and other required
   inventory capabilities. Do not introduce CPU physics fallbacks and call them
   completed TileLang features.
4. Run free motion, impacts, stacks, friction, restitution and joints through full
   scene steps. Compare intermediate stages and short-run checkpoints to Rust;
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
