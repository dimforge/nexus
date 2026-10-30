# Contact-sensitive scene comparisons

The Rust CPU oracle runs with mutable-reference noalias disabled. Floating-point
comparisons use the hashed contracts, which retain exact checks for identities,
counts, topology, copied words and storage guards.

Two 120-frame fixtures amplify small solver differences through contact feedback:

| Fixture | Largest trajectory component difference | Final pose component difference |
| --- | ---: | ---: |
| Triangle tipping onto a box floor | 0.00624, angular velocity | 0.000104 |
| Three-body joint chain | 0.000814, velocity | 0.00000155 |

Both Rust and TileLang trajectories satisfy the unchanged independent energy,
penetration, quaternion, static-body and joint-geometry checks. Fresh Rust
single-frame snapshots isolate the feedback: eleven triangle restarts and eight
chain restarts produced identical numerical body and constraint results between
the tested C and Metal ports; legal adjacency allocation order differed.
Across these restarts, differences from Rust were at most 1.4e-7 in pose
components, 6.04e-5 in velocity components, 1.2e-7 in contact geometry, and 9.19e-5
in joint-row components. These figures describe the tested revisions; current
reports and their source hashes determine acceptance after further edits.

`contracts/general_scene.json` and `contracts/joint_scene.json` declare the exact
case configurations and field budgets. Other cases and unlisted fields retain the
default tolerance. Each accepted long-scene case requires a registered restart
operation, whose oracle rebuilds and runs the native trajectory before exporting
snapshots. Restart replay executes one complete target frame and preserves every
past and future trajectory word exactly.

The triangle's recycled tangent basis changes slightly along the divergent
trajectories. A nearly zero off-diagonal tangent coefficient can consequently
have a large component-relative difference. The scoped comparison uses the
Frobenius norm of each two-by-two tangent response matrix. Every matrix is also
independently reconstructed in f64 from its stored contact arms, directions,
inverse masses and inertia, using the strict default comparison. Orthonormal
contact bases and positive-semidefinite inertia are required. A matching error in
both reference and target coefficients therefore still fails validation.

Maintain the classifications in `changes/0058-contact-sensitive-scene-numerics.json`.
Changing a case, configuration, budget or independent check requires a new change
record and fresh evidence. The numerical policies cover these bounded unit-scale
regressions. They do not resolve the isolated zero-distance CFM stress suite, the
native secondary-witness clipping limitation, or Ascend hardware qualification.
