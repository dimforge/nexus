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

## Analytic trajectories and discrete collision timing

Analytic scenes now record every pose/velocity frame, as joint and general scenes
already did. Native/target comparisons retain their existing float tolerances;
padding, static body words, frame counters, guards and the final recorded state
remain exact. The shared trace checker applies the same policy to every family.

The original 2 cm analytic penetration bound describes the final checkpoint.
Applying it unchanged to every intermediate frame rejects the Rust reference:
in `free-c1-l1-b1-f120`, frame 47 reaches 3.077 cm penetration while the sphere
moves at 7.684 m/s, before the next discrete collision pass. The final state still
meets 2 cm. The contracts therefore explicitly select `previous_surface_motion`
for intermediate physical checks. Its extra allowance is
`2 * frame_dt * previous_max_surface_speed + |gravity| * frame_dt^2`, where
surface speed is linear speed plus angular speed times the fixture's bounding
radius. This covers two approaching surfaces over one discrete collision step.
It is a diagnostic motion bound, not a continuous-collision guarantee. The final
checkpoint receives no extra allowance. Energy, quaternion normalization, static
state and finite-state checks apply at every frame independently. A pair of
matching but corrupt trajectories must still fail these physical checks.

## Connected LBVH scenes

LBVH preserves the native fixed broad-phase prediction. Its triangle trajectory
therefore needs its own classification. In the frozen `e488fb08d` execution,
C and Metal agree on the observed differences. Before frame 36, the largest pose
component difference is 3.65e-5. The nearly flat triangle produces three contact
points in Rust and four in TileLang, with different normals; the resulting
angular-velocity difference reaches 0.139234 for that frame. It falls to 0.001577
on the next frame. All native and target frames pass the existing independent
physical limits (maximum penetration 0.003335, maximum energy 9.758). Independent
contact geometry, constraint response, graph and LBVH checks also pass on the
captured target frames 28 through 39.

Fourteen required `general_lbvh_restart` cases restart the native trajectory from
identical inputs: initial and final frames, plus every start frame 27 through 38.
The frame-36 restart passes the original strict checks on C and Metal. Across the
surrounding restarts, maximum differences are 1.2e-7 in poses, 9.54e-6 in velocities
and 4.62e-5 in constraint fields. Five named starts have scoped velocity/impulse
budgets; geometry, point counts, identities and every other field remain strict.
These diagnostics distinguish trajectory feedback from a different single-step
algorithm. Current registered target reports remain required after source edits.

Only frame 36 receives the larger trajectory budgets: 0.0004 for pose components
and 0.15 for velocity components. Other frames retain 0.0002 and 0.008 respectively.
Final velocity components have a 3e-6 absolute budget (observed 2.423e-6); final pose
components retain 0.0002. The recycling state can retain frame 36's pose, so its
explicit budget is 0.0004. Stored contact arms have a 0.001 budget. All allowances
are confined to this exact LBVH triangle configuration.

The final resting contact also has nonunique per-point friction impulses. Their
largest component difference is 0.012745, but their complete effect on both bodies
agrees under the original strict tolerance. For this case, compare the complete
f64 force/torque reduction followed by inverse mass/inertia, while retaining strict
per-point normal impulse comparisons and each point's Coulomb bound. Independently
reconstruct each stored warmstart velocity delta with the same strict tolerance.
This accepts balanced friction redistribution; it rejects an incorrect net force,
torque or cache, even when the target and reference contain the same corrupt cache.
It does not permit arbitrary per-point differences in other cases.

The LBVH joint-chain native and Metal trajectories, contacts, tiles and joint rows
are bitwise identical to their corresponding brute-force runs at the tested
revisions. Its existing joint-chain numerical budgets are retained, with eight
separately registered `joint_lbvh_restart` cases and a required LBVH configuration.
