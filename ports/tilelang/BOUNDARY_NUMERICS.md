# Zero-distance CFM classification

`rbd.contact.coulomb.boundary3d` retains all 17 original cancellation stress cases.
The Rust and TileLang solver arithmetic is unchanged. A contact distance extremely
close to zero can round to either side of the native `dist <= 0` condition, selecting
a soft CFM factor or one. That discontinuity can exceed the usual output tolerance
even when each arithmetic operation is correctly rounded.

The registered policy is deliberately narrow: fixed input poses, disjoint writable
velocity slots, at most one ambiguous point per manifold, and one to three full
solve passes. `boundary_interval.py` and `reference/src/boundary_interval.rs`
independently propagate outward intervals for binary32 operations. Each operation
expands its binary64 interval by `max_abs * u / (1 - u) + 2^-149`, with `u = 2^-24`,
then rounds the endpoints outwards. The distance interval includes the cross-product
and scalar/vector quaternion rotation identities and both three-term dot-product
association orders. It does not assume the stored quaternion has exact unit norm.
A point is ambiguous only if this interval contains zero. Intervals wider than
`1e-5`, nonfinite bounds, or multiple ambiguous points reject the fixture.

The native oracle extracts the current contact solver arithmetic from
`solver_utils.rs`. It changes only the CFM selector to produce complete soft and
hard alternatives. Extraction fails if the source markers, signature or selector
change. Before exporting either alternative, the unforced extracted solver must
produce bit-identical tiles and velocities to the original native solver. This
keeps the diagnostic tied to the Rust reference rather than maintaining another
handwritten physics implementation.

The target's complete manifold output (all normal/friction impulses and both body
velocities) must match one allowed native outcome at the original `atol=2e-6` and
`rtol=2e-5`. Different components cannot select different branches, and interpolation
between the two outcomes is not accepted. Sign-resolved points and unbiased solves
retain their unique native outcome. Integer identities, padding, guards, readonly
buffers, finite outputs, nonnegative impulses and Coulomb bounds remain checked.
Native outputs must satisfy the same validator as target outputs.

This policy does not assert that every compiler will choose the same branch, or
that arbitrary cancellation is harmless. It cannot be extended to changing poses,
more ambiguous points or longer solve sequences without new analysis and tests.
The registered scene and identical-state restart suites provide separate physical
and trajectory evidence. Original Rust Metal diagnostics remain separate from
TileLang execution reports; unrelated residuals are not excused by this policy.
Actual Ascend compilation, execution and memory-model qualification remain pending.

Changes to this policy, its intervals, source extraction, required cases or output
checks require a classified change record and fresh per-target evidence. Both this
document and the scene numerical policy belong to the conservative source hash
envelope. Historical reports do not validate later sources.
