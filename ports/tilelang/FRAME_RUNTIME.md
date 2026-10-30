# Resident frame execution

`native::frame::FrameSession` advances a compiled frame on an existing khal Metal
device. A `FrameVariant` contains the ordered program, its kernel sources, a
complete initial upload, and a `FrameManifest`. Python/TVM are unnecessary during
execution. Only the small capacity counters are read between stages. Explicit
`read` calls return requested outputs for diagnostics or application use.

A frame manifest identifies capacity gates before contact preparation and solving.
The broad-phase gate reads the original, unclamped pair demand. General collision
adds a second gate after deferred traversal, before contact-plan publication.
The existing native invariant reserves two contact slots per pair-capacity unit,
so these two successful bounds also establish contact-plan capacity.

If demand exceeds capacity, the runtime selects the first later exported variant
that fits it, compiles its kernels, allocates and initializes its storage, and uses
a device blit to move persistent state. It then reruns collision before solving.
Failure to find a fitting variant returns `StepError::Capacity` without advancing
physics or the frame counter. Failed compilation/allocation leaves the previous
specialization available. A dispatch failure poisons the session because a
partial solver update cannot safely be retried.

Persistent region identities describe poses, velocities, previous contact links,
adjacency, colors, sorted indices, tile impulses and old recycling records. Live
record ranges exclude allocation guards. Whole occupied SoA tiles are copied;
new tile capacity starts with native defaults. Traces and their cursor remain
persistent when enabled. Each variant must describe the same physical scene and
parameters, with monotone capacities and no truncated persistent region.
Native recycling records include identity rotations: zero-filling those records
is not equivalent to their Rust defaults.

Construction is unsafe for the same reason as `NativeProgram::dispatch`: shader
source is trusted executable code, and a binding manifest cannot prove index
bounds or race freedom. Additionally, every gate must precede writes to persistent
regions. `frame_manifest.from_program` derives control gates from the collision
controller and verifies that the lowered schedule has the same names and mappings.
The controller annotations do not change shader arithmetic or launch order.

The executor supports a finite ladder of exported capacities. It does not yet
provide arbitrary Nexus scene upload, dynamic specialization compilation, palette
convergence, hub scheduling, or the public backend selector. Those remain required
work. Metal unit tests exercise real dispatch and device migration. Ascend kernel
source lowering does not validate a CANN executor, device memory ordering or
performance; the same frame boundaries must be implemented and qualified there.
