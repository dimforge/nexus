# Writing the Nexus user guide

## Reference and scope

This guide follows the Rapier website at `website/docs/user_guides/templates/` in
`dimforge/rapier`, read on 2026-09-23. The corpus below includes all 54 shared guide
pages and section fragments present in that checkout, including the Rust, Bevy,
and JavaScript introductions. Soft-body pages and soft-body examples were excluded
from the writing-style reference. Nexus's own MPM coverage comes from Nexus code.

The workflow reference was Rapier's `generate_user_guides.sh`,
`inject_code_in_user_guides.sh`, `docs-examples/inject_file/src/main.rs`, example
packages, and `sidebar_docs.js`. Nexus keeps shared templates, inline conditional
replacement, extracted marked regions, language guides, and dimension tabs. A small
Node generator replaces the platform-specific sed/awk shell pipeline, so no GNU
sed or awk installation is needed.

## Observed voice

Rapier starts with the object or behavior, then explains why it is useful. A page
like Rigid-bodies or Joints first defines the concept, gives familiar use-cases,
then introduces construction and settings. It rarely starts with an implementation
detail or a broad motivational introduction.

The prose is direct and explanatory. It talks to the reader with "you", "keep in
mind", and "make sure", but frequently uses impersonal constructions such as "It
is possible to" and "It is recommended to". It repeats names rather than replacing
every repetition with a synonym. A paragraph normally explains what something is,
what it does, and what follows from that behavior.

Keep the recognizable vocabulary: "rigid-body", "rigid-bodies", "world-space",
"local-space", "collision-detection", "wrt.", "i.e.", and "e.g.". Introduce terms
such as Degrees Of Freedom before abbreviating them. Use explicit examples such
as a ground, moving platform, pendulum, or robotic arm. Avoid marketing language.

The English is sometimes uneven (articles, agreement, "This is typically used
for", and repeated explanation). Preserve the plain, slightly formal cadence;
don't polish the guide into a different editorial voice, and don't manufacture
errors merely to imitate it. Technical meaning must remain precise.

## Structure to reuse

- Begin Nexus-specific concepts with a definition and a use-case. For concepts already
  covered by Rapier, link to the relevant Rapier user-guide section instead of defining
  them again. Keep the local explanation about Nexus's behavior and limitations.
- Put "Creation and insertion" early; group the remaining sections by behavior.
- Introduce a code example with the variables or handles it expects. Keep the
  code's full setup in a compiling source file even when only a region is shown.
- Explain local/world frames, units, defaults, and special values near their API.
- Follow code with the caveat that affects how the reader should use it.
- Use `:::info`, `:::note`, and `:::warning` for concrete information. Rapier's
  warnings are about surprising physics or API behavior, not generic disclaimers.
- Link to the dedicated section rather than repeating its entire explanation.
- Fold symptom-based checks into the chapter that provides the corresponding API.
- Keep chapters substantial, with extracted code examples in every language. Merge
  short reference-only chapters into the relevant workflow.
- Separate rigid-body and MPM chapters in the sidebar. Shared initialization and
  viewer operations belong in the common introduction; solver settings stay with
  their own solver.
- Name the callable API or usable type/field when describing it: write
  "call `state.finalize`", not "finalize the scene" alone; write `params.dt`, not
  "the timestep parameter" alone. Use inline language conditions where names differ.
- Use ordinary nouns for example-local objects: "Create a body and collider with",
  not "Create `body` and `collider` with". Don't introduce names from the following
  example merely to put them in backticks. Keep a variable name when it matters,
  such as identifying which returned array is being described or connecting two
  snippets through the same handle. API names, fields, and types still use backticks.
- Link Rapier APIs and concepts to their user-guide sections, including when they
  appear in a Nexus compatibility warning. Don't duplicate Rapier's definitions.

Rapier's shared prose uses inline substitutions even for a single method name.
Keep Nexus's `<rust>` and `<python>` conditions similarly small when the concept
is shared. Use a whole conditional section only for a real difference in API
availability. Keep synchronized 2D/3D tabs for Rust and no misleading 2D tab for Python.

## Nexus-specific accuracy

Reusing Rapier's voice doesn't transfer its feature list. Check the Nexus source
and run its examples before describing a setting. Several Rapier builders expose
properties that Nexus doesn't upload or simulate. Explain the authored scene versus
running buffers, matching batch topology, runtime setters, readbacks, and MPM
coupling using the actual Nexus implementation. A method merely existing in a
builder is not evidence of support by the GPU pipeline.

## Read corpus

- `advanced_collision_detection.mdx`
- `advanced_collision_detection_js.mdx`
- `character_controller.mdx`
- `character_controller_collisions.mdx`
- `character_controller_filtering.mdx`
- `character_controller_offset.mdx`
- `character_controller_setup.mdx`
- `character_controller_slopes.mdx`
- `character_controller_snap_to_ground.mdx`
- `character_controller_stairs.mdx`
- `character_controller_up_vector.mdx`
- `collider_active_collision_types.mdx`
- `collider_active_events.mdx`
- `collider_active_hooks.mdx`
- `collider_collision_groups.mdx`
- `collider_creation_and_insertion.mdx`
- `collider_friction.mdx`
- `collider_mass_properties.mdx`
- `collider_position.mdx`
- `collider_restitution.mdx`
- `collider_type.mdx`
- `colliders.mdx`
- `common_mistakes.mdx`
- `common_recipes.mdx`
- `determinism.mdx`
- `getting_started.mdx`
- `getting_started_bevy.mdx`
- `getting_started_js.mdx`
- `integration_parameters.mdx`
- `joint_constraints.mdx`
- `joints.mdx`
- `multiple_contexts.mdx`
- `rigid_bodies.mdx`
- `rigid_body_ccd.mdx`
- `rigid_body_creation_and_insertion.mdx`
- `rigid_body_damping.mdx`
- `rigid_body_dominance.mdx`
- `rigid_body_forces_and_impulses.mdx`
- `rigid_body_gravity.mdx`
- `rigid_body_locking_translations_rotations.mdx`
- `rigid_body_mass_properties.mdx`
- `rigid_body_position.mdx`
- `rigid_body_sleeping.mdx`
- `rigid_body_type.mdx`
- `rigid_body_velocity.mdx`
- `scene_queries.mdx`
- `scene_queries_filters.mdx`
- `scene_queries_intersection_test.mdx`
- `scene_queries_point_projection.mdx`
- `scene_queries_ray_casting.mdx`
- `scene_queries_shape_casting.mdx`
- `serialization.mdx`
- `simulation_structures.mdx`
- `the_rapier_testbed.mdx`
