"""Python port of `crates/examples3d/dynamic_rbd3.rs`.

Demonstrates adding rigid-bodies to a live scene without rebuilding the whole
GPU state each time: the scene reserves spare GPU slots up-front
(`reserve_rigid_bodies`), then drops in a batch of bodies every few frames via
`add_rigid_bodies`, which appends directly to the existing GPU buffers so the
new bodies are simulated immediately.
"""

from nexus3d import (
    NexusViewer,
    NexusPipeline,
    NexusState,
    RigidBodyBuilder,
    ColliderBuilder,
    GpuTimestamps,
    Vec3,
    Vec4,
    Pose,
)

# Every `SPAWN_PERIOD` simulated frames, drop in a batch of `BODIES_PER_SPAWN`
# bodies at once, up to `MAX_BODIES`.
SPAWN_PERIOD = 2
BODIES_PER_SPAWN = 100
MAX_BODIES = 50000
# Spare GPU slots reserved for the ground + all dynamically-added bodies, so
# none of the appends trigger a full scene rebuild.
RESERVE = MAX_BODIES + 16
# Horizontal grid the falling bodies are laid out on, so a whole batch is
# inserted without self-overlap; successive layers stack upward.
GRID = 20


def run(viewer: NexusViewer, pipeline: NexusPipeline) -> NexusState:
    state = NexusState()

    # A boxed ground: a floor plus four low walls to keep the pile contained.
    floor_half = 50.0
    wall_h = 4.0
    walls_color = Vec4(0.6, 0.8, 1.0, 0.3)
    ground_parts = [
        (Vec3(0.0, -0.5, 0.0), Vec3(floor_half, 0.5, floor_half)),
        (Vec3(floor_half, wall_h, 0.0), Vec3(0.5, wall_h, floor_half)),
        (Vec3(-floor_half, wall_h, 0.0), Vec3(0.5, wall_h, floor_half)),
        (Vec3(0.0, wall_h, floor_half), Vec3(floor_half, wall_h, 0.5)),
        (Vec3(0.0, wall_h, -floor_half), Vec3(floor_half, wall_h, 0.5)),
    ]
    for pos, half in ground_parts:
        body = RigidBodyBuilder.fixed().build()
        collider = (
            ColliderBuilder.cuboid(half.x, half.y, half.z).translation(pos).build()
        )
        shape = collider.shared_shape()
        handle = state.insert_rigid_body(body, collider)
        viewer.insert_shape_with_color(
            handle, shape, Pose.from_translation(pos), walls_color
        )

    # Reserve the spare slots BEFORE the first `finalize`, so the GPU buffers are
    # built large enough to append into later.
    state.reserve_rigid_bodies(RESERVE)

    timestamps = GpuTimestamps(viewer, 2048)
    viewer.add_directional_light(Vec3(1.0, -2.0, 3.0))
    state.finalize(viewer)

    frame = 0
    added = 0

    while viewer.render_frame():
        if viewer.simulating():
            # Drop in a whole batch of bodies periodically — a single in-place
            # GPU append, without rebuilding the scene.
            if frame % SPAWN_PERIOD == 0 and added < MAX_BODIES:
                n = min(BODIES_PER_SPAWN, MAX_BODIES - added)
                bodies = []
                colliders = []
                shapes = []
                for k in range(n):
                    idx = added + k
                    # Unique grid slot per body (a rising column high above the
                    # floor), so a simultaneously-inserted batch never overlaps.
                    cell = idx % (GRID * GRID)
                    gx = float(cell % GRID) - (GRID - 1.0) * 0.5
                    gz = float(cell // GRID) - (GRID - 1.0) * 0.5
                    layer = float(idx // (GRID * GRID))
                    pos = Vec3(gx * 1.4, 18.0 + layer * 1.3, gz * 1.4)

                    bodies.append(RigidBodyBuilder.dynamic().translation(pos).build())
                    # Alternate between balls and cubes (both primitive shapes,
                    # which is what the in-place append path supports).
                    if idx % 3 == 0:
                        collider = ColliderBuilder.ball(0.5).build()
                    else:
                        collider = ColliderBuilder.cuboid(0.4, 0.4, 0.4).build()
                    colliders.append(collider)
                    shapes.append(collider.shared_shape())

                handles = state.add_rigid_bodies(viewer, bodies, colliders)
                for handle, shape in zip(handles, shapes):
                    viewer.insert_shape(handle, shape, Pose.IDENTITY)
                added += n

            pipeline.simulate(viewer, state, timestamps)
            frame += 1
        viewer.sync(state, timestamps)

    return state


def main() -> None:
    viewer = NexusViewer()
    viewer.init_backend()
    pipeline = NexusPipeline()
    pipeline.preload_pipelines(viewer)
    run(viewer, pipeline)


if __name__ == "__main__":
    main()
    import os

    os._exit(0)
