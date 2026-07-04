"""Python port of `crates/examples3d/joints3.rs`.

A showcase of every joint type (prismatic, revolute, fixed, spherical), with
plain and actuated/limited variants, assembled by a set of scene-builder
helpers. Toggle `USE_MULTIBODY` to switch between multibody (articulation) and
impulse joints. Run with:

    maturin develop -m crates/nexus_python3d/Cargo.toml --features metal
    python crates/nexus_python3d/examples/joints3.py
"""

import math

from nexus3d import (
    NexusViewer,
    NexusPipeline,
    NexusState,
    RigidBodyBuilder,
    ColliderBuilder,
    FixedJointBuilder,
    SphericalJointBuilder,
    RevoluteJointBuilder,
    PrismaticJointBuilder,
    JointAxis,
    MotorModel,
    GpuTimestamps,
    Vec3,
    Pose,
)

# Rust `let use_articulations = true;` in `run`.
USE_MULTIBODY = True

# rapier's `Real::MAX` (f32::MAX), used for a one-sided prismatic limit.
REAL_MAX = 3.4028235e38


def normalized(v: Vec3) -> Vec3:
    """`Vector::normalize()` — no `.normalize()` on the Python `Vec3`."""
    m = math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
    return Vec3(v.x / m, v.y / m, v.z / m)


def add_body(state, viewer, body, collider):
    """Inserts a body + collider and registers its render shape."""
    shape = collider.shared_shape()
    handle = state.insert_rigid_body(body, collider)
    viewer.insert_shape(handle, shape, Pose.IDENTITY)
    return handle


def make_body_builder(is_fixed: bool):
    """`RigidBodyBuilder::new(status)` — Python only exposes `.fixed()`/`.dynamic()`."""
    return RigidBodyBuilder.fixed() if is_fixed else RigidBodyBuilder.dynamic()


def insert_joint(state, use_articulations, b1, b2, joint):
    if use_articulations:
        state.insert_multibody_joint(b1, b2, joint)
    else:
        state.insert_impulse_joint(b1, b2, joint)


def create_prismatic_joints(state, viewer, origin: Vec3, num: int, use_articulations: bool):
    rad = 0.4
    shift = 2.0

    curr_parent = add_body(
        state,
        viewer,
        RigidBodyBuilder.fixed().translation(origin).build(),
        ColliderBuilder.cuboid(rad, rad, rad).build(),
    )

    for i in range(num):
        z = origin.z + (i + 1) * shift
        curr_child = add_body(
            state,
            viewer,
            RigidBodyBuilder.dynamic().translation(Vec3(origin.x, origin.y, z)).build(),
            ColliderBuilder.cuboid(rad, rad, rad).build(),
        )

        if i % 2 == 0:
            axis = normalized(Vec3(1.0, 1.0, 0.0))
        else:
            axis = normalized(Vec3(-1.0, 1.0, 0.0))

        prism = (
            PrismaticJointBuilder.new(axis)
            .local_anchor1(Vec3(0.0, 0.0, 0.0))
            .local_anchor2(Vec3(0.0, 0.0, -shift))
            .limits(-2.0, 2.0)
        )

        insert_joint(state, use_articulations, curr_parent, curr_child, prism)
        curr_parent = curr_child


def create_actuated_prismatic_joints(state, viewer, origin: Vec3, num: int, use_articulations: bool):
    rad = 0.4
    shift = 2.0

    curr_parent = add_body(
        state,
        viewer,
        RigidBodyBuilder.fixed().translation(origin).build(),
        ColliderBuilder.cuboid(rad, rad, rad).build(),
    )

    for i in range(num):
        z = origin.z + (i + 1) * shift
        curr_child = add_body(
            state,
            viewer,
            RigidBodyBuilder.dynamic().translation(Vec3(origin.x, origin.y, z)).build(),
            ColliderBuilder.cuboid(rad, rad, rad).build(),
        )

        if i % 2 == 0:
            axis = normalized(Vec3(1.0, 1.0, 0.0))
        else:
            axis = normalized(Vec3(-1.0, 1.0, 0.0))

        prism = (
            PrismaticJointBuilder.new(axis)
            .local_anchor1(Vec3(0.0, 0.0, shift))
            .local_anchor2(Vec3(0.0, 0.0, 0.0))
        )

        # A max force stops the motor from fighting the limits with huge forces.
        if i == 0:
            prism = prism.limits(-2.0, 5.0).motor_velocity(2.0, 1.0e5).motor_max_force(100.0)
        elif i == 1:
            prism = prism.limits(-REAL_MAX, 5.0).motor_velocity(6.0, 1.0e3).motor_max_force(100.0)
        elif i > 1:
            prism = prism.motor_position(2.0, 1.0e3, 1.0e2).motor_max_force(60.0)

        insert_joint(state, use_articulations, curr_parent, curr_child, prism)
        curr_parent = curr_child


def create_revolute_joints(state, viewer, origin: Vec3, num: int, use_articulations: bool):
    rad = 0.4
    shift = 2.0

    curr_parent = add_body(
        state,
        viewer,
        RigidBodyBuilder.fixed().translation(Vec3(origin.x, origin.y, 0.0)).build(),
        ColliderBuilder.cuboid(rad, rad, rad).build(),
    )

    for i in range(num):
        z = origin.z + i * shift * 2.0 + shift
        positions = [
            Pose.from_translation(Vec3(origin.x, origin.y, z)),
            Pose.from_translation(Vec3(origin.x + shift, origin.y, z)),
            Pose.from_translation(Vec3(origin.x + shift, origin.y, z + shift)),
            Pose.from_translation(Vec3(origin.x, origin.y, z + shift)),
        ]

        handles = [curr_parent] * 4
        for k in range(4):
            handles[k] = add_body(
                state,
                viewer,
                RigidBodyBuilder.dynamic().pose(positions[k]).build(),
                ColliderBuilder.cuboid(rad, rad, rad).build(),
            )

        ax = Vec3.X
        az = Vec3.Z
        revs = [
            RevoluteJointBuilder.new(az).local_anchor2(Vec3(0.0, 0.0, -shift)),
            RevoluteJointBuilder.new(ax).local_anchor2(Vec3(-shift, 0.0, 0.0)),
            RevoluteJointBuilder.new(az).local_anchor2(Vec3(0.0, 0.0, -shift)),
            RevoluteJointBuilder.new(ax).local_anchor2(Vec3(shift, 0.0, 0.0)),
        ]

        insert_joint(state, use_articulations, curr_parent, handles[0], revs[0])
        insert_joint(state, use_articulations, handles[0], handles[1], revs[1])
        insert_joint(state, use_articulations, handles[1], handles[2], revs[2])
        insert_joint(state, use_articulations, handles[2], handles[3], revs[3])

        curr_parent = handles[3]


def create_revolute_joints_with_limits(state, viewer, origin: Vec3, use_articulations: bool):
    ground = add_body(
        state,
        viewer,
        RigidBodyBuilder.fixed().translation(origin).build(),
        ColliderBuilder.cuboid(0.1, 0.1, 0.1).build(),
    )

    shift = Vec3(0.0, 0.0, 6.0)
    platform1 = add_body(
        state,
        viewer,
        RigidBodyBuilder.dynamic().translation(origin + shift).build(),
        ColliderBuilder.cuboid(4.0, 0.2, 2.0).build(),
    )

    platform2 = add_body(
        state,
        viewer,
        RigidBodyBuilder.dynamic().translation(origin + shift * 2.0).build(),
        ColliderBuilder.cuboid(4.0, 0.2, 2.0).build(),
    )

    z = Vec3.Z
    joint1 = RevoluteJointBuilder.new(z).local_anchor1(shift).limits(-0.2, 0.2)
    insert_joint(state, use_articulations, ground, platform1, joint1)

    joint2 = RevoluteJointBuilder.new(z).local_anchor2(-shift).limits(-0.2, 0.2)
    insert_joint(state, use_articulations, platform1, platform2, joint2)

    # A couple of cuboids that fall on the platforms, triggering the limits.
    add_body(
        state,
        viewer,
        RigidBodyBuilder.dynamic()
        .translation(origin + shift + Vec3(-2.0, 4.0, 0.0))
        .build(),
        ColliderBuilder.cuboid(0.6, 0.6, 0.6).friction(1.0).build(),
    )

    add_body(
        state,
        viewer,
        RigidBodyBuilder.dynamic()
        .translation(origin + shift * 2.0 + Vec3(2.0, 16.0, 0.0))
        .build(),
        ColliderBuilder.cuboid(0.6, 0.6, 0.6).friction(1.0).build(),
    )


def create_fixed_joints(state, viewer, origin: Vec3, num: int, use_articulations: bool):
    rad = 0.4
    shift = 1.0

    body_handles = []

    for i in range(num):
        for k in range(num):
            fk = float(k)
            fi = float(i)

            # NOTE: the num - 2 test is to avoid two consecutive fixed bodies.
            is_fixed = i == 0 and ((k % 4 == 0 and k != num - 2) or k == num - 1)

            child_handle = add_body(
                state,
                viewer,
                make_body_builder(is_fixed)
                .translation(Vec3(origin.x + fk * shift, origin.y, origin.z + fi * shift))
                .build(),
                ColliderBuilder.ball(rad).build(),
            )

            # Vertical joint.
            if i > 0:
                parent_index = len(body_handles) - num
                parent_handle = body_handles[parent_index]
                joint = FixedJointBuilder.new().local_anchor2(Vec3(0.0, 0.0, -shift))
                insert_joint(state, use_articulations, parent_handle, child_handle, joint)

            # Horizontal joint (always an impulse joint in the Rust source).
            if k > 0:
                parent_index = len(body_handles) - 1
                parent_handle = body_handles[parent_index]
                joint = FixedJointBuilder.new().local_anchor2(Vec3(-shift, 0.0, 0.0))
                state.insert_impulse_joint(parent_handle, child_handle, joint)

            body_handles.append(child_handle)


def create_spherical_joints(state, viewer, num: int, use_articulations: bool):
    rad = 0.4
    shift = 1.0

    body_handles = []

    for k in range(num):
        for i in range(num):
            fk = float(k)
            fi = float(i)

            is_fixed = i == 0 and (k % 4 == 0 or k == num - 1)

            child_handle = add_body(
                state,
                viewer,
                make_body_builder(is_fixed)
                .translation(Vec3(fk * shift, 0.0, fi * shift * 2.0))
                .build(),
                ColliderBuilder.capsule_z(rad * 1.25, rad).build(),
            )

            # Vertical joint.
            if i > 0:
                parent_handle = body_handles[-1]
                joint = SphericalJointBuilder.new().local_anchor2(Vec3(0.0, 0.0, -shift * 2.0))
                insert_joint(state, use_articulations, parent_handle, child_handle, joint)

            # Horizontal joint (always an impulse joint in the Rust source).
            if k > 0:
                parent_index = len(body_handles) - num
                parent_handle = body_handles[parent_index]
                joint = SphericalJointBuilder.new().local_anchor2(Vec3(-shift, 0.0, 0.0))
                state.insert_impulse_joint(parent_handle, child_handle, joint)

            body_handles.append(child_handle)


def create_spherical_joints_with_limits(state, viewer, origin: Vec3, use_articulations: bool):
    shift = Vec3(0.0, 0.0, 3.0)

    ground = add_body(
        state,
        viewer,
        RigidBodyBuilder.fixed().translation(origin).build(),
        ColliderBuilder.cuboid(0.1, 0.1, 0.1).build(),
    )

    ball1 = add_body(
        state,
        viewer,
        RigidBodyBuilder.dynamic()
        .translation(origin + shift)
        .linvel(Vec3(20.0, 20.0, 0.0))
        .build(),
        ColliderBuilder.cuboid(1.0, 1.0, 1.0).build(),
    )

    ball2 = add_body(
        state,
        viewer,
        RigidBodyBuilder.dynamic().translation(origin + shift * 2.0).build(),
        ColliderBuilder.cuboid(1.0, 1.0, 1.0).build(),
    )

    joint1 = (
        SphericalJointBuilder.new()
        .local_anchor2(-shift)
        .limits(JointAxis.LinX, -0.2, 0.2)
        .limits(JointAxis.LinY, -0.2, 0.2)
    )

    joint2 = (
        SphericalJointBuilder.new()
        .local_anchor2(-shift)
        .limits(JointAxis.LinX, -0.3, 0.3)
        .limits(JointAxis.LinY, -0.3, 0.3)
    )

    insert_joint(state, use_articulations, ground, ball1, joint1)
    insert_joint(state, use_articulations, ball1, ball2, joint2)


def create_actuated_revolute_joints(state, viewer, origin: Vec3, num: int, use_articulations: bool):
    rad = 0.4
    shift = 2.0

    z = Vec3.Z
    joint_template = RevoluteJointBuilder.new(z).local_anchor2(Vec3(0.0, 0.0, -shift))

    parent_handle = None

    for i in range(num):
        fi = float(i)

        is_fixed = i == 0
        shifty = -2.0 if i >= 1 else 0.0

        child_handle = add_body(
            state,
            viewer,
            make_body_builder(is_fixed)
            .translation(Vec3(origin.x, origin.y + shifty, origin.z + fi * shift))
            .build(),
            ColliderBuilder.cuboid(rad * 2.0, rad * 6.0 / (fi + 1.0), rad).build(),
        )

        if i > 0:
            joint = joint_template.motor_model(MotorModel.AccelerationBased)

            if i % 3 == 1:
                joint = joint.motor_velocity(-20.0, 100.0)
            elif i == num - 1:
                stiffness = 200.0
                damping = 100.0
                joint = joint.motor_position(math.pi / 2.0, stiffness, damping)

            if i == 1:
                joint = joint.local_anchor2(Vec3(0.0, 2.0, -shift)).motor_velocity(-2.0, 1000.0)

            insert_joint(state, use_articulations, parent_handle, child_handle, joint)

        parent_handle = child_handle


def create_actuated_spherical_joints(state, viewer, origin: Vec3, num: int, use_articulations: bool):
    rad = 0.4
    shift = 2.0

    joint_template = SphericalJointBuilder.new().local_anchor1(Vec3(0.0, 0.0, shift))

    parent_handle = None

    for i in range(num):
        fi = float(i)

        is_fixed = i == 0

        child_handle = add_body(
            state,
            viewer,
            make_body_builder(is_fixed)
            .translation(Vec3(origin.x, origin.y, origin.z + fi * shift))
            .build(),
            ColliderBuilder.capsule_y(rad * 2.0 / (fi + 1.0), rad).build(),
        )

        if i > 0:
            joint = joint_template

            if i == 1:
                joint = (
                    joint.motor_velocity(JointAxis.AngX, 0.0, 0.1)
                    .motor_velocity(JointAxis.AngY, 0.5, 0.1)
                    .motor_velocity(JointAxis.AngZ, -2.0, 0.1)
                )
            elif i == num - 1:
                stiffness = 0.2
                damping = 1.0
                joint = (
                    joint.motor_position(JointAxis.AngX, 0.0, stiffness, damping)
                    .motor_position(JointAxis.AngY, 1.0, stiffness, damping)
                    .motor_position(JointAxis.AngZ, math.pi / 2.0, stiffness, damping)
                )

            insert_joint(state, use_articulations, parent_handle, child_handle, joint)

        parent_handle = child_handle


def run(viewer: NexusViewer, pipeline: NexusPipeline) -> NexusState:
    use_articulations = USE_MULTIBODY

    state = NexusState()

    create_prismatic_joints(state, viewer, Vec3(20.0, 5.0, 0.0), 4, use_articulations)
    create_actuated_prismatic_joints(state, viewer, Vec3(25.0, 5.0, 0.0), 4, use_articulations)
    create_revolute_joints(state, viewer, Vec3(20.0, 0.0, 0.0), 3, use_articulations)
    create_revolute_joints_with_limits(state, viewer, Vec3(34.0, 0.0, 0.0), use_articulations)
    create_fixed_joints(state, viewer, Vec3(0.0, 10.0, 0.0), 10, use_articulations)
    create_actuated_revolute_joints(state, viewer, Vec3(20.0, 10.0, 0.0), 6, use_articulations)
    create_actuated_spherical_joints(state, viewer, Vec3(13.0, 10.0, 0.0), 3, use_articulations)

    create_spherical_joints(state, viewer, 9, use_articulations)
    create_spherical_joints_with_limits(state, viewer, Vec3(-5.0, 0.0, 0.0), use_articulations)

    timestamps = GpuTimestamps(viewer, 2048)
    viewer.add_directional_light(Vec3(1.0, -2.0, 3.0))
    state.finalize(viewer)

    while viewer.render_frame():
        if viewer.simulating():
            pipeline.simulate(viewer, state, timestamps)
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
    # The work is done once the window closes. Exit without running interpreter
    # finalization: some GPU/windowing thread-locals abort if destroyed during
    # Python teardown (an upstream ordering quirk, harmless to skip).
    import os

    os._exit(0)
