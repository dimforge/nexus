"""Native compact-tile ABI counterpart of dynamics/twist_tiles.rs."""


def validate_layout(l):
    if (
        l.get("friction_model") != "twist"
        or l["tile_len"] != 32
        or l["max_points"] != 4
    ):
        raise ValueError("Unsupported native twist ABI")
    if l["twist_arm_words"] != 128:
        raise ValueError("Invalid native twist point stride")
    regions = [(offset, 128) for offset in l["header"].values()]
    regions += [
        (offset, 512 if name == "arm_a" else 128) for name, offset in l["twist"].items()
    ]
    regions += [(l["warmstart"], 64 * l["velocity_words"])]
    occupied = set()
    for offset, width in regions:
        if type(offset) is not int or not 0 <= offset <= l["tile_words"] - width:
            raise ValueError("Invalid native twist field")
        words = set(range(offset, offset + width))
        if words & occupied:
            raise ValueError("Overlapping native twist fields")
        occupied |= words


def make_kernel(layout, case, target):
    from kernel_support import launch

    from . import twist_friction

    operation = {
        "twist_effective_mass": twist_friction.compute_effective_masses,
        "twist_warmstart": twist_friction.warmstart,
        "twist_solve": twist_friction.solve,
        "twist_scale": twist_friction.scale_impulses,
    }[case["kind"]]
    return launch(operation(layout, case), layout, case, target, case["count"])


def constraint_warmstart(l):
    """Cache native warmstart deltas, tagged by ordinary bodies rather than slots."""
    import tilelang.language as T
    from kernel_support import cross, f32

    h, t = l["header"], l["twist"]

    @T.macro
    def mul(Tiles, b, diag, off, v, out):
        out[0] = (
            f32(Tiles, b + diag) * v[0]
            + f32(Tiles, b + off) * v[1]
            + f32(Tiles, b + off + 1) * v[2]
        )
        out[1] = (
            f32(Tiles, b + off) * v[0]
            + f32(Tiles, b + diag + 1) * v[1]
            + f32(Tiles, b + off + 2) * v[2]
        )
        out[2] = (
            f32(Tiles, b + off + 1) * v[0]
            + f32(Tiles, b + off + 2) * v[1]
            + f32(Tiles, b + diag + 2) * v[2]
        )

    @T.macro
    def one(Tiles, i):
        base = i // 32 * l["tile_words"] + i % 32 * 4
        cache = (
            i // 32 * l["tile_words"]
            + l["warmstart"]
            + i % 32 * 2 * l["velocity_words"]
        )
        force = T.alloc_local((3,), "float32")
        torque_a = T.alloc_local((3,), "float32")
        torque_b = T.alloc_local((3,), "float32")
        response_a = T.alloc_local((3,), "float32")
        response_b = T.alloc_local((3,), "float32")
        for d in T.serial(3):
            force[d] = 0.0
            torque_a[d] = 0.0
            torque_b[d] = 0.0
        count = T.cast(Tiles[base + h["normal"] + 3], "int32")
        if count > 0:
            n = T.alloc_local((3,), "float32")
            tangent = T.alloc_local((3,), "float32")
            bitangent = T.alloc_local((3,), "float32")
            ra = T.alloc_local((3,), "float32")
            rb = T.alloc_local((3,), "float32")
            impulse = T.alloc_local((3,), "float32")
            a = T.alloc_local((3,), "float32")
            b = T.alloc_local((3,), "float32")
            for d in T.serial(3):
                n[d] = f32(Tiles, base + h["normal"] + d)
                tangent[d] = f32(Tiles, base + h["tangent"] + d)
            cross(n, tangent, bitangent)
            for k in T.serial(4):
                if k < count:
                    for d in T.serial(3):
                        ra[d] = f32(
                            Tiles, base + t["arm_a"] + k * l["twist_arm_words"] + d
                        )
                        rb[d] = ra[d] + f32(Tiles, base + t["offset_b"] + d)
                        impulse[d] = n[d] * f32(Tiles, base + t["normal_impulse"] + k)
                        force[d] += impulse[d]
                    cross(ra, impulse, a)
                    cross(rb, impulse, b)
                    for d in T.serial(3):
                        torque_a[d] += a[d]
                        torque_b[d] += b[d]
            for d in T.serial(3):
                impulse[d] = tangent[d] * f32(Tiles, base + t["friction"]) + bitangent[
                    d
                ] * f32(Tiles, base + t["friction"] + 1)
                force[d] += impulse[d]
                ra[d] = f32(Tiles, base + t["center_a"] + d)
                rb[d] = ra[d] + f32(Tiles, base + t["offset_b"] + d)
            cross(ra, impulse, a)
            cross(rb, impulse, b)
            for d in T.serial(3):
                twist = n[d] * f32(Tiles, base + h["off_b"] + 3)
                torque_a[d] += a[d] + twist
                torque_b[d] += b[d] + twist
        mul(Tiles, base, h["diag_a"], h["off_a"], torque_a, response_a)
        mul(Tiles, base, h["diag_b"], h["off_b"], torque_b, response_b)
        for d in T.serial(3):
            Tiles[cache + l["velocity_linear"] + d] = T.reinterpret(
                f32(Tiles, base + h["body_a"] + d) * force[d], "uint32"
            )
            Tiles[cache + l["velocity_angular"] + d] = T.reinterpret(
                response_a[d], "uint32"
            )
            Tiles[cache + l["velocity_words"] + l["velocity_linear"] + d] = (
                T.reinterpret(-f32(Tiles, base + h["body_b"] + d) * force[d], "uint32")
            )
            Tiles[cache + l["velocity_words"] + l["velocity_angular"] + d] = (
                T.reinterpret(-response_b[d], "uint32")
            )
        for side in T.serial(2):
            Tiles[cache + side * l["velocity_words"] + l["velocity_padding0"]] = (
                T.uint32(0)
            )
        Tiles[cache + l["velocity_padding1"]] = Tiles[base + h["body_a"] + 3]
        Tiles[cache + l["velocity_words"] + l["velocity_padding1"]] = Tiles[
            base + h["body_b"] + 3
        ]

    return one


def gpu_prepare_constraints(layout, case):
    import tilelang.language as T

    from .solver_utils import contact_to_constraint_header
    from .tile_kernels import transfer_contact_impulses
    from .twist_friction import (
        compute_effective_masses,
        init_points_with_poses,
        recycle_from,
    )

    header = contact_to_constraint_header(layout)
    points = init_points_with_poses(layout, case)
    recycle = recycle_from(layout)
    transfer = transfer_contact_impulses(layout)
    masses = compute_effective_masses(layout, case)
    cache = constraint_warmstart(layout)
    l, t = layout["link"], layout["twist"]

    @T.macro
    def one(
        Links, Contacts, WorldMass, SolverPoses, Velocities, States, OldTiles, Tiles, i
    ):
        if i < case["count"]:
            lb = i * layout["link_words"]
            if Links[lb + l["len"]] != 0:
                j = T.cast(Links[lb + l["contact"]], "int32")
                base = i // 32 * layout["tile_words"] + i % 32 * 4
                for d in T.serial(4):
                    Tiles[base + t["friction"] + d] = T.uint32(0)
                    Tiles[base + t["normal_impulse"] + d] = T.uint32(0)
                    Tiles[base + t["dist"] + d] = T.uint32(0)
                    Tiles[base + t["normal_vel"] + d] = T.uint32(0)
                    Tiles[base + t["radius"] + d] = T.uint32(0)
                header(Links, Contacts, WorldMass, Tiles, i, j)
                if Links[lb + l["recycled"]] == 0:
                    points(
                        Links, Contacts, SolverPoses, Velocities, States, Tiles, i, j
                    )
                if Links[lb + l["previous_constraint"]] != T.uint32(0xFFFFFFFF):
                    old = T.cast(Links[lb + l["previous_constraint_index"]], "int32")
                    if Links[lb + l["recycled"]] != 0:
                        recycle(Links, Velocities, OldTiles, Tiles, i, old)
                    else:
                        transfer(OldTiles, Tiles, i, old)
                masses(Tiles, Velocities, SolverPoses, Contacts, i)
                cache(Tiles, i)

    return one
