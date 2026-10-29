"""Contracts for composed Coulomb TGS frames over supplied contact manifolds."""

BUFFERS = (
    "Poses",
    "SolverPoses",
    "Velocities",
    "SolverVelocities",
    "Increments",
    "WorldMass",
    "LocalMass",
    "Counts",
    "FirstSlot",
    "HubCounts",
    "Tiles",
    "RbdParams",
    "Gravity",
    "BatchIds",
    "Buckets",
    "Links",
    "Contacts",
    "States",
    "OldTiles",
    "Color0",
    "Color1",
    "Color2",
    "Mode",
    "Params",
    "Plan",
)
OPERATIONS = ("tgs_frame",)


def body_case(case, kind):
    return dict(
        case,
        kind=kind,
        count=case["body_count"],
        storage=case["body_storage"],
        slots=case["body_count"],
        bodies_per_batch=case["body_count"] // case["batches"],
        capacity_per_batch=case["body_count"] // case["batches"],
    )


def validate_case(layout, case, arrays):
    import numpy as np
    from contact_replay import (
        validate_case as validate_contact,
        BUFFERS as CONTACT_BUFFERS,
    )
    from body_validation import validate_case as validate_body, BUFFERS as BODY_BUFFERS

    keys = (
        "count",
        "body_count",
        "body_storage",
        "contact_storage",
        "batches",
        "columns",
        "levels",
        "substeps",
        "pgs",
        "mode",
        "storage",
        "bodies",
        "states",
        "new_base",
        "tiles",
        "old_tiles",
        "groups",
        "iterations",
    )
    if case["kind"] != "tgs_frame" or any(
        type(case[k]) is not int or not 0 <= case[k] < 2**20 for k in keys
    ):
        raise ValueError("Invalid TGS frame dimensions")
    if not (
        1 <= case["batches"] <= 64
        and 1 <= case["substeps"] <= 64
        and case["pgs"] <= 64
        and case["mode"] in (1, 2)
        and case["iterations"] == 1
        and 1 <= case["groups"] <= 1024
    ):
        raise ValueError("Invalid TGS frame schedule")
    if (
        case["body_count"] % case["batches"]
        or not case["body_count"] < case["body_storage"]
        or case["bodies"] != case["body_storage"]
        or case["storage"] != case["contact_storage"]
        or case["states"] != case["contact_storage"]
        or case["new_base"] != 0
    ):
        raise ValueError("Invalid TGS frame capacities")
    if case["body_count"] != case["columns"] * (case["levels"] + 1) * case["batches"]:
        raise ValueError("Frame scene dimensions differ from body count")
    if case["count"] != (
        0
        if case["name"] == "free"
        else case["columns"] * case["levels"] * case["batches"]
    ):
        raise ValueError("Frame scene dimensions differ from contact count")
    sizes = {
        "Poses": case["body_storage"] * layout["pose_words"],
        "SolverPoses": case["body_storage"] * layout["pose_words"],
        **{
            n: case["body_storage"] * layout["velocity_words"]
            for n in ("Velocities", "SolverVelocities", "Increments")
        },
        "WorldMass": case["body_storage"] * layout["world_words"],
        "LocalMass": case["body_storage"] * layout["local_words"],
        "Counts": case["body_storage"],
        "FirstSlot": case["body_storage"],
        "HubCounts": layout["hub_words"],
        "Tiles": case["tiles"] * layout["tile_words"],
        "OldTiles": case["old_tiles"] * layout["tile_words"],
        "RbdParams": layout["rbd_params_words"],
        "Gravity": 4,
        "BatchIds": layout["batch_words"],
        "Buckets": 4 * case["batches"],
        "Links": case["contact_storage"] * layout["link_words"],
        "Contacts": case["contact_storage"] * layout["contact_words"],
        "States": case["states"] * layout["state_words"],
        **{n: 2 for n in ("Color0", "Color1", "Color2", "Mode")},
        "Params": 6,
        "Plan": 2 * layout["plan_words"],
    }
    if set(arrays) != set(BUFFERS) or case["buffer_words"] != sizes:
        raise ValueError("Invalid TGS storage manifest")
    if any(a.dtype != np.uint32 or a.shape != (sizes[n],) for n, a in arrays.items()):
        raise ValueError("Invalid TGS buffer shape")
    # Packed Params is device output. Structural body validation uses finite
    # placeholder coefficients; native uniforms are validated independently below.
    body_inputs = {n: arrays[n] for n in BODY_BUFFERS}
    body_inputs["Params"] = np.array(
        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32
    ).view(np.uint32)
    validate_body(layout, body_case(case, "update_mprops"), body_inputs)
    validate_contact(layout, case, {n: arrays[n] for n in CONTACT_BUFFERS})
    mass = arrays["LocalMass"].view(np.float32).reshape(-1, layout["local_words"])
    invmass = mass[:, layout["local_inv_mass"] : layout["local_inv_mass"] + 3]
    if np.any(invmass < 0):
        raise ValueError("Negative frame inverse mass")
    p = arrays["RbdParams"].view(np.float32)
    f = layout["rbd_params"]
    floats = [
        f[n]
        for n in f
        if n
        not in (
            "num_solver_iterations",
            "num_internal_pgs_iterations",
            "friction_in_bias_pass",
        )
    ]
    if (
        not np.isfinite(p[floats]).all()
        or not np.isfinite(arrays["Gravity"].view(np.float32)[:3]).all()
    ):
        raise ValueError("Nonfinite frame uniform")
    if (
        p[f["dt"]] <= 0
        or p[f["length_unit"]] <= 0
        or p[f["normalized_max_linear_velocity"]] < 0
        or p[f["normalized_max_corrective_velocity"]] < 0
        or not 0 <= p[f["warmstart_coefficient"]] <= 1
    ):
        raise ValueError("Invalid frame parameters")
    if (
        int(arrays["RbdParams"][f["num_solver_iterations"]]) != case["substeps"]
        or bool(p[f["warmstart_coefficient"]] != 1) != case["scale_warmstart"]
    ):
        raise ValueError("Uniforms differ from frame schedule")
    for side in ("contact", "static_contact"):
        if p[f[side + "_damping_ratio"]] <= 0 or p[f[side + "_natural_frequency"]] < 0:
            raise ValueError("Invalid frame softness")
    if int(arrays["RbdParams"][f["num_internal_pgs_iterations"]]) != case[
        "pgs"
    ] or bool(arrays["RbdParams"][f["friction_in_bias_pass"]]) != (case["mode"] == 2):
        raise ValueError("PGS uniforms differ from frame schedule")
    if int(arrays["Mode"][0]) != case["mode"] or any(
        arrays[f"Color{c}"][0] != c for c in range(3)
    ):
        raise ValueError("Invalid frame color or mode")
    b = layout["batch"]
    nb = case["batches"]
    ids = arrays["BatchIds"]
    if (
        ids[b["num_batches"]] != nb
        or ids[b["bodies_len"]] * nb != case["body_count"]
        or ids[b["solver_color_buckets_stride"]] != 4
        or ids[b["colliders_batch_capacity"]] * nb != case["body_count"]
    ):
        raise ValueError("Invalid frame batch uniform")
    buckets = arrays["Buckets"].astype(np.int64)
    if (
        np.any(buckets[:nb])
        or np.any(buckets[1:] < buckets[:-1])
        or np.any(buckets[3 * nb :] != case["count"])
    ):
        raise ValueError("Invalid frame bucket ranges")
    links = arrays["Links"].reshape(-1, layout["link_words"])
    lf = layout["link"]
    for color in (1, 2):
        used = set()
        for i in range(
            int(buckets[color * nb - 1]), int(buckets[(color + 1) * nb - 1])
        ):
            link = links[i]
            if link[lf["len"]] == 0:
                raise ValueError("Inactive link in solved frame range")
            if (
                link[lf["previous_constraint"]] != 0xFFFFFFFF
                or link[lf["recycled"]] != 0
            ):
                raise ValueError("Frame contract requires fresh manifold links")
            source = int(link[lf["contact"]])
            manifold_len = arrays["Contacts"][
                source * layout["contact_words"]
                + layout["contact"]["contact"]
                + layout["manifold"]["len"]
            ]
            if manifold_len != link[lf["len"]]:
                raise ValueError("Frame link and manifold lengths differ")
            batch = int(
                np.searchsorted(buckets[color * nb : (color + 1) * nb], i, side="right")
            )
            for side in ("a", "b"):
                bid = int(link[lf[f"solver_body_{side}"]])
                slot = int(link[lf[f"vel_slot_{side}"]])
                if slot != bid or bid >= case["body_count"] or bid % nb != batch:
                    raise ValueError("Frame body or unsplit velocity slot out of range")
                if slot in used:
                    raise ValueError("Velocity writes overlap within a frame color")
                used.add(slot)
                if link.view(np.float32)[lf[f"mass_scale_{side}"]] != 1:
                    raise ValueError("Split masses outside frame contract")
    return write_masks(layout, case, arrays)


def write_masks(layout, case, before):
    import numpy as np
    from body_validation import write_masks as body_masks, BUFFERS as BODY_BUFFERS
    from contact_replay import output_masks

    masks = {n: np.zeros(a.size, dtype=np.uint8) for n, a in before.items()}
    for kind in ("update_mprops", "free_step"):
        for n, m in body_masks(
            layout, body_case(case, kind), {n: before[n] for n in BODY_BUFFERS}
        ).items():
            masks[n] = np.maximum(masks[n], m)
    mutable, floats = output_masks(layout, case, before)
    masks["Tiles"][mutable] = 2
    masks["Tiles"][floats] = 1
    for i in range(case["count"]):
        base = i // 32 * layout["tile_words"] + i % 32 * 4
        count = int(before["Links"][i * layout["link_words"] + layout["link"]["len"]])
        for k in range(count):
            masks["Tiles"][
                base
                + layout["points"]
                + k * layout["point_words"]
                + layout["point"]["arm_a"]
                + 3
            ] = 1
    masks["Params"][:] = 2
    masks["Params"][1:3] = 1
    return masks


def check_outputs(layout, case, before, actual, expected, contract):
    import numpy as np

    masks = validate_case(layout, case, before)
    errors = {}
    if set(actual) != set(before) or set(expected) != set(before):
        raise ValueError("Invalid returned frame buffers")
    for n, mask in masks.items():
        if (
            actual[n].dtype != np.uint32
            or actual[n].shape != before[n].shape
            or expected[n].shape != before[n].shape
        ):
            raise ValueError("Invalid returned frame shape")
        np.testing.assert_array_equal(
            actual[n][mask == 0],
            before[n][mask == 0],
            err_msg=f"Unexpected write to {n}",
        )
        np.testing.assert_array_equal(
            actual[n][mask == 2],
            expected[n][mask == 2],
            err_msg=f"Exact frame field {n}",
        )
        a, e = (
            actual[n].view(np.float32)[mask == 1],
            expected[n].view(np.float32)[mask == 1],
        )
        if not np.isfinite(a).all() or not np.isfinite(e).all():
            raise AssertionError("Nonfinite frame output")
        np.testing.assert_allclose(
            a,
            e,
            atol=contract["atol"],
            rtol=contract["rtol"],
            err_msg=f"Frame floats {n}",
        )
        errors[n] = float(np.max(np.abs(a - e))) if a.size else 0.0
    poses = (
        actual["Poses"]
        .view(np.float32)
        .reshape(-1, layout["pose_words"])[: case["body_count"]]
    )
    q = poses[:, layout["pose_rotation"] : layout["pose_rotation"] + 4]
    if np.any(np.abs(np.linalg.norm(q, axis=1) - 1) > 2e-6):
        raise AssertionError("Frame quaternion normalization")
    mass = (
        before["LocalMass"]
        .view(np.float32)
        .reshape(-1, layout["local_words"])[: case["body_count"]]
    )
    fixed = np.all(
        mass[:, layout["local_inv_mass"] : layout["local_inv_mass"] + 3] == 0, axis=1
    )
    if np.count_nonzero(fixed) != case["columns"] * case["batches"]:
        raise AssertionError("Fixture ground bodies must have zero inverse mass")
    initial_poses = (
        before["Poses"]
        .view(np.float32)
        .reshape(-1, layout["pose_words"])[: case["body_count"]]
    )
    for field, width in (("pose_translation", 3), ("pose_rotation", 4)):
        offset = layout[field]
        np.testing.assert_array_equal(
            poses[fixed, offset : offset + width],
            initial_poses[fixed, offset : offset + width],
            err_msg="Fixed pose changed",
        )
    v = (
        actual["Velocities"]
        .view(np.float32)
        .reshape(-1, layout["velocity_words"])[: case["body_count"]]
    )
    for offset in (layout["velocity_linear"], layout["velocity_angular"]):
        if np.any(v[fixed, offset : offset + 3] != 0):
            raise AssertionError("Fixed body moved")
    h, p = layout["header"], layout["point"]
    tf = actual["Tiles"].view(np.float32)
    for i in range(case["count"]):
        base = i // 32 * layout["tile_words"] + i % 32 * 4
        for k in range(int(actual["Tiles"][base + h["normal"] + 3])):
            pt = base + layout["points"] + k * layout["point_words"]
            normal = float(tf[pt + p["arm_a"] + 3])
            if normal < 0:
                raise AssertionError("Negative frame normal impulse")
            limit = normal * float(tf[base + h["tangent"] + 3])
            friction = tf[pt + p["friction"] : pt + p["friction"] + 2]
            if np.linalg.norm(friction) > limit + contract["atol"] + contract[
                "rtol"
            ] * abs(limit):
                raise AssertionError("Frame Coulomb friction disk")
    return errors


def main():
    from stage_replay import parse_args, run_stage
    from tgs_kernels import make_program

    run_stage(
        parse_args(OPERATIONS, "tgs_frame"),
        oracle="tgs",
        buffers=BUFFERS,
        make_kernel=None,
        make_program=make_program,
        validate_case=validate_case,
        check_outputs=check_outputs,
        launch=lambda c: (c["groups"], 64, 1),
    )


if __name__ == "__main__":
    main()
