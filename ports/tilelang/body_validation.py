"""Strict ABI and write-set checks for the body lifecycle contracts (no physics oracle)."""

BUFFERS = (
    "Velocities",
    "SolverVelocities",
    "Increments",
    "Poses",
    "SolverPoses",
    "LocalMass",
    "WorldMass",
    "Params",
    "Counts",
    "FirstSlot",
    "HubCounts",
)
OPERATIONS = (
    "cleanup",
    "init_bodies",
    "init_increments",
    "apply_increments",
    "finalize",
    "free_step",
)


def validate_case(layout, case, arrays):
    import numpy as np

    if case["kind"] not in OPERATIONS:
        raise ValueError("Unknown body operation")
    for name in (
        "count",
        "slots",
        "storage",
        "batches",
        "bodies_per_batch",
        "capacity_per_batch",
        "iterations",
        "substeps",
    ):
        if type(case[name]) is not int or not 0 <= case[name] <= 2**31 - 1:
            raise ValueError("Invalid body dimensions")
    if (
        case["count"] != case["bodies_per_batch"] * case["batches"]
        or case["slots"] != case["capacity_per_batch"] * case["batches"]
        or case["bodies_per_batch"] > case["capacity_per_batch"]
        or not case["count"] <= case["slots"] <= case["storage"]
        or case["storage"] == 0
        or not 1 <= case["iterations"] <= 1024
        or not 1 <= case["substeps"] <= 64
    ):
        raise ValueError("Invalid batch dimensions")
    groups = {
        "velocity": [("velocity_linear", 3), ("velocity_angular", 3)],
        "pose": [("pose_translation", 3), ("pose_rotation", 4), ("pose_padding", 1)],
        "local": [("local_com", 3)],
        "world": [("world_inv_mass", 3)],
        "hub": [("hub_hubs", 1), ("hub_slots", 1)],
    }
    for group, fields in groups.items():
        stride = layout[f"{group}_words"]
        if type(stride) is not int or not 1 <= stride <= 1024:
            raise ValueError("Invalid Rust layout stride")
        used = set()
        for field, width in fields:
            offset = layout[field]
            if type(offset) is not int or not 0 <= offset <= stride - width:
                raise ValueError("Invalid Rust layout offset")
            words = set(range(offset, offset + width))
            if used & words:
                raise ValueError("Overlapping Rust layout fields")
            used |= words
    storage = case["storage"]
    sizes = {
        "Velocities": storage * layout["velocity_words"],
        "SolverVelocities": storage * layout["velocity_words"],
        "Increments": storage * layout["velocity_words"],
        "Poses": storage * layout["pose_words"],
        "SolverPoses": storage * layout["pose_words"],
        "LocalMass": storage * layout["local_words"],
        "WorldMass": storage * layout["world_words"],
        "Params": 6,
        "Counts": storage,
        "FirstSlot": storage,
        "HubCounts": layout["hub_words"],
    }
    if set(arrays) != set(BUFFERS) or any(
        a.dtype != np.uint32 or a.ndim != 1 or a.size != sizes[n] or a.size > 2**31 - 1
        for n, a in arrays.items()
    ):
        raise ValueError("Invalid body buffer sizes or types")
    params = arrays["Params"].view(np.float32)
    if not np.isfinite(params).all() or params[0] <= 0 or np.any(params[1:3] < 0):
        raise ValueError("Invalid body parameters")
    for name, group in (
        ("Velocities", "velocity"),
        ("SolverVelocities", "velocity"),
        ("Increments", "velocity"),
        ("Poses", "pose"),
        ("SolverPoses", "pose"),
        ("LocalMass", "local"),
        ("WorldMass", "world"),
    ):
        data = arrays[name].view(np.float32).reshape(storage, layout[f"{group}_words"])
        for field, width in groups[group]:
            if field == "pose_padding":
                continue
            if not np.isfinite(data[:, layout[field] : layout[field] + width]).all():
                raise ValueError("Non-finite body input")
        if group == "pose":
            q = data[:, layout["pose_rotation"] : layout["pose_rotation"] + 4]
            if np.any(np.abs(np.linalg.norm(q, axis=1) - 1) > 2e-6):
                raise ValueError("Non-unit input rotation")
    return write_masks(layout, case, arrays)


def write_masks(layout, case, arrays):
    """0 = unchanged bits, 1 = f32 tolerance, 2 = exact output bits."""
    import numpy as np

    masks = {n: np.zeros(a.size, dtype=np.uint8) for n, a in arrays.items()}
    n, slots, kind = case["count"], case["slots"], case["kind"]

    def velocities(name, count, mode):
        data = masks[name].reshape(case["storage"], layout["velocity_words"])
        for field in ("velocity_linear", "velocity_angular"):
            data[:count, layout[field] : layout[field] + 3] = mode

    def poses(name, rotating):
        data = masks[name].reshape(case["storage"], layout["pose_words"])
        data[:n, layout["pose_translation"] : layout["pose_translation"] + 3] = 1
        data[:n, layout["pose_rotation"] : layout["pose_rotation"] + 4] = (
            1 if rotating else 2
        )
        data[:n, layout["pose_padding"]] = 2

    if kind in ("cleanup", "free_step"):
        velocities("SolverVelocities", slots, 2)
        masks["Counts"][:slots] = 2
        masks["FirstSlot"][:slots] = 2
        masks["HubCounts"][[layout["hub_hubs"], layout["hub_slots"]]] = 2
    if kind in ("init_increments", "free_step"):
        velocities("Increments", n, 1)
    if kind in ("apply_increments", "free_step"):
        velocities("SolverVelocities", n, 1)
    if kind in ("init_bodies", "free_step"):
        poses("SolverPoses", kind == "free_step")
    if kind in ("finalize", "free_step"):
        velocities("Velocities", n, 1 if kind == "free_step" else 2)
        poses("Poses", kind == "free_step")
    return masks


def check_outputs(layout, case, before, actual, expected, contract):
    import numpy as np

    masks = validate_case(layout, case, before)
    if set(contract["inputs"]) != set(BUFFERS):
        raise ValueError("Invalid body contract buffers")
    errors = {}
    for name, mask in masks.items():
        if (
            actual[name].shape != before[name].shape
            or actual[name].dtype != np.uint32
            or expected[name].shape != before[name].shape
        ):
            raise ValueError("Invalid body output size or type")
        if mask.any() and name not in contract["outputs"]:
            raise ValueError("Body write set missing from contract")
        np.testing.assert_array_equal(
            actual[name][mask == 0],
            before[name][mask == 0],
            err_msg=f"Unexpected write to {name}",
        )
        np.testing.assert_array_equal(
            actual[name][mask == 2],
            expected[name][mask == 2],
            err_msg=f"Exact output mismatch in {name}",
        )
        a, e = (
            actual[name].view(np.float32)[mask == 1],
            expected[name].view(np.float32)[mask == 1],
        )
        if not np.isfinite(a).all() or not np.isfinite(e).all():
            raise AssertionError(f"Non-finite {name} output")
        np.testing.assert_allclose(
            a,
            e,
            atol=contract["atol"],
            rtol=contract["rtol"],
            err_msg=f"{case['name']}: {name}",
        )
        errors[name] = float(np.max(np.abs(a - e))) if a.size else 0.0
    if case["kind"] == "free_step":
        n = case["count"]
        data = actual["Poses"].view(np.float32).reshape(-1, layout["pose_words"])
        q = data[:n, layout["pose_rotation"] : layout["pose_rotation"] + 4]
        if np.any(np.abs(np.linalg.norm(q, axis=1) - 1) > 2e-6):
            raise AssertionError("Quaternion normalization invariant")
        v = actual["Velocities"].view(np.float32).reshape(-1, layout["velocity_words"])
        for field, limit in zip(
            ("velocity_linear", "velocity_angular"),
            before["Params"].view(np.float32)[1:3],
        ):
            speed = np.linalg.norm(v[:n, layout[field] : layout[field] + 3], axis=1)
            if np.any(speed > limit + contract["atol"] + contract["rtol"] * abs(limit)):
                raise AssertionError("Velocity cap invariant")
    return errors
