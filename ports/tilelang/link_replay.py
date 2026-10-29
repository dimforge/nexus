#!/usr/bin/env python3
"""Native contact-link buffers, strict input guards, and live Rust parity."""

from stage_replay import parse_args, run_stage

BUFFERS = (
    "Contacts",
    "Links",
    "BodyIsMultibody",
    "OldCounts",
    "OldIds",
    "OldLinks",
    "OldIndices",
    "States",
    "ColliderPoses",
    "Params",
    "OldColors",
    "Colors",
    "Colored",
)
OPERATIONS = ("init_links", "match_links", "seed_colors", "contact_persistence")
OUTPUTS = {
    "init_links": ("Links",),
    "match_links": ("Links", "States"),
    "seed_colors": ("Colors", "Colored"),
    "contact_persistence": ("Links", "States", "Colors", "Colored"),
}


def validate_case(layout, case, arrays):
    import numpy as np
    from contact_replay import validate_layout

    validate_layout(layout)

    def fields(stride, items):
        if type(stride) is not int or not 0 < stride < 2**20:
            raise ValueError("Invalid link ABI stride")
        used = set()
        for off, width in items:
            if type(off) is not int or not 0 <= off <= stride - width:
                raise ValueError("Invalid link ABI offset")
            words = set(range(off, off + width))
            if words & used:
                raise ValueError("Overlapping link ABI fields")
            used |= words

    fields(
        layout["state_words"],
        [
            (
                v,
                layout["pose_words"]
                if k in ("pose_a", "pose_b")
                else 2
                if k == "colliders"
                else 1,
            )
            for k, v in layout["state"].items()
        ],
    )
    fields(layout["params_words"], [(v, 1) for v in layout["recycle_params"].values()])
    if layout["state_pose_a"] != layout["state"]["pose_a"]:
        raise ValueError("Inconsistent link state ABI")
    for key in (
        "count",
        "storage",
        "bodies",
        "colliders",
        "old_storage",
        "ids_storage",
        "state_storage",
        "old_base",
        "new_base",
        "iterations",
    ):
        if type(case[key]) is not int or not 0 <= case[key] < 2**31:
            raise ValueError("Invalid link dimensions")
    if (
        case["kind"] not in OPERATIONS
        or case["count"] > case["storage"]
        or not 1 <= case["iterations"] <= 1024
        or min(
            case[k]
            for k in (
                "storage",
                "bodies",
                "colliders",
                "old_storage",
                "ids_storage",
                "state_storage",
            )
        )
        < 1
        or case["old_base"] + case["old_storage"] > case["state_storage"]
        or case["new_base"] + case["storage"] > case["state_storage"]
    ):
        raise ValueError("Invalid link dimensions")
    if not (
        case["old_base"] + case["old_storage"] <= case["new_base"]
        or case["new_base"] + case["storage"] <= case["old_base"]
    ):
        raise ValueError("Old and new recycling states overlap")
    sizes = {
        "Contacts": case["storage"] * layout["contact_words"],
        "Links": case["storage"] * layout["link_words"],
        "BodyIsMultibody": case["bodies"],
        "OldCounts": case["bodies"],
        "OldIds": case["ids_storage"],
        "OldLinks": case["old_storage"] * layout["link_words"],
        "OldIndices": case["old_storage"],
        "States": case["state_storage"] * layout["state_words"],
        "ColliderPoses": case["colliders"] * layout["pose_words"],
        "Params": layout["params_words"],
        "OldColors": case["old_storage"],
        "Colors": case["storage"],
        "Colored": case["storage"],
    }
    if set(arrays) != set(BUFFERS) or any(
        a.dtype != np.uint32 or a.shape != (sizes[k],) or a.size >= 2**31
        for k, a in arrays.items()
    ):
        raise ValueError("Invalid link buffer sizes")
    l, c, m, s = (layout[k] for k in ("link", "contact", "manifold", "state"))
    contacts = arrays["Contacts"].reshape(-1, layout["contact_words"])
    links = arrays["Links"].reshape(-1, layout["link_words"])
    init = case["kind"] in ("init_links", "contact_persistence")
    match = case["kind"] in ("match_links", "contact_persistence")

    def pose(a, base):
        q = a.view(np.float32)[
            base + layout["pose_rotation"] : base + layout["pose_rotation"] + 4
        ]
        t = a.view(np.float32)[
            base + layout["pose_translation"] : base + layout["pose_translation"] + 3
        ]
        if (
            not np.isfinite(q).all()
            or not np.isfinite(t).all()
            or abs(np.linalg.norm(q) - 1) > 2e-6
        ):
            raise ValueError("Invalid recycling pose")

    active = []
    for i in range(case["count"]):
        im = contacts[i]
        link = links[i]
        if init:
            bodies = im[c["bodies"] : c["bodies"] + 2]
            if np.any(bodies >= case["bodies"]):
                raise ValueError("Invalid contact body")
            length = int(im[c["contact"] + m["len"]])
            if length > 4:
                raise ValueError("Invalid contact length")
            length = 0 if any(arrays["BodyIsMultibody"][b] for b in bodies) else length
        else:
            length = int(link[l["len"]])
            if length > 4:
                raise ValueError("Invalid link length")
        if length == 0:
            continue
        active.append(i)
        if match:
            if not init:
                bodies = link[[l["solver_body_a"], l["solver_body_b"]]]
                if np.any(bodies >= case["bodies"]):
                    raise ValueError("Invalid active link body")
            colliders = im[c["colliders"] : c["colliders"] + 2]
            if np.any(colliders >= case["colliders"]):
                raise ValueError("Invalid contact collider")
            for collider in colliders:
                pose(arrays["ColliderPoses"], int(collider) * layout["pose_words"])
            if not np.isfinite(im.view(np.float32)[c["recycle_extent"]]):
                raise ValueError("Invalid recycle extent")
        if case["kind"] == "seed_colors":
            previous = int(link[l["previous_constraint"]])
            if previous != 2**32 - 1 and previous >= case["old_storage"]:
                raise ValueError("Invalid previous color index")
    if match and active:
        counts = arrays["OldCounts"].astype(np.uint64)
        if np.any(counts[1:] < counts[:-1]) or counts[-1] > case["ids_storage"]:
            raise ValueError("Invalid old adjacency counts")
        ids = arrays["OldIds"][: int(counts[-1])]
        if np.any(ids >= case["old_storage"]):
            raise ValueError("Invalid old adjacency ID")
        params = arrays["Params"].view(np.float32)
        distance = params[layout["recycle_params"]["normalized_distance"]]
        unit = params[layout["recycle_params"]["length_unit"]]
        if (
            not np.isfinite([distance, unit]).all()
            or distance < 0
            or unit <= 0
            or not np.isfinite(np.float64(distance) * unit)
            or np.float64(distance) * unit > np.finfo(np.float32).max
        ):
            raise ValueError("Invalid recycle parameters")
        for cid in np.unique(ids):
            base = (case["old_base"] + int(cid)) * layout["state_words"]
            pose(arrays["States"], base + s["pose_a"])
            pose(arrays["States"], base + s["pose_b"])
            values = arrays["States"].view(np.float32)[
                [base + s["max_extent"], base + s["max_drift"]]
            ]
            if not np.isfinite(values).all() or values[1] < 0:
                raise ValueError("Invalid old recycling limits")


def output_masks(layout, case, before):
    import numpy as np

    masks = {n: np.zeros(a.size, dtype=bool) for n, a in before.items()}
    l, c, m = (layout[k] for k in ("link", "contact", "manifold"))
    init = case["kind"] in ("init_links", "contact_persistence")
    match = case["kind"] in ("match_links", "contact_persistence")
    seed = case["kind"] in ("seed_colors", "contact_persistence")
    for i in range(case["count"]):
        lb = i * layout["link_words"]
        if init:
            masks["Links"][lb : lb + layout["link_words"]] = True
            cb = i * layout["contact_words"]
            a, b = before["Contacts"][cb + c["bodies"] : cb + c["bodies"] + 2]
            active = before["Contacts"][cb + c["contact"] + m["len"]] != 0 and not (
                before["BodyIsMultibody"][a] | before["BodyIsMultibody"][b]
            )
        else:
            active = before["Links"][lb + l["len"]] != 0
        if not active:
            continue
        if match:
            for field in (
                "previous_constraint",
                "previous_constraint_index",
                "recycled",
            ):
                masks["Links"][lb + l[field]] = True
            base = (case["new_base"] + i) * layout["state_words"]
            masks["States"][base : base + layout["state_words"]] = True
        if seed:
            masks["Colors"][i] = True
            masks["Colored"][i] = True
    return masks


def check_outputs(layout, case, before, actual, expected, contract):
    import numpy as np

    validate_case(layout, case, before)
    masks = output_masks(layout, case, before)
    errors = {}
    for name in BUFFERS:
        a, e = actual[name], expected[name]
        if (
            a.dtype != np.uint32
            or e.dtype != np.uint32
            or a.shape != before[name].shape
            or e.shape != a.shape
        ):
            raise ValueError("Invalid link output size")
        np.testing.assert_array_equal(
            a[~masks[name]],
            before[name][~masks[name]],
            err_msg=f"Unexpected {name} write",
        )
        np.testing.assert_array_equal(a, e, err_msg=f"{case['name']}: {name}")
        errors[name] = 0.0
    return errors


def main():
    from link_kernels import make_kernel

    run_stage(
        parse_args(OPERATIONS, "contact_persistence"),
        oracle="links",
        buffers=BUFFERS,
        make_kernel=make_kernel,
        validate_case=validate_case,
        check_outputs=check_outputs,
        launch=lambda case: (
            max((case["count"] + 63) // 64, 1),
            64,
            case["iterations"],
        ),
        extra_files=("ContactPlan.bin", "RecycleOffsets.bin"),
    )


if __name__ == "__main__":
    main()
