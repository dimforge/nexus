#!/usr/bin/env python3
"""Live Rust warmstart replay with bounded graph inputs and exact write guards."""

from stage_replay import parse_args, run_stage

BUFFERS = (
    "Tiles",
    "Velocities",
    "Counts",
    "Ids",
    "Indices",
    "HubFirstSlot",
    "HubSlotConstraint",
    "HubCounts",
    "Params",
    "Buckets",
    "BatchIds",
)
OPERATIONS = ("scale_impulses", "gather_warmstart", "hub_warmstart")


def validate_case(layout, case, arrays):
    import numpy as np
    from contact_replay import validate_layout

    validate_layout(layout)
    for stride, offsets in (
        (layout["hub_words"], layout["hub"].values()),
        (layout["batch_words"], layout["batch"].values()),
        (layout["params_words"], [layout["warmstart_coefficient"]]),
    ):
        if type(stride) is not int or not 0 < stride < 2**20:
            raise ValueError("Invalid warmstart ABI stride")
        offsets = list(offsets)
        if len(set(offsets)) != len(offsets) or any(
            type(o) is not int or not 0 <= o < stride for o in offsets
        ):
            raise ValueError("Invalid warmstart ABI fields")
    keys = (
        "count",
        "storage",
        "bodies",
        "constraints",
        "tiles",
        "velocities",
        "ids_storage",
        "hub_storage",
        "bucket_storage",
        "dispatch",
        "iterations",
    )
    if any(type(case[k]) is not int or not 0 <= case[k] < 2**31 for k in keys):
        raise ValueError("Invalid warmstart dimensions")
    if (
        case["kind"] not in OPERATIONS
        or case["count"] > case["constraints"]
        or case["count"] > case["tiles"] * 32
        or case["bodies"] > case["storage"]
        or not 1 <= case["iterations"] <= 1024
        or any(case[k] < 1 for k in keys if k not in ("count", "bodies"))
    ):
        raise ValueError("Invalid warmstart dimensions")
    sizes = {
        "Tiles": case["tiles"] * layout["tile_words"],
        "Velocities": case["velocities"] * layout["velocity_words"],
        "Counts": case["storage"],
        "Ids": case["ids_storage"],
        "Indices": case["constraints"],
        "HubFirstSlot": case["storage"],
        "HubSlotConstraint": case["hub_storage"],
        "HubCounts": layout["hub_words"],
        "Params": layout["params_words"],
        "Buckets": case["bucket_storage"],
        "BatchIds": layout["batch_words"],
    }
    if set(arrays) != set(BUFFERS) or any(
        a.dtype != np.uint32 or a.shape != (sizes[k],) or a.size >= 2**31
        for k, a in arrays.items()
    ):
        raise ValueError("Invalid warmstart buffer sizes")
    batch = {k: int(arrays["BatchIds"][v]) for k, v in layout["batch"].items()}
    hc = {k: int(arrays["HubCounts"][v]) for k, v in layout["hub"].items()}
    if (
        batch["num_batches"] == 0
        or any(v >= 2**31 for v in batch.values())
        or batch["num_batches"] * batch["bodies_len"] != case["bodies"]
        or batch["solver_color_buckets_stride"] == 0
        or batch["num_batches"] * batch["solver_color_buckets_stride"]
        != case["bucket_storage"]
        or any(v >= 2**31 for v in hc.values())
        or hc["base"] < case["bodies"]
        or hc["base"] + hc["pool"] > case["velocities"]
        or hc["pool"] > case["hub_storage"]
    ):
        raise ValueError("Invalid warmstart uniforms")
    live = min(hc["slots"], hc["pool"])
    if case["dispatch"] < max(case["count"], case["bodies"], live):
        raise ValueError("Warmstart dispatch does not cover active ranges")
    buckets = arrays["Buckets"].astype(np.uint64)
    if np.any(buckets[1:] < buckets[:-1]) or buckets[-1] != case["count"]:
        raise ValueError("Invalid warmstart buckets")
    counts = arrays["Counts"].astype(np.uint64)
    if np.any(counts[1:] < counts[:-1]) or counts[-1] > case["ids_storage"]:
        raise ValueError("Invalid warmstart adjacency counts")
    coefficient = arrays["Params"].view(np.float32)[layout["warmstart_coefficient"]]
    if not np.isfinite(coefficient) or not 0 <= coefficient <= 1:
        raise ValueError("Invalid warmstart coefficient")
    h, p = layout["header"], layout["point"]

    def finite(a):
        if not np.isfinite(a).all():
            raise ValueError("Non-finite warmstart input")

    def mapped(cid):
        if cid >= case["constraints"]:
            raise ValueError("Invalid constraint ID")
        index = int(arrays["Indices"][cid])
        if index >= case["count"]:
            raise ValueError("Invalid constraint index")
        return index

    def tile(index):
        base = index // 32 * layout["tile_words"] + index % 32 * 4
        count = int(arrays["Tiles"][base + h["normal"] + 3])
        if count > 4:
            raise ValueError("Invalid warmstart point count")
        f = arrays["Tiles"].view(np.float32)
        for off in h.values():
            finite(f[base + off : base + off + 3])
        for off in ("normal", "tangent"):
            if abs(np.linalg.norm(f[base + h[off] : base + h[off] + 3]) - 1) > 2e-6:
                raise ValueError("Invalid warmstart frame")
        if (
            abs(
                np.dot(
                    f[base + h["normal"] : base + h["normal"] + 3],
                    f[base + h["tangent"] : base + h["tangent"] + 3],
                )
            )
            > 2e-6
        ):
            raise ValueError("Invalid warmstart frame")
        for k in range(count):
            pb = base + layout["points"] + k * layout["point_words"]
            for off in p.values():
                finite(f[pb + off : pb + off + 4])
        finite(f[base + layout["coupling"] : base + layout["coupling"] + count])
        return base

    def velocity(slot):
        f = arrays["Velocities"].view(np.float32)
        for key in ("velocity_linear", "velocity_angular"):
            off = slot * layout["velocity_words"] + layout[key]
            finite(f[off : off + 3])

    if case["kind"] == "scale_impulses":
        for i in range(case["count"]):
            base = tile(i)
            if any(
                int(arrays["Tiles"][base + h[k] + 3]) >= case["bodies"]
                for k in ("body_a", "body_b")
            ):
                raise ValueError("Invalid warmstart body tag")
    elif case["kind"] == "gather_warmstart":
        for body in range(case["bodies"]):
            if arrays["HubFirstSlot"][body] != 2**32 - 1:
                if arrays["HubFirstSlot"][body] >= live:
                    raise ValueError("Invalid first hub slot")
                continue
            velocity(body)
            start = 0 if body == 0 else int(counts[body - 1])
            for cid in arrays["Ids"][start : int(counts[body])]:
                index = mapped(int(cid))
                pair = (
                    index // 32 * layout["tile_words"]
                    + layout["warmstart"]
                    + index % 32 * 2 * layout["velocity_words"]
                )
                side = (
                    0
                    if arrays["Tiles"][pair + layout["velocity_padding1"]] == body
                    else 1
                )
                source = pair + side * layout["velocity_words"]
                if arrays["Tiles"][source + layout["velocity_padding1"]] != body:
                    raise ValueError(
                        "Cached warmstart tag does not match adjacency body"
                    )
                for key in ("velocity_linear", "velocity_angular"):
                    off = source + layout[key]
                    finite(arrays["Tiles"].view(np.float32)[off : off + 3])
    else:
        for slot in range(live):
            vel_slot = hc["base"] + slot
            index = mapped(int(arrays["HubSlotConstraint"][slot]))
            base = tile(index)
            if vel_slot not in [
                int(arrays["Tiles"][base + h[k] + 3]) for k in ("diag_a", "off_a")
            ]:
                raise ValueError("Constraint does not own hub velocity slot")
            velocity(vel_slot)


def output_masks(layout, case, before):
    import numpy as np

    mutable = {k: np.zeros(a.size, dtype=bool) for k, a in before.items()}
    floats = {k: m.copy() for k, m in mutable.items()}
    h, p = layout["header"], layout["point"]
    if case["kind"] == "scale_impulses":
        for i in range(case["count"]):
            base = i // 32 * layout["tile_words"] + i % 32 * 4
            count = int(before["Tiles"][base + h["normal"] + 3])
            for k in ("diag_b", "off_b"):
                mutable["Tiles"][base + h[k] + 3] = True
            mutable["Tiles"][
                base + layout["coupling"] + count : base + layout["coupling"] + 4
            ] = True
            for k in range(count):
                point = base + layout["points"] + k * layout["point_words"]
                for off in (p["arm_a"] + 3, p["friction"], p["friction"] + 1):
                    mutable["Tiles"][point + off] = True
                    floats["Tiles"][point + off] = True
            cache = (
                i // 32 * layout["tile_words"]
                + layout["warmstart"]
                + i % 32 * 2 * layout["velocity_words"]
            )
            mutable["Tiles"][cache : cache + 2 * layout["velocity_words"]] = True
            for side in (0, 1):
                for field in ("velocity_linear", "velocity_angular"):
                    off = cache + side * layout["velocity_words"] + layout[field]
                    floats["Tiles"][off : off + 3] = True
    else:
        if case["kind"] == "gather_warmstart":
            slots = [
                i
                for i in range(case["bodies"])
                if before["HubFirstSlot"][i] == 2**32 - 1
            ]
        else:
            hc = {k: int(before["HubCounts"][v]) for k, v in layout["hub"].items()}
            slots = range(hc["base"], hc["base"] + min(hc["slots"], hc["pool"]))
        for slot in slots:
            for field in ("velocity_linear", "velocity_angular"):
                off = slot * layout["velocity_words"] + layout[field]
                mutable["Velocities"][off : off + 3] = True
                floats["Velocities"][off : off + 3] = True
    return mutable, floats


def check_outputs(layout, case, before, actual, expected, contract):
    import numpy as np

    validate_case(layout, case, before)
    mutable, floats = output_masks(layout, case, before)
    errors = {}
    for name in BUFFERS:
        a, e = actual[name], expected[name]
        if (
            a.dtype != np.uint32
            or e.dtype != np.uint32
            or a.shape != before[name].shape
            or e.shape != a.shape
        ):
            raise ValueError("Invalid warmstart output size")
        np.testing.assert_array_equal(
            a[~mutable[name]],
            before[name][~mutable[name]],
            err_msg=f"Unexpected {name} write",
        )
        np.testing.assert_array_equal(e[~mutable[name]], before[name][~mutable[name]])
        np.testing.assert_array_equal(a[~floats[name]], e[~floats[name]])
        af, ef = a.view(np.float32)[floats[name]], e.view(np.float32)[floats[name]]
        if not np.isfinite(af).all() or not np.isfinite(ef).all():
            raise AssertionError("Non-finite warmstart output")
        np.testing.assert_allclose(af, ef, atol=contract["atol"], rtol=contract["rtol"])
        errors[name] = float(np.max(np.abs(af - ef))) if af.size else 0.0
    return errors


def main():
    from warmstart_kernels import make_kernel

    run_stage(
        parse_args(OPERATIONS, "gather_warmstart"),
        oracle="warmstarts",
        buffers=BUFFERS,
        make_kernel=make_kernel,
        validate_case=validate_case,
        check_outputs=check_outputs,
        launch=lambda case: (
            max((case["dispatch"] + 63) // 64, 1),
            64,
            case["iterations"],
        ),
    )


if __name__ == "__main__":
    main()
