"""Field policies for native twist preparation and cached geometry."""


def validate_old(l, raw, b, n):
    import numpy as np

    t, h = l["twist"], l["header"]
    f = raw.view(np.float32)
    fields = [b + h["off_b"] + 3]
    for key in ("offset_b", "center_a", "friction", "frame_a", "frame_b"):
        fields.extend(range(b + t[key], b + t[key] + 4))
    for k in range(n):
        fields.extend(
            range(
                b + t["arm_a"] + k * l["twist_arm_words"],
                b + t["arm_a"] + k * l["twist_arm_words"] + 4,
            )
        )
        fields.extend(
            b + t[key] + k for key in ("normal_impulse", "dist", "normal_vel", "radius")
        )
    if not np.isfinite(f[fields]).all():
        raise ValueError("Nonfinite old twist constraint")
    for key in ("frame_a", "frame_b"):
        if abs(np.linalg.norm(f[b + t[key] : b + t[key] + 4]) - 1) > 2e-6:
            raise ValueError("Invalid old twist frozen frame")


def output_masks(l, c, before):
    import numpy as np

    mutable = np.zeros(before["Tiles"].size, bool)
    floats = mutable.copy()
    link, contact, manifold, h, t = (
        l[k] for k in ("link", "contact", "manifold", "header", "twist")
    )
    for i in range(c["count"]):
        ln = before["Links"][i * l["link_words"] : (i + 1) * l["link_words"]]
        if not ln[link["len"]]:
            continue
        b = i // 32 * l["tile_words"] + i % 32 * 4
        j = int(ln[link["contact"]])
        count = int(
            before["Contacts"][
                j * l["contact_words"] + contact["contact"] + manifold["len"]
            ]
        )
        recycled = bool(ln[link["recycled"]])
        if recycled:
            old = int(ln[link["previous_constraint_index"]])
            ob = old // 32 * l["tile_words"] + old % 32 * 4
            count = int(before["OldTiles"][ob + h["normal"] + 3])
        for key, off in h.items():
            mutable[b + off : b + off + 4] = True
            floats[b + off : b + off + 3] = not (
                recycled and key in ("normal", "tangent")
            )
        floats[b + h["diag_b"] + 3] = bool(count)
        floats[b + h["off_b"] + 3] = not recycled
        for key in (
            "offset_b",
            "center_a",
            "friction",
            "frame_a",
            "frame_b",
            "normal_impulse",
            "dist",
            "normal_vel",
            "radius",
        ):
            mutable[b + t[key] : b + t[key] + 4] = True
        floats[b + t["offset_b"] : b + t["offset_b"] + 3] = not recycled
        floats[b + t["offset_b"] + 3] = bool(count)
        floats[b + t["center_a"] : b + t["center_a"] + 4] = bool(count)
        floats[b + t["friction"] : b + t["friction"] + 2] = not recycled
        floats[b + t["friction"] + 2] = bool(count)
        floats[b + t["normal_vel"] : b + t["normal_vel"] + count] = True
        floats[b + t["radius"] : b + t["radius"] + count] = True
        for k in range(count):
            p = b + t["arm_a"] + k * l["twist_arm_words"]
            mutable[p : p + 4] = True
            floats[p : p + 3] = not recycled
            floats[p + 3] = True
        cache = (
            i // 32 * l["tile_words"]
            + l["warmstart"]
            + i % 32 * 2 * l["velocity_words"]
        )
        mutable[cache : cache + 2 * l["velocity_words"]] = True
        for side in range(2):
            for field in ("velocity_linear", "velocity_angular"):
                off = cache + side * l["velocity_words"] + l[field]
                floats[off : off + 3] = True
    return mutable, floats
