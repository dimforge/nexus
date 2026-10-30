"""LBVH ABI, independent tree geometry/topology, and pair allocation checks."""

import numpy as np
from lbvh_pair_validation import check as check_pairs
from support_geometry import inverse, support_value
from support_validation import validate_shapes

OPERATIONS = (
    "lbvh_domain",
    "lbvh_domain_merge",
    "lbvh_morton",
    "lbvh_build",
    "lbvh_escapes",
    "lbvh_leaves",
    "lbvh_refit_chunks",
    "lbvh_refit_frontier",
    "lbvh_pairs",
    "lbvh_refit_plan",
    "lbvh_reset",
    "flat_list_dispatch",
)
BUFFERS = (
    "Pairs",
    "Groups",
    "Filters",
    "Poses",
    "Shapes",
    "Vertices",
    "Partials",
    "Domain",
    "Keys",
    "Sorted",
    "Tree",
    "Frontier",
    "FrontierLen",
    "Rounds",
    "PairCount",
    "Indirect",
    "BatchIds",
)


def sizes(l, c):
    _n, nb, cap, storage, pc = (
        c["colliders"],
        c["batches"],
        c["capacity"],
        c["storage"],
        c["pair_capacity"],
    )
    return {
        "Pairs": (pc + 3) * l["pair_words"],
        "Groups": storage * l["group_words"],
        "Filters": storage * 2,
        "Poses": storage * l["pose_words"],
        "Shapes": storage * l["shape_words"],
        "Vertices": 6 * l["vertex_words"],
        "Partials": (64 * nb + 3) * l["aabb_words"],
        "Domain": (nb + 3) * l["aabb_words"],
        "Keys": cap * nb + 3,
        "Sorted": cap * nb + 3,
        "Tree": (2 * cap * nb + 3) * l["lbvh_node_words"],
        "Frontier": cap * nb + 3,
        "FrontierLen": nb + 3,
        "Rounds": 3,
        "PairCount": 3,
        "Indirect": 6,
        "BatchIds": l["batch_words"],
    }


def topology(l, c, r, escapes=False):
    n, nb, cap = c["colliders"], c["batches"], c["capacity"]
    f = l["lbvh_node"]
    w = l["lbvh_node_words"]
    trees = r["Tree"].reshape(-1, w)
    for batch in range(nb):
        tree = trees[2 * batch * cap : 2 * batch * cap + max(2 * n - 1, 1)]
        if not n:
            if escapes and tree[0, f["escape"]] != 0xFFFFFFFF:
                raise AssertionError("Empty tree escape")
            continue
        seen = set()
        ranges = {}
        stack = [(0, False, 0xFFFFFFFF)]
        while stack:
            node, done, escape = stack.pop()
            if node >= 2 * n - 1:
                raise AssertionError("Tree child out of range")
            row = tree[node]
            if not done:
                if node in seen:
                    raise AssertionError("Repeated or cyclic tree node")
                seen.add(node)
                if escapes and row[f["escape"]] != escape:
                    raise AssertionError("Invalid stackless escape")
                if node >= n - 1:
                    ranges[node] = (node - (n - 1), node - (n - 1))
                    continue
                left, right = map(int, row[[f["left"], f["right"]]])
                if (
                    left == right
                    or min(left, right) == 0
                    or max(left, right) >= 2 * n - 1
                ):
                    raise AssertionError("Invalid internal children")
                if tree[left, f["parent"]] != node or tree[right, f["parent"]] != node:
                    raise AssertionError("Invalid parent backlink")
                stack.extend(
                    [(node, True, escape), (right, False, escape), (left, False, right)]
                )
            else:
                left, right = map(int, row[[f["left"], f["right"]]])
                first, mid = ranges[left]
                mid2, last = ranges[right]
                if (
                    mid + 1 != mid2
                    or row[f["first_leaf"]] != first
                    or row[f["last_leaf"]] != last
                ):
                    raise AssertionError("Invalid sorted leaf range")
                ranges[node] = (first, last)
        if len(seen) != 2 * n - 1 or ranges[0] != (0, n - 1):
            raise AssertionError("Incomplete tree coverage")


def validate_case(l, c, b):
    fields = (
        "count",
        "colliders",
        "batches",
        "capacity",
        "storage",
        "groups",
        "iterations",
        "pair_capacity",
    )
    if c["kind"] not in OPERATIONS or any(
        type(c[k]) is not int or not 0 <= c[k] < 2**27 for k in fields
    ):
        raise ValueError("Invalid LBVH dimensions")
    n, nb, cap = c["colliders"], c["batches"], c["capacity"]
    if (
        not 1 <= nb <= 64
        or not 1 <= c["groups"] <= 64
        or c["iterations"] != 1
        or c["count"] != n * nb
        or cap != n + 3
        or c["storage"] != cap * nb + 3
    ):
        raise ValueError("Invalid LBVH batch/dispatch layout")
    expected = sizes(l, c)
    if (
        set(b) != set(BUFFERS)
        or c["buffer_words"] != expected
        or any(
            v.dtype != np.uint32 or v.shape != (expected[name],)
            for name, v in b.items()
        )
    ):
        raise ValueError("Invalid LBVH storage ABI")
    for field, value in [
        ("num_batches", nb),
        ("colliders_len", n),
        ("colliders_batch_capacity", cap),
        ("collision_pairs_capacity", c["pair_capacity"]),
    ]:
        if b["BatchIds"][l["batch"][field]] != value:
            raise ValueError("LBVH uniform mismatch")
    poses = b["Poses"].view(np.float32).reshape(-1, l["pose_words"])[: n * nb]
    p = poses[:, l["pose_translation"] : l["pose_translation"] + 3]
    q = poses[:, l["pose_rotation"] : l["pose_rotation"] + 4]
    if (
        not np.isfinite(np.c_[p, q]).all()
        or np.any(abs(p) > 1e6)
        or not np.allclose(np.sum(q * q, axis=1), 1, atol=2e-6, rtol=0)
    ):
        raise ValueError("Invalid LBVH poses")
    shapes = b["Shapes"].reshape(-1, l["shape_words"])
    vertices = (
        b["Vertices"]
        .view(np.float32)
        .reshape(-1, l["vertex_words"])[:, l["vertex_vector"] : l["vertex_vector"] + 3]
        .astype(float)
    )
    sf, types = l["shape"], l["shape_types"]
    tags = shapes[:, sf["a"] + 3]
    composite = (tags == types["trimesh"]) | (tags == types["polyline"])
    if not np.isfinite(vertices).all():
        raise ValueError("Invalid LBVH geometry vertices")
    validate_shapes(l, shapes[~composite], vertices)
    for shape in shapes[composite]:
        values = shape.view(np.float32)
        lo = values[sf["b"] : sf["b"] + 3]
        hi = values[sf["c"] : sf["c"] + 3]
        if not np.isfinite(np.r_[lo, hi]).all() or np.any(lo > hi):
            raise ValueError("Invalid composite LBVH bounds")
    groups = b["Groups"].reshape(-1, l["group_words"])
    if np.any(groups[:, l["groups"]["test_mode"]] > 1):
        raise ValueError("Invalid collision group mode")
    if c["kind"] in (
        "lbvh_build",
        "lbvh_escapes",
        "lbvh_leaves",
        "lbvh_refit_chunks",
        "lbvh_refit_frontier",
        "lbvh_pairs",
    ):
        for batch in range(nb):
            keys = b["Keys"][batch * cap : batch * cap + n]
            order = b["Sorted"][batch * cap : batch * cap + n]
            if np.any(keys[1:] < keys[:-1]) or np.any(keys >= 2**30):
                raise ValueError("Invalid sorted Morton keys")
            np.testing.assert_array_equal(np.sort(order), np.arange(n, dtype=np.uint32))
            if np.any((keys[1:] == keys[:-1]) & (order[1:] < order[:-1])):
                raise ValueError("Unstable equal-key collider ordering")
        raw = b["Keys"].copy()
        for batch in range(nb):
            start = batch * cap
            order = b["Sorted"][start : start + n].astype(int)
            raw[start + order] = b["Keys"][start : start + n]
        check_morton(l, c, b, {"Keys": raw})
    if c["kind"] in (
        "lbvh_escapes",
        "lbvh_leaves",
        "lbvh_refit_chunks",
        "lbvh_refit_frontier",
        "lbvh_pairs",
    ):
        topology(l, c, b, escapes=c["kind"] != "lbvh_escapes")
    if c["kind"] == "lbvh_refit_frontier":
        for batch in range(nb):
            count = int(b["FrontierLen"][batch])
            items = b["Frontier"][batch * cap : batch * cap + count]
            if (
                count > n
                or len(set(items.tolist())) != count
                or np.any(items == 0)
                or np.any(items >= max(2 * n - 1, 1))
            ):
                raise ValueError("Invalid refit frontier")
        check_frontier(l, c, b, b)


def check_domains(l, c, b, r):
    n, nb = c["colliders"], c["batches"]
    a = l["aabb"]
    aw = l["aabb_words"]
    poses = b["Poses"].view(np.float32).reshape(-1, l["pose_words"])
    p = poses[: n * nb, l["pose_translation"] : l["pose_translation"] + 3]
    if c["kind"] == "lbvh_domain":
        rows = r["Partials"].view(np.float32).reshape(-1, aw)
        for batch in range(nb):
            for group in range(64):
                ids = (
                    [i for i in range(n) if (i // 128) % c["groups"] == group]
                    if group < c["groups"]
                    else []
                )
                points = p[batch::nb][ids]
                lo = np.min(points, axis=0) if ids else np.full(3, np.float32(1e18))
                hi = np.max(points, axis=0) if ids else np.full(3, np.float32(-1e18))
                np.testing.assert_array_equal(
                    rows[batch * 64 + group, a["mins"] : a["mins"] + 3], lo
                )
                np.testing.assert_array_equal(
                    rows[batch * 64 + group, a["maxs"] : a["maxs"] + 3], hi
                )
    else:
        src = b["Partials"].view(np.float32).reshape(-1, aw)
        dst = r["Domain"].view(np.float32).reshape(-1, aw)
        for batch in range(nb):
            for field, reduce in [("mins", np.min), ("maxs", np.max)]:
                off = a[field]
                np.testing.assert_array_equal(
                    dst[batch, off : off + 3],
                    reduce(src[batch * 64 : batch * 64 + 64, off : off + 3], axis=0),
                )


def check_morton(l, c, b, r):
    n, nb, cap = c["colliders"], c["batches"], c["capacity"]
    a = l["aabb"]
    domain = b["Domain"].view(np.float32).reshape(-1, l["aabb_words"])
    poses = b["Poses"].view(np.float32).reshape(-1, l["pose_words"])
    for batch in range(nb):
        lo = domain[batch, a["mins"] : a["mins"] + 3]
        hi = domain[batch, a["maxs"] : a["maxs"] + 3]
        for i in range(n):
            point = poses[
                i * nb + batch, l["pose_translation"] : l["pose_translation"] + 3
            ]
            with np.errstate(divide="ignore", invalid="ignore"):
                v = np.float32(np.float32(point - lo) / np.float32(hi - lo))
            scaled = np.float32(
                np.clip(np.nan_to_num(v, nan=0), 0, np.float32(1023 / 1024))
                * np.float32(1024)
            ).astype(np.uint32)
            key = 0
            for bit in range(10):
                for d in range(3):
                    key |= ((int(scaled[d]) >> bit) & 1) << (bit * 3 + 2 - d)
            if r["Keys"][batch * cap + i] != key:
                raise AssertionError("Morton quantization mismatch")


def check_geometry(l, c, b, r, all_internal=False, chunk_internal=False):
    n, nb, cap = c["colliders"], c["batches"], c["capacity"]
    f, a = l["lbvh_node"], l["aabb"]
    w = l["lbvh_node_words"]
    trees = r["Tree"].reshape(-1, w)
    poses = b["Poses"].view(np.float32).reshape(-1, l["pose_words"])
    shapes = b["Shapes"].reshape(-1, l["shape_words"])
    vertices = (
        b["Vertices"]
        .view(np.float32)
        .reshape(-1, l["vertex_words"])[:, l["vertex_vector"] : l["vertex_vector"] + 3]
        .astype(float)
    )
    sf, types = l["shape"], l["shape_types"]
    for batch in range(nb):
        tree = trees[2 * batch * cap : 2 * batch * cap + max(2 * n - 1, 1)]
        floats = tree.view(np.float32)
        for i in range(n):
            node = n - 1 + i
            row = tree[node]
            index = int(row[f["left"]])
            expected = b["Sorted"][batch * cap + i]
            if (
                index != expected
                or row[f["first_leaf"]] != i
                or row[f["last_leaf"]] != i
            ):
                raise AssertionError("Invalid leaf identity")
            lo = floats[node, f["aabb"] + a["mins"] : f["aabb"] + a["mins"] + 3].astype(
                float
            )
            hi = floats[node, f["aabb"] + a["maxs"] : f["aabb"] + a["maxs"] + 3].astype(
                float
            )
            if not np.isfinite(np.r_[lo, hi]).all() or np.any(lo > hi):
                raise AssertionError("Invalid leaf bounds")
            actual_extent = floats[node, f["max_extent"]]
            extent = np.max(np.float32(hi) - np.float32(lo))
            np.testing.assert_array_equal(actual_extent, extent)
            shape = shapes[index * nb + batch]
            pose = poses[index * nb + batch]
            q = pose[l["pose_rotation"] : l["pose_rotation"] + 4].astype(float)
            p = pose[l["pose_translation"] : l["pose_translation"] + 3].astype(float)
            ty = int(shape[sf["a"] + 3])
            s = shape.view(np.float32)
            for d in range(3):
                direction = np.eye(3)[d]
                local = inverse(q, direction)
                if ty in (types["trimesh"], types["polyline"]):
                    thickness = 0.4 if ty == types["polyline"] else 0
                    lower = s[sf["b"] : sf["b"] + 3].astype(float) - thickness
                    upper = s[sf["c"] : sf["c"] + 3].astype(float) + thickness
                    maximum = np.sum(np.where(local >= 0, upper, lower) * local)
                    minimum = np.sum(np.where(local >= 0, lower, upper) * local)
                else:
                    maximum = support_value(l, shape, vertices, local)
                    minimum = -support_value(l, shape, vertices, -local)
                if lo[d] > p[d] + minimum + 1e-5 or hi[d] < p[d] + maximum - 1e-5:
                    raise AssertionError("Leaf bounds exclude shape support")
        for node in range(max(n - 1, 0)):
            row = tree[node]
            first, last = map(int, row[[f["first_leaf"], f["last_leaf"]]])
            if not all_internal and not (
                chunk_internal and first // 256 == last // 256
            ):
                continue
            left, right = map(int, row[[f["left"], f["right"]]])
            for field, combine in [("mins", np.minimum), ("maxs", np.maximum)]:
                off = f["aabb"] + a[field]
                np.testing.assert_array_equal(
                    floats[node, off : off + 3],
                    combine(floats[left, off : off + 3], floats[right, off : off + 3]),
                )
            np.testing.assert_array_equal(
                floats[node, f["max_extent"]],
                max(floats[left, f["max_extent"]], floats[right, f["max_extent"]]),
            )
            if row[f["refit_count"]] != 2:
                raise AssertionError("Incomplete child arrivals")


def check_frontier(l, c, b, r):
    n, nb, cap = c["colliders"], c["batches"], c["capacity"]
    f = l["lbvh_node"]
    w = l["lbvh_node_words"]
    trees = r["Tree"].reshape(-1, w)
    for batch in range(nb):
        tree = trees[2 * batch * cap : 2 * batch * cap + max(2 * n - 1, 1)]
        expected = set()
        for node in range(1, max(2 * n - 1, 1)):
            row = tree[node]
            first, last = map(int, row[[f["first_leaf"], f["last_leaf"]]])
            parent = tree[int(row[f["parent"]])]
            if (
                first // 256 == last // 256
                and parent[f["first_leaf"]] // 256 != parent[f["last_leaf"]] // 256
            ):
                expected.add(node)
        count = int(r["FrontierLen"][batch])
        items = r["Frontier"][batch * cap : batch * cap + count]
        if count != len(expected) or set(items.tolist()) != expected:
            raise AssertionError("Invalid refit cut membership")
        np.testing.assert_array_equal(
            r["Frontier"][batch * cap + count : (batch + 1) * cap],
            b["Frontier"][batch * cap + count : (batch + 1) * cap],
        )
    np.testing.assert_array_equal(r["Frontier"][cap * nb :], b["Frontier"][cap * nb :])


def write_policy(l, c, b):
    kind = c["kind"]
    n, nb, cap = c["colliders"], c["batches"], c["capacity"]
    f = l["lbvh_node"]
    w = l["lbvh_node_words"]
    af = l["aabb"]
    masks = {name: np.zeros(v.size, bool) for name, v in b.items()}
    floats = np.zeros(b["Tree"].size, bool)
    if kind == "lbvh_domain":
        for i in range(64 * nb):
            for field in ("mins", "maxs"):
                masks["Partials"][
                    i * l["aabb_words"] + af[field] : i * l["aabb_words"]
                    + af[field]
                    + 3
                ] = True
    if kind == "lbvh_domain_merge":
        for i in range(nb):
            for field in ("mins", "maxs"):
                masks["Domain"][
                    i * l["aabb_words"] + af[field] : i * l["aabb_words"]
                    + af[field]
                    + 3
                ] = True
    if kind == "lbvh_morton":
        for batch in range(nb):
            masks["Keys"][batch * cap : batch * cap + n] = True
    for batch in range(nb):
        root = 2 * batch * cap
        if kind == "lbvh_build":
            for i in range(max(n - 1, 0)):
                for field in (
                    "left",
                    "right",
                    "refit_count",
                    "first_leaf",
                    "last_leaf",
                ):
                    masks["Tree"][(root + i) * w + f[field]] = True
            for i in range(1, max(2 * n - 1, 1)):
                masks["Tree"][(root + i) * w + f["parent"]] = True
        if kind == "lbvh_escapes":
            for i in range(max(2 * n - 1, 1)):
                masks["Tree"][(root + i) * w + f["escape"]] = True
        if kind == "lbvh_leaves":
            if n:
                masks["FrontierLen"][batch] = True
            for i in range(n):
                off = (root + n - 1 + i) * w
                masks["Tree"][off + f["aabb"] : off + f["aabb"] + l["aabb_words"]] = (
                    True
                )
                for field in ("left", "first_leaf", "last_leaf", "max_extent"):
                    masks["Tree"][off + f[field]] = True
                for field in ("mins", "maxs"):
                    floats[
                        off + f["aabb"] + af[field] : off + f["aabb"] + af[field] + 3
                    ] = True
                floats[off + f["max_extent"]] = True
        if kind in ("lbvh_refit_chunks", "lbvh_refit_frontier"):
            for i in range(max(n - 1, 0)):
                off = (root + i) * w
                masks["Tree"][off + f["aabb"] : off + f["aabb"] + l["aabb_words"]] = (
                    True
                )
                for field in ("refit_count", "max_extent"):
                    masks["Tree"][off + f[field]] = True
            if kind == "lbvh_refit_chunks" and n >= 2:
                masks["FrontierLen"][batch] = True
                masks["Frontier"][batch * cap : batch * cap + n] = True
    if kind == "lbvh_refit_plan":
        masks["Rounds"][0] = True
    if kind == "lbvh_reset":
        masks["PairCount"][:] = True
    if kind == "flat_list_dispatch":
        masks["Indirect"][:3] = True
    if kind == "lbvh_pairs":
        masks["Pairs"][: c["pair_capacity"] * l["pair_words"]] = True
        masks["PairCount"][0] = True
    return masks, floats


def check_outputs(l, c, b, a, e, k):
    validate_case(l, c, b)
    kind = c["kind"]
    n, nb = c["colliders"], c["batches"]
    masks, floats = write_policy(l, c, b)
    errors = {}
    for r in (a, e):
        if set(r) != set(b) or any(
            v.dtype != np.uint32 or v.shape != b[name].shape for name, v in r.items()
        ):
            raise ValueError("Invalid LBVH returned storage")
        for name, v in r.items():
            np.testing.assert_array_equal(
                v[~masks[name]], b[name][~masks[name]], err_msg=name + " guard/readonly"
            )
        if kind in ("lbvh_domain", "lbvh_domain_merge"):
            check_domains(l, c, b, r)
        if kind == "lbvh_morton":
            check_morton(l, c, b, r)
        if kind in (
            "lbvh_build",
            "lbvh_escapes",
            "lbvh_leaves",
            "lbvh_refit_chunks",
            "lbvh_refit_frontier",
            "lbvh_pairs",
        ):
            topology(l, c, r, escapes=kind != "lbvh_build")
        if kind in (
            "lbvh_leaves",
            "lbvh_refit_chunks",
            "lbvh_refit_frontier",
            "lbvh_pairs",
        ):
            check_geometry(
                l,
                c,
                b,
                r,
                all_internal=kind in ("lbvh_refit_frontier", "lbvh_pairs"),
                chunk_internal=kind == "lbvh_refit_chunks",
            )
        if kind == "lbvh_refit_chunks":
            check_frontier(l, c, b, r)
        if kind == "lbvh_pairs":
            check_pairs(l, c, b, r)
        if kind == "lbvh_refit_plan":
            np.testing.assert_array_equal(
                r["Rounds"][0],
                (max((min(int(v), n) for v in b["FrontierLen"][:nb]), default=0) + 255)
                // 256,
            )
        if kind == "lbvh_reset":
            np.testing.assert_array_equal(r["PairCount"], np.zeros_like(r["PairCount"]))
        if kind == "flat_list_dispatch":
            np.testing.assert_array_equal(
                r["Indirect"][:3],
                [(min(int(b["PairCount"][0]), c["pair_capacity"]) + 63) // 64, 1, 1],
            )
    for name in b:
        if (
            name == "Pairs"
            and kind == "lbvh_pairs"
            or name == "Frontier"
            and kind == "lbvh_refit_chunks"
        ):
            continue
        if name == "Tree" and kind == "lbvh_leaves":
            np.testing.assert_array_equal(a[name][~floats], e[name][~floats])
            av = a[name][floats].view(np.float32)
            ev = e[name][floats].view(np.float32)
            np.testing.assert_allclose(av, ev, atol=k["atol"], rtol=k["rtol"])
            errors[name] = float(np.max(abs(av - ev), initial=0))
        else:
            np.testing.assert_array_equal(a[name], e[name], err_msg=name)
            errors[name] = 0.0
    return errors
