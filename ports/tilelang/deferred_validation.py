"""Deferred work-list membership, native ABI, and manifold/reduction validation."""

import numpy as np
from pfm_validation import check_manifold_row, secondary_inside
from support_validation import validate_shapes

OPERATIONS = (
    "deferred_pairs",
    "deferred_manifolds",
    "pfm_pair_keys",
    "pfm_subshape_keys",
    "pfm_permuted_keys",
    "reduce_contacts",
)
BUFFERS = (
    "Pairs",
    "PairCount",
    "Poses",
    "Shapes",
    "PfmPairs",
    "PfmCount",
    "Vertices",
    "Indices",
    "BatchIds",
    "RbdParams",
    "Contacts",
    "Order",
    "Plan",
    "Parents",
    "Materials",
    "Keys",
    "PfmCandidates",
)


def vertices(l, b):
    return (
        b["Vertices"]
        .view(np.float32)
        .reshape(-1, l["vertex_words"])[:, l["vertex_vector"] : l["vertex_vector"] + 3]
        .astype(float)
    )


def mesh_indices(l, shape, v, indices):
    sf = l["shape"]
    ty = int(shape[sf["a"] + 3])
    polyline = ty == l["shape_types"]["polyline"]
    vb, ib, n = map(int, shape[sf["a"] : sf["a"] + 3])
    f = shape.view(np.float32)
    low = f[sf["b"] : sf["b"] + 3]
    high = f[sf["c"] : sf["c"] + 3]
    if (
        not np.isfinite(np.r_[low, high]).all()
        or np.any(low > high)
        or np.any(np.abs(np.r_[low, high]) > 1e6)
    ):
        raise ValueError("Invalid composite root bounds")
    if n >= 2**27 or vb + 2 * n > len(v) or ib + 3 * n > len(indices):
        raise ValueError("Invalid composite BVH range")
    nodes = indices[ib : ib + 3 * n].reshape(-1, 3)
    seen_parts = set()
    for i, (entry, exit_id, part) in enumerate(nodes.astype(np.int64)):
        if entry != 0xFFFFFFFF and entry >= n:
            raise ValueError("Invalid BVH entry")
        if exit_id != 0xFFFFFFFF and exit_id > n:
            raise ValueError("Invalid BVH exit")
        if entry == 0xFFFFFFFF:
            if int(part) in seen_parts:
                raise ValueError("Repeated composite leaf identity")
            seen_parts.add(int(part))
            width = 2 if polyline else 3
            start = ib + 3 * n + int(part) * width
            if start + width > len(indices):
                raise ValueError("Invalid composite primitive range")
            ids = indices[start : start + width].astype(np.int64)
            if not polyline and (
                part >= shape[sf["b"] + 3] or np.any(ids >= shape[sf["c"] + 3])
            ):
                raise ValueError("Invalid mesh primitive index")
            if np.any(vb + 2 * n + ids >= len(v)):
                raise ValueError("Invalid composite vertex index")
            points = v[vb + 2 * n + ids]
            if np.any(points < low - 2e-6) or np.any(points > high + 2e-6):
                raise ValueError("Composite geometry outside root bounds")
        else:
            a, b = v[vb + 2 * i : vb + 2 * i + 2]
            if np.any(a > b):
                raise ValueError("Invalid BVH node bounds")
    # Both entry and exit arcs must form an acyclic graph. This permits arbitrary
    # physical node order while rejecting repeated visits and nonterminating data.
    color = np.zeros(n, np.uint8)
    for start in range(n):
        stack = [(start, False)]
        while stack:
            node, done = stack.pop()
            if node >= n:
                continue
            if done:
                color[node] = 2
                continue
            if color[node] == 1:
                raise ValueError("Cyclic composite BVH")
            if color[node] == 2:
                continue
            color[node] = 1
            stack.append((node, True))
            entry, exit_id, _ = map(int, nodes[node])
            stack.append((exit_id, False))
            if entry != 0xFFFFFFFF:
                stack.append((entry, False))


def validate_case(l, c, b):
    for name in (
        "count",
        "capacity",
        "candidate_count",
        "candidate_capacity",
        "groups",
        "iterations",
    ):
        if type(c[name]) is not int or not 0 <= c[name] < 2**27:
            raise ValueError("Invalid deferred dimensions")
    n, cap = c["count"], c["capacity"]
    colliders = n * 2 + 3
    if (
        c["kind"] not in OPERATIONS
        or c["groups"] == 0
        or c["iterations"] != 1
        or c["candidate_capacity"] != max(cap, c["candidate_count"])
    ):
        raise ValueError("Invalid deferred dispatch")
    sizes = {
        "Pairs": (max(n, cap) + 3) * l["pair_words"],
        "PairCount": 3,
        "Poses": colliders * l["pose_words"],
        "Shapes": colliders * l["shape_words"],
        "PfmPairs": (cap + 3) * l["pfm_pair_words"],
        "PfmCandidates": (c["candidate_capacity"] + 3) * l["pfm_pair_words"],
        "PfmCount": 3,
        "Vertices": c["buffer_words"]["Vertices"],
        "Indices": c["buffer_words"]["Indices"],
        "BatchIds": l["batch_words"],
        "RbdParams": l["rbd_params_words"],
        "Contacts": (cap * 2 + 3) * l["contact_words"],
        "Order": cap + 3,
        "Plan": l["plan_words"],
        "Parents": colliders,
        "Materials": colliders * l["material_words"],
        "Keys": cap + 3,
    }
    if (
        set(b) != set(BUFFERS)
        or sizes != c["buffer_words"]
        or any(
            v.dtype != np.uint32 or v.shape != (sizes[name],) for name, v in b.items()
        )
    ):
        raise ValueError("Invalid deferred storage ABI")
    if sizes["Vertices"] % l["vertex_words"]:
        raise ValueError("Invalid vertex stride")
    bf = l["batch"]
    batch = b["BatchIds"]
    if (
        batch[bf["collision_pairs_capacity"]] != cap
        or batch[bf["contacts_capacity"]] != 2 * cap
        or b["PairCount"][0] != n
        or b["PfmCount"][0] != 0
    ):
        raise ValueError("Invalid deferred capacity/counter")
    total = min(n, cap)
    length = min(c["candidate_count"], cap)
    plan = b["Plan"]
    pf = l["plan"]
    if (
        plan[pf["pfm_base"]] != total
        or plan[pf["pfm_len"]] != length
        or plan[pf["bound"]] != total + length
    ):
        raise ValueError("Invalid deferred contact plan")
    pairs = b["Pairs"].reshape(-1, l["pair_words"])[
        :total, l["pair_colliders"] : l["pair_colliders"] + 2
    ]
    if np.any(pairs >= colliders):
        raise ValueError("Invalid collider pair")
    poses = b["Poses"].view(np.float32).reshape(-1, l["pose_words"])
    q = poses[:, l["pose_rotation"] : l["pose_rotation"] + 4]
    t = poses[:, l["pose_translation"] : l["pose_translation"] + 3]
    if (
        not np.isfinite(np.r_[q.ravel(), t.ravel()]).all()
        or np.any(np.abs(t) > 1e6)
        or not np.allclose(np.sum(q.astype(float) ** 2, axis=1), 1, atol=2e-6, rtol=0)
    ):
        raise ValueError("Invalid collider pose")
    v = vertices(l, b)
    if not np.isfinite(v).all() or np.any(np.abs(v) > 1e6):
        raise ValueError("Invalid deferred vertices")
    shapes = b["Shapes"].reshape(-1, l["shape_words"])
    sf = l["shape"]
    tags = shapes[:, sf["a"] + 3]
    types = l["shape_types"]
    composite = (tags == types["trimesh"]) | (tags == types["polyline"])
    validate_shapes(l, shapes[~composite], v)
    for shape in shapes[composite]:
        mesh_indices(l, shape, v, b["Indices"])
    for shape in shapes[tags == types["convex_poly"]]:
        first, end = map(int, shape[sf["b"] : sf["b"] + 2])
        a, z = map(int, shape[sf["a"] : sf["a"] + 2])
        if (
            not 0 <= first < end <= len(b["Indices"])
            or (end - first) % 3
            or np.any(b["Indices"][first:end] >= z - a)
        ):
            raise ValueError("Invalid convex face indices")
        tri = v[a + b["Indices"][first:end].astype(np.int64)].reshape(-1, 3, 3)
        if not np.any(
            np.linalg.norm(
                np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
            )
            > 1e-12
        ):
            raise ValueError("Degenerate convex faces")
    params = b["RbdParams"].view(np.float32)
    rf = l["rbd_params"]
    predict, unit, merge = params[
        [
            rf["normalized_prediction_distance"],
            rf["length_unit"],
            rf["contact_merge_cos"],
        ]
    ]
    if (
        not np.isfinite([predict, unit, merge]).all()
        or predict < 0
        or unit <= 0
        or not np.isfinite(predict * unit)
        or not -1 <= merge <= 1
    ):
        raise ValueError("Invalid narrow-phase parameters")
    mats = b["Materials"].reshape(-1, l["material_words"]).view(np.float32)
    mat = l["material"]
    if not np.isfinite(mats[:, [mat["friction"], mat["restitution"]]]).all():
        raise ValueError("Invalid collider material")
    if c["kind"] != "deferred_pairs":
        order = b["Order"][:length]
        if np.any(order >= length) or len(np.unique(order)) != length:
            raise ValueError("Invalid PFM permutation")
        rows = b["PfmPairs"].reshape(-1, l["pfm_pair_words"])[:length]
        if np.any(
            rows[:, l["pfm_pair"]["colliders"] : l["pfm_pair"]["colliders"] + 2]
            >= colliders
        ):
            raise ValueError("Invalid PFM collider")
    if c["kind"] == "reduce_contacts":
        keys = b["Keys"][:length]
        if np.any(keys[1:] < keys[:-1]):
            raise ValueError("Reduction requires sorted pair keys")
        rows = b["Contacts"].reshape(-1, l["contact_words"])[total : total + length]
        cf = l["contact"]
        mf = l["manifold"]
        if np.any(rows[:, cf["contact"] + mf["len"]] > 4):
            raise ValueError("Invalid input reduction cardinality")
        for start, end in runs(keys):
            live = rows[start:end][rows[start:end, cf["contact"] + mf["len"]] > 0]
            if len(live) > 1 and np.any(
                live[:, cf["colliders"] : cf["colliders"] + 2]
                != live[0, cf["colliders"] : cf["colliders"] + 2]
            ):
                raise ValueError("Reduction run mixes collider frames")


def runs(keys):
    start = 0
    while start < len(keys):
        end = start + 1
        while end < len(keys) and keys[end] == keys[start]:
            end += 1
        yield start, end
        start = end


def pair_key(l, row):
    f = l["pfm_pair"]
    return int(row[f["pair_index"]]), int(row[f["subshape"]])


def contact_fields(l, row):
    cf, mf = l["contact"], l["manifold"]
    v = row.view(np.float32).astype(float)
    o = cf["contact"]
    return (
        v[o + mf["points_a"] : o + mf["normal_a"]].reshape(
            4, l["manifold_point_words"]
        ),
        v[o + mf["normal_a"] : o + mf["normal_a"] + 3],
        int(row[o + mf["len"]]),
    )


def pair_geometry(l, row):
    f, sf = l["pfm_pair"], l["shape"]
    shapes = []
    for i in (1, 2):
        shape = row[f[f"shape{i}"] : f[f"shape{i}"] + l["shape_words"]].copy()
        radius = float(
            row[f[f"thickness{i}"] : f[f"thickness{i}"] + 1].view(np.float32)[0]
        )
        if shape[sf["a"] + 3] == l["shape_types"]["capsule"]:
            shape.view(np.float32)[sf["b"] + 3] += radius
        elif radius != 0:
            raise ValueError("Unexpected PFM border radius")
        shapes.append(shape)
    pose = (
        row[f["pose12"] : f["pose12"] + l["pose_words"]].view(np.float32).astype(float)
    )
    return (
        *shapes,
        pose[l["pose_rotation"] : l["pose_rotation"] + 4],
        pose[l["pose_translation"] : l["pose_translation"] + 3],
    )


def check_outputs(l, c, b, a, e, k):
    validate_case(l, c, b)
    for r in (a, e):
        if set(r) != set(b) or any(
            r[n].dtype != np.uint32 or r[n].shape != b[n].shape for n in b
        ):
            raise ValueError("Invalid returned deferred storage")
    kind = c["kind"]
    cap = c["capacity"]
    count = min(c["candidate_count"], cap)
    base = min(c["count"], cap)
    errors = dict.fromkeys(BUFFERS, 0.0)
    written = (
        {"PfmPairs", "PfmCount"}
        if kind == "deferred_pairs"
        else {"Keys"}
        if kind.startswith("pfm_")
        else {"Contacts"}
    )
    for name in set(b) - written:
        for r in (a, e):
            np.testing.assert_array_equal(
                r[name], b[name], err_msg="Deferred readonly " + name
            )
    if kind == "deferred_pairs":
        f = l["pfm_pair"]
        w = l["pfm_pair_words"]
        allrows = b["PfmCandidates"].reshape(-1, w)[: c["candidate_count"]]
        allowed = {pair_key(l, row): row for row in allrows}
        if len(allowed) != c["candidate_count"]:
            raise ValueError("Repeated native candidate identity")
        mask = np.zeros(w, bool)
        mask[
            f["pose12"] + l["pose_rotation"] : f["pose12"] + l["pose_rotation"] + 4
        ] = True
        mask[
            f["pose12"] + l["pose_translation"] : f["pose12"]
            + l["pose_translation"]
            + 3
        ] = True
        mask[f["recycle_extent"]] = True
        for r in (a, e):
            np.testing.assert_array_equal(
                r["PfmCount"][1:], b["PfmCount"][1:], err_msg="PFM counter guard"
            )
            if r["PfmCount"][0] != c["candidate_count"]:
                raise AssertionError("PFM requested count")
            rows = r["PfmPairs"].reshape(-1, w)
            np.testing.assert_array_equal(
                rows[count:],
                b["PfmPairs"].reshape(-1, w)[count:],
                err_msg="PFM work-list guard",
            )
            seen = set()
            for row in rows[:count]:
                key = pair_key(l, row)
                if key not in allowed or key in seen:
                    raise AssertionError("Invalid PFM membership/ownership")
                seen.add(key)
                expected = allowed[key]
                np.testing.assert_array_equal(
                    row[~mask], expected[~mask], err_msg="Exact PFM work item"
                )
                af, ef = row[mask].view(np.float32), expected[mask].view(np.float32)
                np.testing.assert_allclose(af, ef, atol=k["atol"], rtol=k["rtol"])
                errors["PfmPairs"] = max(
                    errors["PfmPairs"], float(np.max(np.abs(af - ef)))
                )
        return errors
    if kind.startswith("pfm_"):
        f = l["pfm_pair"]
        pairs = b["PfmPairs"].reshape(-1, l["pfm_pair_words"])
        order = b["Order"][:count] if kind == "pfm_permuted_keys" else np.arange(count)
        values = pairs[
            order, f["subshape"] if kind == "pfm_subshape_keys" else f["pair_index"]
        ]
        for r in (a, e):
            np.testing.assert_array_equal(
                r["Keys"][:count], values, err_msg="PFM sort keys"
            )
            np.testing.assert_array_equal(
                r["Keys"][count:], b["Keys"][count:], err_msg="PFM key guard"
            )
        return errors
    w = l["contact_words"]
    cf, mf = l["contact"], l["manifold"]
    before = b["Contacts"].reshape(-1, w)
    rows = [r["Contacts"].reshape(-1, w) for r in (a, e)]
    length = cf["contact"] + mf["len"]
    for row in rows:
        np.testing.assert_array_equal(
            row[:base], before[:base], err_msg="Analytic contact prefix"
        )
        np.testing.assert_array_equal(
            row[base + count :], before[base + count :], err_msg="Deferred contact tail"
        )
    if kind == "reduce_contacts":
        np.testing.assert_array_equal(
            a["Contacts"],
            e["Contacts"],
            err_msg="Exact native selection/copy reduction",
        )
        for start, end in runs(b["Keys"][:count]):
            source = before[base + start : base + end]
            candidates = []
            for row in source:
                points, _, size = contact_fields(l, row)
                candidates.extend(points[:size])
            for row in rows[0][base + start : base + end]:
                points, _, size = contact_fields(l, row)
                if size > 4:
                    raise AssertionError("Invalid reduced cardinality")
                for point in points[:size]:
                    if not any(np.array_equal(point, p) for p in candidates):
                        raise AssertionError("Reduction fabricated a point")
        return errors
    geometry_layout = dict(
        l,
        feature_point=l["manifold_point"],
        feature_point_words=l["manifold_point_words"],
    )
    v = vertices(l, b)
    pairs = b["PfmPairs"].reshape(-1, l["pfm_pair_words"])
    f = l["pfm_pair"]
    params = b["RbdParams"].view(np.float32)
    rf = l["rbd_params"]
    prediction = float(
        np.float32(
            params[rf["normalized_prediction_distance"]] * params[rf["length_unit"]]
        )
    )
    for i in range(count):
        actual, expected = (row[base + i] for row in rows)
        pair = pairs[b["Order"][i]]
        if actual[length] != expected[length] or actual[length] > 4:
            raise AssertionError("Exact deferred manifold cardinality")
        if actual[length] == 0:
            for row in (actual, expected):
                np.testing.assert_array_equal(
                    np.delete(row, length),
                    np.delete(before[base + i], length),
                    err_msg="Inert contact payload",
                )
            continue
        metadata = np.ones(w, bool)
        metadata[:length] = False
        for field in ("friction", "restitution"):
            metadata[cf[field]] = False
        np.testing.assert_array_equal(
            actual[metadata],
            expected[metadata],
            err_msg="Exact contact metadata and count",
        )
        # Scalar materials/recycling are native computations and keep strict f32 parity.
        for field in ("friction", "restitution", "recycle_extent"):
            np.testing.assert_allclose(
                actual[cf[field] : cf[field] + 1].view(np.float32),
                expected[cf[field] : cf[field] + 1].view(np.float32),
                atol=k["atol"],
                rtol=k["rtol"],
            )
        ids = pair[f["colliders"] : f["colliders"] + 2]
        for row in (actual, expected):
            np.testing.assert_array_equal(
                row[cf["colliders"] : cf["colliders"] + 2],
                ids,
                err_msg="Deferred collider identity",
            )
            np.testing.assert_array_equal(
                row[cf["bodies"] : cf["bodies"] + 2],
                b["Parents"][ids],
                err_msg="Deferred parent identity",
            )
            if (
                row[cf["subshape"]] != pair[f["subshape"]]
                or row[cf["recycle_extent"]] != pair[f["recycle_extent"]]
            ):
                raise AssertionError("Deferred copied provenance")
        if b["Parents"][ids[0]] == b["Parents"][ids[1]]:
            raise AssertionError("Same-body contact emitted")
        sa, sb, q, shift = pair_geometry(l, pair)
        check_manifold_row(
            geometry_layout,
            sa,
            sb,
            v,
            q,
            shift,
            prediction,
            *contact_fields(l, actual),
            *contact_fields(l, expected),
            k,
        )
        errors["Contacts"] = max(
            errors["Contacts"],
            float(
                np.max(
                    np.abs(
                        actual[:length].view(np.float32)
                        - expected[:length].view(np.float32)
                    )
                )
            ),
        )
    return errors


def diagnostics(l, c, b, a, e, k):
    del k
    if c["kind"] == "deferred_pairs":
        return {
            "requested": c["candidate_count"],
            "written": min(c["candidate_count"], c["capacity"]),
            "overflow": max(0, c["candidate_count"] - c["capacity"]),
        }
    if c["kind"] != "deferred_manifolds":
        return {}
    gl = dict(
        l,
        feature_point=l["manifold_point"],
        feature_point_words=l["manifold_point_words"],
    )
    v = vertices(l, b)
    base = min(c["count"], c["capacity"])
    count = min(c["candidate_count"], c["capacity"])
    pairs = b["PfmPairs"].reshape(-1, l["pfm_pair_words"])
    result = {}
    for label, r in (("actual", a), ("reference", e)):
        misses = []
        rows = r["Contacts"].reshape(-1, l["contact_words"])
        for i in range(count):
            pair = pairs[b["Order"][i]]
            _, sb, q, shift = pair_geometry(l, pair)
            points, normal, size = contact_fields(l, rows[base + i])
            for j in range(size):
                if not secondary_inside(gl, sb, v, q, shift, points[j], normal):
                    misses.append(
                        {
                            "pair": pair_key(l, pair)[0],
                            "subshape": pair_key(l, pair)[1],
                            "point": j,
                        }
                    )
        result[label + "_secondary_containment_misses"] = misses
    return result
