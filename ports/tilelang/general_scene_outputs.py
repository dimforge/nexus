"""Semantic checks for general collision allocation, sorting and canonical contacts."""

import broadphase_replay as broad
import numpy as np
import scene_numeric as numeric_policy
from deferred_validation import contact_fields, vertices
from general_scene_physics import primitives
from joint_scene_physics import multiply
from support_geometry import contains, inverse, support_value


def check_collision(l, c, b, a, e, k):
    per, nb = c["colliders_per_batch"], c["batches"]
    bc = dict(
        c,
        kind="broadphase_pairs",
        count=c["body_count"],
        colliders=per,
        capacity=per + 3,
        storage=c["body_storage"],
        vertices=c["vertex_count"],
        initial_count=0,
        iterations=1,
        dispatch=per * per * nb,
        groups=max((per * per * nb + 63) // 64, 1),
    )
    bb = {n: b[n] for n in broad.BUFFERS}
    bb.update(Poses=a["ColliderPoses"], Pairs=a["Pairs"])
    aa = {n: a[n] for n in broad.BUFFERS}
    aa["Poses"] = a["ColliderPoses"]
    ee = {n: e[n] for n in broad.BUFFERS}
    ee["Poses"] = e["ColliderPoses"]
    bc["buffer_words"] = {n: v.size for n, v in bb.items()}
    errors = broad.check_outputs(
        l, bc, bb, aa, ee, numeric_policy.field_contract(c, k, "Aabbs")
    )
    pf, cf, mf = l["plan"], l["contact"], l["manifold"]
    w = l["contact_words"]
    length = cf["contact"] + mf["len"]
    count = int(a["PairCount"][0])
    pn = int(a["PfmCount"][0])
    bound = count + pn
    if max(count, pn) > c["pair_capacity"]:
        raise AssertionError("General scene overflow")
    for name in (
        "PairCount",
        "PfmCount",
        "PfmSortLen",
        "Plan",
        "ContactsIndirect",
        "PfmIndirect",
        "MbIndirect",
    ):
        np.testing.assert_array_equal(
            a[name], e[name], err_msg=f"General publication {name}"
        )
    f = l["pfm_pair"]
    fw = l["pfm_pair_words"]
    l["shape"]
    vtx = vertices(l, b)
    work = []
    for r in (a, e):
        if (
            r["Plan"][pf["bound"]] != bound
            or r["Plan"][pf["pfm_base"]] != count
            or r["Plan"][pf["pfm_len"]] != pn
            or r["PfmSortLen"][0] != pn
        ):
            raise AssertionError("General contact plan")
        for name, total, threads in [
            ("ContactsIndirect", bound, 64),
            ("PfmIndirect", pn, 64),
        ]:
            # The multibody grid is still published even though this scope has no multibodies.
            np.testing.assert_array_equal(
                r[name][:3],
                np.array([(total + threads - 1) // threads, 1, 1], np.uint32),
                err_msg=name,
            )
        np.testing.assert_array_equal(
            r["MbIndirect"][:3], np.array([0, nb, 1], np.uint32)
        )
        pairs = r["Pairs"].reshape(-1, l["pair_words"])[
            :count, l["pair_colliders"] : l["pair_colliders"] + 2
        ]
        rows = r["PfmPairs"].reshape(-1, fw)[:pn]
        order = r["PfmOrder"][:pn]
        np.testing.assert_array_equal(
            np.sort(order), np.arange(pn, dtype=np.uint32), err_msg="PFM permutation"
        )
        np.testing.assert_array_equal(
            r["PfmSortedKeys"][:pn],
            rows[order, f["pair_index"]],
            err_msg="PFM final pair keys",
        )
        observed = []
        membership = {}
        for row in rows:
            index = int(row[f["pair_index"]])
            ids = tuple(map(int, row[f["colliders"] : f["colliders"] + 2]))
            sub = int(row[f["subshape"]])
            if index >= count or set(ids) != set(map(int, pairs[index])):
                raise AssertionError("PFM originating pair provenance")
            key = (*ids, sub)
            if key in membership:
                raise AssertionError("Duplicate PFM work item")
            membership[key] = row
        for row in rows[order]:
            observed.append((int(row[f["pair_index"]]), int(row[f["subshape"]])))
        if observed != sorted(observed):
            raise AssertionError("PFM stable subshape and pair ordering")
        work.append(membership)
        raw = r["RawContacts"].reshape(-1, w)
        order = r["InOrder"][:bound]
        np.testing.assert_array_equal(
            np.sort(order),
            np.arange(bound, dtype=np.uint32),
            err_msg="General canonical permutation",
        )
        canonical = raw[order].copy()
        canonical[canonical[:, length] == 0] = 0
        np.testing.assert_array_equal(
            r["Contacts"][: bound * w],
            canonical.ravel(),
            err_msg="General canonical copy",
        )
        live = canonical[canonical[:, length] > 0]
        identities = [
            (
                int(row[cf["colliders"]]),
                int(row[cf["colliders"] + 1]),
                int(row[cf["subshape"]]),
            )
            for row in live
        ]
        if (
            identities != sorted(identities)
            or len(identities) != len(set(identities))
            or np.any(canonical[: len(live), length] == 0)
        ):
            raise AssertionError("General canonical identity ordering")
        if r["SortCount"][0] != bound:
            raise AssertionError("General canonical sort count")
    if work[0].keys() != work[1].keys():
        raise AssertionError("General PFM membership differs")
    mask = np.zeros(fw, bool)
    for key, width in (("pose_rotation", 4), ("pose_translation", 3)):
        off = f["pose12"] + l[key]
        mask[off : off + width] = True
    mask[f["recycle_extent"]] = True
    integers = ~mask
    integers[f["pair_index"]] = False
    for key in work[0]:
        ar, er = work[0][key], work[1][key]
        np.testing.assert_array_equal(
            ar[integers], er[integers], err_msg="General exact PFM fields"
        )
        np.testing.assert_allclose(
            ar[mask].view(np.float32),
            er[mask].view(np.float32),
            atol=numeric_policy.tolerances(c, k, "PfmPairs")[0],
            rtol=numeric_policy.tolerances(c, k, "PfmPairs")[1],
            err_msg="General PFM pose/extent",
        )
    contacts = [r["Contacts"].reshape(-1, w)[:bound] for r in (a, e)]
    shapes = b["Shapes"].reshape(-1, l["shape_words"])
    geometry = [primitives(l, s, vtx, b["GeometryIndices"]) for s in shapes]
    params = b["RbdParams"].view(np.float32)
    rf = l["rbd_params"]
    prediction = float(
        np.float32(
            params[rf["normalized_prediction_distance"]] * params[rf["length_unit"]]
        )
    )
    largest = 0.0
    secondary_misses = [set(), set()]
    for i, (ar, er) in enumerate(zip(*contacts)):
        ac, ec = int(ar[length]), int(er[length])
        if ac != ec or not 0 <= ac <= 4:
            raise AssertionError("General contact cardinality")
        if not ac:
            np.testing.assert_array_equal(ar, er)
            continue
        numeric = np.zeros(w, bool)
        numeric[:length] = True
        for key in ("friction", "restitution", "recycle_extent"):
            numeric[cf[key]] = True
        np.testing.assert_array_equal(
            ar[~numeric], er[~numeric], err_msg="General exact contact metadata"
        )
        for side, (r, row) in enumerate(((a, ar), (e, er))):
            ids = row[cf["colliders"] : cf["colliders"] + 2].astype(int)
            if np.any(ids >= c["body_count"]) or ids[0] % nb != ids[1] % nb:
                raise AssertionError("General contact collider/batch identity")
            np.testing.assert_array_equal(
                row[cf["bodies"] : cf["bodies"] + 2], b["Parents"][ids]
            )
            if b["Parents"][ids[0]] == b["Parents"][ids[1]]:
                raise AssertionError("General same-body contact")
            points, normal, num = contact_fields(l, row)
            if not np.isfinite(points).all() or not np.isfinite(normal).all():
                raise AssertionError("General contact nonfinite")
            np.testing.assert_allclose(np.linalg.norm(normal), 1, atol=2e-6, rtol=0)
            poses = (
                r["ColliderPoses"]
                .view(np.float32)
                .reshape(-1, l["pose_words"])
                .astype(float)
            )
            qs = poses[ids, l["pose_rotation"] : l["pose_rotation"] + 4]
            ts = poses[ids, l["pose_translation"] : l["pose_translation"] + 3]
            q = multiply(np.r_[-qs[0, :3], qs[0, 3]], qs[1])
            shift = inverse(qs[0], ts[1] - ts[0])
            primary, secondary = (geometry[x] for x in ids)
            minimum = (
                np.dot(normal, shift)
                - max(support_value(l, s, vtx, normal) for s in primary)
                - max(support_value(l, s, vtx, inverse(q, -normal)) for s in secondary)
            )
            for point_index, point in enumerate(points[:num]):
                pi, di = l["manifold_point"]["pt"], l["manifold_point"]["dist"]
                if not any(contains(l, s, vtx, point[pi : pi + 3]) for s in primary):
                    raise AssertionError(
                        f"General primary witness outside collider {i}"
                    )
                witness_b = inverse(q, point[pi : pi + 3] + normal * point[di] - shift)
                if not any(contains(l, s, vtx, witness_b) for s in secondary):
                    secondary_misses[side].add((tuple(map(int, ids)), point_index))
                if point[di] < minimum - 1e-5 or point[di] > prediction + 2e-6:
                    raise AssertionError("General contact support/prediction bound")
        # Inactive point payload can survive native dedup/reduction and is not a contact.
        active = numeric.copy()
        active[
            cf["contact"] + mf["points_a"] + ac * l["manifold_point_words"] : cf[
                "contact"
            ]
            + mf["normal_a"]
        ] = False
        av, ev = ar[active].view(np.float32), er[active].view(np.float32)
        np.testing.assert_allclose(
            av,
            ev,
            atol=numeric_policy.tolerances(c, k, "Contacts")[0],
            rtol=numeric_policy.tolerances(c, k, "Contacts")[1],
            err_msg=f"General active contact {i}",
        )
        largest = max(largest, float(np.max(np.abs(av - ev), initial=0)))
    if secondary_misses[0] - secondary_misses[1]:
        raise AssertionError("New general secondary witness containment failure")
    errors["Contacts"] = largest
    return errors


def witness_diagnostics(l, c, b, r):
    """Keep the inherited clipping limitation visible in each scene report."""
    shapes = b["Shapes"].reshape(-1, l["shape_words"])
    vtx = vertices(l, b)
    geometry = [primitives(l, s, vtx, b["GeometryIndices"]) for s in shapes]
    poses = (
        r["ColliderPoses"].view(np.float32).reshape(-1, l["pose_words"]).astype(float)
    )
    rows = r["Contacts"].reshape(-1, l["contact_words"])[
        : int(r["Plan"][l["plan"]["bound"]])
    ]
    misses = []
    for row in rows:
        points, normal, num = contact_fields(l, row)
        if not num:
            continue
        ids = row[l["contact"]["colliders"] : l["contact"]["colliders"] + 2].astype(int)
        qs = poses[ids, l["pose_rotation"] : l["pose_rotation"] + 4]
        ts = poses[ids, l["pose_translation"] : l["pose_translation"] + 3]
        q = multiply(np.r_[-qs[0, :3], qs[0, 3]], qs[1])
        shift = inverse(qs[0], ts[1] - ts[0])
        for j, point in enumerate(points[:num]):
            pi, di = l["manifold_point"]["pt"], l["manifold_point"]["dist"]
            if not any(
                contains(
                    l,
                    s,
                    vtx,
                    inverse(q, point[pi : pi + 3] + normal * point[di] - shift),
                )
                for s in geometry[ids[1]]
            ):
                misses.append([int(ids[0]), int(ids[1]), j])
    return misses
