"""Semantic comparison of a resident scene checkpoint, including allocation order."""

import scene_numeric as numeric_policy
from scene_validation import BODY_BUFFERS, READONLY, body_case


def tile_words(l, slot):
    if l.get("friction_model") == "twist":
        from twist_scene_validation import tile_words as twist_words

        yield from twist_words(l, slot)
        return
    base = slot // 32 * l["tile_words"] + slot % 32 * 4
    for off in (*l["header"].values(), l["coupling"]):
        yield from range(base + off, base + off + 4)
    for point in range(4):
        for off in l["point"].values():
            start = base + l["points"] + point * l["point_words"] + off
            yield from range(start, start + 4)
    start = (
        slot // 32 * l["tile_words"]
        + l["warmstart"]
        + slot % 32 * 2 * l["velocity_words"]
    )
    yield from range(start, start + 2 * l["velocity_words"])


def check_guards(l, c, b, a, e):
    import numpy as np
    from body_validation import write_masks

    masks = {name: np.zeros(v.size, dtype=bool) for name, v in b.items()}
    claimed = set(READONLY)

    def prefix(name, count):
        masks[name][:count] = True
        claimed.add(name)

    for kind in ("update_mprops", "free_step"):
        for name, m in write_masks(
            l, body_case(c, kind), {n: b[n] for n in BODY_BUFFERS}
        ).items():
            masks[name] |= m != 0
            claimed.add(name)
    n, cap, cs, nb, per = (
        c[k]
        for k in (
            "body_count",
            "contact_capacity",
            "contact_storage",
            "batches",
            "colliders_per_batch",
        )
    )
    if c.get("hub_pool"):
        pool, base = c["hub_pool"], c["body_storage"]
        for name, size in (
            ("HubList", pool // l["hub_min_constraints"] + 1),
            ("SlotBody", pool),
            ("SlotConstraint", pool),
            ("SlotsIndirect", 3),
            ("AverageIndirect", 3),
        ):
            prefix(name, size)
        masks["SolverVelocities"][
            base * l["velocity_words"] : (base + pool) * l["velocity_words"]
        ] = True
    prefix("ColliderPoses", c.get("collider_count", n) * l["pose_words"])
    for batch in range(nb):
        start = batch * c.get("body_capacity_per_batch", per + 3) * l["aabb_words"]
        masks["Aabbs"][start : start + per * l["aabb_words"]] = True
    claimed.add("Aabbs")
    prefix("Pairs", c["pair_capacity"] * l["pair_words"])
    for name in ("RawContacts", "Contacts"):
        prefix(name, cap * l["contact_words"])
    for name in ("Links", "OldLinks", "SortedLinks"):
        prefix(name, cap * l["link_words"])
    for name in ("Ids", "SortedIds", "OldIds"):
        prefix(name, 2 * cap)
    for name in (
        "Indices",
        "OldIndices",
        "Colors",
        "OldColors",
        "Colored",
        "Pending",
        "InOrder",
        "Keys",
        "Values",
        "SortKeys",
        "PongKeys",
        "PongValues",
    ):
        prefix(name, cap)
    prefix("OldCounts", c["body_storage"])
    for name in ("Tiles", "OldTiles"):
        for slot in range(cap):
            masks[name][list(tile_words(l, slot))] = True
        claimed.add(name)
    for half in (0, cs):
        masks["States"][half * l["state_words"] : (half + cap) * l["state_words"]] = (
            True
        )
    claimed.add("States")
    for name in ("PairCount", "PfmCount", "PfmSortLen", "NumColors", "SortCount"):
        prefix(name, 1)
    prefix("Plan", l["plan_words"])
    for name in (
        "ContactsIndirect",
        "PfmIndirect",
        "MbIndirect",
        "ColoringIndirect",
        "Indirect",
        "SortGrid",
        "ReduceGrid",
    ):
        prefix(name, 3)
    for name in ("Buckets", "Stats", "Params"):
        prefix(name, len(b[name]))
    prefix("RadixCounts", len(b["RadixCounts"]) - 3)
    prefix("Reduced", 1024)
    for i in range(4):
        prefix(f"AdjAux{i}", (c["strides"][i] + 255) // 256 if i < c["levels"] else 0)
        prefix(
            f"BucketAux{i}",
            (c["bucket_strides"][i] + 255) // 256 if i < c["bucket_levels"] else 0,
        )
    prefix("Trajectory", c["frames"] * c["trace_stride"])
    prefix("FrameIndex", 1)
    if claimed != set(b):
        raise ValueError(f"Missing scene write policy: {set(b) - claimed}")
    for name, m in masks.items():
        for result in (a, e):
            np.testing.assert_array_equal(
                result[name][~m],
                b["Colors" if name == "OldColors" else name][~m],
                err_msg=f"Scene guard/readonly {name}",
            )


def check_bodies(l, c, b, a, e, contract):
    import numpy as np
    from body_validation import write_masks

    masks = {name: np.zeros(v.size, dtype=np.uint8) for name, v in b.items()}
    for kind in ("update_mprops", "free_step"):
        for name, m in write_masks(
            l, body_case(c, kind), {n: b[n] for n in BODY_BUFFERS}
        ).items():
            masks[name] = np.maximum(masks[name], m)
    if c.get("hub_pool"):
        masks["FirstSlot"][:] = (
            0  # Checked by exact hub ownership, independent of allocation order.
        )
    # These are body-integration coefficients, all six fields are floating point.
    masks["Params"][:] = 1
    for name in ("ColliderPoses",):
        m = masks[name].reshape(-1, l["pose_words"])
        m[: c.get("collider_count", c["body_count"]), :] = 2
        for field, width in (("pose_rotation", 4), ("pose_translation", 3)):
            m[
                : c.get("collider_count", c["body_count"]), l[field] : l[field] + width
            ] = 1
    errors = {}
    for name, m in masks.items():
        if not np.any(m):
            continue
        np.testing.assert_array_equal(
            a[name][m == 2], e[name][m == 2], err_msg=f"Exact scene body field {name}"
        )
        av, ev = a[name].view(np.float32)[m == 1], e[name].view(np.float32)[m == 1]
        if not np.isfinite(av).all() or not np.isfinite(ev).all():
            raise AssertionError(f"Nonfinite scene {name}")
        np.testing.assert_allclose(
            av,
            ev,
            atol=numeric_policy.tolerances(c, contract, name)[0],
            rtol=numeric_policy.tolerances(c, contract, name)[1],
            err_msg=name,
        )
        errors[name] = float(np.max(np.abs(av - ev))) if av.size else 0.0
    return errors


def check_collision(l, c, b, a, e, contract):
    import broadphase_replay as broad
    import narrow_replay as narrow
    import numpy as np

    per, nb = c["colliders_per_batch"], c["batches"]
    bc = dict(
        c,
        kind="broadphase_pairs",
        count=c["body_count"],
        colliders=per,
        capacity=per + 3,
        storage=c["body_storage"],
        vertices=1,
        initial_count=0,
        iterations=1,
        dispatch=per * per * nb,
        groups=max((per * per * nb + 63) // 64, 1),
    )
    # Last-frame collision uses pre-integration collider poses. Workspace beyond
    # the published pair count can retain preceding frames; capacity guards were
    # checked above independently, so normalize only that dead workspace here.
    bb = {name: b[name] for name in broad.BUFFERS}
    bb.update(Poses=a["ColliderPoses"], Pairs=a["Pairs"])
    aa = {name: a[name] for name in broad.BUFFERS}
    aa["Poses"] = a["ColliderPoses"]
    ee = {name: e[name] for name in broad.BUFFERS}
    ee["Poses"] = e["ColliderPoses"]
    if c.get("broadphase") == "lbvh":
        # The enclosing LBVH scene checker validates its independently reconstructed
        # complete overlap set and tree. Here retain only the common AABB check.
        bc.update(
            kind="world_aabbs",
            dispatch=c["body_count"],
            groups=max((c["body_count"] + 63) // 64, 1),
        )
        bb["PairCount"] = a["PairCount"]
        bc["initial_count"] = int(a["PairCount"][0])
    bc["buffer_words"] = {name: v.size for name, v in bb.items()}
    errors = broad.check_outputs(
        l, bc, bb, aa, ee, numeric_policy.field_contract(c, contract, "Aabbs")
    )
    count = int(a["PairCount"][0])
    pf = l["plan"]
    if count > c["pair_capacity"]:
        raise AssertionError("Scene pair overflow")
    for result in (a, e):
        if (
            result["Plan"][pf["bound"]] != count
            or result["Plan"][pf["pfm_base"]] != count
            or result["Plan"][pf["pfm_len"]] != 0
            or result["PfmCount"][0] != 0
        ):
            raise AssertionError("Scene analytic plan")
    pw, cw = l["pair_words"], l["contact_words"]
    ap = a["Pairs"].reshape(-1, pw)[:count]
    ep = e["Pairs"].reshape(-1, pw)[:count]
    index = {tuple(row): i for i, row in enumerate(ep)}
    rawref = a["RawContacts"].copy().reshape(-1, cw)
    for i, row in enumerate(ap):
        rawref[i] = e["RawContacts"].reshape(-1, cw)[index[tuple(row)]]
    pairs = np.zeros(c["contact_storage"] * pw, dtype=np.uint32)
    pairs[: len(a["Pairs"])] = a["Pairs"]
    bb = {name: a[name] for name in narrow.BUFFERS}
    bb.update(
        Poses=a["ColliderPoses"],
        Pairs=pairs,
        Plan=a["Plan"][: l["plan_words"]],
        Contacts=a["RawContacts"],
    )
    aa = dict(bb)
    ee = dict(bb, Contacts=rawref.ravel())
    nc = dict(
        c,
        kind="analytic_contacts",
        count=count,
        storage=c["contact_storage"],
        colliders=c["body_storage"],
        groups=1,
        iterations=1,
        buffer_words={name: v.size for name, v in bb.items()},
    )
    errors.update(
        narrow.check_outputs(
            l, nc, bb, aa, ee, numeric_policy.field_contract(c, contract, "Contacts")
        )
    )
    length = l["contact"]["contact"] + l["manifold"]["len"]
    for result in (a, e):
        order = result["InOrder"][:count]
        np.testing.assert_array_equal(
            np.sort(order),
            np.arange(count, dtype=np.uint32),
            err_msg="Canonical permutation",
        )
        raw = result["RawContacts"].reshape(-1, cw)
        ordered = raw[order].copy()
        ordered[ordered[:, length] == 0] = 0
        np.testing.assert_array_equal(
            result["Contacts"][: count * cw],
            ordered.ravel(),
            err_msg="Canonical copied contact",
        )
        fields = l["contact"]
        key = lambda row, fields=fields: (
            int(row[fields["colliders"]]),
            int(row[fields["colliders"] + 1]),
            int(row[fields["subshape"]]),
        )
        live = [key(row) for row in ordered if row[length]]
        if live != sorted(live) or len(set(live)) != len(live):
            raise AssertionError("Canonical contact identity order")
        if np.any(ordered[: len(live), length] == 0):
            raise AssertionError("Inert contact inside live canonical prefix")
    # Narrow checks already compared geometry by pair identity; canonicalization
    # must additionally agree on exact published identities and lengths.
    for field, width in (("colliders", 2), ("bodies", 2), ("subshape", 1)):
        off = l["contact"][field]
        np.testing.assert_array_equal(
            a["Contacts"].reshape(-1, cw)[:count, off : off + width],
            e["Contacts"].reshape(-1, cw)[:count, off : off + width],
        )
    np.testing.assert_array_equal(
        a["Contacts"].reshape(-1, cw)[:count, length],
        e["Contacts"].reshape(-1, cw)[:count, length],
    )
    return errors


def check_graph(l, c, a, e):
    import numpy as np

    count = int(a["Plan"][l["plan"]["bound"]])
    nb = c["batches"]
    n = c["body_count"]
    lf = l["link"]
    cf = l["contact"]
    length = cf["contact"] + l["manifold"]["len"]
    for name in (
        "Plan",
        "PairCount",
        "PfmCount",
        "PfmSortLen",
        "ContactsIndirect",
        "PfmIndirect",
        "MbIndirect",
        "Counts",
        "OldCounts",
        "Buckets",
        "Stats",
        "NumColors",
        "ColoringIndirect",
        "Indirect",
        "SortCount",
    ):
        np.testing.assert_array_equal(a[name], e[name], err_msg=f"Scene integer {name}")
    for name in ("Colors", "OldColors", "Colored"):
        np.testing.assert_array_equal(
            a[name][:count], e[name][:count], err_msg=f"Scene integer {name}"
        )
    for result in (a, e):
        contacts = result["Contacts"].reshape(-1, l["contact_words"])
        links = result["Links"].reshape(-1, l["link_words"])
        mass = result["WorldMass"].view(np.float32).reshape(-1, l["world_words"])
        dynamic = np.any(
            mass[:n, l["world_inv_mass"] : l["world_inv_mass"] + 3] != 0, axis=1
        )
        rows = [[] for _ in range(n)]
        buckets = [[] for _ in range(c["color_stride"] * nb)]
        writers = [set() for _ in range(c["color_stride"])]
        for i in range(count):
            link = links[i]
            if not contacts[i, length]:
                if link[lf["len"]] != 0:
                    raise AssertionError("Inert scene link")
                continue
            if not 1 <= link[lf["len"]] <= 4 or link[lf["contact"]] != i:
                raise AssertionError("Invalid live scene link")
            color = int(result["Colors"][i])
            if not 1 <= color < c["color_stride"] - 1:
                raise AssertionError(
                    "Scene has unsolved contacts; palette/iteration exhaustion"
                )
            bodies = contacts[i, cf["bodies"] : cf["bodies"] + 2]
            for side, body in zip(("a", "b"), bodies):
                body = int(body)
                if (
                    body >= n
                    or link[lf["solver_body_" + side]] != body
                    or (not c.get("hub_pool") and link[lf["vel_slot_" + side]] != body)
                ):
                    raise AssertionError("Scene link body ownership")
                if dynamic[body]:
                    rows[body].append(i)
                    writer = int(link[lf["vel_slot_" + side]])
                    if writer in writers[color]:
                        raise AssertionError("Scene color has concurrent writable body")
                    writers[color].add(writer)
            if bodies[0] % nb != bodies[1] % nb:
                raise AssertionError("Cross-batch scene contact")
            buckets[color * nb + int(bodies[0]) % nb].append(i)
        start = 0
        for body, row in enumerate(rows):
            end = start + len(row)
            if result["Counts"][body] != end:
                raise AssertionError("Scene adjacency count")
            np.testing.assert_array_equal(
                np.sort(result["Ids"][start:end]), sorted(row)
            )
            np.testing.assert_array_equal(result["SortedIds"][start:end], sorted(row))
            start = end
        ends = np.cumsum([len(row) for row in buckets], dtype=np.uint64)
        np.testing.assert_array_equal(result["Buckets"], ends)
        start = 0
        sorted_links = result["SortedLinks"].reshape(-1, l["link_words"])
        for row, end in zip(buckets, ends):
            slots = result["Indices"][row]
            np.testing.assert_array_equal(
                np.sort(slots),
                np.arange(start, end, dtype=np.uint32),
                err_msg="Scene bucket membership",
            )
            for source, dest in zip(row, slots):
                np.testing.assert_array_equal(
                    sorted_links[dest], links[source], err_msg="Scene copied link"
                )
            start = int(end)
        for source, dest in (
            ("Links", "OldLinks"),
            ("Counts", "OldCounts"),
            ("SortedIds", "OldIds"),
            ("Indices", "OldIndices"),
            ("Colors", "OldColors"),
            ("Tiles", "OldTiles"),
        ):
            np.testing.assert_array_equal(
                result[source], result[dest], err_msg="Scene persistence"
            )
        half = c["contact_storage"] * l["state_words"]
        np.testing.assert_array_equal(
            result["States"][:half],
            result["States"][half:],
            err_msg="Scene state persistence",
        )
    # Index into previous compacted storage is an allocation-dependent identity.
    # Compare all other native link words, and separately require the index to
    # address a legal previous allocation whenever it is live.
    ar = a["Links"].reshape(-1, l["link_words"])
    er = e["Links"].reshape(-1, l["link_words"])
    for i in range(count):
        if ar[i, lf["len"]]:
            keep = np.arange(l["link_words"]) != lf["previous_constraint_index"]
            if c.get("hub_pool"):
                keep[[lf["vel_slot_a"], lf["vel_slot_b"]]] = (
                    False  # Exact ownership is checked separately.
                )
            np.testing.assert_array_equal(
                ar[i, keep], er[i, keep], err_msg="Scene contact link fields"
            )
            for row in (ar[i], er[i]):
                if (
                    row[lf["previous_constraint"]] != 0xFFFFFFFF
                    and row[lf["previous_constraint_index"]] >= c["contact_capacity"]
                ):
                    raise AssertionError("Previous constraint allocation out of bounds")
    return count


def check_constraints(l, c, a, e, contract, count):
    if l.get("friction_model") == "twist":
        from twist_scene_validation import check_constraints as twist_check

        return twist_check(l, c, a, e, contract, count)
    import numpy as np

    for result in (a, e):
        numeric_policy.constraint_geometry(l, c, result)
    numeric_policy.constraint_response(l, c, a, e, contract)
    h, p = l["header"], l["point"]
    lf = l["link"]
    max_error = 0.0
    for cid in range(count):
        if a["Links"][cid * l["link_words"] + lf["len"]] == 0:
            continue
        response_mode = numeric_policy.policy_for(c, contract).get(
            "friction_comparison"
        )
        av = []
        ev = []
        for result, output in ((a, av), (e, ev)):
            slot = int(result["Indices"][cid])
            base = slot // 32 * l["tile_words"] + slot % 32 * 4
            tile = result["Tiles"]
            floats = tile.view(np.float32)
            length = int(tile[base + h["normal"] + 3])
            if not 1 <= length <= 4:
                raise AssertionError("Invalid scene tile length")
            words = []
            mask = []
            labels = []
            for name, off in h.items():
                words.extend(range(base + off, base + off + 4))
                mask.extend([True] * 3 + [name == "tangent"])
                labels.extend(
                    [
                        "TileDirections"
                        if name in ("normal", "tangent")
                        else "TileHeader"
                    ]
                    * 3
                    + ["TileMaterial"]
                )
            words.extend(range(base + l["coupling"], base + l["coupling"] + 4))
            mask.extend([True] * length + [False] * (4 - length))
            labels.extend(["TileResponse"] * 4)
            for k in range(length):
                pb = base + l["points"] + k * l["point_words"]
                for field, off in p.items():
                    words.extend(range(pb + off, pb + off + 4))
                    mask.extend([True] * 4)
                    labels.extend(
                        {
                            "arm_a": ["TileGeometry"] * 3 + ["TileImpulse"],
                            "arm_b": ["TileGeometry"] * 3 + ["TileCoefficients"],
                            "friction": [
                                "TileFrictionImpulse"
                                if response_mode
                                else "TileImpulse"
                            ]
                            * 2
                            + ["TileResponse"] * 2,
                            "local_a": ["TileGeometry"] * 4,
                            "local_b": ["TileGeometry"] * 3 + ["TileBias"],
                        }[field]
                    )
                normal = float(floats[pb + p["arm_a"] + 3])
                limit = normal * float(floats[base + h["tangent"] + 3])
                if normal < 0 or np.linalg.norm(
                    floats[pb + p["friction"] : pb + p["friction"] + 2]
                ) > limit + contract["atol"] + contract["rtol"] * abs(limit):
                    raise AssertionError("Scene unilateral/Coulomb impulse")
            cache = (
                slot // 32 * l["tile_words"]
                + l["warmstart"]
                + slot % 32 * 2 * l["velocity_words"]
            )
            for side in range(2):
                words.extend(
                    range(
                        cache + side * l["velocity_words"],
                        cache + (side + 1) * l["velocity_words"],
                    )
                )
                cm = np.zeros(l["velocity_words"], dtype=bool)
                for field in ("velocity_linear", "velocity_angular"):
                    cm[l[field] : l[field] + 3] = True
                mask.extend(cm)
                labels.extend(["TileCache"] * l["velocity_words"])
            compared = tile[words].copy()
            if c.get("hub_pool"):
                for side, tag in (("a", "diag_a"), ("b", "off_a")):
                    # Resolve a checked allocation-dependent slot to its exact body identity.
                    compared[words.index(base + h[tag] + 3)] = result["Links"][
                        cid * l["link_words"] + lf["solver_body_" + side]
                    ]
            output.extend((compared, np.array(mask), np.array(labels)))
        if not np.array_equal(av[1], ev[1]):
            raise AssertionError("Scene prepared point count differs")
        mask = av[1]
        np.testing.assert_array_equal(
            av[0][~mask], ev[0][~mask], err_msg="Exact scene tile IDs/cache tags"
        )
        af, ef = av[0].view(np.float32)[mask], ev[0].view(np.float32)[mask]
        if not np.isfinite(af).all() or not np.isfinite(ef).all():
            raise AssertionError("Nonfinite scene constraint")
        labels = av[2][mask]
        for field in np.unique(labels):
            if field == "TileResponse":
                continue
            if field == "TileFrictionImpulse":
                numeric_policy.check_friction_response(l, c, a, e, contract, cid)
                continue
            fallback = (
                contract["cached_warmstart_atol"]
                if field == "TileCache" and c["frames"] > 1
                else contract["atol"]
            )
            numeric_policy.check(
                c, contract, field, af[labels == field], ef[labels == field], fallback
            )
        max_error = max(max_error, float(np.max(np.abs(af - ef))))
    return {"Tiles": max_error}


def check_states(l, c, a, e, contract, count):
    import numpy as np

    s, cf, lf = l["state"], l["contact"], l["link"]
    mask = np.zeros(l["state_words"], dtype=bool)
    for pose in ("pose_a", "pose_b"):
        for field, width in (("pose_rotation", 4), ("pose_translation", 3)):
            start = s[pose] + l[field]
            mask[start : start + width] = True
    mask[[s["max_extent"], s["max_drift"]]] = True
    maximum = 0.0
    for cid in range(count):
        if not a["Links"][cid * l["link_words"] + lf["len"]]:
            continue
        rows = []
        for result in (a, e):
            row = result["States"].reshape(-1, l["state_words"])[c["new_base"] + cid]
            contact = result["Contacts"].reshape(-1, l["contact_words"])[cid]
            np.testing.assert_array_equal(
                row[s["colliders"] : s["colliders"] + 2],
                contact[cf["colliders"] : cf["colliders"] + 2],
            )
            # Fresh states copy the native collider poses exactly. Recycled states
            # are allowed to retain their earlier pose, but must still be finite.
            if result["Links"][cid * l["link_words"] + lf["recycled"]] == 0:
                for side, index in (("pose_a", 0), ("pose_b", 1)):
                    collider = int(contact[cf["colliders"] + index])
                    np.testing.assert_array_equal(
                        row[s[side] : s[side] + l["pose_words"]],
                        result["ColliderPoses"].reshape(-1, l["pose_words"])[collider],
                    )
            rows.append(row)
        av, ev = rows
        np.testing.assert_array_equal(
            av[~mask], ev[~mask], err_msg="Scene persistence state identities"
        )
        af, ef = av.view(np.float32)[mask], ev.view(np.float32)[mask]
        if not np.isfinite(af).all() or not np.isfinite(ef).all():
            raise AssertionError("Nonfinite scene persistence state")
        np.testing.assert_allclose(
            af,
            ef,
            atol=numeric_policy.tolerances(c, contract, "States")[0],
            rtol=numeric_policy.tolerances(c, contract, "States")[1],
            err_msg="Scene recycle state",
        )
        maximum = max(maximum, float(np.max(np.abs(af - ef))))
    return {"States": maximum}
