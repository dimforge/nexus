"""Independent hub ownership, mass scaling, and split-slot identity checks."""

import numpy as np

BUFFERS = ("HubList", "SlotBody", "SlotConstraint", "SlotsIndirect", "AverageIndirect")


def validate(l, c, b):
    pool, base = c["hub_pool"], c["body_storage"]
    if (
        c["native_body_slots"] != (c["body_capacity_per_batch"] - 3) * c["batches"]
        or pool != max(c["native_body_slots"], 1024)
        or c["solver_storage"] != base + pool + 3
    ):
        raise ValueError("Invalid native hub storage")
    sizes = {
        "HubList": pool // l["hub_min_constraints"] + 4,
        "SlotBody": pool + 3,
        "SlotConstraint": pool + 3,
        "SlotsIndirect": 9,
        "AverageIndirect": 9,
        "SolverVelocities": c["solver_storage"] * l["velocity_words"],
    }
    if any(b[name].size != size for name, size in sizes.items()):
        raise ValueError("Invalid native hub buffer dimensions")
    expected = {"base": base, "pool": pool, "hubs": 0, "slots": 0}
    if any(int(b["HubCounts"][l["hub"][key]]) != val for key, val in expected.items()):
        raise ValueError("Invalid initial native hub counters")
    if not np.array_equal(b["BodyGroup"], np.arange(base, dtype=np.uint32)):
        raise ValueError("Ordinary native body groups must use body IDs")


def check(l, c, r):
    base, pool = c["body_storage"], c["hub_pool"]
    h = {name: int(r["HubCounts"][off]) for name, off in l["hub"].items()}
    assert h["base"] == base and h["pool"] == pool, "Hub capacity changed"
    counts = r["Counts"].astype(np.int64)
    degrees = np.diff(np.r_[0, counts])
    assert np.all(degrees >= 0) and np.all(degrees[c["body_count"] :] == 0), (
        "Invalid hub adjacency"
    )
    mass = r["WorldMass"].view(np.float32).reshape(-1, l["world_words"])
    dynamic = np.any(
        mass[:, l["world_inv_mass"] : l["world_inv_mass"] + 3] != 0, axis=1
    )
    candidates = {
        i: int(degrees[i])
        for i in range(base)
        if dynamic[i] and degrees[i] > l["hub_min_constraints"]
    }
    assert h["slots"] == sum(candidates.values()), "Incorrect requested hub slots"
    assert (
        0 <= h["hubs"] <= min(len(candidates), pool // l["hub_min_constraints"] + 1)
    ), "Incorrect live hubs"
    selected = list(map(int, r["HubList"][: h["hubs"]]))
    assert len(selected) == len(set(selected)) and all(
        i in candidates for i in selected
    ), "Invalid hub membership"
    end = 0
    for start, n in sorted((int(r["FirstSlot"][i]), candidates[i]) for i in selected):
        assert start == end and start + n <= pool, (
            "Hub reservations overlap or exceed capacity"
        )
        end += n
    remaining = [n for i, n in candidates.items() if i not in selected]
    assert not remaining or any(n > pool - end for n in remaining), (
        "A fitting hub was omitted"
    )
    assert all(
        r["FirstSlot"][i] == l["not_a_hub"] for i in range(base) if i not in selected
    ), "Nonhub has split slots"
    np.testing.assert_array_equal(
        r["SlotsIndirect"][:3], [(min(h["slots"], pool) + 63) // 64, 1, 1]
    )
    np.testing.assert_array_equal(
        r["AverageIndirect"][:3], [min(h["hubs"], l["hub_average_workgroups"]), 1, 1]
    )
    lf = l["link"]
    links = r["Links"].reshape(-1, l["link_words"])
    slot_by_contact = {}
    for body in selected:
        begin = int(counts[body - 1]) if body else 0
        for offset, cid in enumerate(map(int, r["Ids"][begin : int(counts[body])])):
            slot = int(r["FirstSlot"][body]) + offset
            assert r["SlotBody"][slot] == body and r["SlotConstraint"][slot] == cid, (
                "Split slot ownership mismatch"
            )
            slot_by_contact[body, cid] = base + slot
    for cid in range(int(r["Plan"][l["plan"]["bound"]])):
        row = links[cid]
        if not row[lf["len"]]:
            continue
        compact = int(r["Indices"][cid])
        tile_base = compact // 32 * l["tile_words"] + compact % 32 * 4
        tile = r["Tiles"]
        floats = tile.view(np.float32)
        for side, tag in [("a", "diag_a"), ("b", "off_a")]:
            body = int(row[lf["solver_body_" + side]])
            assert body < c["body_count"], "Split link body out of bounds"
            slot = slot_by_contact.get((body, cid), body)
            scale = np.float32(candidates[body] if body in selected else 1)
            assert row[lf["vel_slot_" + side]] == slot, "Link split slot mismatch"
            assert row.view(np.float32)[lf["mass_scale_" + side]] == scale, (
                "Split mass scale mismatch"
            )
            assert tile[tile_base + l["header"][tag] + 3] == slot, (
                "Prepared split slot mismatch"
            )
            assert tile[tile_base + l["header"]["body_" + side] + 3] == body, (
                "Prepared body tag mismatch"
            )
            # World inertia is a column-major Mat4; the tile stores its symmetric 3x3 part.
            inertia = (
                mass[body, l["world_inertia"] : l["world_inertia"] + 16].reshape(4, 4).T
            )
            expected = {
                "body_" + side: mass[
                    body, l["world_inv_mass"] : l["world_inv_mass"] + 3
                ],
                "diag_" + side: np.diag(inertia)[:3],
                "off_" + side: inertia[[0, 0, 1], [1, 2, 2]],
            }
            for field, values in expected.items():
                start = tile_base + l["header"][field]
                np.testing.assert_allclose(
                    floats[start : start + 3],
                    values * scale,
                    atol=2e-6,
                    rtol=2e-5,
                    err_msg="Independent split mass/inertia",
                )
    return slot_by_contact


def compare_velocities(l, c, a, e, k):
    maps = [check(l, c, r) for r in (a, e)]
    assert maps[0].keys() == maps[1].keys(), "Different accepted hub memberships"
    rows = [r["SolverVelocities"].reshape(-1, l["velocity_words"]) for r in (a, e)]
    floats = [
        l[f] + i for f in ("velocity_linear", "velocity_angular") for i in range(3)
    ]
    pads = [i for i in range(l["velocity_words"]) if i not in floats]
    for identity in maps[0]:
        av, ev = [row[m[identity]] for row, m in zip(rows, maps)]
        assert (
            np.isfinite(av.view(np.float32)[floats]).all()
            and np.isfinite(ev.view(np.float32)[floats]).all()
        ), "Nonfinite split velocity"
        np.testing.assert_array_equal(
            av[pads], ev[pads], err_msg="Hub velocity padding"
        )
        np.testing.assert_allclose(
            av.view(np.float32)[floats],
            ev.view(np.float32)[floats],
            atol=k["atol"],
            rtol=k["rtol"],
            err_msg="Hub velocity by body/contact identity",
        )
