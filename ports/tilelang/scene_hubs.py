"""Launch the mirrored hub stages within a resident native frame."""

BUFFERS = ("HubList", "SlotBody", "SlotConstraint", "SlotsIndirect", "AverageIndirect")


def append(plans, layout, case, target, kind, cache):
    import hub_kernels

    pool = case.get("hub_pool", 0)
    if not pool:
        return
    sizes = case["buffer_words"]
    groups = {
        "hub_assign": max((case["body_storage"] + 63) // 64, 1),
        "hub_dispatch": 1,
        "hub_split": min(
            pool // layout["hub_min_constraints"] + 1, layout["hub_average_workgroups"]
        ),
        "hub_scatter": (pool + 63) // 64,
        "hub_average": min(
            pool // layout["hub_min_constraints"] + 1, layout["hub_average_workgroups"]
        ),
    }[kind]
    threads = 1 if kind == "hub_dispatch" else 64
    mapping = {
        name: name
        for name in (
            "Counts",
            "Ids",
            "WorldMass",
            "BodyGroup",
            "HubCounts",
            "Links",
            "BatchIds",
        )
        + BUFFERS
    }
    mapping.update(FirstSlots="FirstSlot", Velocities="SolverVelocities")
    if kind not in cache:
        config = dict(
            case,
            kind=kind,
            count=case["body_count"],
            groups=groups,
            threads=threads,
            storage=case["body_storage"],
            ids_storage=sizes["Ids"],
            hub_storage=sizes["HubList"],
            constraints=case["contact_storage"],
            slot_storage=sizes["SlotBody"],
            velocities=case["solver_storage"],
        )
        cache[kind] = hub_kernels.make_kernel(layout, config, target)
    plans.append(
        {
            "name": f"{kind}_{len(plans)}",
            "kernel": cache[kind],
            "buffer_map": mapping,
            "blocks": groups,
            "threads": threads,
        }
    )
