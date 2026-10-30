"""Resident frame control ABI, independent of any vendor compiler/runtime."""

import itertools

from ordered import validate_program


def validate_frame(frame):
    program = validate_program(frame["program"])
    if (
        frame.get("schema") != 1
        or program["iterations"] != 1
        or "max_steps" not in frame
        or (
            frame["max_steps"] is not None
            and (
                type(frame["max_steps"]) is not int
                or not 1 <= frame["max_steps"] <= 0xFFFFFFFF
            )
        )
        or (
            frame["max_steps"] is None
            and any(
                n in program["storage_buffers"] for n in ("Trajectory", "FrameIndex")
            )
        )
    ):
        raise ValueError("Invalid resident frame schema, iteration count or step limit")
    names = [s["name"] for s in program["stages"]]
    sizes = {n: b // 4 for n, b in program["buffer_bytes"].items()}
    last = -1
    seen = set()
    for gate in frame["gates"]:
        if not isinstance(gate["id"], str) or not gate["id"] or gate["id"] in seen:
            raise ValueError("Invalid capacity gate identity")
        seen.add(gate["id"])
        end = names.index(gate["after_stage"])
        if (
            not last <= end < len(names) - 1
            or type(gate["capacity"]) is not int
            or not 0 <= gate["capacity"] <= 0xFFFFFFFF
        ):
            raise ValueError("Capacity gates must be ordered before commit")
        last = end
        counters = set()
        if not gate["counters"]:
            raise ValueError("Missing capacity counters")
        for counter in gate["counters"]:
            n, w = counter["buffer"], counter["word"]
            if type(w) is not int or not 0 <= w < sizes.get(n, 0) or (n, w) in counters:
                raise ValueError("Invalid capacity counter")
            counters.add((n, w))
    if not frame["persistent"]:
        raise ValueError("Missing persistent state")
    spans = {}
    for name, r in frame["persistent"].items():
        start, length = r["word_start"], r["word_len"]
        if (
            not isinstance(name, str)
            or not name
            or type(start) is not int
            or type(length) is not int
            or start < 0
            or length < 1
            or start + length > sizes.get(r["buffer"], 0)
        ):
            raise ValueError("Invalid persistent region")
        spans.setdefault(r["buffer"], []).append((start, start + length))
    for ranges in spans.values():
        ranges.sort()
        if any(a[1] > b[0] for a, b in itertools.pairwise(ranges)):
            raise ValueError("Overlapping persistent regions")
    return frame


def persistent_regions(l, c):
    """State read by the following frame, before solver initialization.

    Exclude three outer guards from variable-capacity records. Copy the whole
    occupied tile because its SoA fields span all 32 lanes; unused lanes are
    initialized by native defaults and remain semantically inactive.
    """
    n, cap = c["body_count"], c["contact_capacity"]
    lengths = {
        "Poses": n * l["pose_words"],
        "Velocities": n * l["velocity_words"],
        "OldCounts": n,
        "OldLinks": cap * l["link_words"],
        "OldIds": 2 * cap,
        "OldIndices": cap,
        "OldColors": cap,
        "OldTiles": ((cap + 31) // 32) * l["tile_words"],
        "States": cap * l["state_words"],
    }
    if c.get("kind") == "runtime_scene":
        lengths["Gravity"] = c["buffer_words"]["Gravity"]
    if "FrameIndex" in c["buffer_words"]:
        lengths.update(FrameIndex=1, Trajectory=c["frames"] * c["trace_stride"])
    # Empty scenes still preserve their inert uploaded pose/velocity record.
    if not n:
        lengths.update(Poses=l["pose_words"], Velocities=l["velocity_words"])
    return {
        name: {"buffer": name, "word_start": 0, "word_len": length}
        for name, length in lengths.items()
        if length
    }


def from_program(l, c, plans, program):
    if len(plans) != len(program["stages"]):
        raise ValueError("Frame schedule differs from the compiled program")
    gates = []
    for plan, stage in zip(plans, program["stages"]):
        if plan["name"] != stage["name"] or plan["buffer_map"] != stage["buffer_map"]:
            raise ValueError("Frame stage differs from the compiled ABI")
        local_gates = plan.get("capacity_gates", [])
        if "capacity_gate" in plan:
            local_gates = [plan["capacity_gate"], *local_gates]
        for gate in local_gates:
            gates.append(
                dict(
                    gate,
                    after_stage=stage["name"],
                    counters=[
                        dict(counter, buffer=stage["buffer_map"][counter["buffer"]])
                        for counter in gate["counters"]
                    ],
                )
            )
    expected = (
        ["pairs", "deferred_pairs"]
        if c.get("general_collision", c["kind"] == "general_scene")
        else ["pairs"]
    )
    if c.get("color_checks"):
        expected += ["colors", "coloring_rounds"]
    if [g["id"] for g in gates] != expected:
        raise ValueError("Missing or changed scene capacity gates")
    return validate_frame(
        {
            "schema": 1,
            "program": program,
            "gates": gates,
            "persistent": persistent_regions(l, c),
            "max_steps": c["frames"] if "FrameIndex" in c["buffer_words"] else None,
        }
    )


def validate_growth(a, b):
    """Validate a new specialization before publishing it to a live session."""
    if (
        a["max_steps"] != b["max_steps"]
        or set(a["persistent"]) != set(b["persistent"])
        or len(a["gates"]) != len(b["gates"])
    ):
        raise ValueError("Incompatible variants")
    if any(
        x["id"] != y["id"]
        or x["counters"] != y["counters"]
        or x["capacity"] > y["capacity"]
        for x, y in zip(a["gates"], b["gates"])
    ):
        raise ValueError("Gate meanings changed or capacities shrank")
    if not any(
        x["capacity"] < y["capacity"] for x, y in zip(a["gates"], b["gates"])
    ) or any(
        b["persistent"][n]["word_len"] < region["word_len"]
        for n, region in a["persistent"].items()
    ):
        raise ValueError("Variants must grow without truncating state")
