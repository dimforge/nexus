"""Portable resident-state checkpoints; payloads remain trusted kernel input."""


def validate_checkpoint(value):
    if not isinstance(value, dict) or set(value) != {
        "schema",
        "steps",
        "capacities",
        "regions",
    }:
        raise ValueError("Invalid frame checkpoint schema")
    if (
        type(value["schema"]) is not int
        or value["schema"] != 1
        or type(value["steps"]) is not int
        or not 0 <= value["steps"] <= 0xFFFFFFFFFFFFFFFF
    ):
        raise ValueError("Invalid checkpoint step")
    capacities, regions = value["capacities"], value["regions"]
    if (
        not isinstance(capacities, dict)
        or not capacities
        or any(
            not isinstance(k, str) or type(v) is not int or not 0 <= v <= 0xFFFFFFFF
            for k, v in capacities.items()
        )
    ):
        raise ValueError("Invalid checkpoint capacities")
    if (
        not isinstance(regions, dict)
        or not regions
        or any(
            not isinstance(k, str)
            or not isinstance(v, list)
            or not v
            or any(type(w) is not int or not 0 <= w <= 0xFFFFFFFF for w in v)
            for k, v in regions.items()
        )
    ):
        raise ValueError("Invalid checkpoint region words")
    return value


def fits(checkpoint, manifest):
    capacities = {g["id"]: g["capacity"] for g in manifest["gates"]}
    regions = manifest["persistent"]
    return (
        capacities.keys() == checkpoint["capacities"].keys()
        and regions.keys() == checkpoint["regions"].keys()
        and all(capacities[k] >= v for k, v in checkpoint["capacities"].items())
        and all(
            regions[k]["word_len"] >= len(v) for k, v in checkpoint["regions"].items()
        )
        and (
            manifest["max_steps"] is None
            or checkpoint["steps"] <= manifest["max_steps"]
        )
    )
