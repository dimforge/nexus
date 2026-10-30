"""Hash-inventoried source and initial storage for a resident frame ladder."""

import json

from frame_manifest import validate_frame


def artifacts(directory, contract, safe_file):
    def read(name):
        return json.loads(safe_file(directory, name).read_text())

    bundle = read("bindings.json")
    names = bundle.get("variants", [])
    if (
        bundle.get("schema") != 3
        or bundle.get("execution") != "resident_frames"
        or contract.get("execution") != "resident_frames"
        or type(bundle.get("max_steps")) is not int
        or not names
        or names != [f"variants/{i}" for i in range(len(names))]
        or set(bundle.get("storage_buffers", [])) != set(contract["inputs"])
        or len(bundle["storage_buffers"]) != len(contract["inputs"])
        or read("variants/variants.json") != [str(i) for i in range(len(names))]
    ):
        raise ValueError("Invalid resident frame bundle")
    required = {"bindings.json", "variants/variants.json"}
    previous = None
    for name in names:
        file = name + "/frame.json"
        frame = validate_frame(read(file))
        program = frame["program"]
        if frame["max_steps"] != bundle.get("max_steps") or set(
            program["storage_buffers"]
        ) != set(contract["inputs"]):
            raise ValueError("Inconsistent frame step/storage ABI")
        if previous is not None:
            if set(frame["persistent"]) != set(previous["persistent"]) or len(
                frame["gates"]
            ) != len(previous["gates"]):
                raise ValueError("Incompatible frame state/capacity gates")
            if any(
                a["id"] != b["id"]
                or a["counters"] != b["counters"]
                or a["capacity"] > b["capacity"]
                for a, b in zip(previous["gates"], frame["gates"])
            ) or not any(
                a["capacity"] < b["capacity"]
                for a, b in zip(previous["gates"], frame["gates"])
            ):
                raise ValueError("Frame capacity meanings changed or failed to grow")
            if any(
                frame["persistent"][n]["word_len"] < r["word_len"]
                for n, r in previous["persistent"].items()
            ):
                raise ValueError("Frame state migration truncates persistent data")
        required.add(file)
        for stage in program["stages"]:
            if not stage["source"].endswith(".ascend"):
                raise ValueError("Frame handoff requires Ascend stage sources")
            required.add(name + "/" + stage["source"])
        for n, size in program["buffer_bytes"].items():
            file = name + "/" + n + ".bin"
            if safe_file(directory, file).stat().st_size != size:
                raise ValueError(
                    "Frame initial storage size differs from specialization"
                )
            required.add(file)
        previous = frame
    if bundle["buffer_bytes"] != previous["program"]["buffer_bytes"]:
        raise ValueError("Returned frame storage must use the final specialization")
    for file in required:
        safe_file(directory, file)
    return required
