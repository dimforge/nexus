"""Compiler/export helpers for ordered kernels over persistent native buffers."""

import hashlib
import os
import re
from collections import OrderedDict


class CompilationSession:
    """Bounded reuse within one runner process; no fixtures or outputs are cached."""

    def __init__(self, capacity=256):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("Invalid compiler session capacity")
        self.capacity = capacity
        self.lowered = OrderedDict()
        self.compiled = OrderedDict()
        self.hits = {"lowering": 0, "c_compilation": 0}

    def get(self, cache, key, build, counter):
        if key in cache:
            self.hits[counter] += 1
            cache.move_to_end(key)
            return cache[key]
        value = build()
        cache[key] = value
        if len(cache) > self.capacity:
            cache.popitem(last=False)
        return value

    def lower(self, kernel, target):
        import tilelang
        from tilelang import tvm

        # Serialize the complete IR, including buffers, constants and attributes.
        # Target and full IR equality are required, not a function name or shape.
        key = (target, tvm.ir.save_json(kernel))
        requested = {"kind": "cuda", "arch": "sm_80"} if target == "cuda" else target
        return self.get(
            self.lowered,
            key,
            lambda: tilelang.lower(
                kernel,
                target=requested,
                enable_host_codegen=False,
                enable_device_compile=False,
            ),
            "lowering",
        )

    def compile_c(self, source, bindings, directory):
        from replay import compile_c

        key = (
            os.environ.get("CXX", "c++"),
            source,
            bindings["entry"],
            tuple(bindings["buffers"]),
        )

        def build():
            # dlopen may return an already loaded library at the same pathname.
            # Give changed source/ABI/compiler settings a distinct native path.
            native_dir = directory / hashlib.sha256(repr(key).encode()).hexdigest()
            native_dir.mkdir(parents=True, exist_ok=True)
            return compile_c(source, bindings, native_dir)

        return self.get(self.compiled, key, build, "c_compilation")


def validate_program(program):
    """Validate the export ABI without importing a compiler or executing code."""

    def positive(value):
        return type(value) is int and 0 < value <= (1 << 64) - 1

    def storage(manifest):
        names = manifest.get("storage_buffers", [])
        sizes = manifest.get("buffer_bytes", {})
        if (
            not names
            or len(names) != len(set(names))
            or set(names) != set(sizes)
            or any(not re.fullmatch(r"[A-Za-z0-9_]+", n) for n in names)
            or any(not positive(n) or n % 4 for n in sizes.values())
        ):
            raise ValueError("Invalid program storage")
        if (
            not positive(manifest.get("iterations"))
            or manifest["iterations"] > 0xFFFFFFFF
        ):
            raise ValueError("Invalid program iteration count")
        return sizes

    if program.get("schema") != 2 or not program.get("stages"):
        raise ValueError("Unsupported or empty ordered program")
    physical = storage(program)
    names = set()
    for stage in program["stages"]:
        name = stage.get("name")
        source = stage.get("source", "")
        if (
            not isinstance(name, str)
            or not name
            or name in names
            or not re.fullmatch(r"[A-Za-z0-9_.-]+", source)
            or source in (".", "..")
        ):
            raise ValueError("Invalid program stage name or source")
        names.add(name)
        bindings = stage["bindings"]
        local = storage(bindings)
        order = bindings.get("buffers", [])
        if (
            bindings.get("schema") != 1
            or not bindings.get("entry")
            or len(order) != len(set(order))
            or not set(order).issubset(local)
            or not positive(bindings.get("blocks"))
            or not positive(bindings.get("threads"))
            or bindings["blocks"] * bindings["threads"] > (1 << 64) - 1
        ):
            raise ValueError("Invalid program kernel ABI")
        mapping = stage["buffer_map"]
        if (
            set(mapping) != set(local)
            or len(set(mapping.values())) != len(mapping)
            or any(physical.get(p) != local[n] for n, p in mapping.items())
        ):
            raise ValueError("Missing, invalid or aliased program buffer mapping")
    return program


def lower_program(plans, arrays, target, iterations=1, *, session=None):
    from compiler_intrinsics import register_intrinsics

    register_intrinsics(target)

    program = {
        "schema": 2,
        "storage_buffers": list(arrays),
        "buffer_bytes": {n: a.nbytes for n, a in arrays.items()},
        "iterations": iterations,
        "stages": [],
    }
    sources = {}
    lowered = {}
    session = session or CompilationSession()
    for i, plan in enumerate(plans):
        # Repeated stages may deliberately share the same immutable PrimFunc.
        # Reuse lowering only within this program, by object identity.
        key = id(plan["kernel"])
        if key not in lowered:
            lowered[key] = session.lower(plan["kernel"], target)
        artifact = lowered[key]
        functions = list(artifact.device_mod.functions.values())
        if len(functions) != 1:
            raise ValueError("Expected one entry point per ordered stage")
        function = functions[0]
        mapping = plan["buffer_map"]
        order = [str(p.name) for p in function.params]
        source = artifact.kernel_source
        if target == "metal":
            source = "#pragma clang fp contract(off)\n" + source
        filename = f"kernel{'.' + str(i) if i else ''}.{target}"
        sources[filename] = source
        bindings = {
            "schema": 1,
            "entry": str(function.attrs["global_symbol"]),
            "buffers": order,
            "storage_buffers": list(mapping),
            "buffer_bytes": {n: arrays[p].nbytes for n, p in mapping.items()},
            "blocks": plan["blocks"],
            "threads": plan["threads"],
            "iterations": 1,
        }
        program["stages"].append(
            {
                "name": plan["name"],
                "source": filename,
                "bindings": bindings,
                "buffer_map": mapping,
            }
        )
    return sources, validate_program(program)


def execute_c_program(sources, program, directory, arrays, *, session=None):
    from replay import execute_c

    validate_program(program)
    environment = {}
    session = session or CompilationSession()
    compiled = {}
    # All stages receive views of the same physical arrays. No CPU arithmetic or
    # intermediate readback substitutes for generated kernels.
    for _ in range(program["iterations"]):
        for i, stage in enumerate(program["stages"]):
            stage_dir = directory / f"stage-{i}"
            stage_dir.mkdir(exist_ok=True)
            local = {n: arrays[p] for n, p in stage["buffer_map"].items()}
            source, bindings = sources[stage["source"]], stage["bindings"]
            key = (source, bindings["entry"], tuple(bindings["buffers"]))
            if key not in compiled:
                compiled[key] = session.compile_c(source, bindings, stage_dir)
            environment.update(
                execute_c(source, bindings, stage_dir, local, compiled=compiled[key])
            )
    return environment
