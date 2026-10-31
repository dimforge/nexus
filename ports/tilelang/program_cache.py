"""Cache a compiled schedule, never native inputs, state or expected results."""

import ast
import importlib.util
import json
from pathlib import Path

from frame_manifest import from_program, validate_frame
from kernel_cache import digest, encoded
from ordered import CompilationSession, lower_program

HERE = Path(__file__).resolve().parent


def builder_sources(
    root=HERE, roots=("runtime_compile", "scene_kernels", "program_cache")
):
    """Hash the transitive local Python imports, including function-local imports.

    Re-read the import graph on every request so added/deleted dependencies cannot
    disappear behind a saved file list. External compiler packages are fingerprinted
    separately. The parity evidence envelope is independent and remains unchanged.
    """
    todo, found = list(roots), {}
    while todo:
        module = todo.pop()
        stem = root.joinpath(*module.split("."))
        path = stem.with_suffix(".py")
        package = stem / "__init__.py"
        if package.is_file():
            path = package
        if not path.is_file() or str(path.relative_to(root)) in found:
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Compiler source outside port")
        source = path.read_bytes()
        found[str(path.relative_to(root))] = digest(source)
        # Python executes parent package initializers before importing a module.
        todo.extend(
            ".".join(module.split(".")[:i]) for i in range(1, len(module.split(".")))
        )
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                todo.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                name = node.module or ""
                if node.level:
                    parent = (
                        module
                        if path.name == "__init__.py"
                        else module.rpartition(".")[0]
                    )
                    name = importlib.util.resolve_name("." * node.level + name, parent)
                todo.append(name)
                todo.extend(name + "." + alias.name for alias in node.names)
    return dict(sorted(found.items()))


def compact_sources(sources, program):
    """Identical source files share storage, but retain every ordered dispatch."""
    canonical, unique = {}, {}
    for stage in program["stages"]:
        name = stage["source"]
        source = sources[name]
        abi = stage["bindings"]
        key = (source, abi["entry"], tuple(abi["buffers"]))
        if key not in canonical:
            canonical[key] = name
            unique[name] = source
        stage["source"] = canonical[key]
    return unique


def prepare_program(layout, case, initial, target, cache, *, workers=1, timings=None):
    from time import perf_counter

    timings = {} if timings is None else timings
    sizes = {name: array.nbytes for name, array in initial.items()}
    key = {
        "builder_sources": builder_sources(),
        "layout": layout,
        "case": case,
        "target": target,
        "buffers": {
            name: {
                "bytes": array.nbytes,
                "dtype": str(array.dtype),
                "shape": list(array.shape),
            }
            for name, array in initial.items()
        },
    }

    def build():
        from scene_kernels import make_program

        start = perf_counter()
        plans = make_program(layout, case, target)
        timings["kernel_construction"] = perf_counter() - start
        start = perf_counter()
        session = CompilationSession(persistent_cache=cache, workers=workers)
        sources, program = lower_program(plans, initial, target, session=session)
        timings["lowering"] = perf_counter() - start
        timings["lowering_tasks"] = session.lowering_timings
        sources = compact_sources(sources, program)
        frame = from_program(layout, case, plans, program)
        return encoded({"sources": sources, "frame": frame})

    payload = cache.artifact("program", key, build) if cache is not None else build()
    value = json.loads(payload)
    frame, sources = validate_frame(value["frame"]), value["sources"]
    program = frame["program"]
    if program["storage_buffers"] != list(initial) or program["buffer_bytes"] != sizes:
        raise ValueError("Cached program storage differs from fresh upload")
    if set(sources) != {s["source"] for s in program["stages"]} or any(
        Path(name).name != name or not isinstance(source, str)
        for name, source in sources.items()
    ):
        raise ValueError("Invalid cached program sources")
    return sources, program, frame
