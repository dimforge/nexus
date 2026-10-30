"""Export and execute native resident capacity ladders against fresh Rust scenes."""

import hashlib
import importlib.metadata
import json
import platform

import numpy as np
from cpu_oracle import export_fixtures
from frame_cpu import CpuFrameSession
from frame_manifest import from_program
from frame_scene_validation import check_outputs, validate_variants
from ordered import CompilationSession, lower_program
from parity import digest, evidence_inputs
from replay import HERE, ROOT, build_native, compiler_libraries, run, write_json
from scene_kernels import make_program
from stage_replay import parse_args


def load_case(directory):
    c = json.loads((directory / "case.json").read_text())
    b = {n: np.fromfile(directory / f"{n}.bin", np.uint32) for n in c["buffer_words"]}
    return c, b


def export_bundle(l, c, b, directory, target, compiler):
    variants = [
        load_case(directory / "variants" / str(i))
        for i in range(len(c["capacity_ladder"]))
    ]
    validate_variants(l, c, b, variants)
    loaded = []
    for i, (vc, initial) in enumerate(variants):
        plans = make_program(l, vc, target)
        sources, program = lower_program(plans, initial, target, session=compiler)
        frame = from_program(l, vc, plans, program)
        out = directory / "variants" / str(i)
        write_json(out / "frame.json", frame)
        for name, source in sources.items():
            (out / name).write_text(source)
        loaded.append({"manifest": frame, "sources": sources, "initial": initial})
    write_json(
        directory / "bindings.json",
        {
            "schema": 3,
            "execution": "resident_frames",
            "variants": ["variants/" + str(i) for i in range(len(variants))],
            "storage_buffers": list(b),
            "buffer_bytes": {n: v.nbytes for n, v in b.items()},
            "max_steps": c["frames"],
        },
    )
    # CLI variant directories are relative to their own root.
    write_json(
        directory / "variants/variants.json", [str(i) for i in range(len(variants))]
    )
    return loaded


def main():
    args = parse_args(
        ["analytic_frame_retry", "general_frame_retry", "joint_frame_retry"],
        "analytic_frame_retry",
    )
    output = args.output.resolve() / args.operation / args.target
    output.mkdir(parents=True, exist_ok=True)
    inputs = evidence_inputs(ROOT)
    report = {"schema": 1, "status": "preparing", "target": args.target}
    write_json(output / "report.json", report)
    profile = export_fixtures(
        "frame-scenes", output / "fixtures", extra=(args.operation,)
    )
    l = json.loads((output / "fixtures/layout.json").read_text())
    k = json.loads((HERE / f"contracts/{args.operation}.json").read_text())
    fixtures = output / "fixtures" / args.operation
    suite = json.loads((fixtures / "suite.json").read_text())
    report.update(
        feature=k["feature"],
        inputs=inputs,
        suite="portable",
        level="source_lowered" if args.compile_only else "executed",
        reference_commit=run(["git", "-C", ROOT, "rev-parse", "HEAD"]),
        layout_sha256=digest(output / "fixtures/layout.json"),
        compiler_libraries=compiler_libraries(),
        cases=[],
        environment={
            "python": platform.python_version(),
            "os": platform.platform(),
            "rustflags": profile["rustflags"],
            "rustc": profile["rustc"],
            "cpu_oracle": profile,
            **{
                key: importlib.metadata.version(package)
                for key, package in [
                    ("tilelang", "tilelang"),
                    ("torch", "torch"),
                    ("numpy", "numpy"),
                    ("tvm_ffi", "apache-tvm-ffi"),
                    ("z3", "z3-solver"),
                ]
            },
        },
        limitations=[
            "Ascend source lowering is not vendor compilation, execution or synchronization qualification.",
            "Registered bounded fixture ladders do not establish arbitrary production scene support.",
        ],
    )
    compiler = CompilationSession()
    active = None
    try:
        runner = (
            build_native(frame=True)
            if args.target == "metal" and not args.compile_only
            else None
        )
        for c in suite["cases"]:
            active = c["name"]
            directory = fixtures / active
            _, before = load_case(directory)
            loaded = export_bundle(l, c, before, directory, args.target, compiler)
            row = {
                "name": active,
                "count": c["count"],
                "status": "passed",
                "fixture": c,
                "execution": "resident_frames",
            }
            if not args.compile_only:
                if args.target == "c":
                    runtime = CpuFrameSession(loaded, directory / "native-c")
                    stats = [runtime.step() for _ in range(c["frames"])]
                    actual = runtime.storage
                    report["environment"].update(runtime.environment)
                else:
                    report["environment"].update(
                        json.loads(run([runner, directory / "variants"]))
                    )
                    stats = json.loads((directory / "variants/stats.json").read_text())
                    actual = {
                        n: np.fromfile(
                            directory / "variants" / f"{n}.actual.bin", np.uint32
                        )
                        for n in before
                    }
                write_json(directory / "stats.json", stats)
                for n, v in actual.items():
                    v.tofile(directory / f"{n}.actual.bin")
                expected = {
                    n: np.fromfile(directory / f"{n}.expected.bin", np.uint32)
                    if n in k["outputs"]
                    else v.copy()
                    for n, v in before.items()
                }
                row["max_absolute_errors"] = check_outputs(
                    l, c, before, actual, expected, k, stats
                )
                row["diagnostics"] = {
                    "growth_frames": [s["step"] for s in stats if s["retries"]],
                    "final_variant": stats[-1]["variant"],
                }
            files = {
                str(p.relative_to(directory)): digest(p)
                for p in directory.rglob("*")
                if p.is_file() and "native-c" not in p.relative_to(directory).parts
            }
            row["files"] = dict(sorted(files.items()))
            row["source_sha256"] = hashlib.sha256(
                "".join(
                    source for v in loaded for _, source in sorted(v["sources"].items())
                ).encode()
            ).hexdigest()
            report["cases"].append(row)
            print(f"{args.target}: {args.operation}/{active}: passed", flush=True)
        if inputs != evidence_inputs(ROOT):
            raise ValueError("Sources changed during frame replay")
        report.update(status="passed", compilation_reuse=compiler.hits)
    except Exception as error:
        report.update(status="failed", error=str(error), failed_case=active)
        raise
    finally:
        write_json(output / "report.json", report)


if __name__ == "__main__":
    main()
