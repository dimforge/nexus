"""Compare fresh lowering with an existing prepared scene; no physics execution.

Usage: python source-probe.py REPOSITORY PREPARED_ROOT OUTPUT
PREPARED_ROOT contains verified cuda/ and ascend/ specializations. Native input
buffers and layout are reused deliberately; no expected physics output is read.
"""
import hashlib
import json
from pathlib import Path
import sys
import time

def main():
    repo, prepared, output = map(Path, sys.argv[1:])
    sys.path.insert(0, str(repo / "ports/tilelang"))
    from parity import evidence_inputs
    from program_cache import prepare_program
    from runtime_compile import load_variant
    import tilelang

    output.mkdir(parents=True, exist_ok=False)
    inputs = evidence_inputs(repo)
    report = {
        "schema": 1,
        "suite": "cross_host_source_comparison",
        "inputs": inputs,
        "compiler": tilelang.__version__,
        "status": "running",
        "fresh_reference_execution": "not_run; verified existing native input buffers",
        "device_execution": "not_run",
        "full_matrix": "not_run",
        "targets": [],
    }
    started = time.monotonic()
    try:
        for target in ("cuda", "ascend"):
            before = time.monotonic()
            source = prepared / target
            old = load_variant(source, expected_target=target)
            layout = json.loads((source / "layout.json").read_text())
            case = json.loads((source / "case.json").read_text())
            sources, program, frame = prepare_program(
                layout, case, old["initial"], target, None, workers=4
            )
            destination = output / target
            destination.mkdir()
            for name, code in sources.items():
                (destination / name).write_text(code)
            (destination / "frame.json").write_text(json.dumps(frame, indent=2) + "\n")
            different = sorted(
                n for n in sources.keys() | old["sources"].keys()
                if sources.get(n) != old["sources"].get(n)
            )
            row = {
                "target": target,
                "elapsed_seconds": round(time.monotonic() - before, 3),
                "stages": len(program["stages"]),
                "unique_sources": len(sources),
                "source_hashes": {
                    n: hashlib.sha256(code.encode()).hexdigest()
                    for n, code in sources.items()
                },
                "prepared_manifest_sha256": hashlib.sha256(
                    (source / "specialization.json").read_bytes()
                ).hexdigest(),
                "different_sources": different,
                "frame_equal": frame == old["manifest"],
            }
            row["status"] = "passed" if not different and row["frame_equal"] else "failed"
            report["targets"].append(row)
            print(target, row["status"], row["elapsed_seconds"], flush=True)
        if evidence_inputs(repo) != inputs:
            raise ValueError("Sources changed during comparison")
        report["status"] = "passed" if all(x["status"] == "passed" for x in report["targets"]) else "failed"
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    sys.exit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
