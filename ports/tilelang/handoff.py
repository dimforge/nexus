#!/usr/bin/env python3
"""Build and verify a versioned, hardware-free Ascend kernel handoff.

The bundle is an engineering input, not an Ascend runtime or qualification result.
It contains source, bindings, fixtures, contracts and exact provenance. Executed
outputs from an eventual device session can be checked offline with verify-results.
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from cpu_oracle import verify_probe_file
from parity import (
    PORT,
    ROOT,
    digest,
    evidence_inputs,
    evidence_state,
    feature_contract,
    manifest,
    read_json,
    status,
    write_json,
)


def safe_file(root, relative):
    path = root / relative
    if (
        Path(relative).is_absolute()
        or ".." in Path(relative).parts
        or path.is_symlink()
        or not path.resolve().is_relative_to(root.resolve())
    ):
        raise ValueError(f"Unsafe artifact path: {relative}")
    if not path.is_file():
        raise ValueError(f"Missing artifact: {relative}")
    return path


def ordered_sources(directory, contract):
    """An ordered contract cannot be packaged as its first kernel alone."""
    from ordered import validate_program

    bindings = read_json(safe_file(directory, "bindings.json"))
    if contract.get("execution") == "ordered_kernels" or bindings.get("schema") == 2:
        validate_program(bindings)
        if set(bindings["storage_buffers"]) != set(contract["inputs"]):
            raise ValueError("Ordered storage differs from contract")
        sources = {stage["source"] for stage in bindings["stages"]}
        if bindings["stages"][0]["source"] != "kernel.ascend" or any(
            not source.endswith(".ascend") for source in sources
        ):
            raise ValueError("Ordered handoff requires Ascend stage sources")
        return sources
    return set()


def verify_bundle(root):
    bundle = read_json(root / "manifest.json")
    if bundle.get("schema") != 1:
        raise ValueError("Unsupported handoff schema")
    if (root / "manifest.sha256").read_text().strip() != digest(root / "manifest.json"):
        raise ValueError("Modified handoff manifest")
    for relative, expected in bundle["files"].items():
        if digest(safe_file(root, relative)) != expected:
            raise ValueError(f"Modified handoff artifact: {relative}")
    oracles = bundle.get("cpu_oracles", {})
    if set(oracles) != {case["feature"] for case in bundle["cases"]}:
        raise ValueError("Missing CPU oracle provenance in handoff")
    for oracle in oracles.values():
        if not {oracle["report"], oracle["llvm_ir"]}.issubset(bundle["files"]):
            raise ValueError("CPU oracle files missing from handoff hash inventory")
        report = read_json(safe_file(root, oracle["report"]))
        verify_probe_file(
            safe_file(root, oracle["llvm_ir"]), report["environment"]["cpu_oracle"]
        )
    for case in bundle["cases"]:
        contract = read_json(safe_file(root, case["contract"]))
        required = {case["layout"], case["contract"]} | {
            case["directory"] + "/" + name
            for name in (
                [f"{n}.bin" for n in contract["inputs"]]
                + [f"{n}.expected.bin" for n in contract["outputs"]]
                + ["bindings.json", "kernel.ascend"]
            )
        }
        required.update(
            case["directory"] + "/" + source
            for source in ordered_sources(root / case["directory"], contract)
        )
        if not required.issubset(bundle["files"]):
            raise ValueError("Case artifacts missing from handoff hash inventory")
    keys = [(c["feature"], c["case"]["name"]) for c in bundle["cases"]]
    if not keys or len(keys) != len(set(keys)):
        raise ValueError("Empty or duplicate handoff cases")
    return bundle


def build(root, output, results):
    if output.exists():
        raise ValueError("Choose a new output directory; handoffs are immutable")
    if subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain"], text=True
    ).strip():
        raise ValueError(
            "Commit source and accepted evidence before creating a handoff"
        )
    config = manifest(root)
    inputs = evidence_inputs(root)
    tracked = set(
        subprocess.check_output(
            ["git", "-C", str(root), "ls-files", "-z"], text=True
        ).split("\0")
    )
    if any(path not in tracked for group in inputs.values() for path in group):
        raise ValueError(
            "Every fingerprinted source must be tracked in the handoff commit"
        )
    current = status(root)
    if current["source_reconciliation"] != "current" or current["inventory"]["errors"]:
        raise ValueError("Source/inventory reconciliation is required")
    for feature in config["features"]:
        if current["feature_backends"][feature["id"]]["c"]["state"] != "executed":
            raise ValueError(f"Current CPU evidence required: {feature['id']}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="nexus-handoff-", dir=output.parent
    ) as temp:
        stage = Path(temp) / "bundle"
        stage.mkdir()
        files = {}

        def copy(source, relative):
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            files[relative] = digest(target)

        source_archive = stage / "nexus-source.tar"
        with source_archive.open("wb") as stream:
            subprocess.run(
                ["git", "-C", str(root), "archive", "--format=tar", "HEAD"],
                stdout=stream,
                check=True,
            )
        files["nexus-source.tar"] = digest(source_archive)
        cases = []
        oracles = {}
        for feature in config["features"]:
            op = feature["operation"]
            run_root = results / ("" if op == "solve" else op) / "ascend"
            report = read_json(run_root / "report.json")
            if (
                report.get("feature") != feature["id"]
                or report.get("target") != "ascend"
                or evidence_state(root, report, inputs) != "source_lowered"
            ):
                raise ValueError(
                    f"Current Ascend source report required: {feature['id']}"
                )
            copy(run_root / "report.json", f"reports/{op}.json")
            probe = run_root / "cpu-oracle-noalias.ll"
            verify_probe_file(probe, report["environment"]["cpu_oracle"])
            copy(probe, f"reports/{op}-noalias.ll")
            oracles[feature["id"]] = {
                "report": f"reports/{op}.json",
                "llvm_ir": f"reports/{op}-noalias.ll",
            }
            layout = run_root / "fixtures/layout.json"
            if digest(layout) != report["layout_sha256"]:
                raise ValueError(f"Modified layout: {op}")
            copy(layout, f"layouts/{op}.json")
            contract_path = feature.get("contract", str(PORT / "contract.json"))
            copy(root / contract_path, f"contracts/{op}.json")
            contract = feature_contract(root, feature)
            for outcome in report["cases"]:
                case = outcome["fixture"]
                case_root = (
                    run_root / "fixtures" / ("" if op == "solve" else op) / case["name"]
                )
                relative = f"cases/{op}/{case['name']}"
                required = (
                    {"bindings.json", "kernel.ascend"}
                    | {f"{name}.bin" for name in contract["inputs"]}
                    | {
                        f"{name}.expected.bin"
                        for name in feature_contract(root, feature)["outputs"]
                    }
                )
                required.update(ordered_sources(case_root, contract))
                if not required.issubset(outcome["files"]):
                    raise ValueError(f"Incomplete fixture receipt: {op}/{case['name']}")
                for name, expected in outcome["files"].items():
                    path = safe_file(case_root, name)
                    if digest(path) != expected:
                        raise ValueError(
                            f"Modified fixture: {op}/{case['name']}/{name}"
                        )
                    copy(path, f"{relative}/{name}")
                cases.append(
                    {
                        "feature": feature["id"],
                        "operation": op,
                        "case": case,
                        "directory": relative,
                        "layout": f"layouts/{op}.json",
                        "contract": f"contracts/{op}.json",
                    }
                )
        for name in (
            "compiler.lock.json",
            "requirements.txt",
            "inventory.json",
            "contract.json",
            "HANDOFF.md",
            "session.schema.json",
        ):
            copy(root / PORT / name, name)
        bundle = {
            "schema": 1,
            "git_commit": subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
            ).strip(),
            "inputs": inputs,
            "target_profile": "provisional Ascend 950 SIMT; confirm actual device and CANN with Huawei",
            "qualification": "source_lowered_only",
            "runtime_adapter": "not_implemented; requires selected CANN SDK",
            "known_gaps": current["known_gaps"],
            "cases": cases,
            "cpu_oracles": oracles,
            "files": dict(sorted(files.items())),
        }
        write_json(stage / "manifest.json", bundle)
        (stage / "manifest.sha256").write_text(digest(stage / "manifest.json") + "\n")
        verify_bundle(stage)
        if evidence_inputs(root) != inputs:
            raise ValueError("Sources changed during packaging")
        stage.rename(output)
    return output / "manifest.json"


def probe(output):
    import os
    import platform

    tool = shutil.which("npu-smi")
    result = {
        "schema": 1,
        "level": "environment_probe",
        "os": platform.platform(),
        "machine": platform.machine(),
        "npu_smi": tool,
        "ascend_home_path": os.environ.get("ASCEND_HOME_PATH"),
        "status": "unavailable",
    }
    if tool:
        try:
            command = subprocess.run(
                [tool, "info"], capture_output=True, text=True, timeout=15, check=False
            )
            result.update(
                returncode=command.returncode,
                stdout=command.stdout,
                stderr=command.stderr,
            )
            result["status"] = (
                "probe_succeeded" if command.returncode == 0 else "failed"
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            result.update(status="failed", error=str(error))
    write_json(output, result)
    return result["status"] != "probe_succeeded"


def verify_results(bundle_root, returned, output, root=ROOT):
    """Check returned files; never launch device code or treat missing cases as skips."""
    import numpy as np
    from body_uniform_replay import check_outputs as check_body_uniform_outputs
    from body_validation import check_outputs as check_body_outputs
    from boundary_validation import check_outputs as check_boundary_outputs
    from broadphase_replay import check_outputs as check_broadphase_outputs
    from bucket_replay import check_outputs as check_bucket_outputs
    from canonical_replay import check_outputs as check_canonical_outputs
    from coefficient_replay import check_outputs as check_coefficient_outputs
    from collision_replay import check_outputs as check_collision_outputs
    from colored_replay import check_outputs as check_colored_outputs
    from coloring_replay import check_outputs as check_coloring_outputs
    from contact_plan_replay import check_outputs as check_contact_plan_outputs
    from contact_replay import check_outputs as check_contact_outputs
    from deferred_validation import check_outputs as check_deferred_outputs
    from epa_validation import check_outputs as check_epa_outputs
    from feature_validation import check_outputs as check_feature_outputs
    from general_scene_validation import check_outputs as check_general_scene_outputs
    from geometry_validation import check_outputs as check_geometry_outputs
    from graph_replay import check_outputs as check_graph_outputs
    from hub_replay import check_outputs as check_hub_outputs
    from joint_scene_validation import check_outputs as check_joint_scene_outputs
    from joint_validation import check_outputs as check_joint_outputs
    from lbvh_program_validation import check_outputs as check_lbvh_program_outputs
    from lbvh_validation import check_outputs as check_lbvh_outputs
    from link_replay import check_outputs as check_link_outputs
    from narrow_replay import check_outputs as check_narrow_outputs
    from pfm_validation import check_outputs as check_pfm_outputs
    from pose_replay import check_outputs as check_pose_outputs
    from primitives import check_outputs as check_integer_outputs
    from radix_replay import check_outputs as check_radix_outputs
    from replay import check_outputs
    from scan_replay import check_outputs as check_scan_outputs
    from scene_replay import check_outputs as check_scene_outputs
    from scene_restart_validation import check_outputs as check_scene_restart_outputs
    from support_validation import check_outputs as check_support_outputs
    from tgs_replay import check_outputs as check_tgs_outputs
    from warmstart_replay import check_outputs as check_warmstart_outputs

    result = {
        "schema": 1,
        "status": "failed",
        "level": "external_output_comparison",
        "cases": [],
    }
    try:
        bundle = verify_bundle(bundle_root)
        if bundle["inputs"] != evidence_inputs(root):
            raise ValueError(
                "Use the exact source snapshot from this handoff to compare device outputs"
            )
        receipt = read_json(returned / "session.json")
        if receipt.get("schema") != 1 or receipt.get("bundle_sha256") != digest(
            bundle_root / "manifest.json"
        ):
            raise ValueError("Returned session does not identify this bundle")
        for field in (
            "device_model",
            "device_id",
            "cann_version",
            "driver_version",
            "compiler_version",
        ):
            if (
                not isinstance(receipt.get("environment", {}).get(field), str)
                or not receipt["environment"][field].strip()
            ):
                raise ValueError(f"Missing actual device environment field: {field}")
        result["environment"] = receipt["environment"]
        result["bundle_sha256"] = receipt["bundle_sha256"]
        result["qualification"] = (
            "Numerical output comparison only; execution provenance and runtime adapter require review before parity acceptance"
        )
        for item in bundle["cases"]:
            case = item["case"]
            outcome = {
                "feature": item["feature"],
                "name": case["name"],
                "status": "failed",
            }
            try:
                directory = bundle_root / item["directory"]
                actual_dir = returned / item["directory"]
                contract = read_json(safe_file(bundle_root, item["contract"]))
                layout = read_json(safe_file(bundle_root, item["layout"]))
                buffers = contract["inputs"]
                compare = {
                    None: check_outputs,
                    "boundary_fields": check_boundary_outputs,
                    "exact_u32": check_integer_outputs,
                    "body_fields": check_body_outputs,
                    "pose_fields": check_pose_outputs,
                    "contact_fields": check_contact_outputs,
                    "link_fields": check_link_outputs,
                    "warmstart_fields": check_warmstart_outputs,
                    "hub_fields": check_hub_outputs,
                    "scan_fields": check_scan_outputs,
                    "graph_fields": check_graph_outputs,
                    "bucket_fields": check_bucket_outputs,
                    "coloring_fields": check_coloring_outputs,
                    "colored_fields": check_colored_outputs,
                    "coefficient_fields": check_coefficient_outputs,
                    "tgs_fields": check_tgs_outputs,
                    "scene_fields": check_scene_outputs,
                    "joint_fields": check_joint_outputs,
                    "joint_scene_fields": check_joint_scene_outputs,
                    "general_scene_fields": check_general_scene_outputs,
                    "scene_restart_fields": check_scene_restart_outputs,
                    "lbvh_fields": check_lbvh_outputs,
                    "lbvh_program_fields": check_lbvh_program_outputs,
                    "geometry_fields": check_geometry_outputs,
                    "support_fields": check_support_outputs,
                    "epa_fields": check_epa_outputs,
                    "feature_fields": check_feature_outputs,
                    "pfm_fields": check_pfm_outputs,
                    "deferred_fields": check_deferred_outputs,
                    "broadphase_fields": check_broadphase_outputs,
                    "narrow_fields": check_narrow_outputs,
                    "contact_plan_fields": check_contact_plan_outputs,
                    "collision_fields": check_collision_outputs,
                    "radix_fields": check_radix_outputs,
                    "canonical_fields": check_canonical_outputs,
                    "body_uniform_fields": check_body_uniform_outputs,
                }[contract.get("comparison")]
                before = {
                    n: np.fromfile(safe_file(directory, f"{n}.bin"), dtype=np.uint32)
                    for n in buffers
                }
                actual = {
                    n: np.fromfile(
                        safe_file(returned, f"{item['directory']}/{n}.actual.bin"),
                        dtype=np.uint32,
                    )
                    for n in buffers
                }
                expected = {
                    n: np.fromfile(
                        safe_file(directory, f"{n}.expected.bin"), dtype=np.uint32
                    )
                    if n in contract["outputs"]
                    else before[n]
                    for n in buffers
                }
                if any(
                    (actual_dir / f"{n}.actual.bin").stat().st_size != before[n].nbytes
                    for n in buffers
                ):
                    raise ValueError(
                        "Returned byte sizes differ from compiled specialization"
                    )
                if any(actual[n].shape != before[n].shape for n in buffers):
                    raise ValueError(
                        "Returned buffer sizes differ from compiled specialization"
                    )
                outcome["max_absolute_errors"] = compare(
                    layout, case, before, actual, expected, contract
                )
                outcome["output_sha256"] = {
                    n: digest(actual_dir / f"{n}.actual.bin") for n in buffers
                }
                outcome["status"] = "passed"
            except (ValueError, KeyError, OSError, AssertionError) as error:
                outcome["error"] = str(error)
            result["cases"].append(outcome)
        result["status"] = (
            "passed"
            if all(c["status"] == "passed" for c in result["cases"])
            else "failed"
        )
    except (ValueError, KeyError, OSError) as error:
        result["error"] = str(error)
    write_json(output, result)
    return result["status"] != "passed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("build")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--results", type=Path, default=ROOT / PORT / "results-ascend-source"
    )
    p = sub.add_parser("verify")
    p.add_argument("bundle", type=Path)
    p = sub.add_parser("probe")
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("verify-results")
    p.add_argument("bundle", type=Path)
    p.add_argument("returned", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "build":
            print(build(ROOT, args.output.resolve(), args.results.resolve()))
        elif args.command == "verify":
            bundle = verify_bundle(args.bundle.resolve())
            print(
                f"Verified {len(bundle['cases'])} fixture/source sets; qualification: {bundle['qualification']}"
            )
        elif args.command == "probe":
            return int(probe(args.output))
        else:
            return int(
                verify_results(
                    args.bundle.resolve(), args.returned.resolve(), args.output
                )
            )
        return 0
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print(f"handoff: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
