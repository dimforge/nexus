#!/usr/bin/env python3
"""Machine-readable impact report and evidence gate for the TileLang pilot.

No accelerator or third-party Python modules are needed for this tool. Hashes
only detect changes. Reconciliation also needs an explanatory change record and
successful execution evidence; source lowering cannot stand in for execution.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tomllib

from inventory import audit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PORT = Path("ports/tilelang")
KINDS = {
    "port",
    "feature",
    "bugfix",
    "refactor",
    "optimization",
    "numerical",
    "abi",
    "tooling",
    "test",
}


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest(root):
    result = read_json(root / PORT / "parity.json")
    if result["schema"] != 1:
        raise ValueError("Unsupported parity schema")
    ids = [feature["id"] for feature in result["features"]]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Feature IDs must be nonempty and unique")
    return result


def snapshot(root, patterns):
    files = {}
    for pattern in patterns:
        for path in root.glob(pattern):
            if path.is_file():
                if path.is_symlink() or not path.resolve().is_relative_to(
                    root.resolve()
                ):
                    raise ValueError(f"Source outside repository: {path}")
                files[path.relative_to(root).as_posix()] = digest(path)
    return dict(sorted(files.items()))


def evidence_inputs(root):
    config = manifest(root)
    return {
        "reference": snapshot(root, config["reference_scope"]),
        "implementation": snapshot(root, config["implementation_scope"]),
    }


def changes(previous, current):
    return {
        "added": sorted(current.keys() - previous.keys()),
        "removed": sorted(previous.keys() - current.keys()),
        "modified": sorted(
            k for k in current.keys() & previous.keys() if current[k] != previous[k]
        ),
    }


def evidence_state(root, report, inputs):
    return _evidence_state(root, report, inputs)


def smoke_evidence_state(root, report, inputs, selected_cases):
    """Validate an explicit subset without making it full qualification evidence."""
    if not selected_cases:
        return "incomplete"
    return _evidence_state(root, report, inputs, selected_cases)


def _evidence_state(root, report, inputs, selected_cases=None):
    if report.get("schema") != 1 or report.get("feature") not in {
        f["id"] for f in manifest(root)["features"]
    }:
        return "invalid"
    if (
        report.get("status") != "passed"
        or not report.get("cases")
        or any(c.get("status") != "passed" for c in report["cases"])
    ):
        return "failed"
    suite = "portable" if selected_cases is None else "representative_smoke"
    if report.get("suite") != suite:
        return "diagnostic_only"
    feature = next(
        f for f in manifest(root)["features"] if f["id"] == report["feature"]
    )
    required_cases = feature_contract(root, feature)["required_cases"]
    if selected_cases is not None:
        if (
            len(set(selected_cases)) != len(selected_cases)
            or not set(selected_cases).issubset(required_cases)
            or report.get("selected_cases") != list(selected_cases)
        ):
            return "incomplete"
        required_cases = selected_cases
    names = [c.get("name") for c in report["cases"]]
    if len(names) != len(set(names)) or set(names) != set(required_cases):
        return "incomplete"
    if report.get("inputs") != inputs:
        return "stale"
    pinned = {}
    for line in (root / PORT / "requirements.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            name, version = line.split("==")
            pinned[name] = version
    environment_names = {
        "tilelang": "tilelang",
        "apache-tvm-ffi": "tvm_ffi",
        "numpy": "numpy",
        "z3-solver": "z3",
        "torch": "torch",
    }
    if any(
        report.get("environment", {}).get(environment_names[name], "").split("+")[0]
        != version
        for name, version in pinned.items()
    ):
        return "toolchain_mismatch"
    toolchain = root / PORT / "reference/rust-toolchain.toml"
    if toolchain.exists():
        rust_version = tomllib.loads(toolchain.read_text())["toolchain"]["channel"]
        if (
            not report.get("environment", {})
            .get("rustc", "")
            .startswith(f"rustc {rust_version} ")
        ):
            return "toolchain_mismatch"
    from cpu_oracle import FLAGS, PROFILE

    env = report.get("environment", {})
    oracle = env.get("cpu_oracle", {})
    probe = root / PORT / "reference/noalias_probe.rs"
    if (
        oracle.get("profile") != PROFILE
        or oracle.get("rustflags") != list(FLAGS)
        or env.get("rustflags") != list(FLAGS)
        or oracle.get("rustc") != env.get("rustc")
        or oracle.get("rustc_bootstrap") != "1"
        or oracle.get("mutable_reference_noalias") is not False
        or oracle.get("shared_reference_noalias") is not True
        or not probe.is_file()
        or oracle.get("probe_source_sha256") != digest(probe)
        or len(oracle.get("llvm_ir_sha256", "")) != 64
        or any(c not in "0123456789abcdef" for c in oracle.get("llvm_ir_sha256", ""))
    ):
        return "oracle_configuration_mismatch"
    if report.get("level") == "source_lowered":
        return "source_lowered"
    if report.get("level") != "executed" or any(
        "max_absolute_errors" not in c for c in report["cases"]
    ):
        return "invalid"
    return "executed"


def feature_contract(root, feature):
    return read_json(root / feature.get("contract", str(PORT / "contract.json")))


def saved_evidence(baseline, config):
    # Read the pilot's original schema without treating its evidence as coverage
    # for subsequently added operations.
    result = {
        key: dict(value) for key, value in baseline.get("feature_evidence", {}).items()
    }
    first = config["features"][0]["id"]
    result.setdefault(first, baseline.get("evidence", {}))
    return result


def status(root, extra_evidence=()):
    config = manifest(root)
    current = evidence_inputs(root)
    inventory = audit(root, config)
    baseline_path = root / PORT / "baseline.json"
    baseline = read_json(baseline_path) if baseline_path.exists() else {}
    previous = baseline.get("inputs", {"reference": {}, "implementation": {}})
    diffs = {kind: changes(previous[kind], current[kind]) for kind in current}
    source_current = bool(baseline) and previous == current
    record = baseline.get("change", {})
    record_path = root / record.get("path", "__missing__")
    record_current = record_path.is_file() and digest(record_path) == record.get(
        "sha256"
    )
    matrix = {
        f["id"]: {target: {"state": "unvalidated"} for target in config["targets"]}
        for f in config["features"]
    }
    for feature, targets in saved_evidence(baseline, config).items():
        if feature not in matrix:
            continue
        for target, item in targets.items():
            if target not in matrix[feature]:
                continue
            path = root / item["path"]
            if not path.is_file() or digest(path) != item["sha256"]:
                matrix[feature][target] = {
                    "state": "missing_or_modified_evidence",
                    "path": str(path),
                }
            else:
                report = read_json(path)
                matrix[feature][target] = {
                    "state": evidence_state(root, report, current)
                    if report.get("target") == target
                    and report.get("feature") == feature
                    else "invalid_target",
                    "path": str(path),
                    "environment": report.get("environment"),
                }
    seen = set()
    for path in extra_evidence:
        report = read_json(path)
        feature, target = report["feature"], report["target"]
        if (feature, target) in seen:
            raise ValueError("Duplicate feature/target evidence")
        seen.add((feature, target))
        if feature not in matrix or target not in matrix[feature]:
            raise ValueError(f"Unknown evidence feature/target: {feature}/{target}")
        matrix[feature][target] = {
            "state": evidence_state(root, report, current),
            "path": str(path),
            "environment": report.get("environment"),
        }
    return {
        "schema": 1,
        "source_reconciliation": "current"
        if source_current and record_current
        else "required",
        "change_record_current": record_current,
        "changes": diffs,
        "unmapped_reference_files": diffs["reference"]["added"],
        "affected_features": []
        if source_current
        else [f["id"] for f in config["features"]],
        "features": config["features"],
        "inventory": inventory,
        "feature_backends": matrix,
        # Compatibility alias for existing single-operation consumers only.
        "backends": matrix[config["features"][0]["id"]],
        "known_gaps": [
            gap
            for f in config["features"]
            for gap in feature_contract(root, f).get("known_gaps", [])
        ]
        + [
            {
                "id": c["id"],
                "description": c["remaining"],
                "blocks_release": c.get("required_for_completion", True),
                "scope_reason": c.get("scope_reason"),
            }
            for c in inventory["capabilities"]
            if c["coverage"] != "complete"
        ],
    }


def reconcile(root, change_path, evidence_paths):
    root = root.resolve()
    config = manifest(root)
    current = evidence_inputs(root)
    inventory = audit(root, config)
    if inventory["errors"]:
        raise ValueError("Inventory needs review: " + "; ".join(inventory["errors"]))
    change_path = change_path.resolve()
    if not change_path.is_relative_to((root / PORT / "changes").resolve()):
        raise ValueError("Change records must live in ports/tilelang/changes")
    change = read_json(change_path)
    required = (
        "id",
        "summary",
        "features",
        "kinds",
        "disposition",
        "regressions",
        "differences",
    )
    if any(not change.get(key) for key in required):
        raise ValueError(f"Change record needs nonempty fields: {required}")
    if not set(change["kinds"]).issubset(KINDS):
        raise ValueError("Unknown change classification")
    if set(change["features"]) != {f["id"] for f in config["features"]}:
        raise ValueError("Reconcile every affected feature in this conservative pilot")
    if change["disposition"] not in ("implementation_updated", "behavior_unchanged"):
        raise ValueError("Invalid reconciliation disposition")
    baseline_path = root / PORT / "baseline.json"
    if baseline_path.exists():
        previous = read_json(baseline_path)
        previous_record = root / previous["change"]["path"]
        if previous["inputs"] != current and (
            change_path == previous_record.resolve()
            or (
                previous_record.is_file()
                and read_json(previous_record).get("id") == change["id"]
            )
        ):
            raise ValueError("Use a new change record and ID for changed inputs")
    accepted = {}
    for path in evidence_paths:
        report = read_json(path)
        if report.get("target") not in config["targets"]:
            raise ValueError("Unknown evidence target")
        state = evidence_state(root, report, current)
        if state not in ("executed", "source_lowered"):
            raise ValueError(f"Evidence {path} is {state}; rerun replay")
        key = (report["feature"], report["target"])
        if key in accepted:
            raise ValueError("Duplicate feature/target evidence")
        accepted[key] = (report, state)
    for feature in config["features"]:
        for target in config["required_reconciliation_targets"]:
            key = (feature["id"], target)
            if key not in accepted or accepted[key][1] != "executed":
                raise ValueError(
                    f"Reconciliation requires current {target} execution evidence for {feature['id']}"
                )
    # No convenience operation can simply advance the source hash without checks.
    subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            str(root / PORT / "tests"),
            "-p",
            "test_parity.py",
        ],
        check=True,
    )
    if "inventory" in config:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "unittest",
                "discover",
                "-s",
                str(root / PORT / "tests"),
                "-p",
                "test_inventory.py",
            ],
            check=True,
        )
    if evidence_inputs(root) != current:
        raise ValueError("Sources changed during reconciliation")
    baseline_path = root / PORT / "baseline.json"
    previous = read_json(baseline_path) if baseline_path.exists() else {}
    evidence = saved_evidence(previous, config)
    first = config["features"][0]["id"]
    for (feature, target), (report, _) in accepted.items():
        name = f"{target}.json" if feature == first else f"{feature}.{target}.json"
        path = root / PORT / "evidence" / name
        write_json(path, report)
        evidence.setdefault(feature, {})[target] = {
            "path": path.relative_to(root).as_posix(),
            "sha256": digest(path),
        }
    baseline = {
        "schema": 1,
        "inputs": current,
        "change": {
            "path": change_path.relative_to(root).as_posix(),
            "sha256": digest(change_path),
        },
        "feature_evidence": evidence,
        "evidence": evidence.get(first, {}),
    }
    write_json(baseline_path, baseline)
    return status(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["report", "check", "reconcile"])
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--evidence", type=Path, action="append", default=[])
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        action="append",
        default=[],
        help="Read every report.json under a replay result root; failed reports are rejected",
    )
    parser.add_argument(
        "--require", action="append", choices=manifest(ROOT)["targets"], default=[]
    )
    parser.add_argument(
        "--release",
        action="store_true",
        help="Also reject known blocking gaps in the contract",
    )
    parser.add_argument("--change", type=Path)
    args = parser.parse_args()
    try:
        for directory in args.evidence_dir:
            reports = sorted(directory.rglob("report.json"))
            if not reports:
                raise ValueError(f"No reports in {directory}")
            args.evidence.extend(reports)
        if args.command == "reconcile":
            if args.change is None:
                parser.error("reconcile requires --change")
            result = reconcile(ROOT, args.change, args.evidence)
        else:
            result = status(ROOT, args.evidence)
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print(f"Source reconciliation: {result['source_reconciliation']}")
            for kind, diff in result["changes"].items():
                for action, paths in diff.items():
                    for path in paths:
                        print(f"  {kind} {action}: {path}")
            for error in result["inventory"]["errors"]:
                print(f"Inventory error: {error}")
            for gap in result["known_gaps"]:
                print(f"Known gap: {gap['id']}: {gap['description']}")
            for feature, targets in result["feature_backends"].items():
                for target, item in targets.items():
                    print(f"{feature} / {target}: {item['state']}")
        if args.command == "check":
            targets = args.require or manifest(ROOT)["required_reconciliation_targets"]
            if (
                (
                    args.release
                    and any(g.get("blocks_release") for g in result["known_gaps"])
                )
                or bool(result["inventory"]["errors"])
                or result["source_reconciliation"] != "current"
                or any(
                    backends[t]["state"] != "executed"
                    for backends in result["feature_backends"].values()
                    for t in targets
                )
            ):
                return 1
        return 0
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
        print(f"parity: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
