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
    if report.get("suite") != "portable":
        return "diagnostic_only"
    required_cases = read_json(root / PORT / "contract.json")["required_cases"]
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
    if report.get("level") == "source_lowered":
        return "source_lowered"
    if report.get("level") != "executed" or any(
        "max_absolute_errors" not in c for c in report["cases"]
    ):
        return "invalid"
    return "executed"


def status(root, extra_evidence=()):
    config = manifest(root)
    current = evidence_inputs(root)
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
    backends = {target: {"state": "unvalidated"} for target in config["targets"]}
    for target, item in baseline.get("evidence", {}).items():
        path = root / item["path"]
        if not path.is_file() or digest(path) != item["sha256"]:
            backends[target] = {
                "state": "missing_or_modified_evidence",
                "path": str(path),
            }
        else:
            report = read_json(path)
            backends[target] = {
                "state": evidence_state(root, report, current)
                if report.get("target") == target
                else "invalid_target",
                "path": str(path),
                "environment": report.get("environment"),
            }
    for path in extra_evidence:
        report = read_json(path)
        target = report["target"]
        if target not in backends:
            raise ValueError(f"Unknown evidence target: {target}")
        backends[target] = {
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
        "backends": backends,
        "known_gaps": read_json(root / PORT / "contract.json").get("known_gaps", []),
    }


def reconcile(root, change_path, evidence_paths):
    root = root.resolve()
    config = manifest(root)
    current = evidence_inputs(root)
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
        accepted[report["target"]] = (report, state)
    for target in config["required_reconciliation_targets"]:
        if target not in accepted or accepted[target][1] != "executed":
            raise ValueError(
                f"Reconciliation requires current {target} execution evidence"
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
    if evidence_inputs(root) != current:
        raise ValueError("Sources changed during reconciliation")
    evidence = {}
    for target, (report, _) in accepted.items():
        path = root / PORT / "evidence" / f"{target}.json"
        write_json(path, report)
        evidence[target] = {
            "path": path.relative_to(root).as_posix(),
            "sha256": digest(path),
        }
    # Preserve old backend evidence so it remains visible as stale, not silently lost.
    baseline_path = root / PORT / "baseline.json"
    if baseline_path.exists():
        old = read_json(baseline_path)
        evidence = {**old.get("evidence", {}), **evidence}
    baseline = {
        "schema": 1,
        "inputs": current,
        "change": {
            "path": change_path.relative_to(root).as_posix(),
            "sha256": digest(change_path),
        },
        "evidence": evidence,
    }
    write_json(baseline_path, baseline)
    return status(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["report", "check", "reconcile"])
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--evidence", type=Path, action="append", default=[])
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
            for gap in result["known_gaps"]:
                print(f"Known gap: {gap['id']}: {gap['description']}")
            for target, item in result["backends"].items():
                print(f"{target}: {item['state']}")
        if args.command == "check":
            targets = args.require or manifest(ROOT)["required_reconciliation_targets"]
            if (
                (
                    args.release
                    and any(g.get("blocks_release") for g in result["known_gaps"])
                )
                or result["source_reconciliation"] != "current"
                or any(result["backends"][t]["state"] != "executed" for t in targets)
            ):
                return 1
        return 0
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
        print(f"parity: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
