#!/usr/bin/env python3
"""Audit Rust/TileLang module counterparts without importing a GPU compiler."""

import argparse
import ast
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def counterpart(reference):
    relative = Path(reference).relative_to("src_rbd_shaders").with_suffix(".py")
    if relative.name in ("mod.py", "lib.py"):
        relative = relative.with_name("__init__.py")
    return (Path("ports/tilelang/shaders") / relative).as_posix()


def audit(root, config):
    if "inventory" not in config:
        return {"errors": [], "modules": [], "capabilities": []}
    inventory = json.loads((root / config["inventory"]).read_text())
    errors = []
    if inventory.get("schema") != 1:
        errors.append("Unsupported inventory schema")
    modules = inventory["modules"]
    listed = [m["reference"] for m in modules]
    actual = {
        p.relative_to(root).as_posix() for p in (root / "src_rbd_shaders").rglob("*.rs")
    }
    if len(listed) != len(set(listed)):
        errors.append("Duplicate Rust module dispositions")
    for name in sorted(actual - set(listed)):
        errors.append(f"New Rust module needs an explicit disposition: {name}")
    for name in sorted(set(listed) - actual):
        errors.append(f"Removed Rust module remains in inventory: {name}")
    operations = {f["id"] for f in config["features"]}
    mapped = set()
    implemented_paths = set()
    for module in modules:
        name = module["reference"]
        expected = counterpart(name)
        if module["implementation"] != expected:
            errors.append(f"Shader must mirror Rust module: {name} -> {expected}")
        if module["coverage"] not in (
            "partial",
            "complete",
            "unported",
            "reference_test",
        ):
            errors.append(f"Invalid coverage: {name}")
        if not module.get("remaining"):
            errors.append(f"Missing coverage explanation: {name}")
        ids = set(module["operations"])
        if not ids.issubset(operations):
            errors.append(f"Unknown operation in {name}")
        mapped.update(ids)
        if module["coverage"] in ("partial", "complete"):
            path = root / expected
            implemented_paths.add(expected)
            if not ids or not module["symbols"] or not path.is_file():
                errors.append(
                    f"Implemented module needs operations, symbols and a file: {name}"
                )
                continue
            tree = ast.parse(path.read_text(), filename=expected)
            symbols = {
                node.name
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef))
            }
            if not set(module["symbols"]).issubset(symbols):
                errors.append(f"Missing mapped Python symbols: {expected}")
        elif ids or module["symbols"]:
            errors.append(f"Unported module cannot certify operations: {name}")
    for path in (root / "ports/tilelang/shaders").rglob("*.py"):
        # Empty/import-only Python package initializers need no physics coverage.
        if path.name == "__init__.py":
            if all(
                isinstance(node, (ast.Import, ast.ImportFrom, ast.Expr))
                for node in ast.parse(path.read_text()).body
            ):
                continue
        if path.relative_to(root).as_posix() not in implemented_paths:
            errors.append(f"Unmapped TileLang shader: {path.relative_to(root)}")
    if operations - mapped:
        errors.append(
            f"Operations without shader mappings: {sorted(operations - mapped)}"
        )
    capabilities = inventory["capabilities"]
    ids = [c["id"] for c in capabilities]
    if len(ids) != len(set(ids)):
        errors.append("Duplicate capability IDs")
    covered = set()
    for capability in capabilities:
        required = capability.get("required_for_completion", True)
        if type(required) is not bool or (
            not required and not capability.get("scope_reason")
        ):
            errors.append(f"Invalid completion scope: {capability['id']}")
        if capability["coverage"] not in (
            "partial",
            "complete",
            "unported",
        ) or not capability.get("remaining"):
            errors.append(f"Invalid capability disposition: {capability['id']}")
        if not set(capability["operations"]).issubset(operations):
            errors.append(f"Unknown capability operation: {capability['id']}")
        covered.update(capability["operations"])
    if operations - covered:
        errors.append(
            f"Operations without capability mapping: {sorted(operations - covered)}"
        )
    return {"errors": errors, "modules": modules, "capabilities": capabilities}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = audit(ROOT, json.loads((HERE / "parity.json").read_text()))
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        for module in result["modules"]:
            print(
                f"{module['coverage']:14} {module['reference']} -> {module['implementation']}"
            )
        for error in result["errors"]:
            print(f"ERROR: {error}")
    return int(bool(result["errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
