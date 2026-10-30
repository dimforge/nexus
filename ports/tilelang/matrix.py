"""Deterministic complete-registry sharding for local and CI validation."""

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE_OPERATIONS = {"solve", "effective_mass", "warmstart", "integrate"}


def features(root=HERE):
    rows = json.loads((root / "parity.json").read_text())["features"]
    seen = set()
    for row in rows:
        op = row["operation"]
        runner = row.get("runner", "replay.py")
        command = shlex.split(row["test_command"])
        if op in seen or not op or not op.replace("_", "").isalnum():
            raise ValueError("Duplicate or invalid registered operation")
        seen.add(op)
        if (
            Path(runner).name != runner
            or not runner.endswith(".py")
            or not (root / runner).is_file()
        ):
            raise ValueError(f"Invalid runner for {op}")
        if (
            len(command) < 2
            or Path(command[1]).name != runner
            or (runner == "replay.py" and op not in BASE_OPERATIONS)
        ):
            raise ValueError(f"Registry runner/test command mismatch for {op}")
    return rows


def partitions(rows, count, root=HERE):
    if type(count) is not int or not 1 <= count <= 256:
        raise ValueError("Shard count must be between 1 and 256")
    groups = [[] for _ in range(count)]
    loads = [0] * count

    def weight(row):
        op = row["operation"]
        contract = root / ("contract.json" if op == "solve" else f"contracts/{op}.json")
        cases = len(json.loads(contract.read_text())["required_cases"])
        # Scheduling estimates only. Every operation remains required exactly once.
        multiplier = (
            50
            if op.startswith("runtime_")
            else 20
            if "frame_retry" in op
            else 8
            if "scene" in op or "restart" in op
            else 1
        )
        return max(cases, 1) * multiplier

    weighted = [(weight(row), row) for row in rows]
    for cost, row in sorted(
        weighted, key=lambda item: (-item[0], item[1]["operation"])
    ):
        shard = min(range(count), key=lambda i: (loads[i], i))
        groups[shard].append(row)
        loads[shard] += cost
    return groups


def command(row, target, output, compile_only=False, root=HERE):
    if target in ("ascend", "cuda") and not compile_only:
        raise ValueError("This runner supports source lowering only for Ascend/CUDA")
    result = [
        sys.executable,
        str(root / row.get("runner", "replay.py")),
        "--target",
        target,
        "--operation",
        row["operation"],
        "--output",
        str(output),
    ]
    if compile_only:
        result.append("--compile-only")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target", choices=("c", "metal", "ascend", "cuda"), default="c"
    )
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    groups = partitions(features(), args.shard_count)
    if not 0 <= args.shard_index < args.shard_count:
        parser.error("Shard index must be in [0, shard count)")
    rows = groups[args.shard_index]
    commands = [
        command(row, args.target, args.output.resolve(), args.compile_only)
        for row in rows
    ]
    if args.list:
        print(
            json.dumps(
                {
                    "shard": args.shard_index,
                    "shard_count": args.shard_count,
                    "operations": [r["operation"] for r in rows],
                    "commands": commands,
                },
                indent=2,
            )
        )
        return
    for row, cmd in zip(rows, commands, strict=True):
        print(
            f"Shard {args.shard_index}: {row['operation']} / {args.target}", flush=True
        )
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
