"""Prevent silently omitted operations or incorrect generic runner routing."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matrix


class MatrixTests(unittest.TestCase):
    def test_every_registered_operation_exactly_once(self):
        rows = matrix.features()
        expected = sorted(r["operation"] for r in rows)
        for count in (1, 3, 16, 32):
            groups = matrix.partitions(rows, count)
            self.assertEqual(
                sorted(r["operation"] for group in groups for r in group), expected
            )
            self.assertEqual(groups, matrix.partitions(list(reversed(rows)), count))

    def test_every_routed_command_uses_its_registered_runner(self):
        for row in matrix.features():
            command = matrix.command(row, "c", Path("results"))
            self.assertEqual(Path(command[1]).name, row.get("runner", "replay.py"))
            self.assertEqual(
                command[command.index("--operation") + 1], row["operation"]
            )

    def test_missing_specialized_runner_and_duplicate_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "replay.py").touch()
            (root / "twist_replay.py").touch()
            row = next(
                copy.deepcopy(f)
                for f in matrix.features()
                if f["operation"] == "twist_solve"
            )
            for rows in (
                [dict(row, runner="replay.py")],
                [row, row],
                [dict(row, runner="../replay.py")],
            ):
                (root / "parity.json").write_text(json.dumps({"features": rows}))
                with self.assertRaises(ValueError):
                    matrix.features(root)

    def test_device_execution_is_distinct_from_lowering_evidence(self):
        row = matrix.features()[0]
        self.assertNotIn(
            "--compile-only", matrix.command(row, "ascend", Path("results"))
        )
        self.assertIn(
            "--compile-only", matrix.command(row, "ascend", Path("results"), True)
        )

    def test_invalid_shards_fail(self):
        for value in (0, -1, 257, 1.5):
            with self.assertRaises(ValueError):
                matrix.partitions([], value)


if __name__ == "__main__":
    unittest.main()
