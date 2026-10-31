"""Fast checks cannot silently become full qualification or leak timed-out jobs."""

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import smoke


class SmokeTests(unittest.TestCase):
    def test_fixed_representative_contracts_and_missing_operation(self):
        rows = smoke.selected_features()
        self.assertEqual([row["operation"] for row in rows], list(smoke.OPERATIONS))
        self.assertEqual(len(rows), 5)
        with self.assertRaisesRegex(ValueError, "missing"):
            smoke.selected_features(rows[:-1])
        self.assertNotIn("runtime_scene", smoke.OPERATIONS)

    def test_full_matrix_requires_explicit_opt_in(self):
        result = subprocess.run(
            [sys.executable, str(smoke.HERE / "matrix.py")],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Full matrix is opt-in", result.stderr)
        listing = subprocess.run(
            [sys.executable, str(smoke.HERE / "matrix.py"), "--list"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertGreater(
            len(json.loads(listing.stdout)["operations"]), len(smoke.OPERATIONS)
        )

    def test_failed_command_and_expired_deadline_are_not_success(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaisesRegex(RuntimeError, "exited 7"):
                smoke.run_command(
                    [sys.executable, "-c", "raise SystemExit(7)"],
                    root,
                    root / "failure.log",
                    time.monotonic() + 5,
                )
            with self.assertRaises(TimeoutError):
                smoke.run_command(
                    [sys.executable, "-c", "raise SystemExit(0)"],
                    root,
                    root / "expired.log",
                    time.monotonic() - 1,
                )
            self.assertFalse((root / "expired.log").exists())

    @unittest.skipUnless(os.name == "posix", "POSIX process groups")
    def test_timeout_kills_owned_descendants(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            # The descendant would leave a marker after its parent times out.
            child = "import time; from pathlib import Path; time.sleep(1); Path('leaked').touch()"
            parent = f"import subprocess,sys,time; subprocess.Popen([sys.executable, '-c', {child!r}]); time.sleep(30)"
            before = time.monotonic()
            with self.assertRaises(TimeoutError):
                smoke.run_command(
                    [sys.executable, "-c", parent],
                    root,
                    root / "timeout.log",
                    before + 0.4,
                )
            self.assertLess(time.monotonic() - before, 3)
            time.sleep(1)
            self.assertFalse((root / "leaked").exists())

    def test_incomplete_stale_or_source_only_report_rejected(self):
        row = smoke.selected_features()[0]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "report.json"
            value = {"feature": row["id"], "target": "c", "cases": [{}]}
            path.write_text(json.dumps(value))
            for state in ("stale", "incomplete", "source_lowered", "failed"):
                with (
                    patch.object(smoke, "smoke_evidence_state", return_value=state),
                    self.assertRaisesRegex(ValueError, state),
                ):
                    smoke.validate_report(path, row, "c", {})
            with self.assertRaisesRegex(ValueError, "Wrong feature or target"):
                smoke.validate_report(path, row, "metal", {})

    def test_failure_has_unique_receipt_and_does_not_continue(self):
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.object(smoke, "evidence_inputs", return_value={}),
            patch.object(
                smoke, "run_command", side_effect=TimeoutError("deadline")
            ) as run,
        ):
            self.assertEqual(smoke.run_smoke("c", Path(folder), 2), 1)
            self.assertEqual(run.call_count, 1)
            receipt = json.loads(next(Path(folder).rglob("smoke.json")).read_text())
            self.assertEqual(receipt["status"], "timeout")
            self.assertEqual(len(receipt["steps"]), 1)
            self.assertEqual(receipt["suite"], "representative_smoke")
            self.assertEqual(list(Path(folder).rglob("report.json")), [])

    def test_selection_rejects_typos_and_duplicates(self):
        from stage_replay import select_cases

        cases = [{"name": "a"}, {"name": "b"}]
        self.assertEqual(select_cases(cases, ["b"]), [cases[1]])
        self.assertEqual(select_cases(cases, None), cases)
        for selection in ([], ["typo"], ["a", "a"]):
            with self.assertRaises(ValueError):
                select_cases(cases, selection)

    def test_budget_cannot_be_increased_past_five_minutes(self):
        import argparse

        for value in ("0", "-1", "300", "3600"):
            with self.assertRaises(argparse.ArgumentTypeError):
                smoke.budget_seconds(value)
        self.assertEqual(smoke.budget_seconds("290"), 290)


if __name__ == "__main__":
    unittest.main()
