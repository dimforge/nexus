"""Return protocol for native scene imports; recorded Metal is not Ascend evidence."""

import json
import os
import unittest

from recorded_handoff import RecordedHandoff, handoff
from replay import HERE


@unittest.skipUnless(
    os.environ.get("NEXUS_TILELANG_RUNTIME_RESULTS")
    or all(
        (HERE / "results" / op / target / "report.json").is_file()
        for op in ("runtime_scene",)
        for target in ("metal", "ascend")
    ),
    "Requires recorded Metal execution and Ascend source reports; an explicit results path must be valid",
)
class RuntimeHandoffTests(RecordedHandoff):
    root_environment = "NEXUS_TILELANG_RUNTIME_ROOT"
    results_environment = "NEXUS_TILELANG_RUNTIME_RESULTS"
    operations = ("runtime_scene",)

    def test_all_native_scenes_round_trip(self):
        self.assertEqual(len(handoff.verify_bundle(self.bundle)["cases"]), 6)
        self.assertFalse(self.compare(), self.output.read_text())
        result = json.loads(self.output.read_text())
        self.assertEqual(result["level"], "external_output_comparison")
        self.assertTrue(all(c["status"] == "passed" for c in result["cases"]))

    def test_corrupted_state_and_readonly_parent_fail(self):
        case = self.manifest["cases"][0]
        for name in ("Poses", "Parents"):
            path = self.returned / case["directory"] / f"{name}.actual.bin"
            original = path.read_bytes()
            path.write_bytes(bytes(path.stat().st_size))
            with self.subTest(buffer=name):
                self.assertTrue(self.compare())
            path.write_bytes(original)

    def test_missing_kernel_and_upload_hash_are_rejected(self):
        name = next(n for n in self.manifest["files"] if n.endswith(".ascend"))
        self.manifest["files"].pop(name)
        self.save_manifest()
        with self.assertRaises(ValueError):
            handoff.verify_bundle(self.bundle)


if __name__ == "__main__":
    unittest.main()
