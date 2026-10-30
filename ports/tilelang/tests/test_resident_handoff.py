"""Validate resident Ascend source bundles using recorded Metal returns.

This tests the offline return protocol, not Ascend execution.
"""

import json
import unittest

from recorded_handoff import RecordedHandoff, handoff


class ResidentHandoffTests(RecordedHandoff):
    root_environment = "NEXUS_TILELANG_FRAME_ROOT"
    results_environment = "NEXUS_TILELANG_FRAME_RESULTS"
    operations = ("analytic_frame_retry", "general_frame_retry", "joint_frame_retry")

    def test_all_real_variants_and_metal_outputs_round_trip(self):
        self.assertEqual(len(handoff.verify_bundle(self.bundle)["cases"]), 4)
        self.assertFalse(self.compare(), self.output.read_text())
        result = json.loads(self.output.read_text())
        self.assertEqual(result["level"], "external_output_comparison")
        self.assertTrue(all("stats_sha256" in c for c in result["cases"]))

    def test_missing_and_wrong_growth_statistics_fail(self):
        stats = self.returned / self.manifest["cases"][0]["directory"] / "stats.json"
        original = stats.read_bytes()
        stats.unlink()
        self.assertTrue(self.compare())
        rows = json.loads(original)
        rows[47]["step"] = 47
        stats.write_text(json.dumps(rows))
        self.assertTrue(self.compare())

    def test_missing_variant_artifact_and_hash_inventory_fail(self):
        name = next(
            n
            for n in self.manifest["files"]
            if "/variants/1/" in n and n.endswith(".ascend")
        )
        expected = self.manifest["files"].pop(name)
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "missing from handoff"):
            handoff.verify_bundle(self.bundle)
        self.manifest["files"][name] = expected
        self.save_manifest()
        (self.bundle / name).unlink()
        self.assertTrue(self.compare())

    def test_corrupted_body_state_is_rejected(self):
        path = (
            self.returned / self.manifest["cases"][0]["directory"] / "Poses.actual.bin"
        )
        path.write_bytes(bytes(path.stat().st_size))
        self.assertTrue(self.compare())


if __name__ == "__main__":
    unittest.main()
