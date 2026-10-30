"""Validate real resident Ascend source bundles using recorded Metal returns.

This tests the offline return protocol, not Ascend execution. Fixtures must have
completed frame_replay on both targets at the explicitly selected source snapshot.
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import handoff
from replay import HERE, ROOT


class ResidentHandoffTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bundle = self.root / "bundle"
        self.returned = self.root / "returned"
        self.bundle.mkdir()
        self.returned.mkdir()
        self.snapshot = Path(os.environ.get("NEXUS_TILELANG_FRAME_ROOT", ROOT))
        results = Path(os.environ.get("NEXUS_TILELANG_FRAME_RESULTS", HERE / "results"))
        self.output = self.root / "comparison.json"
        self.manifest = {
            "schema": 1,
            "qualification": "Metal-return protocol regression; not Ascend execution",
            "inputs": handoff.evidence_inputs(self.snapshot),
            "files": {},
            "cpu_oracles": {},
            "cases": [],
        }

        def copy(source, relative):
            target = self.bundle / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            self.manifest["files"][relative] = handoff.digest(target)

        for op in ("analytic_frame_retry", "general_frame_retry", "joint_frame_retry"):
            source = results / op / "ascend"
            report = json.loads((source / "report.json").read_text())
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["inputs"], self.manifest["inputs"])
            metal_report = json.loads((results / op / "metal/report.json").read_text())
            self.assertEqual(metal_report["status"], "passed")
            self.assertEqual(metal_report["inputs"], report["inputs"])
            contract_path = self.snapshot / f"ports/tilelang/contracts/{op}.json"
            contract = json.loads(contract_path.read_text())
            copy(contract_path, f"contracts/{op}.json")
            copy(source / "fixtures/layout.json", f"layouts/{op}.json")
            copy(source / "report.json", f"reports/{op}.json")
            copy(source / "cpu-oracle-noalias.ll", f"reports/{op}.ll")
            self.manifest["cpu_oracles"][contract["feature"]] = {
                "report": f"reports/{op}.json",
                "llvm_ir": f"reports/{op}.ll",
            }
            for row in report["cases"]:
                case = row["fixture"]
                folder = source / "fixtures" / op / case["name"]
                relative = f"cases/{op}/{case['name']}"
                required = handoff.kernel_artifacts(folder, contract)
                required |= {f"{n}.bin" for n in contract["inputs"]}
                required |= {f"{n}.expected.bin" for n in contract["outputs"]}
                for name in required:
                    self.assertEqual(handoff.digest(folder / name), row["files"][name])
                    copy(folder / name, f"{relative}/{name}")
                self.manifest["cases"].append(
                    {
                        "feature": contract["feature"],
                        "case": case,
                        "directory": relative,
                        "layout": f"layouts/{op}.json",
                        "contract": f"contracts/{op}.json",
                    }
                )
                actual = self.returned / relative
                actual.mkdir(parents=True)
                recorded = results / op / "metal/fixtures" / op / case["name"]
                for name in [
                    *[f"{n}.actual.bin" for n in contract["inputs"]],
                    "stats.json",
                ]:
                    shutil.copyfile(recorded / name, actual / name)
        self.save_manifest()

    def save_manifest(self):
        handoff.write_json(self.bundle / "manifest.json", self.manifest)
        (self.bundle / "manifest.sha256").write_text(
            handoff.digest(self.bundle / "manifest.json") + "\n"
        )
        handoff.write_json(
            self.returned / "session.json",
            {
                "schema": 1,
                "bundle_sha256": handoff.digest(self.bundle / "manifest.json"),
                "environment": {
                    "device_model": "Apple M4 Max: recorded Metal outputs for protocol test",
                    "device_id": "local Metal regression",
                    "cann_version": "not used; no Ascend execution",
                    "driver_version": "recorded Metal regression",
                    "compiler_version": "recorded Metal regression",
                },
            },
        )

    def compare(self):
        return handoff.verify_results(
            self.bundle, self.returned, self.output, root=self.snapshot
        )

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
