"""Exercise offline packaging and returned-output checks using real Rust fixtures."""

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import handoff
from replay import HERE


class HandoffTests(unittest.TestCase):
    operation = "solve"
    fixture_index = -1
    mutable_buffer = "Velocities"

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bundle = self.root / "bundle"
        self.bundle.mkdir()
        self.returned = self.root / "returned"
        self.returned.mkdir()
        self.output = self.root / "comparison.json"
        fixtures = (
            Path(os.environ.get("NEXUS_TILELANG_RESULTS", HERE / "results"))
            / ("" if self.operation == "solve" else self.operation)
            / "c/fixtures"
        )
        run_root = fixtures.parent
        layout = fixtures / "layout.json"
        if self.operation != "solve":
            fixtures = fixtures / self.operation
        case = json.loads((fixtures / "suite.json").read_text())["cases"][
            self.fixture_index
        ]
        directory = f"cases/{self.operation}/{case['name']}"
        dest = self.bundle / directory
        dest.mkdir(parents=True)
        self.actual = self.returned / directory
        self.actual.mkdir(parents=True)
        contract_path = HERE / (
            "contract.json"
            if self.operation == "solve"
            else f"contracts/{self.operation}.json"
        )
        contract = json.loads(contract_path.read_text())
        files = {}
        for name in contract["inputs"]:
            shutil.copyfile(
                fixtures / case["name"] / f"{name}.bin", dest / f"{name}.bin"
            )
            expected = (
                fixtures / case["name"] / f"{name}.expected.bin"
                if name in contract["outputs"]
                else dest / f"{name}.bin"
            )
            if name in contract["outputs"]:
                shutil.copyfile(expected, dest / f"{name}.expected.bin")
            shutil.copyfile(expected, self.actual / f"{name}.actual.bin")
        (dest / "bindings.json").write_text("{}")
        (dest / "kernel.ascend").write_text("// test fixture; not executed")
        if contract.get("execution") == "ordered_kernels":
            program = json.loads(
                (fixtures / case["name"] / "bindings.json").read_text()
            )
            for stage in program["stages"]:
                stage["source"] = stage["source"].removesuffix(".c") + ".ascend"
                (dest / stage["source"]).write_text(
                    "// TEST ONLY: source receipt fixture"
                )
            (dest / "bindings.json").write_text(json.dumps(program))
        shutil.copyfile(layout, self.bundle / "layout.json")
        shutil.copyfile(contract_path, self.bundle / "contract.json")
        shutil.copyfile(run_root / "report.json", self.bundle / "oracle-report.json")
        shutil.copyfile(
            run_root / "cpu-oracle-noalias.ll", self.bundle / "oracle-noalias.ll"
        )
        for file in self.bundle.rglob("*"):
            if file.is_file():
                files[file.relative_to(self.bundle).as_posix()] = handoff.digest(file)
        self.manifest = {
            "schema": 1,
            "qualification": "test_fixture",
            "inputs": handoff.evidence_inputs(handoff.ROOT),
            "files": files,
            "cpu_oracles": {
                contract["feature"]: {
                    "report": "oracle-report.json",
                    "llvm_ir": "oracle-noalias.ll",
                }
            },
            "cases": [
                {
                    "feature": contract["feature"],
                    "case": case,
                    "directory": directory,
                    "layout": "layout.json",
                    "contract": "contract.json",
                }
            ],
        }
        self.save_manifest()
        self.session = {
            "schema": 1,
            "bundle_sha256": handoff.digest(self.bundle / "manifest.json"),
            "environment": {
                field: "TEST ONLY"
                for field in (
                    "device_model",
                    "device_id",
                    "cann_version",
                    "driver_version",
                    "compiler_version",
                )
            },
        }
        handoff.write_json(self.returned / "session.json", self.session)

    def save_manifest(self):
        handoff.write_json(self.bundle / "manifest.json", self.manifest)
        (self.bundle / "manifest.sha256").write_text(
            handoff.digest(self.bundle / "manifest.json") + "\n"
        )

    def compare(self):
        return handoff.verify_results(self.bundle, self.returned, self.output)

    def test_bundle_and_return_round_trip(self):
        self.assertEqual(handoff.verify_bundle(self.bundle)["schema"], 1)
        self.assertFalse(self.compare())
        self.assertEqual(
            json.loads(self.output.read_text())["level"], "external_output_comparison"
        )

    def test_modified_manifest_or_artifact_is_rejected(self):
        (self.bundle / "manifest.json").write_text("{}")
        self.assertTrue(self.compare())
        self.save_manifest()
        (self.bundle / "layout.json").write_text("{}")
        self.assertTrue(self.compare())

    def test_unhashed_case_file_is_rejected(self):
        del self.manifest["files"]["layout.json"]
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "missing from handoff"):
            handoff.verify_bundle(self.bundle)

    def test_missing_or_modified_cpu_oracle_is_rejected(self):
        oracles = self.manifest.pop("cpu_oracles")
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "oracle provenance"):
            handoff.verify_bundle(self.bundle)
        self.manifest["cpu_oracles"] = oracles
        probe = self.bundle / "oracle-noalias.ll"
        probe.write_text(probe.read_text() + "\n; modified after fixture export\n")
        self.manifest["files"]["oracle-noalias.ll"] = handoff.digest(probe)
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "Modified CPU oracle LLVM"):
            handoff.verify_bundle(self.bundle)

    def test_missing_corrupt_and_padded_returned_buffers_fail(self):
        file = self.actual / f"{self.mutable_buffer}.actual.bin"
        original = file.read_bytes()
        file.unlink()
        self.assertTrue(self.compare())
        file.write_bytes(original + b"x")
        self.assertTrue(self.compare())
        file.write_bytes(bytes(len(original)))
        self.assertTrue(self.compare())

    def test_path_escape_and_symlink_are_rejected(self):
        outside = self.root / "outside.bin"
        outside.write_bytes(b"test")
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            handoff.safe_file(self.bundle, "../outside.bin")
        (self.bundle / "linked.bin").symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            handoff.safe_file(self.bundle, "linked.bin")

    def test_wrong_session_and_missing_environment_are_rejected(self):
        self.session["bundle_sha256"] = "0" * 64
        handoff.write_json(self.returned / "session.json", self.session)
        self.assertTrue(self.compare())
        self.session["bundle_sha256"] = handoff.digest(self.bundle / "manifest.json")
        del self.session["environment"]["cann_version"]
        handoff.write_json(self.returned / "session.json", self.session)
        self.assertTrue(self.compare())

    def test_no_npu_is_unavailable_not_passed(self):
        with patch.object(handoff.shutil, "which", return_value=None):
            self.assertTrue(handoff.probe(self.output))
        self.assertEqual(json.loads(self.output.read_text())["status"], "unavailable")


class BodyHandoffTests(HandoffTests):
    operation = "free_step"
    fixture_index = 4


class PoseHandoffTests(HandoffTests):
    operation = "sync_colliders"
    fixture_index = 4
    mutable_buffer = "WorldPoses"


class ContactHandoffTests(HandoffTests):
    operation = "prepare"
    fixture_index = 7
    mutable_buffer = "Tiles"


class LinkHandoffTests(HandoffTests):
    operation = "contact_persistence"
    fixture_index = 8
    mutable_buffer = "States"


class WarmstartHandoffTests(HandoffTests):
    operation = "gather_warmstart"
    fixture_index = 8


if __name__ == "__main__":
    unittest.main()


class HubHandoffTests(HandoffTests):
    operation = "hub_average"
    fixture_index = 8


class ScanHandoffTests(HandoffTests):
    operation = "prefix_scan"
    mutable_buffer = "Data"

    def test_unhashed_later_stage_is_rejected(self):
        directory = self.manifest["cases"][0]["directory"]
        del self.manifest["files"][directory + "/kernel.1.ascend"]
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "missing from handoff"):
            handoff.verify_bundle(self.bundle)

    def test_first_kernel_alone_cannot_replace_program(self):
        directory = self.manifest["cases"][0]["directory"]
        path = self.bundle / directory / "bindings.json"
        path.write_text(
            json.dumps(json.loads(path.read_text())["stages"][0]["bindings"])
        )
        self.manifest["files"][directory + "/bindings.json"] = handoff.digest(path)
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "ordered program"):
            handoff.verify_bundle(self.bundle)


class GraphHandoffTests(HandoffTests):
    operation = "adjacency_build"
    mutable_buffer = "Counts"


class BucketHandoffTests(HandoffTests):
    operation = "bucket_build"
    mutable_buffer = "Buckets"


class ColoringHandoffTests(HandoffTests):
    operation = "topo_step"
    mutable_buffer = "Pending"


class BoundedColoringHandoffTests(ScanHandoffTests):
    operation = "topo_bounded"
    mutable_buffer = "Colors"


class ColoredSolverHandoffTests(ScanHandoffTests):
    operation = "colored_program"
    mutable_buffer = "Tiles"


class CoefficientHandoffTests(HandoffTests):
    operation = "contact_coefficients"
    mutable_buffer = "Coefficients"


class TgsHandoffTests(ScanHandoffTests):
    operation = "tgs_frame"
    mutable_buffer = "Velocities"


class BodyUniformHandoffTests(HandoffTests):
    operation = "body_coefficients"
    mutable_buffer = "Params"


class BroadPhaseHandoffTests(ScanHandoffTests):
    operation = "broadphase_pairs"
    mutable_buffer = "PairCount"


class NarrowPhaseHandoffTests(HandoffTests):
    operation = "analytic_contacts"
    mutable_buffer = "Contacts"


class ContactPlanHandoffTests(HandoffTests):
    operation = "contact_plan"
    mutable_buffer = "Plan"


class CollisionPipelineHandoffTests(ScanHandoffTests):
    operation = "analytic_collision"
    mutable_buffer = "Contacts"


class RadixHandoffTests(ScanHandoffTests):
    operation = "radix_sort"
    mutable_buffer = "Values"


class CanonicalHandoffTests(ScanHandoffTests):
    operation = "canonical_contacts"
    mutable_buffer = "Output"


class BoundaryHandoffTests(HandoffTests):
    operation = "solve_boundary"

    def test_counterfactual_inputs_remain_required(self):
        directory = self.manifest["cases"][0]["directory"]
        del self.manifest["files"][directory + "/HardTiles.bin"]
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "missing from handoff"):
            handoff.verify_bundle(self.bundle)


class TwistHandoffTests(HandoffTests):
    operation = "twist_solve"
    mutable_buffer = "Tiles"


class RenderHandoffTests(HandoffTests):
    operation = "prep_render"
    mutable_buffer = "Positions"


class MeshQueryHandoffTests(HandoffTests):
    operation = "mesh_projection"
    mutable_buffer = "Results"


class RayQueryHandoffTests(HandoffTests):
    operation = "ray_point"
    mutable_buffer = "Results"


class ShapeQueryHandoffTests(HandoffTests):
    operation = "shape_project_world_boundary"
    mutable_buffer = "Results"
