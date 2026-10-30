"""Production pipeline comparisons cannot silently accept a different scene."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from production_reference import check_case, check_upload


class ProductionReferenceTests(unittest.TestCase):
    def fixture(self):
        layout = {
            "pose_words": 8,
            "velocity_words": 8,
            "pose_translation": 0,
            "pose_rotation": 4,
            "velocity_linear": 0,
            "velocity_angular": 4,
        }
        case = {
            "name": "production-test",
            "contact_reduction": True,
            "body_count": 1,
            "batches": 1,
            "trace_stride": 16,
            "frames": 2,
        }
        trace = np.zeros(32, np.uint32)
        trace.view(np.float32)[[7, 23]] = 1
        before = {
            "FrameIndex": np.array([0, 0xFEED], np.uint32),
            "Trajectory": np.r_[trace, np.array([11, 22, 33], np.uint32)],
        }
        expected = {
            "FrameIndex": np.array([2, 0xFEED], np.uint32),
            "Trajectory": before["Trajectory"].copy(),
            "Poses": trace[16:24].copy(),
            "Velocities": trace[24:].copy(),
        }
        metadata = {
            "kind": "production_rbd_pipeline",
            "frames": 2,
            "bodies": 1,
            "batches": 1,
            "broadphase": "lbvh",
            "deterministic": True,
            "contact_reduction": True,
        }
        return (
            layout,
            case,
            before,
            expected,
            trace,
            metadata,
            {"atol": 2e-6, "rtol": 2e-5},
        )

    def test_equal_trace_and_frame_prefix(self):
        args = self.fixture()
        self.assertEqual(check_case(*args), 0)
        l, c, b, e, trace, metadata, k = args
        c["frames"] = 1
        e["FrameIndex"][0] = 1
        e["Trajectory"] = np.r_[trace[:16], b["Trajectory"][-3:]]
        self.assertEqual(check_case(l, c, b, e, trace, metadata, k), 0)

    def test_rejects_state_padding_and_nonfinite_changes(self):
        for index, value in [(0, 1.0), (24, 1.0), (3, 1e-20), (7, float("nan"))]:
            args = self.fixture()
            args[4].view(np.float32)[index] = value
            with (
                self.subTest(index=index),
                self.assertRaises((ValueError, AssertionError)),
            ):
                check_case(*args)

    def test_rejects_metadata_and_storage_changes(self):
        for field, value in [
            ("frames", 3),
            ("bodies", 2),
            ("batches", 2),
            ("contact_reduction", False),
            ("deterministic", False),
            ("broadphase", "brute_force"),
        ]:
            args = self.fixture()
            args[5][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                check_case(*args)
        args = list(self.fixture())
        args[4] = args[4][:-1]
        with self.assertRaises(ValueError):
            check_case(*args)

    def test_native_physical_inputs_remain_exact(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = (Path(tmp) / n for n in ("composed", "production"))
            layout = {
                "batch": {
                    "collision_pairs_capacity": 1,
                    "contacts_capacity": 2,
                    "deterministic": 3,
                }
            }
            for d in (a, b):
                d.mkdir()
                (d / "scene.json").write_text(json.dumps({"bodies": 1}))
                np.array([1, 64, 128, 1], np.uint32).tofile(d / "BatchIds.bin")
                np.array([1, 2, 3], np.uint32).tofile(d / "Poses.bin")
            np.array([1, 128, 256, 1], np.uint32).tofile(b / "BatchIds.bin")
            check_upload(layout, a, b)
            np.array([2, 128, 256, 1], np.uint32).tofile(b / "BatchIds.bin")
            with self.assertRaises(AssertionError):
                check_upload(layout, a, b)
            np.array([1, 128, 256, 1], np.uint32).tofile(b / "BatchIds.bin")
            np.array([1, 2, 4], np.uint32).tofile(b / "Poses.bin")
            with self.assertRaisesRegex(ValueError, "scene input"):
                check_upload(layout, a, b)
            (b / "Poses.bin").write_bytes((a / "Poses.bin").read_bytes())
            (b / "extra.bin").write_bytes(b"extra")
            with self.assertRaisesRegex(ValueError, "file set"):
                check_upload(layout, a, b)


if __name__ == "__main__":
    unittest.main()
