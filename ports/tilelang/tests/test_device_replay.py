"""Replay plumbing with explicitly host-only compiled kernels."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from device_replay import execute_program, load_bundle, replay_directory
from test_device_frame import HostTestDevice
from test_frame_cpu import variant
from test_runtime_service import write_variant


class DeviceReplayTests(unittest.TestCase):
    def test_ordered_program_preserves_iterations_and_borrowed_device(self):
        value = variant(8)
        program = dict(value["manifest"]["program"], iterations=3)
        arrays = {n: a.copy() for n, a in value["initial"].items()}
        with tempfile.TemporaryDirectory() as tmp:
            device = HostTestDevice()
            environment = execute_program(
                value["sources"], program, Path(tmp), arrays, "cuda", device=device
            )
            self.assertEqual(arrays["State"][:2].tolist(), [45, 17])
            self.assertEqual(environment["backend"], "host-interface-test-only")
            self.assertEqual(device.allocations, {})
            self.assertFalse(device.closed)
            device.close()

    def test_prepared_scene_checks_target_steps_and_writes_actuals(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "variant"
            value = variant(8)
            write_variant(directory, value)
            meta = directory / "specialization.json"
            metadata = json.loads(meta.read_text())
            metadata["target"] = "cuda"
            meta.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "target differs"):
                load_bundle(directory, "ascend")
            with self.assertRaisesRegex(ValueError, "frame count"):
                load_bundle(directory, "cuda", 0)
            device = HostTestDevice()
            with patch("device_replay.create_device", return_value=device):
                result = replay_directory(directory, "cuda", 2)
            self.assertEqual(result["steps"], 2)
            self.assertEqual(
                result["comparison"], "not_run; use the contract replay runner"
            )
            self.assertTrue(device.closed)
            np.testing.assert_array_equal(
                np.fromfile(directory / "State.actual.bin", np.uint32)[:2], [44, 17]
            )
            self.assertEqual(len(json.loads((directory / "stats.json").read_text())), 2)
