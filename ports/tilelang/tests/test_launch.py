"""GPU launch metadata must retain the original IR extents, including singleton grids."""

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ordered import validate_launch
from primitive_kernels import make_kernel


class LaunchTests(unittest.TestCase):
    target = os.environ.get("NEXUS_TILELANG_LAUNCH_TARGET", "metal")

    def test_singleton_grid_cannot_be_overlaunched(self):
        kernel = make_kernel({"rows": 1, "storage": 70}, self.target)
        validate_launch(kernel, self.target, 1, 64)
        with self.assertRaisesRegex(ValueError, "launch mismatch"):
            validate_launch(kernel, self.target, 2, 64)

    def test_grid_and_thread_extents_are_both_required(self):
        kernel = make_kernel({"rows": 2, "storage": 140}, self.target)
        validate_launch(kernel, self.target, 2, 64)
        for blocks, threads in ((1, 64), (3, 64), (2, 32), (2, 128)):
            with (
                self.subTest(blocks=blocks, threads=threads),
                self.assertRaisesRegex(ValueError, "launch mismatch"),
            ):
                validate_launch(kernel, self.target, blocks, threads)

    def test_serial_c_does_not_dispatch_gpu_workgroups(self):
        kernel = make_kernel({"rows": 2, "storage": 140}, "c")
        validate_launch(kernel, "c", 2, 64)


if __name__ == "__main__":
    unittest.main()
