"""Device transfer/control contracts on host memory, not GPU execution evidence."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from device_frame import DeviceFrameSession
from frame_cpu import CapacityError
from ordered import CompilationSession
from replay import execute_c
from test_frame_cpu import variant


class HostTestDevice:
    """Actual generated C kernels with an instrumented device-shaped interface."""

    def __init__(self):
        self.environment = {"backend": "host-interface-test-only"}
        self.compiler = CompilationSession()
        self.reads, self.copies, self.allocations = [], [], {}
        self.waits = 0
        self.closed = False
        self.fail_wait = False

    def upload(self, words):
        result = words.copy()
        self.allocations[id(result)] = result
        return result

    def free(self, allocation):
        del self.allocations[id(allocation)]

    def read(self, allocation, start, count):
        self.reads.append((start, count))
        return allocation[start : start + count].copy()

    def write(self, allocation, start, words):
        allocation[start : start + len(words)] = words

    def copy(self, dst, ds, src, ss, count):
        self.copies.append((ds, ss, count))
        dst[ds : ds + count] = src[ss : ss + count]

    def compile(self, source, bindings, directory):
        return source, self.compiler.compile_c(source, bindings, directory), directory

    def launch(self, compiled, allocations, bindings):
        source, native, directory = compiled
        execute_c(
            source,
            bindings,
            directory,
            dict(zip(bindings["buffers"], allocations)),
            compiled=native,
        )

    def synchronize(self):
        self.waits += 1
        if self.fail_wait:
            raise RuntimeError("asynchronous device failure")

    def close(self):
        self.closed = True
        self.allocations.clear()


class DeviceFrames(unittest.TestCase):
    def test_growth_uses_device_copies_and_only_reads_gate_counters(self):
        with tempfile.TemporaryDirectory() as tmp:
            device = HostTestDevice()
            runtime = DeviceFrameSession([variant(1), variant(8, 3)], Path(tmp), device)
            progress = runtime.step()
            self.assertEqual((progress["step"], progress["retries"]), (1, 1))
            self.assertEqual(device.copies, [(3, 0, 2)])
            self.assertEqual(device.reads, [(0, 1), (0, 1)])
            self.assertEqual(runtime.read("State")[3:5].tolist(), [43, 17])
            checkpoint = runtime.checkpoint()
            runtime.step()
            runtime.restore(checkpoint)
            self.assertEqual(runtime.checkpoint(), checkpoint)
            runtime.reset()
            self.assertEqual(runtime.checkpoint()["regions"]["state"], [42, 17])
            runtime.close()
            self.assertTrue(device.closed)
            self.assertEqual(device.allocations, {})

    def test_failed_growth_keeps_original_allocations_and_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            device = HostTestDevice()
            runtime = DeviceFrameSession([variant(1)], Path(tmp), device)
            with self.assertRaises(CapacityError):
                runtime.step()
            old_ids = set(device.allocations)
            checkpoint = runtime.checkpoint()
            runtime.extend(variant(8, 3))
            upload = device.upload
            calls = 0

            def fail_later(words):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise MemoryError("allocation failed")
                return upload(words)

            with (
                patch.object(device, "upload", side_effect=fail_later),
                self.assertRaises(MemoryError),
            ):
                runtime.step()
            self.assertEqual(set(device.allocations), old_ids)
            self.assertEqual(runtime.checkpoint(), checkpoint)
            self.assertEqual(runtime.step()["step"], 1)
            runtime.close()

    def test_async_failure_poisoned_before_ownership_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            device = HostTestDevice()
            runtime = DeviceFrameSession([variant(8)], Path(tmp), device)
            device.fail_wait = True
            with self.assertRaisesRegex(RuntimeError, "asynchronous"):
                runtime.step()
            self.assertEqual(runtime.steps, 0)
            with self.assertRaisesRegex(RuntimeError, "Poisoned"):
                runtime.step()
            runtime.close()

    def test_device_rejects_native_indirect_instead_of_reading_dispatch_size(self):
        v = variant(8)
        v["manifest"]["program"]["stages"][0]["indirect"] = {
            "buffer": "State",
            "word": 0,
        }
        with tempfile.TemporaryDirectory() as tmp:
            device = HostTestDevice()
            runtime = DeviceFrameSession([v], Path(tmp), device)
            with self.assertRaisesRegex(ValueError, "virtual-grid"):
                runtime.step()
            self.assertEqual(device.reads, [])
            runtime.close()

    def test_failed_reset_discards_unpublished_storage(self):
        with tempfile.TemporaryDirectory() as tmp:
            device = HostTestDevice()
            runtime = DeviceFrameSession([variant(8)], Path(tmp), device)
            before = set(device.allocations)
            device.fail_wait = True
            with self.assertRaises(RuntimeError):
                runtime.reset()
            self.assertEqual(set(device.allocations), before)
            self.assertTrue(runtime.poisoned)
            runtime.close()

    def test_borrowed_device_survives_close_and_failed_read_poisoned(self):
        with tempfile.TemporaryDirectory() as tmp:
            device = HostTestDevice()
            runtime = DeviceFrameSession(
                [variant(8)], Path(tmp), device, owns_device=False
            )
            with (
                patch.object(device, "read", side_effect=RuntimeError("copy failed")),
                self.assertRaisesRegex(RuntimeError, "copy failed"),
            ):
                runtime.read("State")
            self.assertTrue(runtime.poisoned)
            runtime.close()
            self.assertFalse(device.closed)
            self.assertEqual(device.allocations, {})
            device.close()
