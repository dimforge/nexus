import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from frame_cpu import CapacityError, CpuFrameSession


def variant(cap, offset=0):
    sizes = {"State": 4 * (offset + 4), "Count": 4, "Input": 4}

    def stage(name, names, entry):
        return {
            "name": name,
            "source": name + ".c",
            "bindings": {
                "schema": 1,
                "entry": entry,
                "buffers": names,
                "storage_buffers": names,
                "buffer_bytes": {n: sizes[n] for n in names},
                "blocks": 1,
                "threads": 1,
                "iterations": 1,
            },
            "buffer_map": {n: n for n in names},
        }

    p = {
        "schema": 2,
        "storage_buffers": list(sizes),
        "buffer_bytes": sizes,
        "iterations": 1,
        "stages": [
            stage("count", ["Input", "Count"], "count"),
            stage("commit", ["State"], "commit"),
        ],
    }
    state = np.full(offset + 4, 0xDEADBEEF, np.uint32)
    state[offset : offset + 2] = [42, 17]
    return {
        "manifest": {
            "schema": 1,
            "program": p,
            "max_steps": 3,
            "gates": [
                {
                    "id": "pairs",
                    "after_stage": "count",
                    "counters": [{"buffer": "Count", "word": 0}],
                    "capacity": cap,
                }
            ],
            "persistent": {
                "state": {"buffer": "State", "word_start": offset, "word_len": 2}
            },
        },
        "sources": {
            "count.c": '#include <cstdint>\nextern "C" int32_t count(uint32_t* input,uint32_t* count) {count[0]=input[0];return 0;}',
            "commit.c": f'#include <cstdint>\nextern "C" int32_t commit(uint32_t* state) {{state[{offset}]+=1;return 0;}}',
        },
        "initial": {
            "State": state,
            "Count": np.zeros(1, np.uint32),
            "Input": np.array([5], np.uint32),
        },
    }


class FrameCpuTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def test_updates_migrate_and_rejections_preserve_guards(self):
        s = CpuFrameSession([variant(1), variant(8, 3)], self.temp.name)
        s.write_persistent("state", 0, np.array([100, 200], np.uint32))
        before = s.storage["State"].copy()
        for name, offset, words in [
            ("missing", 0, np.array([9], np.uint32)),
            ("state", -1, np.array([9], np.uint32)),
            ("state", 1, np.array([9, 10], np.uint32)),
            ("state", 0, np.array([9.0], np.float32)),
        ]:
            with self.assertRaises(ValueError):
                s.write_persistent(name, offset, words)
            np.testing.assert_array_equal(s.storage["State"], before)
        self.assertEqual(s.step()["retries"], 1)
        np.testing.assert_array_equal(
            s.storage["State"], [0xDEADBEEF] * 3 + [101, 200] + [0xDEADBEEF] * 2
        )
        s.write_persistent("state", 1, np.array([300], np.uint32))
        s.step()
        self.assertEqual(s.storage["State"][3:5].tolist(), [102, 300])
        s.poisoned = True
        with self.assertRaisesRegex(RuntimeError, "Poisoned"):
            s.write_persistent("state", 0, np.array([0], np.uint32))
        self.assertEqual(s.storage["State"][3], 102)

    def test_growth_after_prior_frame_preserves_state_and_limits(self):
        small, large = variant(1), variant(8, 3)
        small["initial"]["Input"][0] = 1
        large["initial"]["State"][3:5] = 0
        s = CpuFrameSession([small, variant(3, 1), large], self.temp.name)
        self.assertEqual(s.step()["retries"], 0)
        self.assertEqual(s.storage["State"][0], 43)
        s.storage["Input"][0] = 5
        stats = s.step()
        self.assertEqual((stats["variant"], stats["retries"], stats["step"]), (2, 1, 2))
        np.testing.assert_array_equal(
            s.storage["State"], [0xDEADBEEF] * 3 + [44, 17] + [0xDEADBEEF] * 2
        )
        s.step()
        with self.assertRaisesRegex(RuntimeError, "step limit"):
            s.step()
        self.assertEqual(s.storage["State"][3], 45)

    def test_new_capacity_can_be_compiled_after_exhaustion_without_a_trace(self):
        small, large = variant(1), variant(8, 3)
        for v in (small, large):
            v["manifest"]["max_steps"] = None
        s = CpuFrameSession([small], self.temp.name)
        with self.assertRaises(CapacityError):
            s.step()
        s.extend(large)
        self.assertEqual(s.steps, 0)
        self.assertEqual(s.storage["State"][0], 42)
        self.assertEqual(s.step()["retries"], 1)
        for _ in range(9):
            s.step()
        self.assertEqual(s.steps, 10)
        self.assertEqual(s.storage["State"][3:5].tolist(), [52, 17])
        s.steps = 0xFFFFFFFFFFFFFFFF
        with self.assertRaisesRegex(RuntimeError, "step limit"):
            s.step()
        self.assertEqual(s.storage["State"][3], 52)

    def test_rejected_extension_does_not_mutate_the_session(self):
        s = CpuFrameSession([variant(8)], self.temp.name)
        for v in (variant(8), variant(1)):
            with self.assertRaises(ValueError):
                s.extend(v)
            self.assertEqual(len(s.variants), 1)
            self.assertEqual(s.storage["State"][0], 42)
        self.assertEqual(s.step()["step"], 1)

    def test_exhaustion_does_not_advance(self):
        s = CpuFrameSession([variant(1)], self.temp.name)
        for _ in range(2):
            with self.assertRaises(CapacityError) as error:
                s.step()
            self.assertEqual(
                (error.exception.required, error.exception.capacity), (5, 1)
            )
            self.assertEqual(s.steps, 0)
            self.assertEqual(s.storage["State"][0], 42)

    def test_failed_growth_compilation_keeps_state(self):
        bad = variant(8, 2)
        good = bad["sources"]["commit.c"]
        bad["sources"]["commit.c"] = "invalid source"
        s = CpuFrameSession([variant(1), bad], self.temp.name)
        with self.assertRaises(RuntimeError):
            s.step()
        self.assertEqual((s.steps, s.current, s.poisoned), (0, 0, False))
        self.assertEqual(s.storage["State"][0], 42)
        bad["sources"]["commit.c"] = good
        s.step()
        self.assertEqual(s.storage["State"][2], 43)

    def test_partial_commit_failure_poisoned(self):
        v = variant(8)
        v["sources"]["commit.c"] = v["sources"]["commit.c"].replace(
            "return 0", "return 1"
        )
        s = CpuFrameSession([v], self.temp.name)
        with self.assertRaisesRegex(RuntimeError, "kernel failed"):
            s.step()
        with self.assertRaisesRegex(RuntimeError, "Poisoned"):
            s.step()
        self.assertEqual(s.steps, 0)
        self.assertEqual(s.storage["State"][0], 43)

    def test_invalid_variant_cannot_shrink_or_change_meaning(self):
        for a, b in [(variant(8), variant(1)), (variant(1), variant(1))]:
            with self.assertRaises(ValueError):
                CpuFrameSession([a, b], self.temp.name)
        a, b = variant(1), variant(8)
        b["manifest"]["persistent"]["state"]["word_len"] = 1
        with self.assertRaises(ValueError):
            CpuFrameSession([a, b], self.temp.name)
        a, b = variant(1), variant(8)
        b["manifest"]["gates"][0]["counters"][0]["buffer"] = "Input"
        with self.assertRaises(ValueError):
            CpuFrameSession([a, b], self.temp.name)


if __name__ == "__main__":
    unittest.main()
