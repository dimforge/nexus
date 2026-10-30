"""Reject control ABI changes that could retry after committing a frame."""

import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from frame_manifest import from_program, persistent_regions, validate_frame


def frame():
    local = {
        "schema": 1,
        "entry": "kernel",
        "buffers": ["A"],
        "storage_buffers": ["A"],
        "buffer_bytes": {"A": 16},
        "blocks": 1,
        "threads": 1,
        "iterations": 1,
    }
    return {
        "schema": 1,
        "max_steps": 2,
        "program": {
            "schema": 2,
            "storage_buffers": ["State", "Count"],
            "buffer_bytes": {"State": 16, "Count": 16},
            "iterations": 1,
            "stages": [
                {
                    "name": "count",
                    "source": "count.metal",
                    "bindings": copy.deepcopy(local),
                    "buffer_map": {"A": "Count"},
                },
                {
                    "name": "commit",
                    "source": "commit.metal",
                    "bindings": copy.deepcopy(local),
                    "buffer_map": {"A": "State"},
                },
            ],
        },
        "gates": [
            {
                "id": "pairs",
                "after_stage": "count",
                "counters": [{"buffer": "Count", "word": 0}],
                "capacity": 1,
            }
        ],
        "persistent": {"pose": {"buffer": "State", "word_start": 0, "word_len": 2}},
    }


class FrameManifestTests(unittest.TestCase):
    def test_valid_control_abi(self):
        validate_frame(frame())

    def test_unbounded_control_requires_an_explicit_limit_and_no_trace(self):
        f = frame()
        f["max_steps"] = None
        validate_frame(f)
        del f["max_steps"]
        with self.assertRaises(ValueError):
            validate_frame(f)
        f["max_steps"] = None
        f["program"]["storage_buffers"].append("Trajectory")
        f["program"]["buffer_bytes"]["Trajectory"] = 4
        with self.assertRaises(ValueError):
            validate_frame(f)

    def test_gate_after_commit_or_outside_counter_is_rejected(self):
        for edit in (
            lambda f: f["gates"][0].update(after_stage="commit"),
            lambda f: f["gates"][0]["counters"][0].update(word=4),
            lambda f: f["gates"][0].update(capacity=True),
            lambda f: f["gates"][0].update(counters=[]),
        ):
            f = frame()
            edit(f)
            with self.assertRaises(ValueError):
                validate_frame(f)
        f = frame()
        f["gates"][0]["counters"] *= 2
        with self.assertRaises(ValueError):
            validate_frame(f)

    def test_outer_iterations_and_trace_limit_cannot_advance_extra_frames(self):
        for field, value in [
            ("iterations", 2),
            ("max_steps", 0),
            ("max_steps", True),
            ("max_steps", 1 << 32),
        ]:
            f = frame()
            (f["program"] if field == "iterations" else f)[field] = value
            with self.assertRaises(ValueError):
                validate_frame(f)

    def test_persistent_ranges_reject_overlap_and_truncation(self):
        for start, length in [(3, 2), (-1, 1), (0, 0), (0, True)]:
            f = frame()
            f["persistent"]["pose"].update(word_start=start, word_len=length)
            with self.assertRaises(ValueError):
                validate_frame(f)
        f = frame()
        f["persistent"]["velocity"] = {
            "buffer": "State",
            "word_start": 1,
            "word_len": 2,
        }
        with self.assertRaises(ValueError):
            validate_frame(f)

    def test_native_regions_exclude_variable_capacity_guards(self):
        l = {
            "pose_words": 8,
            "velocity_words": 8,
            "link_words": 16,
            "tile_words": 4224,
            "state_words": 20,
        }
        c = {"body_count": 3, "contact_capacity": 66, "buffer_words": {}}
        p = persistent_regions(l, c)
        self.assertEqual(p["OldLinks"]["word_len"], 66 * 16)
        self.assertEqual(p["OldIds"]["word_len"], 132)
        self.assertEqual(p["OldTiles"]["word_len"], 3 * 4224)
        self.assertEqual(
            p["States"], {"buffer": "States", "word_start": 0, "word_len": 1320}
        )
        self.assertNotIn("SolverPoses", p)
        c.update(buffer_words={"FrameIndex": 2}, frames=8, trace_stride=48)
        self.assertEqual(persistent_regions(l, c)["Trajectory"]["word_len"], 384)
        c["kind"] = "runtime_scene"
        c["buffer_words"]["Gravity"] = 4
        self.assertEqual(
            persistent_regions(l, c)["Gravity"],
            {"buffer": "Gravity", "word_start": 0, "word_len": 4},
        )

    def test_export_requires_matching_compiled_schedule_and_real_gates(self):
        f = frame()
        p = f["program"]
        plans = [
            dict(
                s,
                capacity_gate={
                    "id": "pairs",
                    "counters": [{"buffer": "A", "word": 0}],
                    "capacity": 1,
                },
            )
            if i == 0
            else dict(s)
            for i, s in enumerate(p["stages"])
        ]
        l = {
            "pose_words": 1,
            "velocity_words": 1,
            "link_words": 1,
            "tile_words": 1,
            "state_words": 1,
        }
        c = {
            "body_count": 1,
            "contact_capacity": 1,
            "buffer_words": {},
            "kind": "analytic_scene",
            "frames": 2,
        }
        # Reject before attempting to publish any semantic state mappings.
        plans[0]["name"] = "wrong"
        with self.assertRaisesRegex(ValueError, "stage differs"):
            from_program(l, c, plans, p)
        plans[0]["name"] = "count"
        del plans[0]["capacity_gate"]
        with self.assertRaisesRegex(ValueError, "capacity gates"):
            from_program(l, c, plans, p)


if __name__ == "__main__":
    unittest.main()
