"""Warmstart graph ownership, cache semantics, rounding order and mutation guards."""

import copy
import json
import os
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from replay import HERE
from warmstart_replay import BUFFERS, OPERATIONS, check_outputs, validate_case


class WarmstartTests(unittest.TestCase):
    def load(self, op="gather_warmstart", name="n129"):
        root = (
            Path(os.environ.get("NEXUS_TILELANG_RESULTS", HERE / "results"))
            / op
            / "c/fixtures"
        )
        self.layout = json.loads((root / "layout.json").read_text())
        self.cases = json.loads((root / op / "suite.json").read_text())["cases"]
        self.case = next(c for c in self.cases if c["name"] == name)
        folder = root / op / name
        self.contract = json.loads((HERE / f"contracts/{op}.json").read_text())
        self.before = {
            k: np.fromfile(folder / f"{k}.bin", dtype=np.uint32) for k in BUFFERS
        }
        self.actual = {
            k: np.fromfile(folder / f"{k}.actual.bin", dtype=np.uint32) for k in BUFFERS
        }
        self.expected = {
            k: np.fromfile(folder / f"{k}.expected.bin", dtype=np.uint32)
            if k in self.contract["outputs"]
            else self.before[k]
            for k in BUFFERS
        }

    def check(self):
        return check_outputs(
            self.layout,
            self.case,
            self.before,
            self.actual,
            self.expected,
            self.contract,
        )

    def velocities(self, which):
        return getattr(self, which)["Velocities"].reshape(
            -1, self.layout["velocity_words"]
        )

    def test_all_operations(self):
        for op in OPERATIONS:
            self.load(op)
            self.assertEqual(
                set(self.contract["required_cases"]), {c["name"] for c in self.cases}
            )
            for c in self.cases:
                self.load(op, c["name"])
                self.check()

    def test_zero_and_unit_scaling(self):
        for name in ("n31", "n32"):
            self.load("scale_impulses", name)
            l = self.layout
            for i in range(self.case["count"]):
                base = i // 32 * l["tile_words"] + i % 32 * 4
                count = int(self.before["Tiles"][base + l["header"]["normal"] + 3])
                for k in range(count):
                    point = base + l["points"] + k * l["point_words"]
                    for off in (
                        l["point"]["arm_a"] + 3,
                        l["point"]["friction"],
                        l["point"]["friction"] + 1,
                    ):
                        a = self.actual["Tiles"][point + off]
                        if name == "n31":
                            self.assertEqual(a.view(np.float32), 0)
                        else:
                            self.assertEqual(a, self.before["Tiles"][point + off])
                for k in ("diag_b", "off_b"):
                    self.assertEqual(self.actual["Tiles"][base + l["header"][k] + 3], 0)
                np.testing.assert_array_equal(
                    self.actual["Tiles"][
                        base + l["coupling"] + count : base + l["coupling"] + 4
                    ],
                    0,
                )
            self.check()

    def test_static_hub_and_spare_velocities_are_preserved(self):
        self.load()
        np.testing.assert_array_equal(
            self.velocities("actual")[0:2], self.velocities("before")[0:2]
        )
        np.testing.assert_array_equal(
            self.velocities("actual")[self.case["bodies"] :],
            self.velocities("before")[self.case["bodies"] :],
        )
        self.assertFalse(
            np.array_equal(self.velocities("actual")[2], self.velocities("before")[2])
        )
        self.check()

    def test_cache_order_is_preserved(self):
        self.load()
        l = self.layout
        start = int(self.before["Counts"][1])
        end = int(self.before["Counts"][2])
        values = []
        for cid in self.before["Ids"][start:end]:
            i = int(self.before["Indices"][cid])
            pair = (
                i // 32 * l["tile_words"]
                + l["warmstart"]
                + i % 32 * 2 * l["velocity_words"]
            )
            side = 0 if self.before["Tiles"][pair + l["velocity_padding1"]] == 2 else 1
            values.append(
                self.before["Tiles"].view(np.float32)[
                    pair + side * l["velocity_words"] + l["velocity_linear"]
                ]
            )
        np.testing.assert_array_equal(values[:3], [1e8, 3, -1e8])
        original = self.velocities("before").view(np.float32)[2, l["velocity_linear"]]

        def accumulate(sequence):
            result = original
            for _ in range(self.case["iterations"]):
                for value in sequence:
                    result = np.float32(result + value)
            return result

        ordered = accumulate(values)
        self.assertGreater(
            abs(float(ordered - accumulate(list(reversed(values))))), 1.0
        )
        self.assertEqual(
            self.velocities("actual").view(np.float32)[2, l["velocity_linear"]], ordered
        )

    def test_runtime_multi_batch_range(self):
        self.load(name="n65")
        l = self.layout
        count = int(self.before["BatchIds"][l["batch"]["bodies_len"]])
        self.assertEqual(self.case["bodies"], count * 2)
        self.assertFalse(
            np.array_equal(
                self.velocities("before")[count : count * 2],
                self.velocities("actual")[count : count * 2],
            )
        )
        self.check()

    def test_hub_recomputes_and_clips_live_slots(self):
        self.load("hub_warmstart", "n65")
        l = self.layout
        hc = {k: int(self.before["HubCounts"][v]) for k, v in l["hub"].items()}
        self.assertGreater(hc["slots"], hc["pool"])
        np.testing.assert_array_equal(
            self.velocities("actual")[: hc["base"]],
            self.velocities("before")[: hc["base"]],
        )
        np.testing.assert_array_equal(
            self.velocities("actual")[hc["base"] + hc["pool"] :],
            self.velocities("before")[hc["base"] + hc["pool"] :],
        )
        cid = self.before["HubSlotConstraint"][0]
        i = int(self.before["Indices"][cid])
        pair = (
            i // 32 * l["tile_words"]
            + l["warmstart"]
            + i % 32 * 2 * l["velocity_words"]
        )
        side = 0 if self.before["Tiles"][pair + l["velocity_padding1"]] == 1 else 1
        dv = self.before["Tiles"].view(np.float32)[
            pair + side * l["velocity_words"] + l["velocity_linear"]
        ]
        cached = self.velocities("before").view(np.float32)[
            hc["base"], l["velocity_linear"]
        ]
        for _ in range(self.case["iterations"]):
            cached = np.float32(cached + dv)
        self.assertGreater(
            abs(
                float(
                    cached
                    - self.velocities("actual").view(np.float32)[
                        hc["base"], l["velocity_linear"]
                    ]
                )
            ),
            0.1,
        )
        self.check()

    def test_bad_graph_and_uniforms_are_rejected(self):
        self.load()
        l = self.layout
        for name, index, value in (
            ("Counts", 0, 2**32 - 1),
            ("Ids", 0, 2**32 - 1),
            ("Indices", int(self.before["Ids"][0]), self.case["count"]),
            ("Buckets", self.case["bucket_storage"] - 1, self.case["count"] + 1),
            ("BatchIds", l["batch"]["num_batches"], 0),
            ("HubCounts", l["hub"]["pool"], self.case["velocities"]),
        ):
            arrays = copy.deepcopy(self.before)
            arrays[name][index] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_case(l, self.case, arrays)
        arrays = copy.deepcopy(self.before)
        i = int(arrays["Indices"][arrays["Ids"][0]])
        pair = (
            i // 32 * l["tile_words"]
            + l["warmstart"]
            + i % 32 * 2 * l["velocity_words"]
        )
        for side in (0, 1):
            arrays["Tiles"][
                pair + side * l["velocity_words"] + l["velocity_padding1"]
            ] = 2**32 - 1
        with self.assertRaisesRegex(ValueError, "tag"):
            validate_case(l, self.case, arrays)

    def test_bad_points_hub_owner_and_layout_are_rejected(self):
        self.load("hub_warmstart")
        l = self.layout
        i = int(self.before["Indices"][self.before["HubSlotConstraint"][0]])
        base = i // 32 * l["tile_words"] + i % 32 * 4
        for keys in (("normal",), ("diag_a", "off_a")):
            arrays = copy.deepcopy(self.before)
            for key in keys:
                arrays["Tiles"][base + l["header"][key] + 3] = 2**32 - 1
            with self.assertRaises(ValueError):
                validate_case(l, self.case, arrays)
        layout = copy.deepcopy(l)
        layout["hub"]["base"] = layout["hub"]["slots"]
        with self.assertRaises(ValueError):
            validate_case(layout, self.case, self.before)

    def test_padding_readonly_and_nonfinite_outputs_fail(self):
        self.load()
        for name, index, value in (
            ("Velocities", self.layout["velocity_padding0"], 0),
            ("Counts", 0, 0),
            (
                "Velocities",
                2 * self.layout["velocity_words"] + self.layout["velocity_linear"],
                0x7FC00000,
            ),
        ):
            saved = self.actual[name].copy()
            self.actual[name][index] = value
            with self.subTest(name=name), self.assertRaises(AssertionError):
                self.check()
            self.actual[name] = saved
        self.load("scale_impulses")
        index = self.layout["warmstart"] + self.layout["velocity_padding1"]
        self.actual["Tiles"][index] = 2**32 - 1
        with self.assertRaises(AssertionError):
            self.check()


if __name__ == "__main__":
    unittest.main()
