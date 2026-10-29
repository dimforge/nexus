"""Hub allocation, reduction ordering, memory guards and exact copied-field checks."""

import json
import os
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hub_replay as hub
from replay import HERE


class HubTests(unittest.TestCase):
    def fixture(self, op, name="n65-mixed"):
        root = (
            Path(os.environ.get("NEXUS_TILELANG_RESULTS", HERE / "results"))
            / op
            / "c/fixtures"
        )
        layout = json.loads((root / "layout.json").read_text())
        suite = json.loads((root / op / "suite.json").read_text())
        case = next(c for c in suite["cases"] if c["name"] == name)
        directory = root / op / name
        before = {
            n: np.fromfile(directory / f"{n}.bin", dtype=np.uint32) for n in hub.BUFFERS
        }
        contract = json.loads((HERE / f"contracts/{op}.json").read_text())
        expected = {
            n: np.fromfile(directory / f"{n}.expected.bin", dtype=np.uint32)
            if n in contract["outputs"]
            else a.copy()
            for n, a in before.items()
        }
        return layout, case, before, expected, contract

    def test_every_required_case(self):
        for op in hub.OPERATIONS:
            contract = json.loads((HERE / f"contracts/{op}.json").read_text())
            for name in contract["required_cases"]:
                with self.subTest(op=op, case=name):
                    l, c, b, e, k = self.fixture(op, name)
                    hub.check_outputs(l, c, b, e, e, k)

    def test_malformed_graph_and_dimensions(self):
        for mutation in (
            "counts",
            "id",
            "duplicate",
            "owner",
            "mass",
            "batch",
            "dimensions",
        ):
            with self.subTest(mutation=mutation):
                l, c, b, _, _ = self.fixture("hub_assign")
                if mutation == "counts":
                    b["Counts"][1] = 0
                elif mutation == "id":
                    b["Ids"][0] = c["constraint_count"]
                elif mutation == "duplicate":
                    b["Ids"][1] = b["Ids"][0]
                elif mutation == "owner":
                    b["Links"][
                        int(b["Ids"][0]) * l["link_words"] + l["link"]["solver_body_a"]
                    ] = c["count"]
                elif mutation == "mass":
                    b["WorldMass"].view(np.float32)[l["world_mass"]["inv_mass"]] = (
                        np.nan
                    )
                elif mutation == "batch":
                    b["BatchIds"][l["batch"]["colliders_batch_capacity"]] += 1
                else:
                    c["groups"] = 0
                with self.assertRaises(ValueError):
                    hub.validate_case(l, c, b)

    def test_reset_and_hub_ranges(self):
        l, c, b, _, _ = self.fixture("hub_assign")
        b["HubCounts"][l["hub"]["slots"]] = 1
        with self.assertRaisesRegex(ValueError, "reset"):
            hub.validate_case(l, c, b)
        for op in ("hub_split", "hub_average", "hub_scatter"):
            l, c, b, _, _ = self.fixture(op)
            b["HubList"][1] = b["HubList"][0]
            with self.assertRaisesRegex(ValueError, "repeated"):
                hub.validate_case(l, c, b)
            l, c, b, _, _ = self.fixture(op)
            b["FirstSlots"][int(b["HubList"][1])] = b["FirstSlots"][
                int(b["HubList"][0])
            ]
            with self.assertRaisesRegex(ValueError, "ranges"):
                hub.validate_case(l, c, b)
        l, c, b, _, _ = self.fixture("hub_scatter")
        b["SlotBody"][0] = b["HubCounts"][l["hub"]["base"]]
        with self.assertRaisesRegex(ValueError, "scatter"):
            hub.validate_case(l, c, b)

    def test_atomic_order_is_semantic_and_capacity_bounded(self):
        for name in ("n65-mixed", "n65-short", "n33-zero"):
            l, c, b, e, k = self.fixture("hub_assign", name)
            context = hub.validate_case(l, c, b)
            for order in (
                sorted(context["candidates"], reverse=True),
                sorted(context["candidates"], key=context["candidates"].get),
            ):
                actual = {n: a.copy() for n, a in b.items()}
                cursor, selected = 0, []
                for body in order:
                    n = context["candidates"][body]
                    if cursor + n <= context["counts"]["pool"]:
                        actual["FirstSlots"][body] = cursor
                        selected.append(body)
                    cursor += n
                actual["HubList"][: len(selected)] = selected[::-1]
                actual["HubCounts"][l["hub"]["hubs"]] = len(selected)
                actual["HubCounts"][l["hub"]["slots"]] = cursor
                hub.check_outputs(l, c, b, actual, e, k)

    def test_invalid_atomic_results_fail(self):
        for mutation in (
            "total",
            "hub_total",
            "overlap",
            "wrong_body",
            "missing",
            "spare",
        ):
            with self.subTest(mutation=mutation):
                l, c, b, e, k = self.fixture("hub_assign")
                a = {n: v.copy() for n, v in e.items()}
                if mutation == "total":
                    a["HubCounts"][l["hub"]["slots"]] += 1
                elif mutation == "hub_total":
                    a["HubCounts"][l["hub"]["hubs"]] = c["hub_storage"] + 1
                elif mutation == "overlap":
                    a["FirstSlots"][int(a["HubList"][1])] = a["FirstSlots"][
                        int(a["HubList"][0])
                    ]
                elif mutation == "wrong_body":
                    a["HubList"][0] = 0
                elif mutation == "missing":
                    a["HubCounts"][l["hub"]["hubs"]] = 0
                    a["HubList"][:] = b["HubList"]
                    a["FirstSlots"][:] = b["FirstSlots"]
                else:
                    a["HubList"][-1] ^= np.uint32(1)
                with self.assertRaises(AssertionError):
                    hub.check_outputs(l, c, b, a, e, k)

    def test_capacity_not_active_body_count_and_vector_static_test(self):
        l, c, b, e, _ = self.fixture("hub_assign")
        batch = {n: int(b["BatchIds"][v]) for n, v in l["batch"].items()}
        active = batch["bodies_len"] * batch["num_batches"]
        selected = e["HubList"][: int(e["HubCounts"][l["hub"]["hubs"]])]
        self.assertTrue(any(body >= active for body in selected))
        masses = b["WorldMass"].view(np.float32).reshape(-1, l["world_mass_words"])
        self.assertTrue(
            all(masses[body, l["world_mass"]["inv_mass"]] == 0 for body in selected)
        )
        self.assertNotIn(0, selected)
        self.assertNotIn(19, selected)

    def test_clipped_dispatch_and_pool_tail_scatter(self):
        l, c, b, e, _ = self.fixture("hub_dispatch", "n65-clipped")
        pool = int(b["HubCounts"][l["hub"]["pool"]])
        self.assertEqual(int(e["SlotsIndirect"][0]), (pool + 63) // 64)
        self.assertEqual(int(e["AverageIndirect"][0]), min(pool // 32 + 1, 256))
        l, c, b, e, _ = self.fixture("hub_scatter", "n65-short")
        hc = {n: int(b["HubCounts"][v]) for n, v in l["hub"].items()}
        self.assertGreater(hc["slots"], hc["pool"])
        v = e["Velocities"].reshape(-1, l["velocity_words"])
        for slot in range(hc["pool"]):
            np.testing.assert_array_equal(
                v[hc["base"] + slot], v[int(b["SlotBody"][slot])]
            )

    def test_reduction_order_is_observable(self):
        l, c, b, e, k = self.fixture("hub_average", "n129-mixed")
        context = hub.validate_case(l, c, b)
        v = b["Velocities"].view(np.float32).reshape(-1, l["velocity_words"])
        body = next(
            body for body in context["selected"] if context["degrees"][body] >= 129
        )
        start = context["counts"]["base"] + int(b["FirstSlots"][body])
        n = int(context["degrees"][body])
        axis = l["velocity_linear"]
        total = np.float32(0)
        for slot in range(n):
            total = np.float32(total + v[start + slot, axis])
        a = {name: value.copy() for name, value in e.items()}
        a["Velocities"].view(np.float32)[body * l["velocity_words"] + axis] = (
            total * np.float32(1.0 / n)
        )
        with self.assertRaises(AssertionError):
            hub.check_outputs(l, c, b, a, e, k)

    def test_padding_readonly_and_nonfinite_output_fail(self):
        for op in hub.OPERATIONS:
            l, c, b, e, k = self.fixture(op)
            a = {n: v.copy() for n, v in e.items()}
            a["Counts"][-1] ^= np.uint32(1)
            with self.assertRaises(AssertionError):
                hub.check_outputs(l, c, b, a, e, k)
        for index in ("velocity_padding0", "velocity_linear"):
            l, c, b, e, k = self.fixture("hub_average")
            body = int(b["HubList"][0])
            a = {n: v.copy() for n, v in e.items()}
            a["Velocities"][body * l["velocity_words"] + l[index]] = np.float32(
                np.nan
            ).view(np.uint32)
            with self.assertRaises(AssertionError):
                hub.check_outputs(l, c, b, a, e, k)


if __name__ == "__main__":
    unittest.main()
