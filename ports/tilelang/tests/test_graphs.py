"""Adjacency semantics, inactive manifold filtering and ordered graph construction."""

import json, os
from pathlib import Path
import sys, unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from replay import HERE
import graph_replay as graph


class GraphTests(unittest.TestCase):
    def fixture(self, op="adjacency_build", name="n513-r1026-b3"):
        root = (
            Path(os.environ.get("NEXUS_TILELANG_RESULTS", HERE / "results"))
            / op
            / "c/fixtures"
        )
        layout = json.loads((root / "layout.json").read_text())
        case = next(
            c
            for c in json.loads((root / op / "suite.json").read_text())["cases"]
            if c["name"] == name
        )
        d = root / op / name
        b = {n: np.fromfile(d / f"{n}.bin", dtype=np.uint32) for n in graph.BUFFERS}
        k = json.loads((HERE / f"contracts/{op}.json").read_text())
        e = {
            n: np.fromfile(d / f"{n}.expected.bin", dtype=np.uint32)
            if n in k["outputs"]
            else a.copy()
            for n, a in b.items()
        }
        return layout, case, b, e, k

    def test_all_cases(self):
        for op in graph.OPERATIONS:
            contract = json.loads((HERE / f"contracts/{op}.json").read_text())
            for name in contract["required_cases"]:
                with self.subTest(op=op, name=name):
                    l, c, b, e, k = self.fixture(op, name)
                    graph.check_outputs(l, c, b, e, e, k)

    def test_arbitrary_atomic_order_is_accepted(self):
        for op in ("adjacency_scatter", "adjacency_build"):
            l, c, b, e, k = self.fixture(op)
            a = {n: v.copy() for n, v in e.items()}
            _, rows, _, ends = graph.validate_case(l, c, b)
            start = 0
            for row, end in zip(rows, ends):
                end = int(end)
                a["Ids"][start:end] = a["Ids"][start:end][::-1]
                start = end
            graph.check_outputs(l, c, b, a, e, k)

    def test_membership_counts_scratch_and_guards_are_exact(self):
        for name, index in (
            ("Ids", 0),
            ("Counts", 0),
            ("Aux0", 0),
            ("Ids", -1),
            ("Contacts", 0),
            ("Stride0", 0),
        ):
            l, c, b, e, k = self.fixture()
            a = {n: v.copy() for n, v in e.items()}
            a[name][index] ^= np.uint32(1)
            with (
                self.subTest(name=name, index=index),
                self.assertRaises(AssertionError),
            ):
                graph.check_outputs(l, c, b, a, e, k)

    def test_gaps_allow_stale_body_ids_active_contacts_do_not(self):
        l, c, b, e, k = self.fixture()
        fields = b["Contacts"].reshape(-1, l["contact_words"])
        length = l["contact"]["contact"] + l["manifold"]["len"]
        body = l["contact"]["bodies"]
        self.assertEqual(fields[0, length], 0)
        self.assertEqual(fields[0, body], 0xFFFFFFFF)
        graph.check_outputs(l, c, b, e, e, k)
        fields[0, length] = 1
        with self.assertRaisesRegex(ValueError, "active contact body"):
            graph.validate_case(l, c, b)

    def test_reset_prefix_and_plan_are_checked(self):
        for op in graph.OPERATIONS:
            l, c, b, _, _ = self.fixture(op)
            b["Counts"][0] += 1
            with self.assertRaisesRegex(ValueError, "Counts"):
                graph.validate_case(l, c, b)
        l, c, b, _, _ = self.fixture()
        b["Plan"][l["plan"]["bound"]] += 1
        with self.assertRaisesRegex(ValueError, "Plan"):
            graph.validate_case(l, c, b)

    def test_static_and_multibody_rows_are_filtered_and_y_only_mass_is_dynamic(self):
        l, c, b, _, _ = self.fixture()
        _, rows, _, _ = graph.validate_case(l, c, b)
        mass = b["WorldMass"].view(np.float32).reshape(-1, l["world_mass_words"])
        o = l["world_mass"]["inv_mass"]
        active_y = []
        for body, row in enumerate(rows):
            if b["IsMultibody"][body] or np.all(mass[body, o : o + 3] == 0):
                self.assertFalse(row)
            if mass[body, o] == 0 and mass[body, o + 1] != 0 and row:
                active_y.append(body)
        self.assertTrue(active_y)
