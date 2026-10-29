"""Native contact ownership, materials, clipping and axial-symmetry regressions."""

import json, os, sys, unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from replay import HERE
import narrow_replay as narrow


class NarrowPhaseTests(unittest.TestCase):
    def fixture(self, name="n129-g1-mixed"):
        root = (
            Path(os.environ.get("NEXUS_TILELANG_RESULTS", HERE / "results"))
            / "analytic_contacts/c/fixtures"
        )
        l = json.loads((root / "layout.json").read_text())
        cases = json.loads((root / "analytic_contacts/suite.json").read_text())["cases"]
        c = next(c for c in cases if c["name"] == name)
        d = root / "analytic_contacts" / name
        k = json.loads((HERE / "contracts/analytic_contacts.json").read_text())
        b = {n: np.fromfile(d / f"{n}.bin", dtype=np.uint32) for n in narrow.BUFFERS}
        e = {
            n: np.fromfile(d / f"{n}.expected.bin", dtype=np.uint32)
            if n in k["outputs"]
            else a.copy()
            for n, a in b.items()
        }
        return l, c, b, e, k

    def test_all_native_analytic_cases(self):
        k = json.loads((HERE / "contracts/analytic_contacts.json").read_text())
        for name in k["required_cases"]:
            with self.subTest(case=name):
                l, c, b, e, k = self.fixture(name)
                narrow.check_outputs(l, c, b, e, e, k)

    def test_inert_slots_only_change_length(self):
        for name in [
            "n96-g1-far",
            "n96-g1-same-body",
            "n32-g1-shape-5",
            "n96-g1-boxes-far",
        ]:
            l, c, b, e, k = self.fixture(name)
            cw = l["contact_words"]
            li = l["contact"]["contact"] + l["manifold"]["len"]
            a = e["Contacts"].reshape(-1, cw)
            self.assertTrue(np.all(a[: c["count"], li] == 0))
            mask = np.arange(cw) != li
            np.testing.assert_array_equal(
                a[: c["count"], mask], b["Contacts"].reshape(-1, cw)[: c["count"], mask]
            )
            altered = {n: v.copy() for n, v in e.items()}
            altered["Contacts"][0] ^= 1
            with self.assertRaises(AssertionError):
                narrow.check_outputs(l, c, b, altered, e, k)

    def test_box_face_manifolds_have_four_points(self):
        l, c, b, e, k = self.fixture("n96-g1-boxes-coincident")
        li = l["contact"]["contact"] + l["manifold"]["len"]
        self.assertTrue(
            np.all(e["Contacts"].reshape(-1, l["contact_words"])[: c["count"], li] == 4)
        )

    def test_degenerate_capsule_contact_is_not_lost(self):
        for name in ["n96-g1-degenerate-capsule", "n96-g1-zero-radius-capsule"]:
            l, c, b, e, k = self.fixture(name)
            rows = e["Contacts"].reshape(-1, l["contact_words"])
            li = l["contact"]["contact"] + l["manifold"]["len"]
            self.assertTrue(np.all(rows[: c["count"], li] == 1))
            narrow.check_outputs(l, c, b, e, e, k)

    def test_material_rules_and_unknown_rule_priority(self):
        l, c, b, e, k = self.fixture("n36-g1-materials")
        rows = e["Contacts"].reshape(-1, l["contact_words"])
        self.assertTrue(
            np.all(rows[:36, l["contact"]["contact"] + l["manifold"]["len"]] == 1)
        )

        def combine(a, b, rule):
            return (
                abs(min(a, b))
                if rule == 1
                else a * b
                if rule == 2
                else max(a, b)
                if rule == 3
                else min(1, max(0, a + b))
                if rule == 4
                else (a + b) * 0.5
            )

        for i in range(36):
            rule = max(i // 6, i % 6)
            self.assertAlmostEqual(
                float(rows[i].view(np.float32)[l["contact"]["friction"]]),
                combine(-0.25, 0.75, rule),
                places=6,
            )
            self.assertAlmostEqual(
                float(rows[i].view(np.float32)[l["contact"]["restitution"]]),
                combine(0.4, 0.8, rule),
                places=6,
            )

    def test_valid_alternate_axis_contact_is_accepted(self):
        l, c, b, e, k = self.fixture()
        a = {n: v.copy() for n, v in e.items()}
        row = a["Contacts"].reshape(-1, l["contact_words"])[90].view(np.float32)
        m = l["contact"]["contact"]
        p = m + l["manifold"]["points_a"] + l["manifold_point"]["pt"]
        n = m + l["manifold"]["normal_a"]
        for off in [p, n]:
            row[off], row[off + 2] = -row[off + 2], row[off]
        self.assertGreater(
            np.max(
                np.abs(a["Contacts"].view(np.float32) - e["Contacts"].view(np.float32))
            ),
            0.1,
        )
        narrow.check_outputs(l, c, b, a, e, k)

    def test_symmetry_does_not_accept_bad_geometry_or_materials(self):
        for mutation in ["surface", "normal", "distance", "material"]:
            l, c, b, e, k = self.fixture()
            a = {n: v.copy() for n, v in e.items()}
            row = a["Contacts"].reshape(-1, l["contact_words"])[90].view(np.float32)
            m = l["contact"]["contact"]
            p = m + l["manifold"]["points_a"] + l["manifold_point"]["pt"]
            n = m + l["manifold"]["normal_a"]
            if mutation == "surface":
                row[p : p + 3] *= 0.5
            if mutation == "normal":
                row[n : n + 3] *= -1
            if mutation == "distance":
                row[m + l["manifold"]["points_a"] + l["manifold_point"]["dist"]] += 0.1
            if mutation == "material":
                row[l["contact"]["friction"]] += 0.1
            with self.subTest(mutation=mutation), self.assertRaises(AssertionError):
                narrow.check_outputs(l, c, b, a, e, k)

    def test_nonsymmetric_contacts_still_use_strict_float_comparison(self):
        l, c, b, e, k = self.fixture()
        a = {n: v.copy() for n, v in e.items()}
        row = a["Contacts"].reshape(-1, l["contact_words"])[6].view(np.float32)
        p = (
            l["contact"]["contact"]
            + l["manifold"]["points_a"]
            + l["manifold_point"]["pt"]
        )
        row[p : p + 3] *= -1
        with self.assertRaises(AssertionError):
            narrow.check_outputs(l, c, b, a, e, k)

    def test_exact_ids_guards_and_readonly_data(self):
        for name, field in [
            ("Contacts", "bodies"),
            ("Contacts", "colliders"),
            ("Contacts", "subshape"),
            ("Contacts", "guard"),
            ("Pairs", None),
            ("Parents", None),
            ("Materials", None),
            ("Shapes", None),
            ("Poses", None),
            ("RbdParams", None),
            ("Plan", None),
        ]:
            l, c, b, e, k = self.fixture("n96-g1-inside")
            a = {n: v.copy() for n, v in e.items()}
            index = -1 if field == "guard" else l["contact"][field] if field else 0
            a[name][index] ^= 1
            with (
                self.subTest(buffer=name, field=field),
                self.assertRaises(AssertionError),
            ):
                narrow.check_outputs(l, c, b, a, e, k)

    def test_invalid_inputs_are_rejected(self):
        for mutation in [
            "launch",
            "plan",
            "pair",
            "pose",
            "radius",
            "material",
            "prediction",
            "size",
        ]:
            l, c, b, e, k = self.fixture("n32-g1-shape-0")
            if mutation == "launch":
                c["groups"] = 2**26
            if mutation == "plan":
                b["Plan"][l["plan"]["pfm_base"]] += 1
            if mutation == "pair":
                b["Pairs"][l["pair_colliders"]] = c["colliders"]
            if mutation == "pose":
                b["Poses"][l["pose_rotation"] + 3] = np.float32(2).view(np.uint32)
            if mutation == "radius":
                b["Shapes"][l["shape"]["a"]] = np.float32(-1).view(np.uint32)
            if mutation == "material":
                b["Materials"][l["material"]["friction"]] = 0x7FC00000
            if mutation == "prediction":
                b["RbdParams"][l["rbd_params"]["normalized_prediction_distance"]] = (
                    np.float32(-1).view(np.uint32)
                )
            if mutation == "size":
                b["Contacts"] = b["Contacts"][:-1]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                narrow.validate_case(l, c, b)
