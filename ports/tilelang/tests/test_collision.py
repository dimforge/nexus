"""Connected collision stages preserve pair ownership across atomic ordering."""

import json, os, sys, unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from replay import HERE
import collision_replay as collision


class CollisionPipelineTests(unittest.TestCase):
    def fixture(self, name="n33-b2-boxes"):
        root = (
            Path(os.environ.get("NEXUS_TILELANG_RESULTS", HERE / "results"))
            / "analytic_collision/c/fixtures"
        )
        l = json.loads((root / "layout.json").read_text())
        cases = json.loads((root / "analytic_collision/suite.json").read_text())[
            "cases"
        ]
        c = next(x for x in cases if x["name"] == name)
        d = root / "analytic_collision" / name
        b = {n: np.fromfile(d / f"{n}.bin", dtype=np.uint32) for n in collision.BUFFERS}
        e = {n: np.fromfile(d / f"{n}.expected.bin", dtype=np.uint32) for n in b}
        k = json.loads((HERE / "contracts/analytic_collision.json").read_text())
        return l, c, b, e, k

    def test_all_native_collision_cases(self):
        k = json.loads((HERE / "contracts/analytic_collision.json").read_text())
        for name in k["required_cases"]:
            with self.subTest(case=name):
                l, c, b, e, k = self.fixture(name)
                collision.check_outputs(l, c, b, e, e, k)

    def test_atomic_pair_permutation_preserves_manifold_ownership(self):
        l, c, b, e, k = self.fixture()
        a = {n: v.copy() for n, v in e.items()}
        count = int(e["PairCount"][0])
        self.assertGreater(count, 2)
        for name, words in [
            ("Pairs", l["pair_words"]),
            ("Contacts", l["contact_words"]),
        ]:
            rows = a[name].reshape(-1, words)
            rows[:count] = rows[:count][::-1].copy()
        collision.check_outputs(l, c, b, a, e, k)
        a["Contacts"][: l["contact_words"]] = e["Contacts"][: l["contact_words"]]
        with self.assertRaises(AssertionError):
            collision.check_outputs(l, c, b, a, e, k)

    def test_materials_parent_ids_counts_and_guards_stay_strict(self):
        for name, word in [
            ("Contacts", "bodies"),
            ("Contacts", "colliders"),
            ("Contacts", "friction"),
            ("Contacts", "guard"),
            ("Plan", None),
            ("PfmCount", None),
            ("ContactsIndirect", None),
            ("PfmIndirect", None),
            ("MbIndirect", None),
            ("PfmSortLen", None),
            ("Materials", None),
            ("Parents", None),
            ("PairCount", None),
        ]:
            l, c, b, e, k = self.fixture()
            a = {n: v.copy() for n, v in e.items()}
            offset = -1 if word == "guard" else l["contact"][word] if word else 0
            if word == "friction":
                a[name].view(np.float32)[offset] += 0.1
            else:
                a[name][offset] ^= 1
            with (
                self.subTest(buffer=name, word=word),
                self.assertRaises(AssertionError),
            ):
                collision.check_outputs(l, c, b, a, e, k)

    def test_box_roundoff_policy_is_geometric_and_scoped(self):
        l, c, b, e, k = self.fixture("n33-b2-rotated")
        rows = e["Contacts"].reshape(-1, l["contact_words"])
        cf, mf, pf = l["contact"], l["manifold"], l["manifold_point"]
        start = cf["contact"] + mf["points_a"]
        length_word = cf["contact"] + mf["len"]
        live = [i for i in range(int(e["PairCount"][0])) if rows[i, length_word]]
        # A shallow contact makes near-zero absolute error visible; depth is not
        # an appropriate scale for the geometry that produced that depth.
        i = min(live, key=lambda i: abs(rows[i].view(np.float32)[start + pf["dist"]]))
        word = i * l["contact_words"] + start + pf["dist"]
        a = {n: v.copy() for n, v in e.items()}
        a["Contacts"].view(np.float32)[word] += 2.5e-6
        collision.check_outputs(l, c, b, a, e, k)
        base = dict(k)
        del base["box_geometry_roundoff"]
        with self.assertRaises(AssertionError):
            collision.check_outputs(l, c, b, a, e, base)
        a["Contacts"].view(np.float32)[word] += 1e-4
        with self.assertRaises(AssertionError):
            collision.check_outputs(l, c, b, a, e, k)
        # Agreeing implementations are insufficient if both have bad geometry.
        with self.assertRaisesRegex(AssertionError, "off its surface"):
            collision.check_outputs(l, c, b, a, a, k)
        for field in [cf["friction"], cf["contact"] + mf["normal_a"]]:
            a = {n: v.copy() for n, v in e.items()}
            a["Contacts"].view(np.float32)[i * l["contact_words"] + field] += 1e-4
            with self.subTest(field=field), self.assertRaises(AssertionError):
                collision.check_outputs(l, c, b, a, e, k)

    def test_empty_frames_do_not_leave_old_contacts_live(self):
        for name in ["n0-b1-mixed", "n1-b1-mixed", "n33-b2-separated"]:
            l, c, b, e, k = self.fixture(name)
            self.assertEqual(e["PairCount"][0], 0)
            self.assertEqual(e["Plan"][l["plan"]["bound"]], 0)
            np.testing.assert_array_equal(e["Contacts"], b["Contacts"])
            collision.check_outputs(l, c, b, e, e, k)

    def test_invalid_scene_contracts_are_rejected(self):
        for mutation in [
            "shape",
            "parent",
            "overflow",
            "launch",
            "material",
            "size",
            "dtype",
        ]:
            l, c, b, e, k = self.fixture()
            if mutation == "shape":
                b["Shapes"][l["shape"]["a"] + 3] = l["shape_types"]["capsule"]
            if mutation == "parent":
                b["Parents"][0] ^= 1
            if mutation == "overflow":
                c["pair_capacity"] = 1
            if mutation == "launch":
                c["contact_groups"] = 2**26
            if mutation == "material":
                b["Materials"][l["material"]["friction"]] = 0x7FC00000
            if mutation == "size":
                b["Contacts"] = b["Contacts"][:-1]
            if mutation == "dtype":
                b["Contacts"] = b["Contacts"].astype(np.int32)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                collision.validate_case(l, c, b)
