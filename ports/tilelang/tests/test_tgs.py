"""Composed frame ABI, ownership, and independent physical invariants."""

import json
import os
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from replay import HERE
import tgs_replay as tgs
import body_uniform_replay as uniforms


class TgsTests(unittest.TestCase):
    def fixture(self, name="stack", op="tgs_frame"):
        root = (
            Path(os.environ.get("NEXUS_TILELANG_RESULTS", HERE / "results"))
            / op
            / "c/fixtures"
        )
        l = json.loads((root / "layout.json").read_text())
        c = next(
            c
            for c in json.loads((root / op / "suite.json").read_text())["cases"]
            if c["name"] == name
        )
        d = root / op / name
        k = json.loads((HERE / f"contracts/{op}.json").read_text())
        b = {n: np.fromfile(d / f"{n}.bin", dtype=np.uint32) for n in k["inputs"]}
        e = {
            n: np.fromfile(d / f"{n}.expected.bin", dtype=np.uint32)
            if n in k["outputs"]
            else a.copy()
            for n, a in b.items()
        }
        return l, c, b, e, k

    def test_all_frames_and_body_coefficients(self):
        for op, checker in [
            ("tgs_frame", tgs.check_outputs),
            ("body_coefficients", uniforms.check_outputs),
        ]:
            contract = json.loads((HERE / f"contracts/{op}.json").read_text())
            for name in contract["required_cases"]:
                with self.subTest(operation=op, case=name):
                    l, c, b, e, k = self.fixture(name, op)
                    checker(l, c, b, e, e, k)

    def test_frame_rejects_shared_velocity_writers_including_fixed_bodies(self):
        l, c, b, e, k = self.fixture("wide31")
        links = b["Links"].reshape(-1, l["link_words"])
        f = l["link"]
        links[1, f["vel_slot_b"]] = links[0, f["vel_slot_b"]]
        links[1, f["solver_body_b"]] = links[0, f["solver_body_b"]]
        source = int(links[1, f["contact"]])
        b["Contacts"][source * l["contact_words"] + l["contact"]["bodies"] + 1] = links[
            1, f["solver_body_b"]
        ]
        with self.assertRaisesRegex(ValueError, "overlap"):
            tgs.validate_case(l, c, b)

    def test_schedule_uniforms_and_native_ranges_are_checked(self):
        for bad in (
            "substeps",
            "pgs",
            "mode",
            "capacity",
            "buckets",
            "color",
            "slot",
            "scale",
            "recycle",
            "shape",
        ):
            l, c, b, e, k = self.fixture()
            f = l["rbd_params"]
            if bad == "substeps":
                b["RbdParams"][f["num_solver_iterations"]] += 1
            if bad == "pgs":
                b["RbdParams"][f["num_internal_pgs_iterations"]] += 1
            if bad == "mode":
                b["Mode"][0] = 0
            if bad == "capacity":
                b["BatchIds"][l["batch"]["colliders_batch_capacity"]] += 1
            if bad == "buckets":
                b["Buckets"][-1] += 1
            if bad == "color":
                b["Color1"][0] = 2
            if bad == "slot":
                b["Links"][l["link"]["vel_slot_a"]] = c["body_storage"]
            if bad == "scale":
                b["RbdParams"][f["warmstart_coefficient"]] = np.float32(0.7).view(
                    np.uint32
                )
            if bad == "recycle":
                b["Links"][l["link"]["recycled"]] = 1
            if bad == "shape":
                b["Poses"] = b["Poses"][:-1]
            with self.subTest(mutation=bad), self.assertRaises(ValueError):
                tgs.validate_case(l, c, b)

    def test_guards_inactive_tiles_and_readonly_inputs_are_exact(self):
        for name, index in [
            ("Poses", -1),
            ("SolverVelocities", -1),
            ("Tiles", -1),
            ("LocalMass", 0),
            ("Gravity", 0),
            ("Color1", 1),
            ("Contacts", 0),
            ("Plan", 0),
            ("Plan", -1),
        ]:
            l, c, b, e, k = self.fixture()
            a = {n: v.copy() for n, v in e.items()}
            a[name][index] ^= 1
            with self.subTest(buffer=name), self.assertRaises(AssertionError):
                tgs.check_outputs(l, c, b, a, e, k)

    def test_fixed_body_invariants_do_not_trust_oracle_outputs(self):
        for field in ("velocity", "pose"):
            l, c, b, e, k = self.fixture()
            if field == "velocity":
                e["Velocities"][l["velocity_linear"]] = np.float32(0.01).view(np.uint32)
            else:
                e["Poses"][l["pose_translation"]] = np.float32(0.01).view(np.uint32)
            with self.subTest(field=field), self.assertRaises(AssertionError):
                tgs.check_outputs(l, c, b, e, e, k)

    def test_ground_fixture_is_explicitly_static(self):
        l, c, b, e, k = self.fixture()
        m = b["LocalMass"].reshape(-1, l["local_words"])
        self.assertEqual(
            list(m[0, l["local_inv_mass"] : l["local_inv_mass"] + 3]), [0, 0, 0]
        )
        self.assertEqual(
            list(m[0, l["local_inv_inertia"] : l["local_inv_inertia"] + 3]), [0, 0, 0]
        )
        self.assertTrue(
            np.all(m[1, l["local_inv_mass"] : l["local_inv_mass"] + 3] != 0)
        )

    def test_free_motion_matches_semi_implicit_gravity(self):
        l, c, b, e, k = self.fixture("free")
        f = l["rbd_params"]
        dt = float(b["RbdParams"].view(np.float32)[f["dt"]])
        steps = c["substeps"]
        g = b["Gravity"].view(np.float32)[:3].astype(np.float64)
        vo = l["velocity_words"] + l["velocity_linear"]
        po = l["pose_words"] + l["pose_translation"]
        v0 = b["Velocities"].view(np.float32)[vo : vo + 3].astype(np.float64)
        p0 = b["Poses"].view(np.float32)[po : po + 3].astype(np.float64)
        v = e["Velocities"].view(np.float32)[vo : vo + 3]
        p = e["Poses"].view(np.float32)[po : po + 3]
        np.testing.assert_allclose(v, v0 + g * dt * steps, atol=2e-6, rtol=2e-5)
        np.testing.assert_allclose(
            p,
            p0 + v0 * dt * steps + g * dt * dt * (steps * (steps + 1) / 2),
            atol=2e-6,
            rtol=2e-5,
        )

    def test_friction_removes_tangential_speed_in_frame_checkpoint(self):
        l, c, b, e, k = self.fixture("friction")
        i = l["velocity_words"] + l["velocity_linear"]
        before = float(b["Velocities"].view(np.float32)[i])
        after = float(e["Velocities"].view(np.float32)[i])
        self.assertGreater(after, 0)
        self.assertLess(after, before)

    def test_native_body_caps_cover_zero_dt_and_sentinel_neighbors(self):
        l, c, b, e, k = self.fixture("p1", "body_coefficients")
        self.assertEqual(e["Params"][2], 0)
        l, c, b, e, k = self.fixture("p4", "body_coefficients")
        self.assertEqual(e["Params"][1], l["constants"]["max_flt_bits"])
        self.assertNotEqual(e["Params"][1], 0x7F7FFFFF)
        for name in ("p8", "p9"):
            l, c, b, e, k = self.fixture(name, "body_coefficients")
            self.assertGreater(float(e["Params"].view(np.float32)[1]), 1.5e18)
        l, c, b, e, k = self.fixture("p2", "body_coefficients")
        angular0 = e["Params"][2]
        l, c, b, e, k = self.fixture("p3", "body_coefficients")
        self.assertEqual(e["Params"][2], angular0)

    def test_native_body_parameter_guards_and_copies_are_exact(self):
        for name, index in [
            ("Params", 0),
            ("Params", 5),
            ("Params", 8),
            ("Gravity", 3),
            ("RbdParams", 0),
        ]:
            l, c, b, e, k = self.fixture("p0", "body_coefficients")
            a = {n: v.copy() for n, v in e.items()}
            a[name][index] ^= 1
            with (
                self.subTest(buffer=name, index=index),
                self.assertRaises(AssertionError),
            ):
                uniforms.check_outputs(l, c, b, a, e, k)
