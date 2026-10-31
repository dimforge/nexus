"""Bounded native scheduling regressions, independent of solver arithmetic."""

import unittest
import numpy as np
import test_execution_optimizations as execution
import scene_execution as schedule
from ordered import validate_program


class SceneExecution(unittest.TestCase):
    setUpClass = classmethod(execution.ExecutionOptimizations.setUpClass.__func__)
    execute = execution.ExecutionOptimizations.execute
    fixture = execution.ExecutionOptimizations.fixture

    def test_device_grids_bound_live_contacts_colors_and_empty_work(self):
        if self.target != "metal":
            self.skipTest("Native GPU indirect schedule is Metal-only")
        l = {"plan": {"bound": 0, "pfm_base": 1, "pfm_len": 2}}
        c = {
            "contact_capacity": 16385,
            "pair_capacity": 10000,
            "color_stride": 65,
            "batches": 3,
        }
        b = {
            "PairCount": np.array([20000], np.uint32),
            "Plan": np.array([16385, 9999, 6386], np.uint32),
            "Buckets": np.arange(195, dtype=np.uint32) * 19,
            "Dispatches": np.zeros(schedule.grid_words(c), np.uint32),
        }
        c["buffer_words"] = {n: len(v) for n, v in b.items()}
        plans = [
            dict(
                name=k,
                kernel=schedule.make_grids(l, c, k),
                buffer_map={n: n for n in b},
                blocks=1,
                threads=64,
            )
            for k in ("pairs", "contacts", "colors")
        ]
        for empty in (False, True):
            if empty:
                for n in ("PairCount", "Plan", "Buckets"):
                    b[n].fill(0)
            a, _ = self.execute(plans, b)
            g = a["Dispatches"].reshape(-1, 3)
            expected = [1, 1, 1, 1, 0] if empty else [157, 257, 157, 100, 58]
            np.testing.assert_array_equal(g[:5, 0], expected)
            np.testing.assert_array_equal(g[:5, 1:], 1)
            for col in range(1, 64):
                np.testing.assert_array_equal(
                    g[schedule.COLOR_BASE + col], [0 if empty else 1, 1, 1]
                )

    def test_invalid_indirect_offsets_are_rejected(self):
        from test_frame_cpu import variant

        p = variant(8)["manifest"]["program"]
        for descriptor in (
            {"buffer": "Missing", "word": 0},
            {"buffer": "State", "word": 2},
            {"buffer": "State", "word": -1},
            {"buffer": "State", "word": True},
        ):
            p["stages"][0]["indirect"] = descriptor
            with self.subTest(descriptor=descriptor), self.assertRaises(ValueError):
                validate_program(p)
        p["stages"][0]["indirect"] = {"buffer": "State", "word": 1}
        validate_program(p)

    def test_live_color_ranges_match_fresh_rust(self):
        if self.target != "metal":
            self.skipTest("Native GPU indirect schedule is Metal-only")
        l, c, b, e, k = self.fixture("colored", "colored_program", "w65-b2-m2-s22")
        execution.colored.validate_case(l, c, b)
        plans = execution.colored_kernels.make_program(l, c, self.target)
        nb = c["batches"]
        grids = []
        for stage in plans:
            if stage["name"].startswith(
                ("colored_solve", "colored_warmstart", "colored_cached")
            ):
                color = int(b[stage["buffer_map"]["Color"]][0])
                start, end = (
                    b["Buckets"][color * nb - 1],
                    b["Buckets"][(color + 1) * nb - 1],
                )
                groups = min(stage["blocks"], (int(end) - int(start) + 63) // 64)
                stage["indirect"] = {"buffer": "Dispatches", "word": len(grids)}
                grids.extend([groups, 1, 1])
        self.assertTrue(grids)
        b["Dispatches"] = np.array(grids, np.uint32)
        a, _ = self.execute(plans, b, c["iterations"])
        execution.colored.check_outputs(
            l,
            c,
            {n: v for n, v in b.items() if n != "Dispatches"},
            {n: v for n, v in a.items() if n != "Dispatches"},
            e,
            k,
        )

    def test_parallel_color_feedback_matches_rust(self):
        import color_requirement_kernels as kernels
        import color_requirement_replay as replay

        for name in (
            "n0-valid",
            "n65-pending",
            "n65-invalid",
            "n65-inactive",
            "n65-max-rounds",
        ):
            with self.subTest(case=name):
                l, c, b, e, k = self.fixture(
                    "color-requirements", "color_requirements", name
                )
                plans = [
                    dict(
                        name="requirements",
                        kernel=kernels.make_kernel(l, c, self.target),
                        buffer_map={n: n for n in b},
                        blocks=1,
                        threads=64,
                    )
                ]
                a, _ = self.execute(plans, b)
                replay.check_outputs(l, c, b, a, e, k)

    def test_native_sort_pass_policy_matches_rust(self):
        kernels, replay = execution.canonical_kernels, execution.canonical
        for count, composite, expected in (
            (63005, False, [7]),
            (65536, False, [5, 2]),
            (2, True, [4, 1, 2]),
        ):
            self.assertEqual(kernels.scene_passes(count, composite), expected)
        for mode in ("packed", "two-fields"):
            l, c, b, e, k = self.fixture(
                "canonical", "canonical_contacts", f"n65-{mode}-s7"
            )
            rows = b["Contacts"].reshape(-1, l["contact_words"])[: c["count"]]
            offset = l["contact"]["colliders"]
            live = rows[:, l["contact"]["contact"] + l["manifold"]["len"]] != 0
            count = int(rows[live, offset : offset + 2].max()) + 1
            # Packed fixtures carry their native shift. Keep that exact bound.
            if mode == "packed":
                shift = int(b["BatchIds"][l["batch"]["contact_sort_collider_shift"]])
                count = max(count, 2 ** (shift - 1))
                self.assertEqual(count.bit_length(), shift)
            c = dict(
                c,
                collider_count=count,
                packed_pair=mode == "packed",
                groups=kernels.scene_groups(c["count"]),
            )
            a, _ = self.execute(kernels.make_program(l, c, self.target), b)
            spec = replay.expected_outputs(l, c, b)
            replay.check_outputs(l, c, b, a, spec, k)
            for field in ("Output", "InOrder"):
                np.testing.assert_array_equal(a[field], e[field])

    def test_adjacency_grid_stride_covers_bodies_above_native_cap(self):
        import primitive_kernels as kernels

        n = 65537
        b = {
            "Counts": np.full(n, 3, np.uint32),
            "Ids": np.array([9, 1, 1, 8, 2, 0xDEADBEEF], np.uint32),
            "Sorted": np.full(6, 0xDEADBEEF, np.uint32),
        }
        b["Counts"][-1] = 5
        case = {"rows": n, "storage": 6}
        self.assertEqual(kernels.groups(n), 65535)
        plans = [
            dict(
                name="stabilize",
                kernel=kernels.make_kernel(case, self.target),
                buffer_map={k: k for k in b},
                blocks=65535,
                threads=64,
            )
        ]
        a, _ = self.execute(plans, b)
        np.testing.assert_array_equal(a["Sorted"], [1, 1, 9, 2, 8, 0xDEADBEEF])

    def test_state_persistence_touches_only_live_rows(self):
        if self.target != "metal":
            self.skipTest("Native schedule is Metal-only")
        l = {"state_words": 20, "plan": {"bound": 0}}
        c = {
            "contact_capacity": 129,
            "contact_storage": 132,
            "buffer_words": {"States": 5280, "Plan": 1},
        }
        for count in (0, 65, 129):
            b = {
                "States": np.arange(5280, dtype=np.uint32),
                "Plan": np.array([count], np.uint32),
            }
            e = b["States"].copy()
            e[: count * 20] = e[2640 : 2640 + count * 20]
            a, _ = self.execute(
                [
                    dict(
                        name="states",
                        kernel=schedule.copy_live_states(l, c),
                        buffer_map={k: k for k in b},
                        blocks=3,
                        threads=64,
                    )
                ],
                b,
            )
            np.testing.assert_array_equal(a["States"], e)
