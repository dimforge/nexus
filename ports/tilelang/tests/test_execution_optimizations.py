"""Focused optimized schedules, with fresh noalias-disabled Rust outputs."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import canonical_kernels
import canonical_replay as canonical
import colored_kernels
import colored_replay as colored
from cpu_oracle import export_fixtures
from ordered import CompilationSession, execute_c_program, lower_program
from replay import HERE, build_native, run


class ExecutionOptimizations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="nexus-execution-")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        cls.target = os.environ.get("NEXUS_TILELANG_TEST_TARGET", "c")
        cls.session = CompilationSession()
        cls.runner = build_native() if cls.target == "metal" else None
        cls.exports = {}

    def fixture(self, oracle, operation, name):
        if oracle not in self.exports:
            directory = self.root / oracle
            export_fixtures(oracle, directory)
            self.exports[oracle] = directory
        root = self.exports[oracle]
        layout = json.loads((root / "layout.json").read_text())
        suite = json.loads((root / operation / "suite.json").read_text())
        case = next(c for c in suite["cases"] if c["name"] == name)
        contract = json.loads((HERE / f"contracts/{operation}.json").read_text())
        directory = root / operation / name
        before = {n: np.fromfile(directory / f"{n}.bin", np.uint32)
                  for n in contract["inputs"]}
        expected = {n: np.fromfile(directory / f"{n}.expected.bin", np.uint32)
                    if n in contract["outputs"] else a.copy()
                    for n, a in before.items()}
        return layout, case, before, expected, contract

    def execute(self, plans, before, iterations=1):
        directory = Path(tempfile.mkdtemp(prefix="execution-", dir=self.root))
        sources, program = lower_program(plans, before, self.target,
                                         iterations=iterations, session=self.session)
        actual = {n: a.copy() for n, a in before.items()}
        if self.target == "c":
            execute_c_program(sources, program, directory, actual, session=self.session)
        else:
            for n, a in before.items():
                a.tofile(directory / f"{n}.bin")
            for n, source in sources.items():
                (directory / n).write_text(source)
            (directory / "bindings.json").write_text(json.dumps(program))
            run([self.runner, directory])
            actual = {n: np.fromfile(directory / f"{n}.actual.bin", np.uint32)
                      for n in before}
        return actual, sources

    def test_bounded_sort_matches_rust_order_and_exact_workspace(self):
        for name in ("n65-duplicates-s7", "n65-two-fields-s7"):
            with self.subTest(case=name):
                l, c, b, e, k = self.fixture("canonical", "canonical_contacts", name)
                canonical.check_outputs(l, c, b, e, e, k)
                rows = b["Contacts"].reshape(-1, l["contact_words"])[:c["count"]]
                live = rows[:, l["contact"]["contact"] + l["manifold"]["len"]] != 0
                offset = l["contact"]["colliders"]
                bound = int(rows[live, offset:offset + 2].max()) + 1
                optimized = dict(c, collider_count=bound,
                                 groups=canonical_kernels.scene_groups(c["count"]))
                self.assertEqual(optimized["groups"], 2)
                plans = canonical_kernels.make_program(l, optimized, self.target)
                self.assertLess(len(plans), len(canonical_kernels.make_program(l, c, self.target)))
                actual, _ = self.execute(plans, b)
                spec = canonical.expected_outputs(l, optimized, b)
                canonical.check_outputs(l, optimized, b, actual, spec, k)
                for field in ("Output", "InOrder"):
                    np.testing.assert_array_equal(actual[field], e[field], err_msg=field)
                with self.assertRaisesRegex(ValueError, "declared bound"):
                    canonical.validate_case(l, dict(c, collider_count=bound - 1), b)

    def test_canonical_scene_launches_cover_capacity(self):
        for capacity, groups in ((0, 1), (1, 1), (64, 1), (65, 2),
                                 (600000, 9375)):
            self.assertEqual(canonical_kernels.scene_groups(capacity), groups)

    def test_sort_digit_boundaries_reserve_inert_sentinel(self):
        for count, digits in ((0, 2), (255, 2), (256, 4), (65535, 4),
                              (65536, 6), (2**24, 8), (2**32 - 1, 8)):
            for selector in (1, 2, 5, 6):
                self.assertEqual(canonical_kernels.radix_digits(selector, count), digits)
                self.assertGreaterEqual(2**(digits * 4) - 1, count)
            for selector in (0, 3, 4, 7):
                self.assertEqual(canonical_kernels.radix_digits(selector, count), 8)
        self.assertEqual(canonical_kernels.radix_digits(1), 8)
        for invalid in (-1, 2**32, True, 3.5):
            with self.assertRaises(ValueError):
                canonical_kernels.radix_digits(1, invalid)

    def test_fused_colors_match_fresh_rust_and_separate_metal(self):
        if self.target != "metal":
            self.skipTest("Fused device barriers are Metal-specific")
        for oracle, op, name in (
            ("colored", "colored_program", "w33-b3-m1-s21"),
            ("colored", "colored_program", "w65-b2-m2-s22"),
            ("colored", "colored_program", "w32-b2-m0-s20"),
            ("colored", "colored_program", "w129-b1-m2-s42-shared-static"),
            ("twist-colored", "twist_colored_program", "w33-b3-m1-s21"),
            ("twist-colored", "twist_colored_program", "w65-b1-m2-s42-shared-static"),
        ):
            with self.subTest(operation=op, case=name):
                l, c, b, e, k = self.fixture(oracle, op, name)
                colored.validate_case(l, c, b)
                original = colored_kernels.make_program(l, c, self.target)
                optimized = colored_kernels.make_program(l, dict(c, fused_colors=True), self.target)
                self.assertEqual(len(original), 15)
                self.assertEqual(len(optimized), 7)
                a, sources = self.execute(optimized, b, c["iterations"])
                self.assertTrue(any("mem_flags::mem_device" in s for s in sources.values()))
                colored.check_outputs(l, c, b, a, e, k)
                separate, _ = self.execute(original, b, c["iterations"])
                for field in b:
                    np.testing.assert_array_equal(a[field], separate[field], err_msg=field)
                if name == "w33-b3-m1-s21" and oracle == "colored":
                    # Insert an empty color before the same three occupied colors.
                    # Their ascending arithmetic and fresh Rust state stay unchanged.
                    empty = {n: a.copy() for n, a in b.items()}
                    empty["Buckets"] = np.concatenate((np.zeros(c["batches"], np.uint32), b["Buckets"]))
                    for color in range(1, 4):
                        empty[f"Color{color}"][0] += 1
                    empty["BatchIds"][l["batch"]["solver_color_buckets_stride"]] += 1
                    ec = dict(c, fused_colors=True,
                              buffer_words={n: len(a) for n, a in empty.items()})
                    ea, _ = self.execute(colored_kernels.make_program(l, ec, self.target), empty, c["iterations"])
                    for field in b:
                        want = empty[field] if field in ("Buckets", "BatchIds", "Color1", "Color2", "Color3") else separate[field]
                        np.testing.assert_array_equal(ea[field], want, err_msg=field)

    def test_fusion_policy_retains_wide_dispatches_for_large_capacity(self):
        for batches in (1, 3, 64):
            case = {"batches": batches, "contact_capacity": 512*batches}
            self.assertTrue(colored_kernels.fuse_scene_colors(case, "metal"))
            for target in ("c", "ascend", "cuda"):
                self.assertFalse(colored_kernels.fuse_scene_colors(case, target))
            case["contact_capacity"] += 1
            self.assertFalse(colored_kernels.fuse_scene_colors(case, "metal"))

    def test_lbvh_scene_launches_scale_like_native(self):
        from lbvh_program_kernels import scene_groups
        # Balls: spread independent leaves over 985 groups, escape nodes over
        # 1969 groups; domain scratch remains bounded by 64 groups per batch.
        for kind, want in (("lbvh_pairs", 985), ("lbvh_morton", 985),
                           ("lbvh_build", 985), ("lbvh_leaves", 985),
                           ("lbvh_escapes", 1969), ("lbvh_domain", 16)):
            self.assertEqual(scene_groups(kind, 63005), want)
        for n in (0, 1, 63, 64, 65, 4096, 4097, 262144, 1000000):
            self.assertEqual(scene_groups("lbvh_pairs", n), max(1, (n + 63)//64))
            self.assertEqual(scene_groups("lbvh_domain", n),
                             min(64, max(1, (n + 4095)//4096)))

    def test_lbvh_early_exit_preserves_pairs_filters_overflow_and_batches(self):
        import lbvh_kernels as kernels
        import lbvh_validation as validation
        for name in ("n0-b1-g1-mixed", "n1-b2-g1-collapsed",
                     "n65-b2-g2-filters", "n33-b2-g1-overflow",
                     "n257-b2-g3-mixed"):
            with self.subTest(case=name):
                l, c, b, e, k = self.fixture("lbvh", "lbvh_pairs", name)
                blocks, threads, _ = kernels.dimensions(c)
                plans = [{"name": "pairs", "kernel": kernels.make_kernel(l, c, self.target),
                              "buffer_map": {n: n for n in b}, "blocks": blocks, "threads": threads}]
                a, sources = self.execute(plans, b)
                validation.check_outputs(l, c, b, a, e, k)
                if c["colliders"] > 1:
                    self.assertTrue(any("while (" in source for source in sources.values()))

    def test_automatic_lbvh_program_matches_fresh_rust(self):
        import lbvh_program_kernels as kernels
        import lbvh_program_validation as validation
        # Rust's fixture uses one grid-striding workgroup. The scene schedule
        # uses 3 leaf groups and 5 escape groups, including partial groups.
        # Domain geometry agrees here, so every deterministic scratch word can
        # also be compared directly with Rust, alongside legal unordered pairs.
        l, c, b, e, k = self.fixture("lbvh", "lbvh_program", "n129-b1-g1-duplicates")
        plans = kernels.make_program(l, dict(c, groups=None), self.target)
        by_name = {p["name"]: p for p in plans}
        self.assertEqual(by_name["lbvh_pairs"]["blocks"], 3)
        self.assertEqual(by_name["lbvh_escapes"]["blocks"], 5)
        a, _ = self.execute(plans, b)
        validation.check_outputs(l, c, b, a, e, k)

    def test_scene_domain_validation_checks_both_documented_partitions(self):
        from lbvh_scene_validation import checked_domain_groups
        l, _, _, _, _ = self.fixture("lbvh", "lbvh_program", "n129-b1-g1-duplicates")
        n, nb = 4097, 2
        c = {"colliders": n, "batches": nb}
        poses = np.zeros((n * nb, l["pose_words"]), np.float32)
        xyz = l["pose_translation"]
        poses[:, xyz:xyz+3] = np.arange(n * nb, dtype=np.float32)[:, None]
        for groups in (1, 2):
            rows = np.zeros((nb * 64, l["aabb_words"]), np.float32)
            for batch in range(nb):
                for group in range(64):
                    ids = np.arange(n)[(np.arange(n)//128) % groups == group]
                    points = poses[ids * nb + batch, xyz:xyz+3]
                    for field, reduce, empty in (("mins", np.min, 1e18),
                                                  ("maxs", np.max, -1e18)):
                        off = l["aabb"][field]
                        rows[batch*64+group, off:off+3] = reduce(points, axis=0) if ids.size else empty
            arrays = {"Poses": poses.reshape(-1).view(np.uint32),
                      "Partials": rows.reshape(-1).view(np.uint32)}
            self.assertEqual(checked_domain_groups(l, c, arrays, arrays), groups)
            arrays["Partials"][l["aabb"]["mins"]] ^= 1
            with self.assertRaisesRegex(AssertionError, "neither documented"):
                checked_domain_groups(l, c, arrays, arrays)
