"""Coloring contracts: priorities, nondeterministic observations and memory safety."""

import itertools
import json
import os
from pathlib import Path
import random
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from replay import HERE
import coloring_replay as coloring


class ColoringTests(unittest.TestCase):
    def fixture(self, op="topo_step", name="n33-body-zero-d0"):
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
        directory = root / op / name
        before = {
            n: np.fromfile(directory / f"{n}.bin", dtype=np.uint32)
            for n in coloring.BUFFERS
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
        for op in coloring.OPERATIONS:
            contract = json.loads((HERE / f"contracts/{op}.json").read_text())
            for name in contract["required_cases"]:
                with self.subTest(op=op, name=name):
                    l, c, b, e, k = self.fixture(op, name)
                    coloring.check_outputs(l, c, b, e, e, k)

    def test_matching_witness_equals_exhaustive_observations(self):
        rng = random.Random(4271)
        for _ in range(300):
            choices = [
                set(rng.sample(range(5), rng.randint(1, 2)))
                for _ in range(rng.randrange(6))
            ]
            for candidate in range(1, 7):
                wanted = any(
                    candidate not in observed
                    and all(v in observed for v in range(1, candidate))
                    for observed in itertools.product(*choices)
                )
                self.assertEqual(
                    coloring.possible_first_free(candidate, choices),
                    wanted,
                    (candidate, choices),
                )
        self.assertFalse(coloring.possible_first_free(3, [{1, 2}]))
        self.assertTrue(coloring.possible_first_free(3, [{1, 2}, {1, 2}]))

    def test_luby_same_round_priority_regression(self):
        l, c, b, e, k = self.fixture("luby_step", "n4-clique-d0")
        self.assertEqual(list(e["Colors"][:4]), [3, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF])
        bad = {n: a.copy() for n, a in e.items()}
        bad["Colors"][1] = 3
        with self.assertRaises(AssertionError):
            coloring.check_outputs(l, c, b, bad, e, k)
        # Even an identically corrupted oracle must fail the independent priority check.
        with self.assertRaises(AssertionError):
            coloring.check_outputs(l, c, b, bad, bad, k)

    def test_dynamic_body_zero_and_split_hubs_have_correct_neighbors(self):
        l, c, b, e, k = self.fixture()
        _, neighbors, active = coloring.validate_case(l, c, b)
        self.assertGreater(len(neighbors[active[0]]), 20)
        bad = dict(c, static_groups=[0])
        with self.assertRaisesRegex(ValueError, "adjacency"):
            coloring.validate_case(l, bad, b)
        l, c, b, e, k = self.fixture("topo_step", "n65-split-d0")
        _, neighbors, active = coloring.validate_case(l, c, b)
        self.assertTrue(all(not neighbors[i] for i in active))

    def test_guard_uniform_adjacency_and_fixed_output_corruption_fail(self):
        for name, index in (
            ("Colors", -1),
            ("Colored", -1),
            ("Pending", -1),
            ("NumColors", 1),
            ("ColoringIndirect", 3),
            ("BatchIds", 0),
            ("Ids", 0),
        ):
            l, c, b, e, k = self.fixture()
            bad = {n: a.copy() for n, a in e.items()}
            bad[name][index] ^= np.uint32(1)
            with self.subTest(name=name), self.assertRaises(AssertionError):
                coloring.check_outputs(l, c, b, bad, e, k)
        l, c, b, e, k = self.fixture()
        _, _, active = coloring.validate_case(l, c, b)
        fixed = next(i for i in active if b["Colored"][i])
        e["Colors"][fixed] ^= np.uint32(1)
        with self.assertRaises(AssertionError):
            coloring.check_outputs(l, c, b, e, e, k)

    def test_illegal_nondeterministic_pick_fails_even_with_corrupt_oracle(self):
        l, c, b, e, k = self.fixture("topo_step", "n65-split-d0")
        i = next(i for i in range(c["count"]) if b["Colored"][i] == 0)
        e["Colors"][i] = 2  # No neighbors: only first-free color 1 is legal.
        with self.assertRaisesRegex(AssertionError, "first-free"):
            coloring.check_outputs(l, c, b, e, e, k)

    def test_invalid_inputs_are_rejected_before_execution(self):
        for mutation in (
            "palette",
            "pending",
            "counter",
            "adjacency",
            "gap",
            "size",
            "grid",
            "static",
        ):
            op = (
                "topo_fix"
                if mutation == "pending"
                else "luby_step"
                if mutation == "counter"
                else "topo_step"
            )
            l, c, b, e, k = self.fixture(op, "n33-chain-d1")
            if mutation == "palette":
                b["Colors"][1] = 64
            if mutation == "pending":
                b["Pending"][1] = 64
            if mutation == "counter":
                b["NumColors"][0] = 0xFFFFFFFF
            if mutation == "adjacency":
                b["Ids"][0] = 0xFFFFFFFF
            if mutation == "gap":
                b["Colored"][0] = 0
            if mutation == "size":
                b["Colors"] = b["Colors"][:-1]
            if mutation == "grid":
                c["threads"] = 32
            if mutation == "static":
                c["static_groups"] = [0, 0]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                coloring.validate_case(l, c, b)

    def test_exhausted_mask_requires_all_63_colors(self):
        self.assertTrue(coloring.possible_first_free(0, [{c} for c in range(1, 64)]))
        self.assertFalse(coloring.possible_first_free(0, [{c} for c in range(1, 63)]))
        self.assertFalse(coloring.possible_first_free(64, []))

    def test_bounded_output_rejects_retained_conflicts_and_lost_components(self):
        for mode in ("dense", "seeded"):
            l, c, b, e, k = self.fixture("topo_bounded", f"n65-{mode}-d0")
            for corruption in (
                "conflict",
                "discard",
                "false-convergence",
                "stale-color",
            ):
                bad = {n: a.copy() for n, a in e.items()}
                if corruption == "conflict":
                    bad["Colors"][:2] = 1
                    bad["Colored"][:2] = 1
                if corruption == "discard":
                    bad["Colors"][: c["count"]] = 0
                    bad["Colored"][: c["count"]] = 0
                if corruption == "false-convergence":
                    bad["Colors"][0] = 0
                    bad["Colored"][0] = 0
                    bad["NumColors"][0] = 1
                if corruption == "stale-color":
                    bad["Colors"][0] = 1
                    bad["Colored"][0] = 0
                with (
                    self.subTest(mode=mode, corruption=corruption),
                    self.assertRaises(AssertionError),
                ):
                    coloring.check_outputs(l, c, b, bad, bad, k)

    def test_finalize_uses_final_flags_and_preserves_committed_colors(self):
        l, c, b, e, k = self.fixture("topo_finalize", "n33-chain-d0")
        i = next(i for i in range(c["count"]) if b["Colored"][i] == 0)
        bad = {n: a.copy() for n, a in e.items()}
        bad["Colors"][i] = b["Colors"][i]
        with self.assertRaises(AssertionError):
            coloring.check_outputs(l, c, b, bad, bad, k)

    def test_sparse_budget_covers_each_conflict_component(self):
        contract = json.loads((HERE / "contracts/topo_bounded.json").read_text())
        for name in contract["required_cases"]:
            l, c, b, _, _ = self.fixture("topo_bounded", name)
            if not c["expect_complete"]:
                continue
            _, neighbors, active = coloring.validate_case(l, c, b)
            remaining = set(active)
            while remaining:
                stack = [remaining.pop()]
                count = 0
                while stack:
                    i = stack.pop()
                    count += 1
                    for j in neighbors[i]:
                        if j in remaining:
                            remaining.remove(j)
                            stack.append(j)
                self.assertGreater(c["rounds"], count)
