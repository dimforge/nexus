"""Focused execution and control-flow regressions for the scaling audit.

Run explicitly when these paths change; this geometry subset is not appended to
routine smoke, whose total wall-clock budget remains fixed at 290 seconds.
"""

import ast
import unittest

import test_execution_optimizations as execution


class TraversalExits(unittest.TestCase):
    setUpClass = classmethod(execution.ExecutionOptimizations.setUpClass.__func__)
    fixture = execution.ExecutionOptimizations.fixture
    execute = execution.ExecutionOptimizations.execute

    def checked(self, oracle, op, name, kernels, validation, buffers=None):
        l, c, b, e, k = self.fixture(oracle, op, name)
        validation.validate_case(l, c, b)
        if oracle == "deferred":
            kernel = kernels.make_kernel(l, c, self.target, op)
        else:
            kernel = kernels.make_kernel(l, c, self.target)
        blocks = (
            max(1, (c.get("launch_capacity", c["count"]) + 63) // 64)
            if oracle == "links"
            else c["groups"]
        )
        plans = [
            {
                "name": op,
                "kernel": kernel,
                "blocks": blocks,
                "threads": 64,
                "buffer_map": {n: n for n in (buffers or b)},
            }
        ]
        a, sources = self.execute(plans, b, c.get("iterations", 1))
        validation.check_outputs(l, c, b, a, e, k)
        self.assertTrue(any("while (" in s for s in sources.values()))
        return l, c, b, a

    def test_composite_pairs_and_reduction(self):
        import deferred_kernels as kernels
        import deferred_validation as validation

        for op, name in (
            ("deferred_pairs", "n567-c1200-g3-normal"),
            ("deferred_pairs", "n567-c17-g2-overflow"),
            ("reduce_contacts", "n567-c1200-g3-normal"),
            ("reduce_contacts", "n567-c1200-g2-merge-all"),
        ):
            with self.subTest(operation=op, case=name):
                self.checked("deferred", op, name, kernels, validation, kernels.BUFFERS)

    def test_mesh_projection(self):
        import point_query_kernels as kernels
        import point_query_replay as validation

        self.checked("point-queries", "mesh_projection", "n65", kernels, validation)

    def test_first_warmstart_match(self):
        import link_kernels as kernels
        import link_replay as validation

        l, _, _b, a = self.checked("links", "match_links", "n129", kernels, validation)
        rows = a["Links"].reshape(-1, l["link_words"])
        # Duplicate identities must choose the FIRST adjacency entry, not the
        # last matching identity. Misses retain the all-ones sentinel.
        for row, expected in ((0, 1), (2, 11), (6, 27), (1, 2**32 - 1)):
            self.assertEqual(rows[row, l["link"]["previous_constraint"]], expected)

    def test_coloring_conflicts_and_luby_priority(self):
        import coloring_kernels as kernels
        import coloring_replay as validation

        for op, name in (
            ("luby_step", "n4-clique-d0"),
            ("luby_step", "n33-body-zero-d0"),
            ("topo_fix", "n33-body-zero-d0"),
            ("topo_fix", "n33-body-zero-d1"),
            ("topo_fix", "n65-split-d1"),
        ):
            with self.subTest(operation=op, case=name):
                self.checked("colorings", op, name, kernels, validation)

    def test_data_dependent_exits_survive_translation(self):
        # Output parity alone cannot catch executing the inactive tail. Require
        # loop termination predicates, in addition to the execution checks above.
        checks = {
            "queries/gjk/gjk.py": {
                "closest_points": ("iteration[0] < 100", "active[0]")
            },
            "queries/gjk/epa3.py": {
                "closest_points": ("iteration[0] < 100", "done[0] == 0")
            },
            "broad_phase/lbvh.py": {
                "gpu_lbvh_build": ("searching[0]", "t[0] >= 1"),
                "gpu_lbvh_escapes": ("escape_steps[0] < 64", "active[0]"),
            },
            "broad_phase/narrow_phase.py": {
                "composite_convex": (
                    "visited[0] < node_count",
                    "current[0] < node_count",
                ),
                "gpu_reduce_contacts": ("active[0]", "not merged[0]"),
            },
            "shapes/trimesh.py": {
                "project_local_point": ("visited[0] < nodes", "current[0] < nodes"),
            },
            "dynamics/contact_links.py": {
                "find_previous_constraint": ("previous[0] == -1", "entry[0] < last"),
            },
            "dynamics/coloring.py": {
                "gpu_step_graph_coloring_luby": ("greatest[0] != 0",),
                "gpu_fix_conflicts_topo_gc": ("not conflict[0]",),
            },
        }
        for path, symbols in checks.items():
            tree = ast.parse((execution.HERE / "shaders" / path).read_text())
            for symbol, predicates in symbols.items():
                fn = next(
                    n
                    for n in tree.body
                    if isinstance(n, ast.FunctionDef) and n.name == symbol
                )
                conditions = "\n".join(
                    ast.unparse(n.test)
                    for n in ast.walk(fn)
                    if isinstance(n, ast.While)
                )
                for predicate in predicates:
                    with self.subTest(symbol=symbol, predicate=predicate):
                        self.assertIn(predicate, conditions)

    def test_bounded_gjk_epa_exits(self):
        import epa_kernels
        import epa_validation
        import support_kernels
        import support_validation

        for op, kernels, validation in (
            ("gjk_closest", support_kernels, support_validation),
            ("epa_closest", epa_kernels, epa_validation),
            ("gjk_epa", epa_kernels, epa_validation),
        ):
            with self.subTest(operation=op):
                self.checked("geometry", op, "n65", kernels, validation)

    def test_bounded_lbvh_build_escape_exits(self):
        import lbvh_kernels as kernels
        import lbvh_validation as validation

        for op in ("lbvh_build", "lbvh_escapes"):
            for name in ("n129-b1-g1-duplicates", "n257-b2-g3-mixed"):
                l, c, b, e, k = self.fixture("lbvh", op, name)
                blocks, threads, _ = kernels.dimensions(c)
                plans = [
                    {
                        "name": op,
                        "kernel": kernels.make_kernel(l, c, self.target),
                        "blocks": blocks,
                        "threads": threads,
                        "buffer_map": {n: n for n in kernels.BUFFERS},
                    }
                ]
                a, _ = self.execute(plans, b)
                validation.check_outputs(l, c, b, a, e, k)

    def test_cooperative_refit_uniform_completion(self):
        import lbvh_kernels as kernels
        import lbvh_validation as validation

        for op in ("lbvh_refit_chunks", "lbvh_refit_frontier"):
            for name in (
                "n1-b2-g1-collapsed",
                "n129-b1-g1-duplicates",
                "n257-b2-g3-mixed",
            ):
                with self.subTest(operation=op, case=name):
                    l, c, b, e, k = self.fixture("lbvh", op, name)
                    blocks, threads, _ = kernels.dimensions(c)
                    plans = [
                        {
                            "name": op,
                            "kernel": kernels.make_kernel(l, c, self.target),
                            "blocks": blocks,
                            "threads": threads,
                            "buffer_map": {n: n for n in kernels.BUFFERS},
                        }
                    ]
                    a, sources = self.execute(plans, b)
                    validation.check_outputs(l, c, b, a, e, k)
                    if self.target == "metal" and c["colliders"] > 1:
                        self.assertTrue(
                            any("while (" in source for source in sources.values())
                        )
