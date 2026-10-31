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
            dict(
                name=op,
                kernel=kernel,
                blocks=blocks,
                threads=64,
                buffer_map={n: n for n in (buffers or b)},
            )
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

        l, _, b, a = self.checked("links", "match_links", "n129", kernels, validation)
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
