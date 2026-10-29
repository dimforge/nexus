"""Contact graph launches and count/scan/scatter composition."""

import tilelang.language as T
from shaders.dynamics.solver import (
    gpu_solver_count_constraints,
    gpu_solver_sort_constraints,
)


def make_kernel(layout, case, target, operation):
    function = (
        gpu_solver_count_constraints(layout)
        if operation == "adjacency_count"
        else gpu_solver_sort_constraints(layout)
    )
    sizes, groups = case["buffer_words"], case["groups"]
    n0, n1, n2, n3, n4, n5 = [
        sizes[n]
        for n in ("Contacts", "Counts", "Ids", "WorldMass", "IsMultibody", "Plan")
    ]

    @T.macro
    def one(Contacts, Counts, Ids, WorldMass, IsMultibody, Plan, i):
        if case["count"]:
            function(Contacts, Counts, Ids, WorldMass, IsMultibody, Plan, i, groups)

    if target == "c":

        @T.prim_func
        def kernel(
            Contacts: T.Tensor((n0,), "uint32"),
            Counts: T.Tensor((n1,), "uint32"),
            Ids: T.Tensor((n2,), "uint32"),
            WorldMass: T.Tensor((n3,), "uint32"),
            IsMultibody: T.Tensor((n4,), "uint32"),
            Plan: T.Tensor((n5,), "uint32"),
        ):
            with T.Kernel(1):
                for i in T.serial(groups * 64):
                    one(Contacts, Counts, Ids, WorldMass, IsMultibody, Plan, i)
    elif target == "ascend":
        import tilelang.ascend.language as A

        @A.prim_func
        def kernel(
            Contacts: A.Tensor((n0,), "uint32"),
            Counts: A.Tensor((n1,), "uint32"),
            Ids: A.Tensor((n2,), "uint32"),
            WorldMass: A.Tensor((n3,), "uint32"),
            IsMultibody: A.Tensor((n4,), "uint32"),
            Plan: A.Tensor((n5,), "uint32"),
        ):
            with A.Kernel(groups) as group:
                with A.SimtVF(threads=64):
                    one(
                        Contacts,
                        Counts,
                        Ids,
                        WorldMass,
                        IsMultibody,
                        Plan,
                        group * 64 + A.get_thread_binding(),
                    )
    else:

        @T.prim_func
        def kernel(
            Contacts: T.Tensor((n0,), "uint32"),
            Counts: T.Tensor((n1,), "uint32"),
            Ids: T.Tensor((n2,), "uint32"),
            WorldMass: T.Tensor((n3,), "uint32"),
            IsMultibody: T.Tensor((n4,), "uint32"),
            Plan: T.Tensor((n5,), "uint32"),
        ):
            with T.Kernel(groups, threads=64) as group:
                one(
                    Contacts,
                    Counts,
                    Ids,
                    WorldMass,
                    IsMultibody,
                    Plan,
                    group * 64 + T.get_thread_binding(),
                )

    return kernel


def make_program(layout, case, target):
    from scan_kernels import make_program as scan_program

    def stage(operation):
        return {
            "name": operation,
            "kernel": make_kernel(layout, case, target, operation),
            "blocks": case["groups"],
            "threads": 64,
            "buffer_map": {
                n: n
                for n in (
                    "Contacts",
                    "Counts",
                    "Ids",
                    "WorldMass",
                    "IsMultibody",
                    "Plan",
                )
            },
        }

    if case["kind"] != "adjacency_build":
        return [stage(case["kind"])]
    scan_case = {
        **case,
        "kind": "prefix_scan",
        "stride": case["bodies"],
        "count": case["bodies"],
        "batches": 1,
        "buffer_words": {
            **case["buffer_words"],
            "Data": case["buffer_words"]["Counts"],
        },
    }
    middle = scan_program(layout, scan_case, target)
    for plan in middle:
        plan["buffer_map"] = {
            n: ("Counts" if v == "Data" else v) for n, v in plan["buffer_map"].items()
        }
    return [stage("adjacency_count"), *middle, stage("adjacency_scatter")]
