"""Launch adapters for the Rust-mirrored graph-coloring entry points."""

import tilelang.language as T
from shaders.dynamics import coloring as C


def make_kernel(layout, case, target):
    op = case["kind"]
    groups = case["groups"]
    threads = case["threads"]
    sizes = case["buffer_words"]
    n0, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13 = [
        sizes[n]
        for n in (
            "Counts",
            "Ids",
            "Links",
            "Colors",
            "Rands",
            "Colored",
            "Pending",
            "NumColors",
            "BodyGroup",
            "Plan",
            "BatchIds",
            "ContactsIndirect",
            "ColoringIndirect",
            "CurrentColor",
        )
    ]
    luby_reset = C.gpu_reset_luby(layout)
    luby_step = C.gpu_step_graph_coloring_luby(layout)
    topo_reset = C.gpu_reset_topo_gc(layout)
    topo_clear = C.gpu_clear_completion_flag_topo_gc()
    topo_completion = C.gpu_reset_completion_flag_topo_gc()
    topo_step = C.gpu_step_graph_coloring_topo_gc(layout)
    topo_fix = C.gpu_fix_conflicts_topo_gc(layout)

    @T.macro
    def one(
        Counts,
        Ids,
        Links,
        Colors,
        Rands,
        Colored,
        Pending,
        NumColors,
        BodyGroup,
        Plan,
        BatchIds,
        ContactsIndirect,
        ColoringIndirect,
        CurrentColor,
        i,
    ):
        if op == "luby_reset":
            luby_reset(Colors, Rands, Links, Plan, i)
        elif op == "luby_step":
            luby_step(
                Counts,
                Ids,
                Links,
                Colors,
                Rands,
                NumColors,
                BodyGroup,
                CurrentColor,
                Plan,
                i,
                groups,
            )
        elif op == "topo_reset":
            topo_reset(Colors, Colored, Links, Plan, NumColors, Pending, i)
        elif op == "topo_clear":
            if i == 0:
                topo_clear(NumColors, ContactsIndirect, ColoringIndirect)
        elif op == "topo_completion":
            if i == 0:
                topo_completion(NumColors, ContactsIndirect, ColoringIndirect)
        elif op == "topo_step":
            topo_step(
                Counts,
                Ids,
                Links,
                Colors,
                Colored,
                NumColors,
                Plan,
                BodyGroup,
                BatchIds,
                Pending,
                i,
                groups,
            )
        else:
            topo_fix(
                Counts,
                Ids,
                Links,
                Colors,
                Colored,
                NumColors,
                Plan,
                BodyGroup,
                BatchIds,
                Pending,
                i,
                groups,
            )

    if target == "c":

        @T.prim_func
        def kernel(
            Counts: T.Tensor((n0,), "uint32"),
            Ids: T.Tensor((n1,), "uint32"),
            Links: T.Tensor((n2,), "uint32"),
            Colors: T.Tensor((n3,), "uint32"),
            Rands: T.Tensor((n4,), "uint32"),
            Colored: T.Tensor((n5,), "uint32"),
            Pending: T.Tensor((n6,), "uint32"),
            NumColors: T.Tensor((n7,), "uint32"),
            BodyGroup: T.Tensor((n8,), "uint32"),
            Plan: T.Tensor((n9,), "uint32"),
            BatchIds: T.Tensor((n10,), "uint32"),
            ContactsIndirect: T.Tensor((n11,), "uint32"),
            ColoringIndirect: T.Tensor((n12,), "uint32"),
            CurrentColor: T.Tensor((n13,), "uint32"),
        ):
            with T.Kernel(1):
                for i in T.serial(groups * threads):
                    one(
                        Counts,
                        Ids,
                        Links,
                        Colors,
                        Rands,
                        Colored,
                        Pending,
                        NumColors,
                        BodyGroup,
                        Plan,
                        BatchIds,
                        ContactsIndirect,
                        ColoringIndirect,
                        CurrentColor,
                        i,
                    )
    elif target == "ascend":
        import tilelang.ascend.language as A

        @A.prim_func
        def kernel(
            Counts: A.Tensor((n0,), "uint32"),
            Ids: A.Tensor((n1,), "uint32"),
            Links: A.Tensor((n2,), "uint32"),
            Colors: A.Tensor((n3,), "uint32"),
            Rands: A.Tensor((n4,), "uint32"),
            Colored: A.Tensor((n5,), "uint32"),
            Pending: A.Tensor((n6,), "uint32"),
            NumColors: A.Tensor((n7,), "uint32"),
            BodyGroup: A.Tensor((n8,), "uint32"),
            Plan: A.Tensor((n9,), "uint32"),
            BatchIds: A.Tensor((n10,), "uint32"),
            ContactsIndirect: A.Tensor((n11,), "uint32"),
            ColoringIndirect: A.Tensor((n12,), "uint32"),
            CurrentColor: A.Tensor((n13,), "uint32"),
        ):
            with A.Kernel(groups) as group:
                with A.SimtVF(threads=threads):
                    one(
                        Counts,
                        Ids,
                        Links,
                        Colors,
                        Rands,
                        Colored,
                        Pending,
                        NumColors,
                        BodyGroup,
                        Plan,
                        BatchIds,
                        ContactsIndirect,
                        ColoringIndirect,
                        CurrentColor,
                        group * threads + A.get_thread_binding(),
                    )
    else:

        @T.prim_func
        def kernel(
            Counts: T.Tensor((n0,), "uint32"),
            Ids: T.Tensor((n1,), "uint32"),
            Links: T.Tensor((n2,), "uint32"),
            Colors: T.Tensor((n3,), "uint32"),
            Rands: T.Tensor((n4,), "uint32"),
            Colored: T.Tensor((n5,), "uint32"),
            Pending: T.Tensor((n6,), "uint32"),
            NumColors: T.Tensor((n7,), "uint32"),
            BodyGroup: T.Tensor((n8,), "uint32"),
            Plan: T.Tensor((n9,), "uint32"),
            BatchIds: T.Tensor((n10,), "uint32"),
            ContactsIndirect: T.Tensor((n11,), "uint32"),
            ColoringIndirect: T.Tensor((n12,), "uint32"),
            CurrentColor: T.Tensor((n13,), "uint32"),
        ):
            with T.Kernel(groups, threads=threads) as group:
                one(
                    Counts,
                    Ids,
                    Links,
                    Colors,
                    Rands,
                    Colored,
                    Pending,
                    NumColors,
                    BodyGroup,
                    Plan,
                    BatchIds,
                    ContactsIndirect,
                    ColoringIndirect,
                    CurrentColor,
                    group * threads + T.get_thread_binding(),
                )

    return kernel
