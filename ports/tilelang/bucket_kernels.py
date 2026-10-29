"""Color bucket launch plans over persistent device storage."""

import tilelang.language as T
from shaders.dynamics.color_buckets import (
    gpu_color_buckets_reset,
    gpu_color_buckets_count,
    gpu_color_buckets_scatter,
    gpu_color_dispatch_grid,
)


def launch(case, operation):
    if operation == "bucket_reset":
        return (case["buffer_words"]["Buckets"] + 63) // 64
    if operation == "bucket_dispatch":
        return 1
    return case["groups"]


def make_kernel(layout, case, target, operation):
    sizes = case["buffer_words"]
    groups = launch(case, operation)
    n0, n1, n2, n3, n4, n5, n6, n7, n8 = [
        sizes[n]
        for n in (
            "Colors",
            "Links",
            "Plan",
            "Buckets",
            "BatchIds",
            "SortedLinks",
            "Indices",
            "Stats",
            "Indirect",
        )
    ]
    reset = gpu_color_buckets_reset(n3)
    count = gpu_color_buckets_count(layout)
    scatter = gpu_color_buckets_scatter(layout)
    dispatch = gpu_color_dispatch_grid(layout, target == "c")

    @T.macro
    def one(
        Colors,
        Links,
        Plan,
        Buckets,
        BatchIds,
        SortedLinks,
        Indices,
        Stats,
        Indirect,
        Sizes,
        MaxColors,
        group,
        lane,
    ):
        if operation == "bucket_reset":
            reset(Buckets, group * 64 + lane)
        elif operation == "bucket_count":
            count(
                Colors,
                Links,
                Plan,
                Buckets,
                BatchIds,
                SortedLinks,
                group * 64 + lane,
                groups,
            )
        elif operation == "bucket_scatter":
            scatter(
                Colors,
                Links,
                Plan,
                Buckets,
                SortedLinks,
                BatchIds,
                Indices,
                group * 64 + lane,
                groups,
            )
        else:
            dispatch(Buckets, Stats, Indirect, BatchIds, Sizes, MaxColors, lane)

    if target == "c":

        @T.prim_func
        def kernel(
            Colors: T.Tensor((n0,), "uint32"),
            Links: T.Tensor((n1,), "uint32"),
            Plan: T.Tensor((n2,), "uint32"),
            Buckets: T.Tensor((n3,), "uint32"),
            BatchIds: T.Tensor((n4,), "uint32"),
            SortedLinks: T.Tensor((n5,), "uint32"),
            Indices: T.Tensor((n6,), "uint32"),
            Stats: T.Tensor((n7,), "uint32"),
            Indirect: T.Tensor((n8,), "uint32"),
        ):
            with T.Kernel(1):
                Sizes = T.alloc_local((64,), "uint32")
                MaxColors = T.alloc_local((64,), "uint32")
                if operation == "bucket_dispatch":
                    one(
                        Colors,
                        Links,
                        Plan,
                        Buckets,
                        BatchIds,
                        SortedLinks,
                        Indices,
                        Stats,
                        Indirect,
                        Sizes,
                        MaxColors,
                        0,
                        0,
                    )
                else:
                    for group in T.serial(groups):
                        for lane in T.serial(64):
                            one(
                                Colors,
                                Links,
                                Plan,
                                Buckets,
                                BatchIds,
                                SortedLinks,
                                Indices,
                                Stats,
                                Indirect,
                                Sizes,
                                MaxColors,
                                group,
                                lane,
                            )
    elif target == "ascend":
        import tilelang.ascend.language as A

        @A.prim_func
        def kernel(
            Colors: A.Tensor((n0,), "uint32"),
            Links: A.Tensor((n1,), "uint32"),
            Plan: A.Tensor((n2,), "uint32"),
            Buckets: A.Tensor((n3,), "uint32"),
            BatchIds: A.Tensor((n4,), "uint32"),
            SortedLinks: A.Tensor((n5,), "uint32"),
            Indices: A.Tensor((n6,), "uint32"),
            Stats: A.Tensor((n7,), "uint32"),
            Indirect: A.Tensor((n8,), "uint32"),
        ):
            with A.Kernel(groups) as group:
                with A.SimtVF(threads=64):
                    Sizes = T.alloc_shared((64,), "uint32")
                    MaxColors = T.alloc_shared((64,), "uint32")
                    one(
                        Colors,
                        Links,
                        Plan,
                        Buckets,
                        BatchIds,
                        SortedLinks,
                        Indices,
                        Stats,
                        Indirect,
                        Sizes,
                        MaxColors,
                        group,
                        A.get_thread_binding(),
                    )
    else:

        @T.prim_func
        def kernel(
            Colors: T.Tensor((n0,), "uint32"),
            Links: T.Tensor((n1,), "uint32"),
            Plan: T.Tensor((n2,), "uint32"),
            Buckets: T.Tensor((n3,), "uint32"),
            BatchIds: T.Tensor((n4,), "uint32"),
            SortedLinks: T.Tensor((n5,), "uint32"),
            Indices: T.Tensor((n6,), "uint32"),
            Stats: T.Tensor((n7,), "uint32"),
            Indirect: T.Tensor((n8,), "uint32"),
        ):
            with T.Kernel(groups, threads=64) as group:
                Sizes = T.alloc_shared((64,), "uint32")
                MaxColors = T.alloc_shared((64,), "uint32")
                one(
                    Colors,
                    Links,
                    Plan,
                    Buckets,
                    BatchIds,
                    SortedLinks,
                    Indices,
                    Stats,
                    Indirect,
                    Sizes,
                    MaxColors,
                    group,
                    T.get_thread_binding(),
                )

    return kernel


def make_program(layout, case, target):
    from scan_kernels import make_program as scan_program

    def stage(op):
        return {
            "name": op,
            "kernel": make_kernel(layout, case, target, op),
            "blocks": launch(case, op),
            "threads": 64,
            "buffer_map": {
                n: n
                for n in (
                    "Colors",
                    "Links",
                    "Plan",
                    "Buckets",
                    "BatchIds",
                    "SortedLinks",
                    "Indices",
                    "Stats",
                    "Indirect",
                )
            },
        }

    if case["kind"] != "bucket_build":
        return [stage(case["kind"])]
    scan_case = {
        **case,
        "kind": "prefix_scan",
        "batches": 1,
        "stride": case["bucket_count"],
        "count": case["bucket_count"],
        "buffer_words": {
            **case["buffer_words"],
            "Data": case["buffer_words"]["Buckets"],
        },
    }
    middle = scan_program(layout, scan_case, target)
    for plan in middle:
        plan["buffer_map"] = {
            n: ("Buckets" if v == "Data" else v) for n, v in plan["buffer_map"].items()
        }
    return [
        stage("bucket_reset"),
        stage("bucket_count"),
        *middle,
        stage("bucket_scatter"),
        stage("bucket_dispatch"),
    ]
