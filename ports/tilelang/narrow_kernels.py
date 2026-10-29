"""Native buffers and target launches for analytic contacts."""

import tilelang.language as T
from shaders.broad_phase.narrow_phase import gpu_narrow_phase_shape_shape

BUFFERS = (
    "Pairs",
    "Plan",
    "Poses",
    "Shapes",
    "Contacts",
    "Parents",
    "Materials",
    "RbdParams",
)


def make_program(layout, case, target):
    kernel = make_kernel(layout, case, target)
    return [
        dict(
            name="analytic_contacts",
            kernel=kernel,
            buffer_map={n: n for n in BUFFERS},
            blocks=case["groups"],
            threads=64,
        )
    ]


def make_kernel(layout, case, target):
    one = gpu_narrow_phase_shape_shape(layout, case)
    groups = case["groups"]
    n0 = case["buffer_words"]["Pairs"]
    n1 = case["buffer_words"]["Plan"]
    n2 = case["buffer_words"]["Poses"]
    n3 = case["buffer_words"]["Shapes"]
    n4 = case["buffer_words"]["Contacts"]
    n5 = case["buffer_words"]["Parents"]
    n6 = case["buffer_words"]["Materials"]
    n7 = case["buffer_words"]["RbdParams"]
    if target == "c":

        @T.prim_func
        def kernel(
            Pairs: T.Tensor((n0,), "uint32"),
            Plan: T.Tensor((n1,), "uint32"),
            Poses: T.Tensor((n2,), "uint32"),
            Shapes: T.Tensor((n3,), "uint32"),
            Contacts: T.Tensor((n4,), "uint32"),
            Parents: T.Tensor((n5,), "uint32"),
            Materials: T.Tensor((n6,), "uint32"),
            RbdParams: T.Tensor((n7,), "uint32"),
        ):
            with T.Kernel(1):
                for gid in T.serial(groups * 64):
                    one(
                        Pairs,
                        Plan,
                        Poses,
                        Shapes,
                        Contacts,
                        Parents,
                        Materials,
                        RbdParams,
                        gid,
                    )
    elif target == "ascend":
        import tilelang.ascend.language as A

        @A.prim_func
        def kernel(
            Pairs: A.Tensor((n0,), "uint32"),
            Plan: A.Tensor((n1,), "uint32"),
            Poses: A.Tensor((n2,), "uint32"),
            Shapes: A.Tensor((n3,), "uint32"),
            Contacts: A.Tensor((n4,), "uint32"),
            Parents: A.Tensor((n5,), "uint32"),
            Materials: A.Tensor((n6,), "uint32"),
            RbdParams: A.Tensor((n7,), "uint32"),
        ):
            with A.Kernel(groups) as group:
                with A.SimtVF(threads=64):
                    one(
                        Pairs,
                        Plan,
                        Poses,
                        Shapes,
                        Contacts,
                        Parents,
                        Materials,
                        RbdParams,
                        group * 64 + A.get_thread_binding(),
                    )
    else:

        @T.prim_func
        def kernel(
            Pairs: T.Tensor((n0,), "uint32"),
            Plan: T.Tensor((n1,), "uint32"),
            Poses: T.Tensor((n2,), "uint32"),
            Shapes: T.Tensor((n3,), "uint32"),
            Contacts: T.Tensor((n4,), "uint32"),
            Parents: T.Tensor((n5,), "uint32"),
            Materials: T.Tensor((n6,), "uint32"),
            RbdParams: T.Tensor((n7,), "uint32"),
        ):
            with T.Kernel(groups, threads=64) as group:
                one(
                    Pairs,
                    Plan,
                    Poses,
                    Shapes,
                    Contacts,
                    Parents,
                    Materials,
                    RbdParams,
                    group * 64 + T.get_thread_binding(),
                )

    return kernel
