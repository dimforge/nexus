"""Target launch wrappers for the mirrored irregular integer kernels."""

import tilelang.language as T
from shaders.dynamics.canonical_order import gpu_stabilize_body_constraint_ids


def make_kernel(case, target):
    rows, storage = case["rows"], case["storage"]
    counts = max(rows, 1)
    one = gpu_stabilize_body_constraint_ids(case)
    if target == "c":

        @T.prim_func
        def kernel(
            Counts: T.Tensor((counts,), "uint32"),
            Ids: T.Tensor((storage,), "uint32"),
            Sorted: T.Tensor((storage,), "uint32"),
        ):
            with T.Kernel(1):
                for body in T.serial(rows):
                    for lane in T.serial(64):
                        one(Counts, Ids, Sorted, body, lane)
    elif target == "ascend":
        import tilelang.ascend.language as A

        @A.prim_func
        def kernel(
            Counts: A.Tensor((counts,), "uint32"),
            Ids: A.Tensor((storage,), "uint32"),
            Sorted: A.Tensor((storage,), "uint32"),
        ):
            with A.Kernel(max(rows, 1)) as body:
                with A.SimtVF(threads=64):
                    one(Counts, Ids, Sorted, body, A.get_thread_binding())
    else:

        @T.prim_func
        def kernel(
            Counts: T.Tensor((counts,), "uint32"),
            Ids: T.Tensor((storage,), "uint32"),
            Sorted: T.Tensor((storage,), "uint32"),
        ):
            with T.Kernel(max(rows, 1), threads=64) as body:
                one(Counts, Ids, Sorted, body, T.get_thread_binding())

    return kernel
