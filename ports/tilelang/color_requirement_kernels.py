"""Launch the mirrored native palette/round feedback kernel."""

import tilelang.language as T
from shaders.dynamics.coloring import gpu_coloring_requirements

BUFFERS = ("Colors", "Links", "Plan", "Budget", "Requirements")


def make_kernel(layout, case, target):
    sizes = case["buffer_words"]
    nc, nl, np, nb, nr = [sizes[name] for name in BUFFERS]
    one = gpu_coloring_requirements(layout, target == "c")
    if target == "ascend":
        import tilelang.ascend.language as A

        @A.prim_func
        def kernel(
            Colors: A.Tensor((nc,), "uint32"),
            Links: A.Tensor((nl,), "uint32"),
            Plan: A.Tensor((np,), "uint32"),
            Budget: A.Tensor((nb,), "uint32"),
            Requirements: A.Tensor((nr,), "uint32"),
        ):
            with A.Kernel(1):  # noqa: SIM117 -- preserve nested accelerator scopes
                with A.SimtVF(threads=64):
                    Highest = A.alloc_shared((64,), "uint32")
                    Pending = A.alloc_shared((64,), "uint32")
                    one(
                        Colors,
                        Links,
                        Plan,
                        Budget,
                        Requirements,
                        Highest,
                        Pending,
                        A.get_thread_binding(),
                    )
    else:

        @T.prim_func
        def kernel(
            Colors: T.Tensor((nc,), "uint32"),
            Links: T.Tensor((nl,), "uint32"),
            Plan: T.Tensor((np,), "uint32"),
            Budget: T.Tensor((nb,), "uint32"),
            Requirements: T.Tensor((nr,), "uint32"),
        ):
            with T.Kernel(1, threads=64):
                Highest = T.alloc_shared((64,), "uint32")
                Pending = T.alloc_shared((64,), "uint32")
                one(
                    Colors,
                    Links,
                    Plan,
                    Budget,
                    Requirements,
                    Highest,
                    Pending,
                    0 if target == "c" else T.get_thread_binding(),
                )

    return kernel
