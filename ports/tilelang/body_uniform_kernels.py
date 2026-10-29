"""Generate the compact body-kernel parameters from native Rust uniforms on-device."""

import tilelang.language as T
from shaders.dynamics.sim_params import body_coefficients


def make_kernel(layout, case, target):
    n = layout["rbd_params_words"]
    p = case["buffer_words"]["Params"]
    compute = body_coefficients(layout)
    if target == "c":

        @T.prim_func
        def kernel(
            RbdParams: T.Tensor((n,), "uint32"),
            Gravity: T.Tensor((4,), "uint32"),
            Params: T.Tensor((p,), "uint32"),
        ):
            with T.Kernel(1):
                compute(RbdParams, Gravity, Params)
    elif target == "ascend":
        import tilelang.ascend.language as A

        @A.prim_func
        def kernel(
            RbdParams: A.Tensor((n,), "uint32"),
            Gravity: A.Tensor((4,), "uint32"),
            Params: A.Tensor((p,), "uint32"),
        ):
            with A.Kernel(1):
                with A.SimtVF(threads=1):
                    compute(RbdParams, Gravity, Params)
    else:

        @T.prim_func
        def kernel(
            RbdParams: T.Tensor((n,), "uint32"),
            Gravity: T.Tensor((4,), "uint32"),
            Params: T.Tensor((p,), "uint32"),
        ):
            with T.Kernel(1, threads=1):
                compute(RbdParams, Gravity, Params)

    return kernel
