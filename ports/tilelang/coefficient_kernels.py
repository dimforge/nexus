import tilelang.language as T
from shaders.dynamics.sim_params import contact_coefficients


def make_kernel(layout, target):
    n = layout["rbd_params_words"]
    compute = contact_coefficients(layout)

    @T.macro
    def one(RbdParams, Coefficients):
        local = T.alloc_local((6,), "float32")
        compute(RbdParams, local)
        for i in T.serial(6):
            Coefficients[i] = T.reinterpret(local[i], "uint32")

    if target == "c":

        @T.prim_func
        def kernel(
            RbdParams: T.Tensor((n,), "uint32"), Coefficients: T.Tensor((9,), "uint32")
        ):
            with T.Kernel(1):
                one(RbdParams, Coefficients)
    elif target == "ascend":
        import tilelang.ascend.language as A

        @A.prim_func
        def kernel(
            RbdParams: A.Tensor((n,), "uint32"), Coefficients: A.Tensor((9,), "uint32")
        ):
            with A.Kernel(1):
                with A.SimtVF(threads=1):
                    one(RbdParams, Coefficients)
    else:

        @T.prim_func
        def kernel(
            RbdParams: T.Tensor((n,), "uint32"), Coefficients: T.Tensor((9,), "uint32")
        ):
            with T.Kernel(1, threads=1):
                one(RbdParams, Coefficients)

    return kernel
