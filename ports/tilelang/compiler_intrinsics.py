"""Target-specific corrections to the pinned compiler's intrinsic names."""

from functools import cache


@cache
def register_intrinsics(target):
    if target != "metal":
        return
    from tilelang import tvm

    def metal_asin(op):
        # Metal overloads asin(float); the pinned default emits the C name asinf.
        # Use the target's intrinsic without changing precision or the equation.
        return tvm.tirx.call_pure_extern(op.dtype, "asin", *op.args)

    tvm.ir.register_intrin_lowering(
        "tirx.asin", target="metal", f=metal_asin, level=100
    )
