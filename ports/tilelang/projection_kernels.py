# ruff: noqa: SIM117
"""Scalar projection launch adapters; payload packing is an oracle debug format."""

import tilelang.language as T
from kernel_support import f32
from shaders.queries.projection import barycentric_coordinates
from shaders.shapes.segment import project_local_point_and_get_location as segment
from shaders.shapes.tetrahedron import (
    project_local_point_and_get_location as tetrahedron,
)
from shaders.shapes.triangle import project_local_point_and_get_location as triangle


def make_kernel(l, c, target):
    iw = l["projection_input_words"]
    ow = l["projection_output_words"]
    pi = l["projection_input"]
    po = l["projection_output"]
    n = c["count"]
    sizes = c["buffer_words"]
    groups = c["groups"]
    kind = c["kind"]

    @T.macro
    def one(Inputs, Output, gid):
        if gid < n:
            for i in T.serial(gid, n, groups * 64):
                a = T.alloc_local((3,), "float32")
                b = T.alloc_local((3,), "float32")
                cc = T.alloc_local((3,), "float32")
                d = T.alloc_local((3,), "float32")
                pt = T.alloc_local((3,), "float32")
                values = T.alloc_local((6,), "float32")
                flags = T.alloc_local((3,), "uint32")
                bary = T.alloc_local((3,), "float32")
                for k in T.serial(3):
                    a[k] = f32(Inputs, i * iw + pi["a"] + k)
                    b[k] = f32(Inputs, i * iw + pi["b"] + k)
                    cc[k] = f32(Inputs, i * iw + pi["c"] + k)
                    d[k] = f32(Inputs, i * iw + pi["d"] + k)
                    pt[k] = f32(Inputs, i * iw + pi["query"] + k)
                if kind == "segment_projection":
                    segment(a, b, pt, Inputs[i * iw + pi["solid"]] != 0, values, flags)
                elif kind == "triangle_projection":
                    triangle(
                        a, b, cc, pt, Inputs[i * iw + pi["solid"]] != 0, values, flags
                    )
                else:
                    tetrahedron(
                        a,
                        b,
                        cc,
                        d,
                        pt,
                        Inputs[i * iw + pi["solid"]] != 0,
                        values,
                        flags,
                    )
                barycentric_coordinates(values, flags, bary)
                for k in T.serial(3):
                    Output[i * ow + po["point"] + k] = T.reinterpret(
                        values[k], "uint32"
                    )
                    Output[i * ow + po["barycentric"] + k] = T.reinterpret(
                        values[3 + k], "uint32"
                    )
                    Output[i * ow + po["expanded"] + k] = T.reinterpret(
                        bary[k], "uint32"
                    )
                    Output[i * ow + po["padding"] + k] = T.uint32(0)
                Output[i * ow + po["inside"]] = flags[2]
                Output[i * ow + po["feature"]] = flags[0]
                Output[i * ow + po["index"]] = flags[1]

    if target == "c":

        @T.prim_func
        def kernel(
            Inputs: T.Tensor((sizes["Inputs"],), "uint32"),
            Output: T.Tensor((sizes["Output"],), "uint32"),
        ):
            with T.Kernel(1):
                for gid in T.serial(groups * 64):
                    one(Inputs, Output, gid)
    elif target == "ascend":
        import tilelang.ascend.language as A

        @A.prim_func
        def kernel(
            Inputs: A.Tensor((sizes["Inputs"],), "uint32"),
            Output: A.Tensor((sizes["Output"],), "uint32"),
        ):
            with A.Kernel(groups) as group:
                with A.SimtVF(threads=64):
                    one(Inputs, Output, group * 64 + A.get_thread_binding())
    else:

        @T.prim_func
        def kernel(
            Inputs: T.Tensor((sizes["Inputs"],), "uint32"),
            Output: T.Tensor((sizes["Output"],), "uint32"),
        ):
            with T.Kernel(groups, threads=64) as group:
                one(Inputs, Output, group * 64 + T.get_thread_binding())

    return kernel
