# ruff: noqa: SIM117
"""Native-offset support-feature launch adapter."""

import tilelang.language as T
from kernel_support import f32
from shaders.shapes.shape import support_face

BUFFERS = ("Shapes", "Vertices", "Indices", "Inputs", "Output")


def make_kernel(l, c, target):
    n = c["count"]
    sizes = c["buffer_words"]
    groups = c["groups"]
    face = support_face(l)

    @T.macro
    def one(Shapes, Vertices, Indices, Inputs, Output, gid):
        if gid < n:
            for i in T.serial(gid, n, groups * 64):
                direction = T.alloc_local((3,), "float32")
                points = T.alloc_local((4, 3), "float32")
                count = T.alloc_local((1,), "int32")
                for d in T.serial(3):
                    direction[d] = f32(
                        Inputs,
                        i * l["feature_input_words"]
                        + l["feature_input"]["direction"]
                        + d,
                    )
                face(Shapes, i, Vertices, Indices, direction, points, count)
                for v in T.serial(4):
                    for d in T.serial(3):
                        Output[
                            i * l["feature_words"]
                            + l["feature"]["vertices"]
                            + v * l["vector_words"]
                            + d
                        ] = T.reinterpret(points[v, d], "uint32")
                Output[i * l["feature_words"] + l["feature"]["count"]] = T.cast(
                    count[0], "uint32"
                )

    if target == "c":

        @T.prim_func
        def kernel(
            Shapes: T.Tensor((sizes["Shapes"],), "uint32"),
            Vertices: T.Tensor((sizes["Vertices"],), "uint32"),
            Indices: T.Tensor((sizes["Indices"],), "uint32"),
            Inputs: T.Tensor((sizes["Inputs"],), "uint32"),
            Output: T.Tensor((sizes["Output"],), "uint32"),
        ):
            with T.Kernel(1):
                for gid in T.serial(groups * 64):
                    one(Shapes, Vertices, Indices, Inputs, Output, gid)
    elif target == "ascend":
        import tilelang.ascend.language as A

        @A.prim_func
        def kernel(
            Shapes: A.Tensor((sizes["Shapes"],), "uint32"),
            Vertices: A.Tensor((sizes["Vertices"],), "uint32"),
            Indices: A.Tensor((sizes["Indices"],), "uint32"),
            Inputs: A.Tensor((sizes["Inputs"],), "uint32"),
            Output: A.Tensor((sizes["Output"],), "uint32"),
        ):
            with A.Kernel(groups) as group:
                with A.SimtVF(threads=64):
                    one(
                        Shapes,
                        Vertices,
                        Indices,
                        Inputs,
                        Output,
                        group * 64 + A.get_thread_binding(),
                    )
    else:

        @T.prim_func
        def kernel(
            Shapes: T.Tensor((sizes["Shapes"],), "uint32"),
            Vertices: T.Tensor((sizes["Vertices"],), "uint32"),
            Indices: T.Tensor((sizes["Indices"],), "uint32"),
            Inputs: T.Tensor((sizes["Inputs"],), "uint32"),
            Output: T.Tensor((sizes["Output"],), "uint32"),
        ):
            with T.Kernel(groups, threads=64) as group:
                one(
                    Shapes,
                    Vertices,
                    Indices,
                    Inputs,
                    Output,
                    group * 64 + T.get_thread_binding(),
                )

    return kernel
