# ruff: noqa: SIM117
"""Native-offset polygonal contact launch adapter."""

import tilelang.language as T
from kernel_support import f32
from shaders.queries.polygonal_feature import contacts

BUFFERS = ("Features", "Poses", "Inputs", "Output")


def make_kernel(l, c, target):
    n = c["count"]
    sizes = c["buffer_words"]
    groups = c["groups"]

    @T.macro
    def one(Features, Poses, Inputs, Output, gid):
        if gid < n:
            for i in T.serial(gid, n, groups * 64):
                q = T.alloc_local((4,), "float32")
                t = T.alloc_local((3,), "float32")
                normal = T.alloc_local((3,), "float32")
                a = T.alloc_local((4, 3), "float32")
                b = T.alloc_local((4, 3), "float32")
                points = T.alloc_local((4, 4), "float32")
                count = T.alloc_local((1,), "int32")
                for d in T.serial(4):
                    q[d] = f32(Poses, i * l["pose_words"] + l["pose_rotation"] + d)
                for d in T.serial(3):
                    t[d] = f32(Poses, i * l["pose_words"] + l["pose_translation"] + d)
                    normal[d] = f32(
                        Inputs,
                        i * l["polygonal_input_words"]
                        + l["polygonal_input"]["normal"]
                        + d,
                    )
                for v in T.serial(4):
                    for d in T.serial(3):
                        a[v, d] = f32(
                            Features,
                            i * 2 * l["feature_words"]
                            + l["feature"]["vertices"]
                            + v * l["vector_words"]
                            + d,
                        )
                        b[v, d] = f32(
                            Features,
                            (i * 2 + 1) * l["feature_words"]
                            + l["feature"]["vertices"]
                            + v * l["vector_words"]
                            + d,
                        )
                contacts(
                    q,
                    t,
                    a,
                    normal,
                    b,
                    T.cast(
                        Features[i * 2 * l["feature_words"] + l["feature"]["count"]],
                        "int32",
                    ),
                    T.cast(
                        Features[
                            (i * 2 + 1) * l["feature_words"] + l["feature"]["count"]
                        ],
                        "int32",
                    ),
                    f32(
                        Inputs,
                        i * l["polygonal_input_words"]
                        + l["polygonal_input"]["prediction"],
                    ),
                    points,
                    count,
                    Inputs[
                        i * l["polygonal_input_words"] + l["polygonal_input"]["flipped"]
                    ]
                    != 0,
                )
                for v in T.serial(4):
                    for d in T.serial(4):
                        Output[
                            i * l["feature_manifold_words"]
                            + l["feature_manifold"]["points"]
                            + v * l["feature_point_words"]
                            + d
                        ] = T.reinterpret(points[v, d], "uint32")
                for d in T.serial(3):
                    Output[
                        i * l["feature_manifold_words"]
                        + l["feature_manifold"]["normal"]
                        + d
                    ] = T.uint32(0)
                Output[
                    i * l["feature_manifold_words"] + l["feature_manifold"]["count"]
                ] = T.cast(count[0], "uint32")

    if target == "c":

        @T.prim_func
        def kernel(
            Features: T.Tensor((sizes["Features"],), "uint32"),
            Poses: T.Tensor((sizes["Poses"],), "uint32"),
            Inputs: T.Tensor((sizes["Inputs"],), "uint32"),
            Output: T.Tensor((sizes["Output"],), "uint32"),
        ):
            with T.Kernel(1):
                for gid in T.serial(groups * 64):
                    one(Features, Poses, Inputs, Output, gid)
    elif target == "ascend":
        import tilelang.ascend.language as A

        @A.prim_func
        def kernel(
            Features: A.Tensor((sizes["Features"],), "uint32"),
            Poses: A.Tensor((sizes["Poses"],), "uint32"),
            Inputs: A.Tensor((sizes["Inputs"],), "uint32"),
            Output: A.Tensor((sizes["Output"],), "uint32"),
        ):
            with A.Kernel(groups) as group:
                with A.SimtVF(threads=64):
                    one(
                        Features,
                        Poses,
                        Inputs,
                        Output,
                        group * 64 + A.get_thread_binding(),
                    )
    else:

        @T.prim_func
        def kernel(
            Features: T.Tensor((sizes["Features"],), "uint32"),
            Poses: T.Tensor((sizes["Poses"],), "uint32"),
            Inputs: T.Tensor((sizes["Inputs"],), "uint32"),
            Output: T.Tensor((sizes["Output"],), "uint32"),
        ):
            with T.Kernel(groups, threads=64) as group:
                one(
                    Features, Poses, Inputs, Output, group * 64 + T.get_thread_binding()
                )

    return kernel
