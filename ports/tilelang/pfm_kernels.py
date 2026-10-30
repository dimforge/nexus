# ruff: noqa: SIM117
"""Ordered native-layout PFM conversion and manifold launch adapters."""

import tilelang.language as T
from kernel_support import f32
from shaders.queries.contact_pfm_pfm import contact_manifold_pfm_pfm
from shaders.shapes.shape import pfm_subshape

BUFFERS = (
    "Shapes",
    "Poses",
    "Vertices",
    "Indices",
    "Inputs",
    "Output",
    "SubShapes",
    "Radii",
)


def make_kernel(l, c, target, stage):
    n = c["count"] * (2 if stage == "subshape" else 1)
    sizes = c["buffer_words"]
    groups = c["groups"]
    subshape = pfm_subshape(l)
    manifold = contact_manifold_pfm_pfm(l)

    @T.macro
    def one(Shapes, Poses, Vertices, Indices, Inputs, Output, SubShapes, Radii, gid):
        if gid < n:
            for i in T.serial(gid, n, groups * 64):
                if stage == "subshape":
                    sub = T.alloc_local((l["shape_words"],), "uint32")
                    radius = T.alloc_local((1,), "float32")
                    valid = T.alloc_local((1,), "bool")
                    subshape(Shapes, i, sub, radius, valid)
                    for w in T.serial(l["shape_words"]):
                        SubShapes[i * l["shape_words"] + w] = sub[w]
                    Radii[i * l["pfm_radius_words"] + l["pfm_radius"]["thickness"]] = (
                        T.reinterpret(radius[0], "uint32")
                    )
                    Radii[i * l["pfm_radius_words"] + l["pfm_radius"]["valid"]] = (
                        T.cast(valid[0], "uint32")
                    )
                else:
                    q = T.alloc_local((4,), "float32")
                    t = T.alloc_local((3,), "float32")
                    points = T.alloc_local((4, 4), "float32")
                    normal = T.alloc_local((3,), "float32")
                    count = T.alloc_local((1,), "int32")
                    for d in T.serial(4):
                        q[d] = f32(Poses, i * l["pose_words"] + l["pose_rotation"] + d)
                    for d in T.serial(3):
                        t[d] = f32(
                            Poses, i * l["pose_words"] + l["pose_translation"] + d
                        )
                        normal[d] = 0.0
                    for v in T.serial(4):
                        for d in T.serial(4):
                            points[v, d] = 0.0
                    count[0] = 0
                    if (
                        Radii[i * 2 * l["pfm_radius_words"] + l["pfm_radius"]["valid"]]
                        != 0
                    ) & (
                        Radii[
                            (i * 2 + 1) * l["pfm_radius_words"]
                            + l["pfm_radius"]["valid"]
                        ]
                        != 0
                    ):
                        manifold(
                            SubShapes,
                            i * 2,
                            i * 2 + 1,
                            Vertices,
                            Indices,
                            q,
                            t,
                            f32(
                                Radii,
                                i * 2 * l["pfm_radius_words"]
                                + l["pfm_radius"]["thickness"],
                            ),
                            f32(
                                Radii,
                                (i * 2 + 1) * l["pfm_radius_words"]
                                + l["pfm_radius"]["thickness"],
                            ),
                            f32(
                                Inputs,
                                i * l["pfm_input_words"] + l["pfm_input"]["prediction"],
                            ),
                            points,
                            normal,
                            count,
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
                        ] = T.reinterpret(normal[d], "uint32")
                    Output[
                        i * l["feature_manifold_words"] + l["feature_manifold"]["count"]
                    ] = T.cast(count[0], "uint32")

    if target == "c":

        @T.prim_func
        def kernel(
            Shapes: T.Tensor((sizes["Shapes"],), "uint32"),
            Poses: T.Tensor((sizes["Poses"],), "uint32"),
            Vertices: T.Tensor((sizes["Vertices"],), "uint32"),
            Indices: T.Tensor((sizes["Indices"],), "uint32"),
            Inputs: T.Tensor((sizes["Inputs"],), "uint32"),
            Output: T.Tensor((sizes["Output"],), "uint32"),
            SubShapes: T.Tensor((sizes["SubShapes"],), "uint32"),
            Radii: T.Tensor((sizes["Radii"],), "uint32"),
        ):
            with T.Kernel(1):
                for gid in T.serial(groups * 64):
                    one(
                        Shapes,
                        Poses,
                        Vertices,
                        Indices,
                        Inputs,
                        Output,
                        SubShapes,
                        Radii,
                        gid,
                    )
    elif target == "ascend":
        import tilelang.ascend.language as A

        @A.prim_func
        def kernel(
            Shapes: A.Tensor((sizes["Shapes"],), "uint32"),
            Poses: A.Tensor((sizes["Poses"],), "uint32"),
            Vertices: A.Tensor((sizes["Vertices"],), "uint32"),
            Indices: A.Tensor((sizes["Indices"],), "uint32"),
            Inputs: A.Tensor((sizes["Inputs"],), "uint32"),
            Output: A.Tensor((sizes["Output"],), "uint32"),
            SubShapes: A.Tensor((sizes["SubShapes"],), "uint32"),
            Radii: A.Tensor((sizes["Radii"],), "uint32"),
        ):
            with A.Kernel(groups) as group:
                with A.SimtVF(threads=64):
                    one(
                        Shapes,
                        Poses,
                        Vertices,
                        Indices,
                        Inputs,
                        Output,
                        SubShapes,
                        Radii,
                        group * 64 + A.get_thread_binding(),
                    )
    else:

        @T.prim_func
        def kernel(
            Shapes: T.Tensor((sizes["Shapes"],), "uint32"),
            Poses: T.Tensor((sizes["Poses"],), "uint32"),
            Vertices: T.Tensor((sizes["Vertices"],), "uint32"),
            Indices: T.Tensor((sizes["Indices"],), "uint32"),
            Inputs: T.Tensor((sizes["Inputs"],), "uint32"),
            Output: T.Tensor((sizes["Output"],), "uint32"),
            SubShapes: T.Tensor((sizes["SubShapes"],), "uint32"),
            Radii: T.Tensor((sizes["Radii"],), "uint32"),
        ):
            with T.Kernel(groups, threads=64) as group:
                one(
                    Shapes,
                    Poses,
                    Vertices,
                    Indices,
                    Inputs,
                    Output,
                    SubShapes,
                    Radii,
                    group * 64 + T.get_thread_binding(),
                )

    return kernel
