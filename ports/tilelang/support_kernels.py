# ruff: noqa: SIM117
"""Support-map and bounded GJK launch adapters with native-exported offsets."""

import tilelang.language as T
from kernel_support import f32
from shaders.queries.gjk.gjk import closest_points, cso_point_from_shapes
from shaders.shapes.shape import local_support_point, support_point

BUFFERS = ("Shapes", "Poses", "Vertices", "Inputs", "Output", "State")


def make_kernel(l, c, target):
    iw = l["support_input_words"]
    ow = l["support_output_words"]
    sw = l["simplex_words"]
    cw = l["cso_words"]
    si = l["support_input"]
    so = l["support_output"]
    sf = l["simplex"]
    sizes = c["buffer_words"]
    n = c["count"]
    groups = c["groups"]
    kind = c["kind"]
    local = local_support_point(l)
    world = support_point(l)
    cso = cso_point_from_shapes(l)
    gjk = closest_points(l)

    @T.macro
    def one(Shapes, Poses, Vertices, Inputs, Output, State, gid):
        if gid < n:
            for i in T.serial(gid, n, groups * 64):
                q = T.alloc_local((4,), "float32")
                shift = T.alloc_local((3,), "float32")
                direction = T.alloc_local((3,), "float32")
                out = T.alloc_local((3,), "float32")
                result = T.alloc_local((3, 3), "float32")
                status = T.alloc_local((1,), "uint32")
                for d in T.serial(4):
                    q[d] = f32(Poses, i * l["pose_words"] + l["pose_rotation"] + d)
                for d in T.serial(3):
                    shift[d] = f32(
                        Poses, i * l["pose_words"] + l["pose_translation"] + d
                    )
                    direction[d] = f32(Inputs, i * iw + si["direction"] + d)
                if kind == "support_local":
                    local(Shapes, i * 2, Vertices, direction, out)
                    for d in T.serial(3):
                        Output[i * ow + so["a"] + d] = T.reinterpret(out[d], "uint32")
                elif kind == "support_world":
                    world(Shapes, i * 2, Vertices, q, shift, direction, out)
                    for d in T.serial(3):
                        Output[i * ow + so["a"] + d] = T.reinterpret(out[d], "uint32")
                elif kind == "cso_support":
                    cso(Shapes, i * 2, i * 2 + 1, Vertices, q, shift, direction, result)
                    for field in T.serial(3):
                        for d in T.serial(3):
                            Output[i * ow + so["a"] + field * l["vector_words"] + d] = (
                                T.reinterpret(result[field, d], "uint32")
                            )
                else:
                    vertices = T.alloc_local((4, 3, 3), "float32")
                    projection = T.alloc_local((3,), "float32")
                    previous = T.alloc_local((3,), "float32")
                    dims = T.alloc_local((2,), "uint32")
                    indices = T.alloc_local((4,), "uint32")
                    for v in T.serial(4):
                        for field in T.serial(3):
                            for d in T.serial(3):
                                vertices[v, field, d] = f32(
                                    State,
                                    i * sw
                                    + sf["vertices"]
                                    + v * cw
                                    + field * l["vector_words"]
                                    + d,
                                )
                    for v in T.serial(4):
                        indices[v] = State[i * sw + sf["previous_vertices"] + v]
                    for d in T.serial(3):
                        projection[d] = f32(State, i * sw + sf["projection"] + d)
                        previous[d] = f32(State, i * sw + sf["previous_projection"] + d)
                    dims[0] = State[i * sw + sf["dim"]]
                    dims[1] = State[i * sw + sf["previous_dim"]]
                    gjk(
                        Shapes,
                        i * 2,
                        i * 2 + 1,
                        Vertices,
                        q,
                        shift,
                        f32(Inputs, i * iw + si["max_distance"]),
                        Inputs[i * iw + si["exact"]] != 0,
                        vertices,
                        projection,
                        dims,
                        indices,
                        previous,
                        result,
                        status,
                    )
                    for v in T.serial(4):
                        for field in T.serial(3):
                            for d in T.serial(3):
                                State[
                                    i * sw
                                    + sf["vertices"]
                                    + v * cw
                                    + field * l["vector_words"]
                                    + d
                                ] = T.reinterpret(vertices[v, field, d], "uint32")
                    for v in T.serial(4):
                        State[i * sw + sf["previous_vertices"] + v] = indices[v]
                    for d in T.serial(3):
                        State[i * sw + sf["projection"] + d] = T.reinterpret(
                            projection[d], "uint32"
                        )
                        State[i * sw + sf["previous_projection"] + d] = T.reinterpret(
                            previous[d], "uint32"
                        )
                    State[i * sw + sf["dim"]] = dims[0]
                    State[i * sw + sf["previous_dim"]] = dims[1]
                    for field in T.serial(3):
                        for d in T.serial(3):
                            Output[i * ow + so["a"] + field * l["vector_words"] + d] = (
                                T.reinterpret(result[field, d], "uint32")
                            )
                    Output[i * ow + so["status"]] = status[0]

    if target == "c":

        @T.prim_func
        def kernel(
            Shapes: T.Tensor((sizes["Shapes"],), "uint32"),
            Poses: T.Tensor((sizes["Poses"],), "uint32"),
            Vertices: T.Tensor((sizes["Vertices"],), "uint32"),
            Inputs: T.Tensor((sizes["Inputs"],), "uint32"),
            Output: T.Tensor((sizes["Output"],), "uint32"),
            State: T.Tensor((sizes["State"],), "uint32"),
        ):
            with T.Kernel(1):
                for gid in T.serial(groups * 64):
                    one(Shapes, Poses, Vertices, Inputs, Output, State, gid)
    elif target == "ascend":
        import tilelang.ascend.language as A

        @A.prim_func
        def kernel(
            Shapes: A.Tensor((sizes["Shapes"],), "uint32"),
            Poses: A.Tensor((sizes["Poses"],), "uint32"),
            Vertices: A.Tensor((sizes["Vertices"],), "uint32"),
            Inputs: A.Tensor((sizes["Inputs"],), "uint32"),
            Output: A.Tensor((sizes["Output"],), "uint32"),
            State: A.Tensor((sizes["State"],), "uint32"),
        ):
            with A.Kernel(groups) as group:
                with A.SimtVF(threads=64):
                    one(
                        Shapes,
                        Poses,
                        Vertices,
                        Inputs,
                        Output,
                        State,
                        group * 64 + A.get_thread_binding(),
                    )
    else:

        @T.prim_func
        def kernel(
            Shapes: T.Tensor((sizes["Shapes"],), "uint32"),
            Poses: T.Tensor((sizes["Poses"],), "uint32"),
            Vertices: T.Tensor((sizes["Vertices"],), "uint32"),
            Inputs: T.Tensor((sizes["Inputs"],), "uint32"),
            Output: T.Tensor((sizes["Output"],), "uint32"),
            State: T.Tensor((sizes["State"],), "uint32"),
        ):
            with T.Kernel(groups, threads=64) as group:
                one(
                    Shapes,
                    Poses,
                    Vertices,
                    Inputs,
                    Output,
                    State,
                    group * 64 + T.get_thread_binding(),
                )

    return kernel
