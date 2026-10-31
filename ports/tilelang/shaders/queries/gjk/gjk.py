"""Counterpart of queries/gjk/gjk.rs; bounded native GJK and support maps."""

import tilelang.language as T
from kernel_support import dot

from ...shapes.shape import local_support_point, support_point
from .cso_point import from_points
from .voronoi_simplex3 import add_point, project_origin_and_reduce


@T.macro
def result(Vertices, Projection, Dim, PrevVertices, PrevProjection, previous, Output):
    for side in T.serial(2):
        for d in T.serial(3):
            Output[side, d] = 0.0
    for i in T.serial(3):
        if i <= T.cast(T.if_then_else(previous, Dim[1], Dim[0]), "int32"):
            index = T.if_then_else(previous, T.cast(PrevVertices[i], "int32"), i)
            weight = T.if_then_else(previous, PrevProjection[i], Projection[i])
            for d in T.serial(3):
                Output[0, d] = Output[0, d] + Vertices[index, 1, d] * weight
                Output[1, d] = Output[1, d] + Vertices[index, 2, d] * weight


@T.macro
def gjk_result_intersection(Output, Status):
    Status[0] = T.uint32(0)
    for side in T.serial(3):
        for d in T.serial(3):
            Output[side, d] = 0.0


@T.macro
def gjk_result_closest_points(Points, Direction, Output, Status):
    Status[0] = T.uint32(1)
    for d in T.serial(3):
        Output[0, d] = Points[0, d]
        Output[1, d] = Points[1, d]
        Output[2, d] = Direction[d]


@T.macro
def gjk_result_proximity(Direction, Output, Status):
    Status[0] = T.uint32(2)
    for d in T.serial(3):
        Output[0, d] = 0.0
        Output[1, d] = 0.0
        Output[2, d] = Direction[d]


@T.macro
def gjk_result_no_intersection(Direction, Output, Status):
    Status[0] = T.uint32(3)
    for d in T.serial(3):
        Output[0, d] = 0.0
        Output[1, d] = 0.0
        Output[2, d] = Direction[d]


def cso_point_from_shapes(l):
    local = local_support_point(l)
    transformed = support_point(l)

    @T.macro
    def one(Shapes, ia, ib, ShapeVertices, Q, Translation, Direction, Output):
        a = T.alloc_local((3,), "float32")
        b = T.alloc_local((3,), "float32")
        opposite = T.alloc_local((3,), "float32")
        for d in T.serial(3):
            opposite[d] = -Direction[d]
        local(Shapes, ia, ShapeVertices, Direction, a)
        transformed(Shapes, ib, ShapeVertices, Q, Translation, opposite, b)
        from_points(a, b, Output)

    return one


def closest_points(l):
    support = cso_point_from_shapes(l)

    @T.macro
    def one(
        Shapes,
        ia,
        ib,
        ShapeVertices,
        Q,
        Translation,
        max_dist,
        exact_dist,
        Vertices,
        Projection,
        Dim,
        PrevVertices,
        PrevProjection,
        Output,
        Status,
    ):
        vertices = T.alloc_local((4, 3, 3), "float32")
        projection = T.alloc_local((3,), "float32")
        dims = T.alloc_local((2,), "uint32")
        indices = T.alloc_local((4,), "uint32")
        previous = T.alloc_local((3,), "float32")
        point = T.alloc_local((3,), "float32")
        old_dir = T.alloc_local((3,), "float32")
        direction = T.alloc_local((3,), "float32")
        cso = T.alloc_local((3, 3), "float32")
        points = T.alloc_local((2, 3), "float32")
        active = T.alloc_local((1,), "bool")
        publish = T.alloc_local((1,), "bool")
        bound = T.alloc_local((1,), "float32")
        for i in T.serial(4):
            for side in T.serial(3):
                for d in T.serial(3):
                    vertices[i, side, d] = Vertices[i, side, d]
        for i in T.serial(4):
            indices[i] = PrevVertices[i]
        for i in T.serial(2):
            dims[i] = Dim[i]
        for d in T.serial(3):
            projection[d] = Projection[d]
            previous[d] = PrevProjection[d]
            direction[d] = 0.0
        direction[0] = 1.0
        gjk_result_no_intersection(direction, Output, Status)
        project_origin_and_reduce(vertices, projection, dims, indices, previous, point)
        initial_length = T.sqrt(dot(point, point))
        active[0] = initial_length != 0.0
        publish[0] = False
        bound[0] = T.float32(l["query_constants"]["max_flt"])
        if initial_length == 0.0:
            gjk_result_intersection(Output, Status)
        else:
            for d in T.serial(3):
                old_dir[d] = -point[d] / initial_length
        iteration = T.alloc_local((1,), "int32")
        iteration[0] = 0
        while (iteration[0] < 100) & active[0]:
            iteration[0] = iteration[0] + 1
            if active[0]:
                old_bound = bound[0]
                length = T.sqrt(dot(point, point))
                if length > T.float32(1e-6):
                    for d in T.serial(3):
                        direction[d] = -point[d] / length
                    bound[0] = length
                else:
                    publish[0] = True
                    active[0] = False
                    gjk_result_intersection(Output, Status)
                if active[0]:
                    if bound[0] >= old_bound:
                        if exact_dist:
                            result(
                                vertices,
                                projection,
                                dims,
                                indices,
                                previous,
                                True,
                                points,
                            )
                            gjk_result_closest_points(points, old_dir, Output, Status)
                        else:
                            gjk_result_proximity(old_dir, Output, Status)
                        active[0] = False
                    else:
                        support(
                            Shapes,
                            ia,
                            ib,
                            ShapeVertices,
                            Q,
                            Translation,
                            direction,
                            cso,
                        )
                        minimum = -(
                            (direction[0] * cso[0, 0] + direction[1] * cso[0, 1])
                            + direction[2] * cso[0, 2]
                        )
                        if minimum > max_dist:
                            gjk_result_no_intersection(direction, Output, Status)
                            active[0] = False
                        elif (
                            (not exact_dist) & (minimum > 0.0) & (bound[0] <= max_dist)
                        ):
                            gjk_result_proximity(old_dir, Output, Status)
                            active[0] = False
                        elif bound[0] - minimum <= T.sqrt(T.float32(1e-6)) * bound[0]:
                            if exact_dist:
                                result(
                                    vertices,
                                    projection,
                                    dims,
                                    indices,
                                    previous,
                                    False,
                                    points,
                                )
                                gjk_result_closest_points(
                                    points, direction, Output, Status
                                )
                            else:
                                gjk_result_proximity(direction, Output, Status)
                            active[0] = False
                        else:
                            before = dims[0]
                            add_point(
                                cso, vertices, projection, dims, indices, previous
                            )
                            if before == dims[0]:
                                if exact_dist:
                                    result(
                                        vertices,
                                        projection,
                                        dims,
                                        indices,
                                        previous,
                                        False,
                                        points,
                                    )
                                    gjk_result_closest_points(
                                        points, direction, Output, Status
                                    )
                                else:
                                    gjk_result_proximity(direction, Output, Status)
                                active[0] = False
                            else:
                                for d in T.serial(3):
                                    old_dir[d] = direction[d]
                                project_origin_and_reduce(
                                    vertices, projection, dims, indices, previous, point
                                )
                                if dims[0] == 3:
                                    if minimum >= T.float32(1e-6):
                                        if exact_dist:
                                            result(
                                                vertices,
                                                projection,
                                                dims,
                                                indices,
                                                previous,
                                                True,
                                                points,
                                            )
                                            gjk_result_closest_points(
                                                points, old_dir, Output, Status
                                            )
                                        else:
                                            gjk_result_proximity(
                                                old_dir, Output, Status
                                            )
                                    else:
                                        publish[0] = True
                                        gjk_result_intersection(Output, Status)
                                    active[0] = False
        if publish[0]:
            for i in T.serial(4):
                for side in T.serial(3):
                    for d in T.serial(3):
                        Vertices[i, side, d] = vertices[i, side, d]
            for i in T.serial(4):
                PrevVertices[i] = indices[i]
            for i in T.serial(2):
                Dim[i] = dims[i]
            for d in T.serial(3):
                Projection[d] = projection[d]
                PrevProjection[d] = previous[d]

    return one
