"""Counterpart of queries/contact.rs; ball/primitive analytic contacts and materials."""

import tilelang.language as T
from kernel_support import dot, quat_rotate

from ..bounding_volumes.aabb import absolute
from ..shapes.shape import cuboid_support_face, project_local_point_on_boundary
from . import sat
from .polygonal_feature import contacts_face_face


def combine_coefficient(a, b, rule1, rule2):
    rule = T.max(T.cast(rule1, "int64"), T.cast(rule2, "int64"))
    return T.if_then_else(
        rule == T.int64(1),
        absolute(T.min(a, b)),
        T.if_then_else(
            rule == T.int64(2),
            a * b,
            T.if_then_else(
                rule == T.int64(3),
                T.max(a, b),
                T.if_then_else(
                    rule == T.int64(4),
                    T.max(T.float32(0), T.min(T.float32(1), a + b)),
                    (a + b) * T.float32(0.5),
                ),
            ),
        ),
    )


@T.macro
def ball_ball(translation, radius1, radius2, point, normal, distance):
    length = T.sqrt(dot(translation, translation))
    normal[0], normal[1], normal[2] = T.float32(0), T.float32(1), T.float32(0)
    if length != T.float32(0):
        for d in T.serial(3):
            normal[d] = translation[d] / length
    for d in T.serial(3):
        point[d] = normal[d] * radius1
    distance[0] = length - (radius1 + radius2)


def convex_ball(layout):
    project = project_local_point_on_boundary(layout)

    @T.macro
    def one(Shapes, index, translation, radius2, point, normal, distance):
        inside = T.alloc_local((1,), "bool")
        project(Shapes, index, translation, point, inside)
        delta = T.alloc_local((3,), "float32")
        for d in T.serial(3):
            delta[d] = translation[d] - point[d]
        length = T.sqrt(dot(delta, delta))
        normal[0], normal[1], normal[2] = T.float32(0), T.float32(1), T.float32(0)
        if length != T.float32(0):
            for d in T.serial(3):
                normal[d] = delta[d] / length
        distance[0] = length
        if inside[0]:
            for d in T.serial(3):
                normal[d] = -normal[d]
            distance[0] = -distance[0]
        distance[0] = distance[0] - radius2

    return one


def ball_convex(layout):
    convex = convex_ball(layout)

    @T.macro
    def one(Shapes, index, rotation, translation, radius1, point, normal, distance):
        inverse_q = T.alloc_local((4,), "float32")
        neg_t = T.alloc_local((3,), "float32")
        inverse_t = T.alloc_local((3,), "float32")
        for d in T.serial(3):
            inverse_q[d] = -rotation[d]
            neg_t[d] = -translation[d]
        inverse_q[3] = rotation[3]
        quat_rotate(inverse_q, neg_t, inverse_t)
        normal2 = T.alloc_local((3,), "float32")
        point2 = T.alloc_local((3,), "float32")
        rotated = T.alloc_local((3,), "float32")
        convex(Shapes, index, inverse_t, radius1, point2, normal2, distance)
        quat_rotate(rotation, normal2, rotated)
        for d in T.serial(3):
            normal[d] = -rotated[d]
            point[d] = normal[d] * radius1

    return one


@T.macro
def cuboid_cuboid(q12, t12, he1, he2, prediction, points, normal, length):
    q21 = T.alloc_local((4,), "float32")
    neg_t, t21 = T.alloc_local((3,), "float32"), T.alloc_local((3,), "float32")
    for d in T.serial(3):
        q21[d] = -q12[d]
        neg_t[d] = -t12[d]
    q21[3] = q12[3]
    quat_rotate(q21, neg_t, t21)
    axis1, axis2, axis3 = (
        T.alloc_local((3,), "float32"),
        T.alloc_local((3,), "float32"),
        T.alloc_local((3,), "float32"),
    )
    sep1, sep2, sep3 = (
        T.alloc_local((1,), "float32"),
        T.alloc_local((1,), "float32"),
        T.alloc_local((1,), "float32"),
    )
    length[0] = 0
    for i in T.serial(4):
        for d in T.serial(4):
            points[i, d] = T.float32(0)
    for d in T.serial(3):
        normal[d] = T.float32(0)
    sat.cuboid_cuboid_find_local_separating_normal_oneway(
        he1, he2, q12, t12, axis1, sep1
    )
    if sep1[0] <= prediction:
        sat.cuboid_cuboid_find_local_separating_normal_oneway(
            he2, he1, q21, t21, axis2, sep2
        )
        if sep2[0] <= prediction:
            sat.cuboid_cuboid_find_local_separating_edge_twoway(
                he1, he2, q12, t12, axis3, sep3
            )
            if sep3[0] <= prediction:
                for d in T.serial(3):
                    normal[d] = axis1[d]
                if sep2[0] > sep1[0] and sep2[0] > sep3[0]:
                    neg = T.alloc_local((3,), "float32")
                    for d in T.serial(3):
                        neg[d] = -axis2[d]
                    quat_rotate(q12, neg, normal)
                elif sep3[0] > sep1[0]:
                    for d in T.serial(3):
                        normal[d] = axis3[d]
                neg_normal, normal2 = (
                    T.alloc_local((3,), "float32"),
                    T.alloc_local((3,), "float32"),
                )
                for d in T.serial(3):
                    neg_normal[d] = -normal[d]
                quat_rotate(q21, neg_normal, normal2)
                face1, face2 = (
                    T.alloc_local((4, 3), "float32"),
                    T.alloc_local((4, 3), "float32"),
                )
                cuboid_support_face(he1, normal, face1)
                cuboid_support_face(he2, normal2, face2)
                contacts_face_face(
                    q12,
                    t12,
                    face1,
                    normal,
                    face2,
                    4,
                    4,
                    prediction,
                    points,
                    length,
                    False,
                )
