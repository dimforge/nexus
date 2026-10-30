"""Counterpart of queries/projection.rs location constructors."""

import tilelang.language as T


@T.macro
def location(Point, Bary, kind, index, inside, Values, Flags):
    for d in T.serial(3):
        Values[d] = Point[d]
        Values[d + 3] = Bary[d]
    Flags[0] = kind
    Flags[1] = index
    Flags[2] = T.cast(inside, "uint32")


@T.macro
def vertex(Point, index, inside, Values, Flags):
    for d in T.serial(3):
        Values[d] = Point[d]
        Values[d + 3] = 0.0
    Flags[0] = T.uint32(0)
    Flags[1] = index
    Flags[2] = T.cast(inside, "uint32")


@T.macro
def edge(Point, first, second, index, inside, Values, Flags):
    for d in T.serial(3):
        Values[d] = Point[d]
    Values[3] = first
    Values[4] = second
    Values[5] = 0.0
    Flags[0] = T.uint32(1)
    Flags[1] = index
    Flags[2] = T.cast(inside, "uint32")


@T.macro
def solid(Point, Values, Flags):
    for d in T.serial(3):
        Values[d] = Point[d]
        Values[d + 3] = 0.0
    Flags[0] = T.uint32(3)
    Flags[1] = T.uint32(0)
    Flags[2] = T.uint32(1)


@T.macro
def barycentric_coordinates(Values, Flags, Bary):
    for d in T.serial(3):
        Bary[d] = 0.0
    if Flags[0] == 0:
        if Flags[1] < 3:
            Bary[T.cast(Flags[1], "int32")] = 1.0
    elif Flags[0] == 1:
        if Flags[1] == 0:
            Bary[0] = Values[3]
            Bary[1] = Values[4]
        elif Flags[1] == 1:
            Bary[1] = Values[3]
            Bary[2] = Values[4]
        elif Flags[1] == 2:
            Bary[0] = Values[3]
            Bary[2] = Values[4]
    elif Flags[0] == 2:
        for d in T.serial(3):
            Bary[d] = Values[d + 3]
