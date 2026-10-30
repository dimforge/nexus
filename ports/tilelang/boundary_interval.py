"""Outward f32 arithmetic intervals for the two quaternion transform identities."""

import math

U = 2.0**-24
TINY = 2.0**-149


class Interval:
    def __init__(self, lo, hi=None):
        self.lo = float(lo)
        self.hi = self.lo if hi is None else float(hi)

    @classmethod
    def rounded(cls, lo, hi):
        error = max(abs(lo), abs(hi)) * U / (1 - U) + TINY
        return cls(
            math.nextafter(lo - error, -math.inf), math.nextafter(hi + error, math.inf)
        )

    def __add__(self, b):
        if not isinstance(b, Interval):
            b = Interval(b)
        return self.rounded(self.lo + b.lo, self.hi + b.hi)

    def __neg__(self):
        return Interval(-self.hi, -self.lo)

    def __sub__(self, b):
        return self + -b

    def __mul__(self, b):
        if not isinstance(b, Interval):
            b = Interval(b)
        products = [x * y for x in [self.lo, self.hi] for y in [b.lo, b.hi]]
        return self.rounded(min(products), max(products))

    def union(self, b):
        return Interval(min(self.lo, b.lo), max(self.hi, b.hi))


def dot(a, b):
    p = [x * y for x, y in zip(a, b)]
    # Include both association orders used by scalar/SIMD three-term reductions.
    return ((p[0] + p[1]) + p[2]).union(p[0] + (p[1] + p[2]))


def cross(a, b):
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def rotate(q, v):
    b = q[:3]
    w = q[3]
    uv = cross(b, v)
    uuv = cross(b, uv)
    first = [v[i] + (uv[i] * w + uuv[i]) * 2 for i in range(3)]
    twice = [x * 2 for x in uv]
    last = cross(b, twice)
    second = [v[i] + w * twice[i] + last[i] for i in range(3)]
    factor = w * w - dot(b, b)
    dp = dot(b, v)
    third = [v[i] * factor + b[i] * dp * 2 + uv[i] * w * 2 for i in range(3)]
    return [a.union(b).union(c) for a, b, c in zip(first, second, third)]


def separation(q1, t1, p1, q2, t2, p2, normal, dist):
    q1, t1, p1, q2, t2, p2, normal = (
        [Interval(x) for x in a] for a in (q1, t1, p1, q2, t2, p2, normal)
    )
    a = [x + y for x, y in zip(rotate(q1, p1), t1)]
    b = [x + y for x, y in zip(rotate(q2, p2), t2)]
    return Interval(dist) + dot([x - y for x, y in zip(a, b)], normal)


def mask(l, c, b):
    import numpy as np

    result = []
    maxwidth = 0
    maxambiguous = 0
    h, p = l["header"], l["point"]
    tiles = b["Tiles"]
    tf = tiles.view(np.float32)
    poses = b["Poses"].view(np.float32).reshape(-1, l["pose_words"])
    for i in range(c["count"]):
        base, off = (i // 32) * l["tile_words"], (i % 32) * 4
        normal = tf[base + h["normal"] + off : base + h["normal"] + off + 3]
        count = int(tiles[base + h["normal"] + off + 3])
        pa, pb = [
            poses[int(tiles[base + h[k] + off + 3])] for k in ("body_a", "body_b")
        ]
        m = 0
        for j in range(count):
            point = base + l["points"] + j * l["point_words"] + off
            # The signed distance is the scalar of local_a in the native tile.
            aa = tf[point + p["local_a"] : point + p["local_a"] + 3]
            bb = tf[point + p["local_b"] : point + p["local_b"] + 3]
            distance = tf[point + p["local_a"] + 3]
            bounds = separation(
                pa[l["pose_rotation"] : l["pose_rotation"] + 4],
                pa[l["pose_translation"] : l["pose_translation"] + 3],
                aa,
                pb[l["pose_rotation"] : l["pose_rotation"] + 4],
                pb[l["pose_translation"] : l["pose_translation"] + 3],
                bb,
                normal,
                distance,
            )
            if not math.isfinite(bounds.lo) or not math.isfinite(bounds.hi):
                raise ValueError("Nonfinite boundary distance interval")
            if bounds.lo <= 0 <= bounds.hi:
                m |= 1 << j
            maxwidth = max(maxwidth, bounds.hi - bounds.lo)
        result.append(m)
        maxambiguous = max(maxambiguous, m.bit_count())
    return np.array(result, np.uint32), maxwidth, maxambiguous
