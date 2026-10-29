"""Counterpart of src_rbd_shaders/dynamics/constraint.rs::compute_effective_masses."""

import tilelang.language as T
from kernel_support import dot, f32


def compute_effective_masses(layout, case):
    n = case["count"]
    h, p = (layout["header"], layout["point"])

    @T.macro
    def cross(a, b, out):
        out[0] = a[1] * b[2] - a[2] * b[1]
        out[1] = a[2] * b[0] - a[0] * b[2]
        out[2] = a[0] * b[1] - a[1] * b[0]

    @T.macro
    def inertia(diag, off, v, out):
        out[0] = diag[0] * v[0] + off[0] * v[1] + off[1] * v[2]
        out[1] = off[0] * v[0] + diag[1] * v[1] + off[2] * v[2]
        out[2] = off[1] * v[0] + off[2] * v[1] + diag[2] * v[2]

    @T.macro
    def one(Tiles, Velocities, Poses, Params, i):
        if i < n:
            base = i // 32 * layout["tile_words"]
            lane = i % 32
            lane4 = lane * 4
            count = Tiles[base + h["normal"] + lane4 + 3]
            direction = T.alloc_local((3,), "float32")
            tangent0 = T.alloc_local((3,), "float32")
            tangent1 = T.alloc_local((3,), "float32")
            im_a = T.alloc_local((3,), "float32")
            im_b = T.alloc_local((3,), "float32")
            diag_a = T.alloc_local((3,), "float32")
            diag_b = T.alloc_local((3,), "float32")
            off_a = T.alloc_local((3,), "float32")
            off_b = T.alloc_local((3,), "float32")
            for d in T.serial(3):
                direction[d] = f32(Tiles, base + h["normal"] + lane4 + d)
                tangent0[d] = f32(Tiles, base + h["tangent"] + lane4 + d)
                im_a[d] = f32(Tiles, base + h["body_a"] + lane4 + d)
                im_b[d] = f32(Tiles, base + h["body_b"] + lane4 + d)
                diag_a[d] = f32(Tiles, base + h["diag_a"] + lane4 + d)
                diag_b[d] = f32(Tiles, base + h["diag_b"] + lane4 + d)
                off_a[d] = f32(Tiles, base + h["off_a"] + lane4 + d)
                off_b[d] = f32(Tiles, base + h["off_b"] + lane4 + d)
            cross(direction, tangent0, tangent1)
            for k in T.serial(4):
                if k < count:
                    pb = base + layout["points"] + k * layout["point_words"] + lane4
                    ra = T.alloc_local((3,), "float32")
                    rb = T.alloc_local((3,), "float32")
                    ta = T.alloc_local((3,), "float32")
                    tb = T.alloc_local((3,), "float32")
                    ia = T.alloc_local((3,), "float32")
                    ib = T.alloc_local((3,), "float32")
                    impulse = T.alloc_local((3,), "float32")
                    for d in T.serial(3):
                        ra[d] = f32(Tiles, pb + p["arm_a"] + d)
                        rb[d] = f32(Tiles, pb + p["arm_b"] + d)
                    masses = T.alloc_local((3,), "float32")
                    ta0 = T.alloc_local((3,), "float32")
                    tb0 = T.alloc_local((3,), "float32")
                    for axis in T.serial(3):
                        for d in T.serial(3):
                            impulse[d] = direction[d]
                            if axis == 1:
                                impulse[d] = tangent0[d]
                            if axis == 2:
                                impulse[d] = tangent1[d]
                        cross(ra, impulse, ta)
                        cross(rb, impulse, tb)
                        inertia(diag_a, off_a, ta, ia)
                        inertia(diag_b, off_b, tb, ib)
                        lin = (
                            impulse[0] * (im_a[0] + im_b[0]) * impulse[0]
                            + impulse[1] * (im_a[1] + im_b[1]) * impulse[1]
                            + impulse[2] * (im_a[2] + im_b[2]) * impulse[2]
                        )
                        masses[axis] = lin + dot(ia, ta) + dot(ib, tb)
                        if axis == 1:
                            for d in T.serial(3):
                                ta0[d] = ta[d]
                                tb0[d] = tb[d]
                        if axis == 2:
                            Tiles[base + layout["coupling"] + lane4 + k] = (
                                T.reinterpret(
                                    2.0 * (dot(ta0, ia) + dot(tb0, ib)), "uint32"
                                )
                            )
                    if masses[0] != 0:
                        masses[0] = 1.0 / masses[0]
                    Tiles[pb + p["arm_b"] + 3] = T.reinterpret(masses[0], "uint32")
                    Tiles[pb + p["friction"] + 2] = T.reinterpret(masses[1], "uint32")
                    Tiles[pb + p["friction"] + 3] = T.reinterpret(masses[2], "uint32")

    return one
