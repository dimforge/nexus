"""Counterpart of dynamics/solver.rs::gpu_integrate_linearized."""

import tilelang.language as T
from kernel_support import dot, f32, launch
from .body import integrate_linearized


def gpu_integrate_linearized(layout, case, target):
    n = case["active_bodies"]
    integrate = integrate_linearized(layout)

    @T.macro
    def one(Tiles, Velocities, Poses, Params, i):
        if i < n:
            vp = i * layout["velocity_words"]
            v = T.alloc_local((3,), "float32")
            w = T.alloc_local((3,), "float32")
            for d in T.serial(3):
                v[d] = f32(Velocities, vp + layout["velocity_linear"] + d)
                w[d] = f32(Velocities, vp + layout["velocity_angular"] + d)
            vl = T.sqrt(dot(v, v))
            wl = T.sqrt(dot(w, w))
            if vl > Params[1]:
                for d in T.serial(3):
                    v[d] = v[d] * (Params[1] / vl)
            if wl > Params[2]:
                for d in T.serial(3):
                    w[d] = w[d] * (Params[2] / wl)
            for d in T.serial(3):
                Velocities[vp + layout["velocity_linear"] + d] = T.reinterpret(
                    v[d], "uint32"
                )
                Velocities[vp + layout["velocity_angular"] + d] = T.reinterpret(
                    w[d], "uint32"
                )
            integrate(Poses, i, v, w, Params[0])

    return launch(one, layout, case, target, n)
