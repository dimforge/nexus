"""Counterpart of dynamics/joint.rs motor coefficient construction."""

import tilelang.language as T
from kernel_support import f32


def motor_params(l):
    m = l["motor"]
    p = l["rbd_params"]

    @T.macro
    def one(Joints, base, Params, Output):
        dt = f32(Params, p["dt"])
        stiffness = f32(Joints, base + m["stiffness"])
        damping = f32(Joints, base + m["damping"])
        denom = dt * stiffness + damping
        inverse = T.if_then_else(denom == 0.0, 0.0, 1.0 / denom)
        cfmdenom = dt * dt * stiffness + dt * damping
        cfm = T.if_then_else(cfmdenom == 0.0, 0.0, 1.0 / cfmdenom)
        Output[0] = stiffness * inverse
        Output[1] = T.if_then_else(Joints[base + m["model"]] == 0, cfm, 0.0)
        Output[2] = T.if_then_else(Joints[base + m["model"]] == 0, 0.0, cfm)
        Output[3] = f32(Joints, base + m["target_pos"])
        Output[4] = f32(Joints, base + m["target_vel"])
        Output[5] = f32(Joints, base + m["max_force"]) * dt

    return one
