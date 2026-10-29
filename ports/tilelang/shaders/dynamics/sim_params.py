"""Counterpart of dynamics/sim_params.rs; contact regularization coefficients."""

import tilelang.language as T
from kernel_support import f32


def contact_coefficients(layout):
    fields = layout["rbd_params"]

    @T.macro
    def compute(RbdParams, Coefficients):
        dt = f32(RbdParams, fields["dt"])
        Coefficients[0] = T.if_then_else(dt == 0, T.float32(0), T.float32(1) / dt)
        normalized = f32(RbdParams, fields["normalized_max_corrective_velocity"])
        Coefficients[1] = T.if_then_else(
            RbdParams[fields["normalized_max_corrective_velocity"]]
            == T.uint32(layout["constants"]["max_flt_bits"]),
            normalized,
            normalized * f32(RbdParams, fields["length_unit"]),
        )
        for side in T.unroll(2, explicit=True):
            frequency = f32(
                RbdParams,
                T.if_then_else(
                    side == 0,
                    fields["contact_natural_frequency"],
                    fields["static_contact_natural_frequency"],
                ),
            )
            damping = f32(
                RbdParams,
                T.if_then_else(
                    side == 0,
                    fields["contact_damping_ratio"],
                    fields["static_contact_damping_ratio"],
                ),
            )
            omega = frequency * T.reinterpret(
                T.uint32(layout["constants"]["two_pi_bits"]), "float32"
            )
            Coefficients[3 + side * 2] = omega / (dt * omega + T.float32(2) * damping)
            erp = dt * Coefficients[3 + side * 2]
            Coefficients[2 + side * 2] = T.float32(0)
            if erp != 0:
                inv_erp_minus_one = T.float32(1) / erp - T.float32(1)
                coefficient = (
                    inv_erp_minus_one
                    * inv_erp_minus_one
                    / (
                        (T.float32(1) + inv_erp_minus_one)
                        * T.float32(4)
                        * damping
                        * damping
                    )
                )
                Coefficients[2 + side * 2] = T.float32(1) / (T.float32(1) + coefficient)

    return compute


def body_coefficients(layout):
    f = layout["rbd_params"]

    @T.macro
    def one(RbdParams, Gravity, Params):
        dt = f32(RbdParams, f["dt"])
        inv_dt = T.if_then_else(dt == 0, T.float32(0), T.float32(1) / dt)
        v = f32(RbdParams, f["normalized_max_linear_velocity"])
        Params[0] = RbdParams[f["dt"]]
        Params[1] = T.reinterpret(
            T.if_then_else(
                RbdParams[f["normalized_max_linear_velocity"]]
                == T.uint32(layout["constants"]["max_flt_bits"]),
                v,
                v * f32(RbdParams, f["length_unit"]),
            ),
            "uint32",
        )
        Params[2] = T.reinterpret(
            T.reinterpret(T.uint32(layout["constants"]["pi_over_four_bits"]), "float32")
            * inv_dt
            / T.cast(
                T.max(RbdParams[f["num_solver_iterations"]], T.uint32(1)), "float32"
            ),
            "uint32",
        )
        for d in T.serial(3):
            Params[3 + d] = Gravity[d]

    return one


def joint_coefficients(l):
    f = l["rbd_params"]

    @T.macro
    def one(Params, Result):
        dt = f32(Params, f["dt"])
        damping = f32(Params, f["joint_damping_ratio"])
        omega = f32(Params, f["joint_natural_frequency"]) * T.reinterpret(
            T.uint32(l["constants"]["two_pi_bits"]), "float32"
        )
        Result[0] = omega / (dt * omega + 2.0 * damping)
        erp = dt * Result[0]
        Result[1] = 0.0
        if erp != 0.0:
            inv = 1.0 / erp - 1.0
            Result[1] = inv * inv / ((1.0 + inv) * 4.0 * damping * damping)
        Result[2] = T.if_then_else(dt == 0.0, 0.0, 1.0 / dt)

    return one
