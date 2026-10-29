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
