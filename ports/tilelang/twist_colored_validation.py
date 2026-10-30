"""Compact twist state checks within the shared native color schedule validator."""


def validate_tile(l, arrays, b, n):
    import numpy as np
    from twist_contact_validation import validate_old

    validate_old(l, arrays["Tiles"], b, n)
    f = arrays["Tiles"].view(np.float32)
    h = l["header"]
    for offset in h.values():
        if not np.isfinite(f[b + offset : b + offset + 3]).all():
            raise ValueError("Nonfinite twist header")
    normal = f[b + h["normal"] : b + h["normal"] + 3]
    tangent = f[b + h["tangent"] : b + h["tangent"] + 3]
    if (
        abs(np.linalg.norm(normal) - 1) > 2e-6
        or abs(np.linalg.norm(tangent) - 1) > 2e-6
        or abs(np.dot(normal, tangent)) > 2e-6
    ):
        raise ValueError("Invalid twist tangent frame")
    if (
        not np.isfinite(f[b + h["diag_b"] + 3])
        or not np.isfinite(f[b + h["tangent"] + 3])
        or f[b + h["tangent"] + 3] < 0
    ):
        raise ValueError("Invalid twist effective mass or friction")


def mark_impulses(l, b, count, friction, mask):
    t = l["twist"]
    mask[b + t["normal_impulse"] : b + t["normal_impulse"] + count] = True
    if count and friction:
        mask[b + t["friction"] : b + t["friction"] + 2] = True
        mask[b + l["header"]["off_b"] + 3] = True


def check_bounds(l, tiles, b, count, friction, contract):
    import numpy as np

    t, h = l["twist"], l["header"]
    f = tiles.view(np.float32).astype(float)
    impulses = f[b + t["normal_impulse"] : b + t["normal_impulse"] + count]
    if np.any(impulses < 0):
        raise AssertionError("Negative twist normal impulse")
    if count and friction:
        coefficient = f[b + h["tangent"] + 3]
        tangent_limit = np.sum(impulses) * coefficient
        twist_limit = (
            np.dot(impulses, f[b + t["radius"] : b + t["radius"] + count]) * coefficient
        )
        twist = f[b + h["off_b"] + 3]
        for actual, bound in [
            (
                np.linalg.norm(f[b + t["friction"] : b + t["friction"] + 2]),
                tangent_limit,
            ),
            (abs(twist), twist_limit),
        ]:
            if actual > bound + contract["atol"] + contract["rtol"] * abs(bound):
                raise AssertionError("Twist manifold friction limit")
        if count == 1 and twist != 0:
            raise AssertionError("Single-point twist impulse")
