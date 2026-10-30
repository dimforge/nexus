"""Independent all-frame support, containment and energy checks for general scenes."""

import numpy as np
from deferred_validation import vertices
from scene_physics import state
from support_geometry import inverse, rotate, support_value


def primitives(l, shape, v, indices):
    sf, types = l["shape"], l["shape_types"]
    kind = int(shape[sf["a"] + 3])
    if kind not in (types["trimesh"], types["polyline"]):
        return [shape]
    vb, ib, n = map(int, shape[sf["a"] : sf["a"] + 3])
    rows = indices[ib : ib + 3 * n].reshape(-1, 3)
    result = []
    for entry, _, part in rows:
        if entry != 0xFFFFFFFF:
            continue
        width = 2 if kind == types["polyline"] else 3
        start = ib + 3 * n + int(part) * width
        points = v[vb + 2 * n + indices[start : start + width].astype(int)]
        s = np.zeros(l["shape_words"], np.uint32)
        f = s.view(np.float32)
        for field, p in zip(("a", "b", "c"), points):
            f[sf[field] : sf[field] + 3] = p
        s[sf["a"] + 3] = types["capsule"] if width == 2 else types["triangle"]
        if width == 2:
            f[sf["b"] + 3] = np.float32(0.4)
        result.append(s)
    return result


def check(l, c, b, r, k):
    n, nb = c["body_count"], c["batches"]
    pw, vw = l["pose_words"], l["velocity_words"]
    stride = c["trace_stride"]
    shapes = b["Shapes"].reshape(-1, l["shape_words"])
    vtx = vertices(l, b)
    initial, _, iv, iw = state(l, c, b)
    dynamic = np.arange(nb, n)
    inertia = 1 / 6
    energy = lambda p, v, w: float(
        np.sum(
            0.5 * np.sum(v[dynamic] ** 2, axis=1)
            + 0.5 * inertia * np.sum(w[dynamic] ** 2, axis=1)
            + 9.81 * p[dynamic, 1]
        )
    )
    e0 = energy(initial, iv, iw)
    upper = (
        e0 * (1 + k["physics"]["energy_relative_allowance"])
        + len(dynamic) * k["physics"]["energy_per_body_allowance"]
    )
    max_penetration = max_energy = max_speed = max_spin = 0.0
    for frame in range(c["frames"]):
        off = frame * stride
        step = dict(b)
        step["Poses"] = b["Poses"].copy()
        step["Velocities"] = b["Velocities"].copy()
        step["Poses"][: n * pw] = r["Trajectory"][off : off + n * pw]
        step["Velocities"][: n * vw] = r["Trajectory"][off + n * pw : off + stride]
        p, q, v, w = state(l, c, step)
        for name, width in (("Poses", pw), ("Velocities", vw)):
            np.testing.assert_array_equal(
                step[name][: nb * width],
                b[name][: nb * width],
                err_msg="General scene static body changed",
            )
        penetration = 0.0
        for i in dynamic:
            # Every fixture ground is a finite horizontal box/mesh or two thick
            # line segments. Evaluate vertical support in f64 from native inputs.
            ground = i % nb
            down = inverse(q[i], np.array([0.0, -1.0, 0.0]))
            bottom = p[i, 1] - support_value(l, shapes[i], vtx, down)
            if c["scene"] == "general_polyline":
                # Ball vs both finite capsule segments: exact geometric distance.
                for seg in primitives(l, shapes[ground], vtx, b["GeometryIndices"]):
                    f = seg.view(np.float32)
                    sf = l["shape"]
                    a = p[ground] + rotate(q[ground], f[sf["a"] : sf["a"] + 3])
                    bb = p[ground] + rotate(q[ground], f[sf["b"] : sf["b"] + 3])
                    delta = bb - a
                    t = np.clip(np.dot(p[i] - a, delta) / np.dot(delta, delta), 0, 1)
                    penetration = max(
                        penetration, 0.9 - float(np.linalg.norm(p[i] - (a + t * delta)))
                    )
            elif (
                abs(p[i, 0] - p[ground, 0]) < 19.5
                and abs(p[i, 2] - p[ground, 2]) < 19.5
            ):
                penetration = max(penetration, -float(bottom))
            # The only stacked general fixture consists of convex cubes.
            if c["scene"] == "general_convex":
                for j in dynamic:
                    if j <= i or j % nb != i % nb:
                        continue
                    if abs(p[j, 0] - p[i, 0]) < 0.8 and abs(p[j, 2] - p[i, 2]) < 0.8:
                        hi = p[i, 1] + support_value(
                            l, shapes[i], vtx, inverse(q[i], np.array([0.0, 1.0, 0.0]))
                        )
                        lo = p[j, 1] - support_value(
                            l, shapes[j], vtx, inverse(q[j], np.array([0.0, -1.0, 0.0]))
                        )
                        penetration = max(penetration, hi - lo)
        total = energy(p, v, w)
        if penetration > k["physics"]["max_penetration"]:
            raise AssertionError(
                f"General scene penetration at frame {frame + 1}: {penetration}"
            )
        if total > upper:
            raise AssertionError(
                f"General scene excess energy at frame {frame + 1}: {total} > {upper}"
            )
        max_penetration = max(max_penetration, penetration)
        max_energy = max(max_energy, total)
        max_speed = max(
            max_speed, float(np.max(np.linalg.norm(v[dynamic], axis=1), initial=0))
        )
        max_spin = max(
            max_spin, float(np.max(np.linalg.norm(w[dynamic], axis=1), initial=0))
        )
    return {
        "maximum_penetration": max_penetration,
        "maximum_energy": max_energy,
        "maximum_speed": max_speed,
        "maximum_spin": max_spin,
    }
