"""Independent geometric and motion bounds for explicitly named native fixtures."""

import numpy as np
from deferred_validation import vertices
from general_scene_physics import primitives
from joint_scene_physics import multiply, rotate
from support_geometry import inverse, support_value


def check(l, c, b, r, k, frame_indices=None):
    policy = k.get("runtime_physics_cases", {}).get(c["name"])
    if policy is None:
        return {}
    if any(c.get(key) != value for key, value in policy["configuration"].items()):
        raise ValueError("Runtime physics configuration mismatch")
    n, nb = c["body_count"], c["batches"]
    pw, vw = l["pose_words"], l["velocity_words"]
    stride = c["trace_stride"]
    sf, jf, gf = l["shape"], l["impulse_joint"], l["generic_joint"]
    shapes = b["Shapes"].reshape(-1, l["shape_words"])[: c["collider_count"]]
    local = b["LocalPoses"].view(np.float32).reshape(-1, pw).astype(float)
    initial = b["Poses"].view(np.float32).reshape(-1, pw).astype(float)
    mass = b["LocalMass"].view(np.float32).reshape(-1, l["local_words"])
    dynamic = np.any(
        mass[:n, l["local_inv_mass"] : l["local_inv_mass"] + 3] != 0, axis=1
    )
    vtx = vertices(l, b)
    floors = {}
    # These fixtures contain one fixed, horizontal triangle mesh per batch.
    # Reconstruct its world plane from real mesh vertices and parent/local poses.
    for i, shape in enumerate(shapes):
        if shape[sf["a"] + 3] != l["shape_types"]["trimesh"]:
            continue
        body = int(b["Parents"][i])
        if dynamic[body]:
            raise AssertionError("Runtime diagnostic mesh must remain fixed")
        pose = initial[body]
        q = multiply(pose[l["pose_rotation"] : l["pose_rotation"] + 4], local[i, :4])
        t = pose[l["pose_translation"] : l["pose_translation"] + 3] + rotate(
            pose[:4], local[i, l["pose_translation"] : l["pose_translation"] + 3]
        )
        heights = []
        for triangle in primitives(l, shape, vtx, b["GeometryIndices"]):
            values = triangle.view(np.float32)
            heights.extend(
                float((t + rotate(q, values[sf[field] : sf[field] + 3]))[1])
                for field in ("a", "b", "c")
            )
        if not heights or not np.isfinite(heights).all() or np.ptp(heights) > 1e-6:
            raise AssertionError("Runtime diagnostic floor must be horizontal")
        floors[i % nb] = min(heights)
    if set(floors) != set(range(nb)):
        raise AssertionError("Missing runtime diagnostic floor")
    metrics = {
        key: 0.0
        for key in ("anchor", "rotation", "limit", "penetration", "speed", "spin")
    }
    frame_indices = range(c["frames"]) if frame_indices is None else frame_indices
    for index in frame_indices:
        trace = (
            r["Trajectory"][index * stride : (index + 1) * stride]
            .view(np.float32)
            .astype(float)
        )
        if not np.isfinite(trace).all():
            raise AssertionError("Nonfinite runtime trajectory")
        poses, vels = trace[: n * pw].reshape(n, pw), trace[n * pw :].reshape(n, vw)
        q = poses[:, l["pose_rotation"] : l["pose_rotation"] + 4]
        position = poses[:, l["pose_translation"] : l["pose_translation"] + 3]
        for metric, field in (
            ("speed", "velocity_linear"),
            ("spin", "velocity_angular"),
        ):
            metrics[metric] = max(
                metrics[metric],
                float(
                    np.max(
                        np.linalg.norm(vels[:, l[field] : l[field] + 3], axis=1),
                        initial=0,
                    )
                ),
            )
        for i, shape in enumerate(shapes):
            body = int(b["Parents"][i])
            if not dynamic[body]:
                continue
            world_q = multiply(q[body], local[i, :4])
            world_p = position[body] + rotate(
                q[body], local[i, l["pose_translation"] : l["pose_translation"] + 3]
            )
            bottom = world_p[1] - support_value(
                l, shape, vtx, inverse(world_q, np.array([0.0, -1.0, 0.0]))
            )
            metrics["penetration"] = max(
                metrics["penetration"], floors[i % nb] - float(bottom)
            )
        for i, row in enumerate(
            b["Joints"].reshape(-1, l["impulse_joint_words"])[: c["joint_count"]]
        ):
            data = row[jf["data"] : jf["data"] + l["generic_joint_words"]]
            values = data.view(np.float32).astype(float)
            frames = []
            for name, field in (
                ("local_frame_a", "body_a"),
                ("local_frame_b", "body_b"),
            ):
                body = int(row[jf[field]]) * nb + i % nb
                off = gf[name]
                frames.append(
                    (
                        multiply(q[body], values[off : off + 4]),
                        position[body]
                        + rotate(
                            q[body],
                            values[
                                off + l["pose_translation"] : off
                                + l["pose_translation"]
                                + 3
                            ],
                        ),
                    )
                )
            qa, pa = frames[0]
            qb, pb = frames[1]
            inv = np.r_[-qa[:3], qa[3]]
            linear, angular = rotate(inv, pb - pa), multiply(inv, qb)
            locked, limited = int(data[gf["locked_axes"]]), int(data[gf["limit_axes"]])
            for axis in range(3):
                if locked & (1 << axis):
                    metrics["anchor"] = max(metrics["anchor"], abs(float(linear[axis])))
                if locked & (1 << (axis + 3)):
                    metrics["rotation"] = max(
                        metrics["rotation"], abs(float(angular[axis]))
                    )
                if limited & (1 << axis):
                    off = gf["limits"] + axis * l["limits_words"]
                    metrics["limit"] = max(
                        metrics["limit"],
                        float(values[off + l["limits"]["min"]] - linear[axis]),
                        float(linear[axis] - values[off + l["limits"]["max"]]),
                    )
        for field, value in metrics.items():
            if value > policy["bounds"][field]:
                raise AssertionError(f"Runtime {field} at frame {index + 1}: {value}")
    return metrics
