"""Validate native packed geometry, including unused empty BVH leaf bounds."""

import numpy as np
from deferred_validation import mesh_indices, vertices
from support_validation import validate_shapes


def lbvh_vertices(l, buffers):
    """Return a validation-only view with proven-unused leaf bounds normalized.

    Rust's bvh::BVH::flatten writes [+inf, -inf] empty bounds for leaves. Native
    traversal branches on entry == u32::MAX and reads primitive indices instead
    of those bounds. Shape and primitive geometry must still be finite. Kernel
    inputs and output read-only comparisons retain all original words.
    """
    shapes = buffers["Shapes"].reshape(-1, l["shape_words"])
    v = vertices(l, buffers)
    sf, kinds = l["shape"], l["shape_types"]
    tags = shapes[:, sf["a"] + 3]
    composite = (tags == kinds["trimesh"]) | (tags == kinds["polyline"])
    validate_shapes(l, shapes[~composite], v)
    for shape in shapes[tags == kinds["convex_poly"]]:
        start, end = map(int, shape[sf["a"] : sf["a"] + 2])
        if not np.isfinite(v[start:end]).all():
            raise ValueError("Nonfinite convex vertex geometry")
    unused = np.zeros(len(v), bool)
    for shape in shapes[composite]:
        mesh_indices(l, shape, v, buffers["GeometryIndices"])
        vb, ib, count = map(int, shape[sf["a"] : sf["a"] + 3])
        nodes = buffers["GeometryIndices"][ib : ib + 3 * count].reshape(-1, 3)
        for i, (entry, _, _) in enumerate(nodes):
            if entry == 0xFFFFFFFF:
                lo, hi = v[vb + 2 * i : vb + 2 * i + 2]
                if np.isposinf(lo).all() and np.isneginf(hi).all():
                    unused[vb + 2 * i : vb + 2 * i + 2] = True
    if not np.isfinite(v[~unused]).all():
        raise ValueError("Nonfinite native geometry outside unused empty leaf bounds")
    result = buffers["Vertices"].copy()
    words = result.reshape(-1, l["vertex_words"])
    words[unused, l["vertex_vector"] : l["vertex_vector"] + 3] = 0
    return result
