"""Counterpart of broad_phase/lbvh.rs; native 3D LBVH stages."""

import tilelang.language as T
from kernel_support import f32


def scratch_start(BatchIds, fields, batch_id):
    return batch_id * T.cast(BatchIds[fields["colliders_batch_capacity"]], "int32")


def root_id(start):
    return 2 * start


def expand_bits_3d(v):
    v = (v * T.uint32(0x00010001)) & T.uint32(0xFF0000FF)
    v = (v * T.uint32(0x00000101)) & T.uint32(0x0F00F00F)
    v = (v * T.uint32(0x00000011)) & T.uint32(0xC30C30C3)
    return (v * T.uint32(0x00000005)) & T.uint32(0x49249249)


@T.macro
def morton_3d(v, result):
    parts = T.alloc_local((3,), "uint32")
    for d in T.serial(3):
        # Rust's saturating float-to-u32 cast maps NaN (a zero-width domain) to 0.
        # Power-of-two scaling commutes with this clamp exactly in f32.
        # Keep bounds integral to avoid the compiler's six-digit float literals.
        value = T.if_then_else(
            v[d] != v[d],
            T.float32(0),
            T.min(T.max(v[d] * T.float32(1024), T.float32(0)), T.float32(1023)),
        )
        parts[d] = expand_bits_3d(T.cast(value, "uint32"))
    result[0] = parts[0] * T.uint32(4) + parts[1] * T.uint32(2) + parts[2]


@T.macro
def leading_zeros(value, output):
    # Explicit u32 search because the pinned portable C generator lacks clz lowering.
    bits = T.alloc_local((1,), "uint32")
    bits[0] = value
    output[0] = 0
    for step in T.serial(5):
        shift = 16 >> step
        if (bits[0] >> T.cast(32 - shift, "uint32")) == T.uint32(0):
            output[0] = output[0] + shift
            bits[0] = bits[0] << T.cast(shift, "uint32")
    output[0] = output[0] + T.cast(value == T.uint32(0), "int32")


def prefix_len():
    @T.macro
    def one(Keys, base, current, ii, other, n, result):
        result[0] = -1
        if other >= 0 and other < n:
            key = Keys[base + other]
            diff = current ^ key
            index_diff = T.cast(ii ^ other, "uint32")
            # Avoid undefined C clz(0), while retaining Rust leading_zeros(0)=32.
            kp = T.alloc_local((1,), "int32")
            ip = T.alloc_local((1,), "int32")
            leading_zeros(diff, kp)
            leading_zeros(index_diff, ip)
            result[0] = T.if_then_else(diff != T.uint32(0), kp[0], 32 + ip[0])

    return one


def gpu_lbvh_reset_collision_pairs(count):
    @T.macro
    def one(PairCount, gid):
        if gid < count:
            PairCount[gid] = T.uint32(0)

    return one


def gpu_flat_list_dispatch(l):
    b = l["batch"]

    @T.macro
    def one(PairCount, Indirect, BatchIds):
        total = T.min(
            T.atomic_add(PairCount[0], T.uint32(0), return_prev=True),
            BatchIds[b["collision_pairs_capacity"]],
        )
        Indirect[0] = (total + T.uint32(63)) // T.uint32(64)
        Indirect[1] = T.uint32(1)
        Indirect[2] = T.uint32(1)

    return one


def gpu_lbvh_compute_domain(l, cpu):
    b, a = l["batch"], l["aabb"]
    aw = l["aabb_words"]
    pw = l["pose_words"]
    pt = l["pose_translation"]

    @T.macro
    def one(Poses, Partials, BatchIds, Mins, Maxs, group, batch, num_groups, lane):
        n = T.cast(BatchIds[b["colliders_len"]], "int32")
        nb = T.cast(BatchIds[b["num_batches"]], "int32")
        ng = T.min(num_groups, 64)
        for t in T.serial(128 if cpu else 1):
            who = t if cpu else lane
            for d in T.serial(3):
                Mins[who, d] = T.float32(1e18)
                Maxs[who, d] = T.float32(-1e18)
            for it in T.serial((n + ng * 128 - 1) // (ng * 128)):
                i = group * 128 + who + it * ng * 128
                if i < n:
                    for d in T.serial(3):
                        value = f32(Poses, (i * nb + batch) * pw + pt + d)
                        Mins[who, d] = T.min(Mins[who, d], value)
                        Maxs[who, d] = T.max(Maxs[who, d], value)
        for step in T.serial(7):
            if not cpu:
                T.sync_threads()
            for t in T.serial(128 if cpu else 1):
                who = t if cpu else lane
                if who < (64 >> step):
                    for d in T.serial(3):
                        Mins[who, d] = T.min(Mins[who, d], Mins[who + (64 >> step), d])
                        Maxs[who, d] = T.max(Maxs[who, d], Maxs[who + (64 >> step), d])
        if not cpu:
            T.sync_threads()
        if lane == 0 and group < ng:
            for d in T.serial(3):
                Partials[(batch * 64 + group) * aw + a["mins"] + d] = T.reinterpret(
                    Mins[0, d], "uint32"
                )
                Partials[(batch * 64 + group) * aw + a["maxs"] + d] = T.reinterpret(
                    Maxs[0, d], "uint32"
                )
        for t in T.serial(128 if cpu else 1):
            who = t if cpu else lane
            if group == 0 and who >= ng and who < 64:
                for d in T.serial(3):
                    Partials[(batch * 64 + who) * aw + a["mins"] + d] = T.uint32(
                        l["max_flt_bits"]
                    )
                    Partials[(batch * 64 + who) * aw + a["maxs"] + d] = T.uint32(
                        l["max_flt_bits"] ^ 0x80000000
                    )

    return one


def gpu_lbvh_domain_merge(l, cpu):
    a = l["aabb"]
    aw = l["aabb_words"]

    @T.macro
    def one(Partials, Domain, Mins, Maxs, batch, lane):
        for t in T.serial(64 if cpu else 1):
            who = t if cpu else lane
            for d in T.serial(3):
                Mins[who, d] = f32(Partials, (batch * 64 + who) * aw + a["mins"] + d)
                Maxs[who, d] = f32(Partials, (batch * 64 + who) * aw + a["maxs"] + d)
        for step in T.serial(6):
            if not cpu:
                T.sync_threads()
            for t in T.serial(64 if cpu else 1):
                who = t if cpu else lane
                if who < (32 >> step):
                    for d in T.serial(3):
                        Mins[who, d] = T.min(Mins[who, d], Mins[who + (32 >> step), d])
                        Maxs[who, d] = T.max(Maxs[who, d], Maxs[who + (32 >> step), d])
        if not cpu:
            T.sync_threads()
        if lane == 0:
            for d in T.serial(3):
                Domain[batch * aw + a["mins"] + d] = T.reinterpret(Mins[0, d], "uint32")
                Domain[batch * aw + a["maxs"] + d] = T.reinterpret(Maxs[0, d], "uint32")

    return one


def gpu_lbvh_compute_morton(l, groups):
    b, a = l["batch"], l["aabb"]
    aw = l["aabb_words"]

    @T.macro
    def one(Poses, Domain, Keys, BatchIds, batch, gid):
        n = T.cast(BatchIds[b["colliders_len"]], "int32")
        nb = T.cast(BatchIds[b["num_batches"]], "int32")
        v = T.alloc_local((3,), "float32")
        key = T.alloc_local((1,), "uint32")
        for it in T.serial((n + groups * 64 - 1) // (groups * 64)):
            i = gid + it * groups * 64
            if i < n:
                for d in T.serial(3):
                    lo = f32(Domain, batch * aw + a["mins"] + d)
                    hi = f32(Domain, batch * aw + a["maxs"] + d)
                    v[d] = (
                        f32(
                            Poses,
                            (i * nb + batch) * l["pose_words"]
                            + l["pose_translation"]
                            + d,
                        )
                        - lo
                    ) / (hi - lo)
                morton_3d(v, key)
                Keys[scratch_start(BatchIds, b, batch) + i] = key[0]

    return one


def gpu_lbvh_build(l, groups):
    b, f = l["batch"], l["lbvh_node"]
    w = l["lbvh_node_words"]
    prefix = prefix_len()

    @T.macro
    def one(Keys, Tree, BatchIds, batch, gid):
        n = T.cast(BatchIds[b["colliders_len"]], "int32")
        base = scratch_start(BatchIds, b, batch)
        root = root_id(base)
        searching = T.alloc_local((1,), "bool")
        p = T.alloc_local((1,), "int32")
        prev = T.alloc_local((1,), "int32")
        d = T.alloc_local((1,), "int32")
        lmax = T.alloc_local((1,), "int32")
        length = T.alloc_local((1,), "int32")
        t = T.alloc_local((1,), "int32")
        s = T.alloc_local((1,), "int32")
        for it in T.serial((T.max(n - 1, 0) + groups * 64 - 1) // (groups * 64)):
            i = gid + it * groups * 64
            if i < n - 1:
                current = Keys[base + i]
                prefix(Keys, base, current, i, i + 1, n, p)
                prefix(Keys, base, current, i, i - 1, n, prev)
                diff = p[0] - prev[0]
                d[0] = T.if_then_else(diff > 0, 1, T.if_then_else(diff < 0, -1, 0))
                prefix(Keys, base, current, i, i - d[0], n, p)
                delta = p[0]
                lmax[0] = 2
                searching[0] = True
                for _ in T.serial(32):
                    if searching[0]:
                        prefix(Keys, base, current, i, i + lmax[0] * d[0], n, p)
                        if p[0] > delta:
                            lmax[0] = lmax[0] * 2
                        else:
                            searching[0] = False
                length[0] = 0
                t[0] = lmax[0] // 2
                for _ in T.serial(32):
                    if t[0] >= 1:
                        prefix(
                            Keys, base, current, i, i + (length[0] + t[0]) * d[0], n, p
                        )
                        if p[0] > delta:
                            length[0] = length[0] + t[0]
                        t[0] = t[0] // 2
                j = i + length[0] * d[0]
                prefix(Keys, base, current, i, j, n, p)
                node_delta = p[0]
                s[0] = 0
                t[0] = (length[0] + 1) // 2
                for _ in T.serial(32):
                    if t[0] >= 1:
                        prefix(Keys, base, current, i, i + (s[0] + t[0]) * d[0], n, p)
                        if p[0] > node_delta:
                            s[0] = s[0] + t[0]
                        t[0] = (t[0] + 1) // 2
                gamma = i + s[0] * d[0] + T.min(d[0], 0)
                left = T.if_then_else(T.min(i, j) == gamma, n - 1 + gamma, gamma)
                right = T.if_then_else(T.max(i, j) == gamma + 1, n + gamma, gamma + 1)
                out = (root + i) * w
                Tree[out + f["left"]] = T.cast(left, "uint32")
                Tree[out + f["right"]] = T.cast(right, "uint32")
                Tree[out + f["refit_count"]] = T.uint32(0)
                Tree[out + f["first_leaf"]] = T.cast(T.min(i, j), "uint32")
                Tree[out + f["last_leaf"]] = T.cast(T.max(i, j), "uint32")
                Tree[(root + left) * w + f["parent"]] = T.cast(i, "uint32")
                Tree[(root + right) * w + f["parent"]] = T.cast(i, "uint32")

    return one


def gpu_lbvh_escapes(l, groups):
    b, f = l["batch"], l["lbvh_node"]
    w = l["lbvh_node_words"]

    @T.macro
    def one(Tree, BatchIds, batch, gid):
        n = T.cast(BatchIds[b["colliders_len"]], "int32")
        root = root_id(scratch_start(BatchIds, b, batch))
        count = T.max(2 * n, 2) - 1
        active = T.alloc_local((1,), "bool")
        node = T.alloc_local((1,), "int32")
        escape = T.alloc_local((1,), "uint32")
        for it in T.serial((count + groups * 64 - 1) // (groups * 64)):
            i = gid + it * groups * 64
            if i < count:
                escape[0] = T.uint32(0xFFFFFFFF)
                node[0] = i
                active[0] = True
                for _ in T.serial(64):
                    if active[0]:
                        if node[0] == 0:
                            active[0] = False
                        else:
                            parent = T.cast(
                                Tree[(root + node[0]) * w + f["parent"]], "int32"
                            )
                            if Tree[(root + parent) * w + f["left"]] == T.cast(
                                node[0], "uint32"
                            ):
                                escape[0] = Tree[(root + parent) * w + f["right"]]
                                active[0] = False
                            node[0] = parent
                Tree[(root + i) * w + f["escape"]] = escape[0]

    return one


def gpu_lbvh_refit_leaves(l, groups):
    from shaders.shapes.shape import compute_aabb

    compute = compute_aabb(l)
    b, f, a = l["batch"], l["lbvh_node"], l["aabb"]
    w = l["lbvh_node_words"]

    @T.macro
    def one(Poses, Shapes, Sorted, Tree, BatchIds, FrontierLen, Vertices, batch, gid):
        n = T.cast(BatchIds[b["colliders_len"]], "int32")
        nb = T.cast(BatchIds[b["num_batches"]], "int32")
        base = scratch_start(BatchIds, b, batch)
        root = root_id(base)
        lo = T.alloc_local((3,), "float32")
        hi = T.alloc_local((3,), "float32")
        extent = T.alloc_local((1,), "float32")
        for it in T.serial((n + groups * 64 - 1) // (groups * 64)):
            i = gid + it * groups * 64
            if i < n:
                collider = T.cast(Sorted[base + i], "int32")
                out = (root + n - 1 + i) * w
                compute(Shapes, Poses, Vertices, collider * nb + batch, lo, hi)
                extent[0] = hi[0] - lo[0]
                for d in T.serial(3):
                    Tree[out + f["aabb"] + a["mins"] + d] = T.reinterpret(
                        lo[d], "uint32"
                    )
                    Tree[out + f["aabb"] + a["maxs"] + d] = T.reinterpret(
                        hi[d], "uint32"
                    )
                    extent[0] = T.max(extent[0], hi[d] - lo[d])
                Tree[out + f["aabb"] + a["padding0"]] = T.uint32(0)
                Tree[out + f["aabb"] + a["padding1"]] = T.uint32(0)
                Tree[out + f["max_extent"]] = T.reinterpret(extent[0], "uint32")
                Tree[out + f["left"]] = T.cast(collider, "uint32")
                Tree[out + f["first_leaf"]] = T.cast(i, "uint32")
                Tree[out + f["last_leaf"]] = T.cast(i, "uint32")
        if gid == 0:
            FrontierLen[batch] = T.uint32(0)

    return one


def gpu_lbvh_refit_plan(l):
    b = l["batch"]

    @T.macro
    def one(FrontierLen, Rounds, BatchIds):
        largest = T.alloc_local((1,), "uint32")
        largest[0] = T.uint32(0)
        for batch in T.serial(T.cast(BatchIds[b["num_batches"]], "int32")):
            largest[0] = T.max(
                largest[0], T.min(FrontierLen[batch], BatchIds[b["colliders_len"]])
            )
        Rounds[0] = (largest[0] + T.uint32(255)) // T.uint32(256)

    return one


def refit_barrier(target):
    @T.macro
    def one():
        if target == "metal":
            T.evaluate(T.call_extern("int32", "nexus_lbvh_refit_barrier"))
        elif target != "c":
            T.sync_threads()

    return one


def refit_node(l):
    f, a = l["lbvh_node"], l["aabb"]
    w = l["lbvh_node_words"]

    @T.macro
    def one(Tree, root, node):
        out = (root + node) * w
        left = (root + T.cast(Tree[out + f["left"]], "int32")) * w
        right = (root + T.cast(Tree[out + f["right"]], "int32")) * w
        for d in T.serial(3):
            lo = T.min(
                f32(Tree, left + f["aabb"] + a["mins"] + d),
                f32(Tree, right + f["aabb"] + a["mins"] + d),
            )
            hi = T.max(
                f32(Tree, left + f["aabb"] + a["maxs"] + d),
                f32(Tree, right + f["aabb"] + a["maxs"] + d),
            )
            Tree[out + f["aabb"] + a["mins"] + d] = T.reinterpret(lo, "uint32")
            Tree[out + f["aabb"] + a["maxs"] + d] = T.reinterpret(hi, "uint32")
        Tree[out + f["aabb"] + a["padding0"]] = T.uint32(0)
        Tree[out + f["aabb"] + a["padding1"]] = T.uint32(0)
        extent = T.max(
            f32(Tree, left + f["max_extent"]), f32(Tree, right + f["max_extent"])
        )
        Tree[out + f["max_extent"]] = T.reinterpret(extent, "uint32")

    return one


def gpu_lbvh_refit_chunks(l, target):
    cpu = target == "c"
    b, f = l["batch"], l["lbvh_node"]
    w = l["lbvh_node_words"]
    merge = refit_node(l)
    barrier = refit_barrier(target)

    @T.macro
    def one(Tree, Frontier, FrontierLen, BatchIds, chunk, batch, lane):
        n = T.cast(BatchIds[b["colliders_len"]], "int32")
        base = scratch_start(BatchIds, b, batch)
        root = root_id(base)
        active = T.alloc_local((256 if cpu else 1,), "bool")
        nodes = T.alloc_local((256 if cpu else 1,), "uint32")
        children = T.alloc_local((256 if cpu else 1,), "uint32")
        for t in T.serial(256 if cpu else 1):
            who = t if cpu else lane
            slot = t if cpu else 0
            leaf = chunk * 256 + who
            active[slot] = (leaf < n) & (n >= 2)
            children[slot] = T.cast(T.max(n - 1, 0) + leaf, "uint32")
            nodes[slot] = T.uint32(0)
            if active[slot]:
                nodes[slot] = Tree[
                    (root + T.cast(children[slot], "int32")) * w + f["parent"]
                ]
        # Fixed-depth schedule matches native web-compatible semantics; all lanes
        # reach the storage fence even after their arrival is no longer active.
        for _ in T.serial(64):
            for t in T.serial(256 if cpu else 1):
                slot = t if cpu else 0
                if active[slot]:
                    node = T.cast(nodes[slot], "int32")
                    off = (root + node) * w
                    first = T.cast(Tree[off + f["first_leaf"]], "int32")
                    last = T.cast(Tree[off + f["last_leaf"]], "int32")
                    if first // 256 != chunk or last // 256 != chunk:
                        k = T.atomic_add(
                            FrontierLen[batch], T.uint32(1), return_prev=True
                        )
                        Frontier[base + T.cast(k, "int32")] = children[slot]
                        active[slot] = False
                    else:
                        prior = T.cast(
                            T.atomic_add(
                                Tree[off + f["refit_count"]],
                                T.uint32(1),
                                return_prev=True,
                            ),
                            "int32",
                        )
                        if prior == 0:
                            active[slot] = False
                        else:
                            merge(Tree, root, node)
                            if node == 0:
                                active[slot] = False
                            else:
                                children[slot] = nodes[slot]
                                nodes[slot] = Tree[off + f["parent"]]
            barrier()

    return one


def gpu_lbvh_refit_frontier(l, target):
    cpu = target == "c"
    b, f = l["batch"], l["lbvh_node"]
    w = l["lbvh_node_words"]
    merge = refit_node(l)
    barrier = refit_barrier(target)

    @T.macro
    def one(Tree, Frontier, FrontierLen, BatchIds, Rounds, batch, lane):
        base = scratch_start(BatchIds, b, batch)
        root = root_id(base)
        count = T.cast(T.min(FrontierLen[batch], BatchIds[b["colliders_len"]]), "int32")
        active = T.alloc_local((256 if cpu else 1,), "bool")
        nodes = T.alloc_local((256 if cpu else 1,), "uint32")
        for iteration in T.serial((count + 255) // 256):
            for t in T.serial(256 if cpu else 1):
                who = t if cpu else lane
                slot = t if cpu else 0
                k = iteration * 256 + who
                active[slot] = k < count
                nodes[slot] = T.uint32(0)
                if active[slot]:
                    child = T.cast(Frontier[base + k], "int32")
                    nodes[slot] = Tree[(root + child) * w + f["parent"]]
            for _ in T.serial(64):
                for t in T.serial(256 if cpu else 1):
                    slot = t if cpu else 0
                    if active[slot]:
                        node = T.cast(nodes[slot], "int32")
                        off = (root + node) * w
                        prior = T.cast(
                            T.atomic_add(
                                Tree[off + f["refit_count"]],
                                T.uint32(1),
                                return_prev=True,
                            ),
                            "int32",
                        )
                        if prior == 0:
                            active[slot] = False
                        else:
                            merge(Tree, root, node)
                            if node == 0:
                                active[slot] = False
                            else:
                                nodes[slot] = Tree[off + f["parent"]]
                barrier()

    return one


def gpu_lbvh_find_collision_pairs(l, c, target):
    cpu = target == "c"
    # Ascend keeps the eight lane-owned pair slots private. This avoids the
    # compiler inferring a cross-lane dependency inside divergent traversal.
    # The workgroup prefix and atomic overflow allocation remain unchanged.
    pair_stride = 1 if target == "ascend" else 64
    b, f, a, g = l["batch"], l["lbvh_node"], l["aabb"], l["groups"]
    w = l["lbvh_node_words"]
    groups = c["groups"]

    @T.macro
    def one(
        Tree,
        Pairs,
        PairCount,
        Groups,
        BatchIds,
        Filters,
        WgPairs,
        Scan,
        Temp,
        Base,
        group,
        batch,
        lane,
    ):
        n = T.cast(BatchIds[b["colliders_len"]], "int32")
        nb = T.cast(BatchIds[b["num_batches"]], "int32")
        root = root_id(scratch_start(BatchIds, b, batch))
        counts = T.alloc_local((64 if cpu else 1,), "int32")
        current = T.alloc_local((1,), "uint32")
        lo = T.alloc_local((3,), "float32")
        hi = T.alloc_local((3,), "float32")
        hit = T.alloc_local((1,), "bool")
        for t in T.serial(64 if cpu else 1):
            who = t if cpu else lane
            slot = t if cpu else 0
            pair_lane = 0 if target == "ascend" else who
            counts[slot] = 0
            for it in T.serial((n + groups * 64 - 1) // (groups * 64)):
                leaf = group * 64 + who + it * groups * 64
                if leaf < n:
                    leaf_off = (root + n - 1 + leaf) * w
                    i = T.cast(Tree[leaf_off + f["left"]], "int32")
                    gi = (i * nb + batch) * l["group_words"]
                    fi = (i * nb + batch) * 2
                    extent = T.max(
                        T.max(
                            f32(Tree, leaf_off + f["aabb"] + a["maxs"])
                            - f32(Tree, leaf_off + f["aabb"] + a["mins"]),
                            f32(Tree, leaf_off + f["aabb"] + a["maxs"] + 1)
                            - f32(Tree, leaf_off + f["aabb"] + a["mins"] + 1),
                        ),
                        f32(Tree, leaf_off + f["aabb"] + a["maxs"] + 2)
                        - f32(Tree, leaf_off + f["aabb"] + a["mins"] + 2),
                    )
                    for d in T.serial(3):
                        lo[d] = f32(
                            Tree, leaf_off + f["aabb"] + a["mins"] + d
                        ) - T.float32(0.002)
                        hi[d] = f32(
                            Tree, leaf_off + f["aabb"] + a["maxs"] + d
                        ) + T.float32(0.002)
                    current[0] = T.uint32(0)
                    for _ in T.serial(2 * n):
                        if current[0] != T.uint32(0xFFFFFFFF):
                            node = T.cast(current[0], "int32")
                            off = (root + node) * w
                            is_leaf = node >= n - 1
                            maximum = f32(Tree, off + f["max_extent"])
                            owned = (maximum > T.float32(2) * extent) | (
                                (leaf < T.cast(Tree[off + f["last_leaf"]], "int32"))
                                & (maximum >= T.float32(0.5) * extent)
                            )
                            hit[0] = owned
                            for d in T.serial(3):
                                hit[0] = (
                                    hit[0]
                                    & (
                                        lo[d]
                                        <= f32(Tree, off + f["aabb"] + a["maxs"] + d)
                                    )
                                    & (
                                        hi[d]
                                        >= f32(Tree, off + f["aabb"] + a["mins"] + d)
                                    )
                                )
                            current[0] = T.if_then_else(
                                hit[0] & ~is_leaf,
                                Tree[off + f["left"]],
                                Tree[off + f["escape"]],
                            )
                            if hit[0] and is_leaf:
                                j = T.cast(Tree[off + f["left"]], "int32")
                                gj = (j * nb + batch) * l["group_words"]
                                fj = (j * nb + batch) * 2
                                left = (
                                    Groups[gi + g["memberships"]]
                                    & Groups[gj + g["filter"]]
                                ) != T.uint32(0)
                                right = (
                                    Groups[gj + g["memberships"]]
                                    & Groups[gi + g["filter"]]
                                ) != T.uint32(0)
                                either = (
                                    Groups[gi + g["test_mode"]] == T.uint32(1)
                                ) & (Groups[gj + g["test_mode"]] == T.uint32(1))
                                allowed = T.if_then_else(
                                    either, left | right, left & right
                                )
                                if (
                                    allowed
                                    and Filters[fi] != Filters[fj]
                                    and (
                                        Filters[fi + 1] == T.uint32(0)
                                        or Filters[fi + 1] != Filters[fj + 1]
                                    )
                                ):
                                    ci = T.cast(T.min(i, j) * nb + batch, "uint32")
                                    cj = T.cast(T.max(i, j) * nb + batch, "uint32")
                                    if counts[slot] < 8:
                                        WgPairs[
                                            counts[slot] * pair_stride + pair_lane, 0
                                        ] = ci
                                        WgPairs[
                                            counts[slot] * pair_stride + pair_lane, 1
                                        ] = cj
                                        counts[slot] = counts[slot] + 1
                                    else:
                                        dest = T.atomic_add(
                                            PairCount[0], T.uint32(1), return_prev=True
                                        )
                                        if (
                                            dest
                                            < BatchIds[b["collision_pairs_capacity"]]
                                        ):
                                            Pairs[
                                                T.cast(dest, "int32") * l["pair_words"]
                                                + l["pair_colliders"]
                                            ] = ci
                                            Pairs[
                                                T.cast(dest, "int32") * l["pair_words"]
                                                + l["pair_colliders"]
                                                + 1
                                            ] = cj
            Scan[who] = T.cast(counts[slot], "uint32")
        for step in T.serial(6):
            if not cpu:
                T.sync_threads()
            for t in T.serial(64 if cpu else 1):
                who = t if cpu else lane
                Temp[who] = T.uint32(0)
                if who >= (1 << step):
                    Temp[who] = Scan[who - (1 << step)]
            if not cpu:
                T.sync_threads()
            for t in T.serial(64 if cpu else 1):
                who = t if cpu else lane
                Scan[who] = Scan[who] + Temp[who]
        if not cpu:
            T.sync_threads()
        if cpu or lane == 63:
            Base[0] = T.atomic_add(PairCount[0], Scan[63], return_prev=True)
        if not cpu:
            T.sync_threads()
        for t in T.serial(64 if cpu else 1):
            who = t if cpu else lane
            slot = t if cpu else 0
            pair_lane = 0 if target == "ascend" else who
            first = Base[0] + Scan[who] - T.cast(counts[slot], "uint32")
            for k in T.serial(8):
                dest = first + T.cast(k, "uint32")
                if k < counts[slot] and dest < BatchIds[b["collision_pairs_capacity"]]:
                    for side in T.serial(2):
                        Pairs[
                            T.cast(dest, "int32") * l["pair_words"]
                            + l["pair_colliders"]
                            + side
                        ] = WgPairs[k * pair_stride + pair_lane, side]

    return one
