"""Counterpart of dynamics/color_buckets.rs: color-major contact buckets.

The initial count/scatter port uses the Rust generic global-atomic algorithm for
all batches. The single-batch local-histogram optimization is deferred explicitly;
bucket membership and copied link bits are unchanged, while atomic order may vary.
"""

import tilelang.language as T


def bucket_color(color, length, stride):
    return T.if_then_else(
        length == 0,
        T.uint32(0),
        T.if_then_else((color != 0) & (color < stride - 1), color, stride - 1),
    )


def gpu_color_buckets_reset(words):
    @T.macro
    def reset(Buckets, i):
        if i < words:
            Buckets[i] = T.uint32(0)

    return reset


def gpu_color_buckets_count(layout):
    lw = layout["link_words"]
    l = layout["link"]
    b = layout["batch"]

    @T.macro
    def count(Colors, Links, Plan, Buckets, BatchIds, SortedLinks, tid, groups):
        nb = T.Cast("int32", BatchIds[b["num_batches"]])
        stride = BatchIds[b["solver_color_buckets_stride"]]
        bound = T.Cast("int32", Plan[layout["plan"]["bound"]])
        for chunk in T.serial(T.ceildiv(bound, groups * 64)):
            i = tid + chunk * groups * 64
            if i < bound:
                color = T.Cast(
                    "int32", bucket_color(Colors[i], Links[i * lw + l["len"]], stride)
                )
                if color != 0:
                    batch = T.Cast("int32", Links[i * lw + l["solver_body_a"]]) % nb
                    T.atomic_add(Buckets[color * nb + batch], T.uint32(1))
                SortedLinks[i * lw + l["len"]] = T.uint32(0)

    return count


def gpu_color_buckets_scatter(layout):
    lw = layout["link_words"]
    l = layout["link"]
    b = layout["batch"]

    @T.macro
    def scatter(
        Colors, Links, Plan, Buckets, SortedLinks, BatchIds, Indices, tid, groups
    ):
        nb = T.Cast("int32", BatchIds[b["num_batches"]])
        stride = BatchIds[b["solver_color_buckets_stride"]]
        bound = T.Cast("int32", Plan[layout["plan"]["bound"]])
        for chunk in T.serial(T.ceildiv(bound, groups * 64)):
            i = tid + chunk * groups * 64
            if i < bound:
                color = T.Cast(
                    "int32", bucket_color(Colors[i], Links[i * lw + l["len"]], stride)
                )
                if color != 0:
                    batch = T.Cast("int32", Links[i * lw + l["solver_body_a"]]) % nb
                    dst = T.Cast(
                        "int32",
                        T.atomic_add(
                            Buckets[color * nb + batch], T.uint32(1), return_prev=True
                        ),
                    )
                    for word in T.unroll(lw):
                        SortedLinks[dst * lw + word] = Links[i * lw + word]
                    Indices[i] = T.Cast("uint32", dst)

    return scatter


def gpu_color_dispatch_grid(layout, serial):
    b = layout["batch"]
    s = layout["stats"]

    @T.macro
    def lane_values(Buckets, Stats, BatchIds, Sizes, Colors, lane):
        nb = T.Cast("int32", BatchIds[b["num_batches"]])
        num_colors = T.Cast("int32", BatchIds[b["solver_color_buckets_stride"]]) - 2
        Stats[s["sizes"] + lane] = T.uint32(0)
        if (lane > 0) & (lane <= num_colors):
            Stats[s["sizes"] + lane] = (
                Buckets[(lane + 1) * nb - 1] - Buckets[lane * nb - 1]
            )
        Sizes[lane] = T.uint32(0)
        Colors[lane] = T.uint32(0)
        for step in T.serial(T.ceildiv(num_colors, 64)):
            color = 1 + lane + step * 64
            if color <= num_colors:
                size = Buckets[(color + 1) * nb - 1] - Buckets[color * nb - 1]
                Sizes[lane] = T.max(Sizes[lane], size)
                if size != 0:
                    Colors[lane] = T.max(Colors[lane], T.Cast("uint32", color))

    @T.macro
    def reduce(Sizes, Colors, lane, stride):
        Sizes[lane] = T.max(Sizes[lane], Sizes[lane + stride])
        Colors[lane] = T.max(Colors[lane], Colors[lane + stride])

    @T.macro
    def save(Stats, Indirect, Sizes, Colors):
        Stats[s["max_bucket"]] = Sizes[0]
        Stats[s["highest_color"]] = Colors[0]
        Indirect[0] = T.Cast(
            "uint32", T.min(T.ceildiv(T.Cast("int32", Sizes[0]), 64), 256)
        )
        Indirect[1] = T.uint32(1)
        Indirect[2] = T.uint32(1)

    @T.macro
    def dispatch(Buckets, Stats, Indirect, BatchIds, Sizes, Colors, lane):
        if serial:
            for v in T.serial(64):
                lane_values(Buckets, Stats, BatchIds, Sizes, Colors, v)
            for step in T.unroll(6, explicit=True):
                for v in T.serial(32 >> step):
                    reduce(Sizes, Colors, v, 32 >> step)
            save(Stats, Indirect, Sizes, Colors)
        else:
            lane_values(Buckets, Stats, BatchIds, Sizes, Colors, lane)
            for step in T.unroll(6, explicit=True):
                T.sync_threads()
                if lane < (32 >> step):
                    reduce(Sizes, Colors, lane, 32 >> step)
            T.sync_threads()
            if lane == 0:
                save(Stats, Indirect, Sizes, Colors)

    return dispatch
