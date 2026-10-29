"""Launch plans for native radix stages and eight resident stable passes."""

import tilelang.language as T
from shaders.utils.radix_sort.init_indirect_dispatches import gpu_init_sort_dispatch
from shaders.utils.radix_sort.init_batch_ids import gpu_init_sort_batched
from shaders.utils.radix_sort.sort_count import gpu_sort_count
from shaders.utils.radix_sort.sort_reduce import gpu_sort_reduce
from shaders.utils.radix_sort.sort_scan import gpu_sort_scan
from shaders.utils.radix_sort.sort_scan_add import gpu_sort_scan_add
from shaders.utils.radix_sort.sort_scatter import gpu_sort_scatter

LOCAL_BUFFERS = [
    "Config",
    "NumKeys",
    "Keys",
    "Values",
    "Aux",
    "OutKeys",
    "OutValues",
    "OutAux",
    "BatchIds",
    "Counts",
    "Reduced",
    "SortGrid",
    "ReduceGrid",
]
BUFFERS = tuple(LOCAL_BUFFERS) + ("InitConfig",) + tuple(f"Config{i}" for i in range(8))
STAGES = (
    "radix_count",
    "radix_reduce",
    "radix_scan",
    "radix_scan_add",
    "radix_scatter",
)


def dimensions(case, kind):
    cap, b = case["capacity"], case["batches"]
    groups = max((cap + 1023) // 1024, 1) + 1
    reduce = 16 * ((cap + 1048575) // 1048576) + 1
    per_batch = (
        groups
        if kind in ("radix_count", "radix_scatter")
        else reduce
        if kind in ("radix_reduce", "radix_scan_add")
        else 1
    )
    if kind == "radix_dispatch":
        return 1, 1, 1
    if kind == "radix_init_batched":
        return max((cap * b + 255) // 256, 1) + 1, 256, 1
    return per_batch * b, 256, per_batch


def make_kernel(layout, case, target, kind):
    sizes = case["buffer_words"]
    cpu = target == "c"
    blocks, threads, per_batch = dimensions(case, kind)
    dispatch = gpu_init_sort_dispatch(case["batches"])
    init = gpu_init_sort_batched(layout)
    count = gpu_sort_count(layout, cpu)
    reduce = gpu_sort_reduce(layout, cpu)
    scan = gpu_sort_scan(cpu)
    add = gpu_sort_scan_add(layout, cpu, sizes["Counts"])
    scatter = gpu_sort_scatter(layout, cpu)

    @T.macro
    def one(
        Config,
        NumKeys,
        Keys,
        Values,
        Aux,
        OutKeys,
        OutValues,
        OutAux,
        BatchIds,
        Counts,
        Reduced,
        SortGrid,
        ReduceGrid,
        Sums,
        Temp,
        Lds,
        Histogram,
        BinOffsets,
        Local,
        block,
        lane,
    ):
        group, batch = block % per_batch, block // per_batch
        if kind == "radix_dispatch":
            dispatch(NumKeys, SortGrid, ReduceGrid)
        elif kind == "radix_init_batched":
            init(
                Config,
                NumKeys,
                Keys,
                Values,
                OutKeys,
                OutValues,
                BatchIds,
                block * 256 + lane,
            )
        elif kind == "radix_count":
            count(Config, NumKeys, Keys, Counts, Histogram, group, batch, lane)
        elif kind == "radix_reduce":
            reduce(Config, NumKeys, Counts, Reduced, Sums, group, batch, lane)
        elif kind == "radix_scan":
            scan(NumKeys, Reduced, Sums, Temp, Lds, batch, lane)
        elif kind == "radix_scan_add":
            add(Config, NumKeys, Reduced, Counts, Sums, Temp, Lds, group, batch, lane)
        else:
            scatter(
                Config,
                NumKeys,
                Keys,
                Values,
                Counts,
                OutKeys,
                OutValues,
                Aux,
                OutAux,
                Sums,
                Temp,
                BinOffsets,
                Histogram,
                Local,
                group,
                batch,
                lane,
            )

    if target == "c":

        @T.prim_func
        def kernel(
            Config: T.Tensor((sizes["Config"],), "uint32"),
            NumKeys: T.Tensor((sizes["NumKeys"],), "uint32"),
            Keys: T.Tensor((sizes["Keys"],), "uint32"),
            Values: T.Tensor((sizes["Values"],), "uint32"),
            Aux: T.Tensor((sizes["Aux"],), "uint32"),
            OutKeys: T.Tensor((sizes["OutKeys"],), "uint32"),
            OutValues: T.Tensor((sizes["OutValues"],), "uint32"),
            OutAux: T.Tensor((sizes["OutAux"],), "uint32"),
            BatchIds: T.Tensor((sizes["BatchIds"],), "uint32"),
            Counts: T.Tensor((sizes["Counts"],), "uint32"),
            Reduced: T.Tensor((sizes["Reduced"],), "uint32"),
            SortGrid: T.Tensor((sizes["SortGrid"],), "uint32"),
            ReduceGrid: T.Tensor((sizes["ReduceGrid"],), "uint32"),
        ):
            with T.Kernel(1):
                Sums = T.alloc_local((256,), "uint32")
                Temp = T.alloc_local((256,), "uint32")
                Lds = T.alloc_local((4, 256), "uint32")
                Histogram = T.alloc_local((16,), "uint32")
                BinOffsets = T.alloc_local((256,), "uint32")
                Local = T.alloc_local((256, 5), "uint32")
                for block in T.serial(blocks):
                    for lane in T.serial(256 if kind == "radix_init_batched" else 1):
                        one(
                            Config,
                            NumKeys,
                            Keys,
                            Values,
                            Aux,
                            OutKeys,
                            OutValues,
                            OutAux,
                            BatchIds,
                            Counts,
                            Reduced,
                            SortGrid,
                            ReduceGrid,
                            Sums,
                            Temp,
                            Lds,
                            Histogram,
                            BinOffsets,
                            Local,
                            block,
                            lane,
                        )
    elif target == "ascend":
        import tilelang.ascend.language as A

        @A.prim_func
        def kernel(
            Config: A.Tensor((sizes["Config"],), "uint32"),
            NumKeys: A.Tensor((sizes["NumKeys"],), "uint32"),
            Keys: A.Tensor((sizes["Keys"],), "uint32"),
            Values: A.Tensor((sizes["Values"],), "uint32"),
            Aux: A.Tensor((sizes["Aux"],), "uint32"),
            OutKeys: A.Tensor((sizes["OutKeys"],), "uint32"),
            OutValues: A.Tensor((sizes["OutValues"],), "uint32"),
            OutAux: A.Tensor((sizes["OutAux"],), "uint32"),
            BatchIds: A.Tensor((sizes["BatchIds"],), "uint32"),
            Counts: A.Tensor((sizes["Counts"],), "uint32"),
            Reduced: A.Tensor((sizes["Reduced"],), "uint32"),
            SortGrid: A.Tensor((sizes["SortGrid"],), "uint32"),
            ReduceGrid: A.Tensor((sizes["ReduceGrid"],), "uint32"),
        ):
            with A.Kernel(blocks) as block:
                with A.SimtVF(threads=threads):
                    Sums = T.alloc_shared((256,), "uint32")
                    Temp = T.alloc_shared((256,), "uint32")
                    Lds = T.alloc_shared((4, 256), "uint32")
                    Histogram = T.alloc_shared((16,), "uint32")
                    BinOffsets = T.alloc_shared((256,), "uint32")
                    Local = T.alloc_shared((256, 5), "uint32")
                    one(
                        Config,
                        NumKeys,
                        Keys,
                        Values,
                        Aux,
                        OutKeys,
                        OutValues,
                        OutAux,
                        BatchIds,
                        Counts,
                        Reduced,
                        SortGrid,
                        ReduceGrid,
                        Sums,
                        Temp,
                        Lds,
                        Histogram,
                        BinOffsets,
                        Local,
                        block,
                        A.get_thread_binding(),
                    )
    else:

        @T.prim_func
        def kernel(
            Config: T.Tensor((sizes["Config"],), "uint32"),
            NumKeys: T.Tensor((sizes["NumKeys"],), "uint32"),
            Keys: T.Tensor((sizes["Keys"],), "uint32"),
            Values: T.Tensor((sizes["Values"],), "uint32"),
            Aux: T.Tensor((sizes["Aux"],), "uint32"),
            OutKeys: T.Tensor((sizes["OutKeys"],), "uint32"),
            OutValues: T.Tensor((sizes["OutValues"],), "uint32"),
            OutAux: T.Tensor((sizes["OutAux"],), "uint32"),
            BatchIds: T.Tensor((sizes["BatchIds"],), "uint32"),
            Counts: T.Tensor((sizes["Counts"],), "uint32"),
            Reduced: T.Tensor((sizes["Reduced"],), "uint32"),
            SortGrid: T.Tensor((sizes["SortGrid"],), "uint32"),
            ReduceGrid: T.Tensor((sizes["ReduceGrid"],), "uint32"),
        ):
            with T.Kernel(blocks, threads=threads) as block:
                Sums = T.alloc_shared((256,), "uint32")
                Temp = T.alloc_shared((256,), "uint32")
                Lds = T.alloc_shared((4, 256), "uint32")
                Histogram = T.alloc_shared((16,), "uint32")
                BinOffsets = T.alloc_shared((256,), "uint32")
                Local = T.alloc_shared((256, 5), "uint32")
                one(
                    Config,
                    NumKeys,
                    Keys,
                    Values,
                    Aux,
                    OutKeys,
                    OutValues,
                    OutAux,
                    BatchIds,
                    Counts,
                    Reduced,
                    SortGrid,
                    ReduceGrid,
                    Sums,
                    Temp,
                    Lds,
                    Histogram,
                    BinOffsets,
                    Local,
                    block,
                    T.get_thread_binding(),
                )

    return kernel


def make_program(layout, case, target):
    op = case["kind"]
    plans = []
    cache = {}

    def stage(kind, pass_id=None):
        if kind not in cache:
            cache[kind] = make_kernel(layout, case, target, kind)
        mapping = {n: n for n in LOCAL_BUFFERS}
        if kind == "radix_init_batched":
            mapping["Config"] = "InitConfig"
        if pass_id is not None:
            mapping["Config"] = f"Config{pass_id}"
            if pass_id % 2:
                for a, b in [
                    ("Keys", "OutKeys"),
                    ("Values", "OutValues"),
                    ("Aux", "OutAux"),
                ]:
                    mapping[a], mapping[b] = b, a
        blocks, threads, _ = dimensions(case, kind)
        plans.append(
            dict(
                name=f"{kind}_{pass_id}",
                buffer_map=mapping,
                blocks=blocks,
                threads=threads,
                kernel=cache[kind],
            )
        )

    if op == "radix_sort":
        for p in range(8):
            for kind in STAGES:
                stage(kind, p)
    elif op == "radix_pass":
        for kind in STAGES:
            stage(kind)
    else:
        stage(op)
    return plans
