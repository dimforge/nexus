"""Hierarchy launch plan matching src_rbd/utils/prefix_sum.rs."""

import tilelang.language as T
from shaders.utils.prefix_sum import gpu_prefix_sum_sweep, gpu_add_data_grp


def make_kernel(stride, batches, data_words, aux_words, target, kind):
    groups_per_batch = max((stride + 255) // 256, 1)
    groups = groups_per_batch * batches
    sweep, add = gpu_prefix_sum_sweep(target == "c"), gpu_add_data_grp()

    @T.macro
    def one(Data, Aux, Stride, Workspace, group, lane):
        if stride > 0:
            batch, bid = group // groups_per_batch, group % groups_per_batch
            if kind == "prefix_sweep":
                sweep(Data, Aux, Stride, Workspace, batch, bid, lane)
            else:
                add(Data, Aux, Stride, batch, bid, lane)

    if target == "c":

        @T.prim_func
        def kernel(
            Data: T.Tensor((data_words,), "uint32"),
            Aux: T.Tensor((aux_words,), "uint32"),
            Stride: T.Tensor((2,), "uint32"),
        ):
            with T.Kernel(1):
                Workspace = T.alloc_local((256,), "uint32")
                for group in T.serial(groups):
                    if kind == "prefix_sweep":
                        one(Data, Aux, Stride, Workspace, group, 0)
                    else:
                        for lane in T.serial(256):
                            one(Data, Aux, Stride, Workspace, group, lane)
    elif target == "ascend":
        import tilelang.ascend.language as A

        @A.prim_func
        def kernel(
            Data: A.Tensor((data_words,), "uint32"),
            Aux: A.Tensor((aux_words,), "uint32"),
            Stride: A.Tensor((2,), "uint32"),
        ):
            with A.Kernel(groups) as group:
                with A.SimtVF(threads=256):
                    Workspace = T.alloc_shared((256,), "uint32")
                    one(Data, Aux, Stride, Workspace, group, A.get_thread_binding())
    else:

        @T.prim_func
        def kernel(
            Data: T.Tensor((data_words,), "uint32"),
            Aux: T.Tensor((aux_words,), "uint32"),
            Stride: T.Tensor((2,), "uint32"),
        ):
            with T.Kernel(groups, threads=256) as group:
                Workspace = T.alloc_shared((256,), "uint32")
                one(Data, Aux, Stride, Workspace, group, T.get_thread_binding())

    return kernel


def make_program(layout, case, target):
    stages = []
    strides = case["strides"]

    def stage(level, kind):
        stride = strides[level]
        mapping = {
            "Data": "Data" if level == 0 else f"Aux{level - 1}",
            "Aux": f"Aux{level}",
            "Stride": f"Stride{level}",
        }
        sizes = case["buffer_words"]
        stages.append(
            {
                "name": f"{kind}_{level}",
                "buffer_map": mapping,
                "blocks": max((stride + 255) // 256, 1) * case["batches"],
                "threads": 256,
                "kernel": make_kernel(
                    stride,
                    case["batches"],
                    sizes[mapping["Data"]],
                    sizes[mapping["Aux"]],
                    target,
                    kind,
                ),
            }
        )

    if case["kind"] == "prefix_scan":
        for level in range(case["levels"]):
            stage(level, "prefix_sweep")
        for level in reversed(range(case["levels"] - 1)):
            stage(level, "prefix_add")
    else:
        stage(0, case["kind"])
    return stages
