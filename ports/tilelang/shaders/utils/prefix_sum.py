"""Counterpart of src_rbd_shaders/utils/prefix_sum.rs (256-lane Blelloch scan)."""

import tilelang.language as T

WORKGROUP_SIZE = 256


def gpu_prefix_sum_sweep(serial):
    @T.macro
    def load(Data, Stride, Workspace, batch, bid, lane):
        size = T.Cast("int32", Stride[0])
        index = bid * 256 + lane
        Workspace[lane] = T.uint32(0)
        if index < size:
            Workspace[lane] = Data[batch * size + index]

    @T.macro
    def up(Workspace, lane, offset):
        ia = lane * 2 * offset + offset - 1
        ib = (lane * 2 + 1) * offset + offset - 1
        Workspace[ib] = Workspace[ia] + Workspace[ib]

    @T.macro
    def down(Workspace, lane, offset):
        ia = lane * 2 * offset + offset - 1
        ib = (lane * 2 + 1) * offset + offset - 1
        # Materialize both reads before either store, as in Rust.
        a = T.alloc_local((1,), "uint32")
        b = T.alloc_local((1,), "uint32")
        a[0] = Workspace[ia]
        b[0] = Workspace[ib]
        Workspace[ia] = b[0]
        Workspace[ib] = a[0] + b[0]

    @T.macro
    def root(Aux, Stride, Workspace, batch, bid):
        stride = T.ceildiv(T.Cast("int32", Stride[0]), 256)
        Aux[batch * stride + bid] = Workspace[255]
        Workspace[255] = T.uint32(0)

    @T.macro
    def store(Data, Stride, Workspace, batch, bid, lane):
        size = T.Cast("int32", Stride[0])
        index = bid * 256 + lane
        if index < size:
            Data[batch * size + index] = Workspace[lane]

    # Explicit expansion exposes disjoint lane index pairs to Ascend thread-sync
    # analysis; pragma-only unrolling produced a misleading barrier-hoisting warning.
    @T.macro
    def sweep(Data, Aux, Stride, Workspace, batch, bid, lane):
        if serial:
            for v in T.serial(256):
                load(Data, Stride, Workspace, batch, bid, v)
            for step in T.unroll(8, explicit=True):
                for v in T.serial(128 >> step):
                    up(Workspace, v, 1 << step)
            root(Aux, Stride, Workspace, batch, bid)
            for step in T.unroll(8, explicit=True):
                for v in T.serial(1 << step):
                    down(Workspace, v, 128 >> step)
            for v in T.serial(256):
                store(Data, Stride, Workspace, batch, bid, v)
        else:
            load(Data, Stride, Workspace, batch, bid, lane)
            for step in T.unroll(8, explicit=True):
                T.sync_threads()
                if lane < (128 >> step):
                    up(Workspace, lane, 1 << step)
            if lane == 0:
                root(Aux, Stride, Workspace, batch, bid)
            for step in T.unroll(8, explicit=True):
                T.sync_threads()
                if lane < (1 << step):
                    down(Workspace, lane, 128 >> step)
            T.sync_threads()
            store(Data, Stride, Workspace, batch, bid, lane)

    return sweep


def gpu_add_data_grp():
    @T.macro
    def add(Data, Aux, Stride, batch, bid, lane):
        size = T.Cast("int32", Stride[0])
        index = bid * 256 + lane
        if index < size:
            aux_stride = T.ceildiv(size, 256)
            Data[batch * size + index] += Aux[batch * aux_stride + bid]

    return add
