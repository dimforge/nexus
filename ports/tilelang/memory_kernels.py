"""Device transfers corresponding to host buffer clear/copy operations."""

# TileLang parses these nested scopes as distinct launch levels.
# ruff: noqa: SIM117
import tilelang.language as T


def _validate(words, target):
    if type(words) is not int or not 0 < words < 2**31:
        raise ValueError("Invalid device transfer length")
    if target not in ("c", "metal", "cuda", "ascend"):
        raise ValueError("Unsupported device transfer target")


def copy(words, target):
    _validate(words, target)

    @T.macro
    def one(Source, Dest, i):
        if i < words:
            Dest[i] = Source[i]

    if target == "c":

        @T.prim_func
        def kernel(
            Source: T.Tensor((words,), "uint32"), Dest: T.Tensor((words,), "uint32")
        ):
            with T.Kernel(1):
                for i in T.serial(words):
                    one(Source, Dest, i)
    elif target == "ascend":
        import tilelang.ascend.language as A

        @A.prim_func
        def kernel(
            Source: A.Tensor((words,), "uint32"), Dest: A.Tensor((words,), "uint32")
        ):
            with A.Kernel(max((words + 63) // 64, 1)) as group:
                with A.SimtVF(threads=64):
                    one(Source, Dest, group * 64 + A.get_thread_binding())
    else:

        @T.prim_func
        def kernel(
            Source: T.Tensor((words,), "uint32"), Dest: T.Tensor((words,), "uint32")
        ):
            with T.Kernel(max((words + 63) // 64, 1), threads=64) as group:
                one(Source, Dest, group * 64 + T.get_thread_binding())

    return kernel


def single(words, half, target, increment=False):
    _validate(words, target)
    if type(half) is not int or not 0 <= half <= words // 2:
        raise ValueError("State copy ranges must be disjoint and in bounds")
    count = max(half, 1)

    @T.macro
    def one(Data, i):
        if half == 0:
            if i == 0:
                Data[0] = Data[0] + T.uint32(1) if increment else T.uint32(0)
        else:
            if i < half:
                Data[i] = Data[i + half]

    if target == "c":

        @T.prim_func
        def kernel(Data: T.Tensor((words,), "uint32")):
            with T.Kernel(1):
                for i in T.serial(count):
                    one(Data, i)
    elif target == "ascend":
        import tilelang.ascend.language as A

        @A.prim_func
        def kernel(Data: A.Tensor((words,), "uint32")):
            with A.Kernel(max((count + 63) // 64, 1)) as group:
                with A.SimtVF(threads=64):
                    one(Data, group * 64 + A.get_thread_binding())
    else:

        @T.prim_func
        def kernel(Data: T.Tensor((words,), "uint32")):
            with T.Kernel(max((count + 63) // 64, 1), threads=64) as group:
                one(Data, group * 64 + T.get_thread_binding())

    return kernel


def reset(words, target):
    return single(words, 0, target)


def states(words, half, target):
    return single(words, half, target)


def increment(words, target):
    return single(words, 0, target, increment=True)
