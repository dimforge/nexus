"""Dimension checks for native 256-element prefix hierarchies."""


def hierarchy(count):
    if type(count) is not int or not 0 <= count < 2**31 - 256:
        raise ValueError("Invalid runtime scan length")
    strides = [count]
    for _ in range(4):
        strides.append((strides[-1] + 255) // 256)
    levels = next(i for i in range(1, 5) if strides[i] <= 1)
    return strides, levels


def validate_scan(c, b, prefix, count, stride_field, levels_field):
    strides, levels = hierarchy(count)
    if c[stride_field] != strides[:4] or c[levels_field] != levels:
        raise ValueError("Runtime prefix hierarchy differs from active length")
    for i in range(4):
        if (
            b[f"{prefix}Stride{i}"][0] != strides[i]
            or len(b[f"{prefix}Aux{i}"]) != strides[i + 1] + 3
        ):
            raise ValueError("Runtime prefix scratch/uniform differs from hierarchy")
