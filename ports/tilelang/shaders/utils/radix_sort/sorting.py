"""Counterpart of utils/radix_sort/sorting.rs (native 4-bit stable radix passes)."""

import tilelang.language as T

WG = 256
BITS_PER_PASS = 4
DIGIT_MASK = 15
BIN_COUNT = 16
HISTOGRAM_SIZE = 4096
ELEMENTS_PER_THREAD = 4
BLOCK_SIZE = 1024
NUM_PASSES = 8


def div_ceil(a, b):
    return a // b + T.cast(a % b != 0, a.dtype)


def num_workgroups(n):
    return div_ceil(n, 1024)


def extract_digit(key, shift):
    return (key >> shift) & T.uint32(15)
