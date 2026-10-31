"""CUDA Driver API executor: native buffers, one ordered stream, no host physics."""

import ctypes as C
import ctypes.util
import sys
import threading
from dataclasses import dataclass

import numpy as np
from device_compiler import DeviceCompiler

P = C.c_void_p
U64 = C.c_uint64
U32 = C.c_uint32
I32 = C.c_int32
SIZE = C.c_size_t


@dataclass(eq=False)
class Allocation:
    pointer: int
    nbytes: int
    owner: object

    def region(self, start, count):
        if (
            type(start) is not int
            or type(count) is not int
            or start < 0
            or count < 0
            or 4 * (start + count) > self.nbytes
        ):
            raise ValueError("Invalid device buffer range")
        return self.pointer + 4 * start, 4 * count


def words_array(words):
    if not isinstance(words, np.ndarray) or words.dtype != np.uint32 or words.ndim != 1:
        raise ValueError("Device buffers require uint32 word arrays")
    return np.ascontiguousarray(words)


class CudaDevice:
    def __init__(self, ordinal=0, *, library=None):
        if type(ordinal) is not int or ordinal < 0:
            raise ValueError("Invalid CUDA device ordinal")
        self.thread = threading.get_ident()
        self.allocations, self.modules, self.functions = {}, {}, {}
        self.compiler = None
        self.context, self.previous, self.stream = P(), P(), P()
        self.device = I32()
        self.retained, self.closed = False, False
        name = library or ("nvcuda.dll" if sys.platform == "win32" else "libcuda.so.1")
        try:
            self.lib = (C.WinDLL if sys.platform == "win32" else C.CDLL)(name)
        except OSError as error:
            raise RuntimeError(
                "CUDA driver unavailable; this backend requires an NVIDIA device/driver and never falls back to CPU"
            ) from error
        self._bind()
        try:
            self.check("cuInit", 0)
            self.check("cuDeviceGet", C.byref(self.device), ordinal)
            self.check("cuCtxGetCurrent", C.byref(self.previous))
            self.check("cuDevicePrimaryCtxRetain", C.byref(self.context), self.device)
            self.retained = True
            self.check("cuCtxSetCurrent", self.context)
            self.check("cuStreamCreate", C.byref(self.stream), 1)  # nonblocking
            buf = C.create_string_buffer(256)
            self.check("cuDeviceGetName", buf, len(buf), self.device)
            version = I32()
            self.check("cuDriverGetVersion", C.byref(version))
            self.max_threads = self.attribute(1)
            self.max_blocks = self.attribute(5)
            major, minor = self.attribute(75), self.attribute(76)
            if major < 8:
                raise RuntimeError(
                    "Current CUDA lowering targets sm_80; use an Ampere-or-newer GPU"
                )
            self.environment = {
                "device_backend": "cuda",
                "device": buf.value.decode(),
                "device_ordinal": ordinal,
                "compute_capability": [major, minor],
                "driver_version": version.value,
                "driver_library": name,
                "runtime": "CUDA Driver API, ordered native stream",
            }
        except Exception:
            self.close()
            raise

    def _bind(self):
        signatures = {
            "cuInit": [U32],
            "cuDeviceGet": [C.POINTER(I32), I32],
            "cuDeviceGetName": [P, I32, I32],
            "cuDeviceGetAttribute": [C.POINTER(I32), I32, I32],
            "cuDriverGetVersion": [C.POINTER(I32)],
            "cuCtxGetCurrent": [C.POINTER(P)],
            "cuDevicePrimaryCtxRetain": [C.POINTER(P), I32],
            "cuDevicePrimaryCtxRelease_v2": [I32],
            "cuCtxSetCurrent": [P],
            "cuStreamCreate": [C.POINTER(P), U32],
            "cuStreamSynchronize": [P],
            "cuStreamDestroy_v2": [P],
            "cuMemAlloc_v2": [C.POINTER(U64), SIZE],
            "cuMemFree_v2": [U64],
            "cuMemcpyHtoD_v2": [U64, P, SIZE],
            "cuMemcpyDtoH_v2": [P, U64, SIZE],
            "cuMemcpyDtoDAsync_v2": [U64, U64, SIZE, P],
            "cuModuleLoadData": [C.POINTER(P), P],
            "cuModuleUnload": [P],
            "cuModuleGetFunction": [C.POINTER(P), P, C.c_char_p],
            "cuFuncSetAttribute": [P, I32, I32],
            "cuLaunchKernel": [
                P,
                U32,
                U32,
                U32,
                U32,
                U32,
                U32,
                U32,
                P,
                C.POINTER(P),
                C.POINTER(P),
            ],
            "cuGetErrorString": [I32, C.POINTER(C.c_char_p)],
        }
        self.api = {}
        for name, args in signatures.items():
            f = getattr(self.lib, name)
            f.argtypes, f.restype = args, I32
            self.api[name] = f

    def check(self, name, *args):
        if threading.get_ident() != self.thread:
            raise RuntimeError("CUDA session used from a different thread")
        code = self.api[name](*args)
        if code:
            description = C.c_char_p()
            self.api["cuGetErrorString"](code, C.byref(description))
            detail = (
                description.value.decode(errors="replace")
                if description.value
                else "unknown error"
            )
            raise RuntimeError(f"{name}: CUDA error {code}: {detail}")

    def attribute(self, kind):
        value = I32()
        self.check("cuDeviceGetAttribute", C.byref(value), kind, self.device)
        return value.value

    def _owned(self, allocation):
        if (
            self.closed
            or allocation.owner is not self
            or self.allocations.get(allocation.pointer) is not allocation
        ):
            raise ValueError("Foreign or released CUDA allocation")
        self.check("cuCtxSetCurrent", self.context)

    def upload(self, words):
        words = words_array(words)
        if words.nbytes == 0:
            raise ValueError("Empty physical device allocation")
        self.check("cuCtxSetCurrent", self.context)
        ptr = U64()
        self.check("cuMemAlloc_v2", C.byref(ptr), words.nbytes)
        allocation = Allocation(ptr.value, words.nbytes, self)
        self.allocations[ptr.value] = allocation
        try:
            self.write(allocation, 0, words)
        except Exception:
            self.free(allocation)
            raise
        return allocation

    def free(self, allocation):
        self._owned(allocation)
        self.check("cuMemFree_v2", allocation.pointer)
        del self.allocations[allocation.pointer]

    def synchronize(self):
        if self.closed:
            raise RuntimeError("Closed CUDA session")
        self.check("cuCtxSetCurrent", self.context)
        self.check("cuStreamSynchronize", self.stream)

    def read(self, allocation, start, count):
        self._owned(allocation)
        pointer, size = allocation.region(start, count)
        self.synchronize()
        result = np.empty(count, np.uint32)
        if size:
            self.check("cuMemcpyDtoH_v2", result.ctypes.data, pointer, size)
        return result

    def write(self, allocation, start, words):
        self._owned(allocation)
        words = words_array(words)
        pointer, size = allocation.region(start, len(words))
        self.synchronize()
        if size:
            self.check("cuMemcpyHtoD_v2", pointer, words.ctypes.data, size)
            # Pageable H2D can return after host staging, before DMA completes.
            # Our nonblocking compute stream does not wait for the default stream.
            # Publish the upload only after its legacy-default-stream copy finishes.
            self.check("cuStreamSynchronize", None)

    def copy(self, destination, dst_start, source, src_start, count):
        self._owned(destination)
        self._owned(source)
        dst, size = destination.region(dst_start, count)
        src, _ = source.region(src_start, count)
        if size:
            self.check("cuMemcpyDtoDAsync_v2", dst, src, size, self.stream)

    def compile(self, source, bindings, directory):
        if self.compiler is None:
            self.compiler = DeviceCompiler("cuda")
        binary, receipt = self.compiler.compile(source, bindings, directory)
        key = (
            receipt["binary_sha256"],
            bindings["entry"],
            bindings.get("shared_memory_bytes", 0),
        )
        if key not in self.functions:
            self.check("cuCtxSetCurrent", self.context)
            digest = receipt["binary_sha256"]
            if digest not in self.modules:
                module = P()
                data = C.create_string_buffer(binary.read_bytes())
                self.check("cuModuleLoadData", C.byref(module), data)
                self.modules[digest] = module
            function = P()
            self.check(
                "cuModuleGetFunction",
                C.byref(function),
                self.modules[digest],
                bindings["entry"].encode(),
            )
            if key[2] > 49152:
                self.check("cuFuncSetAttribute", function, 8, key[2])
            self.functions[key] = function
        self.environment["device_compiler"] = self.compiler.identity
        return self.functions[key]

    def launch(self, function, args, bindings):
        if self.closed:
            raise RuntimeError("Closed device session")
        self.check("cuCtxSetCurrent", self.context)
        blocks, threads = bindings["blocks"], bindings["threads"]
        shared = bindings.get("shared_memory_bytes", 0)
        if (
            not 0 < blocks <= self.max_blocks
            or not 0 < threads <= self.max_threads
            or not 0 <= shared <= 0xFFFFFFFF
        ):
            raise ValueError("CUDA launch exceeds device limits")
        if len(args) != len(bindings["buffers"]):
            raise ValueError("CUDA argument count mismatch")
        for allocation in args:
            self._owned(allocation)
        pointers = [U64(a.pointer) for a in args]
        parameters = (P * len(pointers))(*(C.addressof(p) for p in pointers))
        self.check(
            "cuLaunchKernel",
            function,
            blocks,
            1,
            1,
            threads,
            1,
            1,
            shared,
            self.stream,
            parameters,
            None,
        )

    def close(self):
        if self.closed:
            return
        # Cleanup every resource even after a failed asynchronous launch.
        try:
            if self.retained:
                self.api["cuCtxSetCurrent"](self.context)
                if self.stream.value:
                    self.api["cuStreamSynchronize"](self.stream)
                for allocation in self.allocations.values():
                    self.api["cuMemFree_v2"](allocation.pointer)
                for module in self.modules.values():
                    self.api["cuModuleUnload"](module)
                if self.stream.value:
                    self.api["cuStreamDestroy_v2"](self.stream)
                self.api["cuCtxSetCurrent"](self.previous)
                self.api["cuDevicePrimaryCtxRelease_v2"](self.device)
        finally:
            self.allocations.clear()
            self.modules.clear()
            self.functions.clear()
            self.closed = True
