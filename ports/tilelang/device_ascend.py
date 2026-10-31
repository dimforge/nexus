"""AscendCL resident execution for the pinned Ascend 950 SIMT compiler profile.

Uses real ACL allocation/copies and Bisheng-compiled host launch wrappers. This
adapter is prepared without hardware; vendor compilation and execution are still
required to qualify the actual CANN/device combination.
"""

import ctypes as C
import os
import threading
from pathlib import Path

import numpy as np
from device_compiler import DeviceCompiler
from device_cuda import Allocation, words_array

P, I32, U32, SIZE = C.c_void_p, C.c_int32, C.c_uint32, C.c_size_t


class AscendDevice:
    def __init__(self, ordinal=0, *, library=None):
        if type(ordinal) is not int or ordinal < 0:
            raise ValueError("Invalid Ascend device ordinal")
        self.ordinal, self.thread = ordinal, threading.get_ident()
        self.allocations, self.libraries = {}, {}
        self.compiler = None
        self.context, self.stream = P(), P()
        self.initialized = self.selected = self.closed = False
        toolkit = os.environ.get("ASCEND_HOME_PATH")
        name = library or (
            str(Path(toolkit) / "lib64/libascendcl.so") if toolkit else "libascendcl.so"
        )
        try:
            self.lib = C.CDLL(name)
        except OSError as error:
            raise RuntimeError(
                "AscendCL unavailable; install/source CANN on the NPU host. No CPU or compiler-stub fallback is used"
            ) from error
        self._bind()
        try:
            self.check("aclInit", None)
            self.initialized = True
            self.check("aclrtSetDevice", ordinal)
            self.selected = True
            self.check("aclrtCreateContext", C.byref(self.context), ordinal)
            self.check("aclrtCreateStream", C.byref(self.stream))
            soc = self.api["aclrtGetSocName"]()
            if not soc:
                raise RuntimeError("Ascend device identification failed")
            major, minor, patch = I32(), I32(), I32()
            self.check(
                "aclrtGetVersion", C.byref(major), C.byref(minor), C.byref(patch)
            )
            self.environment = {
                "device_backend": "ascend",
                "device": soc.decode(),
                "device_ordinal": ordinal,
                "acl_version": [major.value, minor.value, patch.value],
                "driver_library": name,
                "compiler_profile": "dav-3510 (Ascend 950 SIMT)",
                "runtime": "AscendCL, ordered native stream",
            }
        except Exception:
            self.close()
            raise

    def _bind(self):
        signatures = {
            "aclInit": [C.c_char_p],
            "aclFinalize": [],
            "aclrtSetDevice": [I32],
            "aclrtResetDevice": [I32],
            "aclrtCreateContext": [C.POINTER(P), I32],
            "aclrtSetCurrentContext": [P],
            "aclrtDestroyContext": [P],
            "aclrtCreateStream": [C.POINTER(P)],
            "aclrtDestroyStream": [P],
            "aclrtSynchronizeStream": [P],
            "aclrtMalloc": [C.POINTER(P), SIZE, I32],
            "aclrtFree": [P],
            "aclrtMemcpy": [P, SIZE, P, SIZE, I32],
            "aclrtGetVersion": [C.POINTER(I32), C.POINTER(I32), C.POINTER(I32)],
        }
        self.api = {}
        for name, args in signatures.items():
            f = getattr(self.lib, name)
            f.argtypes, f.restype = args, I32
            self.api[name] = f
        for name in ("aclrtGetSocName", "aclGetRecentErrMsg"):
            f = getattr(self.lib, name)
            f.argtypes, f.restype = [], C.c_char_p
            self.api[name] = f

    def check(self, name, *args):
        if threading.get_ident() != self.thread:
            raise RuntimeError("Ascend session used from a different thread")
        code = self.api[name](*args)
        if code:
            detail = self.api["aclGetRecentErrMsg"]()
            raise RuntimeError(
                f"{name}: AscendCL error {code}: {(detail or b'no detail').decode(errors='replace')}"
            )

    def _owned(self, allocation):
        if (
            self.closed
            or allocation.owner is not self
            or self.allocations.get(allocation.pointer) is not allocation
        ):
            raise ValueError("Foreign or released Ascend allocation")
        self.check("aclrtSetCurrentContext", self.context)

    def upload(self, words):
        words = words_array(words)
        if not words.nbytes:
            raise ValueError("Empty physical device allocation")
        self.check("aclrtSetCurrentContext", self.context)
        ptr = P()
        self.check(
            "aclrtMalloc", C.byref(ptr), words.nbytes, 2
        )  # ACL_MEM_MALLOC_NORMAL_ONLY
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
        self.check("aclrtFree", allocation.pointer)
        del self.allocations[allocation.pointer]

    def synchronize(self):
        if self.closed:
            raise RuntimeError("Closed Ascend session")
        self.check("aclrtSetCurrentContext", self.context)
        self.check("aclrtSynchronizeStream", self.stream)

    def read(self, allocation, start, count):
        self._owned(allocation)
        pointer, size = allocation.region(start, count)
        self.synchronize()
        result = np.empty(count, np.uint32)
        if size:
            self.check("aclrtMemcpy", result.ctypes.data, size, pointer, size, 2)
        return result

    def write(self, allocation, start, words):
        self._owned(allocation)
        words = words_array(words)
        pointer, size = allocation.region(start, len(words))
        self.synchronize()
        if size:
            self.check("aclrtMemcpy", pointer, size, words.ctypes.data, size, 1)

    def copy(self, destination, dst_start, source, src_start, count):
        self._owned(destination)
        self._owned(source)
        dst, size = destination.region(dst_start, count)
        src, _ = source.region(src_start, count)
        # Growth may relocate regions at word offsets, not 64-byte boundaries.
        # Synchronous D2D permits these offsets without the older async API's
        # address-alignment restriction. No host staging or physics substitution.
        self.synchronize()
        if size:
            self.check("aclrtMemcpy", dst, size, src, size, 3)

    def compile(self, source, bindings, directory):
        if self.compiler is None:
            self.compiler = DeviceCompiler("ascend")
        binary, receipt = self.compiler.compile(source, bindings, directory)
        key = receipt["binary_sha256"]
        if key not in self.libraries:
            lib = C.CDLL(str(binary))
            lib.nexus_launch.argtypes = [U32, P, C.POINTER(P)]
            lib.nexus_launch.restype = I32
            self.libraries[key] = lib
        self.environment["device_compiler"] = self.compiler.identity
        return self.libraries[key].nexus_launch

    def launch(self, function, args, bindings):
        if self.closed:
            raise RuntimeError("Closed device session")
        self.check("aclrtSetCurrentContext", self.context)
        if not 0 < bindings["blocks"] <= 65535 or not 0 < bindings["threads"] <= 1024:
            raise ValueError("Ascend launch exceeds the prepared SIMT profile")
        if len(args) != len(bindings["buffers"]):
            raise ValueError("Ascend argument count mismatch")
        for allocation in args:
            self._owned(allocation)
        pointers = (P * len(args))(*(a.pointer for a in args))
        code = function(bindings["blocks"], self.stream, pointers)
        if code:
            raise RuntimeError(f"Ascend kernel launch failed: {code}")

    def close(self):
        if self.closed:
            return
        try:
            if self.context.value:
                self.api["aclrtSetCurrentContext"](self.context)
                if self.stream.value:
                    self.api["aclrtSynchronizeStream"](self.stream)
                for allocation in self.allocations.values():
                    self.api["aclrtFree"](allocation.pointer)
                if self.stream.value:
                    self.api["aclrtDestroyStream"](self.stream)
                self.api["aclrtDestroyContext"](self.context)
            if self.selected:
                self.api["aclrtResetDevice"](self.ordinal)
            if self.initialized:
                self.api["aclFinalize"]()
        finally:
            self.allocations.clear()
            self.libraries.clear()
            self.closed = True
