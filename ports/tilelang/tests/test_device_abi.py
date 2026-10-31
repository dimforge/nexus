"""ctypes ABI/resource tests against explicit host test libraries, never GPUs."""

import ctypes as C
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
from device_ascend import AscendDevice
from device_compiler import ascend_wrapper
from device_cuda import CudaDevice

SHIM = r"""
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
static int live=0, fault=0, launches=0, shared=0;
static void* pending_dst=0; static void* pending_data=0; static size_t pending_size=0;
int test_pending(void){return pending_data!=0;}
int test_live(void){return live;} int test_launches(void){return launches;}
int test_shared(void){return shared;} void test_fault(int v){fault=v;}
int cuInit(unsigned x){return 0;}
int cuDeviceGet(int* d,int i){*d=i;return 0;}
int cuDeviceGetName(char* b,int n,int d){strncpy(b,"TEST_ONLY_CUDA",n);return 0;}
int cuDeviceGetAttribute(int* v,int a,int d){*v=a==1?1024:a==5?2147483647:a==75?8:0;return 0;}
int cuDriverGetVersion(int* v){*v=12000;return 0;}
int cuCtxGetCurrent(void** p){*p=0;return 0;}
int cuDevicePrimaryCtxRetain(void** p,int d){*p=(void*)1;return 0;}
int cuDevicePrimaryCtxRelease_v2(int d){return 0;}
int cuCtxSetCurrent(void* c){return 0;}
int cuStreamCreate(void** s,unsigned f){*s=(void*)2;return 0;}
int cuStreamSynchronize(void* s){
 if(!s && pending_data){memcpy(pending_dst,pending_data,pending_size);free(pending_data);pending_dst=0;pending_data=0;}
 return fault;
}
int cuStreamDestroy_v2(void* s){return 0;}
int cuMemAlloc_v2(uint64_t* p,size_t n){*p=(uintptr_t)malloc(n);live++;return 0;}
int cuMemFree_v2(uint64_t p){
 if(pending_dst==(void*)(uintptr_t)p){free(pending_data);pending_data=0;pending_dst=0;}
 free((void*)(uintptr_t)p);live--;return 0;
}
int cuMemcpyHtoD_v2(uint64_t d,void* s,size_t n){
 if(pending_data)return 778;
 pending_dst=(void*)(uintptr_t)d;pending_data=malloc(n);pending_size=n;
 memcpy(pending_data,s,n);return 0;
}
int cuMemcpyDtoH_v2(void* d,uint64_t s,size_t n){memcpy(d,(void*)(uintptr_t)s,n);return 0;}
int cuMemcpyDtoDAsync_v2(uint64_t d,uint64_t s,size_t n,void* st){memcpy((void*)(uintptr_t)d,(void*)(uintptr_t)s,n);return 0;}
int cuModuleLoadData(void** m,void* d){*m=(void*)3;return 0;}
int cuModuleUnload(void* m){return 0;}
int cuModuleGetFunction(void** f,void* m,const char* n){*f=(void*)4;return 0;}
int cuFuncSetAttribute(void* f,int a,int v){return 0;}
int cuLaunchKernel(void* f,unsigned gx,unsigned gy,unsigned gz,unsigned tx,unsigned ty,unsigned tz,unsigned sm,void* s,void** args,void** extra){
 uint32_t* a=(uint32_t*)(uintptr_t)*(uint64_t*)args[0];a[0]+=gx;shared=sm;launches++;return 0;}
int cuGetErrorString(int e,const char** s){*s="host injected failure";return 0;}
int aclInit(const char* s){return 0;} int aclFinalize(void){return 0;}
int aclrtSetDevice(int d){return 0;} int aclrtResetDevice(int d){return 0;}
int aclrtCreateContext(void** c,int d){*c=(void*)1;return 0;}
int aclrtSetCurrentContext(void* c){return 0;} int aclrtDestroyContext(void* c){return 0;}
int aclrtCreateStream(void** s){*s=(void*)2;return 0;}
int aclrtDestroyStream(void* s){return 0;} int aclrtSynchronizeStream(void* s){return fault;}
int aclrtMalloc(void** p,size_t n,int policy){*p=malloc(n);live++;return 0;}
int aclrtFree(void* p){free(p);live--;return 0;}
int aclrtMemcpy(void* d,size_t max,const void* s,size_t n,int kind){if(n>max)return 99;memcpy(d,s,n);return 0;}
int aclrtGetVersion(int* a,int* b,int* c){*a=9;*b=1;*c=0;return 0;}
const char* aclrtGetSocName(void){return "TEST_ONLY_Ascend950";}
const char* aclGetRecentErrMsg(void){return "host injected failure";}
int test_ascend_launch(unsigned blocks,void* stream,void** args){((uint32_t*)args[0])[0]+=blocks;launches++;return 0;}
"""


class DeviceAbi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        (root / "shim.c").write_text(SHIM)
        cls.path = root / "test-only-driver.so"
        subprocess.run(
            [
                shutil.which("cc"),
                "-shared",
                "-fPIC",
                str(root / "shim.c"),
                "-o",
                str(cls.path),
            ],
            check=True,
            capture_output=True,
        )
        cls.lib = C.CDLL(str(cls.path))
        cls.lib.test_fault.argtypes = [C.c_int]
        cls.lib.test_ascend_launch.argtypes = [
            C.c_uint32,
            C.c_void_p,
            C.POINTER(C.c_void_p),
        ]
        cls.lib.test_ascend_launch.restype = C.c_int

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_cuda_and_ascend_pointer_abi_offsets_and_cleanup(self):
        for kind in (CudaDevice, AscendDevice):
            with self.subTest(backend=kind.__name__):
                device = kind(library=str(self.path))
                self.assertIn("TEST_ONLY", device.environment["device"])
                before = np.arange(12, dtype=np.uint32)
                a = device.upload(before)
                b = device.upload(np.full(12, 0xDEADBEEF, np.uint32))
                device.copy(b, 3, a, 2, 5)
                np.testing.assert_array_equal(
                    device.read(b, 0, 12),
                    [0xDEADBEEF] * 3 + list(range(2, 7)) + [0xDEADBEEF] * 4,
                )
                bindings = {
                    "entry": "test",
                    "buffers": ["A"],
                    "blocks": 3,
                    "threads": 64,
                    "shared_memory_bytes": 256,
                }
                function = (
                    C.c_void_p(4) if kind is CudaDevice else self.lib.test_ascend_launch
                )
                device.launch(function, [a], bindings)
                self.assertEqual(device.read(a, 0, 1)[0], 3)
                if kind is CudaDevice:
                    self.assertEqual(self.lib.test_shared(), 256)
                with self.assertRaises(ValueError):
                    device.read(a, 11, 2)
                with self.assertRaises(ValueError):
                    device.write(a, -1, np.zeros(1, np.uint32))
                device.free(b)
                with self.assertRaises(ValueError):
                    device.read(b, 0, 1)
                device.close()
                device.close()
                self.assertEqual(self.lib.test_live(), 0)

    def test_pageable_upload_completes_before_nonblocking_stream_use(self):
        device = CudaDevice(library=str(self.path))
        try:
            allocation = device.upload(np.array([37, 29], np.uint32))
            self.assertEqual(self.lib.test_pending(), 0)
            np.testing.assert_array_equal(device.read(allocation, 0, 2), [37, 29])
            device.write(allocation, 1, np.array([41], np.uint32))
            self.assertEqual(self.lib.test_pending(), 0)
            np.testing.assert_array_equal(device.read(allocation, 0, 2), [37, 41])
        finally:
            device.close()

    def test_async_failure_surfaces_and_close_still_releases(self):
        for kind in (CudaDevice, AscendDevice):
            device = kind(library=str(self.path))
            device.upload(np.arange(4, dtype=np.uint32))
            self.lib.test_fault(777)
            try:
                with self.assertRaisesRegex(RuntimeError, "777"):
                    device.synchronize()
                device.close()
                self.assertEqual(self.lib.test_live(), 0)
            finally:
                self.lib.test_fault(0)

    def test_ascend_wrapper_argument_order_and_unrecognized_abi(self):
        source = 'extern "C" __global__ __vector__ void k(__gm__ uint32_t* B, __gm__ uint32_t* A) {}'
        bindings = {"entry": "k", "buffers": ["B", "A"], "shared_memory_bytes": 128}
        code = ascend_wrapper(source, bindings)
        self.assertIn(
            "k<<<blocks, 128, stream>>>(reinterpret_cast<uint32_t*>(args[0]), reinterpret_cast<uint32_t*>(args[1]))",
            code,
        )
        self.assertIn("aclrtGetLastError(ACL_RT_THREAD_LEVEL)", code)
        with self.assertRaisesRegex(ValueError, "parameter"):
            ascend_wrapper(source.replace("uint32_t", "double"), bindings)
        with self.assertRaisesRegex(ValueError, "count"):
            ascend_wrapper(source, dict(bindings, buffers=["A"]))

        with self.assertRaisesRegex(ValueError, "order"):
            ascend_wrapper(source, dict(bindings, buffers=["A", "B"]))
        typed = ascend_wrapper(source.replace("uint32_t* A", "float* A"), bindings)
        self.assertIn("reinterpret_cast<float*>(args[1])", typed)
