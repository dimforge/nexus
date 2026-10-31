"""Isolated TileLang lowering: TVM compiler globals are never shared by threads."""

import json
from time import perf_counter


def lower_serialized(task):
    ir, target = task
    import tilelang
    from compiler_intrinsics import register_intrinsics
    from kernel_cache import encoded

    register_intrinsics(target)
    kernel = tilelang.tvm.ir.load_json(ir)
    requested = {"kind": "cuda", "arch": "sm_80"} if target == "cuda" else target
    start = perf_counter()
    artifact = tilelang.lower(
        kernel,
        target=requested,
        enable_host_codegen=False,
        enable_device_compile=False,
    )
    return encoded(
        {
            "device_mod": tilelang.tvm.ir.save_json(artifact.device_mod),
            "kernel_source": artifact.kernel_source,
        }
    ), perf_counter() - start


def decode_artifact(payload):
    import tilelang
    from types import SimpleNamespace

    value = json.loads(payload)
    return SimpleNamespace(
        device_mod=tilelang.tvm.ir.load_json(value["device_mod"]),
        kernel_source=value["kernel_source"],
    )
