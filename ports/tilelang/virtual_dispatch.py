"""khal-style bounded physical grids over GPU-resident virtual workgroups.

Transform the original TileLang IR before lowering, so singleton folding and
shared-buffer allocation see the actual virtual index. Never rewrite emitted C++.
"""


def virtualize(plan, arrays, physical_limit=256):
    from tilelang import tvm

    if type(physical_limit) is not int or not 1 <= physical_limit <= 65535:
        raise ValueError("Invalid physical workgroup limit")
    indirect = plan.get("indirect")
    if indirect is None:
        return plan
    function = plan["kernel"]
    tir = getattr(tvm, "tirx", None) or tvm.tir
    capacity = plan["blocks"]
    physical = min(capacity, physical_limit)
    name = "NexusDispatch"
    if name in plan["buffer_map"]:
        raise ValueError("Virtual dispatch argument collision")
    initial = arrays[indirect["buffer"]]
    word = indirect["word"]
    if type(word) is not int or word < 0 or word + 3 > initial.size:
        raise ValueError("Virtual dispatch outside storage")
    handle = tir.Var(name + "_handle", "handle")
    grid = tir.decl_buffer((initial.size,), "uint32", name=name, strides=[1])
    found = 0

    def rewrite(node):
        nonlocal found
        if (
            not isinstance(node, tir.For)
            or node.thread_binding is None
            or node.thread_binding.thread_tag != "blockIdx.x"
        ):
            return None
        found += 1
        if int(node.extent) != capacity or int(node.min) != 0:
            raise ValueError("Virtual dispatch capacity differs from kernel IR")
        wave = tir.Var("nexus_virtual_wave", "int32")
        index = node.loop_var + wave * physical
        # All lanes execute the same number of iterations. The producer grid is
        # immutable during this dispatch; clamp to the compiled safe capacity.
        count = tir.Min(
            tir.Cast("int64", tir.BufferLoad(grid, [word])),
            tir.IntImm("int64", capacity),
        )
        count = tir.Max(
            count - tir.Cast("int64", node.loop_var), tir.IntImm("int64", 0)
        )
        waves = tir.Cast("int32", (count + physical - 1) // physical)
        fence = tir.Evaluate(
            tir.call_intrin("int32", "tirx.tvm_storage_sync", "shared")
        )
        wrapped = 0

        def wrap(block):
            nonlocal wrapped
            simt = (
                isinstance(block, tir.AttrStmt) and block.attr_key == "tl.simtvf_scope"
            )
            gpu = (
                isinstance(block, tir.SBlock)
                and "tl.launch_threads" in block.annotations
            )
            if not (simt or gpu):
                return None
            wrapped += 1
            body = tir.stmt_functor.substitute(block.body, {node.loop_var: index})
            loop = tir.For(
                wave, 0, waves, tir.ForKind.SERIAL, tir.SeqStmt([body, fence])
            )
            if simt:
                return tir.AttrStmt(block.node, block.attr_key, block.value, loop)
            return tir.SBlock(
                block.iter_vars,
                block.reads,
                block.writes,
                block.name_hint,
                loop,
                block.init,
                block.alloc_buffers,
                block.match_buffers,
                block.annotations,
            )

        body = tir.stmt_functor.ir_transform(
            node.body, None, wrap, ["tirx.SBlock", "tirx.AttrStmt"]
        )
        if wrapped != 1:
            raise ValueError("Expected one TileLang launch root for virtual dispatch")
        return tir.For(
            node.loop_var,
            0,
            physical,
            node.kind,
            body,
            node.thread_binding,
            node.annotations,
        )

    body = tir.stmt_functor.ir_transform(function.body, None, rewrite, ["tirx.For"])
    if found != 1:
        raise ValueError("Expected one original blockIdx.x loop for virtual dispatch")
    buffers = dict(function.buffer_map)
    buffers[handle] = grid
    function = tir.PrimFunc(
        [*function.params, handle], body, function.ret_type, buffers, function.attrs
    )
    result = dict(
        plan,
        kernel=function,
        blocks=physical,
        buffer_map={**plan["buffer_map"], name: indirect["buffer"]},
        virtual_grid={**indirect, "capacity_blocks": capacity},
    )
    del result["indirect"]
    return result
