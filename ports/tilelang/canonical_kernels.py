"""Native contact canonicalization using cooperative stable radix stages."""

import tilelang.language as T
from shaders.dynamics.canonical_order import gpu_contact_sort_keys, gpu_permute_contacts
import radix_kernels as radix

BASE_BUFFERS = [
    "Contacts",
    "Output",
    "InOrder",
    "Keys",
    "Values",
    "SortCount",
    "Plan",
    "BatchIds",
    "Selector",
]
BUFFERS = (
    tuple(BASE_BUFFERS)
    + (
        "SortKeys",
        "PongKeys",
        "PongValues",
        "RadixAux",
        "RadixOutAux",
        "RadixBatchIds",
        "Counts",
        "Reduced",
        "SortGrid",
        "ReduceGrid",
        "Config",
        "InitConfig",
    )
    + tuple(f"Config{i}" for i in range(8))
)


def make_kernel(layout, case, target, phase, selector=None):
    if layout["default_contact"] != [0] * layout["contact_words"]:
        raise ValueError("Native default contact changed")
    groups = case["groups"]
    sizes = case["buffer_words"]
    keys = gpu_contact_sort_keys(layout)
    permute = gpu_permute_contacts(layout)

    @T.macro
    def one(
        Contacts,
        Output,
        InOrder,
        Keys,
        Values,
        SortCount,
        Plan,
        BatchIds,
        Selector,
        gid,
    ):
        if phase == "keys":
            if selector is None:
                keys(
                    Contacts,
                    Plan,
                    InOrder,
                    Keys,
                    Values,
                    Selector,
                    SortCount,
                    BatchIds,
                    gid,
                    groups,
                )
            else:
                selected = T.alloc_local((1,), "uint32")
                selected[0] = T.uint32(selector)
                keys(
                    Contacts,
                    Plan,
                    InOrder,
                    Keys,
                    Values,
                    selected,
                    SortCount,
                    BatchIds,
                    gid,
                    groups,
                )
        else:
            permute(Contacts, Output, InOrder, Plan, gid, groups)

    if target == "c":

        @T.prim_func
        def kernel(
            Contacts: T.Tensor((sizes["Contacts"],), "uint32"),
            Output: T.Tensor((sizes["Output"],), "uint32"),
            InOrder: T.Tensor((sizes["InOrder"],), "uint32"),
            Keys: T.Tensor((sizes["Keys"],), "uint32"),
            Values: T.Tensor((sizes["Values"],), "uint32"),
            SortCount: T.Tensor((sizes["SortCount"],), "uint32"),
            Plan: T.Tensor((sizes["Plan"],), "uint32"),
            BatchIds: T.Tensor((sizes["BatchIds"],), "uint32"),
            Selector: T.Tensor((sizes["Selector"],), "uint32"),
        ):
            with T.Kernel(1):
                for gid in T.serial(groups * 64):
                    one(
                        Contacts,
                        Output,
                        InOrder,
                        Keys,
                        Values,
                        SortCount,
                        Plan,
                        BatchIds,
                        Selector,
                        gid,
                    )
    elif target == "ascend":
        import tilelang.ascend.language as A

        @A.prim_func
        def kernel(
            Contacts: A.Tensor((sizes["Contacts"],), "uint32"),
            Output: A.Tensor((sizes["Output"],), "uint32"),
            InOrder: A.Tensor((sizes["InOrder"],), "uint32"),
            Keys: A.Tensor((sizes["Keys"],), "uint32"),
            Values: A.Tensor((sizes["Values"],), "uint32"),
            SortCount: A.Tensor((sizes["SortCount"],), "uint32"),
            Plan: A.Tensor((sizes["Plan"],), "uint32"),
            BatchIds: A.Tensor((sizes["BatchIds"],), "uint32"),
            Selector: A.Tensor((sizes["Selector"],), "uint32"),
        ):
            with A.Kernel(groups) as group:
                with A.SimtVF(threads=64):
                    one(
                        Contacts,
                        Output,
                        InOrder,
                        Keys,
                        Values,
                        SortCount,
                        Plan,
                        BatchIds,
                        Selector,
                        group * 64 + A.get_thread_binding(),
                    )
    else:

        @T.prim_func
        def kernel(
            Contacts: T.Tensor((sizes["Contacts"],), "uint32"),
            Output: T.Tensor((sizes["Output"],), "uint32"),
            InOrder: T.Tensor((sizes["InOrder"],), "uint32"),
            Keys: T.Tensor((sizes["Keys"],), "uint32"),
            Values: T.Tensor((sizes["Values"],), "uint32"),
            SortCount: T.Tensor((sizes["SortCount"],), "uint32"),
            Plan: T.Tensor((sizes["Plan"],), "uint32"),
            BatchIds: T.Tensor((sizes["BatchIds"],), "uint32"),
            Selector: T.Tensor((sizes["Selector"],), "uint32"),
        ):
            with T.Kernel(groups, threads=64) as group:
                one(
                    Contacts,
                    Output,
                    InOrder,
                    Keys,
                    Values,
                    SortCount,
                    Plan,
                    BatchIds,
                    Selector,
                    group * 64 + T.get_thread_binding(),
                )

    return kernel


def radix_digits(selector, collider_count=None):
    """Keep full subshape/packed keys; bounded IDs reserve all-ones for inert rows.

    An even digit count leaves the final permutation in InOrder. Rounding to a
    byte also retains a strictly larger sentinel when count is a power of two.
    """
    if collider_count is not None and (
        type(collider_count) is not int or not 0 <= collider_count < 2**32
    ):
        raise ValueError("Invalid canonical collider count")
    if collider_count is None or selector % 4 not in (1, 2):
        return 8
    return 2 * ((max(1, collider_count.bit_length()) + 7) // 8)


def make_program(layout, case, target):
    plans = []
    cache = {}

    def add(phase, selector=None):
        plans.append(
            dict(
                name=f"{phase}_{len(plans)}",
                kernel=make_kernel(layout, case, target, phase, selector),
                buffer_map={n: n for n in BASE_BUFFERS},
                blocks=case["groups"],
                threads=64,
            )
        )

    def sorting_stage(kind, p):
        mapping = {n: n for n in radix.LOCAL_BUFFERS}
        mapping.update(
            NumKeys="SortCount",
            BatchIds="RadixBatchIds",
            Aux="RadixAux",
            OutAux="RadixOutAux",
            Config=f"Config{p}",
        )
        if p == 0:
            mapping.update(Keys="Keys", Values="Values")
        elif p % 2:
            mapping.update(Keys="PongKeys", Values="PongValues")
        else:
            mapping.update(Keys="SortKeys", Values="InOrder")
        if p % 2:
            mapping.update(OutKeys="SortKeys", OutValues="InOrder")
        else:
            mapping.update(OutKeys="PongKeys", OutValues="PongValues")
        rc = dict(
            kind=kind,
            capacity=case["storage"],
            batches=1,
            iterations=1,
            buffer_words={n: case["buffer_words"][v] for n, v in mapping.items()},
        )
        if kind not in cache:
            cache[kind] = radix.make_kernel(layout, rc, target, kind)
        blocks, threads, _ = radix.dimensions(rc, kind)
        plans.append(
            dict(
                name=f"{kind}_{len(plans)}",
                kernel=cache[kind],
                buffer_map=mapping,
                blocks=blocks,
                threads=threads,
            )
        )

    if case["kind"] == "contact_keys":
        add("keys")
    elif case["kind"] == "permute_contacts":
        add("permute")
    else:
        for selector in case["passes"]:
            add("keys", selector)
            sorting_stage("radix_dispatch", 0)
            for p in range(radix_digits(selector, case.get("collider_count"))):
                for kind in radix.STAGES:
                    sorting_stage(kind, p)
        add("permute")
    return plans
