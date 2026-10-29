"""Target adapters for Rust-mirrored contact persistence entry points."""

import tilelang.language as T
from shaders.dynamics.contact_links import (
    gpu_init_contact_links,
    gpu_match_contact_links,
    gpu_seed_colors_from_warmstart,
)


def make_kernel(layout, case, target):
    n = case["count"]
    init = gpu_init_contact_links(layout, case)
    match = gpu_match_contact_links(layout, case)
    seed = gpu_seed_colors_from_warmstart(layout, case)
    do_init = case["kind"] in ("init_links", "contact_persistence")
    do_match = case["kind"] in ("match_links", "contact_persistence")
    do_seed = case["kind"] in ("seed_colors", "contact_persistence")

    @T.macro
    def one(
        Contacts,
        Links,
        BodyIsMultibody,
        OldCounts,
        OldIds,
        OldLinks,
        OldIndices,
        States,
        ColliderPoses,
        Params,
        OldColors,
        Colors,
        Colored,
        i,
    ):
        if do_init:
            init(Contacts, Links, BodyIsMultibody, i)
        if do_match:
            match(
                Contacts,
                Links,
                OldCounts,
                OldIds,
                OldLinks,
                OldIndices,
                States,
                ColliderPoses,
                Params,
                i,
            )
        if do_seed:
            seed(Links, OldColors, Colors, Colored, i)

    if n == 0:

        @T.macro
        def empty(
            Contacts,
            Links,
            BodyIsMultibody,
            OldCounts,
            OldIds,
            OldLinks,
            OldIndices,
            States,
            ColliderPoses,
            Params,
            OldColors,
            Colors,
            Colored,
            i,
        ):
            T.evaluate(0)

        one = empty
    n0 = case["storage"] * layout["contact_words"]
    n1 = case["storage"] * layout["link_words"]
    n2 = case["bodies"]
    n3 = case["bodies"]
    n4 = case["ids_storage"]
    n5 = case["old_storage"] * layout["link_words"]
    n6 = case["old_storage"]
    n7 = case["state_storage"] * layout["state_words"]
    n8 = case["colliders"] * layout["pose_words"]
    n9 = layout["params_words"]
    n10 = case["old_storage"]
    n11 = case["storage"]
    n12 = case["storage"]
    if target == "c":

        @T.prim_func
        def kernel(
            Contacts: T.Tensor((n0,), "uint32"),
            Links: T.Tensor((n1,), "uint32"),
            BodyIsMultibody: T.Tensor((n2,), "uint32"),
            OldCounts: T.Tensor((n3,), "uint32"),
            OldIds: T.Tensor((n4,), "uint32"),
            OldLinks: T.Tensor((n5,), "uint32"),
            OldIndices: T.Tensor((n6,), "uint32"),
            States: T.Tensor((n7,), "uint32"),
            ColliderPoses: T.Tensor((n8,), "uint32"),
            Params: T.Tensor((n9,), "uint32"),
            OldColors: T.Tensor((n10,), "uint32"),
            Colors: T.Tensor((n11,), "uint32"),
            Colored: T.Tensor((n12,), "uint32"),
        ):
            with T.Kernel(1):
                for i in T.serial(max(n, 1)):
                    one(
                        Contacts,
                        Links,
                        BodyIsMultibody,
                        OldCounts,
                        OldIds,
                        OldLinks,
                        OldIndices,
                        States,
                        ColliderPoses,
                        Params,
                        OldColors,
                        Colors,
                        Colored,
                        i,
                    )
    elif target == "ascend":
        import tilelang.ascend.language as A

        @A.prim_func
        def kernel(
            Contacts: A.Tensor((n0,), "uint32"),
            Links: A.Tensor((n1,), "uint32"),
            BodyIsMultibody: A.Tensor((n2,), "uint32"),
            OldCounts: A.Tensor((n3,), "uint32"),
            OldIds: A.Tensor((n4,), "uint32"),
            OldLinks: A.Tensor((n5,), "uint32"),
            OldIndices: A.Tensor((n6,), "uint32"),
            States: A.Tensor((n7,), "uint32"),
            ColliderPoses: A.Tensor((n8,), "uint32"),
            Params: A.Tensor((n9,), "uint32"),
            OldColors: A.Tensor((n10,), "uint32"),
            Colors: A.Tensor((n11,), "uint32"),
            Colored: A.Tensor((n12,), "uint32"),
        ):
            with A.Kernel(max((n + 63) // 64, 1)) as block:
                with A.SimtVF(threads=64):
                    one(
                        Contacts,
                        Links,
                        BodyIsMultibody,
                        OldCounts,
                        OldIds,
                        OldLinks,
                        OldIndices,
                        States,
                        ColliderPoses,
                        Params,
                        OldColors,
                        Colors,
                        Colored,
                        block * 64 + A.get_thread_binding(),
                    )
    elif target in ("metal", "cuda"):

        @T.prim_func
        def kernel(
            Contacts: T.Tensor((n0,), "uint32"),
            Links: T.Tensor((n1,), "uint32"),
            BodyIsMultibody: T.Tensor((n2,), "uint32"),
            OldCounts: T.Tensor((n3,), "uint32"),
            OldIds: T.Tensor((n4,), "uint32"),
            OldLinks: T.Tensor((n5,), "uint32"),
            OldIndices: T.Tensor((n6,), "uint32"),
            States: T.Tensor((n7,), "uint32"),
            ColliderPoses: T.Tensor((n8,), "uint32"),
            Params: T.Tensor((n9,), "uint32"),
            OldColors: T.Tensor((n10,), "uint32"),
            Colors: T.Tensor((n11,), "uint32"),
            Colored: T.Tensor((n12,), "uint32"),
        ):
            with T.Kernel(max((n + 63) // 64, 1), threads=64) as block:
                one(
                    Contacts,
                    Links,
                    BodyIsMultibody,
                    OldCounts,
                    OldIds,
                    OldLinks,
                    OldIndices,
                    States,
                    ColliderPoses,
                    Params,
                    OldColors,
                    Colors,
                    Colored,
                    block * 64 + T.get_thread_binding(),
                )
    else:
        raise ValueError(f"Unsupported target {target}")
    return kernel
