"""Target launch adapters for native contact buffers; no host physics."""

import tilelang.language as T
from shaders.dynamics.tile_kernels import gpu_prepare_constraints


def make_kernel(layout, case, target):
    n = case["count"]
    one = gpu_prepare_constraints(layout, case)
    n0 = case["storage"] * layout["link_words"]
    n1 = case["storage"] * layout["contact_words"]
    n2 = case["bodies"] * layout["world_mass_words"]
    n3 = case["bodies"] * layout["pose_words"]
    n4 = case["bodies"] * layout["velocity_words"]
    n5 = case["states"] * layout["state_words"]
    n6 = case["old_tiles"] * layout["tile_words"]
    n7 = case["tiles"] * layout["tile_words"]
    if target == "c":

        @T.prim_func
        def kernel(
            Links: T.Tensor((n0,), "uint32"),
            Contacts: T.Tensor((n1,), "uint32"),
            WorldMass: T.Tensor((n2,), "uint32"),
            SolverPoses: T.Tensor((n3,), "uint32"),
            Velocities: T.Tensor((n4,), "uint32"),
            States: T.Tensor((n5,), "uint32"),
            OldTiles: T.Tensor((n6,), "uint32"),
            Tiles: T.Tensor((n7,), "uint32"),
        ):
            with T.Kernel(1):
                for i in T.serial(max(n, 1)):
                    one(
                        Links,
                        Contacts,
                        WorldMass,
                        SolverPoses,
                        Velocities,
                        States,
                        OldTiles,
                        Tiles,
                        i,
                    )
    elif target == "ascend":
        import tilelang.ascend.language as A

        @A.prim_func
        def kernel(
            Links: A.Tensor((n0,), "uint32"),
            Contacts: A.Tensor((n1,), "uint32"),
            WorldMass: A.Tensor((n2,), "uint32"),
            SolverPoses: A.Tensor((n3,), "uint32"),
            Velocities: A.Tensor((n4,), "uint32"),
            States: A.Tensor((n5,), "uint32"),
            OldTiles: A.Tensor((n6,), "uint32"),
            Tiles: A.Tensor((n7,), "uint32"),
        ):
            with A.Kernel(max((n + 63) // 64, 1)) as block:
                with A.SimtVF(threads=64):
                    one(
                        Links,
                        Contacts,
                        WorldMass,
                        SolverPoses,
                        Velocities,
                        States,
                        OldTiles,
                        Tiles,
                        block * 64 + A.get_thread_binding(),
                    )
    elif target in ("metal", "cuda"):

        @T.prim_func
        def kernel(
            Links: T.Tensor((n0,), "uint32"),
            Contacts: T.Tensor((n1,), "uint32"),
            WorldMass: T.Tensor((n2,), "uint32"),
            SolverPoses: T.Tensor((n3,), "uint32"),
            Velocities: T.Tensor((n4,), "uint32"),
            States: T.Tensor((n5,), "uint32"),
            OldTiles: T.Tensor((n6,), "uint32"),
            Tiles: T.Tensor((n7,), "uint32"),
        ):
            with T.Kernel(max((n + 63) // 64, 1), threads=64) as block:
                one(
                    Links,
                    Contacts,
                    WorldMass,
                    SolverPoses,
                    Velocities,
                    States,
                    OldTiles,
                    Tiles,
                    block * 64 + T.get_thread_binding(),
                )
    else:
        raise ValueError(f"Unsupported target {target}")
    return kernel
