"""Launch adapters for native narrow-phase reset and contact plans."""

import tilelang.language as T
from shaders.broad_phase.narrow_phase import gpu_reset_narrow_phase, gpu_contact_plan

BUFFERS = (
    "PairCount",
    "PfmCount",
    "Plan",
    "PfmSortLen",
    "ContactsIndirect",
    "PfmIndirect",
    "MbIndirect",
    "BatchIds",
)
OPERATIONS = ("reset_narrow", "contact_plan")


def make_kernel(layout, case, target):
    reset = gpu_reset_narrow_phase(case["count"])
    publish = gpu_contact_plan(layout)
    groups, threads = case["groups"], case["threads"]
    n0 = case["buffer_words"]["PairCount"]
    n1 = case["buffer_words"]["PfmCount"]
    n2 = case["buffer_words"]["Plan"]
    n3 = case["buffer_words"]["PfmSortLen"]
    n4 = case["buffer_words"]["ContactsIndirect"]
    n5 = case["buffer_words"]["PfmIndirect"]
    n6 = case["buffer_words"]["MbIndirect"]
    n7 = case["buffer_words"]["BatchIds"]

    @T.macro
    def one(
        PairCount,
        PfmCount,
        Plan,
        PfmSortLen,
        ContactsIndirect,
        PfmIndirect,
        MbIndirect,
        BatchIds,
        gid,
    ):
        if case["kind"] == "reset_narrow":
            reset(PfmCount, gid)
        else:
            publish(
                PairCount,
                PfmCount,
                Plan,
                PfmSortLen,
                ContactsIndirect,
                PfmIndirect,
                MbIndirect,
                BatchIds,
            )

    if target == "c":

        @T.prim_func
        def kernel(
            PairCount: T.Tensor((n0,), "uint32"),
            PfmCount: T.Tensor((n1,), "uint32"),
            Plan: T.Tensor((n2,), "uint32"),
            PfmSortLen: T.Tensor((n3,), "uint32"),
            ContactsIndirect: T.Tensor((n4,), "uint32"),
            PfmIndirect: T.Tensor((n5,), "uint32"),
            MbIndirect: T.Tensor((n6,), "uint32"),
            BatchIds: T.Tensor((n7,), "uint32"),
        ):
            with T.Kernel(1):
                for gid in T.serial(groups * threads):
                    one(
                        PairCount,
                        PfmCount,
                        Plan,
                        PfmSortLen,
                        ContactsIndirect,
                        PfmIndirect,
                        MbIndirect,
                        BatchIds,
                        gid,
                    )
    elif target == "ascend":
        import tilelang.ascend.language as A

        @A.prim_func
        def kernel(
            PairCount: A.Tensor((n0,), "uint32"),
            PfmCount: A.Tensor((n1,), "uint32"),
            Plan: A.Tensor((n2,), "uint32"),
            PfmSortLen: A.Tensor((n3,), "uint32"),
            ContactsIndirect: A.Tensor((n4,), "uint32"),
            PfmIndirect: A.Tensor((n5,), "uint32"),
            MbIndirect: A.Tensor((n6,), "uint32"),
            BatchIds: A.Tensor((n7,), "uint32"),
        ):
            with A.Kernel(groups) as group:
                with A.SimtVF(threads=threads):
                    one(
                        PairCount,
                        PfmCount,
                        Plan,
                        PfmSortLen,
                        ContactsIndirect,
                        PfmIndirect,
                        MbIndirect,
                        BatchIds,
                        group * threads + A.get_thread_binding(),
                    )
    else:

        @T.prim_func
        def kernel(
            PairCount: T.Tensor((n0,), "uint32"),
            PfmCount: T.Tensor((n1,), "uint32"),
            Plan: T.Tensor((n2,), "uint32"),
            PfmSortLen: T.Tensor((n3,), "uint32"),
            ContactsIndirect: T.Tensor((n4,), "uint32"),
            PfmIndirect: T.Tensor((n5,), "uint32"),
            MbIndirect: T.Tensor((n6,), "uint32"),
            BatchIds: T.Tensor((n7,), "uint32"),
        ):
            with T.Kernel(groups, threads=threads) as group:
                one(
                    PairCount,
                    PfmCount,
                    Plan,
                    PfmSortLen,
                    ContactsIndirect,
                    PfmIndirect,
                    MbIndirect,
                    BatchIds,
                    group * threads + T.get_thread_binding(),
                )

    return kernel


def make_program(layout, case, target):
    return [
        dict(
            name=case["kind"],
            kernel=make_kernel(layout, case, target),
            buffer_map={n: n for n in BUFFERS},
            blocks=case["groups"],
            threads=case["threads"],
        )
    ]
