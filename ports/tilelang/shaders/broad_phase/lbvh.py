"""Counterpart of broad_phase/lbvh.rs; shared scratch layout only."""

import tilelang.language as T


def scratch_start(BatchIds, fields, batch_id):
    return batch_id * T.cast(BatchIds[fields["colliders_batch_capacity"]], "int32")
