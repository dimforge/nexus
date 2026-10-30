import numpy as np


def check(l, c, b, a):
    n, nb, cap = c["colliders"], c["batches"], c["capacity"]
    f, af = l["lbvh_node"], l["aabb"]
    w = l["lbvh_node_words"]
    gf = l["groups"]
    tree = b["Tree"].reshape(-1, w)
    groups = b["Groups"].reshape(-1, l["group_words"])
    filters = b["Filters"].reshape(-1, 2)
    expected = set()
    for batch in range(nb):
        leaves = (
            tree[2 * batch * cap + n - 1 : 2 * batch * cap + 2 * n - 1]
            if n
            else tree[:0]
        )
        ids = leaves[:, f["left"]].astype(int) * nb + batch
        floats = leaves.view(np.float32)
        mins = floats[:, f["aabb"] + af["mins"] : f["aabb"] + af["mins"] + 3]
        maxs = floats[:, f["aabb"] + af["maxs"] : f["aabb"] + af["maxs"] + 3]
        for i in range(n):
            hits = np.flatnonzero(
                np.all(
                    (mins[i] - np.float32(0.002) <= maxs)
                    & (maxs[i] + np.float32(0.002) >= mins),
                    axis=1,
                )
            )
            gi = groups[ids[i]]
            fi = filters[ids[i]]
            for j in hits:
                if i >= j:
                    continue
                gj = groups[ids[j]]
                fj = filters[ids[j]]
                left = bool(gi[gf["memberships"]] & gj[gf["filter"]])
                right = bool(gj[gf["memberships"]] & gi[gf["filter"]])
                either = gi[gf["test_mode"]] == 1 and gj[gf["test_mode"]] == 1
                if (
                    (left or right if either else left and right)
                    and fi[0] != fj[0]
                    and (fi[1] == 0 or fi[1] != fj[1])
                ):
                    expected.add(tuple(sorted((int(ids[i]), int(ids[j])))))
    assert int(a["PairCount"][0]) == len(expected), (
        int(a["PairCount"][0]),
        len(expected),
    )
    count = min(len(expected), c["pair_capacity"])
    rows = a["Pairs"].reshape(-1, l["pair_words"])
    active = rows[:count, l["pair_colliders"] : l["pair_colliders"] + 2]
    actual = set(map(tuple, active.tolist()))
    assert len(actual) == count and actual <= expected
    np.testing.assert_array_equal(
        a["Pairs"][count * l["pair_words"] :], b["Pairs"][count * l["pair_words"] :]
    )
