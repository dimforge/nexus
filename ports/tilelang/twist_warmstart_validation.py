"""Exact write ownership of native compact-tile scaling and writeback."""


def scale_masks(l, base, count, mutable, floats):
    t, h = l["twist"], l["header"]
    mutable[base + h["off_b"] + 3] = True
    floats[base + h["off_b"] + 3] = True
    mutable[base + t["friction"] : base + t["friction"] + 2] = True
    floats[base + t["friction"] : base + t["friction"] + 2] = True
    mutable[base + t["friction"] + 3] = True
    for k in range(4):
        mutable[base + t["normal_impulse"] + k] = True
        if k < count:
            floats[base + t["normal_impulse"] + k] = True
        else:
            for field in ("dist", "normal_vel", "radius"):
                mutable[base + t[field] + k] = True
