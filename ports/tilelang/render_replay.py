"""Live native renderer data with exact color copies and independently checked transforms."""

import numpy as np
from stage_replay import parse_args, run_stage

BUFFERS = ("Poses", "Descriptors", "Count", "Positions", "Deformations", "Colors")


def validate_case(l, c, b):
    if (
        c["kind"] != "prep_render"
        or any(
            type(c[k]) is not int or not 0 <= c[k] < 2**27
            for k in ("count", "storage", "bodies", "groups")
        )
        or c["count"] > c["storage"]
        or not c["bodies"]
        or c["groups"] * 64 < c["count"]
    ):
        raise ValueError("Invalid render dimensions")
    sizes = {
        "Poses": c["bodies"] * l["pose_words"],
        "Descriptors": c["storage"] * l["descriptor_words"],
        "Count": 2,
        "Positions": c["storage"] * 3,
        "Deformations": c["storage"] * 9,
        "Colors": c["storage"] * 4,
    }
    if (
        c["buffer_words"] != sizes
        or set(b) != set(BUFFERS)
        or any(v.dtype != np.uint32 or v.shape != (sizes[n],) for n, v in b.items())
        or b["Count"][0] != c["count"]
    ):
        raise ValueError("Invalid render storage")
    d = l["descriptor"]
    for i in range(c["count"]):
        row = b["Descriptors"][
            i * l["descriptor_words"] : (i + 1) * l["descriptor_words"]
        ]
        index = int(row[d["pose_index"]])
        if index >= c["bodies"]:
            raise ValueError("Invalid render pose index")
        pose = b["Poses"][index * l["pose_words"] : (index + 1) * l["pose_words"]].view(
            np.float32
        )
        f = row.view(np.float32)
        for a, offset in ((pose, 0), (f, d["local_pose"])):
            q = a[offset + l["pose_rotation"] : offset + l["pose_rotation"] + 4]
            translation = a[
                offset + l["pose_translation"] : offset + l["pose_translation"] + 3
            ]
            if (
                not np.isfinite(q).all()
                or abs(np.linalg.norm(q.astype(float)) - 1) > 2e-6
                or not np.isfinite(translation).all()
            ):
                raise ValueError("Invalid render pose")
        if not np.isfinite(f[d["scale"] : d["scale"] + 3]).all():
            raise ValueError("Invalid render scale")


def check_outputs(l, c, b, a, e, k):
    validate_case(l, c, b)
    errors = {}
    for name in BUFFERS:
        if (
            a[name].dtype != np.uint32
            or e[name].dtype != np.uint32
            or a[name].shape != b[name].shape
            or e[name].shape != b[name].shape
        ):
            raise ValueError("Invalid render result shape")
        width = {"Positions": 3, "Deformations": 9, "Colors": 4}.get(name, 0)
        end = c["count"] * width
        for result in (a, e):
            np.testing.assert_array_equal(
                result[name][end:], b[name][end:], err_msg=f"Render guard {name}"
            )
        if name in ("Positions", "Deformations"):
            af, ef = (v[name][:end].view(np.float32) for v in (a, e))
            if not np.isfinite(af).all() or not np.isfinite(ef).all():
                raise AssertionError("Nonfinite render output")
            np.testing.assert_allclose(af, ef, atol=k["atol"], rtol=k["rtol"])
            errors[name] = float(np.max(np.abs(af - ef))) if end else 0.0
    d = l["descriptor"]
    for i in range(c["count"]):
        row = b["Descriptors"][
            i * l["descriptor_words"] : (i + 1) * l["descriptor_words"]
        ]
        scale = row.view(np.float32)[d["scale"] : d["scale"] + 3].astype(float)
        for result in (a, e):
            np.testing.assert_array_equal(
                result["Colors"][i * 4 : i * 4 + 4],
                row[d["color"] : d["color"] + 4],
                err_msg="Exact render color",
            )
            matrix = (
                result["Deformations"][i * 9 : i * 9 + 9]
                .view(np.float32)
                .astype(float)
                .reshape(3, 3)
                .T
            )
            np.testing.assert_allclose(
                matrix.T @ matrix,
                np.diag(scale * scale),
                atol=1e-5,
                rtol=2e-5,
                err_msg="Scaled orthogonal render axes",
            )
    return errors


def main():
    from render_kernels import make_kernel

    run_stage(
        parse_args(["prep_render"], "prep_render"),
        oracle="render",
        buffers=BUFFERS,
        make_kernel=make_kernel,
        validate_case=validate_case,
        check_outputs=check_outputs,
        launch=lambda c: (c["groups"], 64, c["iterations"]),
    )


if __name__ == "__main__":
    main()
