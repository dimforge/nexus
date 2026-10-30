"""Native compact-tile ABI counterpart of dynamics/twist_tiles.rs."""


def validate_layout(l):
    if (
        l.get("friction_model") != "twist"
        or l["tile_len"] != 32
        or l["max_points"] != 4
    ):
        raise ValueError("Unsupported native twist ABI")
    if l["twist_arm_words"] != 128:
        raise ValueError("Invalid native twist point stride")
    regions = [(offset, 128) for offset in l["header"].values()]
    regions += [
        (offset, 512 if name == "arm_a" else 128) for name, offset in l["twist"].items()
    ]
    regions += [(l["warmstart"], 64 * l["velocity_words"])]
    occupied = set()
    for offset, width in regions:
        if type(offset) is not int or not 0 <= offset <= l["tile_words"] - width:
            raise ValueError("Invalid native twist field")
        words = set(range(offset, offset + width))
        if words & occupied:
            raise ValueError("Overlapping native twist fields")
        occupied |= words


def make_kernel(layout, case, target):
    from kernel_support import launch

    from . import twist_friction

    operation = {
        "twist_effective_mass": twist_friction.compute_effective_masses,
        "twist_warmstart": twist_friction.warmstart,
        "twist_solve": twist_friction.solve,
        "twist_scale": twist_friction.scale_impulses,
    }[case["kind"]]
    return launch(operation(layout, case), layout, case, target, case["count"])
