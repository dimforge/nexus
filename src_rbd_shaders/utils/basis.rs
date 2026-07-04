use glamx::Vec3;

// TODO: refactor to its own file. We already have a copy of this function in capsule.rs
pub fn orthonormal_basis3(v: Vec3) -> [Vec3; 2] {
    // NOTE: not using `sign` because we don't want the 0.0 case to return 0.0.
    let sign = if v.z >= 0.0 { 1.0 } else { -1.0 };
    let a = -1.0 / (sign + v.z);
    let b = v.x * v.y * a;

    [
        Vec3::new(1.0 + sign * v.x * v.x * a, sign * b, -sign * v.x),
        Vec3::new(b, sign + v.y * v.y * a, -v.y),
    ]
}
