use nexus_rbd_shaders3d::{
    glamx::{Vec3, Vec4},
    shapes::*,
};
pub(crate) fn shape(ty: u32) -> Shape {
    match ty {
        SHAPE_TYPE_BALL => Shape::ball(0.75),
        SHAPE_TYPE_CUBOID => Shape::cuboid(Vec3::new(0.3, 0.5, 0.9)),
        SHAPE_TYPE_CAPSULE => {
            Shape::capsule(Vec3::new(-0.4, -0.7, 0.2), Vec3::new(0.3, 0.8, -0.1), 0.25)
        }
        SHAPE_TYPE_CONE => Shape::cone(0.8, 0.4),
        SHAPE_TYPE_CYLINDER => Shape::cylinder(0.6, 0.5),
        SHAPE_TYPE_POLYLINE => Shape::polyline(
            17,
            23,
            11,
            Vec3::new(-0.3, -0.7, -0.9),
            Vec3::new(0.5, 0.4, 0.2),
        ),
        SHAPE_TYPE_TRIMESH => Shape::trimesh(
            17,
            23,
            11,
            5,
            7,
            Vec3::new(-0.3, -0.7, -0.9),
            Vec3::new(0.5, 0.4, 0.2),
        ),
        SHAPE_TYPE_CONVEX_POLY => Shape::convex_poly(1, 6, 0, 0),
        SHAPE_TYPE_TRIANGLE => Shape::from_triangle(&Triangle::new(
            Vec3::new(-0.3, 0.7, -0.4),
            Vec3::new(0.2, -0.8, 0.1),
            Vec3::new(0.6, 0.2, 0.4),
        )),
        _ => Shape::new(
            Vec4::new(1., 2., 3., f32::from_bits(ty)),
            Vec4::ZERO,
            Vec4::ZERO,
        ),
    }
}
