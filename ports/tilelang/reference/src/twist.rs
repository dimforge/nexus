//! Live native twist-friction arithmetic and compact-tile ABI oracle.
#[allow(dead_code)]
#[path = "contacts.rs"]
pub(crate) mod contacts;
use nexus_rbd_shaders3d::{
    Pose,
    dynamics::{
        RbdSimParams, SymInertia, TwistConstraint, TwistContactPoint, Velocity,
        twist_tiles::TwistTile,
    },
    glamx::{Quat, Vec2, Vec3},
    utils::Slice,
};
use serde_json::{Value, json};
#[path = "twist_layout.rs"]
mod twist_layout;
use std::{fs, path::Path};
pub(crate) fn layout() -> Value {
    let mut l = contacts::layout();
    twist_layout::add_layout(&mut l);
    l
}

fn fixture(
    root: &Path,
    n: usize,
    op: &str,
    bias: bool,
    friction: bool,
    iterations: usize,
) -> Value {
    let name = format!(
        "n{n}-b{}-f{}-i{iterations}",
        u8::from(bias),
        u8::from(friction)
    );
    let dir = root.join(&name);
    fs::create_dir_all(&dir).unwrap();
    let bodies = n * 2 + 3;
    let storage = n * 4 + 3;
    let mut tiles = vec![TwistTile::default(); (n + 3).div_ceil(32)];
    bytemuck::cast_slice_mut::<_, u32>(&mut tiles).fill(0xdeadbeef);
    let mut vels: Vec<_> = (0..storage)
        .map(|i| {
            let mut v = Velocity::new(
                Vec3::new(0.2, -0.1 * (i % 5) as f32, 0.3),
                Vec3::new(-0.3, 0.2, 0.07 * (i % 7) as f32),
            );
            v.padding0 = 0x12345678;
            v.padding1 = 0x76543210;
            v
        })
        .collect();
    let poses: Vec<_> = (0..bodies)
        .map(|i| {
            Pose::from_parts(
                Vec3::new(0.01 * (i % 3) as f32, 0.004, -0.002),
                Quat::from_scaled_axis(Vec3::new(0.02, 0.03 * (i % 3) as f32, -0.015)),
            )
        })
        .collect();
    for i in 0..n {
        let q = Quat::from_scaled_axis(Vec3::new(0.02 * (i % 4) as f32, -0.07, 0.1));
        let inertia = SymInertia {
            diag: Vec3::new(0.7, 0.9, 0.6),
            off: Vec3::new(0.04, -0.03, 0.02),
            ..Default::default()
        };
        let mut c = TwistConstraint {
            len: (i % 5) as u32,
            solver_body_a: (2 * i) as u32,
            solver_body_b: (2 * i + 1) as u32,
            vel_slot_a: (2 * n + 2 * (n - 1 - i)) as u32,
            vel_slot_b: (2 * n + 2 * (n - 1 - i) + 1) as u32,
            dir_a: q * Vec3::Y,
            tangent_a: q * Vec3::X,
            limit: if i % 6 == 0 { 0. } else { 0.6 },
            im_a: Vec3::new(0.7, 0., 0.9),
            im_b: Vec3::splat(0.4),
            ii_a: inertia,
            ii_b: inertia,
            frame_a: Quat::from_scaled_axis(Vec3::new(0.03, 0.02, -0.04)),
            frame_b: Quat::from_scaled_axis(Vec3::new(-0.02, 0.01, 0.07)),
            offset_b: Vec3::new(-0.04, 0.03, 0.02),
            ..Default::default()
        };
        if i % 7 == 0 {
            c.im_a = Vec3::ZERO;
            c.ii_a = SymInertia::default();
        }
        if i % 11 == 0 {
            c.im_b = Vec3::ZERO;
            c.ii_b = SymInertia::default();
        }
        for k in 0..c.len as usize {
            c.points[k] = TwistContactPoint {
                r_a: Vec3::new(0.03 * (k + 1) as f32, -0.1, 0.04 * (i % 3) as f32),
                normal_impulse: 0.2 + 0.03 * k as f32,
                dist: if i % 3 == 0 { 0.12 } else { -0.04 },
                normal_vel: -0.05,
                ..Default::default()
            };
        }
        c.compute_effective_masses();
        c.friction.tangent_impulse = Vec2::new(0.03, -0.02);
        c.friction.twist_impulse = if c.len > 1 { 0.01 } else { 0. };
        if op == "twist_effective_mass" {
            for p in &mut c.points[..c.len as usize] {
                p.normal_mass = -3.;
                p.radius = -4.;
            }
            c.friction.center_a = Vec3::splat(-2.);
            c.friction.tangent_k = Vec3::splat(-5.);
            c.friction.twist_mass = -6.;
        }
        tiles[i / 32].write_constraint(i % 32, &c);
    }
    let mut params = RbdSimParams::tgs_soft();
    params.dt = 1. / 90.;
    params.static_contact_natural_frequency *= 1.7;
    let coefficients = [
        params.inv_dt(),
        params.max_corrective_velocity(),
        params.contact_cfm_factor(),
        params.contact_erp_inv_dt(),
        params.static_contact_cfm_factor(),
        params.static_contact_erp_inv_dt(),
    ];
    contacts::pilot::write_buffer(&dir, "Tiles.bin", &tiles);
    contacts::pilot::write_buffer(&dir, "Velocities.bin", &vels);
    contacts::pilot::write_buffer(&dir, "Poses.bin", &poses);
    contacts::pilot::write_buffer(&dir, "Params.bin", &coefficients);
    contacts::pilot::write_buffer(&dir, "RbdParams.bin", &[params]);
    for _ in 0..iterations {
        for i in 0..n {
            let mut c = tiles[i / 32].constraint(i % 32);
            let (a, b) = (c.vel_slot_a as usize, c.vel_slot_b as usize);
            let (mut va, mut vb) = (vels[a], vels[b]);
            match op {
                "twist_effective_mass" => c.compute_effective_masses(),
                "twist_warmstart" => c.warmstart_constraint(&mut va, &mut vb),
                "twist_solve" => c.solve_constraint_gauss_seidel(
                    &Slice(&poses, 0),
                    &params,
                    &mut va,
                    &mut vb,
                    bias,
                    friction,
                ),
                "twist_scale" => c.scale_impulses(0.65),
                _ => panic!("Unknown twist operation"),
            }
            if op != "twist_warmstart" {
                tiles[i / 32].write_constraint(i % 32, &c);
            }
            if c.im_a != Vec3::ZERO || c.ii_a.diag != Vec3::ZERO || c.ii_a.off != Vec3::ZERO {
                vels[a] = va;
            }
            if c.im_b != Vec3::ZERO || c.ii_b.diag != Vec3::ZERO || c.ii_b.off != Vec3::ZERO {
                vels[b] = vb;
            }
        }
    }
    contacts::pilot::write_buffer(&dir, "Tiles.expected.bin", &tiles);
    contacts::pilot::write_buffer(&dir, "Velocities.expected.bin", &vels);
    let case = json!({"name":name,"kind":op,"count":n,"bodies":bodies,"solver_storage":storage,"tiles":tiles.len(),"use_bias":bias,"solve_friction":friction,"iterations":iterations,"scale":0.65});
    fs::write(
        dir.join("case.json"),
        serde_json::to_vec_pretty(&case).unwrap(),
    )
    .unwrap();
    case
}
fn main() {
    let root = std::env::args().nth(1).expect("Output directory");
    let root = Path::new(&root);
    fs::create_dir_all(root).unwrap();
    fs::write(
        root.join("layout.json"),
        serde_json::to_vec_pretty(&layout()).unwrap(),
    )
    .unwrap();
    for op in [
        "twist_effective_mass",
        "twist_warmstart",
        "twist_solve",
        "twist_scale",
    ] {
        let dir = root.join(op);
        fs::create_dir_all(&dir).unwrap();
        let mut cases = Vec::new();
        for n in [0, 1, 5, 32, 65, 129] {
            if op == "twist_solve" {
                for (bias, friction) in [(false, false), (false, true), (true, false), (true, true)]
                {
                    cases.push(fixture(&dir, n, op, bias, friction, 3));
                }
            } else {
                cases.push(fixture(&dir, n, op, true, true, 1));
            }
        }
        fs::write(
            dir.join("suite.json"),
            serde_json::to_vec_pretty(&json!({"cases":cases})).unwrap(),
        )
        .unwrap();
    }
}
