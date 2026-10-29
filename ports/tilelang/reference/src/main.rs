//! Export actual Nexus typed buffers and execute the current Rust solver as an oracle.
//! There is deliberately no second implementation of the solver arithmetic here.
use nexus_rbd_shaders3d::dynamics::contact_tiles::{HeaderTile, TILE_LEN, VecScalar};
use nexus_rbd_shaders3d::dynamics::coulomb_tiles::{CoulombPointTile, CoulombTile, FrictionRows};
use nexus_rbd_shaders3d::dynamics::{RbdSimParams, SymInertia, TwoBodyConstraint, Velocity};
use nexus_rbd_shaders3d::glamx::{Quat, Vec2, Vec3};
use nexus_rbd_shaders3d::{Pose, utils::Slice};
use serde_json::{Value, json};
use std::{
    fs,
    mem::{offset_of, size_of},
    path::Path,
};

fn write_json(path: impl AsRef<Path>, value: &Value) {
    fs::write(path, serde_json::to_vec_pretty(value).unwrap()).unwrap();
}
fn write_buffer<T: bytemuck::Pod>(dir: &Path, name: &str, data: &[T]) {
    const {
        assert!(
            cfg!(target_endian = "little"),
            "Replay ABI requires little endian"
        )
    };
    fs::write(dir.join(name), bytemuck::cast_slice(data)).unwrap();
}
fn layout() -> Value {
    const {
        assert!(offset_of!(CoulombTile, header) == 0);
        assert!(size_of::<VecScalar>() == 16);
        assert!(offset_of!(VecScalar, vec) == 0);
        assert!(offset_of!(VecScalar, scalar) == 12);
        assert!(size_of::<FrictionRows>() == 16);
        assert!(offset_of!(FrictionRows, impulse) == 0);
        assert!(offset_of!(FrictionRows, k) == 8);
    };
    // All offsets/strides are in 32-bit words, obtained from Rust's actual ABI.
    json!({
        "version": 1, "tile_len": TILE_LEN, "max_points": 4,
        "tile_words": size_of::<CoulombTile>() / 4,
        "point_words": size_of::<CoulombPointTile>() / 4,
        "vec_scalar_words": size_of::<VecScalar>() / 4,
        "friction_words": size_of::<FrictionRows>() / 4,
        "header": {
            "normal": offset_of!(HeaderTile, normal) / 4,
            "tangent": offset_of!(HeaderTile, tangent) / 4,
            "body_a": offset_of!(HeaderTile, body_a) / 4,
            "body_b": offset_of!(HeaderTile, body_b) / 4,
            "diag_a": offset_of!(HeaderTile, inertia_diag_a) / 4,
            "off_a": offset_of!(HeaderTile, inertia_off_a) / 4,
            "diag_b": offset_of!(HeaderTile, inertia_diag_b) / 4,
            "off_b": offset_of!(HeaderTile, inertia_off_b) / 4
        },
        "points": offset_of!(CoulombTile, points) / 4,
        "coupling": offset_of!(CoulombTile, coupling) / 4,
        "point": {
            "arm_a": offset_of!(CoulombPointTile, arm_a) / 4,
            "arm_b": offset_of!(CoulombPointTile, arm_b) / 4,
            "local_a": offset_of!(CoulombPointTile, local_a) / 4,
            "local_b": offset_of!(CoulombPointTile, local_b) / 4,
            "friction": offset_of!(CoulombPointTile, friction) / 4
        },
        "pose_words": size_of::<Pose>() / 4,
        "pose_translation": offset_of!(Pose, translation) / 4,
        "pose_rotation": offset_of!(Pose, rotation) / 4,
        "velocity_words": size_of::<Velocity>() / 4,
        "velocity_linear": offset_of!(Velocity, linear) / 4,
        "velocity_angular": offset_of!(Velocity, angular) / 4,
        "params": ["inv_dt", "max_corrective_velocity", "contact_cfm_factor",
            "contact_erp_inv_dt", "static_contact_cfm_factor", "static_contact_erp_inv_dt"]
    })
}
struct Rng(u64);
impl Rng {
    fn value(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / (1_u32 << 24) as f32) * 2.0 - 1.0
    }
    fn vector(&mut self) -> Vec3 {
        Vec3::new(self.value(), self.value(), self.value())
    }
}

fn fixture(
    root: &Path,
    n: usize,
    bias: bool,
    friction: bool,
    iterations: usize,
    seed: u64,
    boundary_stress: bool,
) -> Value {
    let name = format!(
        "n{n}-b{}-f{}-i{iterations}-s{seed}",
        u8::from(bias),
        u8::from(friction)
    );
    let dir = root.join(&name);
    fs::create_dir_all(&dir).unwrap();
    let mut rng = Rng(seed);
    let nb = n.max(1) * 2 + 3; // Also verify untouched sentinel bodies.
    let mut poses = vec![Pose::IDENTITY; nb];
    let mut velocities = vec![Velocity::default(); nb];
    for i in 0..nb {
        poses[i] = Pose::from_parts(
            rng.vector() * 0.05,
            Quat::from_scaled_axis(rng.vector() * 0.2),
        );
        velocities[i] = Velocity::new(rng.vector() * 0.8, rng.vector() * 0.4);
        velocities[i].padding0 = 0x12345678;
        velocities[i].padding1 = 0x7f123456;
    }
    let mut tiles = vec![CoulombTile::default(); n.div_ceil(TILE_LEN).max(1)];
    let mut params = RbdSimParams::tgs_soft();
    params.static_contact_natural_frequency *= 1.7;
    params.static_contact_damping_ratio *= 0.8;
    // Different parameters across fixtures exercise parameter binding as well as static selection.
    params.dt = if seed.is_multiple_of(2) {
        1.0 / 120.0
    } else {
        1.0 / 60.0
    };
    for i in 0..n {
        let dir_a = (rng.vector() + Vec3::new(0.2, 1.5, 0.1)).normalize();
        let tangent_a = dir_a.cross(Vec3::X).normalize();
        let body_a = 2 * i;
        let body_b = 2 * i + 1;
        // Distinct pose and velocity indices exercise split-body addressing.
        let vel_a = 2 * (n - i - 1) + 1;
        let vel_b = 2 * (n - i - 1);
        let inertia = SymInertia {
            diag: Vec3::new(0.7, 0.9, 0.6),
            off: Vec3::new(0.04, -0.03, 0.02),
            ..Default::default()
        };
        let mut c = TwoBodyConstraint {
            dir_a,
            tangent_a,
            len: ((i + 1) % 5) as u32,
            limit: if i % 7 == 0 { 0.0 } else { 0.6 },
            im_a: Vec3::new(0.7, 0.6, 0.8),
            im_b: Vec3::splat(0.4),
            ii_a: inertia,
            ii_b: inertia,
            solver_body_a: body_a as u32,
            solver_body_b: body_b as u32,
            vel_slot_a: vel_a as u32,
            vel_slot_b: vel_b as u32,
            ..Default::default()
        };
        match (i / 5) % 5 {
            1 => {
                c.im_a = Vec3::ZERO;
                c.ii_a = SymInertia::default();
            }
            2 => {
                c.im_b = Vec3::ZERO;
                c.ii_b = SymInertia::default();
            }
            3 => {
                c.im_a.x = 0.0;
                c.im_b.z = 0.0;
            }
            4 => {
                c.im_a = Vec3::ZERO;
                c.im_b = Vec3::ZERO;
                c.ii_a = SymInertia::default();
                c.ii_b = SymInertia::default();
            }
            _ => {}
        }
        for k in 0..c.len as usize {
            let p = &mut c.points[k];
            p.r_a = rng.vector() * 0.3;
            p.r_b = rng.vector() * 0.3;
            p.local_pt_a = rng.vector() * 0.1;
            p.local_pt_b = rng.vector() * 0.1;
            // Exact-zero cases in the portable suite avoid cancellation of rotated
            // points. --boundary-stress deliberately retains that unstable branch.
            if (i + k) % 4 == 1 && !boundary_stress {
                p.local_pt_a = Vec3::ZERO;
                p.local_pt_b = Vec3::ZERO;
            }
            let current_separation =
                (poses[body_a] * p.local_pt_a - poses[body_b] * p.local_pt_b).dot(dir_a);
            let distance = [-0.03, 0.0, 0.025, -0.5][(i + k) % 4];
            p.dist = distance - current_separation;
            p.normal_vel = -0.1 * rng.value().abs();
            p.normal_impulse = 0.25 * rng.value().abs();
            p.tangent_impulse = Vec2::new(rng.value(), rng.value()) * 0.15;
        }
        c.compute_effective_masses();
        // Exercise the determinant threshold independently of fully locked bodies.
        if i % 13 == 0 && c.len > 0 {
            c.points[0].tangent_k = [1.0e-11, 1.0e-11, 0.0];
        }
        tiles[i / TILE_LEN].write_constraint(i % TILE_LEN, &c);
    }
    let coefficients = [
        params.inv_dt(),
        params.max_corrective_velocity(),
        params.contact_cfm_factor(),
        params.contact_erp_inv_dt(),
        params.static_contact_cfm_factor(),
        params.static_contact_erp_inv_dt(),
    ];
    write_buffer(&dir, "Tiles.bin", &tiles);
    write_buffer(&dir, "Velocities.bin", &velocities);
    write_buffer(&dir, "Poses.bin", &poses);
    write_buffer(&dir, "Params.bin", &coefficients);
    write_buffer(&dir, "RbdParams.bin", &[params]);
    for _ in 0..iterations {
        for i in 0..n {
            let tile = &mut tiles[i / TILE_LEN];
            let mut c = tile.constraint(i % TILE_LEN);
            let (a, b) = (c.vel_slot_a as usize, c.vel_slot_b as usize);
            let (mut va, mut vb) = (velocities[a], velocities[b]);
            c.solve_constraint_gauss_seidel(
                &Slice(&poses, 0),
                &params,
                &mut va,
                &mut vb,
                bias,
                friction,
            );
            velocities[a] = va;
            velocities[b] = vb;
            // Only the mutable outputs of solve_points are written back.
            for k in 0..c.len as usize {
                tile.points[k].arm_a[i % TILE_LEN].scalar = c.points[k].normal_impulse;
                tile.points[k].friction[i % TILE_LEN].impulse = c.points[k].tangent_impulse;
            }
        }
    }
    write_buffer(&dir, "Tiles.expected.bin", &tiles);
    write_buffer(&dir, "Velocities.expected.bin", &velocities);
    json!({"name": name, "count": n, "bodies": nb, "tiles": tiles.len(),
        "use_bias": bias, "solve_friction": friction, "iterations": iterations, "seed": seed})
}
fn main() {
    let root = std::env::args()
        .nth(1)
        .expect("Usage: nexus-tilelang-reference OUTPUT_DIRECTORY");
    let root = Path::new(&root);
    fs::create_dir_all(root).unwrap();
    write_json(root.join("layout.json"), &layout());
    let boundary_stress = std::env::args().any(|arg| arg == "--boundary-stress");
    let mut cases = Vec::new();
    for n in [0, 1, 31, 32, 33, 63, 64, 65, 127] {
        cases.push(fixture(
            root,
            n,
            true,
            true,
            1,
            20261029 + n as u64,
            boundary_stress,
        ));
    }
    for bias in [false, true] {
        for friction in [false, true] {
            for seed in [17, 42] {
                cases.push(fixture(root, 129, bias, friction, 3, seed, boundary_stress));
            }
        }
    }
    write_json(
        root.join("suite.json"),
        &json!({"version": 1, "feature": "rbd.contact.coulomb.solve3d", "cases": cases}),
    );
    println!(
        "Exported {} Rust-reference cases to {}",
        cases.len(),
        root.display()
    );
}
