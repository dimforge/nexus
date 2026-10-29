use bytemuck::{Pod, Zeroable};
use nexus_rbd_shaders3d::{
    Pose,
    dynamics::*,
    glamx::{Quat, UVec3, Vec3},
    utils::BatchIndices,
};
use serde_json::{Value, json};
use std::{fs, path::Path};
mod joint_layout;
fn write<T: Pod>(d: &Path, name: &str, data: &[T]) {
    fs::write(d.join(name), bytemuck::cast_slice(data)).unwrap();
}
fn fixture(d: &Path, op: &str, nj: usize, nb: usize, zero_dt: bool) -> Value {
    fs::create_dir_all(d).unwrap();
    let n = nj * nb;
    let storage = n + 3;
    let bs = 2 * n + 3;
    let guard = 0xdeadbeefu32;
    let mut joints = vec![ImpulseJoint::zeroed(); storage];
    let mut local = vec![LocalMassProperties::default(); bs];
    let mut poses = vec![Pose::IDENTITY; bs];
    let mut vels = vec![Velocity::default(); bs];
    for i in 0..bs {
        let f = (i % 19) as f32;
        poses[i] = Pose {
            rotation: Quat::from_scaled_axis(Vec3::new(0.013 * f, -0.009 * f, 0.007 * f)),
            translation: Vec3::new(0.2 * f, 1. + 0.03 * f, -0.11 * f),
            padding: 0,
        };
        local[i].inv_mass = Vec3::new(0.5 + f * 0.02, 0.8 + f * 0.01, 1. + f * 0.03);
        local[i].inv_principal_inertia = Vec3::new(1. + f * 0.03, 0.9 + f * 0.04, 0.7 + f * 0.02);
        local[i].com = Vec3::new(0.01 * f, -0.005 * f, 0.003 * f);
        vels[i] = Velocity::new(
            Vec3::new(0.01 * f, -0.02 * f, 0.03 * f),
            Vec3::new(-0.003 * f, 0.007 * f, 0.005 * f),
        );
        vels[i].padding0 = guard;
        vels[i].padding1 = 0xabcdef12;
    }
    for (i, joint) in joints.iter_mut().take(n).enumerate() {
        let k = i / nb;
        let a = 2 * k * nb + i % nb;
        let b = (2 * k + 1) * nb + i % nb;
        let profile = k % 12;
        if k.is_multiple_of(5) {
            local[a].inv_mass = Vec3::ZERO;
            local[a].inv_principal_inertia = Vec3::ZERO;
            vels[a].linear = Vec3::ZERO;
            vels[a].angular = Vec3::ZERO;
        }
        if k.is_multiple_of(7) {
            local[b].inv_mass = Vec3::ZERO;
            local[b].inv_principal_inertia = Vec3::ZERO;
            vels[b].linear = Vec3::ZERO;
            vels[b].angular = Vec3::ZERO;
        }
        let mut data = GenericJoint::zeroed();
        data.local_frame_a = Pose {
            rotation: Quat::from_rotation_z(0.05 * k as f32),
            translation: Vec3::new(0.2, 0.1, -0.1),
            padding: 0,
        };
        data.local_frame_b = Pose {
            rotation: Quat::from_rotation_y(-0.03 * k as f32),
            translation: Vec3::new(-0.1, 0.2, 0.05),
            padding: 0,
        };
        let (locked, limited, motor, coupled) = match profile {
            0 => (63, 0, 0, 0),
            1 => (7, 0, 0, 0),
            2 => (55, 8, 8, 0),
            3 => (62, 1, 1, 0),
            4 => (0, 63, 63, 0),
            5 => (0, 1, 1, 7),
            6 => (0, 0, 7, 7),
            7 => (0, 7, 0, 7),
            8 => (0, 56, 56, 56),
            9 => (1, 63, 63, 0),
            10 => (0, 0, 0, 0),
            _ => (0, 56, 56, 0),
        };
        data.locked_axes = locked;
        data.limit_axes = limited;
        data.motor_axes = motor;
        data.coupled_axes = coupled;
        for axis in 0..6 {
            data.limits[axis] = JointLimits {
                min: -0.2,
                max: 0.25,
                impulse: 0.01,
            };
            data.motors[axis] = JointMotor {
                target_vel: 0.1 * (axis as f32 - 2.),
                target_pos: 0.05 * (axis as f32 - 3.),
                stiffness: if k.is_multiple_of(4) { 0. } else { 40. },
                damping: if k.is_multiple_of(4) { 0. } else { 2. },
                max_force: if k.is_multiple_of(9) { 0. } else { 100. },
                impulse: 0.03,
                model: (k % 2) as u32,
            };
        }
        *joint = ImpulseJoint {
            body_a: (2 * k) as u32,
            body_b: (2 * k + 1) as u32,
            padding: [0; 2],
            data,
        };
    }
    let world = local
        .iter()
        .zip(&poses)
        .map(|(m, p)| m.to_world(p))
        .collect::<Vec<_>>();
    let mut builders = vec![JointConstraintBuilder::zeroed(); storage];
    let mut constraints = vec![JointConstraint::default(); storage];
    for con in &mut constraints {
        con.padding = [guard; 3];
        for el in &mut con.elements {
            el.joint_id = guard;
        }
    }
    let batch = BatchIndices {
        num_batches: nb as u32,
        bodies_len: (2 * nj) as u32,
        impulse_joints_len: nj as u32,
        ..Default::default()
    };
    let mut params = RbdSimParams::tgs_soft();
    params.dt = if zero_dt { 0. } else { 1. / 240. };
    params.joint_natural_frequency = 25.;
    params.joint_damping_ratio = 0.8;
    let colors = [(nj / 3) as u32, (nj * 2 / 3) as u32, nj as u32, guard];
    let color = if nj < 3 { 2 } else { 1 };
    let bias = (!zero_dt) as u32;
    let groups = 1u32;
    if op != "joint_init" && op != "joint_motor_params" && op != "joint_program" {
        for gid in 0..groups * 64 {
            gpu_init_joint_constraints(
                UVec3::new(gid, 0, 0),
                UVec3::new(groups, 1, 1),
                &joints,
                &mut builders,
                &mut constraints,
                &local,
                &batch,
            );
        }
    }
    if op == "joint_solve" || op == "joint_orthogonalize" {
        for gid in 0..groups * 64 {
            gpu_update_joint_constraints(
                UVec3::new(gid, 0, 0),
                UVec3::new(groups, 1, 1),
                &builders,
                &mut constraints,
                &poses,
                &world,
                &params,
                &batch,
            );
        }
        for (i, con) in constraints.iter_mut().take(n).enumerate() {
            for (k, el) in con.elements.iter_mut().take(con.len as usize).enumerate() {
                el.impulse =
                    (0.003 * (i + k) as f32).clamp(el.impulse_bounds.x, el.impulse_bounds.y);
            }
        }
    }
    let mut motors = vec![guard; storage * 6 * 6];
    macro_rules! buffers {
        ($s:expr) => {{
            write(d, &format!("Joints{}.bin", $s), &joints);
            write(d, &format!("Builders{}.bin", $s), &builders);
            write(d, &format!("Constraints{}.bin", $s), &constraints);
            write(d, &format!("LocalMass{}.bin", $s), &local);
            write(d, &format!("WorldMass{}.bin", $s), &world);
            write(d, &format!("Poses{}.bin", $s), &poses);
            write(d, &format!("Velocities{}.bin", $s), &vels);
            write(d, &format!("Params{}.bin", $s), &[params]);
            write(d, &format!("BatchIds{}.bin", $s), &[batch]);
            write(d, &format!("ColorGroups{}.bin", $s), &colors);
            write(d, &format!("Color{}.bin", $s), &[color, guard]);
            write(d, &format!("UseBias{}.bin", $s), &[bias, guard]);
            write(d, &format!("MotorParams{}.bin", $s), &motors);
        }};
    }
    buffers!("");
    if op == "joint_init" || op == "joint_program" {
        for gid in 0..64 {
            gpu_init_joint_constraints(
                UVec3::new(gid, 0, 0),
                UVec3::new(1, 1, 1),
                &joints,
                &mut builders,
                &mut constraints,
                &local,
                &batch,
            );
        }
    }
    if op == "joint_update" || op == "joint_program" {
        for gid in 0..64 {
            gpu_update_joint_constraints(
                UVec3::new(gid, 0, 0),
                UVec3::new(1, 1, 1),
                &builders,
                &mut constraints,
                &poses,
                &world,
                &params,
                &batch,
            );
        }
    }
    if op == "joint_solve" || op == "joint_program" {
        for gid in 0..64 {
            gpu_solve_joint_constraints(
                UVec3::new(gid, 0, 0),
                UVec3::new(1, 1, 1),
                &mut constraints,
                &mut vels,
                &colors,
                &color,
                &batch,
                &bias,
            );
        }
    }
    if op == "joint_orthogonalize" {
        for con in constraints.iter_mut().take(n) {
            con.orthogonalize_constraints(0, con.len as usize);
        }
    }
    if op == "joint_motor_params" {
        for (i, joint) in joints.iter().take(n).enumerate() {
            for (axis, m) in joint.data.motors.iter().enumerate() {
                let p = m.motor_params(params.dt);
                let values = [
                    p.erp_inv_dt,
                    p.cfm_coeff,
                    p.cfm_gain,
                    p.target_pos,
                    p.target_vel,
                    p.max_impulse,
                ];
                motors[(i * 6 + axis) * 6..(i * 6 + axis + 1) * 6]
                    .copy_from_slice(bytemuck::cast_slice(&values));
            }
        }
    }
    buffers!(".expected");
    let sizes = [
        "Joints",
        "Builders",
        "Constraints",
        "LocalMass",
        "WorldMass",
        "Poses",
        "Velocities",
        "Params",
        "BatchIds",
        "ColorGroups",
        "Color",
        "UseBias",
        "MotorParams",
    ]
    .into_iter()
    .map(|n| {
        (
            n.to_owned(),
            json!(fs::metadata(d.join(format!("{n}.bin"))).unwrap().len() / 4),
        )
    })
    .collect::<serde_json::Map<_, _>>();
    let case = json!({"name":d.file_name().unwrap().to_str().unwrap(),"kind":op,"count":n,"joints_per_batch":nj,"batches":nb,"storage":storage,"bodies":bs,"groups":1,"iterations":1,"buffer_words":sizes});
    fs::write(
        d.join("case.json"),
        serde_json::to_vec_pretty(&case).unwrap(),
    )
    .unwrap();
    case
}
fn main() {
    let p = std::env::args().nth(1).unwrap();
    let root = Path::new(&p);
    fs::create_dir_all(root).unwrap();
    fs::write(
        root.join("layout.json"),
        serde_json::to_vec_pretty(&joint_layout::layout()).unwrap(),
    )
    .unwrap();
    for op in [
        "joint_motor_params",
        "joint_init",
        "joint_update",
        "joint_solve",
        "joint_orthogonalize",
        "joint_program",
    ] {
        let mut cases = vec![];
        for (n, nb, zero) in [
            (0, 1, false),
            (1, 1, false),
            (12, 3, false),
            (33, 1, true),
            (65, 2, false),
            (129, 1, false),
        ] {
            let name = format!("n{n}-b{nb}-z{}", zero as u32);
            cases.push(fixture(&root.join(op).join(name), op, n, nb, zero));
        }
        fs::write(
            root.join(op).join("suite.json"),
            serde_json::to_vec_pretty(&json!({"cases":cases})).unwrap(),
        )
        .unwrap();
    }
}
