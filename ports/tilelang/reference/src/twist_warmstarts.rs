//! Live Rust cache scaling, ordered body gather and split-slot warmstart oracle.
#[allow(dead_code)]
#[path = "twist.rs"]
mod twist;
use nexus_rbd_shaders3d::{
    dynamics::{
        HubCounts, NOT_A_HUB, RbdSimParams, SymInertia, TwistConstraint, Velocity,
        twist_tiles::{
            TwistTile, gpu_gather_warmstart_velocities, gpu_hub_warmstart_constraints,
            gpu_scale_constraint_impulses,
        },
    },
    glamx::{Quat, UVec3, Vec2, Vec3},
    utils::BatchIndices,
};
use serde_json::{Value, json};
use std::{
    fs,
    mem::{offset_of, size_of},
    path::Path,
};
use twist::contacts::pilot;

fn layout() -> Value {
    let mut l = twist::layout();
    l["params_words"] = json!(size_of::<RbdSimParams>() / 4);
    l["warmstart_coefficient"] = json!(offset_of!(RbdSimParams, warmstart_coefficient) / 4);
    l["batch_words"] = json!(size_of::<BatchIndices>() / 4);
    l["batch"] = json!({
        "num_batches": offset_of!(BatchIndices, num_batches)/4,
        "bodies_len": offset_of!(BatchIndices, bodies_len)/4,
        "solver_color_buckets_stride": offset_of!(BatchIndices, solver_color_buckets_stride)/4});
    l["hub_words"] = json!(size_of::<HubCounts>() / 4);
    l["hub"] = json!({"hubs":offset_of!(HubCounts,hubs)/4,
        "slots":offset_of!(HubCounts,slots)/4,"base":offset_of!(HubCounts,base)/4,
        "pool":offset_of!(HubCounts,pool)/4});
    l
}
fn fixture(root: &Path, n: usize, op: &str) -> Value {
    let name = format!("n{n}");
    let dir = root.join(&name);
    fs::create_dir_all(&dir).unwrap();
    let batches = if n == 65 { 2 } else { 1 };
    let body_len = if n == 0 { 0 } else { (n + 4).div_ceil(batches) };
    let bodies = body_len * batches;
    let storage = bodies.max(1) + 3;
    let constraints = n.max(1) + 3;
    let mut tiles = vec![TwistTile::default(); constraints.div_ceil(32)];
    bytemuck::cast_slice_mut::<_, u32>(&mut tiles).fill((-19.0f32).to_bits());
    let mut indices = vec![u32::MAX; constraints];
    let mut rows = vec![Vec::<u32>::new(); storage];
    let mut hub_ids = Vec::new();
    for i in 0..n {
        // Repeated ordinary bodies and one hub; swaps test both sides' tags.
        let batch_base = ((i / 4) % batches) * body_len;
        let a = batch_base + [0, 1, 2, 2][i % 4];
        let b = batch_base + 3 + i % (body_len - 3).min(7);
        let (a, b) = if i % 3 == 0 { (b, a) } else { (a, b) };
        let cid = n - 1 - i;
        indices[cid] = i as u32;
        let mut slots = [a, b];
        for (side, body) in [a, b].into_iter().enumerate() {
            rows[body].push(cid as u32);
            if body == 1 {
                slots[side] = storage + hub_ids.len();
                hub_ids.push(cid as u32);
            }
        }
        let q = Quat::from_scaled_axis(Vec3::new(0.03 * (i % 5) as f32, -0.12, 0.07));
        let inertia = SymInertia {
            diag: Vec3::new(0.7, 0.9, 0.6),
            off: Vec3::new(0.04, -0.03, 0.02),
            ..Default::default()
        };
        let mut c = TwistConstraint {
            dir_a: q * Vec3::Y,
            tangent_a: q * Vec3::X,
            len: ((i + 1) % 5) as u32,
            limit: 0.6,
            im_a: Vec3::new(0.7, 0.0, 0.8),
            im_b: Vec3::splat(0.4),
            ii_a: inertia,
            ii_b: inertia,
            solver_body_a: a as u32,
            solver_body_b: b as u32,
            vel_slot_a: slots[0] as u32,
            vel_slot_b: slots[1] as u32,
            offset_b: Vec3::new(-0.03, 0.02, 0.04),
            frame_a: Quat::IDENTITY,
            frame_b: Quat::IDENTITY,
            ..Default::default()
        };
        if a % body_len == 0 {
            c.im_a = Vec3::ZERO;
            c.ii_a = SymInertia::default();
        }
        if b % body_len == 0 {
            c.im_b = Vec3::ZERO;
            c.ii_b = SymInertia::default();
        }
        for k in 0..c.len as usize {
            let p = &mut c.points[k];
            let t = (i % 11 + k + 1) as f32;
            p.r_a = Vec3::new(0.02 * t, -0.1, 0.07);
            p.normal_impulse = 0.1 + 0.03 * t;
            p.normal_mass = 0.7;
            p.dist = -0.03;
            p.normal_vel = -0.05;
        }
        c.compute_effective_masses();
        c.friction.tangent_impulse = Vec2::new(-0.02, 0.015);
        c.friction.twist_impulse = if c.len > 1 { 0.01 } else { 0.0 };
        let tile = &mut tiles[i / 32];
        let lane = i % 32;
        tile.write_constraint(lane, &c);
        let (mut va, mut vb) = (Velocity::default(), Velocity::default());
        c.warmstart_constraint(&mut va, &mut vb);
        va.padding1 = a as u32;
        vb.padding1 = b as u32;
        // Deliberately distinct cached values prove gather reads the cache, while
        // hub warmstart recomputes it. Scaling must replace the whole cache.
        if a % body_len != 0 {
            va.linear += Vec3::new(0.07, -0.03, 0.02);
        }
        if b % body_len != 0 {
            vb.angular += Vec3::new(-0.04, 0.02, 0.03);
        }
        tile.warmstart.bodies[lane] = [va, vb];
        // Native write_constraint resets inactive compact vectors and friction padding.
        tile.friction[lane]._padding = 17.0;
        for k in c.len as usize..4 {
            tile.normal_impulse[lane][k] = 19.0;
            tile.dist[lane][k] = -18.0;
            tile.normal_vel[lane][k] = 20.0;
            tile.radius[lane][k] = 21.0;
        }
    }
    let live = hub_ids.len();
    let pool = if n == 65 { live } else { live + 2 };
    let hubs = HubCounts {
        hubs: u32::from(live != 0),
        slots: if n == 65 {
            (pool + 5) as u32
        } else {
            live as u32
        },
        base: storage as u32,
        pool: pool as u32,
    };
    hub_ids.resize(pool + 3, u32::MAX);
    let mut first = vec![NOT_A_HUB; storage];
    if live != 0 {
        first[1] = 0;
    }
    let mut counts = Vec::new();
    let mut ids = Vec::new();
    for (i, row) in rows.iter_mut().enumerate() {
        if i % 2 == 0 {
            row.reverse();
        }
        ids.extend(row.iter().copied());
        counts.push(ids.len() as u32);
    }
    // A finite cancellation-sensitive cache checks that gather is not reassociated
    // or reordered. This is an accumulation test, not a physical scene.
    if op == "gather_warmstart" && n == 129 {
        let start = counts[1] as usize;
        for (j, value) in [1.0e8, 3.0, -1.0e8].into_iter().enumerate() {
            let index = indices[ids[start + j] as usize] as usize;
            let pair = &mut tiles[index / 32].warmstart.bodies[index % 32];
            let side = usize::from(pair[0].padding1 != 2);
            pair[side].linear.x = value;
        }
    }
    ids.extend([u32::MAX; 3]);
    let velocities = storage + pool + 3;
    let mut vels: Vec<_> = (0..velocities)
        .map(|i| {
            let mut v = Velocity::new(
                Vec3::new(0.3, -0.17, 0.02 * (i % 7) as f32),
                Vec3::new(-0.2, 0.13, 0.07 * (i % 5) as f32),
            );
            v.padding0 = 0xabcdef01;
            v.padding1 = 0xfedcba98;
            v
        })
        .collect();
    let batch = BatchIndices {
        num_batches: batches as u32,
        bodies_len: body_len as u32,
        solver_color_buckets_stride: 5,
        ..Default::default()
    };
    let bucket_len = 5 * batches;
    let buckets: Vec<_> = (0..bucket_len)
        .map(|i| (n * (i + 1) / bucket_len) as u32)
        .collect();
    let mut params = RbdSimParams::tgs_soft();
    params.warmstart_coefficient = match n {
        31 => 0.0,
        32 => 1.0,
        _ => 0.65,
    };
    let dispatch = n.max(bodies).max(pool).max(1);
    macro_rules! write {($($key:literal=>$value:expr),+)=>{$(pilot::write_buffer(&dir,$key,$value);)+};}
    write!("Tiles.bin"=>&tiles,"Velocities.bin"=>&vels,"Counts.bin"=>&counts,
        "Ids.bin"=>&ids,"Indices.bin"=>&indices,"HubFirstSlot.bin"=>&first,
        "HubSlotConstraint.bin"=>&hub_ids,"HubCounts.bin"=>&[hubs],
        "Params.bin"=>&[params],"Buckets.bin"=>&buckets,"BatchIds.bin"=>&[batch]);
    for _ in 0..3 {
        for i in 0..dispatch.div_ceil(64) * 64 {
            let gid = UVec3::new(i as u32, 0, 0);
            match op {
                "scale_impulses" => {
                    gpu_scale_constraint_impulses(gid, &buckets, &mut tiles, &batch, &params)
                }
                "gather_warmstart" => gpu_gather_warmstart_velocities(
                    gid, &counts, &ids, &indices, &tiles, &mut vels, &first, &batch,
                ),
                "hub_warmstart" => {
                    gpu_hub_warmstart_constraints(gid, &mut vels, &tiles, &hub_ids, &hubs, &indices)
                }
                _ => unreachable!(),
            }
        }
    }
    write!("Tiles.expected.bin"=>&tiles,"Velocities.expected.bin"=>&vels);
    json!({"name":name,"kind":op,"count":n,"storage":storage,"bodies":bodies,
        "constraints":constraints,"tiles":tiles.len(),"velocities":velocities,"ids_storage":ids.len(),
        "hub_storage":hub_ids.len(),"bucket_storage":buckets.len(),"dispatch":dispatch,"iterations":3})
}
fn main() {
    let out = std::env::args().nth(1).expect("Expected output directory");
    let out = Path::new(&out);
    fs::create_dir_all(out).unwrap();
    pilot::write_json(out.join("layout.json"), &layout());
    for op in ["scale_impulses", "gather_warmstart", "hub_warmstart"] {
        let root = out.join(format!("twist_{op}"));
        fs::create_dir_all(&root).unwrap();
        let cases: Vec<_> = [0, 1, 31, 32, 33, 63, 64, 65, 129]
            .into_iter()
            .map(|n| fixture(&root, n, op))
            .collect();
        pilot::write_json(root.join("suite.json"), &json!({"cases":cases}));
    }
}
