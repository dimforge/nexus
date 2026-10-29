//! Live Rust mass-splitting entry points, using the cooperative CPU runtime for barriers.
#[allow(dead_code)]
#[path = "contacts.rs"]
mod contacts;
use contacts::pilot;
mod dispatch;
use khal::backend::{GpuBackend, GpuBuffer, GpuPass};
use nexus_rbd_shaders3d::{
    dynamics::{
        ContactLink, GpuHubAverage, HUB_AVERAGE_WORKGROUPS, HUB_MIN_CONSTRAINTS, HubCounts,
        NOT_A_HUB, Velocity, WorldMassProperties, gpu_hub_assign, gpu_hub_dispatch,
        gpu_hub_scatter, gpu_hub_split_constraints,
    },
    glamx::{UVec3, Vec3},
    utils::BatchIndices,
};
use serde_json::{Value, json};
use std::{
    fs,
    mem::{offset_of, size_of},
    path::Path,
};

fn layout() -> Value {
    let mut l = contacts::layout();
    l["batch_words"] = json!(size_of::<BatchIndices>() / 4);
    l["batch"] = json!({"num_batches": offset_of!(BatchIndices, num_batches)/4,
        "bodies_len": offset_of!(BatchIndices, bodies_len)/4,
        "colliders_batch_capacity": offset_of!(BatchIndices, colliders_batch_capacity)/4});
    l["hub_words"] = json!(size_of::<HubCounts>() / 4);
    l["hub"] = json!({"hubs":offset_of!(HubCounts,hubs)/4,"slots":offset_of!(HubCounts,slots)/4,
        "base":offset_of!(HubCounts,base)/4,"pool":offset_of!(HubCounts,pool)/4});
    l["hub_min_constraints"] = json!(HUB_MIN_CONSTRAINTS);
    l["hub_average_workgroups"] = json!(HUB_AVERAGE_WORKGROUPS);
    l["not_a_hub"] = json!(NOT_A_HUB);
    l
}
struct Graph {
    counts: Vec<u32>,
    ids: Vec<u32>,
    mass: Vec<WorldMassProperties>,
    groups: Vec<u32>,
    first: Vec<u32>,
    list: Vec<u32>,
    hubs: HubCounts,
    links: Vec<ContactLink>,
    slot_body: Vec<u32>,
    slot_constraint: Vec<u32>,
    velocities: Vec<Velocity>,
    slots_indirect: Vec<[u32; 3]>,
    average_indirect: Vec<[u32; 3]>,
    batch: BatchIndices,
}
impl Graph {
    fn write(&self, dir: &Path, expected: bool, op: &str) {
        macro_rules! buffer {
            ($name:literal,$data:expr,$changed:expr) => {
                if !expected || $changed {
                    pilot::write_buffer(
                        dir,
                        &format!("{}{}.bin", $name, if expected { ".expected" } else { "" }),
                        $data,
                    );
                }
            };
        }
        buffer!("Counts", &self.counts, false);
        buffer!("Ids", &self.ids, false);
        buffer!("WorldMass", &self.mass, false);
        buffer!("BodyGroup", &self.groups, false);
        buffer!("FirstSlots", &self.first, op == "hub_assign");
        buffer!("HubList", &self.list, op == "hub_assign");
        buffer!("HubCounts", &[self.hubs], op == "hub_assign");
        buffer!("Links", &self.links, op == "hub_split");
        buffer!("SlotBody", &self.slot_body, op == "hub_split");
        buffer!("SlotConstraint", &self.slot_constraint, op == "hub_split");
        buffer!(
            "Velocities",
            &self.velocities,
            op == "hub_scatter" || op == "hub_average"
        );
        buffer!("SlotsIndirect", &self.slots_indirect, op == "hub_dispatch");
        buffer!(
            "AverageIndirect",
            &self.average_indirect,
            op == "hub_dispatch"
        );
        buffer!("BatchIds", &[self.batch], false);
    }
    fn assign(&mut self, groups: u32) {
        for i in 0..groups * 64 {
            gpu_hub_assign(
                UVec3::new(i, 0, 0),
                UVec3::new(groups, 1, 1),
                &self.counts,
                &self.mass,
                &self.groups,
                &mut self.first,
                &mut self.list,
                &mut self.hubs,
                &self.batch,
            );
        }
    }
    fn split(&mut self, groups: u32) {
        for g in 0..groups {
            for lane in 0..64 {
                gpu_hub_split_constraints(
                    UVec3::new(lane, 0, 0),
                    UVec3::new(g, 0, 0),
                    UVec3::new(groups, 1, 1),
                    &self.counts,
                    &self.ids,
                    &self.first,
                    &mut self.links,
                    &mut self.slot_body,
                    &mut self.slot_constraint,
                    &self.hubs,
                    &self.list,
                );
            }
        }
    }
    fn average(&mut self, groups: u32) {
        let kernel =
            GpuHubAverage::from_bytes(&GpuBackend::Cpu, &[], GpuHubAverage::ENTRY_POINT).unwrap();
        let mut velocities = GpuBuffer::Cpu(std::mem::take(&mut self.velocities));
        kernel
            .call(
                &mut GpuPass::Cpu(None),
                dispatch::workgroups([groups, 1, 1]),
                &mut velocities,
                &GpuBuffer::Cpu(self.list.clone()),
                &GpuBuffer::Cpu(self.first.clone()),
                &GpuBuffer::Cpu(self.counts.clone()),
                &GpuBuffer::Cpu(vec![self.hubs]),
            )
            .unwrap();
        let GpuBuffer::Cpu(v) = velocities else {
            unreachable!()
        };
        self.velocities = v;
    }
}
fn fixture(root: &Path, n: usize, mode: &str, op: &str) -> Value {
    let name = format!("n{n}-{mode}");
    let dir = root.join(&name);
    fs::create_dir_all(&dir).unwrap();
    let batches = if n == 0 {
        0
    } else if n == 65 {
        2
    } else {
        1
    };
    let per_batch = if n == 0 { 0 } else { (n + 1).div_ceil(batches) };
    let capacity = per_batch * batches;
    let storage = capacity + 3;
    let batch = BatchIndices {
        num_batches: batches as u32,
        bodies_len: per_batch.saturating_sub(3) as u32,
        colliders_batch_capacity: per_batch as u32,
        ..Default::default()
    };
    let mut rows = vec![Vec::<u32>::new(); storage];
    let mut links = Vec::new();
    let mut add = |a: usize, b: usize| {
        let cid = links.len() as u32;
        links.push(ContactLink {
            len: 1,
            solver_body_a: a as u32,
            solver_body_b: b as u32,
            vel_slot_a: a as u32,
            vel_slot_b: b as u32,
            mass_scale_a: 1.0,
            mass_scale_b: 1.0,
            previous_constraint: 0xabcdef01,
            previous_constraint_index: 0xabcdef02,
            ..Default::default()
        });
        rows[a].push(cid);
        rows[b].push(cid);
    };
    for body in 1..capacity {
        let degree = if mode == "many" {
            65
        } else if body == 1 {
            33
        } else {
            [0, 1, 31, 32, 33, 63, 64, 65, 129][body % 9]
        };
        for k in 0..degree {
            if k % 2 == 0 {
                add(body, 0)
            } else {
                add(0, body)
            }
        }
    }
    if capacity > 5 {
        add(4, 5);
    }
    let mut ids = Vec::new();
    let counts = rows
        .iter_mut()
        .map(|r| {
            r.reverse();
            ids.extend_from_slice(r);
            ids.len() as u32
        })
        .collect();
    let constraint_count = links.len();
    links.resize(links.len() + 3, ContactLink::default());
    ids.extend([0x12345678; 3]);
    let mass = (0..storage)
        .map(|body| WorldMassProperties {
            inv_mass: if body == 0 || (mode != "many" && body % 17 == 0) {
                Vec3::ZERO
            } else {
                Vec3::new(0.0, 0.7, 0.9)
            },
            ..Default::default()
        })
        .collect();
    let groups = (0..storage)
        .map(|i| {
            if mode != "many" && i % 19 == 0 {
                0
            } else {
                i as u32
            }
        })
        .collect();
    let pool = if mode == "short" {
        69
    } else if mode == "zero" || n == 0 {
        0
    } else {
        constraint_count + 11
    };
    let hubs = HubCounts {
        hubs: 0,
        slots: 0,
        base: storage as u32,
        pool: pool as u32,
    };
    let velocities = (0..storage + pool + 3)
        .map(|i| Velocity {
            linear: Vec3::new(
                (i % 17) as f32 * 0.13 - 0.7,
                (i % 13) as f32 * 0.17 - 0.4,
                (i % 11) as f32 * -0.11,
            ),
            angular: Vec3::new(
                (i % 19) as f32 * -0.07,
                0.8 - (i % 7) as f32 * 0.19,
                (i % 5) as f32 * 0.23,
            ),
            padding0: 0x12345678,
            padding1: 0x87654321,
        })
        .collect();
    let mut graph = Graph {
        counts,
        ids,
        mass,
        groups,
        first: vec![NOT_A_HUB; storage],
        list: vec![u32::MAX; hubs.max_hubs() as usize + 3],
        hubs,
        links,
        slot_body: vec![0; pool + 3],
        slot_constraint: vec![u32::MAX; pool + 3],
        velocities,
        slots_indirect: vec![[0xabcdef01; 3]; 3],
        average_indirect: vec![[0xabcdef02; 3]; 3],
        batch,
    };
    if op != "hub_assign" {
        graph.assign(3);
    }
    if op == "hub_scatter" || op == "hub_average" {
        graph.split(3);
    }
    if op == "hub_average" && n == 129 {
        let body = (1..capacity)
            .find(|&body| rows[body].len() >= 129 && graph.first[body] != NOT_A_HUB)
            .unwrap();
        let start = storage + graph.first[body] as usize;
        for slot in 0..rows[body].len() {
            graph.velocities[start + slot].linear.x = 0.0;
        }
        graph.velocities[start].linear.x = 1.0e8;
        graph.velocities[start + 1].linear.x = 3.0;
        graph.velocities[start + 2].linear.x = -1.0e8;
    }
    if mode == "clipped" {
        assert_eq!(op, "hub_dispatch");
        graph.hubs.hubs = graph.hubs.max_hubs() + 7;
        graph.hubs.slots = graph.hubs.pool + 19;
    }
    let groups = match op {
        "hub_dispatch" => 1,
        "hub_scatter" => graph.hubs.live_slots().div_ceil(64).max(1),
        "hub_average" if mode == "many" => 256,
        _ => {
            if n >= 129 {
                1
            } else {
                3
            }
        }
    };
    let iterations = if op == "hub_assign" { 1 } else { 3 };
    let case = json!({"name":name,"kind":op,"count":capacity,"storage":storage,
        "constraints":graph.links.len(),"constraint_count":constraint_count,"ids_storage":graph.ids.len(),
        "hub_storage":graph.list.len(),"slot_storage":graph.slot_body.len(),"velocities":graph.velocities.len(),
        "groups":groups,"threads":if op=="hub_dispatch" {1} else {64},"iterations":iterations});
    graph.write(&dir, false, op);
    for _ in 0..iterations {
        match op {
            "hub_assign" => graph.assign(groups),
            "hub_split" => graph.split(groups),
            "hub_dispatch" => gpu_hub_dispatch(
                &graph.hubs,
                &mut graph.slots_indirect,
                &mut graph.average_indirect,
            ),
            "hub_scatter" => {
                for i in 0..groups * 64 {
                    gpu_hub_scatter(
                        UVec3::new(i, 0, 0),
                        &mut graph.velocities,
                        &graph.slot_body,
                        &graph.hubs,
                    );
                }
            }
            "hub_average" => graph.average(groups),
            _ => unreachable!(),
        }
    }
    graph.write(&dir, true, op);
    case
}
fn main() {
    let path = std::env::args().nth(1).expect("Expected output directory");
    let root = Path::new(&path);
    fs::create_dir_all(root).unwrap();
    pilot::write_json(root.join("layout.json"), &layout());
    for op in [
        "hub_assign",
        "hub_split",
        "hub_dispatch",
        "hub_scatter",
        "hub_average",
    ] {
        let dir = root.join(op);
        fs::create_dir_all(&dir).unwrap();
        let mut cases: Vec<_> = [0, 1, 31, 32, 33, 63, 64, 65, 129]
            .map(|n| fixture(&dir, n, "mixed", op))
            .into();
        for (n, mode) in [(65, "short"), (33, "zero"), (257, "many")] {
            cases.push(fixture(&dir, n, mode, op));
        }
        if op == "hub_dispatch" {
            cases.push(fixture(&dir, 65, "clipped", op));
        }
        pilot::write_json(dir.join("suite.json"), &json!({"cases":cases}));
    }
}
