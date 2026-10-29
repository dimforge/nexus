//! Live Rust graph-coloring primitives, with phase inputs chosen independently.
#[allow(dead_code)]
#[path = "contacts.rs"]
mod contacts;
use contacts::pilot;
use nexus_rbd_shaders3d::{
    broad_phase::ContactPlan,
    dynamics::{
        ContactLink, gpu_clear_completion_flag_topo_gc, gpu_fix_conflicts_topo_gc,
        gpu_reset_completion_flag_topo_gc, gpu_reset_luby, gpu_reset_topo_gc,
        gpu_step_graph_coloring_luby, gpu_step_graph_coloring_topo_gc,
    },
    glamx::UVec3,
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
    l["plan_words"] = json!(size_of::<ContactPlan>() / 4);
    l["plan"] = json!({"bound":offset_of!(ContactPlan,bound)/4});
    l["batch_words"] = json!(size_of::<BatchIndices>() / 4);
    l["batch"] = json!({"deterministic":offset_of!(BatchIndices,deterministic)/4});
    l
}
fn fixture(root: &Path, n: usize, mode: &str, deterministic: u32, op: &str) -> Value {
    let name = format!("n{n}-{mode}-d{deterministic}");
    let dir = root.join(&name);
    fs::create_dir_all(&dir).unwrap();
    let guard = 0xDEADBEEF;
    let storage = n + 3;
    let bodies = n + 4;
    let mut links = vec![ContactLink::default(); storage];
    bytemuck::cast_slice_mut::<_, u32>(&mut links).fill(0xa5a5a5a5);
    let body_group: Vec<u32> = (0..bodies as u32)
        .map(|b| {
            if mode == "grouped" && b % 3 == 2 {
                b - 1
            } else {
                b
            }
        })
        .collect();
    let mut rows = vec![Vec::<u32>::new(); bodies];
    for (i, link) in links.iter_mut().take(n).enumerate() {
        link.len = if i % 11 == 0 && mode != "clique" {
            0
        } else {
            (i % 4 + 1) as u32
        };
        link.solver_body_a = if link.len == 0 {
            u32::MAX
        } else if mode == "body-zero" {
            0
        } else if mode == "split" || mode == "clique" {
            1
        } else {
            (i + 1) as u32
        };
        link.solver_body_b = if link.len == 0 {
            u32::MAX
        } else if mode == "clique" || i % 7 == 0 {
            0
        } else {
            (i + 2) as u32
        };
        link.vel_slot_a = if mode == "split" {
            (bodies + i) as u32
        } else {
            link.solver_body_a
        };
        link.vel_slot_b = link.solver_body_b;
        if link.len != 0 {
            for b in [link.solver_body_a, link.solver_body_b] {
                let g = body_group[b as usize];
                if g != 0 || mode == "body-zero" {
                    rows[g as usize].push(i as u32);
                }
            }
        }
    }
    let mut ids = Vec::new();
    let counts: Vec<u32> = rows
        .iter()
        .map(|r| {
            ids.extend(r.iter().rev());
            ids.len() as u32
        })
        .collect();
    let active_ids = ids.len();
    ids.extend([guard; 3]);
    let mut colors = vec![guard; storage];
    let mut rands = vec![guard; storage];
    let mut colored = vec![guard; storage];
    let mut pending = vec![guard; storage];
    for i in 0..n {
        let active = links[i].len != 0;
        colors[i] = if active {
            [1, 2, 31, 32, 33, 63][i % 6]
        } else {
            0
        };
        colored[i] = if active && i % 3 == 0 { 0 } else { 1 };
        pending[i] = if active && deterministic != 0 && i % 3 == 1 {
            [1, 31, 32, 33, 63][i % 5]
        } else {
            u32::MAX
        };
        rands[i] = if mode == "clique" {
            [10, 1, 7, 2][i % 4]
        } else if i % 7 == 1 {
            u32::MAX
        } else {
            ((i * 17 + 7) % 23) as u32
        };
        if op == "luby_step" {
            colors[i] = if !active {
                0
            } else if i % 4 == 0 && mode != "clique" {
                2
            } else {
                u32::MAX
            };
        }
        if matches!(op, "luby_reset" | "topo_reset") {
            colors[i] = guard;
            colored[i] = guard;
            pending[i] = guard;
        }
    }
    let mut num_colors = [
        if mode == "converged" {
            7
        } else if op == "luby_step" {
            0
        } else {
            1
        },
        guard,
    ];
    if op == "topo_fix" && mode != "converged" {
        num_colors[0] = 0;
    }
    let plan = ContactPlan {
        bound: n as u32,
        pfm_base: n as u32,
        pfm_len: 0,
    };
    let batch = BatchIndices {
        deterministic,
        ..Default::default()
    };
    let current = [3, guard];
    let scalar = matches!(op, "topo_clear" | "topo_completion");
    let threads = if scalar { 1 } else { 64 };
    let groups = if scalar {
        1
    } else if matches!(op, "luby_reset" | "topo_reset") {
        n.div_ceil(64).max(1) as u32
    } else if n >= 129 {
        1
    } else {
        3
    };
    let contacts_indirect = [
        [
            groups,
            if mode == "axes" { 2 } else { 1 },
            if mode == "axes" { 3 } else { 1 },
        ],
        [guard; 3],
        [guard; 3],
    ];
    let mut coloring_indirect = [[17, 5, 7], [guard; 3], [guard; 3]];
    macro_rules! write_all {
        ($expected:expr) => {{
            let expected = $expected;
            macro_rules! w {
                ($name:literal,$data:expr,$changed:expr) => {
                    if !expected || $changed {
                        pilot::write_buffer(
                            &dir,
                            &format!("{}{}.bin", $name, if expected { ".expected" } else { "" }),
                            $data,
                        );
                    }
                };
            }
            w!("Counts", &counts, false);
            w!("Ids", &ids, false);
            w!("Links", &links, false);
            w!("Colors", &colors, true);
            w!("Rands", &rands, true);
            w!("Colored", &colored, true);
            w!("Pending", &pending, true);
            w!("NumColors", &num_colors, true);
            w!("BodyGroup", &body_group, false);
            w!("Plan", &[plan], false);
            w!("BatchIds", &[batch], false);
            w!("ContactsIndirect", &contacts_indirect, false);
            w!("ColoringIndirect", &coloring_indirect, true);
            w!("CurrentColor", &current, false);
        }};
    }
    write_all!(false);
    for i in 0..groups * threads {
        let id = UVec3::new(i, 0, 0);
        let grid = UVec3::new(groups, 1, 1);
        match op {
            "luby_reset" => gpu_reset_luby(id, &mut colors, &mut rands, &links, &plan),
            "luby_step" => gpu_step_graph_coloring_luby(
                id,
                grid,
                &counts,
                &ids,
                &links,
                &mut colors,
                &rands,
                &mut num_colors[0],
                &body_group,
                &current[0],
                &plan,
            ),
            "topo_reset" => gpu_reset_topo_gc(
                id,
                &mut colors,
                &mut colored,
                &links,
                &plan,
                &mut num_colors[0],
                &mut pending,
            ),
            "topo_clear" => gpu_clear_completion_flag_topo_gc(
                id,
                &mut num_colors[0],
                &contacts_indirect,
                &mut coloring_indirect,
            ),
            "topo_completion" => gpu_reset_completion_flag_topo_gc(
                id,
                &mut num_colors[0],
                &contacts_indirect,
                &mut coloring_indirect,
            ),
            "topo_step" => gpu_step_graph_coloring_topo_gc(
                id,
                grid,
                &counts,
                &ids,
                &links,
                &mut colors,
                &mut colored,
                &mut num_colors[0],
                &plan,
                &body_group,
                &batch,
                &mut pending,
            ),
            "topo_fix" => gpu_fix_conflicts_topo_gc(
                id,
                grid,
                &counts,
                &ids,
                &links,
                &mut colors,
                &mut colored,
                &mut num_colors[0],
                &plan,
                &body_group,
                &batch,
                &pending,
            ),
            _ => unreachable!(),
        }
    }
    write_all!(true);
    let l = layout();
    let sizes = json!({"Counts":counts.len(),"Ids":ids.len(),"Links":storage*l["link_words"].as_u64().unwrap() as usize,"Colors":storage,"Rands":storage,"Colored":storage,"Pending":storage,"NumColors":2,"BodyGroup":bodies,"Plan":l["plan_words"],"BatchIds":l["batch_words"],"ContactsIndirect":9,"ColoringIndirect":9,"CurrentColor":2});
    json!({"name":name,"kind":op,"count":n,"storage":storage,"bodies":bodies,"active_ids":active_ids,"groups":groups,"threads":threads,"mode":mode,"static_groups":if mode=="body-zero" {vec![]}else{vec![0u32]},"deterministic":deterministic,"iterations":1,"buffer_words":sizes})
}
fn main() {
    let path = std::env::args().nth(1).expect("Expected output directory");
    let root = Path::new(&path);
    fs::create_dir_all(root).unwrap();
    pilot::write_json(root.join("layout.json"), &layout());
    for op in [
        "luby_reset",
        "luby_step",
        "topo_reset",
        "topo_clear",
        "topo_completion",
        "topo_step",
        "topo_fix",
    ] {
        let dir = root.join(op);
        fs::create_dir_all(&dir).unwrap();
        let mut cases = Vec::new();
        for deterministic in [0, 1] {
            for (n, mode) in [
                (0, "chain"),
                (1, "chain"),
                (31, "chain"),
                (32, "chain"),
                (33, "chain"),
                (63, "chain"),
                (64, "chain"),
                (65, "chain"),
                (129, "chain"),
                (33, "grouped"),
                (33, "body-zero"),
                (65, "split"),
                (4, "clique"),
                (33, "converged"),
            ] {
                cases.push(fixture(&dir, n, mode, deterministic, op));
            }
        }
        if matches!(op, "topo_clear" | "topo_completion") {
            cases.push(fixture(&dir, 0, "axes", 0, op));
        }
        pilot::write_json(dir.join("suite.json"), &json!({"cases":cases}));
    }
}

#[cfg(test)]
#[path = "../../../../src_rbd_shaders/tests/coloring_luby.rs"]
mod luby_regressions;
