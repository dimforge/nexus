use super::*;

#[test]
fn hub_list_split_matches_full_adjacency_scan() {
    // Two hubs share one constraint. The remaining two dynamic bodies are unsplit.
    // Multiple workgroup counts cover striding, empty groups, and both constraint sides.
    let bodies = 6usize;
    let mut initial = vec![ContactLink::default(); 167];
    let mut lists = vec![Vec::<u32>::new(); bodies];
    for (i, c) in initial.iter_mut().enumerate() {
        let (a, b) = if i < 83 {
            (1, 3)
        } else if i < 166 {
            (4, 2)
        } else {
            (1, 2)
        };
        c.len = 1;
        c.solver_body_a = a;
        c.solver_body_b = b;
        c.vel_slot_a = a;
        c.vel_slot_b = b;
        c.mass_scale_a = 1.0;
        c.mass_scale_b = 1.0;
        lists[a as usize].push(i as u32);
        lists[b as usize].push(i as u32);
    }
    let mut ids = Vec::new();
    let counts: Vec<u32> = lists
        .iter()
        .map(|list| {
            ids.extend_from_slice(list);
            ids.len() as u32
        })
        .collect();
    let first = [NOT_A_HUB, 0, 84, NOT_A_HUB, NOT_A_HUB, NOT_A_HUB];
    for hubs in [0, 1, 2] {
        let hub_counts = [hubs, hubs * 84, bodies as u32, 192];
        let hub_list = [1u32, 2];
        let mut expected = initial.clone();
        let mut expected_bodies = vec![u32::MAX; 192];
        let mut expected_constraints = expected_bodies.clone();
        for body in 0..bodies {
            if !hub_list[..hubs as usize].contains(&(body as u32)) {
                continue;
            }
            let start = if body == 0 { 0 } else { counts[body - 1] };
            let end = counts[body];
            let n = (end - start) as f32;
            for entry in start..end {
                let slot = first[body] + entry - start;
                let cid = ids[entry as usize];
                let c = &mut expected[cid as usize];
                if c.solver_body_a == body as u32 {
                    c.vel_slot_a = bodies as u32 + slot;
                    c.mass_scale_a = n;
                } else {
                    c.vel_slot_b = bodies as u32 + slot;
                    c.mass_scale_b = n;
                }
                expected_bodies[slot as usize] = body as u32;
                expected_constraints[slot as usize] = cid;
            }
        }
        for groups in [1, 2, 4] {
            let mut actual = initial.clone();
            let mut actual_bodies = vec![u32::MAX; 192];
            let mut actual_constraints = actual_bodies.clone();
            for group in 0..groups {
                for lane in 0..64 {
                    gpu_hub_split_constraints(
                        UVec3::new(lane, 0, 0),
                        UVec3::new(group, 0, 0),
                        UVec3::new(groups, 1, 1),
                        &counts,
                        &ids,
                        &first,
                        &mut actual,
                        &mut actual_bodies,
                        &mut actual_constraints,
                        &hub_counts,
                        &hub_list,
                    );
                }
            }
            assert_eq!(actual, expected);
            assert_eq!(actual_bodies, expected_bodies);
            assert_eq!(actual_constraints, expected_constraints);
            let mut slot_grid = [[99u32; 3]];
            let mut hub_grid = [[99u32; 3]];
            gpu_hub_dispatch(&hub_counts, &mut slot_grid, &mut hub_grid);
            assert_eq!(slot_grid, [[(hubs * 84).div_ceil(64), 1, 1]]);
            assert_eq!(hub_grid, [[hubs, 1, 1]]);
        }
    }
}
