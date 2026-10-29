use super::*;

#[test]
fn bounded_topo_cleans_conflicts_and_never_exceeds_its_palette() {
    // Clique graphs force palette exhaustion. Seeded conflicts also exercise
    // budget exhaustion while old nonzero colors remain on losing contacts.
    for n in [0usize, 1, 4, 63, 64, 65, 129] {
        for deterministic in [0, 1] {
            for seeded in [false, true] {
                for reverse in [false, true] {
                    let guard = 0xdeadbeef;
                    let mut links = vec![
                        ContactLink {
                            len: 1,
                            solver_body_a: 1,
                            solver_body_b: 0,
                            vel_slot_a: 1,
                            vel_slot_b: 0,
                            ..Default::default()
                        };
                        n + 3
                    ];
                    // Stale ids in an inert slot must never be dereferenced.
                    if n > 4 {
                        links[4].len = 0;
                        links[4].solver_body_a = u32::MAX;
                        links[4].solver_body_b = u32::MAX;
                    }
                    let ids: Vec<u32> = (0..n as u32)
                        .filter(|&i| links[i as usize].len != 0)
                        .collect();
                    let counts = [0, ids.len() as u32];
                    let body_groups = [0, 1];
                    let plan = ContactPlan {
                        bound: n as u32,
                        ..Default::default()
                    };
                    let batch = BatchIndices {
                        deterministic,
                        ..Default::default()
                    };
                    let mut colors = vec![guard; n + 3];
                    let mut colored = colors.clone();
                    let mut pending = colors.clone();
                    let mut flag = 7;
                    let groups = 2u32;
                    let grid = UVec3::new(groups, 1, 1);
                    // Reset covers the full logical bound, independently of iteration grid.
                    for i in 0..n.max(1) {
                        gpu_reset_topo_gc(
                            UVec3::new(i as u32, 0, 0),
                            &mut colors,
                            &mut colored,
                            &links,
                            &plan,
                            &mut flag,
                            &mut pending,
                        );
                    }
                    if seeded {
                        for i in 0..n {
                            if links[i].len != 0 {
                                colors[i] = 1;
                                colored[i] = 1;
                            }
                        }
                    }
                    let contacts = [[groups, 1, 1]];
                    let mut indirect = [[groups, 1, 1]];
                    gpu_clear_completion_flag_topo_gc(
                        UVec3::ZERO,
                        &mut flag,
                        &contacts,
                        &mut indirect,
                    );
                    let mut order: Vec<u32> = (0..groups * 64).collect();
                    if reverse {
                        order.reverse();
                    }
                    for round in 0..2 {
                        if round > 0 {
                            gpu_reset_completion_flag_topo_gc(
                                UVec3::ZERO,
                                &mut flag,
                                &contacts,
                                &mut indirect,
                            );
                        }
                        if indirect[0][0] != 0 {
                            for &i in &order {
                                gpu_step_graph_coloring_topo_gc(
                                    UVec3::new(i, 0, 0),
                                    grid,
                                    &counts,
                                    &ids,
                                    &links,
                                    &mut colors,
                                    &mut colored,
                                    &mut flag,
                                    &plan,
                                    &body_groups,
                                    &batch,
                                    &mut pending,
                                );
                            }
                            for &i in &order {
                                gpu_fix_conflicts_topo_gc(
                                    UVec3::new(i, 0, 0),
                                    grid,
                                    &counts,
                                    &ids,
                                    &links,
                                    &mut colors,
                                    &mut colored,
                                    &mut flag,
                                    &plan,
                                    &body_groups,
                                    &batch,
                                    &pending,
                                );
                            }
                        }
                    }
                    for &i in &order {
                        gpu_finalize_topo_gc(
                            UVec3::new(i, 0, 0),
                            grid,
                            &mut colors,
                            &colored,
                            &plan,
                        );
                    }
                    for i in 0..n {
                        assert!(
                            colors[i] < 64,
                            "invalid color at {i}, n={n}, deterministic={deterministic}"
                        );
                        if colored[i] != 1 || links[i].len == 0 {
                            assert_eq!(colors[i], 0);
                        }
                        for j in i + 1..n {
                            assert!(
                                colors[i] == 0 || colors[i] != colors[j],
                                "conflict {i}/{j}, n={n}, seeded={seeded}, deterministic={deterministic}"
                            );
                        }
                    }
                    for a in [&colors, &colored, &pending] {
                        assert_eq!(&a[n..], &[guard; 3]);
                    }
                    let once = colors.clone();
                    for &i in &order {
                        gpu_finalize_topo_gc(
                            UVec3::new(i, 0, 0),
                            grid,
                            &mut colors,
                            &colored,
                            &plan,
                        );
                    }
                    assert_eq!(colors, once, "finalization must be idempotent");
                    if n <= 4 {
                        assert!(colors[..n].iter().all(|&c| c != 0));
                    }
                }
            }
        }
    }
}
