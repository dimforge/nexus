use super::*;

#[test]
fn reset_luby_uses_wrapping_gpu_hash_arithmetic() {
    let links = vec![
        ContactLink {
            len: 1,
            ..Default::default()
        };
        129
    ];
    let mut colors = vec![0; 129];
    let mut rands = vec![0; 129];
    let plan = ContactPlan {
        bound: 129,
        ..Default::default()
    };
    for i in 0..129u32 {
        gpu_reset_luby(UVec3::new(i, 0, 0), &mut colors, &mut rands, &links, &plan);
        let first = (u64::from(i) * 0xcc9e2d51) as u32;
        let expected = (u64::from(first.rotate_left(15)) * 0x1b873593) as u32;
        assert_eq!(rands[i as usize], expected);
        assert_eq!(colors[i as usize], u32::MAX);
    }
}

#[test]
fn luby_current_round_winners_still_block_their_neighbors() {
    let links = vec![
        ContactLink {
            len: 1,
            solver_body_a: 1,
            solver_body_b: 0,
            vel_slot_a: 1,
            vel_slot_b: 0,
            ..Default::default()
        };
        4
    ];
    let plan = ContactPlan {
        bound: 4,
        ..Default::default()
    };
    for weights in [[10, 1, 7, 2], [9, 9, 1, 9]] {
        for previous in [false, true] {
            for a in 0..4u32 {
                for b in 0..4u32 {
                    for c in 0..4u32 {
                        for d in 0..4u32 {
                            let order = [a, b, c, d];
                            let mut sorted = order;
                            sorted.sort();
                            if sorted != [0, 1, 2, 3] {
                                continue;
                            }
                            let mut colors = vec![u32::MAX; 4];
                            if previous {
                                colors[0] = 2;
                            }
                            let winner = (0..4usize)
                                .filter(|&i| colors[i] == u32::MAX)
                                .max_by_key(|&i| (weights[i], i))
                                .unwrap();
                            let mut uncolored = 0;
                            for i in order {
                                gpu_step_graph_coloring_luby(
                                    UVec3::new(i, 0, 0),
                                    UVec3::ONE,
                                    &[0, 4],
                                    &[0, 1, 2, 3],
                                    &links,
                                    &mut colors,
                                    &weights,
                                    &mut uncolored,
                                    &[0, 1],
                                    &3,
                                    &plan,
                                );
                            }
                            for (i, &color) in colors.iter().enumerate() {
                                assert_eq!(
                                    color,
                                    if previous && i == 0 {
                                        2
                                    } else if i == winner {
                                        3
                                    } else {
                                        u32::MAX
                                    },
                                    "order {order:?}, weights {weights:?}, previous {previous}"
                                );
                            }
                            assert_eq!(uncolored, if previous { 2 } else { 3 });
                        }
                    }
                }
            }
        }
    }
}
