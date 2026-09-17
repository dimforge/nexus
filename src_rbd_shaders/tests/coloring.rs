use super::*;

#[test]
fn first_free_color_across_both_words() {
    // In particular, an occupied color 32 must not shift a free low color.
    assert_eq!(first_free_color((0b1011, 1)), 2);
    for free in 0..64u32 {
        let mask = !(1u64 << free);
        assert_eq!(first_free_color((mask as u32, (mask >> 32) as u32)), free);
    }
    let mut mask = 0x6a09e667f3bcc909u64;
    for _ in 0..10000 {
        mask = mask.wrapping_mul(6364136223846793005).wrapping_add(1);
        assert_eq!(
            first_free_color((mask as u32, (mask >> 32) as u32)),
            (!mask).trailing_zeros()
        );
    }
}
#[test]
fn conflict_pruning_matches_full_graph_with_seeded_and_pending_colors() {
    use super::*;
    // Include static nodes, multibody groups, split hubs, inert slots and both color words.
    for deterministic in [0, 1] {
        for seed in 0..32u32 {
            let groups = [0u32, 1, 1, 2, 3, 4, 5, 5];
            let mut constraints = vec![ContactLink::default(); 96];
            let mut colors = vec![0; 96];
            let mut pending = vec![MAX_U32; 96];
            let mut colored = vec![1; 96];
            for (i, c) in constraints.iter_mut().enumerate() {
                c.len = u32::from(i % 13 != 0);
                c.solver_body_a = (i % 8) as u32;
                c.solver_body_b = ((i * 3 + 5) % 8) as u32;
                c.vel_slot_a = c.solver_body_a
                    + if groups[c.solver_body_a as usize] == 4 {
                        100
                    } else {
                        0
                    };
                c.vel_slot_b = c.solver_body_b
                    + if groups[c.solver_body_b as usize] == 4 {
                        100
                    } else {
                        0
                    };
                colors[i] = 1 + ((i as u32 * 17 + seed) % 7) * 9;
                if deterministic != 0 && (i as u32 + seed) % 3 == 0 {
                    pending[i] = 1 + ((i as u32 * 13 + seed) % 7) * 9;
                    colored[i] = 0;
                }
            }
            let touches = |i: usize, group: u32| {
                let c = &constraints[i];
                c.len != 0
                    && group != 0
                    && ((groups[c.solver_body_a as usize] == group
                        && c.vel_slot_a == c.solver_body_a)
                        || (groups[c.solver_body_b as usize] == group
                            && c.vel_slot_b == c.solver_body_b))
            };
            let mut counts = Vec::new();
            let mut adjacency = Vec::new();
            for group in 0..6 {
                for i in 0..constraints.len() {
                    if touches(i, group) {
                        adjacency.push(i as u32);
                    }
                }
                counts.push(adjacency.len() as u32);
            }
            let effective: Vec<_> = colors
                .iter()
                .zip(&pending)
                .map(|(&c, &p)| if p == MAX_U32 { c } else { p })
                .collect();
            let mut expected_colors = colors.clone();
            let mut expected_colored = colored.clone();
            for i in 0..constraints.len() {
                if constraints[i].len == 0 {
                    continue;
                }
                let conflict = (i + 1..constraints.len()).any(|j| {
                    effective[i] == effective[j]
                        && (1..6).any(|group| touches(i, group) && touches(j, group))
                });
                if conflict {
                    expected_colored[i] = 0;
                } else if pending[i] != MAX_U32 {
                    expected_colored[i] = 1;
                    expected_colors[i] = pending[i];
                }
            }
            for lane in 0..64 {
                gpu_fix_conflicts_topo_gc(
                    UVec3::new(lane, 0, 0),
                    UVec3::ONE,
                    &counts,
                    &adjacency,
                    &constraints,
                    &mut colors,
                    &mut colored,
                    &mut 0,
                    &ContactPlan {
                        bound: 96,
                        ..Default::default()
                    },
                    &groups,
                    &BatchIndices {
                        deterministic,
                        ..Default::default()
                    },
                    &pending,
                );
            }
            assert_eq!(
                colors, expected_colors,
                "seed {seed}, deterministic {deterministic}"
            );
            assert_eq!(
                colored, expected_colored,
                "seed {seed}, deterministic {deterministic}"
            );
        }
    }
}
