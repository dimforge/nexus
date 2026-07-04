//! Debug validation of the LBVH topology (see [`crate::VALIDATE_LBVH_TOPOLOGY`]).

use crate::shaders::broad_phase::LbvhNode;

pub(crate) fn validate_lbvh_topology(
    tree: &[LbvhNode],
    sorted_colliders: &[u32],
    num_colliders: u32,
) {
    let n = num_colliders as usize;
    if n < 2 {
        println!("[LBVH] Skipping validation: num_colliders={}", n);
        return;
    }

    let num_internal = n - 1;
    let first_leaf = num_internal;
    let total_nodes = 2 * n - 1;
    let mut errors = 0u32;

    println!(
        "[LBVH] Validating topology: {} colliders, {} nodes",
        n, total_nodes
    );

    // 1. Check internal node topology (nodes 0..num_internal)
    for i in 0..num_internal {
        let node = &tree[i];
        let left = node.left as usize;
        let right = node.right as usize;

        if left >= total_nodes {
            eprintln!(
                "  ERROR: internal node {} has left={} (out of bounds, max={})",
                i,
                left,
                total_nodes - 1
            );
            errors += 1;
        }
        if right >= total_nodes {
            eprintln!(
                "  ERROR: internal node {} has right={} (out of bounds, max={})",
                i,
                right,
                total_nodes - 1
            );
            errors += 1;
        }

        // Children should point back to this node as parent
        if left < total_nodes && tree[left].parent as usize != i {
            eprintln!(
                "  ERROR: internal node {} left child {} has parent={} (expected {})",
                i, left, tree[left].parent, i
            );
            errors += 1;
        }
        if right < total_nodes && tree[right].parent as usize != i {
            eprintln!(
                "  ERROR: internal node {} right child {} has parent={} (expected {})",
                i, right, tree[right].parent, i
            );
            errors += 1;
        }

        // left and right should be different
        if left == right {
            eprintln!("  ERROR: internal node {} has left == right == {}", i, left);
            errors += 1;
        }
    }

    // 2. Check leaf nodes (nodes first_leaf..total_nodes)
    let mut collider_seen = vec![false; n];
    for (leaf_offset, node) in tree[first_leaf..total_nodes].iter().enumerate() {
        let leaf_index = first_leaf + leaf_offset;
        let collider_id = node.left as usize;

        if collider_id >= n {
            eprintln!(
                "  ERROR: leaf {} has collider_id={} (out of bounds, max={})",
                leaf_index,
                collider_id,
                n - 1
            );
            errors += 1;
        } else if collider_seen[collider_id] {
            eprintln!(
                "  ERROR: leaf {} has duplicate collider_id={}",
                leaf_index, collider_id
            );
            errors += 1;
        } else {
            collider_seen[collider_id] = true;
        }
    }

    let missing: Vec<usize> = collider_seen
        .iter()
        .enumerate()
        .filter(|(_, seen)| !**seen)
        .map(|(id, _)| id)
        .collect();
    if !missing.is_empty() {
        eprintln!(
            "  ERROR: {} colliders missing from leaves: {:?}",
            missing.len(),
            &missing[..missing.len().min(20)]
        );
        errors += 1;
    }

    // 3. Check sorted_colliders matches leaf assignment
    for i in 0..n {
        let expected_collider = sorted_colliders[i];
        let leaf_collider = tree[first_leaf + i].left;
        if expected_collider != leaf_collider {
            eprintln!(
                "  ERROR: sorted_colliders[{}]={} but tree leaf {}.left={}",
                i,
                expected_collider,
                first_leaf + i,
                leaf_collider
            );
            errors += 1;
            if errors > 50 {
                break;
            }
        }
    }

    // 4. Check AABBs: parent AABB should contain both children
    let mut aabb_errors = 0u32;
    for i in 0..num_internal {
        let node = &tree[i];
        let left = node.left as usize;
        let right = node.right as usize;
        if left >= total_nodes || right >= total_nodes {
            continue;
        }

        let parent_aabb = &node.aabb;
        let left_aabb = &tree[left].aabb;
        let right_aabb = &tree[right].aabb;

        let eps = 1.0e-5;
        let parent_valid = parent_aabb.mins.x <= parent_aabb.maxs.x;
        let left_valid = left_aabb.mins.x <= left_aabb.maxs.x;
        let right_valid = right_aabb.mins.x <= right_aabb.maxs.x;

        if !parent_valid {
            if aabb_errors < 10 {
                eprintln!(
                    "  ERROR: internal node {} has invalid AABB (mins > maxs): mins={:?} maxs={:?}",
                    i, parent_aabb.mins, parent_aabb.maxs
                );
            }
            aabb_errors += 1;
            continue;
        }

        if left_valid
            && (parent_aabb.mins.x > left_aabb.mins.x + eps
                || parent_aabb.mins.y > left_aabb.mins.y + eps
                || parent_aabb.maxs.x < left_aabb.maxs.x - eps
                || parent_aabb.maxs.y < left_aabb.maxs.y - eps)
        {
            if aabb_errors < 10 {
                eprintln!(
                    "  ERROR: node {} AABB does not contain left child {} AABB",
                    i, left
                );
                eprintln!(
                    "    parent: mins={:?} maxs={:?}",
                    parent_aabb.mins, parent_aabb.maxs
                );
                eprintln!(
                    "    left:   mins={:?} maxs={:?}",
                    left_aabb.mins, left_aabb.maxs
                );
            }
            aabb_errors += 1;
        }

        if right_valid
            && (parent_aabb.mins.x > right_aabb.mins.x + eps
                || parent_aabb.mins.y > right_aabb.mins.y + eps
                || parent_aabb.maxs.x < right_aabb.maxs.x - eps
                || parent_aabb.maxs.y < right_aabb.maxs.y - eps)
        {
            if aabb_errors < 10 {
                eprintln!(
                    "  ERROR: node {} AABB does not contain right child {} AABB",
                    i, right
                );
                eprintln!(
                    "    parent: mins={:?} maxs={:?}",
                    parent_aabb.mins, parent_aabb.maxs
                );
                eprintln!(
                    "    right:  mins={:?} maxs={:?}",
                    right_aabb.mins, right_aabb.maxs
                );
            }
            aabb_errors += 1;
        }
    }

    if aabb_errors > 0 {
        eprintln!("  AABB errors total: {} (showing first 10)", aabb_errors);
        errors += aabb_errors;
    }

    // 5. Check reachability from root via BFS
    let mut visited = vec![false; total_nodes];
    let mut queue = std::collections::VecDeque::new();
    queue.push_back(0usize);
    visited[0] = true;
    let mut visited_count = 0usize;

    while let Some(id) = queue.pop_front() {
        visited_count += 1;
        if id < num_internal {
            let left = tree[id].left as usize;
            let right = tree[id].right as usize;
            if left < total_nodes && !visited[left] {
                visited[left] = true;
                queue.push_back(left);
            }
            if right < total_nodes && !visited[right] {
                visited[right] = true;
                queue.push_back(right);
            }
        }
    }

    if visited_count != total_nodes {
        let unreachable: Vec<usize> = visited
            .iter()
            .enumerate()
            .filter(|(_, v)| !**v)
            .map(|(id, _)| id)
            .collect();
        eprintln!(
            "  ERROR: {} nodes unreachable from root. First few: {:?}",
            unreachable.len(),
            &unreachable[..unreachable.len().min(20)]
        );
        errors += 1;
    }

    if errors == 0 {
        println!(
            "[LBVH] Topology OK: all {} nodes valid, all {} colliders present, all AABBs consistent",
            total_nodes, n
        );
    } else {
        eprintln!("[LBVH] VALIDATION FAILED: {} errors found", errors);
    }
}
