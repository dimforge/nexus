//! Checked host state transfer for a native scene reconstruction.
use nexus_rbd3d::{Velocity, math::Pose, pipeline::RbdSceneUpload};
use std::collections::BTreeSet;

pub(crate) fn transfer_bodies(
    upload: &mut RbdSceneUpload,
    poses: &[Pose],
    velocities: &[Velocity],
    survivors: &[(usize, usize)],
) -> Result<(), String> {
    let count = (upload.batch_indices.bodies_len as usize)
        .checked_mul(upload.batch_indices.num_batches as usize)
        .ok_or("Body count overflow")?;
    validate_mapping(poses.len(), count, survivors)?;
    if velocities.len() != poses.len()
        || upload.body_poses.len() < count
        || upload.velocities.len() < count
    {
        return Err("Body state storage differs from active counts".into());
    }
    for &(old, new) in survivors {
        upload.body_poses[new] = poses[old];
        upload.velocities[new] = velocities[old];
    }
    Ok(())
}
fn validate_mapping(
    old_count: usize,
    new_count: usize,
    survivors: &[(usize, usize)],
) -> Result<(), String> {
    let mut from = BTreeSet::new();
    let mut to = BTreeSet::new();
    for &(old, new) in survivors {
        if old >= old_count || new >= new_count || !from.insert(old) || !to.insert(new) {
            return Err("Invalid or duplicate surviving body slot".into());
        }
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn remaps_must_be_one_to_one_and_inside_active_storage() {
        assert!(validate_mapping(6, 4, &[(0, 0), (1, 1), (4, 2), (5, 3)]).is_ok());
        assert!(validate_mapping(0, 0, &[]).is_ok());
        for pairs in [
            vec![(0, 0), (1, 0)],
            vec![(0, 0), (0, 1)],
            vec![(6, 0)],
            vec![(0, 4)],
            vec![(usize::MAX, 0)],
        ] {
            assert!(validate_mapping(6, 4, &pairs).is_err());
        }
    }
}
