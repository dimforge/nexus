//! Explicit units for cooperative CPU oracle dispatches.
use khal::backend::{DispatchGrid, GpuBackend};

pub(crate) fn workgroups(grid: [u32; 3]) -> DispatchGrid<'static, GpuBackend> {
    // `From<[u32; 3]>` means thread counts in khal, not workgroup counts.
    DispatchGrid::Grid(grid)
}

#[cfg(test)]
mod tests {
    use super::*;
    use khal::backend::{GpuBuffer, GpuPass};
    use nexus_rbd_shaders3d::{
        dynamics::{GpuHubScatter, HubCounts, Velocity},
        glamx::Vec3,
    };

    #[test]
    fn explicit_grid_reaches_the_second_workgroup() {
        let kernel =
            GpuHubScatter::from_bytes(&GpuBackend::Cpu, &[], GpuHubScatter::ENTRY_POINT).unwrap();
        let original = Velocity {
            padding0: 0x12345678,
            padding1: 0x87654321,
            ..Velocity::new(Vec3::splat(2.0), Vec3::splat(3.0))
        };
        let untouched = Velocity::new(Vec3::splat(-7.0), Vec3::splat(11.0));
        let mut velocities = GpuBuffer::Cpu(vec![untouched; 130]);
        velocities.unwrap_slice_mut()[0] = original;
        kernel
            .call(
                &mut GpuPass::Cpu(None),
                workgroups([2, 1, 1]),
                &mut velocities,
                &GpuBuffer::Cpu(vec![0u32; 128]),
                &GpuBuffer::Cpu(vec![HubCounts {
                    hubs: 1,
                    slots: 128,
                    base: 1,
                    pool: 128,
                }]),
            )
            .unwrap();
        for velocity in &velocities.unwrap_slice()[1..129] {
            assert_eq!(*velocity, original);
        }
        assert_eq!(velocities.unwrap_slice()[129], untouched);
    }
}
