//! Exercise real shared-memory shader dispatch in both debug and optimized builds.
mod dispatch;
use khal::backend::{GpuBackend, GpuBuffer, GpuPass};
use nexus_rbd_shaders3d::dynamics::{GpuHubAverage, HubCounts, Velocity};
use nexus_rbd_shaders3d::glamx::Vec3;

fn check_average(degrees: &[u32], groups: u32) {
    let kernel =
        GpuHubAverage::from_bytes(&GpuBackend::Cpu, &[], GpuHubAverage::ENTRY_POINT).unwrap();
    let base = degrees.len() + 2;
    let slots: u32 = degrees.iter().sum();
    let mut counts = vec![0u32; base];
    let mut first = vec![u32::MAX; base];
    let sentinel = Velocity {
        padding1: 0x12345678,
        padding0: 0x87654321,
        ..Velocity::new(Vec3::splat(-17.0), Vec3::splat(23.0))
    };
    let mut initial = vec![sentinel; base + slots as usize + 5];
    let mut expected = initial.clone();
    let mut cursor = 0u32;
    for (i, &n) in degrees.iter().enumerate() {
        let body = i + 1;
        first[body] = cursor;
        let mut linear = [0i64; 3];
        let mut angular = [0i64; 3];
        for slot in 0..n {
            let x = ((slot * 11 + body as u32 * 7) % 97) as i32 - 48;
            let y = ((slot * 17 + body as u32 * 3) % 67) as i32 - 33;
            let l = [x, y, -x - y];
            let a = [y - x, -2 * y, x + 7];
            let mut velocity = sentinel;
            velocity.linear = Vec3::from_array(l.map(|v| v as f32));
            velocity.angular = Vec3::from_array(a.map(|v| v as f32));
            initial[base + (cursor + slot) as usize] = velocity;
            expected[base + (cursor + slot) as usize] = velocity;
            for axis in 0..3 {
                linear[axis] += i64::from(l[axis]);
                angular[axis] += i64::from(a[axis]);
            }
        }
        cursor += n;
        counts[body] = cursor;
        if n > 0 {
            // All integer sums are exactly representable. This expectation tests
            // every participating lane without copying the shader's reduction.
            let inv_n = 1.0 / n as f32;
            expected[body].linear = Vec3::from_array(linear.map(|v| v as f32)) * inv_n;
            expected[body].angular = Vec3::from_array(angular.map(|v| v as f32)) * inv_n;
        }
    }
    counts[base - 1] = cursor;
    let mut velocities = GpuBuffer::Cpu(initial);
    let list = GpuBuffer::Cpu((1..=degrees.len() as u32).rev().collect::<Vec<_>>());
    // live_hubs is bounded by pool / 32 + 1. Use enough bookkeeping capacity
    // even for the deliberately empty/small hub fixtures.
    let pool = slots.max(degrees.len() as u32 * 32);
    let hub_counts = GpuBuffer::Cpu(vec![HubCounts {
        hubs: degrees.len() as u32,
        slots,
        base: base as u32,
        pool,
    }]);
    let first = GpuBuffer::Cpu(first);
    let counts = GpuBuffer::Cpu(counts);
    for _ in 0..3 {
        kernel
            .call(
                &mut GpuPass::Cpu(None),
                dispatch::workgroups([groups, 1, 1]),
                &mut velocities,
                &list,
                &first,
                &counts,
                &hub_counts,
            )
            .unwrap();
        assert_eq!(
            bytemuck::cast_slice::<Velocity, u32>(velocities.unwrap_slice()),
            bytemuck::cast_slice::<Velocity, u32>(&expected),
            "groups={groups}, hubs={}",
            degrees.len()
        );
    }
}

fn main() {
    for groups in [1, 3, 16] {
        check_average(&[0, 1, 33, 63, 64, 65, 129], groups);
    }
    check_average(&[], 3);
    check_average(&vec![65; 257], 256);
    println!("Cooperative CPU hub reduction passed, including reused workgroups and padding");
}

#[test]
fn cooperative_hub_average() {
    main();
}
