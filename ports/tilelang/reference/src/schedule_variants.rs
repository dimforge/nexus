//! Native fused/tail entry points versus the ordered color schedule used by TileLang.
#[cfg(test)]
#[path = "dispatch.rs"]
mod dispatch;
fn main() {
    println!("Run this oracle with cargo test --bin schedule-variants through cpu_oracle.py");
}

#[cfg(test)]
mod tests {
    use super::dispatch::workgroups;
    use khal::backend::{GpuBackend, GpuBuffer, GpuPass};
    use nexus_rbd_shaders3d::{
        Pose,
        dynamics::{RbdSimParams, TwistConstraint, TwistContactPoint, TwoBodyConstraint, Velocity},
        glamx::{Vec2, Vec3},
        utils::BatchIndices,
    };

    fn constraints(width: usize, batches: u32) -> (Vec<TwoBodyConstraint>, Vec<u32>) {
        let mut rows = Vec::new();
        let mut buckets = Vec::new();
        for color in 0..=64u32 {
            for batch in 0..batches {
                if [24, 25, 62, 63].contains(&color) {
                    for lane in 0..width {
                        let first = ((2 * lane + color as usize % 2) % (2 * width + 2)) as u32;
                        let a = first * batches + batch;
                        let b = (first + 1) * batches + batch;
                        let mut c = TwoBodyConstraint {
                            len: 1 + (lane % 4) as u32,
                            dir_a: Vec3::Y,
                            tangent_a: Vec3::Z,
                            limit: 0.7,
                            im_a: Vec3::splat(0.4),
                            im_b: Vec3::splat(0.6),
                            solver_body_a: a,
                            solver_body_b: b,
                            vel_slot_a: a,
                            vel_slot_b: b,
                            ..Default::default()
                        };
                        c.ii_a.diag = Vec3::splat(0.2);
                        c.ii_b.diag = Vec3::splat(0.3);
                        for k in 0..c.len as usize {
                            let q = &mut c.points[k];
                            q.r_a = Vec3::new(k as f32 * 0.07, -0.2, 0.1);
                            q.r_b = Vec3::new(k as f32 * 0.07, 0.2, 0.1);
                            q.local_pt_a = q.r_a;
                            q.local_pt_b = q.r_b;
                            q.dist = -0.01;
                            q.normal_impulse = 0.04 * (k + 1) as f32;
                            q.tangent_impulse = Vec2::new(0.01, -0.02);
                        }
                        c.compute_effective_masses();
                        rows.push(c);
                    }
                }
                buckets.push(rows.len() as u32);
            }
        }
        (rows, buckets)
    }
    fn twist(c: &TwoBodyConstraint) -> TwistConstraint {
        let mut t = TwistConstraint {
            dir_a: c.dir_a,
            tangent_a: c.tangent_a,
            len: c.len,
            limit: c.limit,
            im_a: c.im_a,
            im_b: c.im_b,
            ii_a: c.ii_a,
            ii_b: c.ii_b,
            solver_body_a: c.solver_body_a,
            solver_body_b: c.solver_body_b,
            vel_slot_a: c.vel_slot_a,
            vel_slot_b: c.vel_slot_b,
            offset_b: c.points[0].r_b - c.points[0].r_a,
            ..Default::default()
        };
        for k in 0..t.len as usize {
            let q = c.points[k];
            t.points[k] = TwistContactPoint {
                r_a: q.r_a,
                dist: q.dist,
                normal_impulse: q.normal_impulse,
                normal_vel: q.normal_vel,
                ..Default::default()
            };
        }
        t.friction.tangent_impulse = Vec2::new(0.01, -0.02);
        t.friction.twist_impulse = if t.len > 1 { 0.01 } else { 0. };
        t.compute_effective_masses();
        t
    }
    macro_rules! check_model {
        ($module:ident, $tile:ident, $convert:expr) => {{
            use nexus_rbd_shaders3d::dynamics::$module::*;
            let backend=GpuBackend::Cpu;
            let mut scenarios=0;
            for width in [0usize,1,33,95] {for batches in [1u32,3] {for mode in [0u32,1,2] {
                let (input,buckets)=constraints(width,batches);
                let mut initial_tiles=vec![$tile::default();input.len().div_ceil(32)+1];
                for (i,c) in input.iter().enumerate(){initial_tiles[i/32].write_constraint(i%32,&($convert)(c));}
                let body_count=(width*2+3)*batches as usize;
                let initial_vels:Vec<_>=(0..body_count).map(|i|Velocity::new(Vec3::new((i%7) as f32*0.03,-0.1,0.07),Vec3::new(0.1,0.03,(i%3) as f32*0.02))).collect();
                let poses=GpuBuffer::Cpu(vec![Pose::IDENTITY;body_count]);
                let buckets=GpuBuffer::Cpu(buckets);
                let ids=GpuBuffer::Cpu(vec![BatchIndices{num_batches:batches,solver_color_buckets_stride:65,..Default::default()}]);
                let params=GpuBuffer::Cpu(vec![RbdSimParams::tgs_soft()]);
                let mode_buf=GpuBuffer::Cpu(vec![mode]); let last=GpuBuffer::Cpu(vec![63u32]);
                for cached in [false,true] {
                    let mut outputs=Vec::new();
                    for schedule in 0..3 {
                        let mut tiles=GpuBuffer::Cpu(initial_tiles.clone()); let mut vels=GpuBuffer::Cpu(initial_vels.clone());
                        let mut pass=GpuPass::Cpu(None);
                        if schedule == 2 {
                            GpuWarmstartConstraintsFused::from_bytes(&backend,&[],GpuWarmstartConstraintsFused::ENTRY_POINT).unwrap().call(&mut pass,workgroups([1,batches,1]),&tiles,&mut vels,&buckets,&last,&ids).unwrap();
                        } else {
                            for color in 1..=63 {
                                let color=GpuBuffer::Cpu(vec![color]);
                                GpuWarmstartConstraints::from_bytes(&backend,&[],GpuWarmstartConstraints::ENTRY_POINT).unwrap().call(&mut pass,workgroups([2,1,1]),&tiles,&mut vels,&buckets,&color,&ids).unwrap();
                            }
                        }
                        let warm=bytemuck::cast_slice::<_,u32>(vels.unwrap_slice()).to_vec();
                        for _ in 0..3 {
                            if schedule==0 {
                                for color in 1..=63 {
                                    let color=GpuBuffer::Cpu(vec![color]);
                                    if cached {
                                        GpuSolveConstraintsUnbiasedCached::from_bytes(&backend,&[],GpuSolveConstraintsUnbiasedCached::ENTRY_POINT).unwrap().call(&mut pass,workgroups([2,1,1]),&mut tiles,&mut vels,&buckets,&poses,&color,&ids,&mode_buf,&params).unwrap();
                                    } else {
                                        GpuSolveConstraints::from_bytes(&backend,&[],GpuSolveConstraints::ENTRY_POINT).unwrap().call(&mut pass,workgroups([2,1,1]),&mut tiles,&mut vels,&buckets,&poses,&color,&ids,&mode_buf,&params).unwrap();
                                    }
                                }
                            } else if schedule==1 {
                                if cached {
                                    GpuSolveConstraintsTailUnbiasedCached::from_bytes(&backend,&[],GpuSolveConstraintsTailUnbiasedCached::ENTRY_POINT).unwrap().call(&mut pass,workgroups([1,1,1]),&mut tiles,&mut vels,&buckets,&poses,&last,&ids,&mode_buf,&params).unwrap();
                                } else {
                                    GpuSolveConstraintsTail::from_bytes(&backend,&[],GpuSolveConstraintsTail::ENTRY_POINT).unwrap().call(&mut pass,workgroups([1,1,1]),&mut tiles,&mut vels,&buckets,&poses,&last,&ids,&mode_buf,&params).unwrap();
                                }
                            } else if cached {
                                let unbiased=GpuBuffer::Cpu(vec![0u32]);
                                GpuSolveConstraintsFusedCached::from_bytes(&backend,&[],GpuSolveConstraintsFusedCached::ENTRY_POINT).unwrap().call(&mut pass,workgroups([1,batches,1]),&mut tiles,&mut vels,&buckets,&poses,&last,&ids,&unbiased,&params).unwrap();
                            } else {
                                GpuSolveConstraintsFused::from_bytes(&backend,&[],GpuSolveConstraintsFused::ENTRY_POINT).unwrap().call(&mut pass,workgroups([1,batches,1]),&mut tiles,&mut vels,&buckets,&poses,&last,&ids,&mode_buf,&params).unwrap();
                            }
                        }
                        outputs.push((warm,bytemuck::cast_slice::<_,u32>(tiles.unwrap_slice()).to_vec(),bytemuck::cast_slice::<_,u32>(vels.unwrap_slice()).to_vec()));
                    }
                    assert!(outputs[0] == outputs[1],"tail model {} width {width} batches {batches} mode {mode} cached {cached}",stringify!($module));
                    assert!(outputs[0] == outputs[2],"fused model {} width {width} batches {batches} mode {mode} cached {cached}",stringify!($module));
                    scenarios+=1;
                }
            }}}
            println!("{}: {scenarios} native schedule comparisons bitwise equal",stringify!($module));
        }};
    }
    #[test]
    fn coulomb_native_fused_tail_equal_ordered() {
        check_model!(coulomb_tiles, CoulombTile, |c: &TwoBodyConstraint| *c);
    }
    #[test]
    fn twist_native_fused_tail_equal_ordered() {
        check_model!(twist_tiles, TwistTile, twist);
    }
}
