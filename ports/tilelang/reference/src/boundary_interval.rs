//! f32 roundoff envelope for scalar and SIMD quaternion transforms.
use nexus_rbd_shaders3d::{Pose, dynamics::TwoBodyConstraint, glamx::Vec3};
#[derive(Clone, Copy, Debug)]
struct Interval {
    lo: f64,
    hi: f64,
}
impl Interval {
    fn exact(x: f32) -> Self {
        Self {
            lo: x as f64,
            hi: x as f64,
        }
    }
    fn rounded(lo: f64, hi: f64) -> Self {
        let u = 2.0f64.powi(-24);
        let error = lo.abs().max(hi.abs()) * u / (1.0 - u) + 2.0f64.powi(-149);
        Self {
            lo: (lo - error).next_down(),
            hi: (hi + error).next_up(),
        }
    }
    fn add(self, b: Self) -> Self {
        Self::rounded(self.lo + b.lo, self.hi + b.hi)
    }
    fn neg(self) -> Self {
        Self {
            lo: -self.hi,
            hi: -self.lo,
        }
    }
    fn sub(self, b: Self) -> Self {
        self.add(b.neg())
    }
    fn mul(self, b: Self) -> Self {
        let p = [
            self.lo * b.lo,
            self.lo * b.hi,
            self.hi * b.lo,
            self.hi * b.hi,
        ];
        Self::rounded(
            p.into_iter().fold(f64::INFINITY, f64::min),
            p.into_iter().fold(f64::NEG_INFINITY, f64::max),
        )
    }
    fn twice(self) -> Self {
        self.mul(Self::exact(2.0))
    }
    fn union(self, b: Self) -> Self {
        Self {
            lo: self.lo.min(b.lo),
            hi: self.hi.max(b.hi),
        }
    }
}
fn dot(a: [Interval; 3], b: [Interval; 3]) -> Interval {
    let p = std::array::from_fn::<_, 3, _>(|i| a[i].mul(b[i]));
    p[0].add(p[1]).add(p[2]).union(p[0].add(p[1].add(p[2])))
}
fn cross(a: [Interval; 3], b: [Interval; 3]) -> [Interval; 3] {
    [
        a[1].mul(b[2]).sub(a[2].mul(b[1])),
        a[2].mul(b[0]).sub(a[0].mul(b[2])),
        a[0].mul(b[1]).sub(a[1].mul(b[0])),
    ]
}
fn transform(pose: Pose, point: Vec3) -> [Interval; 3] {
    let q = pose.rotation.to_array().map(Interval::exact);
    let b = [q[0], q[1], q[2]];
    let w = q[3];
    let v = point.to_array().map(Interval::exact);
    let uv = cross(b, v);
    let uuv = cross(b, uv);
    let doubled = uv.map(Interval::twice);
    let last = cross(b, doubled);
    let factor = w.mul(w).sub(dot(b, b));
    let dp = dot(b, v);
    let t = pose.translation.to_array().map(Interval::exact);
    std::array::from_fn(|i| {
        let first = v[i].add(uv[i].mul(w).add(uuv[i]).twice());
        let second = v[i].add(w.mul(doubled[i])).add(last[i]);
        let third = v[i]
            .mul(factor)
            .add(b[i].mul(dp).twice())
            .add(uv[i].mul(w).twice());
        first.union(second).union(third).add(t[i])
    })
}
pub fn mask(c: &TwoBodyConstraint, poses: &[Pose]) -> u32 {
    let mut mask = 0;
    for k in 0..c.len as usize {
        let point = &c.points[k];
        let pa = poses[c.solver_body_a as usize];
        let pb = poses[c.solver_body_b as usize];
        let a = transform(pa, point.local_pt_a);
        let b = transform(pb, point.local_pt_b);
        let bounds = Interval::exact(point.dist).add(dot(
            std::array::from_fn(|i| a[i].sub(b[i])),
            c.dir_a.to_array().map(Interval::exact),
        ));
        let native = point.dist + (pa * point.local_pt_a - pb * point.local_pt_b).dot(c.dir_a);
        assert!(
            bounds.lo.is_finite()
                && bounds.hi.is_finite()
                && bounds.lo <= native as f64
                && native as f64 <= bounds.hi
        );
        if bounds.lo <= 0.0 && 0.0 <= bounds.hi {
            mask |= 1 << k;
        }
    }
    mask
}
