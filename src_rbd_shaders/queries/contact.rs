//! Contact computation for shape pairs (ball-ball, cuboid-ball, cuboid-cuboid, etc).
//!
//! All functions operate in the local frame of the first shape, with the second
//! shape's pose provided as a relative transform.

use crate::queries::contact_manifold::ContactManifold;
use crate::queries::polygonal_feature;
use crate::queries::sat::{self, SeparatingAxis};
use crate::shapes::{Ball, Cuboid, Shape};
#[cfg(feature = "dim3")]
use crate::{Pad, Pose};
#[cfg(feature = "dim2")]
use crate::{PaddedVector, Pose};
use glamx::UVec2;
use khal_std::index::MaybeIndexUnchecked;

use super::contact_pfm_pfm;
#[cfg(feature = "dim2")]
use crate::MAX_FLT;
use crate::Vector;

/// Per-collider contact material, mirroring rapier's `ColliderMaterial`.
///
/// Friction and restitution are stored per collider; when two colliders touch,
/// their coefficients are merged with [`ColliderMaterial::combined_friction`] /
/// [`ColliderMaterial::combined_restitution`] (the rapier
/// `CoefficientCombineRule`). The `*_combine_rule` fields hold the rule as
/// `CoefficientCombineRule as u32` (Average = 0 .. ClampedSum = 4).
#[derive(Clone, Copy)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct ColliderMaterial {
    /// Coulomb friction coefficient (μ).
    pub friction: f32,
    /// Restitution coefficient (0 = no bounce, 1 = fully elastic).
    pub restitution: f32,
    /// `CoefficientCombineRule as u32` used to merge friction with the other collider's.
    pub friction_combine_rule: u32,
    /// `CoefficientCombineRule as u32` used to merge restitution with the other collider's.
    pub restitution_combine_rule: u32,
}

impl Default for ColliderMaterial {
    #[inline]
    fn default() -> Self {
        // Matches rapier's `ColliderMaterial::default()`: μ = 0.5, no bounce,
        // Average combine rule (0) for both coefficients.
        ColliderMaterial {
            friction: 0.5,
            restitution: 0.0,
            friction_combine_rule: 0,
            restitution_combine_rule: 0,
        }
    }
}

/// Combine two coefficients following rapier's `CoefficientCombineRule::combine`:
/// the "stronger" of the two rules wins (Average < Min < Multiply < Max < ClampedSum).
#[inline(always)]
fn combine_coefficient(coeff1: f32, coeff2: f32, rule1: u32, rule2: u32) -> f32 {
    let effective_rule = if rule1 > rule2 { rule1 } else { rule2 };
    if effective_rule == 1 {
        // Min — godot use-case relies on the `.abs()`, mirror rapier.
        coeff1.min(coeff2).abs()
    } else if effective_rule == 2 {
        // Multiply.
        coeff1 * coeff2
    } else if effective_rule == 3 {
        // Max.
        coeff1.max(coeff2)
    } else if effective_rule == 4 {
        // ClampedSum.
        (coeff1 + coeff2).clamp(0.0, 1.0)
    } else {
        // Average (default, rule 0).
        (coeff1 + coeff2) * 0.5
    }
}

impl ColliderMaterial {
    /// Effective friction for a contact between `self` and `other`.
    #[inline(always)]
    pub fn combined_friction(&self, other: &ColliderMaterial) -> f32 {
        combine_coefficient(
            self.friction,
            other.friction,
            self.friction_combine_rule,
            other.friction_combine_rule,
        )
    }

    /// Effective restitution for a contact between `self` and `other`.
    #[inline(always)]
    pub fn combined_restitution(&self, other: &ColliderMaterial) -> f32 {
        combine_coefficient(
            self.restitution,
            other.restitution,
            self.restitution_combine_rule,
            other.restitution_combine_rule,
        )
    }
}

/// Contact manifold with collider pair indices for solver integration.
///
/// This structure extends ContactManifold with the collider indices,
/// allowing the physics solver to identify which bodies are in contact.
#[derive(Clone, Copy, Default)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct IndexedManifold {
    /// The contact information.
    pub contact: ContactManifold,
    /// Collider pair that resulted in this contact.
    pub colliders: UVec2,
    /// Parent rigid bodies of `colliders.x` / `colliders.y`.
    pub bodies: UVec2,
    /// Combined Coulomb friction coefficient of the two colliders (see
    /// [`ColliderMaterial::combined_friction`]). Resolved at narrow-phase time
    /// and consumed by both the rigid-body and multibody contact solvers.
    pub friction: f32,
    /// Combined restitution coefficient of the two colliders (see
    /// [`ColliderMaterial::combined_restitution`]).
    pub restitution: f32,
    /// Padding so the struct size stays a multiple of 16 bytes — std430 storage
    /// buffers require the array stride to satisfy the 16-byte alignment of the
    /// inner vector members.
    pub _padding: [f32; 2],
}

/// Computes the contact between two balls.
pub fn ball_ball(pose12: Pose, ball1: &Ball, ball2: &Ball) -> ContactManifold {
    let r1 = ball1.radius;
    let r2 = ball2.radius;
    let center2_1 = pose12.translation;
    let mut normal1 = Vector::Y;

    let distance = center2_1.length();
    let sum_radius = r1 + r2;

    if distance != 0.0 {
        normal1 = center2_1 / distance;
    }

    let point1 = normal1 * r1;

    ContactManifold::single_point(point1, distance - sum_radius, normal1)
}

/// Computes the contact between a convex shape and a ball.
pub fn convex_ball(pose12: Pose, shape1: &Shape, ball2: &Ball) -> ContactManifold {
    let center2_1 = pose12.translation;
    let proj = shape1.project_local_point_on_boundary(center2_1);
    let proj_vec = center2_1 - proj.point;
    let mut dist = proj_vec.length();
    let mut normal1 = if dist != 0.0 {
        proj_vec / dist
    } else {
        Vector::Y
    };

    if proj.is_inside {
        normal1 = -normal1;
        dist = -dist;
    }

    ContactManifold::single_point(proj.point, dist - ball2.radius, normal1)
}

/// Computes the contact between a ball and a convex shape.
pub fn ball_convex(pose12: Pose, ball1: &Ball, shape2: &Shape) -> ContactManifold {
    let pose21 = pose12.inverse();
    let mut result = convex_ball(pose21, shape2, ball1);
    let normal1 = -(pose12.rotation * result.normal_a);
    result.points_a.at_mut(0).pt = normal1 * ball1.radius;
    result.normal_a = normal1;
    result
}

/// Computes the contact manifold between two polygonal feature-based shapes.
#[cfg(feature = "dim2")]
pub fn pfm_pfm(
    pose12: Pose,
    shape1: &Shape,
    thickness1: f32,
    shape2: &Shape,
    thickness2: f32,
    prediction: f32,
    vertices: &[PaddedVector],
) -> ContactManifold {
    contact_pfm_pfm::contact_manifold_pfm_pfm(
        pose12, shape1, thickness1, shape2, thickness2, prediction, vertices,
    )
}

/// Computes the contact manifold between two polygonal feature-based shapes.
#[cfg(feature = "dim3")]
pub fn pfm_pfm(
    pose12: Pose,
    shape1: &Shape,
    thickness1: f32,
    shape2: &Shape,
    thickness2: f32,
    prediction: f32,
    vertices: &[Pad<crate::Vector, u32>],
    indices: &[u32],
) -> ContactManifold {
    contact_pfm_pfm::contact_manifold_pfm_pfm(
        pose12, shape1, thickness1, shape2, thickness2, prediction, vertices, indices,
    )
}

/// Computes the contact between two cuboids.
pub fn cuboid_cuboid(
    pose12: Pose,
    cuboid1: &Cuboid,
    cuboid2: &Cuboid,
    prediction: f32,
) -> ContactManifold {
    let pose21 = pose12.inverse();

    /*
     *
     * Point-Face
     *
     */
    let sep1 = sat::cuboid_cuboid_find_local_separating_normal_oneway(cuboid1, cuboid2, pose12);

    // Early-exit: any contact point's distance is >= the separation along a
    // separating axis, so the caller would drop the manifold anyway.
    if sep1.separation > prediction {
        return ContactManifold::default();
    }

    let sep2 = sat::cuboid_cuboid_find_local_separating_normal_oneway(cuboid2, cuboid1, pose21);

    if sep2.separation > prediction {
        return ContactManifold::default();
    }

    /*
     *
     * Edge-Edge cases
     *
     */
    #[cfg(feature = "dim2")]
    let sep3 = SeparatingAxis::new(-MAX_FLT, Vector::new(1.0, 0.0)); // This case does not exist in 2D.
    #[cfg(feature = "dim3")]
    let sep3 = sat::cuboid_cuboid_find_local_separating_edge_twoway(cuboid1, cuboid2, pose12);

    if sep3.separation > prediction {
        return ContactManifold::default();
    }

    /*
     *
     * Select the best combination of features
     * and get the polygons to clip.
     *
     */
    let mut best_sep = sep1;

    if sep2.separation > sep1.separation && sep2.separation > sep3.separation {
        best_sep = SeparatingAxis::new(sep2.separation, pose12.rotation * -sep2.axis);
    } else if sep3.separation > sep1.separation {
        best_sep = sep3;
    }

    let local_n2 = pose21.rotation * -best_sep.axis;

    // Now the reference feature is from `cuboid1` and the best separation is `best_sep`.
    // Everything must be expressed in the local-space of `cuboid1` for contact clipping.
    let face1 = cuboid1.support_face(best_sep.axis);
    let face2 = cuboid2.support_face(local_n2);
    let mut manifold = polygonal_feature::contacts(
        pose12,
        pose21,
        best_sep.axis,
        local_n2,
        &face1.into(),
        &face2.into(),
        prediction,
        false,
    );
    manifold.normal_a = best_sep.axis;
    manifold
}
