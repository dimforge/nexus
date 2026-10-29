use nexus_rbd_shaders3d::{dynamics::*, utils::BatchIndices};
use serde_json::{Value, json};
use std::mem::{offset_of, size_of};
#[path = "scene_layout.rs"]
mod scene_layout;
pub fn layout() -> Value {
    let mut l = scene_layout::layout();
    macro_rules! fields{($key:literal,$t:ty,$($f:ident),+)=>{l[$key]=json!({$(stringify!($f):offset_of!($t,$f)/4),+});};}
    fields!("impulse_joint", ImpulseJoint, body_a, body_b, padding, data);
    fields!(
        "generic_joint",
        GenericJoint,
        local_frame_a,
        local_frame_b,
        locked_axes,
        limit_axes,
        motor_axes,
        coupled_axes,
        limits,
        motors
    );
    fields!(
        "motor", JointMotor, target_vel, target_pos, stiffness, damping, max_force, impulse, model
    );
    fields!("limits", JointLimits, min, max, impulse);
    fields!(
        "builder",
        JointConstraintBuilder,
        body1,
        body2,
        joint_id,
        constraint_id,
        joint
    );
    fields!(
        "joint_constraint",
        JointConstraint,
        im_a,
        im_b,
        solver_vel_a,
        solver_vel_b,
        elements,
        len,
        padding
    );
    fields!(
        "joint_element",
        JointConstraintElement,
        lin_jac,
        joint_id,
        ang_jac_a,
        impulse,
        ang_jac_b,
        inv_lhs,
        ii_ang_jac_a,
        rhs,
        ii_ang_jac_b,
        rhs_wo_bias,
        cfm_gain,
        cfm_coeff,
        impulse_bounds
    );
    for (name, words) in [
        ("impulse_joint_words", size_of::<ImpulseJoint>() / 4),
        ("generic_joint_words", size_of::<GenericJoint>() / 4),
        ("motor_words", size_of::<JointMotor>() / 4),
        ("limits_words", size_of::<JointLimits>() / 4),
        ("builder_words", size_of::<JointConstraintBuilder>() / 4),
        ("joint_constraint_words", size_of::<JointConstraint>() / 4),
        (
            "joint_element_words",
            size_of::<JointConstraintElement>() / 4,
        ),
    ] {
        l[name] = json!(words);
    }
    l["batch"]["impulse_joints_len"] = json!(offset_of!(BatchIndices, impulse_joints_len) / 4);
    l["rbd_params"]["joint_damping_ratio"] =
        json!(offset_of!(RbdSimParams, joint_damping_ratio) / 4);
    l["rbd_params"]["joint_natural_frequency"] =
        json!(offset_of!(RbdSimParams, joint_natural_frequency) / 4);
    l
}
