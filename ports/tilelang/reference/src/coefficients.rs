#[allow(dead_code)]
#[path = "colored.rs"]
mod colored;
use nexus_rbd_shaders3d::dynamics::RbdSimParams;
use serde_json::json;
use std::{fs, path::Path};
fn main() {
    let out = std::env::args().nth(1).unwrap();
    let root = Path::new(&out);
    fs::create_dir_all(root).unwrap();
    fs::write(
        root.join("layout.json"),
        serde_json::to_vec_pretty(&colored::layout()).unwrap(),
    )
    .unwrap();
    let root = root.join("contact_coefficients");
    fs::create_dir_all(&root).unwrap();
    let mut cases = Vec::new();
    for kind in 0..10 {
        let mut p = RbdSimParams::tgs_soft();
        match kind {
            1 => p.dt = 0.,
            2 => p.contact_natural_frequency = 0.,
            3 => p.static_contact_natural_frequency = 0.,
            4 => {
                p.normalized_max_corrective_velocity = nexus_rbd_shaders3d::MAX_FLT;
                p.length_unit = 2.;
            }
            5 => {
                p.dt = 1. / 120.;
                p.length_unit = 0.01;
            }
            6 => {
                p.contact_damping_ratio = 0.7;
                p.static_contact_damping_ratio = 12.;
            }
            7 => {
                p.contact_natural_frequency = 1e6;
                p.static_contact_natural_frequency = 1.;
            }
            8 | 9 => {
                let bits = nexus_rbd_shaders3d::MAX_FLT.to_bits();
                p.normalized_max_corrective_velocity =
                    f32::from_bits(if kind == 8 { bits - 1 } else { bits + 1 });
                p.length_unit = 2.;
            }
            _ => {}
        }
        let d = root.join(format!("p{kind}"));
        fs::create_dir_all(&d).unwrap();
        fs::write(d.join("RbdParams.bin"), bytemuck::bytes_of(&p)).unwrap();
        let mut expected = [0xdeadbeefu32; 9];
        for (i, v) in [
            p.inv_dt(),
            p.max_corrective_velocity(),
            p.contact_cfm_factor(),
            p.contact_erp_inv_dt(),
            p.static_contact_cfm_factor(),
            p.static_contact_erp_inv_dt(),
        ]
        .into_iter()
        .enumerate()
        {
            expected[i] = v.to_bits();
        }
        fs::write(
            d.join("Coefficients.expected.bin"),
            bytemuck::cast_slice(&expected),
        )
        .unwrap();
        fs::write(
            d.join("Coefficients.bin"),
            bytemuck::cast_slice(&[0xdeadbeefu32; 9]),
        )
        .unwrap();
        cases.push(json!({"name":format!("p{kind}"),"kind":"contact_coefficients","count":6,"iterations":3,"buffer_words":{"RbdParams":std::mem::size_of::<RbdSimParams>()/4,"Coefficients":9}}));
    }
    fs::write(
        root.join("suite.json"),
        serde_json::to_vec_pretty(&json!({"cases":cases})).unwrap(),
    )
    .unwrap();
}
