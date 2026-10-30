//! Native control feedback, including stale inactive links and palette exhaustion.
use nexus_rbd_shaders3d::{
    broad_phase::ContactPlan,
    dynamics::{ContactLink, gpu_coloring_requirements},
    glamx::UVec3,
};
use serde_json::json;
use std::{
    fs,
    mem::{offset_of, size_of},
    path::Path,
};
fn write<T: bytemuck::Pod>(dir: &Path, name: &str, values: &[T]) {
    fs::write(dir.join(name), bytemuck::cast_slice(values)).unwrap();
}
fn main() {
    let root = std::env::args().nth(1).expect("output directory");
    let root = Path::new(&root);
    fs::create_dir_all(root).unwrap();
    let layout = json!({"link_words":size_of::<ContactLink>()/4,"link":{"len":offset_of!(ContactLink,len)/4},"plan_words":size_of::<ContactPlan>()/4,"plan":{"bound":offset_of!(ContactPlan,bound)/4}});
    fs::write(
        root.join("layout.json"),
        serde_json::to_vec_pretty(&layout).unwrap(),
    )
    .unwrap();
    let root = root.join("color_requirements");
    fs::create_dir_all(&root).unwrap();
    let mut cases = Vec::new();
    for n in [0usize, 1, 31, 32, 33, 63, 64, 65, 127] {
        for mode in ["valid", "pending", "invalid", "inactive", "max-rounds"] {
            let name = format!("n{n}-{mode}");
            let dir = root.join(&name);
            fs::create_dir_all(&dir).unwrap();
            let mut links = vec![ContactLink::default(); n + 3];
            bytemuck::cast_slice_mut::<_, u32>(&mut links).fill(u32::MAX);
            let mut colors = vec![0xDEADBEEF; n + 3];
            for i in 0..n {
                links[i].len = if mode == "inactive" || i % 5 == 4 {
                    0
                } else {
                    1 + (i % 4) as u32
                };
                colors[i] = if links[i].len == 0 {
                    u32::MAX
                } else {
                    1 + (i % 63) as u32
                };
            }
            if n > 0 && matches!(mode, "pending" | "max-rounds") {
                colors[0] = 0;
            }
            if n > 0 && mode == "invalid" {
                colors[0] = 64;
            }
            let mut plan: ContactPlan = bytemuck::Zeroable::zeroed();
            plan.bound = n as u32;
            let budget = [if mode == "max-rounds" { u32::MAX } else { 7 }, 0xDEADBEEF];
            let mut requirements = [0xDEADBEEF; 5];
            write(&dir, "Colors.bin", &colors);
            write(&dir, "Links.bin", &links);
            write(&dir, "Plan.bin", &[plan]);
            write(&dir, "Budget.bin", &budget);
            write(&dir, "Requirements.bin", &requirements);
            gpu_coloring_requirements(
                UVec3::ZERO,
                &colors,
                &links,
                &plan,
                &budget[0],
                &mut requirements,
            );
            write(&dir, "Requirements.expected.bin", &requirements);
            cases.push(json!({"name":name,"kind":"color_requirements","count":n,"storage":n+3,"groups":1,"threads":1,"iterations":1,"buffer_words":{"Colors":colors.len(),"Links":links.len()*size_of::<ContactLink>()/4,"Plan":size_of::<ContactPlan>()/4,"Budget":2,"Requirements":5}}));
        }
    }
    fs::write(
        root.join("suite.json"),
        serde_json::to_vec_pretty(&json!({"cases":cases})).unwrap(),
    )
    .unwrap();
}
