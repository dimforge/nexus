use super::builder::{self, Scene};
use super::{box_rain, joint_lattice, pyramid, wrecking_ball};
use nexus_viewer2d::{DemoKind, NexusViewer};
use nexus2d::prelude::NexusPipeline;

/// A stress scene: its display name and its builder.
type SceneEntry = (&'static str, fn() -> Scene);

const SCENES: &[SceneEntry] = &[
    ("Pyramid 50 (1.3k)", pyramid::pyramid_50),
    ("Pyramid 100 (5k)", pyramid::pyramid_100),
    ("Pyramid 200 (20k)", pyramid::pyramid_200),
    ("Box rain 40x25 (1k)", box_rain::box_rain_40x25),
    ("Box rain 100x50 (5k)", box_rain::box_rain_100x50),
    ("Box rain 900x100 (90k)", box_rain::box_rain_900x100),
    (
        "Joint lattice 64x64 (4k)",
        joint_lattice::joint_lattice_64x64,
    ),
    (
        "Joint lattice 320x320 (100k)",
        joint_lattice::joint_lattice_320x320,
    ),
    (
        "Joint lattice 512x512 (262k)",
        joint_lattice::joint_lattice_512x512,
    ),
    (
        "Wrecking ball 100x40 (4k)",
        wrecking_ball::wrecking_ball_100x40,
    ),
    (
        "Wrecking ball 400x100 (40k)",
        wrecking_ball::wrecking_ball_400x100,
    ),
];

/// The stress scenes for the demo picker.
pub fn demo_list() -> Vec<(String, DemoKind)> {
    SCENES
        .iter()
        .map(|(name, _)| (name.to_string(), DemoKind::Stress))
        .collect()
}

/// Runs the stress scene called `name`, or returns `None` if there is none.
pub async fn run(
    name: &str,
    viewer: &mut NexusViewer,
    pipeline: &mut NexusPipeline,
) -> Option<anyhow::Result<()>> {
    let (_, build) = SCENES.iter().find(|(scene, _)| *scene == name)?;
    Some(builder::run(viewer, pipeline, build()).await.map(|_| ()))
}
