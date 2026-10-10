use super::builder::{self, Scene};
use super::{box_columns, box_pile, brick_ring, brick_walls, jointed_drop, ragdolls_on_cloth};
use nexus_viewer3d::{DemoKind, NexusViewer};
use nexus3d::prelude::NexusPipeline;

/// A stress scene: its display name and its builder.
type SceneEntry = (&'static str, fn() -> Scene);

const SCENES: &[SceneEntry] = &[
    ("Brick ring (28k)", brick_ring::brick_ring_28k),
    ("Brick ring (110k)", brick_ring::brick_ring_110k),
    ("Brick walls (27k)", brick_walls::brick_walls_27k),
    ("Pyramids (500k)", brick_walls::pyramids_500k),
    (
        "Ragdolls on cloth (24k)",
        ragdolls_on_cloth::ragdolls_on_cloth,
    ),
    ("Box pile (4k)", box_pile::box_pile_4k),
    ("Box pile (32k)", box_pile::box_pile_32k),
    ("Jointed drop (34k)", jointed_drop::jointed_drop),
    ("Box columns (100k)", box_columns::box_columns_100k),
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
