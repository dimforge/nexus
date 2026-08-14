//! Sand draining through the neck of an hourglass.

use khal::backend::GpuTimestamps;
use nexus_viewer2d::NexusViewer;
use nexus2d::mpm::solver::{BoundaryCondition, Particle, ParticleModel, SimulationParams};
use nexus2d::prelude::{NexusPipeline, NexusState, RbdCoupling};

use glamx::{Vec4, vec2};
use rapier2d::prelude::{Collider, ColliderBuilder, Pose, RigidBodyBuilder};

const DENSITY: f32 = 1500.0;
const YOUNG_MODULUS: f32 = 1.0e7;
const POISSON_RATIO: f32 = 0.2;

/// Half-width of the neck the sand drains through.
const NECK_HALF_WIDTH: f32 = 0.9;
/// Height at which the funnel walls meet the vertical chamber walls.
const FUNNEL_TOP: f32 = 12.0;
/// Half-width of both chambers.
const CHAMBER_HALF_WIDTH: f32 = 12.0;

fn insert_boundary(
    state: &mut NexusState,
    viewer: &mut NexusViewer,
    collider: Collider,
    center: glamx::Vec2,
    angle: f32,
) {
    let shape = collider.shared_shape().clone();
    let body = RigidBodyBuilder::fixed()
        .translation(center)
        .rotation(angle)
        .build();
    let coupling = RbdCoupling::MpmOneWay(BoundaryCondition::separate(0.6));
    let handle = state.insert_rigid_body(body, collider, coupling);
    viewer.insert_shape_with_color(
        handle,
        &shape,
        Pose::IDENTITY,
        Vec4::new(0.4, 0.6, 0.8, 0.8),
    );
}

/// Inserts a bar spanning `start` -> `end` with the given half-thickness.
fn insert_bar(
    state: &mut NexusState,
    viewer: &mut NexusViewer,
    start: glamx::Vec2,
    end: glamx::Vec2,
    half_thickness: f32,
) {
    let delta = end - start;
    let half_len = delta.length() / 2.0;
    let angle = delta.y.atan2(delta.x);
    let center = (start + end) / 2.0;
    insert_boundary(
        state,
        viewer,
        ColliderBuilder::cuboid(half_len, half_thickness).build(),
        center,
        angle,
    );
}

pub async fn run(
    viewer: &mut NexusViewer,
    pipeline: &mut NexusPipeline,
) -> anyhow::Result<NexusState> {
    let mut state = NexusState::default();

    let cell_width = 0.2;
    let radius = cell_width / 4.0;
    let spacing = radius * 2.0;

    /*
     * Sand column filling the upper chamber.
     */
    let mut particles = vec![];
    let fill_min = vec2(-CHAMBER_HALF_WIDTH + 1.0, FUNNEL_TOP + 1.0);
    let fill_max = vec2(CHAMBER_HALF_WIDTH - 1.0, FUNNEL_TOP + 23.0);
    let nx = ((fill_max.x - fill_min.x) / spacing) as i32;
    let ny = ((fill_max.y - fill_min.y) / spacing) as i32;
    let model = ParticleModel::sand(YOUNG_MODULUS, POISSON_RATIO);
    // One group per row, shaded by height, so the draining order stays
    // readable: the top of the pile ends up on top of the heap below.
    let shades: Vec<_> = (0..ny)
        .map(|j| {
            let t = j as f32 / ny as f32;
            Vec4::new(0.95, 0.75 - 0.35 * t, 0.35 - 0.25 * t, 1.0)
        })
        .collect();
    viewer.set_particle_group_colors(&shades);
    for i in 0..nx {
        for j in 0..ny {
            let position = fill_min + vec2(i as f32 + 0.5, j as f32 + 0.5) * spacing;
            particles.push(Particle::with_group(
                position, radius, DENSITY, model, j as u32,
            ));
        }
    }

    let params = SimulationParams {
        gravity: vec2(0.0, -9.81),
        padding: 0.0,
        dt: 1.0 / 60.0,
    };
    state.set_mpm_params(viewer.backend(), params, cell_width)?;
    state.set_mpm_substeps(15);
    state.add_particles(viewer.backend(), particles)?;

    /*
     * Hourglass walls.
     */
    let thickness = 0.4;
    // Upper funnel.
    insert_bar(
        &mut state,
        viewer,
        vec2(NECK_HALF_WIDTH, 0.0),
        vec2(CHAMBER_HALF_WIDTH, FUNNEL_TOP),
        thickness,
    );
    insert_bar(
        &mut state,
        viewer,
        vec2(-NECK_HALF_WIDTH, 0.0),
        vec2(-CHAMBER_HALF_WIDTH, FUNNEL_TOP),
        thickness,
    );
    // Upper chamber sides.
    for side in [-1.0f32, 1.0] {
        insert_bar(
            &mut state,
            viewer,
            vec2(side * CHAMBER_HALF_WIDTH, FUNNEL_TOP),
            vec2(side * CHAMBER_HALF_WIDTH, FUNNEL_TOP + 26.0),
            thickness,
        );
    }
    // Lower chamber sides and floor.
    for side in [-1.0f32, 1.0] {
        insert_bar(
            &mut state,
            viewer,
            vec2(side * CHAMBER_HALF_WIDTH, -18.0),
            vec2(side * CHAMBER_HALF_WIDTH, 0.0),
            thickness,
        );
    }
    insert_bar(
        &mut state,
        viewer,
        vec2(-CHAMBER_HALF_WIDTH, -18.0),
        vec2(CHAMBER_HALF_WIDTH, -18.0),
        thickness,
    );

    let mut timestamps = GpuTimestamps::new(viewer.backend(), 2048);
    state.finalize(viewer.backend()).await?;

    while viewer.render_frame().await {
        if viewer.simulating() {
            pipeline
                .simulate(viewer.backend(), &mut state, Some(&mut timestamps))
                .await?;
        }
        viewer.sync(&mut state, Some(&mut timestamps)).await?;
    }

    Ok(state)
}
