use inflector::Inflector;
use nexus_viewer2d::{BackendType, DemoKind, NexusViewer};
use nexus2d::prelude::{NexusPipeline, NexusPipelineMask};

mod rbd_balls2;
mod rbd_boxes2;
mod rbd_boxes_and_balls2;
mod rbd_compound2;
mod rbd_dynamic2;
mod rbd_joint_ball2;
mod rbd_joint_fixed2;
mod rbd_joint_prismatic2;
mod rbd_polyline2;
mod rbd_primitives2;
mod rbd_pyramid2;

// MPM examples.
mod mpm_centilever_beam2;
mod mpm_cohesion_sweep2;
mod mpm_dam_break2;
mod mpm_elastic_cut2;
mod mpm_elasticity2;
mod mpm_emitter2;
mod mpm_hourglass2;
mod mpm_sand2;
mod mpm_snowball2;

/// Declares the demo registry: a `(name, kind)` list for the picker UI and a
/// name -> `run()` dispatcher. Keeping both in one macro keeps them in sync.
macro_rules! demos {
    ( $( $name:literal => $kind:ident : $module:ident ),* $(,)? ) => {
        fn demo_list() -> Vec<(String, DemoKind)> {
            let mut demos: Vec<(String, DemoKind)> =
                vec![ $( ($name.to_string(), DemoKind::$kind) ),* ];
            demos.sort_by(|a, b| match (a.0.starts_with('('), b.0.starts_with('(')) {
                (true, true) | (false, false) => a.0.cmp(&b.0),
                (true, false) => std::cmp::Ordering::Greater,
                (false, true) => std::cmp::Ordering::Less,
            });
            demos
        }

        async fn dispatch(name: &str, viewer: &mut NexusViewer, pipeline: &mut NexusPipeline) {
            match name {
                // `run` may return `()` (legacy demos) or a `Result` (demos
                // migrated to the `NexusState` API); discard whatever it yields
                // so every arm has the same `()` type.
                $( $name => { let _ = $module::run(viewer, pipeline).await; }, )*
                _ => eprintln!("Unknown demo: '{name}'"),
            }
        }
    };
}

demos! {
    "Balls" => Rbd : rbd_balls2,
    "Boxes" => Rbd : rbd_boxes2,
    "Boxes & balls" => Rbd : rbd_boxes_and_balls2,
    "Compound" => Rbd : rbd_compound2,
    "Dynamic insertion" => Rbd : rbd_dynamic2,
    "Pyramid" => Rbd : rbd_pyramid2,
    "Primitives" => Rbd : rbd_primitives2,
    "Polyline" => Rbd : rbd_polyline2,
    "Joints (spherical)" => Rbd : rbd_joint_ball2,
    "Joints (prismatic)" => Rbd : rbd_joint_prismatic2,
    "Joints (fixed)" => Rbd : rbd_joint_fixed2,
    // MPM demos.
    "Cantilever beam" => Mpm : mpm_centilever_beam2,
    "Sand" => Mpm : mpm_sand2,
    "Sand emitter" => Mpm : mpm_emitter2,
    "Elasticity" => Mpm : mpm_elasticity2,
    "Elastic cut" => Mpm : mpm_elastic_cut2,
    "Dam break" => Mpm : mpm_dam_break2,
    "Cohesion sweep" => Mpm : mpm_cohesion_sweep2,
    "Snowballs" => Mpm : mpm_snowball2,
    "Hourglass" => Mpm : mpm_hourglass2,
}

struct CliOptions {
    example: Option<String>,
    list: bool,
    cpu: bool,
    metal: bool,
    run: bool,
}

fn parse_command_line() -> CliOptions {
    let mut args = std::env::args();
    let mut opts = CliOptions {
        example: None,
        list: false,
        cpu: false,
        metal: false,
        run: false,
    };

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--example" => opts.example = args.next(),
            "--list" => opts.list = true,
            "--cpu" => opts.cpu = true,
            "--metal" => opts.metal = true,
            "--run" => opts.run = true,
            _ => {}
        }
    }

    opts
}

#[kiss3d::main]
pub async fn main() {
    let opts = parse_command_line();
    let demos = demo_list();

    if opts.list {
        for (name, _) in &demos {
            println!("{}", name.to_camel_case());
        }
        return;
    }

    let mut selected = 0;
    if let Some(ref demo) = opts.example {
        match demos
            .iter()
            .position(|(name, _)| name.to_camel_case().as_str() == demo.as_str())
        {
            Some(i) => selected = i,
            None => {
                eprintln!("Invalid example to run provided: '{demo}'");
                return;
            }
        }
    }

    let mut viewer = NexusViewer::new(demos.clone()).await;
    viewer = viewer.with_selected_demo(selected);
    if opts.cpu {
        viewer = viewer.with_cpu();
    }
    #[cfg(feature = "metal")]
    if opts.metal {
        viewer = viewer.with_backend(nexus_viewer2d::BackendType::Metal);
    }
    if opts.run {
        viewer = viewer.with_running();
    }

    // The GPU pipelines are owned here (not by `NexusState`) so they can be
    // compiled once up-front and reused across demos. A backend switch drops and
    // recompiles them.
    let mut pipeline = NexusPipeline::default();
    let mut compiled_backend: Option<BackendType> = None;

    loop {
        // Initialize the currently-selected backend (it may have just changed
        // via the UI backend selector). Idempotent for already-created backends.
        viewer.init_backend();

        // Compile all pipelines up-front the first time each backend is used,
        // showing the "Compiling shaders…" banner while it blocks — so the
        // freeze is explained and never happens mid-demo. A backend switch
        // drops the stale (other-device) pipelines and recompiles.
        let backend_type = viewer.backend_type();
        if compiled_backend != Some(backend_type) {
            pipeline = NexusPipeline::default();
            if backend_type != BackendType::Cpu {
                viewer.show_compile_banner().await;
            }
            if let Err(err) = pipeline.preload_pipelines(viewer.backend(), NexusPipelineMask::all())
            {
                eprintln!("Failed to preload GPU pipelines: {err:?}");
            }
            compiled_backend = Some(backend_type);
        }

        let sel = viewer.selected_demo();
        dispatch(&demos[sel].0, &mut viewer, &mut pipeline).await;
        if viewer.quitting() {
            break;
        }
        viewer.clear_scene();
        viewer.clear_transition();
    }
}
