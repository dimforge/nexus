use inflector::Inflector;
use nexus_viewer3d::{BackendType, DemoKind, NexusViewer};
use nexus3d::prelude::{NexusPipeline, NexusPipelineMask};

mod balls3;
mod boxes3;
mod boxes_and_balls3;
mod compound3;
mod dynamic_rbd3;
mod joint_ball3;
mod joint_fixed3;
mod joint_prismatic3;
mod joint_revolute3;
mod joint_revolute_batch3;
mod joints3;
mod keva3;
mod many_pyramids3;
mod many_pyramids_batch3;
// The robot loaders read URDF/MJCF assets from the filesystem, so they are
// native-only.
#[cfg(not(target_arch = "wasm32"))]
mod mujoco_menagerie3;
mod multibody_pendulum3;
mod primitives3;
mod pyramid3;
mod trimesh3;
#[cfg(not(target_arch = "wasm32"))]
mod urdf3;

/// Declares the demo registry: a `(name, kind)` list for the picker UI and a
/// name -> `run()` dispatcher. Keeping both in one macro keeps them in sync.
/// Entries can carry attributes (e.g. `#[cfg(...)]`) to exclude a demo from
/// some targets.
macro_rules! demos {
    ( $( $(#[$attr:meta])* $name:literal => $kind:ident : $module:ident ),* $(,)? ) => {
        // Built with `push` (not `vec![]`) so per-entry `#[cfg]` attributes
        // can exclude entries on some targets.
        #[allow(clippy::vec_init_then_push)]
        fn demo_list() -> Vec<(String, DemoKind)> {
            let mut demos: Vec<(String, DemoKind)> = Vec::new();
            $(
                $(#[$attr])*
                demos.push(($name.to_string(), DemoKind::$kind));
            )*
            // Lexicographic sort, with stress tests (names starting with '(')
            // moved to the end of the list.
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
                $( $(#[$attr])* $name => { let _ = $module::run(viewer, pipeline).await; }, )*
                _ => eprintln!("Unknown demo: '{name}'"),
            }
        }
    };
}

demos! {
    "Balls" => Rbd : balls3,
    "Boxes" => Rbd : boxes3,
    "Boxes & balls" => Rbd : boxes_and_balls3,
    "Compound" => Rbd : compound3,
    "Dynamic insertion" => Rbd : dynamic_rbd3,
    "Primitives" => Rbd : primitives3,
    "Pyramid" => Rbd : pyramid3,
    "Many pyramids" => Rbd : many_pyramids3,
    "Many pyramids (batched)" => Rbd : many_pyramids_batch3,
    "Keva tower" => Rbd : keva3,
    "Joints (multibody)" => Rbd : joints3,
    "Joints (Spherical)" => Rbd : joint_ball3,
    "Joints (Fixed)" => Rbd : joint_fixed3,
    "Joints (Prismatic)" => Rbd : joint_prismatic3,
    "Joints (Revolute)" => Rbd : joint_revolute3,
    "Joints (Revolute - Batched)" => Rbd : joint_revolute_batch3,
    "Multibody (Pendulum)" => Rbd : multibody_pendulum3,
    "Trimesh" => Rbd : trimesh3,
    #[cfg(not(target_arch = "wasm32"))]
    "URDF (multibody)" => Rbd : urdf3,
    #[cfg(not(target_arch = "wasm32"))]
    "MuJoCo Menagerie" => Rbd : mujoco_menagerie3,
}

struct CliOptions {
    example: Option<String>,
    list: bool,
    cpu: bool,
    cuda: bool,
    metal: bool,
    run: bool,
}

fn parse_command_line() -> CliOptions {
    let mut args = std::env::args();
    let mut opts = CliOptions {
        example: None,
        list: false,
        cpu: false,
        cuda: false,
        metal: false,
        run: false,
    };

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--example" => opts.example = args.next(),
            "--list" => opts.list = true,
            "--cpu" => opts.cpu = true,
            "--cuda" => opts.cuda = true,
            "--metal" => opts.metal = true,
            "--run" => opts.run = true,
            _ => {}
        }
    }

    opts
}

#[kiss3d::main]
pub async fn main() {
    env_logger::init();
    let opts = parse_command_line();
    let demos = demo_list();

    if opts.list {
        for (name, _) in &demos {
            println!("{}", name.to_camel_case());
        }
        return;
    }

    // Resolve `--example NAME` to a starting demo index.
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
    #[cfg(feature = "cuda")]
    if opts.cuda {
        viewer = viewer.with_backend(nexus_viewer3d::BackendType::Cuda);
    }
    #[cfg(feature = "metal")]
    if opts.metal {
        viewer = viewer.with_backend(nexus_viewer3d::BackendType::Metal);
    }
    if opts.run {
        viewer = viewer.with_running();
    }

    // The GPU pipelines are owned here (not by `NexusState`) so they can be
    // compiled once up-front and reused across demos. A backend switch drops and
    // recompiles them.
    let mut pipeline = NexusPipeline::default();
    let mut compiled_backend: Option<BackendType> = None;

    // Each selected demo owns its own loop (`run`); it returns when the user
    // closes the window or picks another demo (via the picker, which makes
    // `viewer.render()` return false).
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
