use std::time::Duration;
// use crate::rbd::BackendType;
use crate::viewer::UiState;
use crate::{DemoKind, RunState, Transition};
use kiss3d::egui;
use nexus::rbd::pipeline::RunStats;
use nexus::state::NexusCounts;

use crate::backend::BackendType;
use egui::{Button, CollapsingHeader, Color32, CornerRadius, RichText, Stroke};

/// Sets up a custom warm theme that complements the app's off-white background.
pub fn setup_custom_theme(ctx: &egui::Context) {
    let bg_fill = Color32::from_rgb(250, 250, 245);
    let window_fill = Color32::from_rgb(252, 252, 248);
    let faint_bg = Color32::from_rgb(240, 240, 232);
    let extreme_bg = Color32::from_rgb(255, 255, 252);

    let text_color = Color32::from_rgb(60, 58, 52);

    let accent = Color32::from_rgb(82, 130, 150);
    let accent_active = Color32::from_rgb(70, 115, 135);

    let widget_bg = Color32::from_rgb(235, 235, 225);
    let widget_bg_hover = Color32::from_rgb(225, 225, 215);
    let widget_bg_active = Color32::from_rgb(215, 215, 205);

    let stroke_color = Color32::from_rgb(200, 198, 190);
    let stroke_hover = Color32::from_rgb(180, 178, 170);

    let rounding = CornerRadius::same(6);
    let small_rounding = CornerRadius::same(4);

    ctx.global_style_mut(|style| {
        let v = &mut style.visuals;
        v.dark_mode = false;

        v.widgets.noninteractive.bg_fill = faint_bg;
        v.widgets.noninteractive.weak_bg_fill = faint_bg;
        v.widgets.noninteractive.bg_stroke = Stroke::new(1.0, stroke_color);
        v.widgets.noninteractive.corner_radius = rounding;
        v.widgets.noninteractive.fg_stroke = Stroke::new(1.0, text_color);

        v.widgets.inactive.bg_fill = widget_bg;
        v.widgets.inactive.weak_bg_fill = widget_bg;
        v.widgets.inactive.bg_stroke = Stroke::new(1.0, stroke_color);
        v.widgets.inactive.corner_radius = small_rounding;
        v.widgets.inactive.fg_stroke = Stroke::new(1.0, text_color);

        v.widgets.hovered.bg_fill = widget_bg_hover;
        v.widgets.hovered.weak_bg_fill = widget_bg_hover;
        v.widgets.hovered.bg_stroke = Stroke::new(1.0, stroke_hover);
        v.widgets.hovered.corner_radius = small_rounding;
        v.widgets.hovered.fg_stroke = Stroke::new(1.5, text_color);

        v.widgets.active.bg_fill = widget_bg_active;
        v.widgets.active.weak_bg_fill = widget_bg_active;
        v.widgets.active.bg_stroke = Stroke::new(1.0, accent);
        v.widgets.active.corner_radius = small_rounding;
        v.widgets.active.fg_stroke = Stroke::new(2.0, accent_active);

        v.widgets.open.bg_fill = widget_bg;
        v.widgets.open.weak_bg_fill = widget_bg;
        v.widgets.open.bg_stroke = Stroke::new(1.0, stroke_color);
        v.widgets.open.corner_radius = small_rounding;
        v.widgets.open.fg_stroke = Stroke::new(1.0, text_color);

        v.selection.bg_fill = accent.gamma_multiply(0.25);
        v.selection.stroke = Stroke::new(1.0, accent);

        v.hyperlink_color = accent;
        v.faint_bg_color = faint_bg;
        v.extreme_bg_color = extreme_bg;
        v.code_bg_color = Color32::from_rgb(230, 230, 220);
        v.warn_fg_color = Color32::from_rgb(180, 120, 60);
        v.error_fg_color = Color32::from_rgb(180, 70, 70);

        v.window_corner_radius = CornerRadius::same(8);
        v.window_fill = window_fill;
        v.window_stroke = Stroke::new(1.0, stroke_color);

        v.panel_fill = bg_fill;

        v.slider_trailing_fill = true;
        v.handle_shape = egui::style::HandleShape::Circle;

        style.spacing.item_spacing = egui::vec2(6.0, 3.0);
        style.spacing.window_margin = egui::Margin::same(10);
        style.spacing.button_padding = egui::vec2(6.0, 3.0);
        style.spacing.slider_width = 130.0;
        style.spacing.indent = 14.0;
        style.spacing.interact_size = egui::vec2(32.0, 18.0);
        style.spacing.combo_width = 100.0;
    });
}

/// Draws a centered "compiling shaders" banner into an existing egui context.
///
/// Shown as an overlay on the first GPU frame of a demo so it stays on screen
/// while the (blocking) shader compilation freezes the app for a few seconds —
/// otherwise the window looks frozen with no explanation. Uses a distinct
/// window id from [`main_panel`] to avoid an egui id collision.
pub fn compiling_overlay(ctx: &egui::Context) {
    egui::Window::new("⏳ Compiling shaders")
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .collapsible(false)
        .resizable(false)
        .show(ctx, |ui| {
            ui.colored_label(
                Color32::from_rgb(82, 130, 150),
                "Compiling shaders…\nThe app will freeze for a few seconds.\n\nIf nothing happens after a minute or two, check the dev console for an error.",
            );
        });
}

/// Builds the viewer control panel. Mutates `state` in place (run state, demo
/// selection, backend choice) and queries the scene for scene-specific widgets.
pub fn main_panel(ctx: &egui::Context, state: &mut UiState, gpu_available: bool) {
    egui::Window::new("Nexus Viewer")
        .default_width(300.0)
        .show(ctx, |ui| {
            // GPU error banner.
            if let Some(error_msg) = &state.gpu_init_error {
                ui.colored_label(
                    Color32::from_rgb(180, 70, 70),
                    format!("GPU: {}", error_msg),
                );
                ui.separator();
            }

            // Section toggles.
            ui.horizontal(|ui| {
                ui.toggle_value(&mut state.ui_sections.show_performance, "Performance");
                ui.toggle_value(&mut state.ui_sections.show_settings, "Settings");
                ui.toggle_value(&mut state.ui_sections.show_examples, "Examples");
            });

            egui::ScrollArea::vertical()
                .max_height(500.0)
                .show(ui, |ui| {
                    if state.ui_sections.show_settings {
                        ui.separator();
                        backend_selector(ui, state, gpu_available);
                        ui.add_space(4.0);
                        simulation_settings(ui, state);
                    }

                    if state.ui_sections.show_performance {
                        ui.separator();
                        performance_ui(ui, &state.counts, &state.run_stats, state.sync_time);
                    }

                    if state.ui_sections.show_examples && !state.demos.is_empty() {
                        ui.separator();
                        examples_section(ui, state);
                    }
                });

            ui.separator();

            // Bottom controls.
            ui.horizontal(|ui| {
                let (play_label, play_hover) = if state.run_state == RunState::Running {
                    ("Pause", "Pause simulation (T)")
                } else {
                    ("Play", "Start simulation (T)")
                };

                if ui.button(play_label).on_hover_text(play_hover).clicked() {
                    state.run_state = if state.run_state == RunState::Running {
                        RunState::Paused
                    } else {
                        RunState::Running
                    };
                }

                if ui.button("Step").on_hover_text("Single step (S)").clicked() {
                    state.run_state = RunState::Step;
                }

                if ui
                    .button("Restart")
                    .on_hover_text("Restart example (R)")
                    .clicked()
                {
                    state.transition = Some(Transition::Switch);
                }
            });
        });
}

/// Per-scene simulation settings. The viewer seeds these from the scene on load
/// and pushes edits back into the running `NexusState` each frame (see
/// [`crate::NexusViewer::sync`]). Only the groups relevant to the current scene
/// are shown.
fn simulation_settings(ui: &mut egui::Ui, state: &mut UiState) {
    let has_rbd = state.has_rbd;
    if !has_rbd {
        return;
    }

    ui.label(RichText::new("Simulation").strong());
    ui.add_space(2.0);
    let s = &mut state.sim_settings;

    ui.label("Rigid bodies");
    ui.add(egui::Slider::new(&mut s.rbd_steps_per_frame, 1..=20).text("steps / frame"));
}

fn performance_ui(
    ui: &mut egui::Ui,
    counts: &NexusCounts,
    run_stats: &RunStats,
    sync_time: Duration,
) {
    // Scene entity counts.
    ui.label(RichText::new("Scene").strong());
    ui.add_space(2.0);
    egui::Grid::new("scene_counts")
        .num_columns(2)
        .spacing([20.0, 2.0])
        .show(ui, |ui| {
            let mut row = |label: &str, value: usize| {
                ui.label(label);
                ui.label(format!("{value}"));
                ui.end_row();
            };

            if counts.rigid_bodies > 0 {
                row("Environments:", counts.num_environments);
                row("Rigid bodies:", counts.rigid_bodies);
                row("Colliders:", counts.colliders);
                row("Collision pairs:", counts.collision_pairs);
                row("Collision capacity:", counts.collision_pairs_capacity);
                row("Impulse joints:", counts.impulse_joints);
                row("Multibodies:", counts.multibodies);
                row("Multibody DOFs:", counts.multibody_dofs);
            }
        });

    ui.add_space(8.0);
    ui.separator();
    ui.add_space(4.0);

    // Timing.
    let encoding_time = run_stats.encoding_time_ms();
    let sync_time = sync_time.as_secs_f32() * 1000.0;
    ui.label(
        RichText::new(format!(
            "Encoding: {:.2}ms\nGPU: {:.2}ms\nSync: {:.2}ms",
            encoding_time, run_stats.gpu_total_time_ms, sync_time
        ))
        .strong(),
    );
    if !run_stats.gpu_pass_times.is_empty() {
        CollapsingHeader::new(format!("GPU passes: {:.2}ms", run_stats.gpu_total_time_ms))
            .id_salt("rbd_gpu_passes")
            .default_open(false)
            .show(ui, |ui| {
                egui::Grid::new("rbd_timestamp_grid")
                    .num_columns(2)
                    .spacing([20.0, 2.0])
                    .show(ui, |ui| {
                        for (label, ms) in &run_stats.gpu_pass_times {
                            ui.label(format!("{}:", label));
                            ui.label(format!("{:.2}ms", ms));
                            ui.end_row();
                        }
                    });
            });
    }

    // Slow performance warning.
    if run_stats.gpu_total_time_ms > 100.0 {
        ui.add_space(4.0);
        ui.colored_label(
            Color32::from_rgb(180, 120, 60),
            #[cfg(not(target_arch = "wasm32"))]
            "Running slow? If you have both an integrated and discrete GPU, ensure the discrete GPU is in use.",
            #[cfg(target_arch = "wasm32")]
            "Running slow? If you have both an integrated and discrete GPU, ensure your browser runs exclusively on the discrete GPU.",
        );
    }
}

impl UiState {
    /// Order in which demos appear in the picker: grouped by kind, preserving
    /// each group's listing order. Prev/Next walks this sequence so it matches
    /// the visible list rather than the raw (lexicographically-sorted) `demos`
    /// index order.
    fn demo_display_order(&self) -> Vec<usize> {
        let mut order = Vec::with_capacity(self.demos.len());
        for kind in [DemoKind::Rbd] {
            for (i, (_, k)) in self.demos.iter().enumerate() {
                if *k == kind {
                    order.push(i);
                }
            }
        }
        order
    }
}

fn examples_section(ui: &mut egui::Ui, state: &mut UiState) {
    // Previous/Next navigation + current demo name. Navigation follows the
    // grouped listing order (see `demo_display_order`), not the raw index.
    let order = state.demo_display_order();
    let pos = order
        .iter()
        .position(|&i| i == state.selected_demo)
        .unwrap_or(0);

    ui.horizontal(|ui| {
        if ui
            .add_enabled(pos > 0, Button::new("<"))
            .on_hover_text("Previous example")
            .clicked()
        {
            state.selected_demo = order[pos - 1];
            state.transition = Some(Transition::Switch);
        }

        if ui
            .add_enabled(pos + 1 < order.len(), Button::new(">"))
            .on_hover_text("Next example")
            .clicked()
        {
            state.selected_demo = order[pos + 1];
            state.transition = Some(Transition::Switch);
        }

        ui.label(
            RichText::new(state.demos[state.selected_demo].0.clone())
                .strong()
                .italics(),
        );
    });

    ui.add_space(4.0);
    ui.separator();
    ui.add_space(4.0);

    demo_group(ui, state, DemoKind::Rbd, "Rigid Bodies");
}

fn demo_group(ui: &mut egui::Ui, state: &mut UiState, kind: DemoKind, label: &str) {
    // Collect owned (index, name) so the closure can freely mutate `state`.
    let demos: Vec<(usize, String)> = state
        .demos
        .iter()
        .enumerate()
        .filter(|(_, (_, k))| *k == kind)
        .map(|(i, (name, _))| (i, name.clone()))
        .collect();

    if demos.is_empty() {
        return;
    }

    CollapsingHeader::new(format!("{} ({})", label, demos.len()))
        .default_open(true)
        .show(ui, |ui| {
            for (idx, name) in &demos {
                let is_selected = state.selected_demo == *idx;
                let text = if is_selected {
                    RichText::new(name).strong()
                } else {
                    RichText::new(name)
                };
                if ui
                    .selectable_label(is_selected, text)
                    .on_hover_text("Click to run this example")
                    .clicked()
                    && !is_selected
                {
                    state.selected_demo = *idx;
                    state.transition = Some(Transition::Switch);
                }
            }
        });
}

/// Unified backend selector.
fn backend_selector(ui: &mut egui::Ui, state: &mut UiState, gpu_available: bool) {
    ui.label(RichText::new("Physics Backend").strong());
    ui.add_space(2.0);

    let mut new_backend: Option<BackendType> = None;

    if gpu_available
        && ui
            .radio(state.backend_type == BackendType::Gpu, "GPU (nexus)")
            .on_hover_text("GPU-accelerated physics with nexus")
            .clicked()
        && state.backend_type != BackendType::Gpu
    {
        new_backend = Some(BackendType::Gpu);
    }

    #[cfg(feature = "cuda")]
    if ui
        .radio(state.backend_type == BackendType::Cuda, "CUDA (nexus)")
        .on_hover_text("GPU-accelerated physics with nexus via CUDA")
        .clicked()
        && state.backend_type != BackendType::Cuda
    {
        new_backend = Some(BackendType::Cuda);
    }

    #[cfg(feature = "metal")]
    if ui
        .radio(state.backend_type == BackendType::Metal, "Metal (nexus)")
        .on_hover_text("GPU-accelerated physics with nexus via native Metal")
        .clicked()
        && state.backend_type != BackendType::Metal
    {
        new_backend = Some(BackendType::Metal);
    }

    #[cfg(feature = "cpu")]
    if ui
        .radio(state.backend_type == BackendType::Cpu, "CPU (nexus)")
        .on_hover_text("CPU physics using the nexus GPU pipeline executed on CPU")
        .clicked()
        && state.backend_type != BackendType::Cpu
    {
        new_backend = Some(BackendType::Cpu);
    }

    if let Some(bt) = new_backend {
        state.backend_type = bt;
        state.transition = Some(Transition::Switch);
    }
}

// // ===========================================================================
// // Per-scene UI, implemented through the `Scene` trait.
// // ===========================================================================
//
// impl Scene for crate::rbd::RbdScene {
//     fn is_rbd(&self) -> bool {
//         true
//     }
//
//     fn performance_ui(&mut self, ui: &mut egui::Ui, run_stats: &RunStats, backend_type: BackendType) {
//         let physics = &self.physics;
//
//         // Scene info.
//         ui.label(RichText::new("Scene").strong());
//         ui.add_space(2.0);
//
//         egui::Grid::new("rbd_scene_grid")
//             .num_columns(2)
//             .spacing([20.0, 2.0])
//             .show(ui, |ui| {
//                 ui.label("Bodies:");
//                 ui.label(format!("{}", physics.backend.num_bodies()));
//                 ui.end_row();
//
//                 ui.label("Joints:");
//                 ui.label(format!("{}", physics.backend.num_joints()));
//                 ui.end_row();
//
//                 ui.label("Batches:");
//                 ui.label(format!("{}", physics.backend.num_batches()));
//                 ui.end_row();
//             });
//
//         ui.add_space(8.0);
//         ui.separator();
//         ui.add_space(4.0);
//
//         // Timing.
//         let total_ms_with_readback = run_stats.total_simulation_time_with_readback_ms();
//         let total_ms_without_readback = run_stats.total_simulation_time_without_readback_ms();
//         let total_readback_time = total_ms_with_readback - total_ms_without_readback;
//         let fps = if total_ms_with_readback > 0.0 {
//             (1000.0f32 / total_ms_with_readback).round()
//         } else {
//             0.0
//         };
//
//         ui.label(
//             RichText::new(format!(
//                 "Total: {:.2}ms (+ readback: {:.2}ms) - {:.0} FPS",
//                 total_ms_without_readback, total_readback_time, fps
//             ))
//             .strong(),
//         );
//         ui.add_space(4.0);
//
//         if !matches!(backend_type, BackendType::Rapier) {
//             CollapsingHeader::new("Simulation details")
//                 .id_salt("rbd_sim_details")
//                 .default_open(false)
//                 .show(ui, |ui| {
//                     ui.label(format!("Colors: {}", run_stats.num_colors));
//                     ui.label(format!(
//                         "Coloring: {:.2}ms",
//                         run_stats.coloring_time.as_secs_f32() * 1000.0
//                     ));
//                     ui.label(format!(
//                         "Coloring iterations: {} x 10",
//                         run_stats.coloring_iterations
//                     ));
//                     ui.label(format!(
//                         "Start to pairs count: {:.2}ms",
//                         run_stats.start_to_pairs_count_time.as_secs_f32() * 1000.0
//                     ));
//                     ui.label(format!(
//                         "Coloring fallback: {:.2}ms",
//                         run_stats.coloring_fallback_time.as_secs_f32() * 1000.0
//                     ));
//                 });
//
//             if !run_stats.gpu_pass_times.is_empty() {
//                 CollapsingHeader::new(format!("GPU passes: {:.2}ms", run_stats.gpu_total_time))
//                     .id_salt("rbd_gpu_passes")
//                     .default_open(false)
//                     .show(ui, |ui| {
//                         egui::Grid::new("rbd_timestamp_grid")
//                             .num_columns(2)
//                             .spacing([20.0, 2.0])
//                             .show(ui, |ui| {
//                                 for (label, ms) in &run_stats.gpu_pass_times {
//                                     ui.label(format!("{}:", label));
//                                     ui.label(format!("{:.2}ms", ms));
//                                     ui.end_row();
//                                 }
//                             });
//                     });
//             }
//
//             // Slow performance warning.
//             if run_stats.total_simulation_time_with_readback.as_secs_f32() > 0.1 {
//                 ui.add_space(4.0);
//                 ui.colored_label(
//                     Color32::from_rgb(180, 120, 60),
//                     #[cfg(not(target_arch = "wasm32"))]
//                     "Running slow? If you have both an integrated and discrete GPU, ensure the discrete GPU is in use.",
//                     #[cfg(target_arch = "wasm32")]
//                     "Running slow? If you have both an integrated and discrete GPU, ensure your browser runs exclusively on the discrete GPU.",
//                 );
//             }
//         }
//     }
// }
