//! **`$WOW_SSR_PIP=1` — the reflection march's raw output as a picture-in-picture inset.**
//!
//! `$WOW_SSR_SHOW=2` paints the march's raw colour in place of the water, which answers "what did
//! the march return" but hides the finished picture it is supposed to explain. This keeps the frame
//! as it is and puts the raw march beside it: the water shader stores `ssr_trace`'s colour for
//! every water fragment into a half-resolution storage image, and a pass after the main pass draws
//! that image into an inset at the left edge of the frame.
//!
//! Three pieces, in graph order:
//! * a **clear** between the prepass and the water, so pixels the water no longer covers do not
//!   keep last frame's answer (alpha zero means "not water" to the inset shader);
//! * the **water's own store**, in `liquid.wgsl` at binding 116;
//! * the **inset**, after the main pass and BEFORE tonemapping, because the stored colour is the
//!   scene snapshot's linear HDR and the inset has to go through the same tonemapper as the frame
//!   to be comparable with it.
//!
//! Off, the image is 1x1 and the shader's store is bounds-gated to nothing but that texel, so the
//! debug lever costs nothing it is not asked for.

use bevy::asset::RenderAssetUsages;
use bevy::core_pipeline::core_3d::graph::{Core3d, Node3d};
use bevy::core_pipeline::prepass::ViewPrepassTextures;
use bevy::core_pipeline::FullscreenShader;
use bevy::prelude::*;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_graph::{
    Node, NodeRunError, RenderGraphContext, RenderGraphExt, RenderLabel,
};
use bevy::render::render_resource::binding_types::texture_2d;
use bevy::render::render_resource::{
    BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries, CachedRenderPipelineId,
    ColorTargetState, ColorWrites, Extent3d, FragmentState, LoadOp, Operations, PipelineCache,
    RenderPassColorAttachment, RenderPassDescriptor, RenderPipelineDescriptor, ShaderStages,
    StoreOp, TextureDimension, TextureFormat, TextureSampleType, TextureUsages,
};
use bevy::render::renderer::RenderContext;
use bevy::render::texture::GpuImage;
use bevy::render::view::ViewTarget;

use super::WaterStyle;

/// The inset's image and whether it is being filled.
#[derive(Resource, Clone, ExtractResource)]
pub(crate) struct WaterSsrPip {
    /// The storage image the liquid materials write into.
    pub(crate) image: Handle<Image>,
    size: UVec2,
    armed: bool,
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct SsrPipClearLabel;

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct SsrPipDrawLabel;

fn pip_requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("WOW_SSR_PIP").is_some_and(|v| v != "0"))
}

/// `$WOW_SSR_PIP=2` shows the **depth the march traverses** in the inset instead of its output —
/// the finest level of `liquid::hiz`'s pyramid. Anything missing from it is invisible to the march:
/// the ray walks straight through it and reads its colour from wherever it stops behind it.
fn pip_shows_depth() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("WOW_SSR_PIP").as_deref() == Ok("2"))
}

/// `Rgba16Float`: the stored colour is the scene snapshot's linear HDR, and an 8-bit format would
/// clip exactly the bright sky the reflection's artefacts have been made of.
fn pip_image(size: UVec2) -> Image {
    let mut image = Image::new_fill(
        Extent3d {
            width: size.x.max(1),
            height: size.y.max(1),
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0u8; 8],
        TextureFormat::Rgba16Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.usage = TextureUsages::STORAGE_BINDING
        | TextureUsages::TEXTURE_BINDING
        | TextureUsages::RENDER_ATTACHMENT;
    image.data = None;
    image
}

/// Half the window on each axis while armed, 1x1 while not. The water stores by screen UV, so the
/// render scale does not change what lands where — only how many fragments share a texel.
fn drive_pip(
    style: Res<WaterStyle>,
    mut pip: ResMut<WaterSsrPip>,
    mut images: ResMut<Assets<Image>>,
    cameras: Query<&Camera, With<crate::view::WorldCamera>>,
) {
    pip.armed = pip_requested() && style.is_stylised();
    let want = if pip.armed {
        let Ok(camera) = cameras.single() else {
            return;
        };
        let Some(px) = camera.physical_target_size() else {
            return;
        };
        (px / 2).max(UVec2::ONE)
    } else {
        UVec2::ONE
    };
    if want != pip.size {
        pip.image = images.add(pip_image(want));
        pip.size = want;
    }
}

/// Re-point every liquid material at the image after a rebuild; a no-op on every other frame.
fn restamp_pip(
    pip: Res<WaterSsrPip>,
    mut materials: ResMut<Assets<benilla_assets::materials::LiquidMaterial>>,
    mut stamped: Local<Option<AssetId<Image>>>,
) {
    if *stamped == Some(pip.image.id()) {
        return;
    }
    *stamped = Some(pip.image.id());
    for (_, material) in materials.iter_mut() {
        material.extension.ssr_pip = pip.image.clone();
    }
}

/// One inset pipeline per view format: the main texture is `Rgba16Float` on an HDR camera and
/// Bevy's default otherwise, and a colour target has to match the attachment exactly.
#[derive(Resource)]
struct SsrPipPipelines {
    layout: BindGroupLayoutDescriptor,
    hdr: CachedRenderPipelineId,
    sdr: CachedRenderPipelineId,
}

fn init_pip_pipelines(
    mut commands: Commands,
    fullscreen_shader: Res<FullscreenShader>,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
) {
    let shader: Handle<Shader> = asset_server.load("embedded://benilla_world/shaders/ssr_pip.wgsl");
    let layout = BindGroupLayoutDescriptor::new(
        "ssr_pip_layout",
        &BindGroupLayoutEntries::with_indices(
            ShaderStages::FRAGMENT,
            (
                (
                    0,
                    texture_2d(TextureSampleType::Float { filterable: false }),
                ),
                (
                    1,
                    texture_2d(TextureSampleType::Float { filterable: false }),
                ),
            ),
        ),
    );
    let defs = if pip_shows_depth() {
        vec!["PIP_DEPTH".into()]
    } else {
        vec![]
    };
    let desc = |format: TextureFormat| RenderPipelineDescriptor {
        label: Some("ssr_pip".into()),
        layout: vec![layout.clone()],
        vertex: fullscreen_shader.to_vertex_state(),
        fragment: Some(FragmentState {
            shader: shader.clone(),
            shader_defs: defs.clone(),
            entry_point: Some("fs_pip".into()),
            targets: vec![Some(ColorTargetState {
                format,
                blend: None,
                write_mask: ColorWrites::ALL,
            })],
        }),
        ..default()
    };
    let hdr = pipeline_cache.queue_render_pipeline(desc(ViewTarget::TEXTURE_FORMAT_HDR));
    let sdr = pipeline_cache.queue_render_pipeline(desc(TextureFormat::bevy_default()));
    commands.insert_resource(SsrPipPipelines { layout, hdr, sdr });
}

/// The image, if the lever is on AND this is the view the water marches in.
///
/// **The view gate is not optional.** Both nodes are in `Core3d`, which runs for every 3D view —
/// the mirror camera and the probe's cube faces included — and without it the inset is drawn into
/// the mirror's capture and turns up reflected in the lake. The depth prepass is the discriminator
/// `liquid::hiz` already uses: only the world camera carries one, and it is what the march reads.
fn armed_gpu_image(world: &World, view: Entity) -> Option<&GpuImage> {
    let pip = world.get_resource::<WaterSsrPip>()?;
    if !pip.armed {
        return None;
    }
    world.get::<ViewPrepassTextures>(view)?.depth.as_ref()?;
    world
        .get_resource::<RenderAssets<GpuImage>>()?
        .get(&pip.image)
}

#[derive(Default)]
struct SsrPipClearNode;

impl Node for SsrPipClearNode {
    fn run(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let Some(gpu) = armed_gpu_image(world, graph.view_entity()) else {
            return Ok(());
        };
        // An empty pass whose only job is its load op — `clear_texture` would need wgpu's
        // `CLEAR_TEXTURE` feature, and this needs nothing.
        let _pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("ssr_pip_clear"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &gpu.texture_view,
                depth_slice: None,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(Default::default()),
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        Ok(())
    }
}

#[derive(Default)]
struct SsrPipDrawNode;

impl Node for SsrPipDrawNode {
    fn run(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let Some(gpu) = armed_gpu_image(world, graph.view_entity()) else {
            return Ok(());
        };
        let Some(target) = world.get::<ViewTarget>(graph.view_entity()) else {
            return Ok(());
        };
        let (Some(pipes), Some(cache)) = (
            world.get_resource::<SsrPipPipelines>(),
            world.get_resource::<PipelineCache>(),
        ) else {
            return Ok(());
        };
        let id = if target.main_texture_format() == ViewTarget::TEXTURE_FORMAT_HDR {
            pipes.hdr
        } else {
            pipes.sdr
        };
        let Some(pipeline) = cache.get_render_pipeline(id) else {
            return Ok(());
        };
        let Some(hiz) = world.get_resource::<super::hiz::WaterHiz>().and_then(|h| {
            world
                .get_resource::<RenderAssets<GpuImage>>()?
                .get(&h.image)
        }) else {
            return Ok(());
        };
        let bind = render_context.render_device().create_bind_group(
            "ssr_pip_bind",
            &cache.get_bind_group_layout(&pipes.layout),
            &BindGroupEntries::with_indices(((0, &gpu.texture_view), (1, &hiz.texture_view))),
        );
        // The inset: a third of the frame's width at the view's own aspect, at the left edge and
        // vertically centred — the one stretch of the 1.12 layout that no frame covers by default.
        let size = target.main_texture().size();
        let (w, h) = (size.width as f32, size.height as f32);
        let iw = (w / 3.0).floor();
        let ih = (iw * h / w).floor();
        let (x, y) = ((w * 0.01).floor(), ((h - ih) * 0.5).floor());
        let mut pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("ssr_pip"),
            color_attachments: &[Some(target.get_unsampled_color_attachment())],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_viewport(x, y, iw, ih, 0.0, 1.0);
        pass.set_render_pipeline(pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.draw(0..3, 0..1);
        Ok(())
    }
}

pub(super) fn register(app: &mut App) {
    let size = UVec2::ONE;
    let image = app
        .world_mut()
        .resource_mut::<Assets<Image>>()
        .add(pip_image(size));
    app.insert_resource(WaterSsrPip {
        image,
        size,
        armed: false,
    })
    .add_plugins(ExtractResourcePlugin::<WaterSsrPip>::default())
    .add_systems(
        PostUpdate,
        (drive_pip, restamp_pip)
            .chain()
            .before(bevy::transform::TransformSystems::Propagate)
            .before(bevy::camera::CameraUpdateSystems)
            .after(super::probe::drive_probe),
    );

    let Some(render_app) = app.get_sub_app_mut(bevy::render::RenderApp) else {
        return;
    };

    render_app
        .add_systems(bevy::render::RenderStartup, init_pip_pipelines)
        .add_render_graph_node::<SsrPipClearNode>(Core3d, SsrPipClearLabel)
        .add_render_graph_edges(Core3d, (Node3d::EndPrepasses, SsrPipClearLabel))
        .add_render_graph_edges(Core3d, (SsrPipClearLabel, Node3d::MainTransparentPass))
        .add_render_graph_node::<SsrPipDrawNode>(Core3d, SsrPipDrawLabel)
        // **Before post-processing starts, not merely before tonemapping.** The world emits gamma
        // bytes and `ffx_glow`'s combine — which hangs off `StartMainPassPostProcessing` — takes
        // the frame's one decode. Pinned only between `EndMainPass` and `Tonemapping`, the inset
        // landed on either side of the glow depending on the run: decoded with the frame in one
        // (its 0.12 grey read 33) and skipping the decode in the next (97, the whole inset washed
        // out against the frame it is compared with).
        .add_render_graph_edges(Core3d, (Node3d::EndMainPass, SsrPipDrawLabel))
        .add_render_graph_edges(
            Core3d,
            (SsrPipDrawLabel, Node3d::StartMainPassPostProcessing),
        );
}
