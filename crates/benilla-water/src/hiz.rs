//! **The reflection march's hierarchical depth buffer.**
//!
//! A pyramid of nearest-per-tile depths, rebuilt from the depth prepass every frame the stylised
//! water is on. `shaders/hiz_reduce.wgsl` carries the reasoning for the pyramid and for why it
//! reduces with `max` where Bevy's own depth pyramid reduces with `min`; this module is the
//! plumbing that builds it.
//!
//! The shape is `scene_color`'s, because the problem is the same one: an image the liquid material
//! samples, sized to the view, rebuilt when the view resizes, re-pointed into every material, and
//! filled by a render-graph node. The differences are that this one carries a mip chain and is
//! rendered into rather than copied into.
//!
//! ## From the VIEW's depth, after the opaque pass — not from the prepass
//!
//! It is built **after the opaque and transmissive passes and before the transparent pass**, from
//! the view's own depth attachment: the only window where the depth of the opaque world is complete
//! and the water has not yet been drawn. It is also the window [`super::scene_color`] copies the
//! colour in, so the depth the march walks and the colour it reads describe the same frame.
//!
//! It used to be seeded from the depth PREPASS, and the prepass is not the opaque world: models on
//! the entity path opt out of it (`WowModelExt::enable_prepass` — characters, creatures, fading and
//! refused doodads), the WDL horizon opts out, and the static world is in it only through a pass of
//! its own that draws nothing until all of its pipelines have compiled. Anything missing from the
//! depth is invisible to the march while still drawn in the colour it reads, and a ray walks
//! straight through it, stops in the "sky" behind it and reads its colour there — every water row
//! in that column the same few pixels. That is the stretched palm Stefan reported from
//! Stranglethorn, reproduced exactly by dropping the static world from the prepass
//! (`$WOW_GX_PREPASS=0`). Every engine that ships this builds the pyramid from the scene depth for
//! the same reason (AMD FidelityFX SSSR's input is the opaque depth buffer), and Bevy's own
//! occlusion culling binds the view depth exactly as this does.
//!
//! The view depth is only bindable once `liquid::depth` has added `TEXTURE_BINDING` to the world
//! camera's `depth_texture_usages`, which takes a frame to reach the texture; until then the seed
//! falls back to the prepass depth.

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
use bevy::render::render_resource::binding_types::{
    texture_2d, texture_depth_2d, texture_depth_2d_multisampled,
};
use bevy::render::render_resource::{
    BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries, CachedRenderPipelineId,
    ColorTargetState, ColorWrites, Extent3d, FragmentState, Operations, PipelineCache,
    RenderPassColorAttachment, RenderPassDescriptor, RenderPipelineDescriptor, ShaderStages,
    TextureAspect, TextureDimension, TextureFormat, TextureSampleType, TextureUsages,
    TextureViewDescriptor, TextureViewDimension,
};
use bevy::render::renderer::RenderContext;
use bevy::render::texture::GpuImage;
use bevy::render::view::ViewDepthTexture;

use super::WaterStyle;

/// The pyramid, and the size it was built for.
#[derive(Resource, Clone, ExtractResource)]
pub(crate) struct WaterHiz {
    /// The NEAREST-depth pyramid — the one the traversal reads on every step.
    pub(crate) image: Handle<Image>,
    /// The FARTHEST-depth pyramid, same size — read only on the steps where the ray is behind a
    /// tile's nearest surface. See [`hiz_image`] for why it is a second image and not a channel.
    pub(crate) far: Handle<Image>,
    /// Its base size in physical pixels — a power of two on both axes.
    size: UVec2,
    /// How many levels it carries.
    mips: u32,
    /// Only the stylised looks march, so only they need this.
    armed: bool,
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct WaterHizLabel;

/// The largest power of two not greater than `v`, floored at one.
fn prev_pot(v: u32) -> u32 {
    if v <= 1 {
        1
    } else {
        1 << (31 - (v.max(1)).leading_zeros())
    }
}

/// How many levels a pyramid of this size carries, down to a single texel.
fn mip_count(size: UVec2) -> u32 {
    32 - size.x.max(size.y).max(1).leading_zeros()
}

/// One pyramid level chain, `R32Float`: full-precision float, because it holds raw reverse-Z NDC
/// depth and the traversal compares it against the ray's own — a normalised or half-precision
/// format would quantise exactly the near-plane precision reverse-Z exists to provide.
///
/// There are two of these, the NEAREST surface per tile ([`WaterHiz::image`]) and the FARTHEST
/// ([`WaterHiz::far`]). **Why the farthest at all:** the nearest alone can prove a tile EMPTY in
/// front of a ray, which is all FFX SSSR asks of it; it cannot prove the ray passed BEHIND a tile, so
/// a ray going under a tree crown — which the water sees and the camera does not — met the crown's
/// front leaves 24+ yards in front of it and was thrown away: the black fragments in reflected
/// canopies. With each depth sample given a thickness (McGuire & Mara, "Efficient GPU Screen-Space
/// Ray Tracing", JCGT 2014), a ray behind the slab of a tile's farthest surface has passed behind
/// everything in it; min-max hierarchies (Hofmann et al., "Hierarchical Multi-Layer Screen-Space
/// Ray Tracing", HPG 2017) skip such tiles whole.
///
/// **Why a second image and not a second channel.** It was first an `Rg32Float` channel, and the
/// water's pass got 0.9 ms slower at Mirror Lake (GPU journal: transparent 4.05 → 4.94 ms, the
/// pyramid build itself unchanged) — the march reads the pyramid on every step, a 64-bit texel
/// fetches at half rate on this GPU, and most steps cross empty space in front of everything and
/// never needed the farthest at all. Apart, the nearest is read as before and the farthest only on
/// the steps where the ray is behind a tile's nearest surface.
fn hiz_image(size: UVec2, mips: u32) -> Image {
    let mut image = Image::new_fill(
        Extent3d {
            width: size.x.max(1),
            height: size.y.max(1),
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &0f32.to_ne_bytes(),
        TextureFormat::R32Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.usage =
        TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT;
    image.texture_descriptor.mip_level_count = mips;
    // **The view has to span the chain.** Bevy's default view is built from this descriptor, and
    // without saying so the shader sees one level — `textureNumLevels` reports it, the traversal
    // never ascends, and every read below level zero is out of range.
    image.texture_view_descriptor = Some(TextureViewDescriptor {
        label: Some("hiz_all_mips"),
        dimension: Some(TextureViewDimension::D2),
        base_mip_level: 0,
        mip_level_count: Some(mips),
        ..default()
    });
    // Every texel is written by a render pass before anything reads it, and `new_fill` would
    // otherwise upload a base level that is immediately overwritten.
    image.data = None;
    image
}

/// Keep the pyramid the size of the view, and armed only for the looks that march.
fn drive_hiz(
    style: Res<WaterStyle>,
    mut hiz: ResMut<WaterHiz>,
    mut images: ResMut<Assets<Image>>,
    cameras: Query<&Camera, With<crate::view::WorldCamera>>,
) {
    hiz.armed = style.is_stylised();
    let Ok(camera) = cameras.single() else {
        return;
    };
    let Some(px) = camera.physical_target_size() else {
        return;
    };
    // **Previous power of two, deliberately.** Every level is then exactly half the one above, so
    // the traversal's cell arithmetic is a shift and a tile's bounds are exact. The cost is that
    // the pyramid is slightly smaller than the view, which is harmless: it is addressed by UV and
    // a conservative nearest-depth is still conservative when its tiles are a little larger.
    let want = UVec2::new(prev_pot(px.x), prev_pot(px.y)).max(UVec2::splat(8));
    if want != hiz.size {
        let mips = mip_count(want);
        hiz.image = images.add(hiz_image(want, mips));
        hiz.far = images.add(hiz_image(want, mips));
        hiz.size = want;
        hiz.mips = mips;
    }
}

/// Re-point every liquid material at the pyramid after a rebuild — change-gated on the handle, so
/// it is a no-op on every frame that is not a resize.
fn restamp_hiz(
    hiz: Res<WaterHiz>,
    mut materials: ResMut<Assets<benilla_assets::materials::LiquidMaterial>>,
    mut stamped: Local<Option<AssetId<Image>>>,
) {
    if *stamped == Some(hiz.image.id()) {
        return;
    }
    *stamped = Some(hiz.image.id());
    for (_, material) in materials.iter_mut() {
        material.extension.hiz = hiz.image.clone();
        material.extension.hiz_far = hiz.far.clone();
    }
}

/// The pipelines: seed mip 0 from a single- or multi-sampled depth, then halve repeatedly.
#[derive(Resource)]
struct HizPipelines {
    seed_layout: BindGroupLayoutDescriptor,
    seed_ms_layout: BindGroupLayoutDescriptor,
    reduce_layout: BindGroupLayoutDescriptor,
    seed: CachedRenderPipelineId,
    seed_ms: CachedRenderPipelineId,
    reduce: CachedRenderPipelineId,
}

/// Whether the pyramid seeds from the depth prepass rather than the view's depth; it does not.
fn seed_from_prepass() -> bool {
    false
}

fn init_hiz_pipelines(
    mut commands: Commands,
    fullscreen_shader: Res<FullscreenShader>,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
) {
    let shader: Handle<Shader> =
        asset_server.load("embedded://benilla_water/shaders/hiz_reduce.wgsl");
    // **Each entry point gets only the binding it uses, at its own index.** The two live in one
    // file, so both globals exist in the module; a layout built from a running count would hand
    // `fs_seed` a sampler where its shader declares a texture and wgpu rejects the pipeline. No
    // sampler at all: every read is a `textureLoad`.
    let seed_layout = BindGroupLayoutDescriptor::new(
        "hiz_seed_layout",
        &BindGroupLayoutEntries::with_indices(ShaderStages::FRAGMENT, ((0, texture_depth_2d()),)),
    );
    let seed_ms_layout = BindGroupLayoutDescriptor::new(
        "hiz_seed_ms_layout",
        &BindGroupLayoutEntries::with_indices(
            ShaderStages::FRAGMENT,
            ((2, texture_depth_2d_multisampled()),),
        ),
    );
    let reduce_layout = BindGroupLayoutDescriptor::new(
        "hiz_reduce_layout",
        &BindGroupLayoutEntries::with_indices(
            ShaderStages::FRAGMENT,
            (
                (
                    1,
                    texture_2d(TextureSampleType::Float { filterable: false }),
                ),
                (
                    3,
                    texture_2d(TextureSampleType::Float { filterable: false }),
                ),
            ),
        ),
    );
    // Two targets, one per pyramid: location 0 the nearest, location 1 the farthest.
    let one = || {
        Some(ColorTargetState {
            format: TextureFormat::R32Float,
            blend: None,
            write_mask: ColorWrites::RED,
        })
    };
    let target =
        |shader: Handle<Shader>, entry: &'static str, layout: &BindGroupLayoutDescriptor| {
            RenderPipelineDescriptor {
                label: Some("hiz_reduce".into()),
                layout: vec![layout.clone()],
                vertex: fullscreen_shader.to_vertex_state(),
                fragment: Some(FragmentState {
                    shader,
                    shader_defs: vec![],
                    entry_point: Some(entry.into()),
                    targets: vec![one(), one()],
                }),
                ..default()
            }
        };
    let seed =
        pipeline_cache.queue_render_pipeline(target(shader.clone(), "fs_seed", &seed_layout));
    let seed_ms =
        pipeline_cache.queue_render_pipeline(target(shader.clone(), "fs_seed_ms", &seed_ms_layout));
    let reduce = pipeline_cache.queue_render_pipeline(target(shader, "fs_reduce", &reduce_layout));
    commands.insert_resource(HizPipelines {
        seed_layout,
        seed_ms_layout,
        reduce_layout,
        seed,
        seed_ms,
        reduce,
    });
}

#[derive(Default)]
struct WaterHizNode;

impl Node for WaterHizNode {
    fn run(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let Some(hiz) = world.get_resource::<WaterHiz>() else {
            return Ok(());
        };
        if !hiz.armed {
            return Ok(());
        }
        let (Some(pipes), Some(cache)) = (
            world.get_resource::<HizPipelines>(),
            world.get_resource::<PipelineCache>(),
        ) else {
            return Ok(());
        };
        let (Some(seed_pipeline), Some(seed_ms_pipeline), Some(reduce_pipeline)) = (
            cache.get_render_pipeline(pipes.seed),
            cache.get_render_pipeline(pipes.seed_ms),
            cache.get_render_pipeline(pipes.reduce),
        ) else {
            return Ok(()); // still compiling
        };
        let Some(images) = world.get_resource::<RenderAssets<GpuImage>>() else {
            return Ok(());
        };
        let (Some(gpu), Some(gpu_far)) = (images.get(&hiz.image), images.get(&hiz.far)) else {
            return Ok(());
        };
        // **Only the view the water marches in.** Core3d runs for every 3D view — the mirror and
        // the probe's cube faces too — and a depth prepass is what marks the world camera: only it
        // carries one, and only while a look that marches is on.
        let Some(prepass) = world
            .get::<ViewPrepassTextures>(graph.view_entity())
            .and_then(|p| p.depth.as_ref())
        else {
            return Ok(());
        };
        // **The view's own depth, complete after the opaque pass** — see the module doc. The
        // prepass is the fallback for the frames before the attachment is bindable, and the
        // lever's answer.
        let view_depth = world
            .get::<ViewDepthTexture>(graph.view_entity())
            .filter(|d| d.texture.usage().contains(TextureUsages::TEXTURE_BINDING))
            .filter(|_| !seed_from_prepass());
        let (source_texture, source_view) = match view_depth {
            Some(d) => (
                &d.texture,
                d.texture.create_view(&TextureViewDescriptor {
                    label: Some("hiz_view_depth"),
                    aspect: TextureAspect::DepthOnly,
                    ..default()
                }),
            ),
            None => (
                &prepass.texture.texture,
                prepass.texture.default_view.clone(),
            ),
        };
        let multisampled = source_texture.sample_count() > 1;

        let view_of = |texture: &bevy::render::render_resource::Texture, level: u32| {
            texture.create_view(&TextureViewDescriptor {
                label: Some("hiz_level"),
                dimension: Some(TextureViewDimension::D2),
                base_mip_level: level,
                mip_level_count: Some(1),
                ..default()
            })
        };
        let level_view = |level: u32| {
            (
                view_of(&gpu.texture, level),
                view_of(&gpu_far.texture, level),
            )
        };
        let pass = |ctx: &mut RenderContext,
                    views: &(
            bevy::render::render_resource::TextureView,
            bevy::render::render_resource::TextureView,
        ),
                    pipeline: &bevy::render::render_resource::RenderPipeline,
                    bind: &bevy::render::render_resource::BindGroup| {
            let attach = |view| {
                Some(RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations::default(),
                })
            };
            let mut p = ctx.begin_tracked_render_pass(RenderPassDescriptor {
                label: Some("hiz_level"),
                color_attachments: &[attach(&views.0), attach(&views.1)],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            p.set_render_pipeline(pipeline);
            p.set_bind_group(0, bind, &[]);
            p.draw(0..3, 0..1);
        };

        // Mip 0, from the chosen depth.
        let (seed_bind, seed_pipe) = if multisampled {
            (
                render_context.render_device().create_bind_group(
                    "hiz_seed_ms_bind",
                    &cache.get_bind_group_layout(&pipes.seed_ms_layout),
                    &BindGroupEntries::with_indices(((2, &source_view),)),
                ),
                seed_ms_pipeline,
            )
        } else {
            (
                render_context.render_device().create_bind_group(
                    "hiz_seed_bind",
                    &cache.get_bind_group_layout(&pipes.seed_layout),
                    &BindGroupEntries::with_indices(((0, &source_view),)),
                ),
                seed_pipeline,
            )
        };
        let dst0 = level_view(0);
        pass(render_context, &dst0, seed_pipe, &seed_bind);

        // **Top down, and it cannot be reordered**: each level reads the one above it.
        for level in 1..hiz.mips {
            let src = level_view(level - 1);
            let dst = level_view(level);
            let bind = render_context.render_device().create_bind_group(
                "hiz_reduce_bind",
                &cache.get_bind_group_layout(&pipes.reduce_layout),
                &BindGroupEntries::with_indices(((1, &src.0), (3, &src.1))),
            );
            pass(render_context, &dst, reduce_pipeline, &bind);
        }
        Ok(())
    }
}

pub(super) fn register(app: &mut App) {
    let size = UVec2::splat(8);
    let mips = mip_count(size);
    let (image, far) = {
        let mut images = app.world_mut().resource_mut::<Assets<Image>>();
        (
            images.add(hiz_image(size, mips)),
            images.add(hiz_image(size, mips)),
        )
    };
    app.insert_resource(WaterHiz {
        image,
        far,
        size,
        mips,
        armed: false,
    })
    .add_plugins(ExtractResourcePlugin::<WaterHiz>::default())
    .add_systems(
        PostUpdate,
        (drive_hiz, restamp_hiz)
            .chain()
            .before(bevy::transform::TransformSystems::Propagate)
            .before(bevy::camera::CameraUpdateSystems)
            .after(super::probe::drive_probe),
    );

    let Some(render_app) = app.get_sub_app_mut(bevy::render::RenderApp) else {
        return;
    };

    render_app
        .add_systems(bevy::render::RenderStartup, init_hiz_pipelines)
        .add_render_graph_node::<WaterHizNode>(Core3d, WaterHizLabel)
        // After the opaque world has written the view's depth, before the water is drawn — the
        // same window `scene_color` copies the colour in.
        .add_render_graph_edges(Core3d, (Node3d::MainTransmissivePass, WaterHizLabel))
        .add_render_graph_edges(Core3d, (WaterHizLabel, Node3d::MainTransparentPass));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pyramid's levels have to reach a single texel, or the traversal's top level still
    /// covers more than one tile and the march can walk off the end of the hierarchy.
    #[test]
    fn the_pyramid_reaches_one_texel() {
        for (size, want) in [
            (UVec2::new(1024, 512), 11),
            (UVec2::new(2048, 1024), 12),
            (UVec2::new(8, 8), 4),
        ] {
            assert_eq!(mip_count(size), want, "levels for {size:?}");
            // The last level is one texel on the longest axis.
            assert_eq!(size.x.max(size.y) >> (want - 1), 1);
        }
    }

    /// The base is a power of two on both axes, which is what makes a level exactly half the one
    /// above and the traversal's cell arithmetic a shift.
    #[test]
    fn the_base_is_a_power_of_two() {
        for v in [1u32, 2, 3, 7, 8, 9, 1600, 1920, 2560] {
            let p = prev_pot(v);
            assert!(p.is_power_of_two(), "{v} -> {p}");
            assert!(p <= v.max(1), "{v} -> {p} grew");
            assert!(
                p * 2 > v || v < 2,
                "{v} -> {p} is not the NEAREST power below"
            );
        }
    }
}
