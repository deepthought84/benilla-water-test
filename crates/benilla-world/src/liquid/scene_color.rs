//! The **scene colour the stylised water reflects** — one copy of the main texture, taken after the
//! opaque world has drawn and before anything transparent does.
//!
//! ## Why a copy, and why our own
//!
//! Screen-space reflection needs two things: where the reflected ray lands (depth, which
//! [`super::depth`]'s prepass now supplies) and what colour is there. The second cannot be read
//! from the frame being drawn — the water draws in the transparent phase, where the main colour
//! target is bound as an attachment and sampling it is undefined. So a snapshot has to be taken
//! while it is still readable.
//!
//! **Bevy has exactly this texture and we cannot have it.** `ViewTransmissionTexture` is a copy of
//! the main texture made for specular transmission, and `mesh_view_bindings` even exposes it at
//! binding 24 — but `core_3d` only prepares it when the **`Transmissive3d` phase has items**, and
//! liquid is alpha-blended, so it lands in `Transparent3d` and that phase is empty. Making the
//! water transmissive to borrow the texture would change which phase it draws in, how it is sorted
//! and how it blends, which is a great deal of blast radius for a texture we can copy ourselves in
//! one command.
//!
//! So this is the same shape [`super::reflect`] already uses: an image asset the material binds,
//! rebuilt when the view resizes, filled by a render-graph node. The difference is that the mirror
//! *renders* into its target and this only *copies* into one.
//!
//! ## Where it sits in the frame
//!
//! After [`Node3d::MainTransmissivePass`] and before [`Node3d::MainTransparentPass`]: late enough
//! that the opaque world, the alpha-masked lane and the sky are all in it, early enough that no
//! transparent draw has happened yet. That ordering is the whole contract — the snapshot must not
//! contain the water, or the water would reflect itself.
//!
//! What it therefore cannot show is any *other* transparent surface: a second sheet of water, a
//! particle, a ribbon. That is the same exclusion the depth prepass makes and for the same reason,
//! and it is the right one — a reflection of a puff of smoke that is itself being blended over the
//! water is not a thing to chase.

use bevy::asset::RenderAssetUsages;
use bevy::core_pipeline::core_3d::graph::{Core3d, Node3d};
use bevy::prelude::*;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_graph::{
    Node, NodeRunError, RenderGraphContext, RenderGraphExt, RenderLabel,
};
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages};
use bevy::render::renderer::RenderContext;
use bevy::render::texture::GpuImage;
use bevy::render::view::ViewTarget;

use super::WaterStyle;

/// How much smaller than the main view the snapshot is, per side.
///
/// **One, and it is not an oversight.** The mirror can afford half resolution because it is sampled
/// through a distorting normal at a fraction of its strength; a screen-space march cannot, because
/// it is looking for a *hit* — it walks the depth buffer pixel by pixel, and a snapshot at half the
/// main view's density halves the distance it can resolve before two steps land in one texel. If
/// this ever needs to come down it is the march that has to be told, not just the texture.
const SNAPSHOT_SCALE: u32 = 1;

/// The colour snapshot, and the size it was built for.
#[derive(Resource, Clone, ExtractResource)]
pub(crate) struct WaterSceneColor {
    /// The image the liquid materials sample. Rebuilt on a view resize, exactly as the mirror's
    /// target is, and re-pointed by [`restamp_scene_color`].
    pub(crate) image: Handle<Image>,
    /// Its current size in physical pixels.
    size: UVec2,
    /// Whether the copy should run at all — the stylised look only, like everything else here.
    armed: bool,
}

/// The render-graph node's label.
#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct WaterSceneColorLabel;

/// An HDR image matching the view target's own format, so the copy is a straight blit and the
/// colours arrive on the scale the water is mixing them into.
fn snapshot_image(size: UVec2) -> Image {
    let mut image = Image::new_fill(
        Extent3d {
            width: size.x.max(1),
            height: size.y.max(1),
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0, 0, 0, 0, 0, 0, 0, 0],
        TextureFormat::Rgba16Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    // COPY_DST is the point of this image: the node writes it with `copy_texture_to_texture`
    // rather than rendering into it.
    image.texture_descriptor.usage =
        TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST | TextureUsages::RENDER_ATTACHMENT;
    image
}

/// Keep the snapshot the size of the view, and armed only for the stylised look.
fn drive_scene_color(
    style: Res<WaterStyle>,
    mut snap: ResMut<WaterSceneColor>,
    mut images: ResMut<Assets<Image>>,
    cameras: Query<&Camera, With<crate::view::WorldCamera>>,
) {
    snap.armed = *style == WaterStyle::Stylised;
    let Ok(camera) = cameras.single() else {
        return;
    };
    let want = camera
        .physical_target_size()
        .map_or(snap.size, |s| (s / SNAPSHOT_SCALE).max(UVec2::splat(64)));
    if want != snap.size {
        snap.image = images.add(snapshot_image(want));
        snap.size = want;
    }
}

/// Re-point every liquid material at the snapshot after a rebuild. Change-gated on the handle, so
/// it is a no-op on every frame that is not a resize.
fn restamp_scene_color(
    snap: Res<WaterSceneColor>,
    mut materials: ResMut<Assets<benilla_assets::materials::LiquidMaterial>>,
    mut stamped: Local<Option<AssetId<Image>>>,
) {
    if *stamped == Some(snap.image.id()) {
        return;
    }
    *stamped = Some(snap.image.id());
    for (_, material) in materials.iter_mut() {
        material.extension.scene_color = snap.image.clone();
    }
}

/// The copy itself: one `copy_texture_to_texture` per frame, and nothing else.
#[derive(Default)]
struct WaterSceneColorNode;

impl Node for WaterSceneColorNode {
    fn run(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let Some(snap) = world.get_resource::<WaterSceneColor>() else {
            return Ok(());
        };
        if !snap.armed {
            return Ok(());
        }
        let Some(gpu) = world
            .get_resource::<RenderAssets<GpuImage>>()
            .and_then(|images| images.get(&snap.image))
        else {
            return Ok(()); // the image has not reached the render world yet
        };
        let Some(target) = world.get::<ViewTarget>(graph.view_entity()) else {
            return Ok(());
        };
        // Only when the two agree exactly — a mismatched blit is a validation error, and a frame
        // where the view has resized but the rebuilt image has not yet arrived is a real one.
        let main = target.main_texture();
        if main.width() != gpu.size.width || main.height() != gpu.size.height {
            return Ok(());
        }
        render_context.command_encoder().copy_texture_to_texture(
            main.as_image_copy(),
            gpu.texture.as_image_copy(),
            gpu.size,
        );
        Ok(())
    }
}

pub(super) fn register(app: &mut App) {
    let image = app
        .world_mut()
        .resource_mut::<Assets<Image>>()
        .add(snapshot_image(UVec2::new(64, 64)));
    app.insert_resource(WaterSceneColor {
        image,
        size: UVec2::new(64, 64),
        armed: false,
    })
    .add_plugins(ExtractResourcePlugin::<WaterSceneColor>::default())
    .add_systems(
        PostUpdate,
        (drive_scene_color, restamp_scene_color)
            .chain()
            .before(bevy::transform::TransformSystems::Propagate),
    );

    // After the transmissive pass and before the transparent one — see the module doc. The snapshot
    // must hold the opaque world and NOT the water.
    app.sub_app_mut(bevy::render::RenderApp)
        .add_render_graph_node::<WaterSceneColorNode>(Core3d, WaterSceneColorLabel)
        .add_render_graph_edges(
            Core3d,
            (
                Node3d::MainTransmissivePass,
                WaterSceneColorLabel,
                Node3d::MainTransparentPass,
            ),
        );
}
