//! The seam the Improved Water plugs into: what the reference water and the rest of the engine
//! must name whether or not that plugin is built. The render layers, the camera markers the static
//! pass and the cull filter on, the probe overlay the minimap reads, the layout of the reflection
//! block `liquid.wgsl` binds, and the pass images every liquid material holds.
//!
//! Without the plugin the pass images are one-texel placeholders and the reflection block is
//! zeros, which the shader reads as "no tier contributing"; the plugin publishes its own targets
//! into [`WaterPassImages`] in [`WaterPassSet`], before the materials are built.

use bevy::asset::RenderAssetUsages;
use bevy::prelude::*;
use bevy::render::render_resource::{
    BufferDescriptor, BufferUsages, Extent3d, TextureDimension, TextureFormat,
    TextureViewDescriptor, TextureViewDimension,
};
use bevy::render::renderer::RenderDevice;

/// The render layer liquid surfaces ride, and the one the reflection camera does not render.
///
/// Public because the app's booth-layer ladder (`portrait`) must stay clear of it, and asserts that
/// it does — a silent layer collision there is exactly the class of bug that ladder was created to
/// stop, and this is the engine's one claim on a layer index outside it.
pub const WATER_RENDER_LAYER: usize = 31;

/// The render layer for **world geometry that must never appear in the mirror** — the overhead unit
/// names and the raid target marks.
///
/// They are not UI. The 1.12 client draws overhead names inside the world pass, depth-tested, as
/// real camera-facing billboard meshes (`benilla_app::nameplates`), which is why walls occlude
/// them — and it is also why the mirrored camera drew them, floating under the surface of every
/// lake, back to front. A name is a label attached to the viewer's own eye; it has no reflection,
/// any more than a cursor does.
///
/// Distinct from [`WATER_RENDER_LAYER`] even though both mean "the world camera draws it, the
/// mirror does not", because the two say different things and only one of them is also a statement
/// about the water: liquid is excluded so it cannot wash the image teal from in front of the
/// mirrored lens, and merging them would make that reason cover labels too.
pub const UNMIRRORED_RENDER_LAYER: usize = 30;

/// The world camera renders the world, the water, and the labels over it; the reflection camera
/// renders only the world. One system rather than a component at the spawn, because there are three
/// spawn sites for the same camera (the client's, its no-data fallback, and the world viewer's) and
/// none of them should have to know that water is layered.
pub(super) fn stamp_world_camera_layers(
    mut commands: Commands,
    cameras: Query<
        Entity,
        (
            With<crate::view::WorldCamera>,
            Added<crate::view::WorldCamera>,
        ),
    >,
) {
    for entity in &cameras {
        commands
            .entity(entity)
            .insert(bevy::camera::visibility::RenderLayers::from_layers(&[
                0,
                UNMIRRORED_RENDER_LAYER,
                WATER_RENDER_LAYER,
            ]));
    }
}

/// The mirror's **view-key shape** — the components that decide which pipelines its draws need,
/// kept in one place because a second camera has to reproduce them exactly.
///
/// A pipeline is specialized against the VIEW as well as the material, so this bundle is the whole
/// reason the mirror does not share the world camera's pipelines: no multisampling and no glow pass
/// (this image is sampled through a rippling normal at half resolution, where neither is
/// recoverable), and — the axis that actually bites — **no `DepthPrepass`**. The water's depth
/// pass puts one on the `WorldCamera` alone, so every material this camera draws needs a second
/// pipeline compiled without `DEPTH_PREPASS`. Measured at the Elwynn golden: the mirrored pass adds
/// ten pipelines over `$WOW_NO_REFLECT=1`, and all ten differ from their main-view twin by that one
/// def and nothing else.
///
/// Extracted so `pipe_warm`'s warm mirror can be built from the SAME bundle. Warming a view key by
/// hand-copying four components is how a warm pass silently stops covering the thing it was written
/// for: the copy keeps compiling, the keys quietly diverge, and the only symptom is the tripwire
/// firing months later with no obvious connection to whatever changed here.
pub fn mirror_view_shape() -> impl Bundle {
    (
        Camera3d::default(),
        bevy::render::view::Msaa::Off,
        bevy::render::view::Hdr,
        bevy::core_pipeline::tonemapping::Tonemapping::None,
    )
}

/// Marks a mirrored camera; the retained static pass draws into exactly the views it is told to,
/// and the water's mirrors are among them (see `static_gx::render`'s marker).
#[derive(Component)]
pub struct ReflectionCamera(pub usize);

/// Which cube face a capture camera owns, in wgpu's layer order: +X, -X, +Y, -Y, +Z, -Z.
///
/// Extracted to the render world so the probe's node can pair each face's colour target with that
/// view's `ViewDepthTexture` — the depth is what becomes the cube's alpha.
#[derive(Component, Clone, Copy, bevy::render::extract_component::ExtractComponent)]
pub struct ProbeFace(pub usize);

/// What the probe needs the static cull to admit while it is capturing.
///
/// **A cubemap needs every direction and the cull list only has one.** `static_gx` builds a single
/// visible set from the WORLD camera's frustum and every view draws it, so a probe face aimed
/// anywhere the player is not looking receives terrain and sky and none of the trees, doodads or
/// buildings that make a reflection worth having. The mirror accepts that knowingly — it is aimed
/// roughly where the camera is — but for a cube it is not an approximation, it is the whole failure.
///
/// So during a capture the cull additionally admits everything within [`radius`](Self::radius) of
/// the probe, frustum and farclip ignored. It is a sphere test per cell, it runs on six frames out
/// of every three hundred and sixty, and the extra cells it admits are real world geometry that the
/// main view simply clips — so the only cost is a little draw time on those frames. A second full
/// cull walk was rejected for the mirror on cost grounds and would be the wrong answer here too.
#[derive(Resource, Clone, Copy, Default)]
pub struct ProbeCull {
    /// Set for the whole capture and a few frames either side of it.
    ///
    /// **Held rather than pulsed, because system order is not guaranteed.** The cull and the
    /// capture driver both run in `Update` with no ordering between them, so a flag raised as a
    /// face is activated can easily be read by a cull that already ran — and that face captures a
    /// world with no trees in it while its neighbours capture one with. It showed as two of the six
    /// faces coming back as flat sky panels in the unwrapped cube while the other four held the
    /// forest. A countdown that spans the whole six-frame capture cannot be raced.
    pub active: bool,
    /// The capture point, in Bevy world yards.
    pub at: Vec3,
    /// How far around it to admit, in yards — the box's reach plus its ceiling, so anything the
    /// projection can land on has been drawn.
    pub radius: f32,
    /// Frames remaining on the hold. Counted down by the capture driver, read as `active` by the
    /// cull.
    pub hold: u32,
}

/// One probe as the debug overlays see it: where it stands, and whether the water can actually read
/// it this frame.
///
/// A flat published view rather than access to the slots themselves, because the overlays want
/// exactly two facts and the slot carries eight — including the resident/target split, which is an
/// invariant of the capture scheduler and nobody else's business.
#[derive(Clone, Copy, Debug)]
pub struct ProbeMark {
    /// The capture point, in **Bevy** world yards.
    pub at: Vec3,
    /// Whether this probe holds a cube the water is reading. False while a slot is waiting for its
    /// first capture.
    pub live: bool,
    /// **This probe's share of the blend at the player's position**, 0 to 1 — normalised across the
    /// taps, so the lit markers are the probes the water is actually reading and the rest are dark.
    pub fade: f32,
}

/// The live probes, republished each frame for the minimap and world overlays; empty without the
/// Improved Water plugin.
#[derive(Resource, Default)]
pub struct ProbeMarks(pub Vec<ProbeMark>);

/// The probe slot ceiling the shader's array is built for.
pub const PROBE_SLOT_MAX: usize = 16;
/// Where the probe slots sit in the reflection block: four `vec4`s per slot.
pub const PROBE_LANES: std::ops::Range<usize> = 48..48 + PROBE_SLOT_MAX * 16;
/// Where the dome's seven rows sit in the reflection block — `sky0..sky4`, `fog`, `warp`, as
/// [`crate::sky::dome_uniforms`] hands them to the dome itself. The shader's `WaterReflect::dome`.
pub const DOME_LANES: std::ops::Range<usize> = PROBE_LANES.end..PROBE_LANES.end + 28;
/// The march's own switches — the shader's `WaterReflect::march`: `x` rays may pass behind tiles,
/// `y` the ripple's origin shift, `z` the probe's softening, `w` probes bound to fixed spots.
pub const MARCH_LANES: std::ops::Range<usize> = DOME_LANES.end..DOME_LANES.end + 4;
/// The second mirror's row: plane, strength, distortion, tolerance — the same four as the first
/// mirror's `params`, zero while it is off.
pub const MIRROR2_LANES: std::ops::Range<usize> = MARCH_LANES.end..MARCH_LANES.end + 4;
/// The reflection block's length in floats — `liquid.wgsl`'s `WaterReflect`.
pub const REFLECT_FLOATS: usize = MIRROR2_LANES.end;

/// The pass targets and the reflection block every liquid material binds. The reference water
/// never samples them; the Improved Water plugin publishes its own in [`WaterPassSet`], and
/// re-points the materials itself when it rebuilds one.
#[derive(Resource, Clone)]
pub struct WaterPassImages {
    /// The first planar mirror's capture.
    pub reflection: Handle<Image>,
    /// The second planar mirror's capture.
    pub reflection2: Handle<Image>,
    /// The scene colour snapshot the march reads.
    pub scene_color: Handle<Image>,
    /// The probe cube array.
    pub probe: Handle<Image>,
    /// The nearest-depth pyramid.
    pub hiz: Handle<Image>,
    /// The farthest-depth pyramid.
    pub hiz_far: Handle<Image>,
    /// The wave simulation's field.
    pub wake: Handle<Image>,
    /// The reflection block, [`REFLECT_FLOATS`] floats, written once a frame in the render world.
    pub reflect_buf: bevy::render::render_resource::Buffer,
}

/// Where a water plugin publishes [`WaterPassImages`]; the liquid materials are built after it.
#[derive(bevy::ecs::schedule::SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WaterPassSet;

/// A one-texel image of `format`, `layers` deep, viewed as `view`.
fn placeholder(format: TextureFormat, layers: u32, view: Option<TextureViewDimension>) -> Image {
    let texel = vec![0u8; format.block_copy_size(None).unwrap_or(4) as usize];
    let mut image = Image::new_fill(
        Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: layers,
        },
        TextureDimension::D2,
        &texel,
        format,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_view_descriptor = view.map(|dimension| TextureViewDescriptor {
        dimension: Some(dimension),
        ..default()
    });
    image
}

/// Publish placeholder pass images when no water plugin has: one texel each, typed as the bindings
/// declare them, and a zeroed reflection block the shader reads as every tier off.
pub(super) fn placeholder_pass_images(
    mut commands: Commands,
    device: Option<Res<RenderDevice>>,
    mut images: ResMut<Assets<Image>>,
) {
    let Some(device) = device else {
        return;
    };
    let flat = images.add(placeholder(TextureFormat::Rgba16Float, 1, None));
    let depth = images.add(placeholder(TextureFormat::R32Float, 1, None));
    let cube = images.add(placeholder(
        TextureFormat::Rgba16Float,
        6,
        Some(TextureViewDimension::CubeArray),
    ));
    let reflect_buf = device.create_buffer(&BufferDescriptor {
        label: Some("water_reflect_params"),
        size: (REFLECT_FLOATS * 4) as u64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    commands.insert_resource(WaterPassImages {
        reflection: flat.clone(),
        reflection2: flat.clone(),
        scene_color: flat.clone(),
        probe: cube,
        hiz: depth.clone(),
        hiz_far: depth,
        wake: flat,
        reflect_buf,
    });
}
