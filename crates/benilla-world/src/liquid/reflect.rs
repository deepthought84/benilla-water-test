//! The stylised water's **planar reflection** — a second camera, mirrored through the water plane,
//! rendering the world into an off-screen image the water samples by screen UV.
//!
//! **This belongs to the stylised look and to nothing else.** The 1.12 client has no reflection of
//! any kind on this path (its water is a depth swatch, an animated sheet and a specular sheen), so
//! everything here is off, and costs nothing, while `waterStyle` is Reference. What it reproduces is
//! the shape the *reference's own Ultra water* has — a real mirrored-camera capture, which is the
//! only mechanical difference between High and Ultra in the 4.3.4 renderer this project's sibling
//! sandbox reverse-engineered.
//!
//! ## The three things that are load-bearing
//!
//! - **The mirror is not a rotation.** A true planar reflection is `view · R` with `R` the mirror
//!   through the plane, and that matrix has a negative determinant — no `Transform` can hold it.
//!   `looking_to(R·forward, R·up)` differs from it in exactly one respect: its right vector is
//!   negated, which flips the image horizontally. The shader samples at `1 − u` to undo that. It is
//!   the exact correction, not a fudge, and it holds at any camera roll.
//! - **Alpha is coverage.** The camera clears to fully transparent, so a texel's alpha says whether
//!   the mirrored view drew anything there. Where it did not, the water keeps the Fresnel sky mix it
//!   would have had — which is why the sky needs no special case and a reflection that fails to
//!   render degrades to the look without one rather than to a hole.
//! - **The water itself must not be in it.** The mirrored camera sits below the plane looking up
//!   through it, so the water surface is between it and everything it is there to capture — and
//!   liquid draws blended with depth-write off, so it would not occlude the reflection, it would
//!   wash it teal. Liquid surfaces therefore ride [`WATER_RENDER_LAYER`], which the world camera
//!   renders and this one does not.
//!
//! ## What it costs, and what that buys back
//!
//! A second pass over the world's draws. Two things hold it down: the target is **half the main
//! view's resolution** (it is sampled through a rippling normal — the detail is not recoverable and
//! not missed), and the pass is **inactive** whenever the look is off, no water is near the eye, or
//! the eye is under the surface. The oblique clip below pays for part of the rest: the mirror draws
//! nothing under the waterline.
//!
//! It renders **every frame**. Alternating was the third saving and it is gone: it judders (see
//! [`drive_reflection`]) and, measured, it bought 1.7 %.

use bevy::asset::RenderAssetUsages;
use bevy::camera::{ClearColorConfig, PerspectiveProjection, RenderTarget};
use bevy::math::Mat3;
use bevy::prelude::*;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::render_resource::{
    Buffer, BufferDescriptor, BufferUsages, Extent3d, TextureDimension, TextureFormat,
    TextureUsages,
};
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy::render::view::Hdr;
use bevy::render::{Render, RenderApp, RenderSystems};

use super::query::WaterChunkInfo;
use super::WaterStyle;
use crate::view::WorldCamera;
use benilla_assets::coords::{bevy_to_wow, wow_to_bevy};

/// How far from the eye a water surface may be and still be the **fallback** plane — the "water
/// beside me" answer used when the view ray hits nothing, in yards. Finite, because the pass is a
/// second world draw and a player crossing a continent should not pay for it.
///
/// This is deliberately NOT the reach of the look test; see [`look_reach`].
const REFLECT_RADIUS: f32 = 120.0;

/// How far the look test reaches — **the player's own view distance**, not a constant.
///
/// Split from [`REFLECT_RADIUS`] because the two answer different questions and sharing a number
/// made the far case wrong. The surface you are *watching* is routinely past 120 yd — a lake seen
/// from a ridge, the sea from a cliff path — and with one radius the ray test found nothing out
/// there and fell through to "nearest", which is the stream at your feet. The capture was then
/// built for the puddle's plane while the lake filled the screen: not a missing reflection, a wrong
/// one, and the reported "at some distance the reflection is not correct".
///
/// **It is `farclip` because any other number is a second, invisible view distance.** This was
/// briefly a flat 600, which is inside the 717 a player on the slider's upper half is running: the
/// world drew water for another 117 yd that the reflection had already given up on, and the seam
/// moved whenever they touched the slider. `farclip` is also the honest ceiling in the other
/// direction — past it the liquid is not resident (`terrain_stream::window` sizes residency from
/// the same number), so there is nothing to find and the WDL horizon is drawing instead.
///
/// The proximity fallback keeps [`REFLECT_RADIUS`], because a plane picked for a surface half a
/// zone away is a worse answer for the water you are standing in than the water you are standing
/// in is.
fn look_reach(farclip: f32) -> f32 {
    farclip.max(REFLECT_RADIUS)
}

/// The mirror's own far clip, in yards — **not** the world camera's. `$WOW_MIRROR_FAR` overrides it.
///
/// The pass inherited the world lens whole (only its near plane was replaced), and the world lens
/// reaches about 3000 yd: far beyond `farclip` on purpose, so the coarse WDL horizon can draw
/// behind the wall (`view::within_farclip`). The mirror inherited that reach for an image that is
/// downscaled, sampled through a rippling normal, and mixed in at [`REFLECT_MAX`] at its very
/// strongest — a second horizon's worth of culling and draws, for a picture that cannot show it.
///
/// **This is a frustum bound, not a clip plane.** Bevy's perspective matrix is
/// `perspective_infinite_reverse_rh` and does not carry `far` at all; `far` builds the [`Frustum`]
/// the per-view visibility pass tests against, so what this buys is entities never reaching the
/// mirror's draw list. Nothing is clipped mid-mesh and there is no wall in the image — geometry
/// simply stops being submitted.
///
/// Five hundred rather than the two or three hundred the near field would justify, because the
/// reflection is at its *strongest* where it shows the most distant things: Fresnel runs to
/// [`REFLECT_MAX`] at grazing angles, which is exactly the long view down a lake or out to sea. Cut
/// too close and the reflected far shore goes missing from the one shot that shows it off. Sweep it
/// with `$WOW_MIRROR_FAR` against `$WOW_GPU_MS` before moving the default.
const MIRROR_FAR_YARDS: f32 = 500.0;

/// How much smaller than the main view the mirror's target is, per side. `$WOW_REFLECT_SCALE`
/// overrides it.
///
/// Left at the 2 this pass shipped with, because raising it is a *look* change and belongs to
/// whoever is looking at the water — but named and levered, because it is the cheapest knob here:
/// the image is sampled through a distorting normal, so 3 or 4 may well be indistinguishable at a
/// third or a quarter of the mirror's fragments.
const REFLECT_DOWNSCALE: u32 = 2;

/// `$WOW_MIRROR_FAR=<yards>` — the mirror's frustum reach, for sweeping [`MIRROR_FAR_YARDS`]
/// against the frame meter without a rebuild. Read once, like every other lever in this module.
fn mirror_far() -> f32 {
    static FAR: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *FAR.get_or_init(|| {
        std::env::var("WOW_MIRROR_FAR")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .filter(|v| *v > 0.0)
            .unwrap_or(MIRROR_FAR_YARDS)
    })
}

/// `$WOW_REFLECT_SCALE=<n>` — the mirror's resolution divisor, for the same reason. Clamped to a
/// sane range: 1 is the main view's own size (the most this could ever want) and 8 is past the
/// point where the capture holds a recognisable image at all.
fn reflect_downscale() -> u32 {
    static SCALE: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *SCALE.get_or_init(|| {
        std::env::var("WOW_REFLECT_SCALE")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(REFLECT_DOWNSCALE)
            .clamp(1, 8)
    })
}

/// How far from the capture plane a surface is still served at **full** reflection strength, in
/// yards — the shader fades it out over a wider band past this rather than cutting it off
/// (`liquid.wgsl`'s `plane_trust`).
///
/// **This is not a "which surface may reflect" threshold, and it must not become one again.** A
/// frame regularly holds more than one water body — a roadside pool and the sea, a river and the
/// pond above it — and they all have to reflect at the same time. One capture can do that because
/// the shader reprojects each fragment by its own distance from the capture plane; what this number
/// bounds is where that correction is exact enough to trust at full weight.
///
/// It replaced a hard 3.0 that was a cutoff, and the cutoff was wrong twice over. A liquid grid is
/// a heightfield — Felwood's river drops about two yards across a single MCNK and Elwynn's stream
/// has 3.04 in one chunk — so three yards could not even cover one surface; and across two bodies
/// it meant exactly one of them reflected while the other went flat, with the choice flipping as
/// the camera moved.
///
/// ## What is still only approximate across bodies
///
/// The mirror clips its own waterline at the capture plane ([`WATERLINE_CLIP_BIAS`]), so a body
/// *below* that plane is missing whatever geometry stands between the two heights, and one above it
/// keeps a little it should have cut. The clip follows the plane the prominence walk picked — the
/// water covering most of the screen — so the error lands on the bodies occupying least of it.
const PLANE_TOLERANCE: f32 = 10.0;

/// How far below the capture plane the mirror's clip may be dropped to spare a lower water body,
/// in yards.
///
/// The two artefacts this sits between are both real and both have been seen. Too small and a
/// lower surface loses its reflection outright (the Savage Coast seam). Too large and the capture
/// keeps the submerged world — hulls, pilings, the sea bed — which the capture plane's own body
/// then paints as hard-edged slabs standing in the water (Booty Bay, the reason the oblique clip
/// exists at all).
///
/// Twelve yards covers the spread a view actually holds — a river mouth against the sea is about
/// ten, terraced pools less — while staying well inside the depth at which the reference world's
/// large submerged geometry sits.
/// **Zero, and it stays zero until the clip plane's `w` convention is settled.** Dropping the clip
/// was meant to spare a lower water body's reflection; measured, it does the opposite — it removes
/// the geometry NEAREST the waterline from the capture, which is the bank and the base of every
/// tree trunk. At a Stranglethorn river camera a 12 yd drop emptied the top of the mirror image and
/// the water went flat; `WOW_CLIP_DROP=0` brought the trunk reflections straight back.
///
/// That means `pos.y - (clip_at - WATERLINE_CLIP_BIAS)` does not move the plane the way lowering
/// `clip_at` implies — Bevy's contract wants the **negative** signed distance from the camera to
/// the plane, and this passes the positive one, so the effective plane is mirrored about the
/// camera and a lower `clip_at` raises it. The existing value worked because `clip_at == plane`
/// was the only case ever exercised. Correcting the sign is a separate change that has to be
/// re-validated against Booty Bay, which is what the oblique clip exists for.
///
/// `$WOW_CLIP_DROP=<yards>` overrides it for that investigation.
const MAX_CLIP_DROP: f32 = 0.0;

/// The share of the most prominent body's score a surface needs before it may lower the clip.
const CLIP_SHARE: f32 = 0.10;

/// How far below the water the mirror's clip plane sits, in yards. See the plane's use: clipping
/// exactly on the surface leaves a hairline gap where a hull meets its own reflection.
const WATERLINE_CLIP_BIAS: f32 = 0.25;

/// How far the surface normal displaces the reflection's UV — the wobble that makes it read as a
/// reflection in *water* rather than as a mirror. Small: past a few hundredths the image tears off
/// the geometry that casts it.
///
/// Raised from 0.035 along with the roughness pass in `liquid.wgsl`: the complaint that the water
/// looked "too perfect" is answered half by reflecting less and half by reflecting less *cleanly*,
/// and this is the second half.
const REFLECT_DISTORT: f32 = 0.055;

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

/// The reflection target and the buffer the water samples it through.
#[derive(Resource)]
pub(crate) struct WaterReflect {
    /// The off-screen colour target. Rebuilt (a new asset, as the world backdrop does it) when the
    /// main view's size changes; [`restamp_reflection_target`] re-points the materials.
    pub(crate) image: Handle<Image>,
    /// The image's current size in physical pixels.
    size: UVec2,
}

/// The per-frame parameters the water shader reads — **two `vec4`s**, and only the first is about
/// the reflection:
///
/// - `0..4` — the mirror: `x` = the plane's world Y, `y` = the reflection's strength (0 disables it
///   — no plane, no water near, an eye under the surface, or the reference look), `z` = the UV
///   distortion, `w` = the plane tolerance.
/// - `4..8` — **the sun the player can actually see**: `xyz` = the to-sun direction
///   ([`crate::lighting::WowLighting::celestial_dir`]) and `w` = [`crate::sun::SunVisibility`].
/// - `8..12` / `12..16` — **the sky the surface reflects**: the zenith and horizon stops of the
///   dome's own gradient (`WowLighting::sky` rows 0 and 4 — the same colours the dome is drawn
///   with, so the water agrees with the sky above it at every hour and in every zone). The water
///   reads them at the reflected view ray's elevation, which is what makes a wave visible when the
///   sun is behind you; a single flat sky colour cannot, because at a grazing angle a tilted facet
///   and a flat one both return it.
/// - `20..24` — **the moon**, the same pair as the sun: `xyz` = the to-moon direction and `w` =
///   [`crate::sun::MoonVisibility`]. A separate body rather than a switch on the sun's, because on
///   the nights both are up they are up together and in different parts of the sky.
/// - `16..20` — **the wave simulation's window** (`super::ripple_sim`): `xy` = its lower corner in
///   Bevy XZ yards, `z` = one over its side length, `w` = strength (0 on the reference lane, where
///   the client's own painted splash decals are the wake instead).
///
/// The sun lanes live here rather than in the shared light buffer because that buffer's layout is
/// mirrored by three shaders and read by every draw in the frame, and this is one consumer's
/// per-frame scalar. They are written unconditionally — the water's glitter needs them on exactly
/// the frames the reflection is off.
#[derive(Resource, Clone, Copy, Default, ExtractResource)]
pub(crate) struct WaterReflectData(pub(crate) [f32; 24]);

/// The buffer every liquid material binds (`#[storage(107, …)]`), written once a frame in the render
/// world — the same shape as the shared light buffer, and for the same reason: a per-frame material
/// mutation would mark every liquid material Modified and rebuild its bind group, which this project
/// has already measured and retired once.
#[derive(Resource, Clone, ExtractResource)]
pub(crate) struct WaterReflectBuffer(pub(crate) Buffer);

/// Marks the mirrored camera.
///
/// `pub(crate)` because the retained static pass has to know this view exists: it draws into
/// exactly the views it is told to, and the water's mirror is the second one (see
/// `static_gx::render`'s marker).
#[derive(Component)]
pub(crate) struct ReflectionCamera;

/// The mirrored camera as [`drive_reflection`] writes it: its pose, its gate, its lens and its
/// target, all four of which move together when the plane or the main view does.
type MirrorCamera<'w, 's> = Query<
    'w,
    's,
    (
        &'static mut Transform,
        &'static mut Camera,
        &'static mut Projection,
        &'static mut RenderTarget,
    ),
    (With<ReflectionCamera>, Without<WorldCamera>),
>;

pub(super) fn register(app: &mut App) {
    app.init_resource::<WaterReflectData>()
        .add_plugins(ExtractResourcePlugin::<WaterReflectData>::default())
        .add_plugins(ExtractResourcePlugin::<WaterReflectBuffer>::default())
        // Before the liquid materials are built: they bind the buffer and the image, and a
        // material built before either exists would have to be rebuilt to get them.
        .add_systems(
            Startup,
            setup_reflection
                .after(benilla_assets::AssetSet::Open)
                .before(super::surface::setup_liquid),
        )
        .add_systems(Update, stamp_world_camera_layers)
        .add_systems(
            Last,
            trace_reflection.run_if(resource_exists::<WaterReflect>),
        )
        .add_systems(
            PostUpdate,
            (
                // After the camera has been moved for the frame and before transforms propagate,
                // so the mirrored pose reaches the render world in the same frame as the pose it
                // mirrors. A frame-late mirror swims against the camera on every turn.
                drive_reflection.before(bevy::transform::TransformSystems::Propagate),
                restamp_reflection_target,
            )
                .chain(),
        );
    app.sub_app_mut(RenderApp).add_systems(
        Render,
        upload_reflect.in_set(RenderSystems::PrepareResources),
    );
}

/// The world camera renders the world, the water, and the labels over it; the reflection camera
/// renders only the world. One system rather than a component at the spawn, because there are three
/// spawn sites for the same camera (the client's, its no-data fallback, and the world viewer's) and
/// none of them should have to know that water is layered.
fn stamp_world_camera_layers(
    mut commands: Commands,
    cameras: Query<Entity, (With<WorldCamera>, Added<WorldCamera>)>,
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

fn setup_reflection(
    mut commands: Commands,
    device: Res<RenderDevice>,
    mut images: ResMut<Assets<Image>>,
) {
    let size = UVec2::new(640, 360);
    let image = images.add(reflection_image(size));
    let camera = commands
        .spawn((
            Name::new("water reflection camera"),
            Camera3d::default(),
            // Never `WorldCamera`: that marker means *the* viewer, and every "where is the eye"
            // consumer in the client filters on it.
            ReflectionCamera,
            // No multisampling and no glow pass: this image is sampled through a rippling normal at
            // half resolution, where neither is recoverable.
            bevy::render::view::Msaa::Off,
            Hdr,
            bevy::core_pipeline::tonemapping::Tonemapping::None,
            RenderTarget::Image(image.clone().into()),
            Camera {
                // Ahead of the world camera (order 0), so the water samples this frame's image rather
                // than last frame's on the frames it does render.
                order: -1,
                // Transparent, because alpha is coverage — see the module doc.
                clear_color: ClearColorConfig::Custom(Color::NONE),
                is_active: false,
                ..default()
            },
            Transform::default(),
        ))
        .id();
    if reflect_debug() {
        // Straight to the window, over the world (the world camera is order 0), so the mirrored
        // image is what you see. The water still samples its own target, which nothing writes —
        // this mode is for looking at the capture, not at the water.
        commands.entity(camera).insert(RenderTarget::default());
        commands.entity(camera).insert(Camera {
            order: 1,
            clear_color: ClearColorConfig::Custom(Color::BLACK),
            is_active: false,
            ..default()
        });
    }
    commands.insert_resource(WaterReflect { image, size });
    commands.insert_resource(WaterReflectBuffer(device.create_buffer(
        &BufferDescriptor {
            label: Some("water_reflect_params"),
            size: 96,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        },
    )));
}

/// `$WOW_NO_REFLECT=1` — the reflection pass's kill switch, in the mould of `$WOW_NO_LIQUID` and
/// `$WOW_NO_PARTICLES`: the stylised look keeps its ripples, glitter, sky mix and foam, and only the
/// mirrored pass goes. It is the A/B this feature is priced with (a second world draw is the one
/// thing here that can cost a frame rate) and the answer for a machine that cannot afford it.
///
/// Read once — an env var cannot change mid-session, and this is asked every frame.
fn no_reflect() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| std::env::var_os("WOW_NO_REFLECT").is_some())
}

/// `$WOW_REFLECT_DEBUG=1` — draw the mirrored camera's image **to the window** instead of into the
/// water, over the top of the world view.
///
/// A reflection is the hardest thing in this module to reason about from the outside: what reaches
/// the water is a mirrored image sampled by screen UV through a rippling normal, so an artefact in
/// it (the plane a yard off, geometry that should have been clipped away, a flipped axis) arrives
/// as "the water looks wrong" and every one of those causes looks the same. This shows the image
/// itself. Read once, like every other lever here.
fn reflect_debug() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("WOW_REFLECT_DEBUG").is_some())
}

/// `$WOW_REFLECT_TRACE=1` — one line a frame: the plane, whether the pass ran, and the target's
/// size. It exists because everything this module decides is invisible in the frame it decides it
/// (a reflection that is not drawn looks exactly like water that is not reflective), and because
/// the line rate is also this feature's frame-rate meter — the viewer has no FPS readout.
fn trace_reflection(
    data: Res<WaterReflectData>,
    reflect: Res<WaterReflect>,
    eye: Query<&GlobalTransform, With<WorldCamera>>,
    chunks: Query<&WaterChunkInfo>,
) {
    if std::env::var_os("WOW_REFLECT_TRACE").is_some() {
        let at = eye.single().map(|e| bevy_to_wow(e.translation()));
        info!(
            "reflect: plane {:.2} strength {:.0} size {}x{} eye {:?} surfaces {}",
            data.0[0],
            data.0[1],
            reflect.size.x,
            reflect.size.y,
            at.map(|[x, y, z]| [x.round(), y.round(), z.round()]),
            chunks.iter().count()
        );
    }
}

/// An HDR colour target with an alpha channel — the same `Rgba16Float` the world camera renders,
/// so the reflected colours arrive on the same scale the water is mixing them into.
fn reflection_image(size: UVec2) -> Image {
    let mut image = Image::new_fill(
        Extent3d {
            width: size.x.max(1),
            height: size.y.max(1),
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0; 8],
        TextureFormat::Rgba16Float,
        RenderAssetUsages::default(),
    );
    image.texture_descriptor.usage =
        TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST | TextureUsages::RENDER_ATTACHMENT;
    image
}

/// A wet height on this surface near `(x, y)`, and how far away that sample is — the surface's own
/// answer to "how high is the water over there".
///
/// The clamped box point is only the first guess, and taking its `None` for an answer was a bug
/// with a very visible face. [`WaterChunkInfo::surface_z_at`] answers *wet or dry at this point*,
/// which is the right question for a swimmer and the wrong one for a plane: a WMO pool's footprint
/// is a box around a whole district, and a camera standing on a bridge inside that box is over a
/// DRY cell of it. Read as "this surface is not here", that left Stormwind's canals reflecting from
/// some positions and not from others a few yards away — the reported symptom, and reproducible in
/// the world viewer at one pose out of four.
///
/// So when the near point is dry, a short lattice over the footprint answers instead: any wet cell
/// will do, and the nearest one is the best reading of "the water beside me".
///
/// It is rarer than it looks, and worth knowing why before pricing it: `xy_bounds` is the box of
/// the **wet** cells, so clamping into it already lands on water for any convex surface, and only a
/// wet region shaped like an L — a canal bend — leaves the clamped corner dry. That is what the
/// lattice is for. It runs only when the cheap point missed, and only for the surfaces near enough
/// to be the fallback plane (`lattice`); see [`look_reach`] for why the walk now visits
/// many more surfaces than it can afford to probe.
fn wet_height_near(
    chunk: &WaterChunkInfo,
    x: f32,
    y: f32,
    lattice: bool,
) -> Option<(f32, f32, [f32; 2])> {
    let [[min_x, min_y], [max_x, max_y]] = chunk.xy_bounds()?;
    let (cx, cy) = (x.clamp(min_x, max_x), y.clamp(min_y, max_y));
    let d2 = |px: f32, py: f32| (px - x) * (px - x) + (py - y) * (py - y);
    if let Some(z) = chunk.surface_z_at(cx, cy) {
        return Some((z, d2(cx, cy), [cx, cy]));
    }
    // Only for the surfaces near enough to be the fallback plane. The lattice is 25 grid probes and
    // the walk now reaches out to [`look_reach`], which is several times the surfaces —
    // paying it on every one of them, once a frame, to rescue the rare distant WMO pool whose
    // near corner happens to be dry is the wrong trade. A far surface that misses simply is not a
    // candidate, which is the behaviour this walk had at every distance before.
    if !lattice {
        return None;
    }
    const N: i32 = 4;
    let mut best: Option<(f32, f32, [f32; 2])> = None;
    for i in 0..=N {
        for j in 0..=N {
            let px = min_x + (max_x - min_x) * i as f32 / N as f32;
            let py = min_y + (max_y - min_y) * j as f32 / N as f32;
            let Some(z) = chunk.surface_z_at(px, py) else {
                continue;
            };
            let d = d2(px, py);
            if best.is_none_or(|(bd, _, _)| d < bd) {
                best = Some((d, z, [px, py]));
            }
        }
    }
    best.map(|(d, z, at)| (z, d, at))
}

/// The water plane to mirror through: **the surface the camera is looking at**, out to
/// [`look_reach`], and only failing that the nearest one ahead within [`REFLECT_RADIUS`].
///
/// Nearest-to-the-eye alone was the first rule and it is the wrong question. A capture is right for
/// one plane, so the plane has to be the one the player is *watching*: standing on a bank between a
/// stream and a lake, "nearest" picks the stream, the lake falls outside the tolerance and reflects
/// nothing — and because zooming the camera in or out moves the eye tens of yards, which of the two
/// is nearest flips as the camera moves.
///
/// The look test is a ray-plane hit, evaluated per surface: take a wet sample near the eye for the
/// surface's height, intersect the view ray with that height, and keep the hit only if the point it
/// lands on is wet too. One iteration is enough — a liquid grid is a heightfield, but a gentle one.
/// The nearest hit along the ray wins, which is what "looking at" means when two surfaces overlap.
///
/// **A full walk, not the `WaterIndex`**, which is the shape that index's own module doc prescribes
/// for a once-a-frame consumer ("fine for the consumers that ask once or twice a frame … and it
/// detonates the moment a consumer asks per *draw*"). It is also the reading that works: at a
/// Stormwind camera the index answered `over()` empty for a cell whose surface the walk finds
/// containing the point — 22 of 673 loaded surfaces were unreachable through it while 651 were — so
/// an index query here would have left the canals with no reflection and no error.
fn plane_near(
    eye: Vec3,
    forward: Vec3,
    up: Vec3,
    fov_y: f32,
    aspect: f32,
    reach: f32,
    held: Option<f32>,
    chunks: &Query<&WaterChunkInfo>,
) -> Option<(f32, f32)> {
    let [x, y, _] = bevy_to_wow(eye);
    let frame = Frame::new(forward, up, fov_y, aspect);
    let mut seen = Prominence::default();
    let mut nearest: Option<(f32, f32)> = None; // (distance² to a wet sample, plane)
    for chunk in chunks {
        // The cheap reject first, on the footprint alone: everything below samples the grid.
        let Some([[min_x, min_y], [max_x, max_y]]) = chunk.xy_bounds() else {
            continue;
        };
        let (bx, by) = (x.clamp(min_x, max_x), y.clamp(min_y, max_y));
        let foot_d2 = (bx - x) * (bx - x) + (by - y) * (by - y);
        if foot_d2 > reach * reach {
            continue;
        }
        // Near enough to be the *fallback* plane, which is a shorter reach than the look test's.
        let near = foot_d2 <= REFLECT_RADIUS * REFLECT_RADIUS;
        let Some((z, d2, at)) = wet_height_near(chunk, x, y, near) else {
            continue; // a footprint with no wet cell in reach
        };
        if near && nearest.is_none_or(|(bd2, _)| d2 < bd2) {
            nearest = Some((d2, z));
        }
        // **How much of the screen does this surface cover?** — see [`Prominence`]. Not "is the
        // centre ray on it" (which the sea fails from up the beach) and not "is it the nearest one
        // in frame" (which hands a roadside pool the capture while the sea fills the window).
        if eye.y <= z {
            continue;
        }
        let Some((depth, on_screen)) = frame.depth_if_visible(eye, wow_to_bevy([at[0], at[1], z]))
        else {
            continue;
        };
        if depth > reach {
            continue;
        }
        let area = (max_x - min_x) * (max_y - min_y);
        seen.add(z, on_screen * area / (depth * depth));
    }
    if std::env::var_os("WOW_REFLECT_TRACE").is_some() {
        info!(
            "plane_near: reach {reach:.0} fov {fov_y:.2} aspect {aspect:.2} \
             seen {} best {:?} nearest {:?} slots {:?}",
            seen.used,
            seen.best(),
            nearest.map(|(_, z)| z),
            &seen.slots[..seen.used],
        );
    }
    let plane = seen.best_sticky(held).or(nearest.map(|(_, z)| z))?;
    // The lowest surface actually on screen, which is what the mirror's clip has to spare — see
    // `drive_reflection`. Falls back to the chosen plane when nothing else was in frame.
    // **Only bodies with a real share of the screen may pull the clip down.** A surface scoring
    // 0.013 against the river's 3.9 — a puddle at the edge of the frame — was setting `lowest` and
    // moving the clip a full twelve yards, which is a large change to the capture driven by
    // something invisible in it.
    let top = seen.slots[..seen.used]
        .iter()
        .map(|(_, s)| *s)
        .fold(0.0_f32, f32::max);
    let lowest = seen.slots[..seen.used]
        .iter()
        .filter(|(_, s)| *s >= top * CLIP_SHARE)
        .map(|(z, _)| *z)
        .fold(plane, f32::min);
    Some((plane, lowest))
}

/// How much screen each distinct water height covers, accumulated over the surfaces in frame.
///
/// **The third rule this function has had, and the reasons the first two failed are the argument
/// for this one.** "What the centre ray lands on" cannot see a sea that fills the top half of the
/// window while the middle of the screen is beach. "The nearest surface in frame" can see it, but
/// ranks a pool at the roadside above it — measured on the Savage Coast as plane 10.07 at every
/// distance from 100 to 700 yd, with the ocean right there at 0.00.
///
/// What actually decides which plane a capture should serve is **which water the player is mostly
/// looking at**, and that is a question about screen area. A surface's projected area falls as
/// `area / depth²`, so one chunk of pool close by and a hundred chunks of ocean far away can be
/// compared on the same scale — and the ocean, being made of many chunks, accumulates.
///
/// Heights are bucketed because a water body is many surfaces at one height: the ocean is hundreds
/// of ADT chunks all at 0.0, and they have to add up rather than compete. [`BUCKET_YARDS`] is well
/// under `PLANE_TOLERANCE`, so two heights that land in one bucket are two the capture could serve
/// together anyway.
#[derive(Default)]
struct Prominence {
    /// `(height, accumulated area/depth²)`, in no order.
    slots: [(f32, f32); PROMINENCE_SLOTS],
    used: usize,
}

/// How far apart two surface heights must be to be ranked separately, in yards.
const BUCKET_YARDS: f32 = 0.5;

/// How much better a challenger must score than the plane already in use before the capture is
/// handed over.
///
/// **Continuity is worth more here than being right by a nose.** Switching plane changes every
/// fragment's error, trust, reprojection and the mirror's clip in a single frame, so a handover is
/// expensive to look at however well-judged it is; two bodies scoring within a few percent of each
/// other would otherwise trade the capture back and forth every time the camera breathed. A
/// challenger that is genuinely what the player is looking at clears this easily — at the Savage
/// Coast the pool outscored the ocean seven-fold — while a near-tie leaves the picture alone.
const HYSTERESIS: f32 = 1.5;

/// How many distinct water heights can be ranked at once.
///
/// Twelve is far past what a view holds in the shipped world — a terraced spot like Stormwind's
/// canals shows three or four — and the overflow path folds a thirteenth into its nearest
/// neighbour rather than dropping it, so a pathological view degrades in accuracy, never into
/// having no reflection.
const PROMINENCE_SLOTS: usize = 12;

impl Prominence {
    fn add(&mut self, z: f32, score: f32) {
        if !score.is_finite() || score <= 0.0 {
            return;
        }
        let mut closest = usize::MAX;
        let mut closest_gap = f32::MAX;
        for (i, (bz, _)) in self.slots[..self.used].iter().enumerate() {
            let gap = (bz - z).abs();
            if gap < closest_gap {
                (closest, closest_gap) = (i, gap);
            }
        }
        if closest_gap <= BUCKET_YARDS {
            self.slots[closest].1 += score;
        } else if self.used < PROMINENCE_SLOTS {
            self.slots[self.used] = (z, score);
            self.used += 1;
        } else {
            // Full: fold into the nearest height rather than lose the surface entirely.
            self.slots[closest].1 += score;
        }
    }

    /// The height covering the most screen, **with the plane already in use given right of
    /// first refusal** — see [`HYSTERESIS`].
    fn best_sticky(&self, incumbent: Option<f32>) -> Option<f32> {
        let challenger = self.best()?;
        let Some(held) = incumbent else {
            return Some(challenger);
        };
        // Is the plane we are already serving still on screen at all? Find its bucket.
        let standing = self.slots[..self.used]
            .iter()
            .find(|(z, _)| (z - held).abs() <= BUCKET_YARDS)
            .map(|(z, score)| (*z, *score));
        let Some((held_z, held_score)) = standing else {
            return Some(challenger); // gone from the frame; nothing to defend
        };
        let top = self.slots[..self.used]
            .iter()
            .map(|(_, s)| *s)
            .fold(0.0_f32, f32::max);
        if top > held_score * HYSTERESIS {
            Some(challenger)
        } else {
            Some(held_z)
        }
    }

    /// The height covering the most screen, or `None` when nothing was in frame.
    fn best(&self) -> Option<f32> {
        self.slots[..self.used]
            .iter()
            .copied()
            .fold(None, |best: Option<(f32, f32)>, (z, score)| match best {
                Some((_, bs)) if bs >= score => best,
                _ => Some((z, score)),
            })
            .map(|(z, _)| z)
    }
}

/// The camera's frame, as the three numbers a "is that point on screen" test needs.
///
/// **This replaces a ray march, and the reason is worth keeping.** The look test used to intersect
/// the camera's forward ray with each surface's height and ask whether the crossing point was wet.
/// That answers for the middle of the screen and nothing else, so standing back from a shore and
/// looking out to sea — where the middle of the screen is *beach* — it found no water at all, and
/// the ocean filling the upper half of the frame got no reflection until the player walked close
/// enough for the centre of the screen to land on it. Measured at the Savage Coast: strength 0 from
/// 400 yd up the beach, with the sea plainly in view.
///
/// Fanning the ray over the vertical FOV was the first fix and it **did not work**, for a reason
/// worth recording: ground distance runs as `height / tan(θ)`, so a fan spread evenly in *angle* is
/// spread wildly unevenly in *distance*. At that same spot the ocean subtended about 0.4° of
/// depression angle, and seven rays across a 46° frame stepped straight over the band — the sweep
/// came back strength 0 at every distance it had before.
///
/// So the question is asked directly instead: take the surface's nearest wet point, and test
/// whether it falls inside the view frustum's four sides. No marching, no sampling density to get
/// wrong, and one grid lookup per surface rather than one per ray.
struct Frame {
    forward: Vec3,
    /// The frame's up and right, squared against `forward` — a rolled or degenerate basis must not
    /// tilt the test out of the camera's own plane.
    up: Vec3,
    right: Vec3,
    /// Tangents of the half angles: vertical, then horizontal.
    tan_v: f32,
    tan_h: f32,
}

/// How far past the frustum's true edge a surface still counts as "looked at".
///
/// Generous on purpose. The test stands on ONE point of a surface that is usually far wider than
/// the screen, so a sea whose nearest wet cell sits just off the left edge is still the thing being
/// looked at; being strict about the edge would reintroduce the popping this exists to remove, just
/// at a different boundary.
const FRAME_SLOP: f32 = 1.35;

/// How a surface's contribution falls off as it leaves the frame — 1 well inside, 0 well outside.
///
/// **A step function here is what made the capture plane switch**, and it is worth being precise
/// about why, because the symptom looked like a reflection bug rather than a scoring one. Each
/// surface is tested at ONE point, so an in-or-out test makes its whole score appear and vanish the
/// instant that point crosses the edge: measured at the Savage Coast, a 10 deg turn took a pool
/// from 37.7 to absent, the winner flipped to the ocean, and every fragment in the frame changed
/// its plane error, its trust, its reprojection and the mirror's clip in one frame. The water had
/// not moved; the arithmetic had.
///
/// Fading the contribution across the frame edge instead makes the score continuous, so a
/// crossover between two bodies is a slow trade rather than a jump.
fn edge_fade(offset: f32, half_extent: f32) -> f32 {
    let outer = half_extent * FRAME_SLOP;
    1.0 - ((offset - half_extent) / (outer - half_extent)).clamp(0.0, 1.0)
}

impl Frame {
    fn new(forward: Vec3, up: Vec3, fov_y: f32, aspect: f32) -> Self {
        let forward = forward.normalize_or_zero();
        let up = (up - forward * forward.dot(up)).normalize_or_zero();
        let tan_v = (fov_y * 0.5).tan();
        Self {
            forward,
            up,
            right: forward.cross(up).normalize_or_zero(),
            tan_v,
            tan_h: tan_v * aspect.max(0.1),
        }
    }

    /// How far in front of the eye `point` is, or `None` when it is behind the camera or outside
    /// the frame. The depth is along `forward`, not the straight-line distance: it is the ordering
    /// the phrase "the nearest water on screen" actually wants, and it is what a projection uses.
    fn depth_if_visible(&self, eye: Vec3, point: Vec3) -> Option<(f32, f32)> {
        let v = point - eye;
        let depth = v.dot(self.forward);
        if depth <= 0.0 {
            return None;
        }
        let vertical = (v.dot(self.up) / depth).abs();
        let horizontal = (v.dot(self.right) / depth).abs();
        let weight = edge_fade(vertical, self.tan_v) * edge_fade(horizontal, self.tan_h);
        (weight > 0.0).then_some((depth, weight))
    }
}

#[allow(clippy::too_many_arguments)]
fn drive_reflection(
    style: Res<WaterStyle>,
    light: Res<crate::lighting::WowLighting>,
    sun_visible: Res<crate::sun::SunVisibility>,
    moon_visible: Res<crate::sun::MoonVisibility>,
    mut reflect: ResMut<WaterReflect>,
    mut data: ResMut<WaterReflectData>,
    mut images: ResMut<Assets<Image>>,
    chunks: Query<&WaterChunkInfo>,
    view_distance: Res<crate::view::ViewDistance>,
    mut held_plane: Local<Option<f32>>,
    world_cam: Query<(&GlobalTransform, &Camera, &Projection), With<WorldCamera>>,
    mut mirror: MirrorCamera,
) {
    // The sun lanes first and unconditionally: they are not the mirror's, and every path out of
    // this system below is a frame on which the water still has to draw its glitter.
    let to_sun = light.celestial_dir.normalize_or_zero();
    data.0[4..8].copy_from_slice(&[to_sun.x, to_sun.y, to_sun.z, sun_visible.0]);
    let to_moon = light.moon_dir_white.normalize_or_zero();
    // `$WOW_SCENE_SHOW` rides the moon lane's spare headroom, the way `$WOW_WAKE_SHOW` rides the
    // sim's: the visibility term is a 0..1 fraction, so anything past 1.5 is unambiguous.
    let moon_w = if std::env::var_os("WOW_SCENE_SHOW").is_some() {
        2.0
    } else {
        moon_visible.0
    };
    data.0[20..24].copy_from_slice(&[to_moon.x, to_moon.y, to_moon.z, moon_w]);
    let (zenith, horizon) = (light.sky[0], light.sky[4]);
    // `$WOW_WATER_DEPTH_SHOW` paints the water column instead of the water — see the shader.
    let show_depth = f32::from(std::env::var_os("WOW_WATER_DEPTH_SHOW").is_some());
    data.0[8..12].copy_from_slice(&[zenith[0], zenith[1], zenith[2], show_depth]);
    // `$WOW_WATER_TILT_SHOW` paints the geometric tilt instead of the water — see the shader.
    let show_tilt = f32::from(std::env::var_os("WOW_WATER_TILT_SHOW").is_some());
    data.0[12..16].copy_from_slice(&[horizon[0], horizon[1], horizon[2], show_tilt]);
    let Ok((mut mirror_tf, mut mirror_cam, mut mirror_proj, mut target)) = mirror.single_mut()
    else {
        return;
    };
    let off = |cam: &mut Camera, data: &mut WaterReflectData| {
        cam.is_active = false;
        // The mirror's lanes only — see [`WaterReflectData`].
        data.0[..4].copy_from_slice(&[0.0; 4]);
    };
    if *style != WaterStyle::Stylised || no_reflect() {
        off(&mut mirror_cam, &mut data);
        return;
    }
    let Ok((eye, camera, projection)) = world_cam.single() else {
        off(&mut mirror_cam, &mut data);
        return;
    };
    let eye_pos = eye.translation();
    let fov_y = match projection {
        Projection::Perspective(p) => p.fov,
        _ => PerspectiveProjection::default().fov,
    };
    let aspect = camera
        .physical_target_size()
        .map_or(16.0 / 9.0, |s| s.x as f32 / s.y.max(1) as f32);
    let Some((plane, lowest_visible)) = plane_near(
        eye_pos,
        eye.forward().into(),
        eye.up().into(),
        fov_y,
        aspect,
        look_reach(view_distance.farclip),
        *held_plane,
        &chunks,
    ) else {
        *held_plane = None;
        off(&mut mirror_cam, &mut data);
        return;
    };
    // Under the surface there is nothing to reflect: the sky is on the other side of it, and the
    // mirrored camera would be above the water looking down at the bed.
    if eye_pos.y <= plane {
        // Release the incumbent too. Defending a plane the eye has sunk below would have this
        // frame's rejection re-elect it next frame — a swimmer who dips under one surface would
        // hold the pass off even where another body could have served it.
        *held_plane = None;
        off(&mut mirror_cam, &mut data);
        return;
    }

    // Half the main view, and re-made when that moves (render scale moves it too). A new asset
    // rather than a resize in place, which is how the world backdrop does the same job — the
    // materials are re-pointed at it by `restamp_reflection_target`.
    let want = camera.physical_target_size().map_or(reflect.size, |s| {
        (s / reflect_downscale()).max(UVec2::splat(64))
    });
    if want != reflect.size {
        reflect.image = images.add(reflection_image(want));
        reflect.size = want;
        // Not while the debug view owns the target: it points this camera at the WINDOW, and a
        // resize would quietly take it back to the image — which is how the lever looked broken
        // the first time it was used.
        if !reflect_debug() {
            *target = RenderTarget::Image(reflect.image.clone().into());
        }
    }

    // The mirror. `looking_to` of the mirrored basis is the reflection matrix up to a negated right
    // vector; the shader's `1 − u` is that negation's exact undo (module doc).
    let mirror_v = |v: Vec3| Vec3::new(v.x, -v.y, v.z);
    let pos = Vec3::new(eye_pos.x, 2.0 * plane - eye_pos.y, eye_pos.z);
    *mirror_tf = Transform::from_translation(pos)
        .looking_to(mirror_v(eye.forward().into()), mirror_v(eye.up().into()));
    // The same lens, so the reflection lines up with the view it is sampled by screen UV against —
    // **with its near plane replaced by the water plane**.
    //
    // Without that replacement the mirrored camera, which sits under the surface looking up through
    // it, draws everything BELOW the waterline too: a moored ship's submerged hull, a pier's
    // pilings, the sea bed. Mirrored into the image and sampled by screen UV, that arrives as huge
    // hard-edged slabs standing in the water — Booty Bay, where a galleon's hull filled half the
    // bay, is its worst case, and it is why a lake looked fine while a harbour did not.
    //
    // `PerspectiveProjection::near_clip_plane` is Bevy's own oblique-clip (Lengyel 2005) in the
    // reverse-Z convention, which is exactly this problem's textbook solution. Its contract: the
    // normal is in VIEW space, unit length, pointing at the half-space to KEEP, and `w` is the
    // negative signed distance from the camera to the plane. Ours is world up (+Y) rotated into
    // view — a rotation, so it stays unit — and the camera's own height below the plane.
    let mut lens = match projection {
        Projection::Perspective(p) => p.clone(),
        // The world camera is perspective in both binaries; a non-perspective one would be a new
        // kind of view and this pass would need thinking about, not a silent fallback.
        _ => PerspectiveProjection::default(),
    };
    let rot = Mat3::from_quat(mirror_tf.rotation);
    let up_in_view = Vec3::new(rot.x_axis.y, rot.y_axis.y, rot.z_axis.y);
    // Biased a little UNDER the surface: clipped dead on the waterline, a hull and its reflection
    // meet across a hairline of missing pixels. A quarter of a yard is far below anything the
    // artefact is made of and hides the seam.
    // **The clip is dropped to the lowest water on screen, not held at the capture plane.**
    //
    // Clipping at the capture plane is right for the body that plane belongs to and ruinous for
    // every body below it: everything under the waterline is cut from the capture, so a surface
    // ten yards lower reflects its rays into parts of the image that were never drawn, reads
    // `alpha = 0`, and falls back to the sky mix. That is not a soft error — it is the whole
    // reflection missing, with a hard seam along the boundary between the two bodies. Measured at
    // the Savage Coast river mouth, where the pool at 10.07 reflected and the ocean at 0.00 beside
    // it went flat.
    //
    // Dropping the clip costs the opposite artefact — submerged geometry the capture plane's own
    // body should not show, which is the hull that filled half of Booty Bay when this pass had no
    // clip at all. So the drop is **bounded** by [`MAX_CLIP_DROP`]: enough for the water bodies a
    // real view holds at once, never enough to reach a galleon's keel.
    let drop = std::env::var("WOW_CLIP_DROP")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(MAX_CLIP_DROP);
    let clip_at = lowest_visible.clamp(plane - drop, plane);
    lens.near_clip_plane = up_in_view.extend(pos.y - (clip_at - WATERLINE_CLIP_BIAS));
    // ...and its FAR plane replaced too, which the world lens does not get to decide. See
    // [`MIRROR_FAR_YARDS`]: the mirror inherited the player's whole view distance and drew a second
    // horizon into an image that is downscaled, wobbled and mixed in at a fraction of its strength.
    // Never widened beyond what the world camera itself draws — a mirror that reaches further than
    // the view it belongs to would reflect geometry that is not streamed in.
    lens.far = lens.far.min(mirror_far());
    *mirror_proj = Projection::Perspective(lens);

    // **Every frame.**
    //
    // This alternated once, on the reasoning that a second pass over the world is the one thing that
    // can most afford to be a frame stale. It cannot, and the reason is the sampling: on a skipped
    // frame the water reads an image built for the PREVIOUS pose while sampling it with THIS
    // frame's screen UVs, so the reflected world slides by one frame of camera motion and back
    // again — a judder at half the frame rate, reported as "sometimes behind and stuttering,
    // sometimes in complete sync", which is the two parities.
    //
    // And it was not even paying: measured at the Stormwind quay (1920x1045, uncapped), the pass
    // costs 168.6 → 152.6 fps every frame and 155.2 fps on alternate frames. Halving the rate buys
    // 1.7 % and costs the judder, so there is no knob here — a choice that cheap is not a choice.
    mirror_cam.is_active = true;
    *held_plane = Some(plane);
    data.0[..4].copy_from_slice(&[plane, 1.0, REFLECT_DISTORT, PLANE_TOLERANCE]);
}

/// Re-point every liquid material at the reflection image after a rebuild. Change-gated on the
/// handle, so it is a no-op on every frame that is not a resize.
fn restamp_reflection_target(
    reflect: Res<WaterReflect>,
    mut materials: ResMut<Assets<benilla_assets::materials::LiquidMaterial>>,
    mut stamped: Local<Option<AssetId<Image>>>,
) {
    if *stamped == Some(reflect.image.id()) {
        return;
    }
    *stamped = Some(reflect.image.id());
    for (_, material) in materials.iter_mut() {
        material.extension.reflection = reflect.image.clone();
    }
}

/// Render world: one 16-byte write a frame, before anything draws.
fn upload_reflect(
    queue: Res<RenderQueue>,
    buffer: Option<Res<WaterReflectBuffer>>,
    data: Option<Res<WaterReflectData>>,
) {
    let (Some(buffer), Some(data)) = (buffer, data) else {
        return;
    };
    queue.write_buffer(&buffer.0, 0, bytemuck::cast_slice(&data.0));
}

#[cfg(test)]
mod tests {
    use super::super::query::LiquidSource;
    use super::*;
    use benilla_formats::LiquidKind;

    /// A 3x3-vertex grid at height `z` with exactly ONE wet cell — the far corner one — so three
    /// quarters of its footprint is dry ground inside a wet surface's box, which is the shape a
    /// WMO pool has around a bridge or a quay.
    fn holed_chunk(z: f32) -> WaterChunkInfo {
        let positions = (0..3)
            .flat_map(|j| (0..3).map(move |i| [10.0 * i as f32, 10.0 * j as f32, z]))
            .collect();
        WaterChunkInfo::new(
            LiquidSource::AdtChunk,
            LiquidKind::Still,
            [3, 3],
            positions,
            vec![false, false, false, true],
        )
    }

    /// **The reported bug** (the Stormwind screencast): standing over a DRY cell of a pool whose
    /// box you are inside must not read as "no water here". `surface_z_at` answers wet/dry at a
    /// point, and taking its `None` for the surface's answer left the canals reflecting from some
    /// positions and not from others a few yards away.
    #[test]
    fn a_dry_point_inside_a_pools_footprint_still_finds_its_water() {
        let chunk = holed_chunk(42.0);
        // The dry corner of the same footprint: the point test says nothing is here …
        assert!(chunk.surface_z_at(1.0, 1.0).is_none());
        // … and the surface's own answer is still its water, with the distance to it.
        let (z, d2, _) = wet_height_near(&chunk, 1.0, 1.0, true).expect("the wet cell is found");
        assert!(
            (z - 42.0).abs() < 1e-3,
            "the pool's height, not a guess: {z}"
        );
        assert!(d2 > 0.0, "the sample is somewhere else on the surface");
        // A point over the wet cell answers directly, at no distance.
        let (z, d2, _) = wet_height_near(&chunk, 15.0, 15.0, false).expect("wet under the point");
        assert!((z - 42.0).abs() < 1e-3);
        assert!(d2 < 1e-6, "the near point itself was wet: {d2}");
        // Note this surface never needs the lattice: `xy_bounds` is the bounding box of the WET
        // cells, so clamping a point outside the water into it already lands on water. What the
        // lattice is for is the box whose own nearest corner is dry — see [`elled_chunk`].
    }

    /// A wet region that is not convex: an L, so the bounding box of its wet cells has a DRY corner
    /// in it. This is the only shape that reaches the lattice, and the shape a canal bend has.
    fn elled_chunk(z: f32) -> WaterChunkInfo {
        let positions = (0..3)
            .flat_map(|j| (0..3).map(move |i| [10.0 * i as f32, 10.0 * j as f32, z]))
            .collect();
        WaterChunkInfo::new(
            LiquidSource::AdtChunk,
            LiquidKind::Still,
            [3, 3],
            positions,
            // Cells are `j * (cols - 1) + i`: the two off-diagonal ones, so the box spans the whole
            // grid while the corner the query sits over is dry.
            vec![false, true, true, false],
        )
    }

    /// The lattice is what the far surfaces skip, and skipping it is the only difference.
    ///
    /// The walk reaches out to [`look_reach`] now, several times the surfaces it used to
    /// visit, and 25 grid probes on every one of them once a frame is not what that reach is for.
    /// Near, the L still answers with its water; far, it declines rather than pays — and the cheap
    /// clamped point test is untouched at both distances, which is what keeps every convex surface
    /// (all but this shape) answering exactly as it did.
    #[test]
    fn only_the_far_surfaces_skip_the_lattice() {
        let chunk = elled_chunk(7.0);
        // The corner of the box the query sits over really is dry, or this proves nothing.
        assert!(
            chunk.surface_z_at(1.0, 1.0).is_none(),
            "the fixture's corner is dry"
        );
        let (z, d2, _) = wet_height_near(&chunk, 1.0, 1.0, true).expect("the lattice finds the L");
        assert!((z - 7.0).abs() < 1e-3, "the L's height: {z}");
        assert!(d2 > 0.0, "the water is somewhere else on the surface");
        assert!(
            wet_height_near(&chunk, 1.0, 1.0, false).is_none(),
            "without the lattice a dry corner is simply not a candidate"
        );
    }

    /// **The ocean regression, as arithmetic.** A camera 12 yd above sea level, 400 yd up the
    /// beach, pitched 3 deg down: the sea is plainly in the upper half of its frame, and the old
    /// centre-ray test could not see it because the middle of the screen was sand.
    ///
    /// The numbers are the measured ones from the Savage Coast sweep, and the depression angle to
    /// the water — about 1.7 deg — is why the ray fan that replaced the single ray failed too: at
    /// seven rays over a 46 deg frame the gaps are 7.7 deg wide.
    #[test]
    fn the_sea_up_the_beach_is_in_frame_even_when_the_centre_ray_misses() {
        // Bevy: −Z is WoW +x, −X is WoW +y. Facing WoW +y (out to sea) is Bevy −X.
        let forward = Vec3::new(-1.0, -0.052, 0.0).normalize(); // ~3 deg down
        let frame = Frame::new(forward, Vec3::Y, 0.8, 16.0 / 9.0);
        let eye = Vec3::new(0.0, 12.0, 0.0);

        // The sea, 400 yd ahead at sea level: 1.7 deg below the horizon, well inside a 23 deg
        // half-frame, so it is on screen and the test must say so.
        let sea = Vec3::new(-400.0, 0.0, 0.0);
        let (depth, weight) = frame
            .depth_if_visible(eye, sea)
            .expect("the sea is in the upper half of the frame");
        assert!(
            weight > 0.9,
            "well inside the frame, so barely faded: {weight}"
        );
        assert!(
            (depth - 400.0).abs() < 5.0,
            "depth should be the distance ahead: {depth}"
        );

        // The centre ray, for contrast: it crosses sea level at 12/tan(3 deg) ≈ 229 yd — dry sand,
        // 170 yd short of the water. This is the whole bug in one number.
        let centre_hit = 12.0 / (3.0_f32.to_radians()).tan();
        assert!(
            centre_hit < 300.0,
            "the centre ray lands on the beach, not the sea: {centre_hit}"
        );
    }

    /// **The Savage Coast ranking, with the measured geometry.** A single pool at the roadside
    /// against the ocean 400 yd out: the ocean is further and each of its chunks is no bigger, but
    /// there are a hundred of them and they are all one height, so it covers the screen and must
    /// win. Ranking by nearest-in-frame instead is what returned the pool's 10.07 at every distance
    /// from 100 to 700 yd.
    #[test]
    fn the_ocean_outranks_a_roadside_pool_it_is_further_away_than() {
        let mut seen = Prominence::default();
        // One pool chunk, 60x60 yd of it, 150 yd off.
        seen.add(10.07, 3600.0 / (150.0 * 150.0));
        // The sea: ADT chunks are 33.3 yd square, and a farclip's worth of them is in frame.
        for i in 0..100 {
            let depth = 400.0 + i as f32 * 3.0;
            seen.add(0.0, 1109.0 / (depth * depth));
        }
        assert_eq!(
            seen.best(),
            Some(0.0),
            "the sea covers more screen than the pool"
        );
    }

    /// …and the same arithmetic the other way: standing AT a pond, it is what you are looking at,
    /// however much ocean is on the horizon behind it.
    #[test]
    fn a_pond_you_are_standing_at_outranks_a_distant_sea() {
        let mut seen = Prominence::default();
        seen.add(10.07, 3600.0 / (12.0 * 12.0));
        for i in 0..100 {
            let depth = 600.0 + i as f32 * 1.0;
            seen.add(0.0, 1109.0 / (depth * depth));
        }
        assert_eq!(seen.best(), Some(10.07));
    }

    /// One body is many surfaces at one height, so they must ADD rather than compete — that is the
    /// whole reason the score is bucketed. Heights further apart than the bucket stay separate.
    #[test]
    fn one_water_body_accumulates_across_its_chunks() {
        let mut seen = Prominence::default();
        for _ in 0..10 {
            seen.add(57.63, 1.0);
        }
        seen.add(57.9, 1.0); // within BUCKET_YARDS — the same body, folded in
        seen.add(70.0, 5.0); // a different body, on its own
        assert_eq!(seen.used, 2, "two bodies, not twelve surfaces");
        assert_eq!(seen.best(), Some(57.63), "11.0 of river beats 5.0 of pond");
    }

    /// Overflow folds into the nearest height instead of dropping the surface — a view with more
    /// water heights than slots must still answer, just less precisely.
    #[test]
    fn more_heights_than_slots_still_answers() {
        let mut seen = Prominence::default();
        for i in 0..(PROMINENCE_SLOTS + 4) {
            seen.add(i as f32 * 10.0, 1.0);
        }
        assert_eq!(seen.used, PROMINENCE_SLOTS);
        assert!(
            seen.best().is_some(),
            "an overfull view still has an answer"
        );
    }

    /// **The switch, as arithmetic.** The measured Savage Coast numbers: the pool outscores the
    /// ocean seven-fold while it is in frame, so it wins outright — and when the camera turns and
    /// its score decays, the incumbent holds until the ocean is clearly better, instead of the two
    /// trading the capture the instant they cross.
    #[test]
    fn the_incumbent_holds_until_a_challenger_is_clearly_better() {
        let mut seen = Prominence::default();
        seen.add(10.069875, 23.4); // the pool, in frame
        seen.add(0.0, 3.9); // the ocean behind it
        assert_eq!(
            seen.best_sticky(None),
            Some(10.069875),
            "no incumbent: the top score"
        );
        assert_eq!(
            seen.best_sticky(Some(10.069875)),
            Some(10.069875),
            "it defends its own"
        );

        // The camera turns: the pool fades toward the frame edge and the ocean gains. A near-tie
        // must NOT flip — that is the twitch.
        let mut turning = Prominence::default();
        turning.add(10.069875, 4.2);
        turning.add(0.0, 4.8);
        assert_eq!(
            turning.best_sticky(Some(10.069875)),
            Some(10.069875),
            "a 14% lead is not enough to move the capture"
        );

        // Further round, the ocean is unambiguously the subject and the handover happens.
        let mut past = Prominence::default();
        past.add(10.069875, 2.0);
        past.add(0.0, 9.0);
        assert_eq!(
            past.best_sticky(Some(10.069875)),
            Some(0.0),
            "4.5x clears the bar"
        );
    }

    /// A plane that has left the frame entirely cannot defend itself — otherwise turning your back
    /// on a pool would keep the whole scene captured for it.
    #[test]
    fn an_incumbent_that_left_the_frame_gives_way_at_once() {
        let mut seen = Prominence::default();
        seen.add(0.0, 2.1); // only the ocean is left in frame
        assert_eq!(seen.best_sticky(Some(10.069875)), Some(0.0));
    }

    /// The contribution fades across the frame edge rather than switching off, which is what makes
    /// the score continuous enough for hysteresis to have anything to hold.
    #[test]
    fn a_surface_fades_out_of_frame_instead_of_vanishing() {
        let half = 0.4_f32;
        assert_eq!(edge_fade(0.0, half), 1.0, "dead centre");
        assert_eq!(
            edge_fade(half, half),
            1.0,
            "the frustum edge itself is still full"
        );
        let mid = edge_fade(half * 1.17, half);
        assert!(
            mid > 0.2 && mid < 0.8,
            "partway into the slop it is partial: {mid}"
        );
        assert_eq!(
            edge_fade(half * FRAME_SLOP, half),
            0.0,
            "the far side of the slop"
        );
        assert_eq!(edge_fade(half * 5.0, half), 0.0, "and it stays there");
    }

    /// The frame has sides, and things behind the camera are not in it.
    #[test]
    fn the_frame_rejects_what_is_behind_it_and_off_its_edges() {
        let frame = Frame::new(Vec3::new(0.0, 0.0, -1.0), Vec3::Y, 0.8, 16.0 / 9.0);
        let eye = Vec3::ZERO;
        assert!(
            frame
                .depth_if_visible(eye, Vec3::new(0.0, 0.0, 100.0))
                .is_none(),
            "a pool behind the camera is not what the player is looking at"
        );
        assert!(
            frame
                .depth_if_visible(eye, Vec3::new(0.0, 0.0, -100.0))
                .is_some(),
            "dead ahead is in frame"
        );
        // Straight up, and far off to the side, are both out.
        assert!(frame
            .depth_if_visible(eye, Vec3::new(0.0, 100.0, -1.0))
            .is_none());
        assert!(frame
            .depth_if_visible(eye, Vec3::new(500.0, 0.0, -1.0))
            .is_none());
    }

    /// The edge slop is real but bounded — a surface a little outside the frustum still counts
    /// (see [`FRAME_SLOP`]), one far outside does not.
    #[test]
    fn the_frame_is_forgiving_at_its_edge_but_not_boundless() {
        let frame = Frame::new(Vec3::new(0.0, 0.0, -1.0), Vec3::Y, 0.8, 16.0 / 9.0);
        let eye = Vec3::ZERO;
        let tan_v = (0.8_f32 * 0.5).tan();
        let at = |up: f32| Vec3::new(0.0, up * 100.0, -100.0);
        assert!(
            frame.depth_if_visible(eye, at(tan_v * 1.2)).is_some(),
            "just past the edge still reads as looked-at"
        );
        assert!(
            frame.depth_if_visible(eye, at(tan_v * 2.0)).is_none(),
            "well past it does not"
        );
    }

    /// The mirrored pose: the eye reflects through the plane, and the basis reflects with it. The
    /// handedness flip this leaves behind is the shader's `1 − u`, and is asserted there in prose
    /// rather than here — what this pins is that the *pose* is a reflection and not a rotation.
    #[test]
    fn the_mirrored_pose_reflects_through_the_plane() {
        let plane = 12.0;
        let eye =
            Transform::from_xyz(3.0, 20.0, -7.0).looking_to(Vec3::new(0.0, -1.0, 1.0), Vec3::Y);
        let mirror_v = |v: Vec3| Vec3::new(v.x, -v.y, v.z);
        let pos = Vec3::new(
            eye.translation.x,
            2.0 * plane - eye.translation.y,
            eye.translation.z,
        );
        let mirrored = Transform::from_translation(pos)
            .looking_to(mirror_v(eye.forward().into()), mirror_v(eye.up().into()));
        // As far below the plane as the eye is above it.
        assert!((mirrored.translation.y - 4.0).abs() < 1e-5);
        // Looking as far up as the eye looks down.
        let f: Vec3 = eye.forward().into();
        let mf: Vec3 = mirrored.forward().into();
        assert!((mf.y + f.y).abs() < 1e-5, "{f:?} vs {mf:?}");
        assert!((mf.x - f.x).abs() < 1e-5 && (mf.z - f.z).abs() < 1e-5);
    }
}
