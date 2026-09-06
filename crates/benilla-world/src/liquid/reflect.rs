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
use benilla_assets::coords::bevy_to_wow;

/// How far from the eye a water surface may be and still be worth a reflection pass, in yards.
/// Generous, because the subject is usually the lake you are *looking at* rather than the puddle
/// you are standing in — but finite, because the pass is a second world draw and a player crossing
/// a continent should not pay for it.
const REFLECT_RADIUS: f32 = 120.0;

/// Fragments whose surface sits further than this from the plane the image was rendered for keep
/// the sky mix instead of the reflection, in yards.
///
/// Three yards rather than one: a liquid grid is a heightfield, not a plane — Felwood's river drops
/// about two yards across a single MCNK — so a tolerance tight enough to be exact would cut the
/// reflection off halfway down a river. What a few yards of plane error costs is a little parallax
/// in an image already displaced by the ripple normal; what it buys is a reflection that covers the
/// whole surface you are looking at.
///
/// One capture can only be right for one plane, and a world of streams and terraced pools has many:
/// Elwynn's river is one height, the pond above it another. Rather than pick a compromise plane and
/// be subtly wrong on both, the far surface simply does not take the reflection — which reads as
/// calmer water, not as an error.
const PLANE_TOLERANCE: f32 = 3.0;

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
/// will do, and the nearest one is the best reading of "the water beside me". It runs only for the
/// surfaces that pass the radius reject, and only when the cheap point missed.
fn wet_height_near(chunk: &WaterChunkInfo, x: f32, y: f32) -> Option<(f32, f32)> {
    let [[min_x, min_y], [max_x, max_y]] = chunk.xy_bounds()?;
    let (cx, cy) = (x.clamp(min_x, max_x), y.clamp(min_y, max_y));
    let d2 = |px: f32, py: f32| (px - x) * (px - x) + (py - y) * (py - y);
    if let Some(z) = chunk.surface_z_at(cx, cy) {
        return Some((z, d2(cx, cy)));
    }
    const N: i32 = 4;
    let mut best: Option<(f32, f32)> = None;
    for i in 0..=N {
        for j in 0..=N {
            let px = min_x + (max_x - min_x) * i as f32 / N as f32;
            let py = min_y + (max_y - min_y) * j as f32 / N as f32;
            let Some(z) = chunk.surface_z_at(px, py) else {
                continue;
            };
            let d = d2(px, py);
            if best.is_none_or(|(bd, _)| d < bd) {
                best = Some((d, z));
            }
        }
    }
    best.map(|(d, z)| (z, d))
}

/// The water plane to mirror through: **the surface the camera is looking at**, and only failing
/// that the nearest one within [`REFLECT_RADIUS`].
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
fn plane_near(eye: Vec3, forward: Vec3, chunks: &Query<&WaterChunkInfo>) -> Option<f32> {
    let [x, y, _] = bevy_to_wow(eye);
    let mut looked_at: Option<(f32, f32)> = None; // (distance along the ray, plane)
    let mut nearest: Option<(f32, f32)> = None; // (distance² to a wet sample, plane)
    for chunk in chunks {
        // The cheap reject first, on the footprint alone: everything below samples the grid.
        let Some([[min_x, min_y], [max_x, max_y]]) = chunk.xy_bounds() else {
            continue;
        };
        let (bx, by) = (x.clamp(min_x, max_x), y.clamp(min_y, max_y));
        if (bx - x) * (bx - x) + (by - y) * (by - y) > REFLECT_RADIUS * REFLECT_RADIUS {
            continue;
        }
        let Some((z, d2)) = wet_height_near(chunk, x, y) else {
            continue; // a footprint with no wet cell in reach of the lattice
        };
        if nearest.is_none_or(|(bd2, _)| d2 < bd2) {
            nearest = Some((d2, z));
        }
        // Where the view ray crosses that height, if it crosses it ahead of the eye at all.
        if forward.y >= -1e-4 || eye.y <= z {
            continue;
        }
        let t = (eye.y - z) / -forward.y;
        if t <= 0.0 || t > REFLECT_RADIUS || looked_at.is_some_and(|(bt, _)| bt <= t) {
            continue;
        }
        let [hx, hy, _] = bevy_to_wow(eye + forward * t);
        if let Some(hit_z) = chunk.surface_z_at(hx, hy) {
            looked_at = Some((t, hit_z));
        }
    }
    looked_at.or(nearest).map(|(_, z)| z)
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
    world_cam: Query<(&GlobalTransform, &Camera, &Projection), With<WorldCamera>>,
    mut mirror: MirrorCamera,
) {
    // The sun lanes first and unconditionally: they are not the mirror's, and every path out of
    // this system below is a frame on which the water still has to draw its glitter.
    let to_sun = light.celestial_dir.normalize_or_zero();
    data.0[4..8].copy_from_slice(&[to_sun.x, to_sun.y, to_sun.z, sun_visible.0]);
    let to_moon = light.moon_dir_white.normalize_or_zero();
    data.0[20..24].copy_from_slice(&[to_moon.x, to_moon.y, to_moon.z, moon_visible.0]);
    let (zenith, horizon) = (light.sky[0], light.sky[4]);
    // `$WOW_WATER_DEPTH_SHOW` paints the water column instead of the water — see the shader.
    let show_depth = f32::from(std::env::var_os("WOW_WATER_DEPTH_SHOW").is_some());
    data.0[8..12].copy_from_slice(&[zenith[0], zenith[1], zenith[2], show_depth]);
    data.0[12..16].copy_from_slice(&[horizon[0], horizon[1], horizon[2], 0.0]);
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
    let Some(plane) = plane_near(eye_pos, eye.forward().into(), &chunks) else {
        off(&mut mirror_cam, &mut data);
        return;
    };
    // Under the surface there is nothing to reflect: the sky is on the other side of it, and the
    // mirrored camera would be above the water looking down at the bed.
    if eye_pos.y <= plane {
        off(&mut mirror_cam, &mut data);
        return;
    }

    // Half the main view, and re-made when that moves (render scale moves it too). A new asset
    // rather than a resize in place, which is how the world backdrop does the same job — the
    // materials are re-pointed at it by `restamp_reflection_target`.
    let want = camera
        .physical_target_size()
        .map_or(reflect.size, |s| (s / 2).max(UVec2::splat(64)));
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
    lens.near_clip_plane = up_in_view.extend(pos.y - (plane - WATERLINE_CLIP_BIAS));
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
        let (z, d2) = wet_height_near(&chunk, 1.0, 1.0).expect("the wet cell is found");
        assert!(
            (z - 42.0).abs() < 1e-3,
            "the pool's height, not a guess: {z}"
        );
        assert!(d2 > 0.0, "the sample is somewhere else on the surface");
        // A point over the wet cell answers directly, at no distance.
        let (z, d2) = wet_height_near(&chunk, 15.0, 15.0).expect("wet under the point");
        assert!((z - 42.0).abs() < 1e-3);
        assert!(d2 < 1e-6, "the near point itself was wet: {d2}");
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
