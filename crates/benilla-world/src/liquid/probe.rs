//! The water's **cubemap reflection probe** — the fourth reflection tier.
//!
//! ## What it is for, and what the other three cannot do
//!
//! The march is right at any surface orientation and blind to everything off the screen. The planar
//! capture sees off-screen and behind the eye and is right for exactly one plane at one orientation.
//! The sky mix is always available and always plausible and knows nothing about the world. What is
//! missing between them is **off-screen content at the correct orientation**: the headland behind
//! you, the trees past the frame edge, reflected on water the march cannot reach and at an angle the
//! plane gets wrong. That is this tier's whole job.
//!
//! ## One probe, anchored to the WATER
//!
//! A probe captured at the mirror of the eye is arithmetically the planar capture — it needs no
//! parallax correction and it serves exactly one water height, so it re-inherits the single-plane
//! limitation that probes exist to escape. Anchoring to the water instead is what buys anything, and
//! it buys a parallax error in exchange, which is what [`probe_dir`]'s sphere proxy is for.
//!
//! **This is the second time a probe tier has been built here**, and the first one was measured and
//! removed. What that work established, and what this design is shaped by:
//!
//! * One probe per body closed **54-59%** of the gap to the planar capture on rivers and terraces,
//!   and **1%** on open coast. Enclosed water is the favourable case; open water is not, and if this
//!   measures like open coast at Mirror Lake then it has not earned its slot and should go again.
//! * Nothing ever demonstrated a *second* probe earning its slot — one and eight measured
//!   identically once a coverage gate was added. So this starts at one, deliberately.
//! * **The hard problem is not where probes go, it is which probe a fragment reads.** A metric
//!   selector hands a fragment the cube of a pond behind a bank, and the failure is invisible in a
//!   screenshot because a cube of somewhere else is still a plausible picture. With one probe that
//!   problem does not exist — which is most of why starting at one is right.
//!
//! ## The falloff is three-dimensional, and that is not fussiness
//!
//! A 220-yard box around this lake holds **three** separate water bodies: the lake at y = 48.64, a
//! second at 55.9, and a stream running 108 to 130 with slopes of up to 12 yards across one chunk.
//! A fragment sixty yards above the lake reading the lake's cubemap gets a picture of somewhere
//! else — the "cube of a pond behind a bank" failure, reproduced by a single probe with no selector
//! in sight. So the weight falls off with **height as well as distance** ([`PROBE_FADE_UP`]).
//!
//! ## What is Bevy's and what is ours
//!
//! Bevy 0.18 ships the *filtering* half: `EnvironmentMapGenerationPlugin` with its downsample chain
//! and `environment_filter.wgsl` turn a runtime cubemap into irradiance and prefiltered specular
//! mips. It does not ship a scene-to-cubemap **capture**, and there is no parallax correction
//! anywhere in its light-probe shaders — its environment maps are sampled at infinite distance. So
//! the capture is ours and the correction is ours, and only the filtering is worth borrowing.
//!
//! The capture is six cameras into six 2D images plus one node copying them into the cube's layers,
//! rather than six cameras aimed at cube faces directly, because `ImageRenderTarget` carries a
//! handle and a scale factor and no layer — a camera cannot address one face. The copy is the same
//! `copy_texture_to_texture` shape `scene_color` already uses.

use bevy::asset::RenderAssetUsages;
use bevy::core_pipeline::core_3d::graph::{Core3d, Node3d};
use bevy::core_pipeline::FullscreenShader;
use bevy::gizmos::config::{GizmoConfigGroup, GizmoConfigStore};
use bevy::prelude::*;
use bevy::reflect::Reflect;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_graph::{
    Node, NodeRunError, RenderGraphContext, RenderGraphExt, RenderLabel,
};
use bevy::render::render_resource::binding_types::{sampler, texture_2d, texture_depth_2d};
use bevy::render::render_resource::{
    BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries, CachedRenderPipelineId,
    ColorTargetState, ColorWrites, Extent3d, FilterMode, FragmentState, Operations, PipelineCache,
    RenderPassColorAttachment, RenderPassDescriptor, RenderPipelineDescriptor, SamplerBindingType,
    SamplerDescriptor, ShaderStages, TextureDimension, TextureFormat, TextureSampleType,
    TextureUsages, TextureViewDescriptor, TextureViewDimension,
};
use bevy::render::renderer::{RenderContext, RenderDevice};
use bevy::render::texture::GpuImage;

use super::WaterStyle;

/// Where the one probe stands, in **Bevy world yards**.
///
/// Mirror Lake, Elwynn Forest — raw WoW `(-9385, 420, 48.64)` through
/// `benilla_assets::coords::wow_to_bevy`, which is `(-y, z, -x)`. The position was not guessed: the
/// MCLQ census (`benilla-formats --example water_here`) reports this point ringed by chunks that are
/// 64/64 wet at a dead-flat surface height of 48.640, which is open water rather than a bank.
///
/// It is a constant because there is exactly one probe. The day there are two, this becomes a
/// component on a probe entity and the shader gains a selector — and the note above about which
/// probe a fragment reads becomes the whole problem again.
const PROBE_AT: Vec3 = Vec3::new(-420.0, 48.64, 9385.0);

/// How far ABOVE the water the cube is captured from, in yards.
///
/// **A probe standing exactly on the surface puts its own equator at the waterline**, and that is
/// the wrong place for it. A reflected ray at a grazing angle leaves the surface a few degrees above
/// horizontal, so it asks the cube for directions just above the equator — and at the waterline what
/// stands there is the bank's dirt, a few yards away. The treeline and the buildings, which are what
/// a reflection of a wooded shore should actually contain, sit HIGHER in the cube than those rays
/// ever reach. The first capture did exactly this and the water came back browner and flatter than
/// the march alone had left it.
///
/// **It is zero, and the reasoning above is wrong** — kept because the mistake is instructive. Box
/// projection returns `d * scalar + (p - centre)`, so the offset between the fragment and the
/// capture point enters the lookup directly. A probe twelve yards above the water gives every
/// surface fragment a twelve-yard DOWNWARD bias, and a shallow ray that lands on the box wall only
/// ten yards up then points below the probe's horizon — where the cube holds bank and lakebed. The
/// lift that was supposed to bring the treeline into view pushed the lookup under it instead.
///
/// For a PLANAR reflector the probe belongs on the plane, so zero was tried — and zero is worse,
/// for a reason neither the optics nor the projection could have predicted. **A capture point
/// exactly on the water plane is UNDERWATER as far as the client is concerned**, and the submerged
/// treatment then fogs the whole cube: the distance dissolves into green haze, near trees survive
/// as silhouettes standing in nothing, and buildings a little further off vanish entirely. Every one
/// of those was read off the unwrapped cube and blamed on something else first — a broken face, a
/// missing cull, sky bleeding through a face seam.
///
/// So the lift is small and load-bearing: enough to clear the waterline and the submerged test, few
/// enough yards that the `(p - centre)` term it adds to the box lookup stays a minor bias rather
/// than the twelve-yard one that pushed shallow rays under the horizon. Four is the compromise, and
/// the two failure modes it sits between are both real and both were observed.
const PROBE_LIFT: f32 = 4.0;

/// How far PAST the shoreline the projection box reaches, in yards.
///
/// **The box must bound the SCENERY, not the water**, and getting that wrong is subtle enough to
/// survive a working implementation. Box projection assumes the environment lies on the box
/// surface: every reflected ray is told the world stands where the wall is. Walls derived from the
/// liquid footprint alone sit exactly at the waterline — true of the bank, false of the treeline a
/// few yards behind it and the buildings behind that — so the entire scene collapses onto the
/// shoreline plane and the reflection compresses into a thin band of bank colour. Against a true
/// mirrored render the difference is stark: the mirror shows trunks descending at their real
/// distance and the house below the house, while the unexpanded box showed orange bank and almost
/// nothing else.
///
/// Unreal states the rule as *place the shape so the geometry you want reflected sits just inside
/// it*, and the trap is that "the geometry you want reflected" is not the surface you are
/// reflecting it in. Thirty yards puts the near treeline and the lakeside buildings inside the box;
/// what stands further out than that is far enough away that the parallax error is small anyway.
const PROBE_BOX_OUT: f32 = 30.0;

/// How far above the water the projection box reaches, in yards — its ceiling.
///
/// **The box is not a region of influence. It is a stand-in for where the reflected scenery
/// actually stands**, and that distinction is the whole technique. Unity calls this box projection
/// and Unreal sizes its capture shapes by the same rule: *place them so the part of the level you
/// want reflected sits just inside the shape, because the level is reprojected onto it*. A shape
/// that does not match the geometry reflects the wrong thing at the wrong size, and the failure is
/// smooth and plausible rather than obviously broken.
///
/// A sphere was tried first and is the wrong shape for a lake. A lake is a floor with walls: the
/// bank stands at the shoreline in every direction and the sky is overhead. A box has exactly that
/// form — shallow rays meet a WALL and find the bank and its trees, steep rays meet the CEILING and
/// find sky. A sphere curves away uniformly, so at any radius large enough to clear the near bank
/// it also arcs over the far treeline, and every lookup lands in open sky. That is precisely what
/// happened: at 110 yards and again at 45, the probe returned a soft blue wash and the lane looked
/// identical to the one without it.
///
/// The walls come from the water body itself ([`super::query::WaterChunkInfo`]'s footprint), so
/// they sit on the shoreline by construction rather than by tuning. Only the ceiling is authored,
/// because no liquid footprint knows how tall the trees are.
const PROBE_BOX_UP: f32 = 55.0;

/// How far below the surface the projection box reaches, in yards — its floor.
///
/// Shallow, because water reflects UP: the only rays that ever meet this face are near-vertical
/// ones from a fragment almost directly above the probe, and what is under a lake is its bed.
const PROBE_BOX_DOWN: f32 = 8.0;

/// `$WOW_PROBE_SPHERE=<yards>` — intersect the reflected ray with a SPHERE of this radius about the
/// probe instead of with the scenery box. `0` (the default) keeps the box.
///
/// **The perspective this builds is "from inside the sphere".** The reflected ray leaves the water,
/// meets the sphere's inner surface, and the cube is read in the direction from the probe's centre
/// to that meeting point — so the environment is treated as painted on a shell at one finite
/// distance, and the water shows the inside of that shell. Against a box the difference is what the
/// environment is assumed to sit on: a box has flat walls at different distances per bearing, a
/// sphere one distance in all of them.
///
/// It is a lever rather than a replacement because the sphere was tried first, at 110 yards and
/// again at 45, and abandoned on evidence that turned out to be worthless — the cube it was reading
/// had its faces upside down and no trees in it at all. Neither result said anything about the
/// sphere. With a correct cube the comparison is finally meaningful, and a lake ringed at a roughly
/// even distance is exactly the case a single radius should suit.
/// `$WOW_PROBE_RAW=1` — draw the probe lane as a flat reflection with no Fresnel, body or tint.
///
/// The diagnostic that made the tier debuggable at all; see [`probe_lanes`] for why it was the
/// default while the cube was wrong, and why it no longer is.
/// `$WOW_WATER_RIPPLE=<0..1>` — how much wave the surface has at all. 1 is the real look, 0 is
/// dead flat.
///
/// Flat water is the only honest way to compare two REFLECTIONS. With waves in, the planar tier's
/// ripple is a small bounded offset into an image that is already correct, while the cube's goes
/// through `reflect()` — violently sensitive to the normal at a grazing eye — so the same tilt asks
/// the two tiers completely different questions. Flattening removes the wave from both at once and
/// leaves only what each tier believes is behind the water.
/// `$WOW_PROBE_NORMAL=<0..1>` — how much of the water's ripple reaches the CUBE lookup.
///
/// **Separate from the water's actual ripple, and it has to be.** The cube is sampled through
/// `reflect()`, which at a grazing eye swings across a huge arc for a small tilt, and the cube has
/// no mip chain — so a rippled normal scatters the lookup and returns speckle where the planar tier,
/// whose ripple is a bounded offset into an already-correct image, returns a gentle wobble. Measured
/// against the planar mirror at Mirror Lake: the probe matches it +0.526 on flat water and only
/// +0.260 with the ripple in.
///
/// **This is an approximation, and the real fix is a prefiltered cube.** The standard answer is a
/// mip chain convolved for increasing roughness (a PMREM), sampled at the mip the surface's
/// roughness asks for, so a perturbed normal averages a wide lobe instead of point-sampling a sharp
/// one. Damping the normal buys the same stability for five lines and costs some of the ripple's
/// motion in the reflection; it is a stand-in until the cube carries mips.
pub(crate) fn probe_normal() -> f32 {
    static N: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("WOW_PROBE_NORMAL")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .map(|v| v.clamp(0.0, 1.0))
            // 0.15 measured best against the planar mirror at Mirror Lake: 1.0 scores +0.260,
            // 0.5 +0.407, 0.25 +0.493, 0.15 +0.522, and flat 0.0 +0.515 — so a little ripple is
            // better than none, and a lot is far worse than a little.
            .unwrap_or(0.15)
    })
}

/// `$WOW_PROBE_BOX_SCALE=<f>` — scale the derived proxy box about the probe.
///
/// **The proxy's walls have to sit where the reflected geometry actually is.** That is Unreal's
/// sizing rule and Lagarde's: the scene is reprojected onto the shape, so a wall at the wrong
/// distance rescales everything behind it. Measured against the planar mirror at Mirror Lake, the
/// probe's reflection of the cottage came back **1.23x too large** — proportions right, scale
/// wrong, which is the signature of a proxy at the wrong remove rather than of a broken projection.
///
/// The box is derived from the WATER's footprint plus [`PROBE_BOX_OUT`], and the treeline it needs
/// to represent stands at the shoreline, not out where that puts the walls.
/// How far out the proxy box gathers water chunks, in yards — **not** [`PROBE_FADE_YD`].
///
/// These are two different questions and they were sharing one number. The fade radius asks *how
/// far this probe's influence reaches*; the box radius asks *where the geometry it reflects stands*.
///
/// Unreal's sizing rule is the one to follow: **the shape must contain the geometry you want
/// reflected**, because the scene is reprojected onto it. A wall that stands closer than the real
/// geometry does not merely blur the reflection, it substitutes a different part of the cube: a
/// grazing ray from far water travels a long way almost horizontally, and if it meets the proxy
/// wall while still only a few yards above the surface, the direction that comes back points at the
/// BANK rather than at the treeline standing behind it.
///
/// **60 yards, measured — and it used to be 28 for a reason that turned out to be wrong.** 28 was
/// chosen when there was one probe, by sweeping the box until the reflection matched the planar
/// mirror's SIZE at one camera. That made the box a magnification trim rather than a proxy, and it
/// left the walls well inside the treeline; the visible result was a broad orange band across the
/// far water, the bank showing where the canopy belonged. With probes on a lattice the density
/// bounds the magnification error, so the box is free to do its actual job.
///
/// The measurement is the far-water strip at the Mirror Lake pin, scored as red-minus-blue against
/// the planar mirror's `+39.7`: reach 28 gives `+70.5` (the orange band), 40 `+47.8`, 50 `+46.9`,
/// 55 `+41.7`, **60 `+39.9`**, 65 `+38.6`, 80 `+37.8` — crossing the mirror between 60 and 65 and
/// flat thereafter. Agreement with the mirror over the whole surface more than doubled at the same
/// time, from `+0.258` to `+0.538`.
///
/// `$WOW_PROBE_BOX_REACH=<yards>` overrides it.
const PROBE_BOX_REACH: f32 = 60.0;

fn box_reach() -> f32 {
    static R: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *R.get_or_init(|| {
        std::env::var("WOW_PROBE_BOX_REACH")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(PROBE_BOX_REACH)
    })
}

fn box_scale() -> f32 {
    static B: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *B.get_or_init(|| {
        std::env::var("WOW_PROBE_BOX_SCALE")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .map(|v| v.clamp(0.1, 5.0))
            // **1.0 — a diagnostic lever, never a correction.**
            //
            // This defaulted to 1.15 for a while, because that made the cottage reflect at exactly
            // the planar mirror's size at the Mirror Lake pin (match +0.853 against +0.839 at 1.0).
            // The number was measured honestly and was still wrong, for a reason that no amount of
            // sweeping would have found: a box-projected cube's magnification is a function of
            // where the SHADING POINT stands relative to the capture point. The error is identically
            // zero at the capture point for any proxy at all — there `H - c` collapses to `t*d`, the
            // true reflected direction — and grows as the fragment moves away from it. So a scalar
            // fitted at one camera is guaranteed wrong at the next one, which is exactly what
            // showed up when the camera moved.
            //
            // A residual that varies with something a constant cannot see is not correctable by a
            // constant. The two honest moves are to put the proxy walls where the geometry actually
            // stands, and to shorten the distance from any fragment to its nearest probe. Both are
            // geometry; neither is a multiplier.
            .unwrap_or(1.0)
    })
}

fn ripple_scale() -> f32 {
    static N: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("WOW_WATER_RIPPLE")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .map(|v| v.clamp(0.0, 1.0))
            .unwrap_or(1.0)
    })
}

fn probe_raw() -> bool {
    static R: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *R.get_or_init(|| std::env::var("WOW_PROBE_RAW").as_deref() == Ok("1"))
}

fn probe_sphere() -> f32 {
    static R: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *R.get_or_init(|| {
        std::env::var("WOW_PROBE_SPHERE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.0)
    })
}

/// The fallback half-extent in yards, for the frames before any liquid footprint has arrived.
///
/// A box of zero size makes every ray-box factor zero and sends the whole lookup to one texel, so
/// there has to be something. This is only ever seen for the frames between boot and the first
/// water chunk streaming in.
const PROBE_BOX_FALLBACK: f32 = 60.0;

/// The near plane of every face camera, in yards.
///
/// **`shaders/probe_face.wgsl` hardcodes this same number** and cannot read it from anywhere: the
/// blit turns the face's depth buffer into a distance with `near / ndc`, which is only correct for
/// the near plane the face was actually rendered with. Bevy builds a `perspective_infinite_reverse`
/// matrix, so the projection's `far` never enters it and only `near` matters. Changing this without
/// changing the shader silently scales every stored distance.
pub(crate) const PROBE_NEAR: f32 = 0.1;

/// Face resolution of the cube, per side: `slots * 6 * px * px * 8` bytes — 150 MB for the live
/// probe's three slots, 800 MB for the lattice's sixteen.
///
/// At 256 the cube's texels read as blocks on the water at grazing angles (alpha-cut foliage
/// magnified), and the depth correction's hits step at the same texels; 512 halves them and 1024
/// halves them again.
const PROBE_FACE: u32 = 1024;

/// `$WOW_PROBE_BLUR=1` — prefilter the cube and read it softened with distance. **Off.**
///
/// The tier was built this way and Stefan turned it off: "no just don't blur the probes". The blur
/// existed to stop the parallax error being READABLE — a cube is exact at its capture point and
/// drifts away from it, and no amount of march depth, cube resolution or landing consensus changed
/// that — so softening it was meant to trade a reflection that is sharp and wrong for one that is
/// vague and unobjectionable. In practice it traded it for one that could barely be seen at all.
///
/// With this off the water reads level 0 and the cube is allocated with a single level, so
/// [`probe_mips`] is 1, the filter passes in `WaterProbeNode` have no levels to fill and cost
/// nothing, and `liquid.wgsl`'s `probe_lod` is clamped to zero by its own lanes without needing a
/// branch. Turning it on restores the chain and the distance ramp exactly.
///
/// Note the blur index is separately miscalibrated and was never fixed: it derives from
/// `d_probe / d_geom`, the parallax of a cube read as a DISTANT ENVIRONMENT, while this tier
/// reprojects against the cube's own stored depths first and removes most of that. It saturates
/// within about six yards of a probe, so with the chain on nearly all water reads the bottom level
/// — an 8x8 face, an average colour. Anyone turning this back on should expect that and fix the
/// index first.
pub(crate) fn probe_blur() -> bool {
    static B: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *B.get_or_init(|| std::env::var("WOW_PROBE_BLUR").as_deref() == Ok("1"))
}

/// How many mip levels the probe cube carries, derived from its face size.
///
/// The chain stops at eight pixels a face. Past that a level holds barely more than an average
/// colour, and the cone that produced it has already swallowed most of a hemisphere — the levels
/// below add cost and nothing the level above did not already say.
///
/// Six levels at the default 256, which is the range the water's distance bias indexes across.
fn probe_mips() -> u32 {
    if !probe_blur() {
        // One level: nothing reads a chain, so nothing builds or stores one.
        return 1;
    }
    PROBE_FACE.ilog2().saturating_sub(2).clamp(1, 6)
}

/// How far apart probes stand on the world lattice, in yards.
///
/// **This is the parameter that replaced a fudge factor.** A box-projected cube is exact only where
/// its proxy wall coincides with real geometry, and the error is identically zero at the capture
/// point — there `H - c` collapses to `t*d`, the true reflected direction, whatever the proxy is —
/// and grows as the shading point moves away from it. One probe on a lake therefore cannot be right
/// everywhere at any proxy size, which is why a magnification scalar fitted at one camera read
/// 1.00x there and wrong at the next. Shortening the distance from a fragment to its nearest probe
/// is the only lever that REDUCES that error rather than relocating it.
///
/// **12 yards, measured across four cameras — and it only became reachable once
/// [`PROBE_LIVE_RADIUS`] stopped being derived from this number.** While the two were welded,
/// asking for a denser lattice also asked each probe to reach less far, so density and coverage
/// fought and 24 was the best compromise available. Unwelded, denser wins outright: at 12 yards
/// with sixteen slots the four-camera mean is `+0.673` and the worst camera `+0.510`, against
/// `+0.492` and `+0.428` for 24 yards with eight, and the pale far band that prompted all of this
/// measures `-0.17` against the planar mirror — indistinguishable from it.
///
/// The cost is that crossing a 12-yard cell retargets slots about twice as often as a 24-yard one,
/// so [`PROBE_BURST_GAP_S`] is doing more work than it used to; it is the first thing to look at if
/// the frame-time tail regresses.
///
/// The older sweep this doc used to quote: Scored against the planar mirror, masked to the
/// water surface, mean and worst-camera agreement over the sweep:
///
/// | spacing | 24 | 32 | 48 | 64 |
/// |---|---|---|---|---|
/// | mean  | **+0.633** | +0.621 | +0.550 | +0.500 |
/// | worst | **+0.464** | +0.406 | +0.415 | +0.310 |
///
/// Monotonic, and tighter wins — which is what the error model predicts, since the farthest a
/// fragment can sit from its nearest probe is `spacing / sqrt(2)`: about 17 yards at 24, against
/// the 100-plus the single Mirror Lake pin left on the far bank.
///
/// **One camera cannot choose this number.** Swept at the pin alone the same four values came back
/// non-monotonic (24 and 64 beating 32 and 48), because the lattice is anchored to the world and
/// its phase relative to any one camera changes with the spacing. Reading a winner off that would
/// have been the single-camera fit this tier has already been burned by once — see [`box_scale`].
///
/// `$WOW_PROBE_SPACING=<yards>` overrides it, clamped to 8..=256.
const PROBE_SPACING: f32 = 12.0;

fn probe_trace() -> bool {
    static T: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *T.get_or_init(|| std::env::var("WOW_PROBE_TRACE").as_deref() == Ok("1"))
}

fn probe_spacing() -> f32 {
    static S: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *S.get_or_init(|| {
        std::env::var("WOW_PROBE_SPACING")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .map(|v| v.clamp(8.0, 256.0))
            .unwrap_or(PROBE_SPACING)
    })
}

/// How many probes are live at once. The cube array is `6 * PROBE_SLOTS` layers deep.
///
/// Eight covers a three-by-three lattice neighbourhood around the player at [`PROBE_SPACING`] with
/// one spare for the body they are walking toward. Memory is `slots * 6 * face^2 * 8` bytes — at
/// eight slots and 256-pixel faces, **25 MB**, against 50 MB for the single 1024 probe it replaces.
///
/// **Sixteen — and the reason eight once looked sufficient is instructive.** Sixteen slots measured
/// no better than eight while [`PROBE_LIVE_RADIUS`] was `spacing * 2`: the extra probes were placed
/// and then faded out at the very same radius, so they could not add coverage. With the two
/// unwelded and the lattice at 12 yards, sixteen slots are what fill the reach.
///
/// Memory is `slots * 6 * face^2 * 8` bytes — 50 MB here, which is what the single 1024-pixel probe
/// this tier began as already cost.
///
/// The superseded reading: Raise this only with evidence from a scene that needs it — a wide body seen
/// from a shore, where the water in view genuinely outruns a three-by-three neighbourhood.
///
/// `$WOW_PROBE_SLOTS=<n>` overrides it, clamped to 1..=16.
const PROBE_SLOTS: usize = 16;

fn probe_slots() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        // The live probe rotates three cubes — see [`probe_live`].
        if probe_live() {
            return 3;
        }
        std::env::var("WOW_PROBE_SLOTS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .map(|v| v.clamp(1, PROBE_SLOT_MAX))
            .unwrap_or(PROBE_SLOTS)
    })
}

/// The shader's array is fixed-size, so this is the ceiling the uniform is built for.
pub(crate) const PROBE_SLOT_MAX: usize = 16;

/// One live probe: the lattice cell it answers for, the point its cube was taken from, and its own
/// proxy box.
///
/// **The lattice is anchored to the WORLD, not to the player.** A cube is only valid for the point
/// it was taken from, so a probe that follows the player is invalidated by every step they take and
/// re-captures continuously. Snapping capture points to a fixed world grid means a probe captured
/// once stays correct for as long as the world is static, and walking merely changes which subset is
/// live. It also settles the even-distribution question by construction rather than by a placement
/// heuristic.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ProbeSlot {
    /// **Resident state — what is actually in this slot's six layers right now.** The cell the cube
    /// was taken for, the point it was taken from, and the proxy fitted to that point. `None` until
    /// the slot's first capture has landed.
    ///
    /// These four move together and only ever at the instant a capture completes, so the cube's
    /// contents and the projection applied to it can never disagree.
    cell: Option<IVec2>,
    at: Vec3,
    box_min: Vec3,
    box_max: Vec3,
    /// Whether the resident cube is valid. Read by the shader as the slot's `at.w`.
    captured: bool,
    /// **Target state — the cell this slot SHOULD hold.** Differs from the resident cell exactly
    /// while a capture is owed, and placement only ever writes here.
    ///
    /// **The separation is what stops the reflection popping when the player moves.** Reassignment
    /// used to overwrite the resident state and clear `captured`, so the moment a lattice cell fell
    /// out of range its slot went dark and stayed dark for the whole burst — invisible standing
    /// still, constant while running. Now the old cube keeps being drawn, correctly projected for
    /// where it was taken, until the replacement is ready; the swap is then one atomic promotion.
    want: Option<IVec2>,
    want_at: Vec3,
    want_box_min: Vec3,
    want_box_max: Vec3,
    /// How much of this slot's contribution is currently being drawn, 0 to 1.
    ///
    /// **The shape of this is DISTANCE, not time.** A probe does not switch on when the lattice
    /// admits it; it grows as the player walks toward it, weak at the edge of its reach and full
    /// once they are properly inside it, so a new probe joins the picture a little at a time
    /// instead of arriving. See [`fade_target`](Self::fade_target) for the ramp.
    ///
    /// Time enters only as a rate limit on top of that shape, and it earns its place twice: it
    /// keeps a teleport from snapping every probe on at once, and it covers the one transition
    /// distance cannot — a slot that changes CELL changes the content of its cube, and that is a
    /// jump even though both cubes are valid. Such a slot fades out, captures while invisible, and
    /// fades back in. A slot merely refreshing the cell it already holds keeps its picture and
    /// never fades.
    reach: f32,
    /// Fading out for good: this slot's cell is no longer wanted and its replacement is showing,
    /// or its slot is needed for a new cell. Its cube stays valid until the fade reaches zero.
    retire: bool,
    /// This slot's share of the blend, 0 to 1, separate from its availability: the live probe's
    /// crossfade moves share between two cubes without dimming the tier. 1 on the lattice.
    mix: f32,
    /// Whether the resident cube wants re-taking even though it is still of the right cell — the
    /// slow round-robin refresh for a moving sun. Distinct from a pending reassignment, and from
    /// `captured`, because none of the three implies either of the others.
    dirty: bool,
}

impl Default for ProbeSlot {
    fn default() -> Self {
        Self {
            cell: None,
            at: PROBE_AT,
            box_min: PROBE_AT - Vec3::splat(PROBE_BOX_FALLBACK),
            box_max: PROBE_AT + Vec3::splat(PROBE_BOX_FALLBACK),
            captured: false,
            reach: 0.0,
            retire: false,
            mix: 1.0,
            want: None,
            want_at: PROBE_AT,
            want_box_min: PROBE_AT - Vec3::splat(PROBE_BOX_FALLBACK),
            want_box_max: PROBE_AT + Vec3::splat(PROBE_BOX_FALLBACK),
            dirty: false,
        }
    }
}

impl ProbeSlot {
    /// Where this slot's fade is heading — a **smooth function of how far the player is from it**,
    /// zero for a slot that has nothing valid to show.
    ///
    /// Full strength within [`fade_inner`], nothing past [`live_radius`], and a smoothstep between.
    /// Walking toward a lake therefore adds each probe gradually rather than admitting it whole:
    /// the far ones contribute a little, and grow as they are approached.
    /// Where this slot's availability is heading: **1 as soon as it holds a cube, and it never
    /// falls off with distance.**
    ///
    /// This used to be a smoothstep on the PLAYER's distance, cut hard at `live_radius`. That is
    /// what left a stretch between two probes with no reflection at all: a curving channel puts
    /// probes far apart, and midway between two of them both sat past the cutoff, so neither was
    /// eligible and the water had nothing to read. A probe's usefulness does not end at a radius
    /// drawn around the player — it ends where its cube stops resembling what the fragment should
    /// see, which is a per-FRAGMENT question that `probe_weight` already answers over 150 yards.
    ///
    /// So availability is now binary in intent and ramped only so a newly captured probe arrives
    /// rather than appears. Distance is left to the two mechanisms that were always better placed
    /// to judge it: the kernel, which weights by fragment distance, and `probe_weight`.
    fn reach_target(&self) -> f32 {
        f32::from(self.captured && !self.retire)
    }

    /// Free to be aimed at a new cell: no cube, or retired and fully faded out.
    fn is_free(&self) -> bool {
        !self.captured || (self.retire && self.reach <= 0.001)
    }
}

/// A **rate limit** on the distance ramp, in fractions of full strength per second — not the fade
/// itself, which is [`ProbeSlot::fade_target`]'s business.
///
/// Half a second from nothing to full at the most. Walking never hits this limit; a teleport does,
/// and so does the cube-content swap when a slot changes cell, which are the two cases where
/// distance alone would still produce a jump.
const PROBE_FADE_RATE: f32 = 2.0;

/// The least time between one burst finishing and the next starting, in seconds — **and it applies
/// only to slots that already hold a cube**.
///
/// **This is the frame-time tail, measured.** Driving the character around Mirror Lake at 1600x900
/// against the same run with the tier off: eight slots cost +0.72 ms of mean frame time but
/// **+2.95 ms at p95**. Dropping the face resolution from 256 to 128 took the mean cost to +0.11 and
/// left the tail at +2.48 — so the tail is not fill cost. One slot instead of eight took the tail to
/// +0.10. What the tail is made of is the NUMBER OF BURSTS: crossing a 24-yard lattice at running
/// speed retargets a slot every few seconds, and every retarget fires six views into one frame.
///
/// A re-pointed slot has already faded out before it captures (see [`ProbeSlot::fade`]), so making
/// it wait costs nothing visible — it stays faded a little longer while its neighbours cover the
/// gap through the two-nearest blend. A slot with NO cube is exempt, because that is arrival
/// somewhere new, where the wait would be the whole reflection.
const PROBE_BURST_GAP_S: f32 = 0.9;

/// How much nearer a cell the probes already hold is treated as being, when placement decides which
/// cells to keep. 0.85 is a fifteen per cent discount — comfortably more than the distance jitter
/// that makes two cells swap, comfortably less than a lattice step, so a genuinely closer cell still
/// wins.
const PROBE_KEEP_BIAS: f32 = 0.85;

/// Slots kept free for the double buffer: a new cell is captured into one of these while the probe
/// it replaces is still showing — see [`retire_replaced`].
const PROBE_SPARE: usize = 4;

/// How far from the player a probe still contributes anything, in yards.
///
/// Twice the lattice spacing, which is about where the eight-slot neighbourhood ends anyway — so
/// the ramp reaches zero at the boundary the placement was going to enforce regardless, and a probe
/// leaving the live set has already faded out by the time it goes.
/// The lattice cell a Bevy world position falls in — **a hexagon, not a square.**
///
/// Honeycomb for two reasons, and the second is the one that matters here.
///
/// **Coverage.** Hexagonal is the densest packing of the plane, so for a given centre-to-centre
/// spacing the farthest any point can sit from the nearest probe is `s / sqrt(3)` ≈ 0.577 s, against
/// a square lattice's `s * sqrt(2) / 2` ≈ 0.707 s. Since a box-projected cube's error is measured in
/// exactly that distance, the same probe count buys about eighteen per cent less worst-case error.
///
/// **Handover conditioning.** At a square lattice's corner FOUR probes are equidistant, which is the
/// worst-conditioned choice a two-tap blend can be asked to make: the second tap's identity is a
/// coin toss between three candidates, and it flips as the fragment crosses. A honeycomb's vertices
/// are where only THREE cells meet, so the ambiguity is always between one incoming and one
/// outgoing probe — which is the case the vanishing kernel is built to make continuous.
///
/// Axial coordinates on the basis `a1 = (s, 0)`, `a2 = (s/2, s*sqrt(3)/2)`. Rounding to the nearest
/// centre needs CUBE rounding rather than rounding the axial pair, because the basis is not
/// orthogonal and componentwise rounding picks the wrong hexagon near its edges.
fn cell_of(p: Vec3) -> IVec2 {
    let s = probe_spacing();
    let r = p.z / (s * SQRT_3_OVER_2);
    let q = (p.x - r * s * 0.5) / s;
    hex_round(q, r)
}

/// `sqrt(3)/2` — the vertical step of a hex row, as a fraction of the spacing.
const SQRT_3_OVER_2: f32 = 0.866_025_4;

/// Round a fractional axial coordinate to the nearest hexagon.
///
/// Through cube space, where the three coordinates sum to zero: round all three, then correct
/// whichever moved furthest so the sum is restored. Rounding `q` and `r` independently would
/// misplace points near a hexagon's edge, which is where the handover happens and therefore exactly
/// where a mistake would show.
fn hex_round(q: f32, r: f32) -> IVec2 {
    let (x, z) = (q, r);
    let y = -x - z;
    let (mut rx, mut ry, mut rz) = (x.round(), y.round(), z.round());
    let (dx, dy, dz) = ((rx - x).abs(), (ry - y).abs(), (rz - z).abs());
    if dx > dy && dx > dz {
        rx = -ry - rz;
    } else if dy > dz {
        ry = -rx - rz;
    } else {
        rz = -rx - ry;
    }
    let _ = ry;
    IVec2::new(rx as i32, rz as i32)
}

/// How far apart accepted probes must be, as a fraction of the lattice spacing.
///
/// **"The sixteen nearest" is the wrong selection rule**, and the bench says so. Nearness alone
/// packs probes into a blob around the player — in open water that left a coverage horizon of under
/// seven yards with sixteen probes placed, because they were all beside each other rather than
/// spread over the water anyone was looking at. It gets worse once cells are seated at the water
/// rather than judged by their centres, since that admits marginal edge cells which are near and
/// contribute nothing new.
///
/// So candidates are taken nearest-first but only accepted when they stand clear of everything
/// already accepted. The probes then spread over the visible water instead of crowding the player,
/// which is the same suppression rule the medial-axis literature uses and the same one Unreal
/// recommends by hand: a few big captures spread out, not a heap in one place.
/// **1.1, swept on the stream bench against the worst case.** Horizons by channel, for separations
/// 0.5 / 0.8 / 1.1 / 1.4: open water 0.0 / 6.9 / **25.7** / 25.7, straight 58.3 throughout, curving
/// 18.9 / 41.7 / 29.7 / 18.2, narrow-and-curving 40.4 / 36.1 / 41.3 / 40.0. Taking the worst of each
/// column — which is the number that decides whether anywhere looks bad — gives 0.0 / 6.9 / **25.7**
/// / 18.2. Past 1.0 no two probes may sit in adjacent cells, which is what stops them heaping.
const PROBE_MIN_SEPARATION: f32 = 1.1;

/// How many samples across a cell decide where its probe should sit. Five by five is twenty-five
/// wetness lookups per candidate cell, which is cheap next to the chunk walk that answers them and
/// fine enough to find the channel inside a twelve-yard hexagon.
const PROBE_SEAT_SAMPLES: i32 = 9;

/// `$WOW_PROBE_SEAT_SAMPLES=<n>` — the seat grid's resolution, `n` by `n` across a cell.
///
/// **The grid caps how well centred a probe can be, independently of the rule.** Samples sit
/// `spacing / n` apart, so the expected miss from a channel's centreline is about half that: at the
/// old 5 across a 12-yard cell the grid alone was 1.2 yards out, on channels whose whole ideal
/// clearance is 2. Nine puts it near 0.67 and is why the default moved. It costs `n^2` liquid
/// queries per candidate cell, which is the reason not to simply set it large.
fn probe_seat_samples() -> i32 {
    static N: std::sync::OnceLock<i32> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("WOW_PROBE_SEAT_SAMPLES")
            .ok()
            .and_then(|v| v.parse::<i32>().ok())
            .map(|v| v.clamp(3, 25))
            .unwrap_or(PROBE_SEAT_SAMPLES)
    })
}

/// **Where in this cell should a probe stand — if anywhere?**
///
/// Testing the cell's CENTRE and nothing else is what made a curving channel lose half its
/// coverage: the lattice is fixed to the world, so wherever the water wanders between two centres
/// the cell is judged dry and no probe is placed, however much water crosses its corner. Measured on
/// the stream bench, a narrow wandering channel dropped from a 58 yard horizon to 28.
///
/// So the cell is sampled across its span and the probe is seated at the **centroid of the wet
/// samples**. The cell still supplies the probe's IDENTITY — it is world-anchored, so a probe keeps
/// its cube as the player moves — while the seat follows the water, landing mid-channel rather than
/// wherever the lattice happened to fall. Identity from the grid, position from the scene.
fn probe_seat(
    centre: Vec2,
    span: f32,
    n: i32,
    wet: &impl Fn(Vec2) -> Option<f32>,
) -> Option<(Vec2, f32, f32)> {
    let t = |k: i32| (k as f32 + 0.5) / n as f32 - 0.5;
    let mut pts: Vec<(Vec2, Option<f32>)> = Vec::with_capacity((n * n) as usize);
    for j in 0..n {
        for i in 0..n {
            let p = centre + Vec2::new(t(i), t(j)) * span;
            let h = wet(p);
            pts.push((p, h));
        }
    }
    if pts.iter().all(|(_, h)| h.is_none()) {
        return None;
    }
    // **The seat is the wet sample FURTHEST FROM DRY GROUND, not the average of the wet ones.**
    //
    // Averaging was the previous rule and it fails in two opposite directions, both measured by
    // `stream_seat_clearance`. On a channel that wanders off the lattice row the centroid of a
    // clipped sliver leans toward whichever bank the sliver leans to, and the worst seat came out
    // five centimetres from the bank on a four-yard channel where two metres was available. On WIDE
    // water it drifts toward the bulk of wet area in the cell rather than the middle of anything:
    // on a straight sixteen-yard channel the centroid scored 47% of ideal clearance where simply
    // taking the cell centre scored 100%.
    //
    // Maximum clearance has neither failure because it is the definition of the thing wanted. It is
    // the medial axis restricted to the cell — on a channel it lands on the centreline, on open
    // water it lands in the middle — and it can never be dry, which removes the fallback the
    // centroid needed. It matters because a cube magnifies whatever is near its capture point
    // (Unreal documents exactly that), so a probe seated against a bank paints that bank across the
    // reflection.
    //
    // Dryness is only known where it was sampled, so a wet sample with no dry sample anywhere in the
    // grid is unconstrained and the cell centre wins on the tie-break below — which is the right
    // answer for a cell wholly inside a lake.
    let mut best: Option<(f32, f32, Vec2, Option<f32>)> = None;
    for (p, h) in &pts {
        if h.is_none() {
            continue;
        }
        let clear = pts
            .iter()
            .filter(|(_, o)| o.is_none())
            .map(|(q, _)| p.distance(*q))
            .fold(f32::MAX, f32::min);
        // Ties — every sample in an all-wet cell, or a flat ridge along a channel — go to the
        // sample nearest the cell centre, so a probe does not wander along the middle of a lake
        // between placement passes for no reason.
        let tie = p.distance(centre);
        let better = best
            .as_ref()
            .is_none_or(|(bc, bt, _, _)| clear > *bc || (clear == *bc && tie < *bt));
        if better {
            best = Some((clear, tie, *p, *h));
        }
    }
    let (clear, _, seat, h) = best?;
    // **A seat must be demonstrably INTERIOR, not a sliver.** The threshold is the grid's own step:
    // a wet region so thin that no sample inside it is further than one step from a dry one is
    // thinner than this function can see, and a probe dropped in it captures from a hand's breadth
    // off the bank. Rejecting those is not fussiness — it is what stops them being admitted at all,
    // and admission is the real harm. The old centroid rule required the cell CENTRE to be wet,
    // which quietly excluded bank slivers; clearance seating accepts any cell with one wet sample,
    // so without this it admitted every sliver along both banks and they then crowded better cells
    // out of the nearest-N selection. Measured on a straight sixteen-yard channel that took ideal
    // clearance from 100% to 41%.
    //
    // Derived from the grid rather than chosen: at a finer grid a genuinely thin channel becomes
    // visible and is admitted, which is the behaviour wanted.
    let step = span / n as f32;
    if clear < step {
        return None;
    }
    // **The height AT the seat, never the average.** A cell that spans two surfaces — a stream
    // beside a pond, water either side of a fall — has no fragment at the mean of their heights,
    // and a probe placed there floats visibly off the water with a capture point belonging to
    // neither. The seat is a sampled point, so its own height is always available.
    Some((seat, h?, clear))
}

/// A hexagon's centre, in Bevy world XZ — the inverse of [`cell_of`]'s axial basis.
fn cell_centre(c: IVec2) -> Vec2 {
    let s = probe_spacing();
    let (q, r) = (c.x as f32, c.y as f32);
    Vec2::new((q + r * 0.5) * s, r * s * SQRT_3_OVER_2)
}

/// Bevy world XZ back to WoW XY — the inverse of `wow_to_bevy`'s `(-y, z, -x)` on the two axes that
/// survive it, so `wow.x = -bevy.z` and `wow.y = -bevy.x`.
fn wow_xy(p: Vec2) -> (f32, f32) {
    (-p.y, -p.x)
}

/// How far from [`PROBE_AT`], horizontally, the probe's weight reaches zero — in yards.
const PROBE_FADE_YD: f32 = 150.0;

/// How far above or below the probe's own height its weight reaches zero — in yards.
///
/// **Much tighter than the horizontal reach, and the module doc says why**: the bodies stacked
/// around this lake are separated by tens of yards vertically and a few hundred horizontally, so
/// height is the axis that actually distinguishes them. A stream sixty yards up must not read the
/// lake's cube.
///
/// Measured from the CAPTURE point, which [`PROBE_LIFT`] puts twelve yards above the water — so
/// this has to clear the lift and still exclude the stream. At 26 the falloff has not begun by the
/// time it reaches the surface twelve yards below (it starts at half the reach), the second pool
/// seven yards up is comfortably inside, and the stream forty-eight yards further up is outside it
/// entirely. Shrink the lift and this shrinks with it; they are one number in two halves.
const PROBE_FADE_UP: f32 = 26.0;

/// How often the cube is re-captured, in seconds.
///
/// **A probe captured at startup is a picture of an empty sky**, and that is not a hypothetical —
/// it is what the first build of this did. The six faces went in over the first six frames, before
/// any terrain, doodad or building had streamed in, and the cube held nothing but the sky dome
/// forever after. The water dutifully reflected it: flat blue, at a grazing camera where the
/// treeline should have filled the reflection. `$WOW_PROBE_SHOW` is the only reason that was
/// visible at all, because flat blue water under a blue sky looks entirely plausible.
///
/// So the capture repeats. It is also the cheapest answer to the two other things that stale a
/// cube — the world streaming in around the probe, and the sun moving across it — without either
/// needing to be detected. Six seconds is far more often than a static world needs and far less
/// often than the cost would matter: one face per frame, six frames out of every three hundred.
const PROBE_REFRESH_S: f32 = 6.0;

/// `$WOW_PROBE_REFRESH=<seconds>` — override [`PROBE_REFRESH_S`]. A diagnostic lever: the capture
/// harness has about four seconds of clock in it, so the default fires exactly ONE cycle and a
/// fault that a second cycle would correct is indistinguishable from one that is permanent.
fn refresh_s() -> f32 {
    static S: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *S.get_or_init(|| {
        std::env::var("WOW_PROBE_REFRESH")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(PROBE_REFRESH_S)
    })
}

/// How many frames the six faces are held active before the cube is taken.
///
/// **A view that was inactive last frame has no specialized pipelines, and Bevy draws nothing for
/// the meshes that need them — silently.** `bevy_pbr`'s `SpecializedMaterialPipelineCache` is keyed
/// by `retained_view_entity`, and `specialize_material_meshes` ends with
/// `.retain(|view, _| all_views.contains(view))` (bevy_pbr 0.18.1, `material.rs:1147`): a view
/// absent from a frame has its whole cache dropped and must rebuild it on reactivation.
///
/// The retained `static_gx` pass does not go through that path — it draws its own list — so the
/// cube came back holding a forest with **no ground under it**, floating in the clear colour. That
/// symptom is badly misleading. It reads as a broken face orientation, a cull gap, or fog, and was
/// blamed on each in turn. It is none of them: the faces are aimed correctly and their visible sets
/// are full. A per-face trace (`$WOW_PROBE_TRACE=1`) counted 148, 190, 19, 6, 302 and 261 `Mesh3d`
/// rows across the six views, including both that came back empty. The rows were always there; the
/// pipelines to draw them were not.
///
/// Two reproducible facts pin the cause down:
///
/// * The fault is **ordinal, not directional** — it lands on whichever faces go first.
///   `$WOW_PROBE_ORDER=1` rotated the capture order while each face kept its own cube layer, and
///   the empty panel moved from +X to +Z exactly as the rotation predicts.
/// * **More cycles never helped** (`$WOW_PROBE_REFRESH`), because each fresh cycle re-broke its own
///   first faces identically. Only more consecutive frames per face helped — which is what a cache
///   rebuilt per activation predicts, and what a compiled-pipeline warm-up does not.
///
/// **Holding one face at a time does not fix it, and the measurement is worth keeping.** A dwell
/// sweep at Mirror Lake (2/3/4/6/8) first suggested four was clean, on two runs. Four more runs at
/// four came back three-bad: the fault is FLAKY, so a couple of green captures prove nothing, and
/// every earlier single-capture verdict in this file's history was worth less than it looked. Long
/// dwells also fail from the other end — at six and eight the LAST faces come back black, because
/// six faces times the dwell stops fitting before the capture harness's shutter.
///
/// So the faces are held **together** instead. All six stay active for the whole burst, so none is
/// the unlucky one reactivated after a gap, and the copy is taken only on the final frame, once
/// every view has rebuilt. Six runs, six clean cubes, and — unlike every one-at-a-time variant —
/// byte-identical results across all six, which is what removing a race looks like from outside.
///
/// **It costs one dropped frame per burst, and that is not a bug to be tuned away.** Interleaved
/// against `$WOW_PROBE=0`, three runs each: max frame 31.8/33.5/32.7 ms against 20.9/19.9/19.9,
/// with p99 essentially unmoved (19.2-19.6 against 18.9-19.3) — so it is roughly ONE slow frame per
/// burst, not six. Ramping the faces in two at a time to spread the cold specialization was tried
/// and measured and **did not work** (32.4/26.6/32.9 ms — two of three identical to the flat
/// burst), because the peak comes from six views sharing a frame at all, and correctness is exactly
/// what requires them to share one. The ramp was reverted rather than kept as complexity that buys
/// nothing. If the stutter ever matters, the lever is [`PROBE_REFRESH_S`] — it makes the hitch
/// rarer — or a smaller [`PROBE_FACE`]; it is not the burst shape.
const PROBE_BURST: u32 = 6;

/// `$WOW_PROBE_BURST=<frames>` — override [`PROBE_BURST`].
fn burst_frames() -> u32 {
    static B: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *B.get_or_init(|| {
        std::env::var("WOW_PROBE_BURST")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(PROBE_BURST)
    })
}

/// How far a face camera sees, in yards.
///
/// `$WOW_PROBE_FAR=<yards>` overrides it. Measured as a non-issue at the capture pin — 600 and 4000
/// produced a byte-identical cube, because `WOW_FARCLIP` governs what terrain exists at all and
/// nothing is resident past it — but the default stays clear of the sky dome, which
/// `crate::sky::follow_camera` scales to the WORLD camera's far plane times 0.9 and which a shorter
/// far plane would clip away entirely.
fn probe_far() -> f32 {
    static F: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *F.get_or_init(|| {
        std::env::var("WOW_PROBE_FAR")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4000.0)
    })
}

/// `$WOW_PROBE_TRACE=1` — log what each active face view actually resolved to see.
///
/// The cube's faces failed in a way that looked like a maths bug and was not one: the geometry was
/// right and the views were simply not drawing rows they could see. Counting `VisibleEntities` per
/// face answered in one run what staring at the cube could not — see [`PROBE_BURST`].
fn trace_faces(
    probe: Res<WaterProbe>,
    views: Query<(
        &ProbeFace,
        &Camera,
        Option<&bevy::camera::visibility::VisibleEntities>,
    )>,
) {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var("WOW_PROBE_TRACE").as_deref() == Ok("1")) {
        return;
    }
    for (which, cam, vis) in &views {
        if !cam.is_active {
            continue;
        }
        // `VisibleEntities` is keyed by visibility CLASS, so ask it two questions: the total across
        // every class, and the `Mesh3d` bucket alone. Terrain cells are ordinary `Mesh3d` rows and
        // the retained `static_gx` scenery is not, so the second number is the one that separates
        // "this view never saw the ground" from "it saw it and did not draw it".
        let (total, meshes) = match vis {
            Some(v) => (
                v.entities.values().map(|e| e.len()).sum::<usize>(),
                v.len(std::any::TypeId::of::<Mesh3d>()),
            ),
            None => (usize::MAX, usize::MAX),
        };
        warn!(
            "probe trace: face {} active, burst {}, copy {}, rows {} mesh3d {}",
            which.0, probe.dwell, probe.copy, total, meshes
        );
    }
}

/// How long the FIRST capture waits, in seconds.
///
/// The world streams in around the player, so a cube taken on frame six is a cube of an empty sky —
/// no terrain, no doodads, no buildings, and the water reflecting flat blue forever after. Waiting
/// is the whole fix, and waiting is enough: by the time a couple of seconds have passed the
/// neighbourhood is resident.
///
/// **Deliberately shorter than [`PROBE_REFRESH_S`], because the capture harness has about four
/// seconds of frozen clock in it.** A settle of ~97 frames and an age of ~150 is roughly four
/// seconds at the pinned frame time, so a first capture armed on the refresh interval would never
/// fire before the shutter and every golden shot would show the empty-sky cube. That is not a
/// hypothesis: it is what the first two attempts at this produced, and `$WOW_PROBE_SHOW` is the
/// only reason it was visible, since flat blue water under a blue sky looks entirely correct.
const PROBE_FIRST_S: f32 = 2.0;

/// `$WOW_PROBE=0` — turn the tier off. On by default once the lane exists; the capture is a handful
/// of frames at startup and the sample is one cube fetch.
fn probe_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("WOW_PROBE").as_deref() != Ok("0"))
}

/// `$WOW_PROBE_LIVE=1` — one probe that follows the player, re-captured continuously, in place of
/// the lattice (the single player-centred cube GTA V renders each frame). Time-sliced: one camera
/// renders one face a frame, so a cube completes every six frames. Slots 0 to 2 rotate: one being
/// written and hidden, one fading in over the next cycle, one fading out.
pub(crate) fn probe_live() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("WOW_PROBE_LIVE").as_deref() == Ok("1"))
}

/// `$WOW_PROBE_SHOW=1` — paint the probe's own sample instead of the water.
///
/// In the `$WOW_SSR_SHOW` / `$WOW_MIRROR_SHOW` family, and needed for the same reason those are:
/// **a cubemap of the wrong place is still a plausible picture.** A face captured with the wrong
/// orientation, or a cube that never received its copy, produces water that looks like water and is
/// reflecting somewhere else entirely. That cannot be caught by looking at the final image, so the
/// raw sample has to be paintable.
fn probe_show() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("WOW_PROBE_SHOW").is_some())
}

/// `$WOW_REFL_SHOW=1` — paint the **whole composited reflection** on the water and nothing else.
///
/// Not the same lever as [`probe_show`], and the difference is the point. That one paints one
/// TIER's raw sample; this one paints what all of them add up to — sky, then probe, then capture,
/// then march — with the water's own body colour and the Fresnel weight both taken out of the way.
///
/// **It exists because the Fresnel weight makes every tier nearly invisible at a normal camera.**
/// The final mix is `mix(body, refl, fres)` and `fres` is `mix(0.02, 0.45, (1 - cos)^5)`, which at
/// twenty to twenty-five degrees above the surface is four to seven per cent: ninety-five per cent
/// of the pixel is the water's own colour, and a tier can be working perfectly and still look like
/// a faint wash. That cost a long stretch of misdiagnosis — a probe measured at 99.7% coverage of
/// the water and still read as "nothing is happening", and two innocent constants were tuned
/// before the ceiling was suspected. With this lever the question "is the reflection right" is
/// asked directly, instead of being inferred through a twentieth of its own strength.
fn refl_show() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("WOW_REFL_SHOW").is_some())
}

/// `$WOW_PROBE_CUBE=1` — paint the **whole cube**, unwrapped, across the water.
///
/// Every other view of a probe shows it through something: through a reflected direction, through
/// the ripple, through a Fresnel weight. This one shows the cube itself as a latitude-longitude
/// map — screen X is the compass bearing, screen Y is the elevation — so "what did the six cameras
/// actually capture" is answerable by looking rather than by inference.
///
/// **It should have been the first thing built.** Without it, an empty cube, a cube missing the
/// static scenery, and a cube being sampled in the wrong directions all look the same from the
/// water: a plausible, slightly dull reflection. Three separate causes were diagnosed and tuned
/// against that one symptom — capture height, proxy radius, proxy shape — and none of them was
/// what was wrong. A cubemap of the wrong place is still a plausible picture, and the only cure is
/// to look at the map.
fn probe_cube_show() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("WOW_PROBE_CUBE").is_some())
}

/// Which cube face a capture camera owns, in wgpu's layer order: +X, -X, +Y, -Y, +Z, -Z.
///
/// Extracted to the render world so `WaterProbeNode` can pair each face's colour target with that
/// view's `ViewDepthTexture` — the depth is what becomes the cube's alpha.
#[derive(Component, Clone, Copy, bevy::render::extract_component::ExtractComponent)]
pub(crate) struct ProbeFace(pub(crate) usize);

/// The look and up vectors for each face, in wgpu's layer order.
///
/// **World up, and it is the OPENGL convention negated — which is correct here, for a reason.**
///
/// Every canonical listing of this table (LearnOpenGL's point-shadow cubemap, the GL spec's own
/// face selection) gives the four side faces `up = (0, -1, 0)` and the poles `(0, 0, ±1)`. Those
/// are written for **OpenGL**, whose NDC Y axis points up. Vulkan — and so wgpu, and so Bevy —
/// inverts NDC Y, so identical geometry rendered into a texture lands vertically flipped, while
/// cubemap *sampling* follows the same left-handed face convention in both APIs. Copying the GL
/// table verbatim therefore flips all six faces, and the fix is to negate the up vectors back.
///
/// This was found the other way round — from the symptom, then explained — and the symptom is worth
/// recording because it is so unlike what it sounds like:
/// the unwrapped cube showed ground at forty to sixty degrees of elevation, where the sky belongs,
/// and every shallow reflected ray — which is most of them on water — sampled that band and came
/// back the colour of dirt. No amount of correct parallax recovers from a cube whose sky is made of
/// ground, and three rounds of tuning the projection were spent before the map was read.
///
/// The failure is legible exactly once, in `$WOW_PROBE_CUBE`: a lat-long map has a horizon, and a
/// horizon in the wrong place is obvious. Through a reflection it is invisible — it just looks like
/// a dull reflection of a bank.
///
/// The lesson for the next table copied out of a tutorial: check whose NDC it was written for.
/// GL-to-Vulkan Y inversion silently flips anything rendered into a texture, and a flipped cube
/// face is not a crash, a warning, or an artefact — it is a plausible picture of the wrong world.
const FACES: [([f32; 3], [f32; 3]); 6] = [
    ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
    ([-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
    ([0.0, 1.0, 0.0], [0.0, 0.0, -1.0]),
    ([0.0, -1.0, 0.0], [0.0, 0.0, 1.0]),
    ([0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
    ([0.0, 0.0, -1.0], [0.0, 1.0, 0.0]),
];

/// The probe's cube, the six faces feeding it, and how much of the capture is still owed.
#[derive(Resource, Clone, ExtractResource)]
pub(crate) struct WaterProbe {
    /// The cubemap the liquid materials sample.
    pub(crate) cube: Handle<Image>,
    /// The six 2D render targets, one per face, in [`FACES`] order.
    faces: [Handle<Image>; 6],
    /// Frames the burst has been running for — see [`PROBE_BURST`].
    dwell: u32,
    /// Set for the frames on which the node should copy faces into the cube.
    copy: bool,
    /// Which slot's six layers the copy frame writes. Carried separately from
    /// [`capturing`](Self::capturing) because the burst ENDS on the copy frame — by the time the
    /// render world extracts the resource, `capturing` is already `None` and the node would have
    /// nothing to aim at.
    copy_slot: usize,
    /// Seconds until the next capture is armed. Counts down only while the tier is armed.
    refresh_in: f32,
    /// Whether a complete cube has ever been built. Until it has, the strength lane stays at zero
    /// and the water falls back to the sky mix rather than reading a half-built or empty cube.
    ever: bool,
    /// Whether the tier is live at all — the stylised looks only, like everything else here.
    armed: bool,
    /// The live probes, one per six layers of [`cube`](Self::cube). Length is [`probe_slots`].
    pub(crate) slots: Vec<ProbeSlot>,
    /// Which slot the six capture cameras are currently pointed at, if a burst is running.
    capturing: Option<usize>,
    /// The lattice cell the player was in when the slots were last placed.
    anchor: Option<IVec2>,
    /// Seconds until placement runs again regardless of whether the player changed cell.
    place_in: f32,
    /// The cells placement currently wants — see [`next_capture`] and [`retire_replaced`].
    wanted: Vec<Wanted>,
    /// `$WOW_PROBE_LIVE`: where the probe should stand now, and its proxy box.
    live_target: Option<Wanted>,
    /// `$WOW_PROBE_LIVE`: this cycle's capture point, the face rendered this frame, the slots
    /// being written, fading in and fading out, and completed cycles.
    live_cycle: Option<Wanted>,
    live_face: usize,
    live_back: usize,
    live_front: Option<usize>,
    live_old: Option<usize>,
    live_cycles: u32,
    /// The single face to copy this frame, from face camera 0 — the time-sliced capture.
    pub(crate) copy_face: Option<usize>,
    /// Seconds left before a slot that already holds a cube may start a burst — see
    /// [`PROBE_BURST_GAP_S`]. Never gates a slot that has nothing to show.
    burst_gap: f32,
    /// Round-robin position for the slow refresh, so a moving sun reaches every cube in
    /// `slots * refresh` seconds at a cost of one burst per window.
    cursor: usize,
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
    /// first capture, and false for a slot the lattice has not given a cell to.
    pub live: bool,
    /// **This probe's share of the blend at the player's position**, 0 to 1 — normalised across the
    /// taps, so the lit markers are the probes the water is actually reading and the rest are dark.
    ///
    /// Not the same as "available": since availability stopped falling off with distance, every
    /// captured probe is available, and a marker showing that lit every probe on the map at once.
    pub fade: f32,
}

/// The live probes, republished each frame for the minimap and world overlays.
#[derive(Resource, Default)]
pub struct ProbeMarks(pub Vec<ProbeMark>);

/// Copy the slot list into [`ProbeMarks`]. Its own system rather than a tail on `drive_probe`,
/// which returns early on several paths and would leave the overlay showing the frame before.
/// A cell placement wants: where its probe stands, and its proxy box.
#[derive(Clone, Copy, Debug)]
struct Wanted {
    cell: IVec2,
    at: Vec3,
    lo: Vec3,
    hi: Vec3,
}

/// Which slot to capture next, **double-buffered**: a wanted cell with no cube is captured into a
/// FREE slot (never captured, or retired and faded out), so no probe that is showing ever changes
/// content. With no slot free, the old probe furthest from the player starts retiring — one at a
/// time, because captures land one per burst and a slot retired early would only sit invisible.
///
/// Among missing cells, the one that closes the **biggest hole** goes first: its distance from the
/// nearest showing probe, less a quarter of its distance from the player. Nearest-first spends the
/// bursts on probes being walked away from.
///
/// The burst gap holds every capture except into a slot never captured, which is arrival. With nothing
/// missing, a slot the refresh marked dirty is re-taken in place.
fn next_capture(
    slots: &mut [ProbeSlot],
    wanted: &[Wanted],
    eye: Vec2,
    gap_ready: bool,
) -> Option<usize> {
    let is_wanted = |c: Option<IVec2>| c.is_some_and(|c| wanted.iter().any(|w| w.cell == c));
    // A probe retiring from a cell that is wanted again comes back rather than being re-taken.
    for s in slots.iter_mut() {
        if s.retire && s.captured && s.reach > 0.001 && is_wanted(s.cell) {
            s.retire = false;
        }
    }
    let showing: Vec<Vec2> = slots
        .iter()
        .filter(|s| s.captured && !s.retire)
        .map(|s| Vec2::new(s.at.x, s.at.z))
        .collect();
    let missing = wanted.iter().filter(|w| {
        !slots
            .iter()
            .any(|s| s.captured && !s.retire && s.cell == Some(w.cell))
    });
    let score = |w: &&Wanted| {
        let at = Vec2::new(w.at.x, w.at.z);
        let hole = showing
            .iter()
            .map(|p| p.distance(at))
            .fold(f32::MAX, f32::min);
        if hole == f32::MAX {
            1e6 - at.distance(eye)
        } else {
            hole - at.distance(eye) * 0.25
        }
    };
    let Some(next) = missing
        .max_by(|a, b| score(a).total_cmp(&score(b)))
        .copied()
    else {
        if !gap_ready {
            return None;
        }
        let i = slots
            .iter()
            .position(|s| s.captured && !s.retire && s.dirty)?;
        slots[i].want = slots[i].cell;
        return Some(i);
    };
    let far = |s: &ProbeSlot| {
        if s.captured {
            Vec2::new(s.at.x, s.at.z).distance(eye)
        } else {
            f32::MAX
        }
    };
    let free = slots
        .iter()
        .enumerate()
        .filter(|(_, s)| s.is_free() && !(s.captured && is_wanted(s.cell)))
        .max_by(|(_, a), (_, b)| far(a).total_cmp(&far(b)))
        .map(|(i, _)| i);
    let Some(i) = free else {
        if !slots.iter().any(|s| s.retire && s.reach > 0.001) {
            if let Some((_, s)) = slots
                .iter_mut()
                .enumerate()
                .filter(|(_, s)| s.captured && !s.retire && !is_wanted(s.cell))
                .max_by(|(_, a), (_, b)| far(a).total_cmp(&far(b)))
            {
                s.retire = true;
            }
        }
        return None;
    };
    if !(gap_ready || !slots[i].captured) {
        return None;
    }
    let s = &mut slots[i];
    s.want = Some(next.cell);
    s.want_at = next.at;
    s.want_box_min = next.lo;
    s.want_box_max = next.hi;
    Some(i)
}

fn ramp_probe_fades(time: Res<Time>, mut probe: ResMut<WaterProbe>) {
    let step = PROBE_FADE_RATE * time.delta_secs();
    for slot in probe.slots.iter_mut() {
        let r = slot.reach_target();
        slot.reach = (slot.reach + (r - slot.reach).clamp(-step, step)).clamp(0.0, 1.0);
    }
    if probe_live() {
        return;
    }
    let probe = &mut *probe;
    retire_replaced(&mut probe.slots, &probe.wanted);
}

/// **Double buffering: an old probe leaves only once the new set is showing.** When every wanted
/// cell has a captured probe at full strength, the probes of cells no longer wanted fade out. Until
/// then they stay, so the water always blends valid cubes and a probe changes content only while
/// invisible (see [`ProbeSlot::may_capture`]).
fn retire_replaced(slots: &mut [ProbeSlot], wanted: &[Wanted]) {
    let complete = wanted.iter().all(|w| {
        slots
            .iter()
            .any(|s| s.cell == Some(w.cell) && s.captured && !s.retire && s.reach >= 0.999)
    });
    if !complete {
        return;
    }
    for s in slots.iter_mut() {
        if s.captured && s.cell.is_none_or(|c| !wanted.iter().any(|w| w.cell == c)) {
            s.retire = true;
        }
    }
}

fn publish_probe_marks(
    probe: Res<WaterProbe>,
    eye: Query<&GlobalTransform, With<crate::view::WorldCamera>>,
    mut marks: ResMut<ProbeMarks>,
) {
    // **The marker shows CONTRIBUTION where the player stands, not mere availability.**
    //
    // It used to show `reach * hand`, and once availability stopped falling off with distance that
    // made every captured probe read as fully active — true, and useless: the overlay exists to
    // answer "which probes is the water actually reading", and every one of them answered yes.
    //
    // So it now mirrors, on the CPU and for display only, what the shader computes for a fragment
    // at the player's feet: rank by weight, apply the same vanishing kernel with the first
    // unsampled probe as the cutoff, normalise. Two or three probes come out lit and the rest dark,
    // which is what the blend is really doing. Nothing here feeds the render — if this and the
    // shader ever disagree, the shader is right and this is the thing to fix.
    let eye = eye
        .single()
        .map(|e| Vec2::new(e.translation().x, e.translation().z))
        .unwrap_or(Vec2::ZERO);
    let taps = probe_taps() as usize;
    // Heaviest first by availability over distance, as the shader ranks them.
    let mut rank: Vec<(f32, usize)> = probe
        .slots
        .iter()
        .enumerate()
        .filter(|(_, s)| s.reach > 0.0)
        .map(|(i, s)| {
            (
                s.reach / Vec2::new(s.at.x, s.at.z).distance(eye).max(1e-4),
                i,
            )
        })
        .collect();
    rank.sort_by(|a, b| b.0.total_cmp(&a.0));
    let cut = rank.get(taps).map_or(0.0, |(u, _)| *u);
    let mut weight = vec![0.0f32; probe.slots.len()];
    let mut total = 0.0;
    for &(u, i) in rank.iter().take(taps) {
        let w = (u - cut).max(0.0);
        weight[i] = w;
        total += w;
    }
    if total > 1e-8 {
        for w in &mut weight {
            *w /= total;
        }
    }

    marks.0.clear();
    // A slot with no resident cell and no target has never captured anything and stands nowhere —
    // it would draw at the fallback position, which is a marker for a probe that does not exist.
    marks.0.extend(
        probe
            .slots
            .iter()
            .enumerate()
            .filter(|(_, s)| s.cell.is_some() || s.want.is_some())
            .map(|(i, s)| ProbeMark {
                at: if s.cell.is_some() { s.at } else { s.want_at },
                live: s.captured && probe.armed,
                fade: if probe.armed { weight[i] } else { 0.0 },
            }),
    );
}

/// The probe markers ride [`UNMIRRORED_RENDER_LAYER`](super::UNMIRRORED_RENDER_LAYER) — the layer
/// this crate already keeps for things the world camera draws and the mirror does not.
///
/// They needed a layer at all because every face camera draws whatever the default gizmo group
/// draws: eight green rings and their masts went into the probe's own cube and came back as a
/// bright green line across the horizon of every reflection. Measured in the lat-long unwrap, 3590
/// marker-coloured texels before and 7 after. The planar mirror would have taken them too. A debug
/// overlay that appears inside the thing it is measuring is worse than no overlay, because it
/// changes the picture it exists to explain.
///
/// **The right layer already existed**, and its own doc describes a nameplate as "a label attached
/// to the viewer's own eye; it has no reflection, any more than a cursor does" — which is a probe
/// marker word for word. A private layer was tried first and drew nothing, for a reason worth
/// keeping: `reflect.rs` *inserts* the world camera's `RenderLayers` every frame, so a layer added
/// at the spawn site is silently replaced. Layers for that camera are that system's to grant, and
/// the way to get one is to use a layer it already lists.
///
/// The group exists so that layer can be set without moving the fishing line and the bowstring off
/// the default group with them.
#[derive(Default, Reflect, GizmoConfigGroup)]
pub struct ProbeMarkGizmos;

/// Colour for a probe the water is reading, and for one placed but not yet captured.
///
/// Green and grey rather than two hues: the question the overlay answers is binary and the answer
/// should survive being glanced at on a minimap eight pixels wide.
const PROBE_MARK_LIVE: Color = Color::srgb(0.25, 0.95, 0.35);
const PROBE_MARK_IDLE: Color = Color::srgb(0.55, 0.55, 0.58);

/// Put the marker group on the unmirrored layer so no capture camera can see it.
fn confine_probe_gizmos(mut store: ResMut<GizmoConfigStore>) {
    store.config_mut::<ProbeMarkGizmos>().0.render_layers =
        bevy::camera::visibility::RenderLayers::layer(super::UNMIRRORED_RENDER_LAYER);
}

/// Draw each probe in the world as a ring at its capture point, under the panel's water toggle.
///
/// A ring lying in the water plane plus a short mast: the ring says where the cube was taken from,
/// which is the thing the projection rebases onto, and the mast makes it findable from a distance
/// without hiding the surface under a filled shape.
fn draw_probe_gizmos(
    dev: Res<crate::dev_state::DebugState>,
    marks: Res<ProbeMarks>,
    mut gizmos: Gizmos<ProbeMarkGizmos>,
) {
    if !dev.water.probe_world {
        return;
    }
    for m in &marks.0 {
        // Grey ramps to green with the fade, so the marker shows how much the water is actually
        // taking from this probe rather than a state the reflection never passes through.
        let colour = PROBE_MARK_IDLE.mix(&PROBE_MARK_LIVE, m.fade);
        // **Drawn on the water, not at the capture point.** The capture sits [`PROBE_LIFT`] above
        // the surface for reasons that are load-bearing — a capture point exactly on the plane
        // reads as underwater and fogs the whole cube — but a marker floating four yards up reads
        // as a misplaced probe, which is the one thing this overlay must not do. So the ring lies
        // on the surface where the probe belongs, and the mast rises to where it actually captures.
        let seat = m.at - Vec3::Y * PROBE_LIFT;
        gizmos.circle(
            Isometry3d::new(seat, Quat::from_rotation_x(std::f32::consts::FRAC_PI_2)),
            1.5,
            colour,
        );
        gizmos.line(seat, m.at, colour);
    }
}

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
pub(crate) struct ProbeCull {
    /// Set for the whole capture and a few frames either side of it.
    ///
    /// **Held rather than pulsed, because system order is not guaranteed.** `cull_cells` and
    /// `drive_probe` both run in `Update` with no ordering between them, so a flag raised as a face
    /// is activated can easily be read by a cull that already ran — and that face captures a world
    /// with no trees in it while its neighbours capture one with. It showed as two of the six faces
    /// coming back as flat sky panels in the unwrapped cube while the other four held the forest.
    /// A countdown that spans the whole six-frame capture cannot be raced.
    pub(crate) active: bool,
    /// The capture point, in Bevy world yards.
    pub(crate) at: Vec3,
    /// How far around it to admit, in yards — the box's reach plus its ceiling, so anything the
    /// projection can land on has been drawn.
    pub(crate) radius: f32,
    /// Frames remaining on the hold. Counted down by `drive_probe`, read as `active` by the cull.
    hold: u32,
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct WaterProbeLabel;

/// A cube-shaped image the water samples. `COPY_DST` because the node writes it with
/// `copy_texture_to_texture` rather than rendering into it — see the module doc on why a camera
/// cannot address a face.
fn cube_image() -> Image {
    let mut image = Image::new_fill(
        Extent3d {
            width: PROBE_FACE,
            height: PROBE_FACE,
            // Six layers per probe, `probe_slots()` probes deep — one cube array rather than one
            // texture per probe, so the water picks a probe with an array index instead of the
            // shader needing a binding per slot.
            depth_or_array_layers: 6 * probe_slots() as u32,
        },
        TextureDimension::D2,
        &[0, 0, 0, 0, 0, 0, 0, 0],
        TextureFormat::Rgba16Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    // RENDER_ATTACHMENT, not COPY_DST: each face is written by a mirroring blit into that
    // layer's own view rather than copied — see `shaders/probe_face.wgsl` for why a copy
    // cannot produce a correct cube face.
    image.texture_descriptor.usage =
        TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT;
    // **The mip chain is the tier's whole purpose now** — see [`probe_mips`] and
    // `shaders/probe_filter.wgsl`. Level 0 is the blit's, every level below it is the filter's.
    image.texture_descriptor.mip_level_count = probe_mips();
    // No CPU data: every texel is written by a render pass, and `new_fill` would otherwise
    // allocate and upload the base level for all ninety-six layers — fifty megabytes at the
    // default face size, to be overwritten before anything reads it. A slot with no cube is
    // already gated by its `at.w` reach flag, so uninitialised is not observable.
    image.data = None;
    image.texture_view_descriptor = Some(TextureViewDescriptor {
        dimension: Some(TextureViewDimension::CubeArray),
        ..default()
    });
    image
}

/// One face's render target — HDR, to match the cube and the scale the water mixes on.
fn face_image() -> Image {
    let mut image = Image::new_fill(
        Extent3d {
            width: PROBE_FACE,
            height: PROBE_FACE,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0, 0, 0, 0, 0, 0, 0, 0],
        TextureFormat::Rgba16Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.usage =
        TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT;
    image
}

/// Spawn the six capture cameras, inactive, aimed at the probe's six faces.
///
/// They share the mirror's view-key shape for the same reason the mirror has one: a second view
/// needs its pipelines to match something, and `Msaa::Off` + `Hdr` + `Tonemapping::None` with no
/// prepass is the shape `pipe_warm`'s warm mirror already compiles.
fn setup_probe(mut commands: Commands, mut images: ResMut<Assets<Image>>, probe: Res<WaterProbe>) {
    for (i, (look, up)) in FACES.iter().enumerate() {
        let _ = &mut images;
        commands
            .spawn((
                Name::new(format!("water probe face {i}")),
                super::reflect::mirror_view_shape(),
                ProbeFace(i),
                Projection::Perspective(PerspectiveProjection {
                    // A cube face is a 90-degree square, and it has to be exact: anything else leaves a
                    // gap or an overlap at every seam.
                    fov: std::f32::consts::FRAC_PI_2,
                    aspect_ratio: 1.0,
                    far: probe_far(),
                    near: PROBE_NEAR,
                    ..default()
                }),
                bevy::camera::RenderTarget::Image(probe.faces[i].clone().into()),
                Camera {
                    // Ahead of the world camera and of the mirror, so a face captured this frame is
                    // ready for the water that samples it in the same frame.
                    order: -20 + i as isize,
                    is_active: false,
                    ..default()
                },
                Transform::from_translation(PROBE_AT + Vec3::Y * PROBE_LIFT)
                    .looking_to(Vec3::from_array(*look), Vec3::from_array(*up)),
            ))
            // Overridden AFTER the bundle rather than inside it: `mirror_view_shape` already carries a
            // `Camera3d`, and a tuple bundle holding the same component twice panics on spawn.
            //
            // TEXTURE_BINDING on top of the usual attachment use, so the mirroring blit can sample this
            // view's depth and write distance into the cube's alpha — see `probe_face.wgsl`.
            .insert(Camera3d {
                depth_texture_usages: (TextureUsages::RENDER_ATTACHMENT
                    | TextureUsages::TEXTURE_BINDING)
                    .into(),
                ..default()
            });
    }
}

/// Run the capture a face at a time, and publish the probe's lanes.
///
/// **One face per frame, not six.** Six extra views in one frame is a visible hitch on the frame the
/// player walks into range, and the cube is a picture of a static world — there is nothing to
/// synchronise, so spreading it costs only that the first few frames sample an incomplete cube,
/// which the strength lane holds at zero until it is whole.
/// The projection box for the water under the probe, in Bevy world yards.
///
/// Unions the footprints of every liquid chunk whose box lies within the probe's horizontal reach,
/// so the walls land on the shoreline of the body the probe is standing in. WoW XY maps to Bevy XZ
/// with both axes negated and swapped (`benilla_assets::coords::wow_to_bevy` is `(-y, z, -x)`),
/// which is why a WoW x-range becomes a Bevy z-range here.
fn derive_box(chunks: &Query<&super::WaterChunkInfo>, at: Vec3) -> Option<(Vec3, Vec3)> {
    let (mut lo, mut hi) = (Vec2::splat(f32::MAX), Vec2::splat(f32::MIN));
    let mut any = false;
    let reach = box_reach();
    let (mut seen, mut bounded, mut near, mut nearest) = (0usize, 0usize, 0usize, f32::MAX);
    for info in chunks {
        seen += 1;
        let Some([[min_x, min_y], [max_x, max_y]]) = info.xy_bounds() else {
            continue;
        };
        bounded += 1;
        // WoW (x, y) -> Bevy (x = -y, z = -x), so each range flips and swaps axis.
        let b_lo = Vec2::new(-max_y, -max_x);
        let b_hi = Vec2::new(-min_y, -min_x);
        let centre = (b_lo + b_hi) * 0.5;
        let d = centre.distance(Vec2::new(at.x, at.z));
        nearest = nearest.min(d);
        if d > box_reach() {
            continue;
        }
        near += 1;
        lo = lo.min(b_lo);
        hi = hi.max(b_hi);
        any = true;
    }
    if std::env::var("WOW_PROBE_TRACE").as_deref() == Ok("1") {
        use std::sync::atomic::{AtomicU32, Ordering};
        static TICK: AtomicU32 = AtomicU32::new(0);
        let t = TICK.fetch_add(1, Ordering::Relaxed);
        if t.is_multiple_of(120) {
            warn!(
                "derive_box: {seen} chunks seen, {bounded} with bounds, {near} within {reach} yd                  of the probe; nearest centre {nearest:.1} yd; probe at ({}, {})",
                at.x, at.z
            );
        }
    }
    any.then(|| {
        // Out past the shoreline, so the walls stand where the TREES do — see [`PROBE_BOX_OUT`].
        let out = Vec2::splat(PROBE_BOX_OUT);
        let (lo, hi) = (lo - out, hi + out);
        (
            Vec3::new(lo.x, at.y - PROBE_BOX_DOWN, lo.y),
            Vec3::new(hi.x, at.y + PROBE_BOX_UP, hi.y),
        )
    })
}

/// Place the live probes on the world lattice around the player.
///
/// Walks the lattice cells within reach of the camera, keeps those that have water at their centre,
/// takes the [`probe_slots`] nearest, and reconciles them against the slots already held — a slot
/// whose cell survives keeps its cube, and only the ones that changed hands are marked for
/// re-capture. That reconciliation is the whole point of anchoring the lattice to the world: walking
/// across a lake reassigns at most the trailing slot per step instead of invalidating all eight.
fn place_probes(
    style: Res<WaterStyle>,
    time: Res<Time>,
    chunks: Query<&super::WaterChunkInfo>,
    eye: Query<&GlobalTransform, With<crate::view::WorldCamera>>,
    viewer: Res<crate::view::Viewer>,
    mut probe: ResMut<WaterProbe>,
) {
    if !(style.wants_probe() && probe_enabled()) {
        return;
    }
    let Ok(eye) = eye.single() else {
        return;
    };
    let eye = eye.translation();
    if probe_live() {
        // Over the player (the camera where there is no player), on land as on water: at the
        // higher of the feet and the water surface nearby, so it is never under the ground on a
        // bank nor under the surface when wading.
        let body = viewer.at.unwrap_or(eye);
        let wet = |p: Vec2| {
            let (wx, wy) = wow_xy(p);
            chunks.iter().find_map(|c| c.surface_z_at(wx, wy))
        };
        let xz = Vec2::new(body.x, body.z);
        let water = wet(xz).or_else(|| {
            probe_seat(xz, probe_spacing(), probe_seat_samples(), &wet).map(|(_, z, _)| z)
        });
        let floor = if viewer.at.is_some() {
            body.y
        } else {
            f32::MIN
        };
        let Some(y) = water.map(|z| z.max(floor)).or(viewer.at.map(|b| b.y)) else {
            return;
        };
        let at = Vec3::new(xz.x, y + PROBE_LIFT, xz.y);
        let (lo, hi) = derive_box(&chunks, at).unwrap_or((
            at - Vec3::splat(PROBE_BOX_FALLBACK),
            at + Vec3::splat(PROBE_BOX_FALLBACK),
        ));
        probe.live_target = Some(Wanted {
            cell: IVec2::ZERO,
            at,
            lo,
            hi,
        });
        return;
    }
    let here = cell_of(eye);
    // Re-place on a cell change, and otherwise once a second — the second condition is what picks up
    // water that streamed in while the player stood still.
    probe.place_in -= time.delta_secs();
    if probe.anchor == Some(here) && probe.place_in > 0.0 {
        return;
    }
    probe.anchor = Some(here);
    probe.place_in = 1.0;

    // A block wide enough to hold the slot count at this spacing, centred on the player's cell.
    let want = probe_slots().saturating_sub(PROBE_SPARE).max(1);
    let r = ((want as f32).sqrt().ceil() as i32).max(1) + 1;
    let eye_xz = Vec2::new(eye.x, eye.z);
    let mut found: Vec<(f32, IVec2, Vec3)> = Vec::new();
    for dz in -r..=r {
        for dx in -r..=r {
            let cell = here + IVec2::new(dx, dz);
            // Seated at the water inside the cell, not at the cell's own centre — see [`probe_seat`].
            let wet = |p: Vec2| {
                let (wx, wy) = wow_xy(p);
                chunks.iter().find_map(|c| c.surface_z_at(wx, wy))
            };
            let Some((seat, z, _)) = probe_seat(
                cell_centre(cell),
                probe_spacing(),
                probe_seat_samples(),
                &wet,
            ) else {
                continue;
            };
            found.push((
                seat.distance(eye_xz),
                cell,
                Vec3::new(seat.x, z + PROBE_LIFT, seat.y),
            ));
        }
    }
    // **Hysteresis: a cell already held counts as nearer than it is.** Without it two cells at
    // nearly the same distance trade the last slot back and forth every time placement runs, and
    // each trade costs a full fade-out, capture and fade-in — which is a blink, not a fade. The
    // discount has to be larger than the jitter in the distances and smaller than a lattice step.
    let held: Vec<IVec2> = probe
        .slots
        .iter()
        .filter(|s| s.captured && !s.retire)
        .filter_map(|s| s.cell)
        .collect();
    for f in found.iter_mut() {
        if held.contains(&f.1) {
            f.0 *= PROBE_KEEP_BIAS;
        }
    }
    found.sort_by(|a, b| a.0.total_cmp(&b.0));
    // Nearest-first, but spread — see [`PROBE_MIN_SEPARATION`].
    let sep = probe_spacing() * PROBE_MIN_SEPARATION;
    let mut spread: Vec<(f32, IVec2, Vec3)> = Vec::with_capacity(want);
    for c in found {
        if spread.len() >= want {
            break;
        }
        if spread
            .iter()
            .any(|(_, _, p)| Vec2::new(p.x, p.z).distance(Vec2::new(c.2.x, c.2.z)) < sep)
        {
            continue;
        }
        spread.push(c);
    }
    let found = spread;

    // Placement only says which cells are wanted; `next_capture` decides which slot takes each.
    probe.wanted = found
        .iter()
        .map(|&(_, cell, at)| {
            let (lo, hi) = derive_box(&chunks, at).unwrap_or((
                at - Vec3::splat(PROBE_BOX_FALLBACK),
                at + Vec3::splat(PROBE_BOX_FALLBACK),
            ));
            Wanted { cell, at, lo, hi }
        })
        .collect();
}

/// The time-sliced live probe: face camera 0 stays active (a view that goes inactive loses its
/// specialized pipelines — see [`PROBE_BURST`]) and renders face `live_face` into that face's
/// image, which the copy node blits into the back slot. The capture point holds for a cycle of six
/// faces; when a cycle completes the back slot is shown whole and the other hidden.
fn drive_live(
    probe: &mut WaterProbe,
    cull: &mut ProbeCull,
    cameras: &mut Query<(
        &ProbeFace,
        &mut Camera,
        &mut Transform,
        &mut bevy::camera::RenderTarget,
    )>,
) {
    let f = probe.live_face;
    if f == 0 {
        if let Some(done) = probe
            .live_cycle
            .filter(|_| probe.live_cycles >= LIVE_WARM_CYCLES)
        {
            let back = probe.live_back;
            let sl = &mut probe.slots[back];
            sl.cell = Some(done.cell);
            sl.at = done.at;
            sl.box_min = done.lo;
            sl.box_max = done.hi;
            sl.captured = true;
            sl.reach = 1.0;
            // The finished cube fades in over the next cycle while the one it replaces fades out;
            // the one before that has faded out entirely and is written next.
            let gone = probe.live_old;
            probe.live_old = probe.live_front;
            probe.live_front = Some(back);
            probe.live_back = gone.unwrap_or((back + 1) % 3);
            if probe.live_back == back || Some(probe.live_back) == probe.live_old {
                probe.live_back = (0..3)
                    .find(|i| Some(*i) != probe.live_front && Some(*i) != probe.live_old)
                    .unwrap_or(0);
            }
            let hidden = probe.live_back;
            probe.slots[hidden].reach = 0.0;
            probe.slots[hidden].mix = 0.0;
            probe.ever = true;
        }
        if probe.live_cycle.is_some() {
            probe.live_cycles = probe.live_cycles.saturating_add(1);
        }
        probe.live_cycle = probe.live_target;
    }
    // Crossfade across the cycle: the new cube's share rises by a sixth a frame as the old one's
    // falls, and both stay available, so the tier's strength does not dip.
    let t = (f + 1) as f32 / 6.0;
    if let Some(i) = probe.live_front {
        probe.slots[i].mix = if probe.live_old.is_some() { t } else { 1.0 };
    }
    if let Some(i) = probe.live_old {
        probe.slots[i].mix = 1.0 - t;
        if f == 5 {
            probe.slots[i].reach = 0.0;
        }
    }
    let Some(cycle) = probe.live_cycle else {
        probe.copy = false;
        probe.copy_face = None;
        return;
    };
    cull.at = cycle.at;
    cull.radius = PROBE_FADE_YD + PROBE_BOX_UP;
    cull.hold = PROBE_BURST + 4;
    cull.active = true;
    for (face, mut cam, mut xf, mut target) in cameras.iter_mut() {
        cam.is_active = face.0 == 0;
        if face.0 != 0 {
            continue;
        }
        let (look, up) = FACES[f];
        *xf = Transform::from_translation(cycle.at)
            .looking_to(Vec3::from_array(look), Vec3::from_array(up));
        *target = bevy::camera::RenderTarget::Image(probe.faces[f].clone().into());
    }
    probe.copy = true;
    probe.copy_face = Some(f);
    probe.copy_slot = probe.live_back;
    probe.capturing = None;
    probe.live_face = (f + 1) % 6;
}

/// Cycles the live probe renders before it is first shown, so its one camera has specialized the
/// pipelines of everything around it.
const LIVE_WARM_CYCLES: u32 = 3;

pub(super) fn drive_probe(
    style: Res<WaterStyle>,
    time: Res<Time>,
    eye: Query<&GlobalTransform, With<crate::view::WorldCamera>>,
    mut probe: ResMut<WaterProbe>,
    mut cull: ResMut<ProbeCull>,
    mut cameras: Query<(
        &ProbeFace,
        &mut Camera,
        &mut Transform,
        &mut bevy::camera::RenderTarget,
    )>,
) {
    // Capture order is by distance from here — see [`next_to_capture`].
    let eye_xz = eye
        .single()
        .map(|e| Vec2::new(e.translation().x, e.translation().z))
        .unwrap_or(Vec2::ZERO);
    // The hold spans the whole capture and outlives it by a few frames — see [`ProbeCull`].
    cull.hold = cull.hold.saturating_sub(1);
    cull.active = cull.hold > 0;
    // `wants_probe`, not `is_stylised`: this tier is armed on its own lane and nowhere else. With
    // the mirror running its contribution measured at zero pixels — see `WaterStyle::StylisedProbe`
    // — so arming it beside one would be capturing six faces a refresh in order to discard them.
    let want = style.wants_probe() && probe_enabled();
    probe.armed = want;
    probe.ever = probe.slots.iter().any(|s| s.captured);
    if !want {
        for (_, mut cam, _, _) in &mut cameras {
            cam.is_active = false;
        }
        probe.capturing = None;
        probe.copy = false;
        return;
    }

    if probe_live() {
        drive_live(&mut probe, &mut cull, &mut cameras);
        return;
    }
    probe.copy_face = None;

    // **One probe per burst, and the six cameras are shared between them.** Serialising the
    // captures is what makes the probe COUNT free: whatever the slot count, exactly one burst is
    // ever in flight, so the per-frame cost is the single-probe cost this tier already paid. The
    // price is latency on arrival somewhere new, which is a warm-up rather than a running cost.
    let Some(slot) = probe.capturing else {
        for (_, mut cam, _, _) in &mut cameras {
            cam.is_active = false;
        }
        probe.copy = false;

        // **A cube of a static world does not expire.** So the timer no longer drives captures; it
        // only marks one slot dirty per window, round-robin, to let a moving sun reach every cube
        // in `slots * refresh` seconds. The real trigger is a slot that has no cube for its current
        // cell, which is what placement produces when the player walks somewhere new.
        probe.refresh_in -= time.delta_secs();
        if probe.refresh_in <= 0.0 {
            probe.refresh_in = refresh_s();
            let n = probe.slots.len();
            if n > 0 {
                let c = probe.cursor % n;
                probe.cursor = (c + 1) % n;
                if probe.slots[c].cell.is_some() {
                    probe.slots[c].dirty = true;
                }
            }
        }

        probe.burst_gap = (probe.burst_gap - time.delta_secs()).max(0.0);
        let gap_ready = probe.burst_gap <= 0.0;
        let p = &mut *probe;
        let Some(next) = next_capture(&mut p.slots, &p.wanted, eye_xz, gap_ready) else {
            return;
        };
        probe.capturing = Some(next);
        probe.dwell = 0;
        // Arm the cull a frame BEFORE the faces render, so the burst never captures a world with
        // no trees in it. Around the TARGET, which is where the cameras are about to look.
        cull.at = probe.slots[next].want_at;
        cull.radius = PROBE_FADE_YD + PROBE_BOX_UP;
        cull.hold = PROBE_BURST + 4;
        cull.active = true;
        return;
    };

    let at = probe.slots[slot].want_at;
    // Re-arm every frame of the capture, so the hold always outruns it.
    cull.at = at;
    cull.radius = PROBE_FADE_YD + PROBE_BOX_UP;
    cull.hold = PROBE_BURST + 4;
    cull.active = true;

    // **Every face at once, for the whole burst.** Not one at a time: a view that went inactive
    // had its specialization cache dropped, so whichever face went first would be the one drawing
    // no terrain — see [`PROBE_BURST`]. Held together, the six warm up together and none of them
    // is the unlucky first.
    //
    // The six cameras move to whichever slot is being captured; they belong to the burst, not to a
    // probe. `PostUpdate` before `TransformSystems::Propagate`, so the pose lands this frame.
    for (face, mut cam, mut xf, _) in &mut cameras {
        cam.is_active = true;
        let (look, up) = FACES[face.0];
        *xf = Transform::from_translation(at)
            .looking_to(Vec3::from_array(look), Vec3::from_array(up));
    }

    // Copy on the LAST frame of the burst only. The early frames are exactly the ones whose
    // pipelines are still missing, and copying them would put the empty faces into the cube that
    // the warm frames were meant to replace.
    probe.dwell += 1;
    probe.copy = probe.dwell >= burst_frames();
    if !probe.copy {
        return;
    }
    probe.dwell = 0;
    probe.burst_gap = PROBE_BURST_GAP_S;
    probe.copy_slot = slot;
    if probe_trace() {
        warn!(
            "probe: slot {slot} {} -> {:?}",
            probe.slots[slot]
                .cell
                .map_or("(empty)".to_string(), |c| format!("{c:?}")),
            probe.slots[slot].want,
        );
    }
    // **Promote target to resident in one step**, on the same frame the blit writes the layers.
    // Cube contents, capture point and proxy change together or not at all; a slot is never showing
    // one place's cube through another place's projection.
    {
        let sl = &mut probe.slots[slot];
        sl.cell = sl.want;
        sl.at = sl.want_at;
        sl.box_min = sl.want_box_min;
        sl.box_max = sl.want_box_max;
        sl.captured = true;
        sl.dirty = false;
        sl.retire = false;
    }
    probe.ever = true;
    probe.capturing = None;
}

/// Re-point every liquid material at the cube. Change-gated on the handle, so a steady state is a
/// no-op — the same shape `scene_color`'s restamp takes.
fn restamp_probe(
    probe: Res<WaterProbe>,
    mut materials: ResMut<Assets<benilla_assets::materials::LiquidMaterial>>,
    mut stamped: Local<Option<AssetId<Image>>>,
) {
    if *stamped == Some(probe.cube.id()) {
        return;
    }
    *stamped = Some(probe.cube.id());
    for (_, material) in materials.iter_mut() {
        material.extension.probe = probe.cube.clone();
    }
}

/// Copy whichever faces have been rendered into the cube's layers.
/// The mirroring blit that writes each rendered face into its cube layer.
///
/// See `shaders/probe_face.wgsl`: a right-handed camera cannot render a cube face in the face's own
/// left-handed basis, so the face comes out horizontally mirrored and is un-mirrored here. A plain
/// `copy_texture_to_texture` cannot do that, which is why this pass exists at all.
#[derive(Resource)]
struct ProbeBlit {
    layout: BindGroupLayoutDescriptor,
    sampler: bevy::render::render_resource::Sampler,
    pipeline: CachedRenderPipelineId,
}

fn init_probe_blit(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    fullscreen_shader: Res<FullscreenShader>,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
) {
    let shader: Handle<Shader> =
        asset_server.load("embedded://benilla_world/shaders/probe_face.wgsl");
    let layout = BindGroupLayoutDescriptor::new(
        "probe_face_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                texture_2d(TextureSampleType::Float { filterable: true }),
                sampler(SamplerBindingType::Filtering),
                texture_depth_2d(),
            ),
        ),
    );
    let sampler_handle = render_device.create_sampler(&SamplerDescriptor {
        // CLAMP, not REPEAT: a face edge must not fetch from the far side of its own image, which
        // would smear the opposite wall of the world into the seam this pass exists to close.
        address_mode_u: bevy::render::render_resource::AddressMode::ClampToEdge,
        address_mode_v: bevy::render::render_resource::AddressMode::ClampToEdge,
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        ..default()
    });
    let pipeline = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("probe_face_mirror".into()),
        layout: vec![layout.clone()],
        vertex: fullscreen_shader.to_vertex_state(),
        fragment: Some(FragmentState {
            shader,
            shader_defs: vec![],
            entry_point: Some("fs_mirror".into()),
            targets: vec![Some(ColorTargetState {
                format: TextureFormat::Rgba16Float,
                blend: None,
                write_mask: ColorWrites::ALL,
            })],
        }),
        ..default()
    });
    commands.insert_resource(ProbeBlit {
        layout,
        sampler: sampler_handle,
        pipeline,
    });
}

/// The prefilter's pipeline — see `shaders/probe_filter.wgsl` for what it is for and why it is a
/// cone rather than Bevy's GGX importance sampler.
#[derive(Resource)]
struct ProbeFilter {
    layout: BindGroupLayoutDescriptor,
    sampler: bevy::render::render_resource::Sampler,
    pipeline: CachedRenderPipelineId,
}

fn init_probe_filter(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    fullscreen_shader: Res<FullscreenShader>,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
) {
    let shader: Handle<Shader> =
        asset_server.load("embedded://benilla_world/shaders/probe_filter.wgsl");
    let layout = BindGroupLayoutDescriptor::new(
        "probe_filter_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                bevy::render::render_resource::binding_types::texture_cube_array(
                    TextureSampleType::Float { filterable: true },
                ),
                sampler(SamplerBindingType::Filtering),
                bevy::render::render_resource::binding_types::uniform_buffer::<Vec4>(false),
            ),
        ),
    );
    let sampler_handle = render_device.create_sampler(&SamplerDescriptor {
        // The cone reaches across face edges by construction — that is the point of filtering a
        // cube rather than six pictures — so this samples the CUBE, where the hardware crosses
        // seams for us, and the address modes never come into it.
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        ..default()
    });
    let pipeline = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("probe_filter".into()),
        layout: vec![layout.clone()],
        vertex: fullscreen_shader.to_vertex_state(),
        fragment: Some(FragmentState {
            shader,
            shader_defs: vec![],
            entry_point: Some("fs_filter".into()),
            targets: vec![Some(ColorTargetState {
                format: TextureFormat::Rgba16Float,
                blend: None,
                write_mask: ColorWrites::ALL,
            })],
        }),
        ..default()
    });
    commands.insert_resource(ProbeFilter {
        layout,
        sampler: sampler_handle,
        pipeline,
    });
}

/// The cone half-angle for each level of the chain, in radians.
///
/// Each level convolves the one above it, so these compound: the width at level `n` is roughly the
/// sum of the levels before it, which is why they start small. The top of the chain is a gentle
/// softening the water uses right next to a probe — never fully sharp, because a probe that looks
/// like a mirror invites the comparison it loses — and the bottom is wide enough to be an ambient
/// term with no recognisable geometry left in it at all.
fn probe_cone(level: u32) -> f32 {
    // 3, 6, 12, 22, 36 degrees as the chain descends.
    const CONES: [f32; 5] = [0.052, 0.105, 0.209, 0.384, 0.628];
    CONES[(level as usize - 1).min(CONES.len() - 1)]
}

/// Holds a query for the face views, so the blit can reach each face's depth texture. A plain
/// `Node` rather than a `ViewNode` because this pass is not per-view: it runs once and writes six
/// layers, and the views it reads from are the capture cameras rather than the view it runs under.
struct WaterProbeNode {
    faces: bevy::ecs::query::QueryState<(
        &'static ProbeFace,
        &'static bevy::render::view::ViewDepthTexture,
    )>,
}

impl FromWorld for WaterProbeNode {
    fn from_world(world: &mut World) -> Self {
        Self {
            faces: bevy::ecs::query::QueryState::new(world),
        }
    }
}

impl Node for WaterProbeNode {
    fn update(&mut self, world: &mut World) {
        self.faces.update_archetypes(world);
    }

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let Some(probe) = world.get_resource::<WaterProbe>() else {
            return Ok(());
        };
        if !probe.armed || !probe.copy {
            return Ok(());
        }
        let Some(images) = world.get_resource::<RenderAssets<GpuImage>>() else {
            return Ok(());
        };
        let Some(cube) = images.get(&probe.cube) else {
            return Ok(()); // the cube has not reached the render world yet
        };
        let Some(blit) = world.get_resource::<ProbeBlit>() else {
            return Ok(());
        };
        let Some(cache) = world.get_resource::<PipelineCache>() else {
            return Ok(());
        };
        let Some(pipeline) = cache.get_render_pipeline(blit.pipeline) else {
            return Ok(()); // still compiling; the cube keeps last refresh's faces
        };
        // Pair each face index with its view's depth texture. A face whose depth has not been
        // allocated yet is skipped rather than blitted without distance — half a cube with alpha and
        // half without is the kind of state that looks fine and reads wrong.
        let mut depth: [Option<&bevy::render::render_resource::TextureView>; 6] = [None; 6];
        for (face, tex) in self.faces.iter_manual(world) {
            if let Some(slot) = depth.get_mut(face.0) {
                *slot = Some(tex.view());
            }
        }

        for (i, handle) in probe.faces.iter().enumerate() {
            // Time-sliced: only the face rendered this frame, and its depth is face camera 0's.
            let face_depth = match probe.copy_face {
                Some(f) if f != i => continue,
                Some(_) => depth[0],
                None => depth[i],
            };
            let Some(face) = images.get(handle) else {
                continue;
            };
            let Some(face_depth) = face_depth else {
                continue; // no depth for this face yet
            };
            if face.size.width != cube.size.width || face.size.height != cube.size.height {
                continue;
            }
            // One D2 view onto this layer of the cube — the cube's own view is a Cube view and
            // cannot be a render target.
            // Six layers per probe, so this slot's faces start at `slot * 6` — the same packing
            // the shader's `array_index` assumes.
            let dst = cube.texture.create_view(&TextureViewDescriptor {
                label: Some("probe_cube_layer"),
                dimension: Some(TextureViewDimension::D2),
                base_array_layer: (probe.copy_slot * 6 + i) as u32,
                array_layer_count: Some(1),
                // **Level 0 explicitly, now that the cube has a chain.** A render target may carry
                // exactly one mip level, and a view that defaults to "all of them" is not
                // renderable — which is a hard validation failure rather than a wrong picture, and
                // the reason this line exists at all.
                base_mip_level: 0,
                mip_level_count: Some(1),
                ..default()
            });
            let bind = render_context.render_device().create_bind_group(
                "probe_face_bind",
                &cache.get_bind_group_layout(&blit.layout),
                &BindGroupEntries::sequential((&face.texture_view, &blit.sampler, face_depth)),
            );
            let mut pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
                label: Some("probe_face_mirror"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &dst,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations::default(),
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_render_pipeline(pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.draw(0..3, 0..1);
        }

        // ---- the prefilter chain ---------------------------------------------------------------
        //
        // Runs on the slot that was just blitted, and only on it: the chain is a function of that
        // slot's own base level, so a slot whose faces did not change this frame has nothing to
        // redo. See `shaders/probe_filter.wgsl` for what the water does with the result.
        //
        // Level `l` reads level `l - 1`, so the passes MUST go top down and cannot be reordered or
        // batched — each one's source is the previous one's target.
        let mips = cube.texture.mip_level_count();
        let (Some(filter), Some(filter_pipeline)) = (
            world.get_resource::<ProbeFilter>(),
            world
                .get_resource::<ProbeFilter>()
                .and_then(|f| cache.get_render_pipeline(f.pipeline)),
        ) else {
            return Ok(()); // still compiling; the cube keeps level 0 and the water reads it sharp
        };
        // One uniform per (level, face), padded to the 256-byte offset alignment uniforms require.
        const STRIDE: usize = 256;
        let mut cfg = vec![0u8; STRIDE * 6 * mips.max(1) as usize];
        for level in 1..mips {
            for face in 0..6u32 {
                let at = STRIDE * ((level as usize - 1) * 6 + face as usize);
                let v = [face as f32, probe.copy_slot as f32, probe_cone(level), 0.0];
                cfg[at..at + 16].copy_from_slice(bytemuck::cast_slice(&v));
            }
        }
        let cfg_buf = render_context.render_device().create_buffer_with_data(
            &bevy::render::render_resource::BufferInitDescriptor {
                label: Some("probe_filter_cfg"),
                contents: &cfg,
                usage: bevy::render::render_resource::BufferUsages::UNIFORM,
            },
        );
        for level in 1..mips {
            // The whole array at the level above, as a cube array — the cone crosses face edges and
            // needs the hardware's seamless cube filtering, which a D2 view would not give it.
            let src = cube.texture.create_view(&TextureViewDescriptor {
                label: Some("probe_filter_src"),
                dimension: Some(TextureViewDimension::CubeArray),
                base_mip_level: level - 1,
                mip_level_count: Some(1),
                ..default()
            });
            for face in 0..6u32 {
                let dst = cube.texture.create_view(&TextureViewDescriptor {
                    label: Some("probe_filter_dst"),
                    dimension: Some(TextureViewDimension::D2),
                    base_mip_level: level,
                    mip_level_count: Some(1),
                    base_array_layer: probe.copy_slot as u32 * 6 + face,
                    array_layer_count: Some(1),
                    ..default()
                });
                let at = (STRIDE * ((level as usize - 1) * 6 + face as usize)) as u64;
                let bind = render_context.render_device().create_bind_group(
                    "probe_filter_bind",
                    &cache.get_bind_group_layout(&filter.layout),
                    &BindGroupEntries::sequential((
                        &src,
                        &filter.sampler,
                        bevy::render::render_resource::BindingResource::Buffer(
                            bevy::render::render_resource::BufferBinding {
                                buffer: &cfg_buf,
                                offset: at,
                                size: core::num::NonZeroU64::new(16),
                            },
                        ),
                    )),
                );
                let mut pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
                    label: Some("probe_filter"),
                    color_attachments: &[Some(RenderPassColorAttachment {
                        view: &dst,
                        depth_slice: None,
                        resolve_target: None,
                        ops: Operations::default(),
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                pass.set_render_pipeline(filter_pipeline);
                pass.set_bind_group(0, &bind, &[]);
                pass.draw(0..3, 0..1);
            }
        }
        Ok(())
    }
}

/// The probe's two uniform rows, for [`super::reflect::WaterReflectData`] to publish.
///
/// Returned rather than written here because the reflection buffer has one writer, and a second
/// system reaching into it is how two tiers end up disagreeing about which frame they are on.
pub(crate) fn probe_lanes(probe: Option<&WaterProbe>) -> ([f32; 4], [f32; 4], [f32; 4], [f32; 4]) {
    // `ever`, not `pending.is_none()`: a REFRESH must not blank the tier while it runs, or the
    // water flickers back to the sky mix for six frames every time the timer comes round.
    let live = probe.is_some_and(|p| p.armed && p.ever);
    if std::env::var("WOW_PROBE_TRACE").as_deref() == Ok("1") {
        use std::sync::atomic::{AtomicU32, Ordering};
        static TICK2: AtomicU32 = AtomicU32::new(0);
        let t2 = TICK2.fetch_add(1, Ordering::Relaxed);
        if let (0, Some(p)) = (t2 % 120, probe) {
            for (i, slot) in p.slots.iter().enumerate() {
                let Some(cell) = slot.cell else { continue };
                warn!(
                    "probe slot {i}: cell {:?} at {:?} half-extent {:?} yd | captured {} dirty {} \
                     reach {:.3}",
                    cell,
                    slot.at,
                    (slot.box_max - slot.box_min) * 0.5,
                    slot.captured,
                    slot.dirty,
                    slot.reach,
                );
            }
        }
    }
    (
        // The CAPTURE point, not the water anchor: a cubemap's directions are only meaningful about
        // the place it was taken from, so the box projection has to subtract THIS position.
        //
        // `w` marks the lane as **reflection-only** — and it is now OFF by default, because the
        // reason for it has expired.
        //
        // While the probe was being made correct this lane drew its reflection and nothing else: no
        // body colour, no Fresnel weight, no depth tint. A reflection delivered at four to seven
        // per cent over a lit green lake cannot be compared against anything, every error in it is
        // invisible, and that is how three wrong diagnoses survived in a row. Painting it flat made
        // "does this match what a mirror would show" a question the eye could answer.
        //
        // The cube is correct now — the faces are un-mirrored (see `shaders/probe_face.wgsl`), the
        // burst captures all six together (see [`PROBE_BURST`]) and the live reflection scores
        // +0.312 against the shore and -0.210 against the shore flipped. So the tier goes back to
        // being water: Fresnel, body and depth tint like every other tier. `$WOW_PROBE_RAW=1`
        // restores the flat reflection, which is still the right first move on any probe that
        // starts looking wrong.
        [
            PROBE_AT.x,
            PROBE_AT.y + PROBE_LIFT,
            PROBE_AT.z,
            f32::from(probe.is_some_and(|p| p.armed) && probe_raw()),
        ],
        // A pure gate now: the look knob is a Fresnel boost two rows down, because scaling the
        // tier's own weight could not make it more visible — it already owns the composited
        // reflection outright wherever it contributes.
        [
            f32::from(live),
            PROBE_FADE_YD,
            PROBE_FADE_UP,
            // The debug lane is a small enum, not a flag: 0 = off, 1 = this tier's own sample,
            // 2 = the whole composited reflection. Two switches would have wanted two lanes, and
            // this row has no spare.
            if probe_dist_show() {
                4.0
            } else if probe_cube_show() {
                3.0
            } else if refl_show() {
                2.0
            } else {
                f32::from(probe_show())
            },
        ],
        // **These two rows are `w` carriers now and nothing else.** The proxy box moved onto the
        // slots, where it belongs — every probe needs its own, because a box fitted to one capture
        // point's surroundings says nothing about another's. The `xyz` here would be a box for a
        // probe that no longer exists, so it is written as zero rather than left looking meaningful.
        //
        // `w` carries the surface ripple scale — see [`ripple_scale`].
        // `x` = `$WOW_PROBE_FORCE`, the single-slot debug; `y` the angle a mip-0 texel subtends,
        // `z` the blur floor — both read by `liquid.wgsl`'s `probe_lod`; `w` the ripple scale.
        [
            probe_force(),
            core::f32::consts::FRAC_PI_2 / PROBE_FACE as f32,
            if probe_blur() { probe_lod_floor() } else { 0.0 },
            ripple_scale(),
        ],
        // `x` = the probe tier's Fresnel boost; `y` the tap count; `z` the deepest mip the water
        // may read; `w` spare. The sphere radius is per-slot — see [`probe_slot_lanes`].
        [
            probe_strength(),
            probe_taps(),
            (probe_mips() - 1) as f32,
            0.0,
        ],
    )
}

/// The per-slot rows: capture point + validity, proxy minimum, proxy maximum, for every slot the
/// shader's fixed-length array can hold.
///
/// A slot that holds no cube writes `w = 0` on its first row and the shader skips it — that flag is
/// the only thing standing between the water and a cube of somewhere else, which is a failure mode
/// that looks entirely plausible and so has to be made structural rather than watched for.
/// `$WOW_PROBE_PROXY=box` — use the analytic proxy shape instead of the cube's stored distances.
///
/// The depth correction is the default because it is the one that does not need a per-site number.
/// This exists so the two can be compared inside one build, which is the only way the comparison is
/// worth anything: across two builds the world streams differently and the capture noise floor
/// (about 4.5 % of pixels between identical runs) swallows the difference.
pub(crate) fn probe_depth_proxy() -> f32 {
    static D: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    // 2 = march the reflected ray against the stored distances (default), 1 = the fixed-point
    // correction (`$WOW_PROBE_PROXY=fixed`), 0 = the analytic box (`$WOW_PROBE_PROXY=box`).
    *D.get_or_init(|| match std::env::var("WOW_PROBE_PROXY").as_deref() {
        Ok("box") => 0.0,
        Ok("fixed") => 1.0,
        _ => 2.0,
    })
}

/// `$WOW_PROBE_MARCH=<n>` — refinement steps for the depth correction, clamped to 1..=8.
///
/// Four is where the picture stops changing at these face sizes. Each step is one more cube fetch
/// per water fragment, and this is the tier's only genuinely per-frame cost, so it is worth knowing
/// what the lower numbers give.
pub(crate) fn probe_march_steps() -> f32 {
    static M: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *M.get_or_init(|| {
        std::env::var("WOW_PROBE_MARCH")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .map(|v| v.clamp(1, 8))
            .unwrap_or(4) as f32
    })
}

/// `$WOW_PROBE_DIST=1` — unwrap the cube's stored DISTANCE instead of its colour.
///
/// The cube's alpha carries yards from the capture point (see `shaders/probe_face.wgsl`). A cube
/// holding the right picture at the wrong distances gives a reflection that is entirely plausible
/// and in the wrong place, which is indistinguishable from a dozen other faults by eye — so the
/// distances get their own direct view rather than being inferred from the water.
fn probe_dist_show() -> bool {
    std::env::var("WOW_PROBE_DIST").as_deref() == Ok("1")
}

/// `$WOW_PROBE_STRENGTH=<f>` — a dimmer on the whole tier, on top of the Fresnel weight.
///
/// The planar mirror reaches the water through four attenuations —
/// `params.y * mirrored.a * trust * (1 - ssr.conf)` — where this tier had only its distance
/// falloff, which is 1.0 over most of a lake. Restoring the march stand-down (see the composite in
/// `liquid.wgsl`) removes the structural half of that gap; this covers the rest, and exists because
/// the probe has no equivalent of the capture's own `params.y` strength lane.
pub(crate) fn probe_strength() -> f32 {
    static S: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *S.get_or_init(|| {
        std::env::var("WOW_PROBE_STRENGTH")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .map(|v| v.clamp(0.0, 12.0))
            // **0.8 on Stefan's report that the tier reads heavier than the mirror, and the
            // measurement here does NOT reproduce that.** At Mirror Lake, across two cameras, the
            // probe at full strength already sits slightly *under* the planar tier — water-pixel
            // standard deviation 6.67 against the mirror's 8.35 at one camera, 6.11 against 6.75 at
            // the other — and the signed difference map is mixed red and blue, which is a content
            // disagreement rather than a uniform gain.
            //
            // Two static cameras at one lake is exactly the evidence that has been wrong before, so
            // the report wins over the metric and the dimmer is real. Note the sign is
            // scene-dependent: where the reflection is DARKER than the water body, as it is under a
            // canopy here, turning this down makes the surface brighter rather than dimmer. It is a
            // weight on the tier, not a brightness.
            // **1.0 — parity with the planar mirror, measured.** This was pushed to 4.0 to answer
            // "the reflection is barely visible", and at 4.0 the surface reads as liquid metal:
            // water-band contrast std 23.2 against the mirror's 10.1, with the ripple moving it
            // three times as far. At 1.0 the three looks measure within a tenth of each other —
            // std 10.09 planar, 10.63 march, 9.71 probe — which is the parity this tier is meant
            // to have. Faintness is a property of the shared Fresnel weight and belongs to all
            // three tiers at once; it is not a thing to fix inside this one.
            .unwrap_or(1.0)
    })
}

/// `$WOW_PROBE_FORCE=<n>` — make every water fragment read slot `n` and nothing else.
///
/// **The instrument for "how visible is the change from one probe to another".** A cross-fade
/// between probes is a dissolve between two pictures, so what the transition looks like is decided
/// by how much those two pictures disagree AFTER the parallax correction has had its say. Comparing
/// raw cubes answers a different and easier question — of course two captures from different places
/// differ; the correction exists to reconcile them.
///
/// Forcing each slot in turn and differencing the rendered water measures the residual the blend
/// actually has to hide. It is also the yardstick for whether density, a sharper depth correction,
/// or prefiltered mips is the right lever: all three work by shrinking this number.
///
/// `-1`, the default, leaves selection alone.
pub(crate) fn probe_force() -> f32 {
    static F: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *F.get_or_init(|| {
        std::env::var("WOW_PROBE_FORCE")
            .ok()
            .and_then(|v| v.parse::<i32>().ok())
            .map(|v| v.clamp(-1, PROBE_SLOT_MAX as i32 - 1))
            .unwrap_or(-1) as f32
    })
}

/// `$WOW_PROBE_LOD_FLOOR=<f>` — the softest the probe is ever read, even standing on top of one.
///
/// `liquid.wgsl`'s `probe_lod` derives how much to blur from the parallax error it has to hide, and
/// at the capture point that error is zero, so the derivation alone would read the cube perfectly
/// sharp there. This stops it. The reason is a look decision rather than a correction: a probe that
/// is sometimes a crisp mirror invites the comparison with the planar tier that it loses, and the
/// switch between sharp-at-the-probe and soft-between-probes is itself the artefact that started
/// this whole line of work. A tier that is never sharp has nothing to switch between.
///
/// # What was tried before the blur, and did not work
///
/// The probe tier spent a long time trying to be geometrically correct and every implementation-
/// quality explanation for its failure was eliminated in turn. A world-space landing consensus,
/// which resolves the reflected ray's hit point and blends probes by whether they can see it, moved
/// displacement against the planar mirror from 23.63 to 17.11 on a lattice stretched to 40 yards and
/// was a WASH at the 12-yard spacing this ships — near a probe there is nothing for it to improve.
/// Raising the march from 4 steps to 12: no effect. Raising the cube face from 256 to 512: no
/// effect. The consensus visibility test's tolerance across a 66-fold sweep: no effect, and its
/// diagnostic said why — two probes on open water see the same surfaces, so 96-99% of fragments
/// accepted every probe with nothing in between.
///
/// What is left is inherent: a cube is exact at its capture point and degrades with distance from
/// it. So the tier stopped trying to be right and started being unreadable instead, which is the
/// same conclusion every shipping engine reached.
fn probe_lod_floor() -> f32 {
    static F: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *F.get_or_init(|| {
        std::env::var("WOW_PROBE_LOD_FLOOR")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .map(|v| v.clamp(0.0, 6.0))
            .unwrap_or(0.75)
    })
}

/// `$WOW_PROBE_TAPS=<2|3>` — how many probes may be blended into one fragment.
///
/// **Three by default, and more than three is not on offer.** Blending more cubes does not make the
/// reflection more correct: their errors are displacements in different directions, so averaging
/// ghosts rather than converges — which is why Unity caps its own probe blend at two. What extra
/// taps buy is a smoother HANDOVER, and that is worth paying for here because continuity outranks
/// per-frame sharpness in this tier.
///
/// The cost is not what it looks like. Considering every slot is distance arithmetic and free;
/// sampling one is a colour fetch plus its depth march, five cube reads at the default depth. Since
/// the march depth now follows the blend weight — full for the dominant tap, one step for the minor
/// ones, on the grounds that a probe contributing a sixth of the picture does not need its parallax
/// resolved precisely — three taps land at roughly what two used to cost.
pub(crate) fn probe_taps() -> f32 {
    static T: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *T.get_or_init(|| {
        std::env::var("WOW_PROBE_TAPS")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .map(|v| v.clamp(1, 3))
            .unwrap_or(3) as f32
    })
}

/// `$WOW_PROBE_SLOT=<n>` — which slot the `$WOW_PROBE_CUBE` lat-long debug unwraps.
///
/// The unwrap has to show ONE probe. A lat-long of "whichever probe is nearest this fragment" is a
/// mosaic of several cubes and cannot answer the question the debug exists for, which is whether a
/// given cube holds the right place at the right orientation.
pub(crate) fn probe_debug_slot() -> f32 {
    static N: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("WOW_PROBE_SLOT")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .map(|v| v.min(PROBE_SLOT_MAX - 1) as f32)
            .unwrap_or(0.0)
    })
}

pub(crate) fn probe_slot_lanes(probe: Option<&WaterProbe>) -> [f32; PROBE_SLOT_MAX * 12] {
    let mut out = [0.0f32; PROBE_SLOT_MAX * 12];
    let Some(probe) = probe else {
        return out;
    };
    if !probe.armed {
        return out;
    }
    for (i, slot) in probe.slots.iter().take(PROBE_SLOT_MAX).enumerate() {
        if !slot.captured || slot.reach <= 0.001 {
            continue;
        }
        // Scale the proxy about THIS probe's capture point, which is what its projection rebases
        // onto — see [`box_scale`], which is 1.0 and a diagnostic.
        let k = box_scale();
        let (lo, hi) = (
            slot.at + (slot.box_min - slot.at) * k,
            slot.at + (slot.box_max - slot.at) * k,
        );
        let k = i * 12;
        // `w` is the fade, not a flag: the shader weights the probe by it, so a slot on its way
        // somewhere else leaves the picture smoothly instead of vanishing on one frame.
        // `at.w` is REACH — has a cube, and how far it carries — which is what makes a slot
        // eligible at all. `box_min.w` is its share of the blend — see [`ProbeSlot::mix`].
        out[k..k + 4].copy_from_slice(&[slot.at.x, slot.at.y, slot.at.z, slot.reach]);
        out[k + 4..k + 8].copy_from_slice(&[lo.x, lo.y, lo.z, slot.mix]);
        out[k + 8..k + 12].copy_from_slice(&[hi.x, hi.y, hi.z, probe_sphere()]);
    }
    out
}

/// **Characters stay out of the probe's cube.** A unit stands next to the probe — the viewer's
/// own body is always right under it — so it fills a large part of the cube and the water across
/// from it reflects the body, or reads it through where it should see past. The screen-space
/// march reflects bodies on screen; the cube keeps the surroundings. Each face camera's visible
/// list is rebuilt by the visibility pass every frame, so the unit meshes are dropped from it
/// after that pass, for these cameras alone.
fn keep_units_out_of_probe(
    units: Query<Entity, With<crate::world_unit::WorldUnit>>,
    children: Query<&Children>,
    mut faces: Query<(&Camera, &mut bevy::camera::visibility::VisibleEntities), With<ProbeFace>>,
) {
    if !faces.iter().any(|(cam, _)| cam.is_active) {
        return;
    }
    let mut unit_meshes = bevy::ecs::entity::EntityHashSet::default();
    for unit in &units {
        unit_meshes.insert(unit);
        unit_meshes.extend(children.iter_descendants(unit));
    }
    if unit_meshes.is_empty() {
        return;
    }
    for (cam, mut visible) in &mut faces {
        if !cam.is_active {
            continue;
        }
        for list in visible.entities.values_mut() {
            list.retain(|e| !unit_meshes.contains(e));
        }
    }
}

pub(super) fn register(app: &mut App) {
    let mut images = app.world_mut().resource_mut::<Assets<Image>>();
    let cube = images.add(cube_image());
    let faces = std::array::from_fn(|_| images.add(face_image()));
    app.insert_resource(WaterProbe {
        cube,
        faces,
        dwell: 0,
        copy: false,
        copy_slot: 0,
        refresh_in: PROBE_FIRST_S,
        ever: false,
        armed: false,
        slots: vec![ProbeSlot::default(); probe_slots()],
        capturing: None,
        anchor: None,
        place_in: 0.0,
        wanted: Vec::new(),
        live_target: None,
        live_cycle: None,
        live_face: 0,
        live_back: 0,
        live_front: None,
        live_old: None,
        live_cycles: 0,
        copy_face: None,
        burst_gap: 0.0,
        cursor: 0,
    })
    .init_resource::<ProbeCull>()
    .init_resource::<ProbeMarks>()
    .add_plugins((
        ExtractResourcePlugin::<WaterProbe>::default(),
        bevy::render::extract_component::ExtractComponentPlugin::<ProbeFace>::default(),
    ))
    .add_systems(Startup, setup_probe)
    .add_systems(
        PostUpdate,
        (
            place_probes,
            drive_probe,
            ramp_probe_fades,
            publish_probe_marks,
            restamp_probe,
        )
            .chain()
            .before(bevy::transform::TransformSystems::Propagate)
            .before(bevy::camera::CameraUpdateSystems)
            .after(super::reflect::drive_reflection),
    )
    .add_systems(
        PostUpdate,
        (keep_units_out_of_probe, trace_faces)
            .chain()
            .after(bevy::camera::visibility::VisibilitySystems::CheckVisibility),
    )
    .init_gizmo_group::<ProbeMarkGizmos>()
    .add_systems(Update, draw_probe_gizmos)
    .add_systems(Startup, confine_probe_gizmos);

    // Before the main opaque pass: the faces are rendered by their own cameras earlier in the frame
    // (negative order), and the copy only has to land before anything samples the cube.
    let Some(render_app) = app.get_sub_app_mut(bevy::render::RenderApp) else {
        return;
    };
    render_app
        .add_systems(
            bevy::render::RenderStartup,
            (init_probe_blit, init_probe_filter),
        )
        .add_render_graph_node::<WaterProbeNode>(Core3d, WaterProbeLabel)
        .add_render_graph_edges(Core3d, (WaterProbeLabel, Node3d::MainOpaquePass));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **Every hexagon must round-trip**, because `place_probes` identifies a probe by its cell and
    /// the shader is handed the centre: a cell that does not survive `centre -> cell_of` would make
    /// placement retarget a slot it already holds, every frame, forever.
    #[test]
    fn a_hex_cell_survives_the_trip_through_its_own_centre() {
        for q in -6..=6 {
            for r in -6..=6 {
                let cell = IVec2::new(q, r);
                let c = cell_centre(cell);
                let back = cell_of(Vec3::new(c.x, 0.0, c.y));
                assert_eq!(
                    cell, back,
                    "hex {cell:?} centre {c:?} came back as {back:?}"
                );
            }
        }
    }

    /// Walk a straight stream and report, for every frame, how far the nearest USABLE probe is.
    ///
    /// A deliberately small model of the real loop: a chain of water cells along one axis, a player
    /// walking it at running speed, placement re-running once a second and keeping the nearest
    /// slots, and the real [`next_to_capture`] policy driving one burst at a time. No GPU, no Bevy,
    /// no world — which is the point. The symptom it exists for ("the next probe only comes alive
    /// when I am almost there") is a scheduling question, and scheduling can be answered on a
    /// bench far faster than by walking a river and squinting.
    ///
    /// Returns `(smallest lead ahead of the player, frames with no probe ahead at all)`.
    /// **How far from the bank a stream probe actually sits**, in yards, over the probes the
    /// spread selection would really use.
    ///
    /// This is the direct measure of "are the probes centred". A cube's worst artefact on a narrow
    /// channel is that geometry close to the capture point is magnified in the reflection — Unreal
    /// documents exactly that — so a probe seated two yards off a bank paints that bank across the
    /// water. A perfectly centred probe on a channel `w` wide has clearance `w/2`; one sitting on
    /// the bank has zero.
    ///
    /// Returns `(mean, worst, ideal)`. `seated` picks between the real [`probe_seat`] and the older
    /// rule of taking the cell centre when the centre happens to be wet, so the two can be compared
    /// on one run.
    fn stream_seat_clearance(width: f32, amplitude: f32, seated: bool, n: i32) -> (f32, f32, f32) {
        let wavelength = 60.0_f32;
        let spacing = 12.0_f32;
        let slots_n = 16usize;
        let bank = |x: f32| amplitude * (x / wavelength).sin();
        let wet = |p: Vec2| ((p.y - bank(p.x)).abs() <= width * 0.5).then_some(0.0f32);
        // Perpendicular distance to the nearer bank. The channel's centreline wanders, so this is
        // measured against the bank BELOW the seat rather than against a straight line.
        let clearance = |p: Vec2| (width * 0.5 - (p.y - bank(p.x)).abs()).max(0.0);

        let mut all: Vec<f32> = Vec::new();
        for step in 0..120 {
            let x = step as f32 * 2.0;
            let eye = Vec2::new(x, bank(x));
            let mut cand: Vec<(f32, Vec2)> = Vec::new();
            for q in -6..=6 {
                for r in -3..=3 {
                    let cell = IVec2::new((x / spacing).round() as i32 + q, r);
                    let seat = if seated {
                        probe_seat(cell_centre(cell), spacing, n, &wet).map(|(p, _, _)| p)
                    } else {
                        let c = cell_centre(cell);
                        wet(c).map(|_| c)
                    };
                    let Some(seat) = seat else { continue };
                    cand.push((seat.distance(eye), seat));
                }
            }
            cand.sort_by(|a, b| a.0.total_cmp(&b.0));
            let sep = spacing * PROBE_MIN_SEPARATION;
            let mut spread: Vec<Vec2> = Vec::new();
            for (_, p) in cand {
                if spread.len() >= slots_n {
                    break;
                }
                if spread.iter().any(|q| q.distance(p) < sep) {
                    continue;
                }
                spread.push(p);
            }
            // **The clearance of the probe a fragment would actually READ, not the mean over every
            // probe placed.** Water is shaded from its nearest probe, so a cell seated off toward
            // the bank harms nothing while a better-centred probe sits closer to the water being
            // drawn. Averaging over all placed probes counts those as damage and reports a
            // regression where the picture has none — it did, on a straight sixteen-yard channel,
            // where extra row-offset cells are admitted beside the centred ones that still serve
            // mid-channel.
            let here = Vec2::new(x, bank(x));
            if let Some(p) = spread
                .iter()
                .min_by(|a, b| a.distance(here).total_cmp(&b.distance(here)))
            {
                all.push(clearance(*p));
            }
        }
        let mean = all.iter().sum::<f32>() / all.len() as f32;
        let worst = all.iter().copied().fold(f32::MAX, f32::min);
        (mean, worst, width * 0.5)
    }

    /// The bench behind "we need to center the probes more, especially in streams".
    ///
    /// Prints rather than only asserting, because the number that matters is how far off centre the
    /// probes are TODAY — the assertion is a tripwire so the figures in the report cannot rot
    /// silently.
    #[test]
    fn stream_probes_sit_off_centre_and_this_says_by_how_much() {
        println!("  -- the RULE, at the shipping grid --");
        println!(
            "{:>7}{:>10}{:>12}{:>9}{:>9}{:>10}",
            "width", "amplitude", "seating", "mean", "worst", "centred"
        );
        for width in [4.0_f32, 6.0, 10.0, 16.0] {
            for amplitude in [0.0_f32, 8.0] {
                for (name, seated) in [("centre", false), ("clearance", true)] {
                    let (mean, worst, ideal) =
                        stream_seat_clearance(width, amplitude, seated, PROBE_SEAT_SAMPLES);
                    println!(
                        "{width:>7.0}{amplitude:>10.0}{name:>12}{mean:>9.2}{worst:>9.2}{:>9.0}%",
                        100.0 * mean / ideal,
                    );
                }
            }
        }
        println!("\n  -- the GRID, on a wandering channel (clearance seating) --");
        println!(
            "{:>7}{:>7}{:>9}{:>9}{:>10}",
            "width", "n", "mean", "worst", "centred"
        );
        for width in [4.0_f32, 6.0, 10.0] {
            for n in [5_i32, 7, 9, 13, 19] {
                let (mean, worst, ideal) = stream_seat_clearance(width, 8.0, true, n);
                println!(
                    "{width:>7.0}{n:>7}{mean:>9.2}{worst:>9.2}{:>9.0}%",
                    100.0 * mean / ideal,
                );
            }
        }
        // **The claim this rule is kept on**, stated where it can fail: on channels that WANDER
        // off the lattice row — the case that prompted it — a clearance seat puts markedly more
        // water between the probe and the nearest bank than taking a wet cell centre does. A
        // straight channel was already perfect and must stay near it; clearance gives up a little
        // there (89% against 100% at sixteen yards wide) because it admits row-offset cells beside
        // the centred ones, and that is the price for everything above.
        for (width, amplitude) in [(4.0_f32, 8.0_f32), (6.0, 8.0), (10.0, 8.0), (16.0, 8.0)] {
            let (centre, _, ideal) =
                stream_seat_clearance(width, amplitude, false, PROBE_SEAT_SAMPLES);
            let (clear, _, _) = stream_seat_clearance(width, amplitude, true, PROBE_SEAT_SAMPLES);
            assert!(
                clear > centre * 1.2,
                "a wandering {width}-yard channel should seat well clear of its banks: \
                 clearance {clear:.2} against centre {centre:.2} of {ideal:.2}",
            );
        }
        for width in [4.0_f32, 6.0, 10.0, 16.0] {
            let (clear, _, ideal) = stream_seat_clearance(width, 0.0, true, PROBE_SEAT_SAMPLES);
            assert!(
                clear > ideal * 0.85,
                "a straight {width}-yard channel must stay near its centreline: \
                 {clear:.2} of {ideal:.2}",
            );
        }
    }

    fn walk_a_stream(
        gap_s: f32,
        burst_frames: u32,
        width: f32,
        amplitude: f32,
        seated: bool,
        sep_frac: f32,
    ) -> (f32, usize) {
        let wavelength = 60.0_f32;
        let spacing = 12.0_f32;
        let slots_n = 16usize;
        let speed = 7.0_f32; // yards a second, about a run
        let dt = 1.0 / 60.0;

        let mut slots = vec![ProbeSlot::default(); slots_n];
        let mut capturing: Option<usize> = None;
        let mut dwell = 0u32;
        let mut place_in = 0.0_f32;
        let mut wanted: Vec<Wanted> = Vec::new();
        let mut gap = 0.0_f32;
        let (mut worst, mut blind) = (f32::MAX, 0usize);

        for frame in 0..(60 * 40) {
            let x = frame as f32 * dt * speed;
            let eye = Vec2::new(x, 0.0);

            // Placement: the nearest cells of a stream running along +x.
            place_in -= dt;
            if place_in <= 0.0 {
                place_in = 1.0;
                // **The real placement rule: a cell counts only if WATER IS AT ITS CENTRE**, and
                // the lattice is fixed to the world. A straight channel lines up with one row of
                // cells and every step has a probe; a channel that WANDERS drifts off the row, and
                // wherever the centreline passes between cell centres the lattice finds nothing.
                // That is the "part of a river curve where no probe is placed" case.
                // The channel, as a wetness oracle — the same shape of question the real
                // placement asks of the liquid chunks.
                let wet = |p: Vec2| {
                    let bank = amplitude * (p.x / wavelength).sin();
                    ((p.y - bank).abs() <= width * 0.5).then_some(0.0f32)
                };
                let mut cand: Vec<(f32, IVec2, Vec2)> = Vec::new();
                for q in -6..=6 {
                    for r in -3..=3 {
                        let cell = IVec2::new((x / spacing).round() as i32 + q, r);
                        // The REAL seating function, so the bench measures placement rather than a
                        // paraphrase of it that can drift.
                        let seat = if seated {
                            probe_seat(cell_centre(cell), spacing, PROBE_SEAT_SAMPLES, &wet)
                                .map(|(p, _, _)| p)
                        } else {
                            let c = cell_centre(cell);
                            wet(c).map(|_| c)
                        };
                        let Some(seat) = seat else { continue };
                        cand.push((seat.distance(eye), cell, seat));
                    }
                }
                cand.sort_by(|a, b| a.0.total_cmp(&b.0));
                // The same spread rule the real placement applies — see PROBE_MIN_SEPARATION.
                let sep = spacing * sep_frac;
                let mut spread: Vec<(f32, IVec2, Vec2)> = Vec::new();
                for c in cand {
                    if spread.len() >= slots_n - PROBE_SPARE {
                        break;
                    }
                    if spread.iter().any(|(_, _, p)| p.distance(c.2) < sep) {
                        continue;
                    }
                    spread.push(c);
                }
                let cand = spread;
                wanted = cand
                    .iter()
                    .map(|&(_, cell, c)| {
                        let at = Vec3::new(c.x, 0.0, c.y);
                        Wanted {
                            cell,
                            at,
                            lo: at,
                            hi: at,
                        }
                    })
                    .collect();
            }

            // The fade ramp and the double buffer's retirement, as `ramp_probe_fades` runs them.
            let step = PROBE_FADE_RATE * dt;
            for s in slots.iter_mut() {
                let r = s.reach_target();
                s.reach = (s.reach + (r - s.reach).clamp(-step, step)).clamp(0.0, 1.0);
            }
            retire_replaced(&mut slots, &wanted);

            // The capture loop, one burst in flight.
            gap = (gap - dt).max(0.0);
            match capturing {
                Some(i) => {
                    dwell += 1;
                    if dwell >= burst_frames {
                        dwell = 0;
                        slots[i].cell = slots[i].want;
                        slots[i].at = slots[i].want_at;
                        slots[i].captured = true;
                        slots[i].retire = false;
                        gap = gap_s;
                        capturing = None;
                    }
                }
                None => {
                    capturing = next_capture(&mut slots, &wanted, eye, gap <= 0.0);
                }
            }

            // **How far AHEAD is the nearest usable probe?** Measuring the nearest in any
            // direction flatters the scheduler: the probes already behind keep that number small
            // while the water in front has nothing. The reported symptom is about what is ahead —
            // "the next probe only activates when I am almost there" — so that is what to measure.
            // A lead of zero means the player has walked past the last probe that exists.
            // The HORIZON: how far ahead usable coverage extends. The nearest probe ahead is the
            // wrong question — it necessarily falls to zero every time the player passes one — and
            // the nearest in any direction is flattered by the probes already behind. What the
            // symptom is about is whether the player is outrunning the captured set.
            let lead = slots
                .iter()
                .filter(|s| s.captured && !s.retire && s.at.x > eye.x)
                .map(|s| s.at.x - eye.x)
                .fold(f32::MIN, f32::max);
            if lead == f32::MIN {
                blind += 1;
            } else if frame > 60 * 6 {
                // Past the initial fill, which is arrival and a different question. We want the
                // SMALLEST lead seen — the moment the player came closest to outrunning coverage.
                worst = worst.min(lead);
            }
        }
        (worst, blind)
    }

    /// **Capturing the nearest owed slot beats capturing the lowest-indexed one**, on a bench, with
    /// a number. Slot order is assignment order, so the old policy made the probe directly ahead
    /// wait behind however many happened to sort before it.
    #[test]
    fn walking_a_stream_keeps_a_probe_close_by() {
        println!(
            "{:>10}{:>12}{:>12}{:>10}{:>8}",
            "width", "wander", "separation", "horizon", "blind"
        );
        for &(w, a, label) in &[
            (40.0_f32, 0.0_f32, "open"),
            (10.0, 0.0, "straight"),
            (10.0, 12.0, "curving"),
            (6.0, 12.0, "narrow+curve"),
        ] {
            for &sep in &[0.5_f32, 0.8, 1.1, 1.4] {
                let (h, b) = walk_a_stream(PROBE_BURST_GAP_S, 7, w, a, true, sep);
                println!("{label:>10}{a:>12.0}{sep:>12.1}{h:>10.1}{b:>8}");
            }
        }
        // **Open water must not leave the player uncovered**: a probe at least a lattice step ahead
        // at a run. The wanted set is the nearest `slots - PROBE_SPARE` cells, so the lead here is
        // bounded by that blob's radius (15.5 yd at twelve cells, 18.9 at sixteen).
        let (open_h, open_blind) =
            walk_a_stream(PROBE_BURST_GAP_S, 7, 40.0, 0.0, true, PROBE_MIN_SEPARATION);
        assert!(
            open_blind < 100 && open_h > probe_spacing(),
            "open water left {open_blind} frames with no probe ahead and a horizon of \
             {open_h:.1} yd — capture order has stopped closing the largest gap first"
        );

        // Streams, straight and curving: a probe at least a lattice step ahead at a run.
        let (straight_h, _) =
            walk_a_stream(PROBE_BURST_GAP_S, 7, 10.0, 0.0, true, PROBE_MIN_SEPARATION);
        let (curve_h, _) =
            walk_a_stream(PROBE_BURST_GAP_S, 7, 6.0, 12.0, true, PROBE_MIN_SEPARATION);
        assert!(
            straight_h > probe_spacing() && curve_h > probe_spacing(),
            "a stream outran its probes: straight {straight_h:.1} yd, curving {curve_h:.1} yd ahead"
        );
    }

    /// **A captured probe stays available however far away it is.**
    ///
    /// Availability used to fall off with the PLAYER's distance and cut out hard at a radius. On a
    /// curving channel, where the lattice leaves probes far apart, that put stretches between two
    /// probes where both sat past the cutoff — neither eligible, nothing to blend, and no reflection
    /// at all for part of the walk. Distance belongs to the per-fragment weighting, never to
    /// whether a probe exists.
    #[test]
    fn a_captured_probe_is_available_at_any_distance() {
        let mut slot = ProbeSlot {
            cell: Some(IVec2::new(0, 0)),
            at: Vec3::new(0.0, 0.0, 0.0),
            captured: true,
            ..Default::default()
        };
        slot.want = slot.cell;
        assert_eq!(slot.reach_target(), 1.0, "a captured probe is available");

        // Far enough away that every old radius would have excluded it.
        slot.at = Vec3::new(5_000.0, 0.0, 5_000.0);
        assert_eq!(
            slot.reach_target(),
            1.0,
            "a captured probe must stay available at ANY distance — a radius drawn around the \
             player is what left stretches of a stream with no reflection at all"
        );

        slot.captured = false;
        assert_eq!(
            slot.reach_target(),
            0.0,
            "a probe with no cube is not available"
        );
    }

    /// **A probe leaves only once its replacement is showing.** A new cell is captured into a
    /// free slot while the old probe stays at full strength; the old one retires when the new set
    /// is complete. With no slot free, the furthest old probe retires first, and is re-used only
    /// once it has faded out.
    #[test]
    fn a_replaced_probe_stays_until_the_new_one_shows() {
        let old = IVec2::new(0, 0);
        let new = IVec2::new(1, 0);
        let showing = ProbeSlot {
            cell: Some(old),
            want: Some(old),
            captured: true,
            reach: 1.0,
            ..Default::default()
        };
        let wanted = [Wanted {
            cell: new,
            at: Vec3::new(12.0, 0.0, 0.0),
            lo: Vec3::ZERO,
            hi: Vec3::ZERO,
        }];

        let mut slots = vec![showing, ProbeSlot::default()];
        assert_eq!(
            next_capture(&mut slots, &wanted, Vec2::ZERO, true),
            Some(1),
            "the new cell goes into the free slot, not over the showing probe"
        );
        retire_replaced(&mut slots, &wanted);
        assert!(
            !slots[0].retire,
            "the old probe stays while the new cell has no cube"
        );

        slots[1].cell = Some(new);
        slots[1].captured = true;
        slots[1].reach = 0.5;
        retire_replaced(&mut slots, &wanted);
        assert!(
            !slots[0].retire,
            "the old probe stays while the new one fades in"
        );

        slots[1].reach = 1.0;
        retire_replaced(&mut slots, &wanted);
        assert!(
            slots[0].retire,
            "the old probe retires once the new one is at full strength"
        );

        let mut full = vec![showing];
        assert_eq!(next_capture(&mut full, &wanted, Vec2::ZERO, true), None);
        assert!(
            full[0].retire,
            "with no slot free, an old probe starts retiring"
        );
        assert_eq!(
            next_capture(&mut full, &wanted, Vec2::ZERO, true),
            None,
            "and is not re-used while it is still visible"
        );
        full[0].reach = 0.0;
        assert_eq!(next_capture(&mut full, &wanted, Vec2::ZERO, true), Some(0));
    }

    /// **The point of a honeycomb is the worst case, so measure the worst case.** Sampled densely
    /// over one cell's neighbourhood, no point may sit further from its own hexagon's centre than
    /// `spacing / sqrt(3)`. A square lattice's bound is `spacing * sqrt(2) / 2`, about 22 % worse,
    /// and componentwise rounding instead of cube rounding lands near that — which is how this test
    /// catches the bug it exists for.
    #[test]
    fn no_point_is_further_from_its_hex_than_the_packing_bound() {
        let s = probe_spacing();
        let bound = s / 3.0_f32.sqrt();
        let mut worst = 0.0_f32;
        for i in 0..200 {
            for j in 0..200 {
                let p = Vec2::new(
                    (i as f32 / 200.0 - 0.5) * 4.0 * s,
                    (j as f32 / 200.0 - 0.5) * 4.0 * s,
                );
                let c = cell_centre(cell_of(Vec3::new(p.x, 0.0, p.y)));
                worst = worst.max(p.distance(c));
            }
        }
        assert!(
            worst <= bound * 1.01,
            "worst distance to a hex centre was {worst:.3} yd, past the packing bound {bound:.3}"
        );
    }

    /// Placement variants on the stream bench, dumped per frame for a top-down video.
    /// `BENCH_OUT=<dir> cargo test -p benilla-world --release --lib placement_variants -- --ignored`.
    #[test]
    #[ignore]
    fn placement_variants() {
        use std::io::Write;
        let out = std::env::var("BENCH_OUT").unwrap_or_else(|_| "/data/scratch/water/bench".into());
        std::fs::create_dir_all(&out).unwrap();
        let spacing = 12.0_f32;
        let wavelength = 60.0_f32;
        let knob = |k: &str, d: f32| {
            std::env::var(k)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(d)
        };
        let radius = knob("BENCH_RADIUS", 22.0);
        let slots_n = knob("BENCH_SLOTS", PROBE_SLOT_MAX as f32) as usize;
        let cap = slots_n - PROBE_SPARE;
        let gap_s = knob("BENCH_GAP", PROBE_BURST_GAP_S);
        let speed = 7.0_f32;
        let dt = 1.0 / 60.0;
        let mut summary = std::fs::File::create(format!("{out}/summary.txt")).unwrap();
        for &(scene, width, amplitude) in &[("stream", 6.0_f32, 12.0_f32), ("open", 40.0, 0.0)] {
            for variant in ["A-nearest12-water", "B-radius-land", "C-radius-water"] {
                let wet = |p: Vec2| {
                    let bank = amplitude * (p.x / wavelength).sin();
                    ((p.y - bank).abs() <= width * 0.5).then_some(0.0f32)
                };
                let mut slots = vec![ProbeSlot::default(); slots_n];
                let mut wanted: Vec<Wanted> = Vec::new();
                let (mut capturing, mut dwell, mut place_in, mut gap) =
                    (None::<usize>, 0u32, 0.0_f32, 0.0_f32);
                let mut first_wanted: std::collections::HashMap<IVec2, f32> = Default::default();
                let (mut caps, mut ahead_sum, mut late, mut wait_sum) =
                    (0usize, 0.0_f32, 0usize, 0.0_f32);
                let (mut cover_sum, mut cover_n) = (0.0_f32, 0usize);
                let mut f = std::io::BufWriter::new(
                    std::fs::File::create(format!("{out}/{scene}-{variant}.csv")).unwrap(),
                );
                writeln!(
                    f,
                    "#scene={scene} width={width} amplitude={amplitude} wavelength={wavelength}"
                )
                .unwrap();
                for frame in 0..(60 * 40) {
                    let t = frame as f32 * dt;
                    let eye = Vec2::new(t * speed, amplitude * (t * speed / wavelength).sin());
                    place_in -= dt;
                    if place_in <= 0.0 {
                        place_in = 1.0;
                        let base = (eye.x / spacing).round() as i32;
                        let mut cand: Vec<(f32, IVec2, Vec2)> = Vec::new();
                        for q in -10..=10 {
                            for r in -7..=7 {
                                let cell = IVec2::new(base + q - r / 2, r);
                                let c = cell_centre(cell);
                                let seat = probe_seat(c, spacing, PROBE_SEAT_SAMPLES, &wet)
                                    .map(|(p, _, _)| p);
                                let at = match (variant, seat) {
                                    ("B-radius-land", s) => s.unwrap_or(c),
                                    (_, Some(s)) => s,
                                    (_, None) => continue,
                                };
                                if variant != "A-nearest12-water" && c.distance(eye) > radius {
                                    continue;
                                }
                                cand.push((at.distance(eye), cell, at));
                            }
                        }
                        cand.sort_by(|a, b| a.0.total_cmp(&b.0));
                        let sep = spacing * PROBE_MIN_SEPARATION;
                        let mut spread: Vec<(f32, IVec2, Vec2)> = Vec::new();
                        for c in cand {
                            if spread.len() >= cap {
                                break;
                            }
                            if variant != "B-radius-land"
                                && spread.iter().any(|(_, _, p)| p.distance(c.2) < sep)
                            {
                                continue;
                            }
                            spread.push(c);
                        }
                        wanted = spread
                            .iter()
                            .map(|&(_, cell, c)| {
                                let at = Vec3::new(c.x, 0.0, c.y);
                                Wanted {
                                    cell,
                                    at,
                                    lo: at,
                                    hi: at,
                                }
                            })
                            .collect();
                        for w in &wanted {
                            first_wanted.entry(w.cell).or_insert(t);
                        }
                    }
                    let step = PROBE_FADE_RATE * dt;
                    for s in slots.iter_mut() {
                        let r = s.reach_target();
                        s.reach = (s.reach + (r - s.reach).clamp(-step, step)).clamp(0.0, 1.0);
                    }
                    retire_replaced(&mut slots, &wanted);
                    gap = (gap - dt).max(0.0);
                    match capturing {
                        Some(i) => {
                            dwell += 1;
                            if dwell >= PROBE_BURST {
                                dwell = 0;
                                let sl = &mut slots[i];
                                sl.cell = sl.want;
                                sl.at = sl.want_at;
                                sl.captured = true;
                                sl.retire = false;
                                gap = gap_s;
                                capturing = None;
                                if t > 6.0 {
                                    let rel = Vec2::new(sl.at.x, sl.at.z) - eye;
                                    let ahead = rel.x;
                                    caps += 1;
                                    ahead_sum += ahead;
                                    if ahead > 0.0 && ahead < spacing {
                                        late += 1;
                                    }
                                    if let Some(t0) = sl.cell.and_then(|c| first_wanted.get(&c)) {
                                        wait_sum += t - t0;
                                    }
                                }
                            }
                        }
                        None => capturing = next_capture(&mut slots, &wanted, eye, gap <= 0.0),
                    }
                    // Water ahead, 0..48 yd along the stream: distance to the nearest showing probe,
                    // weighted by availability (a probe at half strength counts half as near).
                    if t > 6.0 {
                        for k in 0..13 {
                            let x = eye.x + k as f32 * 4.0;
                            let p = Vec2::new(x, amplitude * (x / wavelength).sin());
                            let d = slots
                                .iter()
                                .filter(|s| s.captured && s.reach > 0.01)
                                .map(|s| Vec2::new(s.at.x, s.at.z).distance(p) / s.reach)
                                .fold(200.0_f32, f32::min);
                            cover_sum += d;
                            cover_n += 1;
                        }
                    }
                    if frame % 2 == 0 {
                        write!(
                            f,
                            "{t:.3},{:.2},{:.2},{}",
                            eye.x,
                            eye.y,
                            capturing.map_or(-1, |c| c as i32)
                        )
                        .unwrap();
                        for s in &slots {
                            let state = if !s.captured {
                                0
                            } else if s.retire {
                                2
                            } else {
                                1
                            };
                            write!(f, ",{:.2},{:.2},{:.3},{state}", s.at.x, s.at.z, s.reach)
                                .unwrap();
                        }
                        write!(f, ",|").unwrap();
                        for w in &wanted {
                            write!(f, ",{:.2},{:.2}", w.at.x, w.at.z).unwrap();
                        }
                        writeln!(f).unwrap();
                    }
                }
                let line = format!(
                    "{scene:7} {variant:18} captures {caps:3}  mean ahead at capture {:6.1} yd  captured <12 yd ahead {late:3}  mean wait {:4.1} s  water ahead: mean distance to a probe {:5.1} yd",
                    ahead_sum / caps.max(1) as f32,
                    wait_sum / caps.max(1) as f32,
                    cover_sum / cover_n.max(1) as f32
                );
                println!("{line}");
                writeln!(summary, "{line}").unwrap();
            }
        }
    }
}
