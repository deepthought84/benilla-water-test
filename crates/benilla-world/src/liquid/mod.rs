//! Liquid: the animated surfaces (`surface`), where the liquid is and whether a subject or the
//! camera eye is in it (`query`, indexed by `spatial`), and the underwater drift cloud (`drift`).
//! The MCLQ parse and flat mesh are `benilla_formats::liquid`; the shading is `liquid.wgsl`.
//!
//! The reference has three liquid renderers, each with its own arm in the shader: ADT MCLQ, WMO
//! MLIQ water (`0x6b62e0` category 0, exterior `0x6b6630` or interior `0x6b6420` by the group's
//! `MOGP.flags & 0x48`: one texture stage, no depth ramp, alpha from a per-vertex authored byte)
//! and magma/slime.
//!
//! The ADT combiner is the asset `Shaders\Pixel\ocean0_s.bls` in `patch.MPQ`:
//! `rgb = primary·colorTex.rgb + detailTex.rgb + (secondary + 0.25)·detailTex.a`,
//! `alpha = colorTex.a`. `primary` is the lit vertex colour `clamp(ambient + N·L·sun)`;
//! `detailTex` is the animated `lake_a`/`ocean_h` frame, near-black RGB and a ripple alpha.
//! `colorTex` is a 64-row swatch, RGB between the zone's raw `Light.dbc` water rows (IntBand 16/17
//! river/lake, 14/15 ocean) and alpha between its `LightParams` shallow and deep alphas, filled by
//! `0x68a830` as `row(i) = c0 + floor(i·(c1−c0)/64)`, so it never reaches the deep endpoint. The
//! ocean's last row alone is darkened by an HSV `V ×= 0.9` and forced opaque; about 80% of ocean
//! vertices sample it. The dirty flags `[0xc8117c]`/`[0xc81b70]` clear every world frame
//! (`0x680b90`/`0x680b97`, refill `0x58acd0`), so colour and opacity track the zone and the clock.
//!
//! One per-vertex depth `V` indexes colour and alpha alike: `min(byte/42, 1)` on river/lake
//! (`0xc81768`, read at `0x68d818` by `0x68d790`), `min(byte/255, 1)` on ocean (`0xc7fcd8`, read
//! at `0x68d718` by `0x68d690`), both tables built by `0x68c4c0`. `0xc7fbc0`'s `1.6·(i/63)^8` is no
//! water alpha: it fills the sky glare texture `[0xc7fbb8]` (`0x68c250` via `0x68c4a0`), whose one
//! liquid binding, `0x685257`, sits in a branch that never runs (`[0xc800ec]` is only ever 0).

use bevy::pbr::MaterialPlugin;
use bevy::prelude::*;

use benilla_assets::materials::LiquidMaterial;
use benilla_assets::AssetSet;

mod depth;
mod drift;
mod hiz;
mod probe;
mod query;
#[cfg(test)]
mod real_data;
mod reflect;
mod scene_color;
mod ssr_pip;
pub use probe::{ProbeMark, ProbeMarks};
mod ripple;
mod ripple_sim;
mod spatial;
mod surface;

// The submodules are private: this list is everything the rest of the client may name.
/// **Which water look the client draws** — the `waterStyle` CVar's knob, and the only setting in
/// this subsystem that is a matter of taste rather than of fidelity.
///
/// [`WaterStyle::Reference`] is the default and is what every other line under `liquid/` is about:
/// the 1.12 client's own combine, its swatch, its sheet, its opacity. [`WaterStyle::Stylised`] is
/// benilla's own — an animated surface normal, a specular glitter path, a Fresnel sky mix and shore
/// foam, ported from the `water-test` sandbox and described in `liquid.wgsl`'s `stylised_water`. It
/// keeps the zone's colours, depth and lighting and changes only what is done with them, so it
/// follows the world's day/night and its zones exactly as the faithful path does.
///
/// The switch is a **uniform write**, not a rebuild: [`surface::apply_water_style`] rewrites
/// `path.y` on the handful of shared liquid materials when this resource changes, so a player
/// flipping the dropdown pays one uniform upload and no reload.
#[derive(Resource, Clone, Copy, PartialEq, Eq, Debug)]
pub enum WaterStyle {
    /// The 1.12 client's water, which is what the rest of this subsystem implements.
    Reference,
    /// benilla's own stylised water, with the planar capture and the screen-space march layered
    /// over the sky mix.
    Stylised,
    /// The stylised water with **the screen-space march as its only reflection** — no second
    /// camera, no capture, no blur.
    ///
    /// This is the cheap lane, and it is cheap by subtraction rather than by cutting quality out
    /// of the march. [`Stylised`](Self::Stylised) draws the world twice: the mirror camera is a
    /// whole extra view with its own culling, its own pipelines and its own 500-yard reach, and it
    /// costs about a millisecond of every frame near water even before the march runs. Standing it
    /// down leaves the march holding the whole reflection, which it can do — it traces the
    /// fragment's own normal, so it is right at any surface orientation, and a stream in a gorge
    /// or a pond ringed by banks is exactly the case it handles best.
    ///
    /// What is lost is what a screen-space trace structurally cannot see: anything off the frame
    /// or behind the eye. On open water viewed at a grazing angle the ray runs off the top of the
    /// screen within a few steps, so the reflection there falls back to the sky mix underneath —
    /// which is a plausible image for open water and the reason this lane is offered at all rather
    /// than being a strictly worse [`Stylised`](Self::Stylised). Trees on the far bank of a lake
    /// will pop in as they enter frame; that is the honest cost, and the tooltip says so.
    ///
    /// **How much water it actually answers for, measured:** on `water-noon` the march reaches
    /// about 14% of water pixels and changes about 4% of the frame against a no-reflection
    /// baseline, where the planar mirror changes about 15%. Open water at a grazing angle is the
    /// case it cannot serve — those rays leave the top of the frame — so this lane is for water
    /// with something standing around it: a stream in a gorge, a pond ringed by banks, exactly the
    /// geometry a single horizontal plane gets wrong. Offering it as a general replacement for the
    /// mirror would be overselling it.
    ///
    /// **It is not currently the cheaper lane, and that is measured rather than assumed.** On
    /// `water-noon` at 1600x900 with MSAA off (RX 6500 XT, RADV), the whole-frame GPU spans are
    /// 1.17 ms for [`Reference`](Self::Reference), 1.73 ms for [`Stylised`](Self::Stylised) and
    /// 2.36 ms for this — because the march cost 0.98 ms where the mirror it replaces costs
    /// 0.36 ms. (Casting the ray flat later took the march to 0.70 ms; the gap narrowed, it did
    /// not close.) The march is full-resolution and per water pixel; the capture is half-resolution
    /// and once per frame, and no amount of tuning the step count closes a gap of that shape. What
    /// would is tracing into a downscaled buffer in a pass of its own (`ssr_trace`'s doc carries
    /// the argument).
    ///
    /// Two things it *is* better at today, both worth keeping in view. It is the only stylised
    /// lane that survives `WOW_MSAA=4` — the mirror view is `Msaa::Off` while `static_gx` keys its
    /// pipeline on the world camera's sample count, so the planar lane panics there on an
    /// incompatible sample count. And it costs no second view on the CPU: the mirror's culling and
    /// queueing are gone, which showed as roughly 1.5 ms of `cpu_ms` on the same shot.
    ///
    /// Take the absolute numbers as one batch's, not as constants: this rig moved every figure
    /// above by a factor of three between sessions, so what is durable here is the *ordering* —
    /// which held in every batch — and not the milliseconds. Compare arms inside one batch.
    StylisedSsr,
    /// The stylised water with **the march and a cubemap probe**, and still no second camera.
    ///
    /// [`StylisedSsr`](Self::StylisedSsr) gives up everything off the frame: a ray that leaves the
    /// screen falls back to a two-colour sky ramp, which is plausible over open water and plainly
    /// wrong with a treeline behind you. This lane fills that with a cubemap captured at the water
    /// itself (`liquid::probe`) — off-screen content at the fragment's own reflected orientation,
    /// which is the one thing none of the other three tiers can supply.
    ///
    /// On the full [`Stylised`](Self::Stylised) lane the same probe runs under the mirrors and
    /// shows only where they stand down (`liquid::planar`).
    StylisedProbe,
    /// **The hybrid: the screen-space march over the cubemap probe, with no planar camera at all.**
    ///
    /// Each tier covers the other's blind spot, and between them they leave none. The march traces
    /// the fragment's own normal so it is right at any surface orientation and exact for anything
    /// on screen — which is precisely the bank trees and near shoreline that a probe reflects
    /// worst. The probe sees off-frame and behind the eye, which is exactly what the march cannot,
    /// and it is available everywhere without a second scene render.
    ///
    /// **No mirror camera.** [`Stylised`](Self::Stylised) draws the world twice, and that second
    /// view — its own culling, its own pipelines, its own 500-yard reach — costs about a
    /// millisecond of every frame near water. The whole reason a probe exists is to escape the
    /// single plane the mirror is built on, so keeping the mirror alongside it re-imports the cost
    /// this lane is meant to shed, for a tier the other two already cover.
    ///
    /// **The probe here is deliberately soft.** It is not trying to place reflections correctly —
    /// see [`probe_lod_floor`](super::probe::probe_lod_floor) for the measurements that closed that
    /// question — it is an ambient term blurred by exactly the parallax error it would otherwise
    /// show, sitting under a march that has the last word wherever the screen has the answer.
    StylisedSsrProbe,
}

/// `$WOW_WATER_STYLE=1` boots into the stylised look, `=2` into its SSR-only lane — the A/B lever, in the mould of
/// `$WOW_RENDER_SCALE` and `$WOW_CLUTTER_DENSITY`, and the only way to reach this look from the
/// world viewer, which has no CVar host to read a config with. Session-only: `cvars` marks the row
/// env-overridden so a comparison run cannot pin itself into `config.toml`.
impl Default for WaterStyle {
    fn default() -> Self {
        match std::env::var("WOW_WATER_STYLE").as_deref() {
            Ok("1") => Self::Stylised,
            Ok("2") => Self::StylisedSsr,
            Ok("3") => Self::StylisedProbe,
            Ok("4") => Self::StylisedSsrProbe,
            _ => Self::Reference,
        }
    }
}

impl WaterStyle {
    /// The CVar's parse. Anything that is not the stylised lane reads as the reference — the same
    /// posture `FollowStyle::from_cvar` takes, and the right one for a look: an unknown value must
    /// land on what the client would draw with no setting at all, never on a dead surface.
    pub fn from_cvar(v: f32) -> Self {
        // `2` is exact in f32 and the CVar layer hands this the parsed number, so an equality test
        // is safe here — but it is written as a window anyway, because the rule this posture is
        // built on is that an unrecognised value lands on a look the client can draw, and a
        // half-open window is the shape that keeps holding when a fourth lane arrives.
        if v >= 4.0 {
            Self::StylisedSsrProbe
        } else if v >= 3.0 {
            Self::StylisedProbe
        } else if v >= 2.0 {
            Self::StylisedSsr
        } else if v != 0.0 {
            Self::Stylised
        } else {
            Self::Reference
        }
    }

    /// Is this one of benilla's own looks, as opposed to the 1.12 client's?
    ///
    /// **Almost every gate in this subsystem wants this rather than an equality test.** The
    /// stylised lane's machinery — the ripple simulation, the scene-colour snapshot, the depth
    /// prepass, the silenced splash decals — is shared by both stylised variants, and the only
    /// things that genuinely distinguish them are the mirror camera and the march's own strength,
    /// both of which live in `reflect.rs`. A `== Stylised` anywhere else is a bug waiting for the
    /// day someone selects the SSR lane and finds the water has no ripples.
    pub fn is_stylised(self) -> bool {
        matches!(
            self,
            Self::Stylised | Self::StylisedSsr | Self::StylisedProbe | Self::StylisedSsrProbe
        )
    }

    /// Does this look drive the planar reflection camera?
    ///
    /// Only the full stylised lane does. Kept as a name rather than an inline comparison because
    /// the mirror is spawned, aimed, sized and torn down in four different places, and they have
    /// to agree.
    pub(crate) fn wants_mirror(self) -> bool {
        matches!(self, Self::Stylised)
    }

    /// Does this look drive the cubemap probe?
    ///
    /// Every lane but the march alone. Beside the mirrors it serves the water they stand down
    /// over — falls, descending rivers, small pools (`liquid::planar`).
    pub(crate) fn wants_probe(self) -> bool {
        matches!(
            self,
            Self::Stylised | Self::StylisedProbe | Self::StylisedSsrProbe
        )
    }

    /// What `GetCVar("waterStyle")` answers for this state.
    pub fn cvar(self) -> &'static str {
        match self {
            Self::Reference => "0",
            Self::Stylised => "1",
            Self::StylisedSsr => "2",
            Self::StylisedProbe => "3",
            Self::StylisedSsrProbe => "4",
        }
    }

    /// The shader lane (`LiquidParams.path.y`).
    pub(crate) fn shader_flag(self) -> f32 {
        match self {
            Self::Reference => 0.0,
            // **Both stylised lanes are the same shader lane.** What separates them is not the
            // surface — the ripple, the glitter, the Fresnel sky mix and the foam are identical —
            // but which reflection tiers are armed, and those arrive on the reflection uniform
            // (`WaterReflectData`) rather than here. Giving the SSR lane its own `path.y` would
            // make the dropdown a pipeline rebuild for a difference the fragment shader does not
            // need to know about.
            Self::Stylised | Self::StylisedSsr | Self::StylisedProbe | Self::StylisedSsrProbe => {
                1.0
            }
        }
    }
}

pub(crate) use probe::{ProbeCull, ProbeFace};
pub use query::{
    camera_claim, describe_at, liquid_at, player_claim, surfaces_at, unit_claim, water_surface_at,
    EyeLiquid, FoamPatch, LiquidClaim, LiquidHit, LiquidSource, RoomPlacements, SubmergedEye,
    Underwater, WaterChunkInfo, WmoPool,
};
pub(crate) use reflect::ReflectionCamera;
pub use reflect::{mirror_view_shape, UNMIRRORED_RENDER_LAYER, WATER_RENDER_LAYER};
pub(crate) use spatial::{maintain_water_index, WaterIndex};
pub(crate) use surface::{
    spawn_liquids, spawn_wmo_liquids, LiquidAssets, LiquidSoundSource, LiquidSurface, WaterMapRef,
    WetLattice,
};

/// `WOW_FORCE_SUB=<frames>`: hold the camera-eye verdict submerged for the first `<frames>` frames,
/// then release it, so the wet-to-dry crossing lands on a known frame with no water or server.
fn forced_submersion_frames() -> u32 {
    std::env::var("WOW_FORCE_SUB")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
}

/// The [`forced_submersion_frames`] override, run after the real probe so it overwrites the
/// verdict rather than racing it; logs the release frame.
fn force_submersion(mut underwater: ResMut<Underwater>, mut frame: Local<u32>) {
    let hold = forced_submersion_frames();
    *frame += 1;
    if *frame <= hold {
        underwater.0 = benilla_formats::Submersion::Water;
    } else if *frame == hold + 1 {
        info!(
            "WOW_FORCE_SUB: frame {} — eye verdict released to Dry",
            *frame
        );
    }
}

/// The set that writes [`Underwater`]; every consumer of the verdict runs after it, since a
/// frame-old verdict shows one mixed frame (murk with the sky still up) on every surface crossing.
#[derive(bevy::ecs::schedule::SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct SubmersionVerdict;

/// The liquid subsystem: the shared materials at startup, the submersion verdict and the drift
/// cloud each frame. The terrain streamer spawns the surfaces with their tile ([`spawn_liquids`]).
pub(crate) struct LiquidPlugin;

impl Plugin for LiquidPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<LiquidMaterial>::default())
            .init_resource::<Underwater>()
            .init_resource::<SubmergedEye>()
            .init_resource::<WaterIndex>()
            .init_resource::<WaterStyle>()
            // PreUpdate: a surface streamed by last frame's commands is indexed before anyone asks.
            .add_systems(PreUpdate, maintain_water_index)
            .add_systems(Startup, surface::setup_liquid.after(AssetSet::Open))
            .add_systems(
                Update,
                (
                    // After the portal pass publishes this frame's `CameraInteriorClaim`: a
                    // frame-old room flashes the wrong atmosphere on every doorway crossing.
                    query::detect_submersion
                        .after(crate::wmo_portal::WmoPvsSet)
                        .in_set(SubmersionVerdict),
                    // The look toggle: change-gated, so the steady state schedules nothing and a
                    // flip touches the shared materials once. It runs on the frame the CVar lands
                    // and on the first frame after startup, which is what carries a persisted
                    // `waterStyle` from `config.toml` onto materials built before it was read.
                    surface::apply_water_style.run_if(resource_changed::<WaterStyle>),
                ),
            )
            // The `WOW_NO_LIQUID` override: after both per-frame `Visibility` owners of a surface
            // (the exterior cull, the model-visibility authority) and before Bevy reads it.
            .add_systems(
                PostUpdate,
                surface::hide_liquid_surfaces
                    .after(crate::exterior_cull::ExteriorCullSet)
                    .before(bevy::camera::visibility::VisibilitySystems::VisibilityPropagate)
                    .run_if(|| std::env::var_os("WOW_NO_LIQUID").is_some()),
            );
        // Added only when `WOW_FORCE_SUB` names a hold: an ordinary run carries no system for it.
        if forced_submersion_frames() > 0 {
            app.add_systems(
                Update,
                force_submersion
                    .after(query::detect_submersion)
                    .in_set(SubmersionVerdict),
            );
        }
        drift::register(app);
        // The stylised look's planar reflection — a whole subsystem of its own, and inert unless
        // that look is selected.
        reflect::register(app);
        // ...and its wave simulation, which is the same story: a second subsystem, inert on the
        // reference lane, where the client's own painted splash decals are the wake instead.
        ripple_sim::register(app);
        // ...and the scene depth it reads to know what is behind it, on the same terms: a second
        // geometry pass, attached only while the stylised look is selected.
        hiz::register(app);
        ssr_pip::register(app);
        scene_color::register(app);
        depth::register(app);
        // ...and the cubemap probe, the fourth tier: one capture of one lake's surroundings, which
        // is the only tier that can answer for what is off the screen AND at the right orientation.
        probe::register(app);
    }
}
