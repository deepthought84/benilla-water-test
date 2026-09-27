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
/// flipping the checkbox pays one uniform upload and no reload.
#[derive(Resource, Clone, Copy, PartialEq, Eq, Debug)]
pub enum WaterStyle {
    /// The 1.12 client's water, which is what the rest of this subsystem implements.
    Reference,
    /// benilla's Improved Water: the stylised surface, planar mirrors over flat water and fixed
    /// cube probes over falls and small pools (`benilla_formats::PlanarMap`), with the
    /// screen-space march a debug option.
    Stylised,
}

/// `$WOW_WATER_STYLE=1` boots into Improved Water — the lever for the world viewer, which has no
/// CVar host, and for captures. Session-only: `cvars` marks the row env-overridden so a run cannot
/// pin itself into `config.toml`.
impl Default for WaterStyle {
    fn default() -> Self {
        match std::env::var("WOW_WATER_STYLE").as_deref() {
            Ok("0") | Err(_) => Self::Reference,
            Ok(_) => Self::Stylised,
        }
    }
}

impl WaterStyle {
    /// The CVar's parse: 0 is the reference water, anything else Improved Water.
    pub fn from_cvar(v: f32) -> Self {
        if v != 0.0 {
            Self::Stylised
        } else {
            Self::Reference
        }
    }

    /// Is this benilla's own look, as opposed to the 1.12 client's?
    pub fn is_stylised(self) -> bool {
        self == Self::Stylised
    }

    /// Does this look drive the planar reflection cameras?
    pub(crate) fn wants_mirror(self) -> bool {
        self.is_stylised()
    }

    /// Does this look drive the cube probes?
    pub(crate) fn wants_probe(self) -> bool {
        self.is_stylised()
    }

    /// What `GetCVar("waterStyle")` answers for this state.
    pub fn cvar(self) -> &'static str {
        match self {
            Self::Reference => "0",
            Self::Stylised => "1",
        }
    }

    /// The shader lane (`LiquidParams.path.y`).
    pub(crate) fn shader_flag(self) -> f32 {
        f32::from(self.is_stylised())
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
        scene_color::register(app);
        depth::register(app);
        // ...and the cubemap probe, the fourth tier: one capture of one lake's surroundings, which
        // is the only tier that can answer for what is off the screen AND at the right orientation.
        probe::register(app);
    }
}
