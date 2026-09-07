//! The **scene depth** the stylised water reads — a depth prepass on the world camera, and nothing
//! else.
//!
//! ## Why the water needs to know what is behind it
//!
//! Until this existed, the water had no idea. How deep it looked was the authored MCLQ depth byte
//! lerped between two swatch colours, and that byte says how deep the *designer* called the water,
//! not how much water actually stands between this pixel and whatever is under it. Two things fall
//! out of the real distance and neither can be faked from a per-vertex byte:
//!
//! * **A soft edge where geometry passes through the surface.** Every rock, pier and hull currently
//!   meets the water on a hard, aliased line, because the water's alpha knows nothing about how far
//!   away the thing behind it is. Fading the last hand's breadth is the single cheapest thing that
//!   separates "a plane with a texture" from water, and it needs one number: the distance from this
//!   fragment to the opaque surface behind it.
//! * **Absorption that follows the water column.** Real water attenuates light exponentially and
//!   unevenly — red dies within a couple of yards, blue carries. A two-colour lerp cannot express
//!   that at all, which is why the deep end of several zones had to be capped by hand to stop it
//!   going to tar. With a thickness in yards the extinction is Beer–Lambert and the palette stays
//!   the zone's own.
//!
//! ## Why it is a prepass and not the main pass's depth
//!
//! The water draws in the transparent phase, so by the time it runs, the main depth buffer is bound
//! as a read/write attachment and cannot be sampled. A **prepass** writes the opaque scene's depth
//! into a separate texture before anything transparent draws, which is exactly the "what is behind
//! the water" question. Transparent geometry is not in it, and that is right: water should not fade
//! against another sheet of water.
//!
//! ## What it costs, and why it is a toggle
//!
//! A second geometry pass over the opaque scene. That is a real cost and this project does not hand
//! them out — so, like the planar reflection beside it, the prepass exists only while `waterStyle`
//! selects the stylised look, and the reference lane pays nothing. It is attached and detached in
//! place rather than at the camera's spawn because the CVar moves at runtime and there are three
//! spawn sites for the world camera, none of which should have to know this.
//!
//! ## It was off, and the reason was the HORIZON
//!
//! For a long time this was behind `$WOW_WATER_DEPTH` and disabled, because arming it punched holes
//! in the world: terrain vanished wherever a river or lake ran over it and the sky showed through.
//! It is on now, and the diagnosis that kept it off was wrong, which is worth recording because it
//! sent two plausible fixes into the bin before the real one.
//!
//! The suspicion was depth *disagreement* — Bevy renders the prepass with its own vertex path, so a
//! material whose real vertex stage differs from it could write a depth there that its main pass
//! then fails `GreaterEqual` against, dropping the fragment. That reads well and it is not what was
//! happening. Both remedies it implies were built and measured: a `prepass_fragment_shader` twin
//! carrying terrain's far-clip discard, and a `prepass_vertex_shader` twin computing terrain's
//! position under `@invariant`. Each changed the frame by **MAE 0.000**. The first was never even
//! called — Bevy only attaches a prepass fragment when `MeshPipelineKey::MAY_DISCARD` is set, which
//! comes from `AlphaMode::Mask`, and terrain is opaque.
//!
//! The cause was the **WDL horizon**, and it was a *discard* mismatch rather than a vertex one.
//! `wdl.wgsl`'s fragment cuts everything NEARER than `farclip − 33`, so the coarse hull only draws
//! past the wall and the fine terrain owns the near field. Bevy's stock prepass has no such cut, so
//! an armed prepass wrote the whole hull's depth — including the near part that never draws. The
//! coarse surface lies above the fine one wherever the ground is flat, which is exactly the river
//! valleys and lake beds, so the detailed terrain there failed the depth test against a horizon
//! that was not drawn either, and nothing was left but sky. The hole followed the water because the
//! water is where the flat ground is.
//!
//! The fix is `WdlExt::enable_prepass() -> false`: the same "a missing answer beats a wrong one"
//! trade the model lane already takes, and it costs the water nothing, because the coarse hull only
//! exists beyond the far-clip wall where there is no water to measure. `terrain.wgsl`'s position
//! carries `@invariant` beside it — it measured as free and it is the standard guard for a material
//! that draws in both passes, not a fix for anything observed here.
//!
//! Pinned by capture: a frame with no liquid in it is **pixel-identical** armed and unarmed
//! (MAE 0.000), which is the bar this section used to set for turning the gate off. `water-noon`
//! moves MAE 1.786 over 29 % of its pixels, and that is the feature — absorption and the soft edge
//! reading a true column instead of the authored byte.
//!
//! `$WOW_WATER_DEPTH=0` still forces it off, for bisecting a frame against the byte-only look.
//!
//! ## What is in it, and what is not
//!
//! Terrain is, which is what matters most: the seabed and the banks are what the water's column is
//! measured against nearly everywhere. Alpha-blended materials are not, and Bevy excludes them by
//! alpha mode — which is right, since water should not fade against another sheet of water, and it
//! is also why the liquid material itself is never in the prepass.
//!
//! **Models are not, deliberately** — see `WowModelExt::enable_prepass`. Their material rebuilds
//! the vertex layout around attributes Bevy does not know, and Bevy hands a material's `specialize`
//! to the prepass pipeline as well as the main one, so the prepass pipeline fails validation and
//! the client aborts the moment a skinned character is in view. The cost of opting them out is a
//! missing answer rather than a wrong one: the water measures its column past a character to the
//! terrain behind, so depth and colour stay right and only the soft edge around a body in the water
//! is absent. A prepass vertex shader of our own is the fix, and it is a job in the model renderer.
//! The same is true of the static-gx pass, which draws on its own pipeline and is in no prepass at
//! all.
//!
//! **The WDL horizon is not, deliberately** — see `WdlExt::enable_prepass` and the section above.
//! Its fragment discards the near field and Bevy's prepass cannot reproduce that, so it wrote depth
//! for a hull it never drew. Nothing is lost: the coarse horizon lives beyond the far-clip wall,
//! and there is no water out there to measure a column through.

use bevy::core_pipeline::prepass::DepthPrepass;
use bevy::prelude::*;

use super::WaterStyle;
use crate::view::WorldCamera;

/// Is the prepass armed? **On**, with `$WOW_WATER_DEPTH=0` as the way back to the authored-byte
/// look — see the module doc for the horizon bug that kept it off, and for what it costs.
///
/// Anything other than `0` reads as on, so the `=1` that every recipe in this repo carries keeps
/// working.
fn depth_enabled() -> bool {
    std::env::var("WOW_WATER_DEPTH").as_deref() != Ok("0")
}

/// Attach the depth prepass while the stylised look is on, and take it away again when it is not.
///
/// Change-gated on both sides: a steady state does nothing at all, and the insert/remove only fire
/// on the frame the answer actually differs from what the camera already carries.
fn maintain_depth_prepass(
    mut commands: Commands,
    style: Res<WaterStyle>,
    cameras: Query<(Entity, Has<DepthPrepass>), With<WorldCamera>>,
) {
    let want = *style == WaterStyle::Stylised && depth_enabled();
    for (entity, has) in &cameras {
        if want == has {
            continue;
        }
        let mut cam = commands.entity(entity);
        if want {
            cam.insert(DepthPrepass);
        } else {
            cam.remove::<DepthPrepass>();
        }
    }
}

pub(super) fn register(app: &mut App) {
    // Every frame, but it is a query over one camera and a comparison — the work is the insert,
    // which happens on a style flip and on the first frame a camera exists.
    app.add_systems(Update, maintain_depth_prepass);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The gate's polarity, which has now flipped once. It reads **on** unarmed — the prepass is
    /// the shipped behaviour — and only the explicit `0` turns it off; anything else, including the
    /// `=1` every recipe in this repo and its decisions carry, leaves it on.
    #[test]
    fn the_prepass_is_on_unless_it_is_explicitly_switched_off() {
        let restore = std::env::var("WOW_WATER_DEPTH").ok();
        // SAFETY: single-threaded test, and the variable is restored before it returns.
        unsafe {
            std::env::remove_var("WOW_WATER_DEPTH");
            assert!(depth_enabled(), "the default is armed");
            std::env::set_var("WOW_WATER_DEPTH", "1");
            assert!(depth_enabled(), "the historical =1 still means on");
            std::env::set_var("WOW_WATER_DEPTH", "0");
            assert!(
                !depth_enabled(),
                "and 0 is the way back to the authored byte"
            );
            match restore {
                Some(v) => std::env::set_var("WOW_WATER_DEPTH", v),
                None => std::env::remove_var("WOW_WATER_DEPTH"),
            }
        }
    }
}
