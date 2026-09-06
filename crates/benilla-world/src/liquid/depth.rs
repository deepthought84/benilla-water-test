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
//! ## It is OFF, and the reason is not the water
//!
//! **A global depth prepass is not usable in this engine as it stands, and the way it fails is
//! quiet.** With it armed, terrain vanishes wherever a river or lake runs over it and the sky shows
//! through the hole — reproducible in Elwynn from a single vista, and provably the prepass: with
//! the stylised look on and the prepass alone disabled, the ground is whole again.
//!
//! It is the same root as the loud failure the model materials give. Bevy renders the prepass with
//! its own vertex path, and any material whose real vertex stage differs from it writes a different
//! depth there than it computes in the main pass; the main pass compares `GreaterEqual` against
//! that, and where the prepass came out nearer, the fragment is simply dropped. The model materials
//! at least fail validation outright and abort, which is how they were caught. Terrain returns a
//! wrong picture instead, and a wrong picture that only appears over water is one that survives an
//! empty-coastline vista, several live sessions and a commit — which is exactly what it did.
//!
//! So this is behind `$WOW_WATER_DEPTH` and off. Everything downstream of it falls back to the
//! authored depth byte (`thickness < 0.0` in `liquid.wgsl`), which is the look the water had before
//! any of this. Turning it on for real means giving each custom vertex path a prepass twin —
//! terrain's included, not just the models' — via `MaterialExtension::prepass_vertex_shader`, and
//! then proving on screen that a prepass-armed frame is pixel-identical to an unarmed one wherever
//! no water is involved.
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

use bevy::core_pipeline::prepass::DepthPrepass;
use bevy::prelude::*;

use super::WaterStyle;
use crate::view::WorldCamera;

/// `$WOW_WATER_DEPTH=1` — arm the prepass. **Off by default, and it must stay that way until the
/// custom vertex paths have prepass twins** — see the module doc.
fn depth_enabled() -> bool {
    std::env::var_os("WOW_WATER_DEPTH").is_some()
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
