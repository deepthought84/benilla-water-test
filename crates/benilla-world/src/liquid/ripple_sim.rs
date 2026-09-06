//! The stylised water's **wave simulation** — a height field the world pushes on, so that a
//! swimmer's wake is water actually moving rather than a picture of a wake.
//!
//! **This is the alternative to `water_fx`, not an addition to it.** That module is the reference's
//! own `CWater0Ripple`: a pool of *records*, each a splash stencil (`splash.blp` / `wake.blp`)
//! stretched over a static patch of geometry by a texture matrix as the record ages. It is
//! byte-verified against the 1.12 client and it stays exactly as it is on the reference lane. What
//! it cannot do is what a painted thing never can: two wakes crossing do not interfere, a wake does
//! not reflect off the bank it runs into, and nothing that is not a unit can disturb the water at
//! all. So on the stylised lane the decals are silenced ([`super::WaterStyle::Stylised`]) and this
//! runs instead.
//!
//! ## What it solves and how
//!
//! The 2-D wave equation, `∂²h/∂t² = c²∇²h`, integrated explicitly on a square lattice:
//!
//! ```text
//! h' = 2h − h_prev + C² · (h[l] + h[r] + h[u] + h[d] − 4h)
//! ```
//!
//! with `C = c·Δt/Δx` the Courant number. That single line is the whole physics, and everything a
//! wake is falls out of it rather than being drawn: a source dragged through the lattice leaves a
//! V behind it because the disturbance it made a moment ago has spread a fixed distance while the
//! source has moved further than that; two swimmers' wakes add where they cross and cancel where
//! they meet out of phase; a wave running into shallow water piles up. None of those is a case
//! anywhere in this file.
//!
//! **The scheme is only conditionally stable** — `C ≤ 1/√2` in 2-D, and past it the field does not
//! degrade, it explodes to NaN within a second. Two things hold it: [`SIM_C2`] is `C²` fixed well
//! under the limit, and the step runs at a **fixed** [`SIM_DT`] with the frame's elapsed time
//! accumulated into whole steps, so a frame-rate spike cannot raise the Courant number at all. The
//! substep count is capped, which drops simulated time on a stall rather than trying to catch up
//! (a spiral this cheap is still a spiral).
//!
//! ## The window
//!
//! The lattice is not the world — it is a [`SIM_YARDS`]-square window centred on the viewer,
//! carried along as they move. It moves in **whole cells only**: a fractional scroll would need the
//! field resampled every frame, and resampling a wave field is a low-pass filter, so the ripples
//! would visibly dissolve whenever the player walked. A whole-cell shift is a memmove, and it is
//! exact.
//!
//! Waves reflect off the window's edge, which the world has no edge to justify — the result is a
//! square standing pattern that reads as a bug. [`EDGE_ABSORB`] cells of progressive damping around
//! the rim swallow them instead, and the shader fades the whole contribution out over the same
//! band, so a wave leaving the simulated area leaves quietly.

use std::sync::Arc;

use bevy::asset::RenderAssetUsages;
use bevy::ecs::entity::EntityHashMap;
use bevy::image::{ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor};
use bevy::prelude::*;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::{
    Extent3d, TexelCopyBufferLayout, TextureDimension, TextureFormat, TextureUsages,
};
use bevy::render::renderer::RenderQueue;
use bevy::render::texture::GpuImage;
use bevy::render::{Render, RenderApp, RenderSystems};

use super::reflect::WaterReflectData;
use super::{WaterChunkInfo, WaterIndex, WaterStyle};
use crate::world_unit::{ViewerUnit, WorldUnit};
use benilla_assets::coords::bevy_to_wow;

/// Lattice resolution, per side. Square, and a power of two only by taste — nothing here needs it.
const SIM_CELLS: usize = 192;

/// How much world the window covers, per side, in yards. With [`SIM_CELLS`] this puts a cell at a
/// quarter of a yard: fine enough that the shortest wave the lattice can carry (two cells, half a
/// yard) is a ripple rather than a swell, and coarse enough that the window still reaches
/// twenty-four yards out — past the distance at which a wake is worth resolving.
const SIM_YARDS: f32 = 48.0;

/// Cell pitch in yards.
const SIM_CELL: f32 = SIM_YARDS / SIM_CELLS as f32;

/// The integrator's fixed timestep. Fixed, not the frame's — see the module doc: the stability
/// condition is on `c·Δt/Δx`, so a Δt that moves with the frame rate is a Δt that can leave the
/// stable region on any hitch.
const SIM_DT: f32 = 1.0 / 90.0;

/// Most substeps one frame may run. Four at [`SIM_DT`] is 44 ms — past that the simulation simply
/// loses time, which looks like the water briefly slowing and is the right failure.
const SIM_MAX_STEPS: u32 = 4;

/// `C² = (c·Δt/Δx)²`, the wave speed in lattice units — about **0.7 yd/s**. The 2-D stability limit
/// is `C² ≤ 0.5`, so this is nowhere near it; the number is chosen for shape, not stability. It
/// also sets the wedge's WIDTH — half-angle `arcsin(c/v)` — and the reference's is narrow, perhaps
/// fifteen degrees, which is what pins `c` at roughly a quarter of a swimmer's pace.
///
/// **This one number decides what a wake looks like**, through its ratio to the swimmer's own
/// speed. A source slower than its waves radiates concentric circles and nothing else. A source
/// FASTER than them leaves everything behind, in a wedge of half-angle `arcsin(c/v)` — and that
/// wedge is the wake. At the 14 yd/s this started from, the waves left a 2.5 yd/s swimmer standing
/// and drew a symmetric bullseye; at 5.5 they still outran her and it was still rings, just
/// lopsided ones.
///
/// The reference settles it: 1.14.2 footage of the same character swimming the same stretch of
/// Darkshore shows a long, narrow, very low-contrast V opening backwards from the shoulders, and no
/// circles anywhere. Against a 2.5 yd/s swim that wants `c` around half her speed, which is also
/// where the physics of short surface ripples actually sits — a swimmer really does outrun them.
const SIM_C2: f32 = 0.0010;

/// Per-step amplitude retention. Ripples have to die out eventually or the window fills with the
/// memory of everywhere the player has been; at 90 steps a second this holds a wave for a few.
///
/// It was briefly a third of this, to stop energy piling into a standing mound around a stationary
/// swimmer — which was the wrong fix for that, since the mound came from pushing every frame and
/// from pulsing while idle, and both of those are gone. A wake needs the length back: the reference
/// footage shows a trail still legible fifteen yards behind the swimmer, which at swimming pace is
/// several seconds old.
const SIM_DAMP: f32 = 0.9955;

/// Width of the absorbing rim, in cells — see the module doc.
const EDGE_ABSORB: usize = 14;

/// How far a unit travels between the rings it sheds, in yards.
///
/// **Distance only — a body that is not going anywhere makes no waves.** This carried a second,
/// time-based cadence so that treading water still rang, on the theory that a swimmer is never
/// quite still. On screen that is not what it reads as: a stationary character sits inside a
/// permanent set of expanding rings that nothing explains, and the surface never returns to rest.
/// Water at rest is worth having, and it is the thing a wake is legible against.
///
/// Short enough that consecutive rings overlap into one continuous trail rather than a row of
/// beads — at swimming pace this is about three a second, and the waves spread further than that
/// between them.
const PULSE_STRIDE: f32 = 0.7;

/// One pulse's displacement. Height here is arbitrary; what makes it a physical quantity is only
/// the slope the shader reads out of it, and that is scaled once, at the end, by [`WAKE_SLOPE`].
///
/// Cut hard when the wave speed came down. The two are coupled and it is easy to miss: fast waves
/// carry a pulse's energy away over a wide area, slow ones leave it where it was put, so the same
/// displacement that read as a faint ring at 5.5 yd/s piles into a solid white stripe along the
/// swimmer's track at 1.3.
const PULSE_AMOUNT: f32 = 0.030;

/// Radius of a unit's push, in yards — **a body plus the bow wave it makes**, not the body.
///
/// The size of the source is what sets the wavelength of everything downstream of it, and the first
/// pass at three-quarters of a yard is why the first live capture came back with a wake that was
/// unmistakably a V and unmistakably too small to see: ripples a yard from crest to crest, dying
/// inside a body length. Wider is not louder here, it is *longer* — the same energy in waves that
/// carry, which is what a wake looks like.
const PUSH_RADIUS: f32 = 0.8;

/// How far AHEAD of a body its wake is born, in yards.
///
/// A wedge's apex sits where the source is, so pushing at the body's centre opens the V from the
/// body's centre — and since the newest rings have barely spread, the arms only become visible a
/// stride or two further back, leaving a gap between the swimmer and their own wake. A real bow
/// wave starts at the bow. Moving the source forward by about half a body closes that gap: the
/// arms now leave the shoulders, which is where the reference's do.
const PUSH_BOW_OFFSET: f32 = 0.55;

/// How far a body must move in one frame to count as moving at all, in yards. Small, but not zero:
/// a streamed position that jitters by a thousandth of a yard is a body standing still.
const MOVED_EPSILON: f32 = 1.0e-3;

/// How deep the water has to be under a unit's feet before it disturbs anything, in yards. Ankle
/// deep: shallower than this and the "wake" is a unit walking on wet sand.
///
/// The **upper** limit — a body that has gone under rather than along the surface — is not a
/// number here at all: it is the reference's own gate, borrowed whole from
/// [`crate::water_fx::params::depth_strength`]. A submerged body makes no wake because there is no
/// longer a surface being pushed on, and the client already draws exactly that line at about two
/// body heights, with a linear ramp over the last half of it. Reproducing it rather than inventing
/// a second line keeps the two wake systems agreeing about where water stops being disturbed,
/// which is the sort of thing that silently drifts apart otherwise.
const PUSH_MIN_DEPTH: f32 = 0.35;

/// The slope the encoded texture's full range stands for. The height field's units are arbitrary
/// (see [`PUSH_PER_YARD`]); this is the one number that turns it into something the shader can add
/// to a normal, and therefore the knob for how pronounced wakes are.
/// Set against the AMBIENT waves rather than in isolation, because that is what the wake has to be
/// seen against: the tiling ripple map's summed octaves reach about ±0.5, and the shader multiplies
/// them by `NORMAL_STRENGTH`, so the surface a swimmer is crossing already tilts by ~0.7 in the
/// same encoded units. A wake quieter than that is not subtle, it is invisible — which is exactly
/// what happened the frame after the ambient waves were strengthened to break the sun's highlight.
///
/// Originally set from the geometry, and by eye before that, and wrong both times in both directions.
/// A pulse is a raised-cosine bowl of depth `D` over [`PUSH_RADIUS`], whose steepest slope is about
/// `1.43·D` per yard; the encode's central difference spans half a yard, so the byte sees
/// `1.43·D · 0.5 · (1/SIM_CELL) · WAKE_SLOPE`. With pulses overlapping to `D ≈ 0.5`, anything much
/// above 1.2 drives that past the byte's range — and a CLIPPED slope is why raising the shader's
/// foam threshold did nothing to shrink the white patch around the swimmer: over several yards the
/// value was already pinned at its maximum, so there was no gradient left for a threshold to bite
/// on.
const WAKE_SLOPE: f32 = 1.7;

/// The height the encoded blue channel's full range stands for.
///
/// Height, not curvature — the first pass encoded the Laplacian here on the theory that a crest is
/// where a surface is most sharply bent, which is true and useless: measured live with a swimmer
/// driving it, the field's curvature peaks around 0.05 against a height of 0.25, so once it is
/// scaled to fill a byte the ordinary parts of a wake are down in the bottom few percent and the
/// whole thing is invisible on screen. Height has the range, and it is also the more useful
/// quantity at the other end: the shader reads it as **how much water is under this pixel**, which
/// shows a wake even when you are looking straight down at it and Fresnel has taken every other
/// term away. Foam is derived from the SLOPE in the shader instead, which has range to spare.
const WAKE_HEIGHT: f32 = 2.0;

/// The live height field and the image the shader reads it through.
#[derive(Resource)]
pub(crate) struct RippleSim {
    /// The texture the liquid materials bind: R/G the surface slope, B the foam the disturbance has
    /// whipped up, A unused. Same channel convention as the tiling ripple map beside it
    /// (`super::ripple`), so the shader reads the two the same way.
    pub(crate) image: Handle<Image>,
    /// Height now, and height one step ago — the two the explicit scheme needs.
    h: Vec<f32>,
    prev: Vec<f32>,
    /// Scratch for the step, kept allocated across frames.
    next: Vec<f32>,
    /// The encoded RGBA the render world uploads — staged here so the step and the encode stay in
    /// the main world and the render side only ever copies bytes.
    pixels: Vec<u8>,
    /// The window's lower corner, as a whole number of cells in **Bevy** XZ. Whole cells is the
    /// point — see the module doc.
    origin: IVec2,
    /// Unconsumed real time, in seconds, waiting to become whole [`SIM_DT`] steps.
    accum: f32,
    /// Has the window ever been placed? Until it has, there is nothing to scroll *from*.
    placed: bool,
}

impl RippleSim {
    fn new(image: Handle<Image>) -> Self {
        let n = SIM_CELLS * SIM_CELLS;
        Self {
            image,
            h: vec![0.0; n],
            prev: vec![0.0; n],
            next: vec![0.0; n],
            pixels: vec![0u8; n * 4],
            origin: IVec2::ZERO,
            accum: 0.0,
            placed: false,
        }
    }

    /// The window's lower corner in world yards.
    fn corner(&self) -> Vec2 {
        self.origin.as_vec2() * SIM_CELL
    }

    /// Carry the window to a new centre, shifting the field by whole cells and clearing whatever
    /// scrolls in. A jump further than the window is a teleport: nothing of the old field is
    /// relevant, so it all goes.
    fn recentre(&mut self, centre: Vec2) {
        let want = IVec2::new(
            (centre.x / SIM_CELL).round() as i32 - (SIM_CELLS / 2) as i32,
            (centre.y / SIM_CELL).round() as i32 - (SIM_CELLS / 2) as i32,
        );
        if !self.placed {
            self.origin = want;
            self.placed = true;
            return;
        }
        let d = want - self.origin;
        if d == IVec2::ZERO {
            return;
        }
        self.origin = want;
        if d.x.unsigned_abs() as usize >= SIM_CELLS || d.y.unsigned_abs() as usize >= SIM_CELLS {
            self.h.fill(0.0);
            self.prev.fill(0.0);
            return;
        }
        // The cell that was at (x + d) is now at x. Copying in the direction of travel would
        // overwrite sources before reading them, so each axis walks the way that cannot.
        for buf in [&mut self.h, &mut self.prev] {
            let rows: Box<dyn Iterator<Item = usize>> = if d.y > 0 {
                Box::new(0..SIM_CELLS)
            } else {
                Box::new((0..SIM_CELLS).rev())
            };
            for y in rows {
                let sy = y as i32 + d.y;
                let cols: Box<dyn Iterator<Item = usize>> = if d.x > 0 {
                    Box::new(0..SIM_CELLS)
                } else {
                    Box::new((0..SIM_CELLS).rev())
                };
                for x in cols {
                    let sx = x as i32 + d.x;
                    let v = if (0..SIM_CELLS as i32).contains(&sx)
                        && (0..SIM_CELLS as i32).contains(&sy)
                    {
                        buf[sy as usize * SIM_CELLS + sx as usize]
                    } else {
                        0.0
                    };
                    buf[y * SIM_CELLS + x] = v;
                }
            }
        }
    }

    /// Push the surface down over a disc — a body displacing water, which is the only kind of
    /// source this simulation has. Everything else a wake is comes out of the integrator.
    fn push(&mut self, at: Vec2, radius: f32, amount: f32) {
        let corner = self.corner();
        let c = (at - corner) / SIM_CELL;
        let r = radius / SIM_CELL;
        let lo = ((c - r).floor().max(Vec2::ZERO)).as_ivec2();
        let hi = ((c + r).ceil().min(Vec2::splat((SIM_CELLS - 1) as f32))).as_ivec2();
        if lo.x > hi.x || lo.y > hi.y {
            return;
        }
        let r2 = r * r;
        for y in lo.y..=hi.y {
            for x in lo.x..=hi.x {
                let d2 = (Vec2::new(x as f32 + 0.5, y as f32 + 0.5) - c).length_squared();
                if d2 > r2 {
                    continue;
                }
                // A raised cosine, so the source has no corner in it: a discontinuous push is a
                // step function, and a step function is every frequency at once — including the
                // lattice's own, which is the one frequency the scheme cannot carry.
                let f = 0.5 * (1.0 + (std::f32::consts::PI * (d2 / r2).sqrt()).cos());
                self.h[y as usize * SIM_CELLS + x as usize] -= amount * f;
            }
        }
    }

    /// One step of the wave equation over the whole lattice. The border ring is held at zero and
    /// the [`EDGE_ABSORB`] cells inside it are damped progressively — the outward wave loses its
    /// amplitude on the way out instead of turning around at a wall.
    fn step(&mut self) {
        for y in 1..SIM_CELLS - 1 {
            let edge_y = y.min(SIM_CELLS - 1 - y);
            for x in 1..SIM_CELLS - 1 {
                let i = y * SIM_CELLS + x;
                let lap = self.h[i - 1] + self.h[i + 1] + self.h[i - SIM_CELLS]
                    + self.h[i + SIM_CELLS]
                    - 4.0 * self.h[i];
                let edge = edge_y.min(x.min(SIM_CELLS - 1 - x));
                let absorb = if edge < EDGE_ABSORB {
                    // Linear from a hard 0.72 at the outermost live cell up to no extra loss at
                    // the band's inner edge. Gradual on purpose: an abrupt change in damping is
                    // itself an impedance step, and a wave reflects off one of those too.
                    0.72 + 0.28 * (edge as f32 / EDGE_ABSORB as f32)
                } else {
                    1.0
                };
                self.next[i] =
                    ((2.0 * self.h[i] - self.prev[i] + SIM_C2 * lap) * SIM_DAMP) * absorb;
            }
        }
        // The outer ring never carries anything, so it costs nothing to leave at zero, and having
        // it there is what lets the loop above skip its bounds checks.
        for x in 0..SIM_CELLS {
            self.next[x] = 0.0;
            self.next[(SIM_CELLS - 1) * SIM_CELLS + x] = 0.0;
            self.next[x * SIM_CELLS] = 0.0;
            self.next[x * SIM_CELLS + SIM_CELLS - 1] = 0.0;
        }
        std::mem::swap(&mut self.prev, &mut self.h);
        std::mem::swap(&mut self.h, &mut self.next);
    }

    /// Peak height and peak curvature over the field — the two numbers the encoding's scales are
    /// guesses about, reported under `$WOW_WAKE_DEBUG` so they can be measured instead.
    fn peaks(&self) -> (f32, f32) {
        let at = |x: usize, y: usize| self.h[y * SIM_CELLS + x];
        let mut peak_h = 0.0f32;
        let mut peak_lap = 0.0f32;
        for y in 1..SIM_CELLS - 1 {
            for x in 1..SIM_CELLS - 1 {
                peak_h = peak_h.max(at(x, y).abs());
                let lap = at(x - 1, y) + at(x + 1, y) + at(x, y - 1) + at(x, y + 1) - 4.0 * at(x, y);
                peak_lap = peak_lap.max(lap.abs());
            }
        }
        (peak_h, peak_lap)
    }
}

/// One frame's encoded field on its way to the GPU.
///
/// **This exists because the obvious way does not work.** Writing the new pixels into the `Image`
/// asset each frame — `Assets::get_mut`, then mutate — makes Bevy tear the old GPU texture down and
/// build a new one, while the liquid materials' bind groups go on pointing at the old one. Nothing
/// errors: the water simply shows the texture's startup contents for ever, which in this case is a
/// flat field, which looks exactly like a simulation that is running and producing nothing. A whole
/// live capture went into finding that out (and a wrong reading of a swimmer's shadow as a wake on
/// the way).
///
/// So the asset is created once and never touched, and the per-frame contents are copied into the
/// texture the render world already made — the same shape as the reflection's parameter buffer next
/// door, and for the same underlying reason. The `Arc` is what makes the per-frame extract a
/// pointer copy rather than 147 KB.
#[derive(Resource, Clone, Default, ExtractResource)]
struct RippleFrame {
    id: AssetId<Image>,
    pixels: Option<Arc<Vec<u8>>>,
}

/// Copy this frame's field into the texture the materials are already bound to.
fn upload_wake(
    queue: Res<RenderQueue>,
    images: Res<RenderAssets<GpuImage>>,
    frame: Option<Res<RippleFrame>>,
) {
    let Some(frame) = frame else { return };
    let Some(pixels) = frame.pixels.as_ref() else {
        return;
    };
    // Absent for the first frame or two, while the image is still being prepared.
    let Some(gpu) = images.get(frame.id) else {
        return;
    };
    queue.write_texture(
        gpu.texture.as_image_copy(),
        pixels,
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * SIM_CELLS as u32),
            rows_per_image: Some(SIM_CELLS as u32),
        },
        Extent3d {
            width: SIM_CELLS as u32,
            height: SIM_CELLS as u32,
            depth_or_array_layers: 1,
        },
    );
}

/// Encode a height field into the bound texture's bytes: central-difference slope in R/G and the
/// height itself in B, both signed around the byte's middle. Same layout as the tiling ripple map
/// next door, which is not a coincidence — the shader reads the two the same way.
fn encode(h: &[f32], data: &mut [u8]) {
    let at = |x: usize, y: usize| h[y * SIM_CELLS + x];
    for y in 0..SIM_CELLS {
        for x in 0..SIM_CELLS {
            let (xm, xp) = (x.saturating_sub(1), (x + 1).min(SIM_CELLS - 1));
            let (ym, yp) = (y.saturating_sub(1), (y + 1).min(SIM_CELLS - 1));
            let dx = (at(xp, y) - at(xm, y)) * (0.5 / SIM_CELL) * WAKE_SLOPE;
            let dy = (at(x, yp) - at(x, ym)) * (0.5 / SIM_CELL) * WAKE_SLOPE;
            let i = (y * SIM_CELLS + x) * 4;
            data[i] = ((dx * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u8;
            data[i + 1] = ((dy * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u8;
            data[i + 2] =
                ((at(x, y) * WAKE_HEIGHT * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u8;
            data[i + 3] = 255;
        }
    }
}

/// The lattice's own texture: no mips (it is rewritten every frame and never minified far), and
/// **clamped**, so a sample taken outside the window returns the rim — which the absorbing band has
/// already brought to zero.
fn sim_image() -> Image {
    let mut image = Image::new_uninit(
        Extent3d {
            width: SIM_CELLS as u32,
            height: SIM_CELLS as u32,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        TextureFormat::Rgba8Unorm,
        // Render world only. The asset is created once and **never touched again** — see
        // [`upload_wake`] for why the per-frame contents do not go through it.
        RenderAssetUsages::RENDER_WORLD,
    );
    // The render side copies into this texture every frame rather than replacing it.
    image.texture_descriptor.usage |= TextureUsages::COPY_DST;
    // Flat water: zero slope and zero height, all three signed around the byte's middle.
    image.data = Some(
        [128u8, 128, 128, 255]
            .iter()
            .copied()
            .cycle()
            .take(SIM_CELLS * SIM_CELLS * 4)
            .collect(),
    );
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        address_mode_u: ImageAddressMode::ClampToEdge,
        address_mode_v: ImageAddressMode::ClampToEdge,
        mag_filter: ImageFilterMode::Linear,
        min_filter: ImageFilterMode::Linear,
        ..default()
    });
    image
}

fn setup_ripple_sim(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let image = images.add(sim_image());
    commands.insert_resource(RippleSim::new(image));
}

/// Carry the window, take this frame's disturbances, integrate, and publish.
///
/// Inert on the reference lane, where `water_fx`'s decals are the wake and this would be a second
/// one: the strength lane goes to zero and the field is left alone.
#[allow(clippy::too_many_arguments)]
fn drive_ripple_sim(
    style: Res<WaterStyle>,
    time: Res<Time>,
    mut sim: ResMut<RippleSim>,
    mut frame: ResMut<RippleFrame>,
    mut data: ResMut<WaterReflectData>,
    viewer: Res<crate::view::Viewer>,
    cam: Query<&GlobalTransform, With<crate::view::WorldCamera>>,
    units: Query<(Entity, &Transform, &WorldUnit), Without<ViewerUnit>>,
    water: Query<&WaterChunkInfo>,
    index: Res<WaterIndex>,
    // Last frame's position per streamed unit — a unit's transform is streamed without a velocity,
    // and a wake is made of velocity.
    mut was: Local<EntityHashMap<Vec3>>,
    // Fractions of a pulse owed to each unit, from distance travelled and from time standing still.
    mut credit: Local<EntityHashMap<f32>>,
    // Seconds since the last `$WOW_WAKE_DEBUG` line.
    mut report: Local<f32>,
) {
    // x/y = the window's lower corner, z = 1/extent, w = strength.
    let mut params = [0.0f32; 4];
    if *style != WaterStyle::Stylised {
        data.0[16..20].copy_from_slice(&params);
        return;
    }
    // The window follows the avatar when there is one and the camera otherwise — a free-flying
    // eye (the world viewer, a capture run) still wants water that moves under it.
    let Some(centre) = viewer
        .at
        .or_else(|| cam.iter().next().map(|t| t.translation()))
    else {
        data.0[16..20].copy_from_slice(&params);
        return;
    };
    sim.recentre(Vec2::new(centre.x, centre.z));

    // The surface height under a point, through the same index the foam decals use — a hash lookup
    // per unit, not a walk over every loaded surface.
    let depth_at = |p: Vec3| -> Option<f32> {
        let w = bevy_to_wow(p);
        index
            .over(w[0], w[1])
            .iter()
            .find_map(|&e| water.get(e).ok()?.surface_z_at(w[0], w[1]))
            .map(|surface| surface - w[2])
    };

    let dt = time.delta_secs().clamp(0.0, 0.25);

    // **Discrete pulses, not a continuous press.** Pressing on the surface every frame under the
    // swimmer does not make a wake, it digs a permanent bowl and holds it there: the water reaches
    // a steady state in which the disturbance is a bright dish moving with the body, with no
    // structure in it at all. That is what the first two live captures showed, and no amount of
    // scaling the amplitude would have fixed it, because the shape was wrong rather than the size.
    //
    // A body in water sheds ONE ring, then another, then another. Each is a transient that leaves
    // and does not come back, and a wake is what several of them look like once the source has
    // moved on between each. So the surface is struck once every [`PULSE_STRIDE`] yards of travel —
    // the same "one per so-many yards" cadence the reference's own decal wake uses — and at no
    // other time.
    let mut fire = |sim: &mut RippleSim,
                    entity: Entity,
                    p: Vec3,
                    step: Vec2,
                    height: f32| {
        let moved = step.length();
        if moved <= MOVED_EPSILON {
            return;
        }
        let credit = credit.entry(entity).or_insert(0.0);
        *credit += moved / PULSE_STRIDE;
        if *credit < 1.0 {
            return;
        }
        // At most one pulse a frame however long the frame was: the point of the stride is that
        // pulses are spaced along the path, and a stall that fired six at once would put them all
        // in the same place.
        *credit = (*credit - 1.0).min(1.0);
        let Some(depth) = depth_at(p) else { return };
        if depth < PUSH_MIN_DEPTH {
            return;
        }
        // …and nothing from a body that has gone under rather than along — the reference's own
        // gate, with its own ramp (see [`PUSH_MIN_DEPTH`]).
        let Some(reach) = crate::water_fx::params::depth_strength(height, depth) else {
            return;
        };
        // Born at the bow, not at the middle — see [`PUSH_BOW_OFFSET`].
        let at = Vec2::new(p.x, p.z) + step / moved * PUSH_BOW_OFFSET;
        sim.push(at, PUSH_RADIUS, PULSE_AMOUNT * reach);
    };
    let mut seen: Vec<Entity> = Vec::new();
    // The avatar goes through the same last-position channel as everyone else, rather than through
    // its commanded speed: the wake now needs a DIRECTION as well as a distance, and the direction
    // a body actually moved is the one its bow points along — a commanded speed has none.
    if let Some(body) = viewer.at {
        seen.push(Entity::PLACEHOLDER);
        let step = was
            .insert(Entity::PLACEHOLDER, body)
            .map_or(Vec2::ZERO, |prev| {
                Vec2::new(body.x, body.z) - Vec2::new(prev.x, prev.z)
            });
        fire(&mut sim, Entity::PLACEHOLDER, body, step, viewer.height);
    }
    for (entity, transform, unit) in &units {
        if !unit.wades {
            continue;
        }
        let p = transform.translation;
        seen.push(entity);
        let step = was.insert(entity, p).map_or(Vec2::ZERO, |prev| {
            Vec2::new(p.x, p.z) - Vec2::new(prev.x, prev.z)
        });
        fire(&mut sim, entity, p, step, unit.height);
    }
    // A unit that has gone (out of range, dead, despawned) must not keep state here, or both maps
    // grow for as long as the session runs.
    if was.len() > seen.len() {
        was.retain(|e, _| seen.contains(e));
        credit.retain(|e, _| seen.contains(e));
    }

    sim.accum += dt;
    let mut steps = 0;
    while sim.accum >= SIM_DT && steps < SIM_MAX_STEPS {
        sim.accum -= SIM_DT;
        sim.step();
        steps += 1;
    }
    if steps == SIM_MAX_STEPS {
        sim.accum = 0.0;
    }
    if steps > 0 {
        let RippleSim {
            h, pixels, image, ..
        } = &mut *sim;
        encode(h, pixels);
        *frame = RippleFrame {
            id: image.id(),
            pixels: Some(Arc::new(std::mem::take(pixels))),
        };
        // `Arc` is what the extract clones each frame, so the staging buffer is handed over rather
        // than copied; take a fresh one for the next step.
        *pixels = vec![0u8; SIM_CELLS * SIM_CELLS * 4];
        if std::env::var_os("WOW_WAKE_DEBUG").is_some() && *report >= 1.0 {
            *report = 0.0;
            let (h, lap) = sim.peaks();
            info!("WAKE peak_h={h:.4} peak_lap={lap:.5} slope_full={WAKE_SLOPE} height_full={WAKE_HEIGHT}");
        }
        *report += dt;
    }

    let corner = sim.corner();
    // `$WOW_WAKE_SHOW=1` raises the strength lane past 1 and the shader paints the wave field
    // itself onto the water instead of shading with it. It exists because the thing this file
    // produces is *subtle by design*, and subtle is exactly what cannot be verified by looking: the
    // first live capture of it was reported as a clear V-shaped wake and was the swimmer's shadow.
    // A view with no interpretation in it settles that in one frame.
    let show = std::env::var_os("WOW_WAKE_SHOW").is_some();
    params = [
        corner.x,
        corner.y,
        1.0 / SIM_YARDS,
        if show { 2.0 } else { 1.0 },
    ];
    data.0[16..20].copy_from_slice(&params);
}

pub(super) fn register(app: &mut App) {
    app.init_resource::<RippleFrame>()
        .add_plugins(ExtractResourcePlugin::<RippleFrame>::default())
        .add_systems(
            Startup,
            setup_ripple_sim
            .after(benilla_assets::AssetSet::Open)
                // The materials bind this image, and a material built before it exists would have
                // to be rebuilt to get it — the same ordering the reflection target needs.
                .before(super::surface::setup_liquid),
        )
        .add_systems(Update, drive_ripple_sim);
    app.sub_app_mut(RenderApp)
        .add_systems(Render, upload_wake.in_set(RenderSystems::PrepareResources));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sim() -> RippleSim {
        RippleSim::new(Handle::default())
    }

    /// The whole thing is worthless if it diverges, and an explicit scheme diverges silently — the
    /// field is finite, then a second later every cell is NaN. Drive it hard and check it stays
    /// bounded.
    #[test]
    fn the_integrator_stays_bounded_under_a_hammering() {
        let mut s = sim();
        s.placed = true;
        for i in 0..2000 {
            let t = i as f32 * SIM_DT;
            s.push(Vec2::new(24.0 + t, 24.0), PUSH_RADIUS, 0.05);
            s.step();
        }
        assert!(
            s.h.iter().all(|v| v.is_finite() && v.abs() < 10.0),
            "the wave field left its stable region"
        );
    }

    /// A disturbance has to travel, and to travel at [`SIM_C2`]'s own speed.
    ///
    /// Propagation is the one property that separates a simulation from a decal, so it gets a test
    /// rather than an eye — but the SPEED has to be in the test too, because it is the number that
    /// decides the wake's whole shape (a source slower than its waves makes rings; faster, a V) and
    /// because a test that only asked "did anything arrive at cell 20" had to be rewritten the day
    /// that speed was retuned. Both bounds are derived from the constants, so the test moves with
    /// them instead of going stale.
    #[test]
    fn a_push_travels_at_the_wave_speed() {
        let mut s = sim();
        s.placed = true;
        let mid = SIM_CELLS / 2;
        s.push(Vec2::new(SIM_YARDS * 0.5, SIM_YARDS * 0.5), 0.5, 1.0);
        let at = |s: &RippleSim, cells: usize| s.h[mid * SIM_CELLS + mid + cells].abs();

        // c = C · Δx/Δt, the lattice's phase speed in yards per second.
        let speed = SIM_C2.sqrt() * SIM_CELL / SIM_DT;
        let seconds = 2.0_f32;
        let front = (speed * seconds / SIM_CELL) as usize;
        assert!(front > 4, "the front must clear the source to be measurable");

        assert!(at(&s, front) < 1e-6, "arrived before a single step ran");
        for _ in 0..(seconds / SIM_DT) as usize {
            s.step();
        }
        let behind = at(&s, front / 2);
        assert!(
            behind > 1e-4,
            "nothing behind the front — the wave is not propagating"
        );
        // A ratio rather than a floor: an explicit lattice scheme is dispersive, so a little
        // amplitude always runs ahead of the analytic front. What must hold is that the
        // disturbance is LOCALISED around where `c` puts it — if it were not, the field would be
        // a uniform swell and there would be no wake shape to speak of.
        assert!(
            at(&s, front * 2) < behind * 0.1,
            "energy well ahead of where the wave speed allows — it is not obeying c"
        );
    }

    /// Scrolling must carry the field with the world, not smear it: a feature sitting at a world
    /// position has to still be at that world position after the window moves under it.
    #[test]
    fn scrolling_the_window_keeps_features_at_their_world_position() {
        let mut s = sim();
        s.recentre(Vec2::splat(SIM_YARDS * 0.5));
        let at = Vec2::new(SIM_YARDS * 0.5 + 4.0, SIM_YARDS * 0.5);
        s.push(at, 0.5, 1.0);
        let sample = |s: &RippleSim, p: Vec2| {
            let c = ((p - s.corner()) / SIM_CELL).as_ivec2();
            s.h[c.y as usize * SIM_CELLS + c.x as usize]
        };
        let before = sample(&s, at);
        assert!(before.abs() > 0.1, "the push did not land");
        s.recentre(Vec2::new(SIM_YARDS * 0.5 + 5.0, SIM_YARDS * 0.5 + 3.0));
        assert!(
            (sample(&s, at) - before).abs() < 1e-6,
            "the field did not travel with the window"
        );
    }

    /// A teleport out of the window leaves nothing behind — otherwise the wake of wherever you were
    /// standing arrives with you.
    #[test]
    fn a_jump_past_the_window_clears_the_field() {
        let mut s = sim();
        s.recentre(Vec2::splat(SIM_YARDS * 0.5));
        s.push(Vec2::splat(SIM_YARDS * 0.5), 1.0, 1.0);
        assert!(s.h.iter().any(|v| v.abs() > 0.1));
        s.recentre(Vec2::splat(SIM_YARDS * 0.5 + 4000.0));
        assert!(
            s.h.iter().all(|v| *v == 0.0),
            "a teleport carried the old water with it"
        );
    }
}
