//! **The Bevy render glue** — the animated liquid surfaces themselves.
//!
//! One shared material per (kind, [`LiquidPath`], scroll) over a `texture_2d_array` of the kind's
//! frames, the two spawn paths (an ADT chunk's MCLQ and a WMO group's MLIQ), and the flat mesh build. The faithful
//! shading model is `super`'s header and `liquid.wgsl`'s; the *position* side — the grid each
//! spawned surface publishes and the queries against it — is [`super::query`].
//!
//! **The 24 fps frame-flip and the lava scroll are computed IN the shader off `globals.time`**
//! (liquid.wgsl `anim_time`): both are pure functions of the wall clock and two per-material
//! constants baked at build, so no CPU ever mutates a liquid material again. The 24 Hz
//! `Assets::get_mut` cycler this replaced cost a measured 0.28 cpu_ms/frame at the SW pin (the
//! WOW_NO_LIQUID_ANIM bracket, 2026-08-18, negative in all three rounds): its per-tick Modified
//! chain re-uploaded ~14 material uniforms, rebuilt their Metal bind groups, and armed the
//! whole-population `AssetChanged` walks 24 times a second. A deterministic run bakes the freeze
//! at BUILD time (`anim.w = 0` → the shader clock reads 0 forever), so captures keep frame 0 /
//! scroll 0 bit-exactly — the same pin the old tick enforced (0600). Two clocks, unchanged in
//! spirit: animation = wall clock; day/night = server game-time.

use std::collections::{HashMap, HashSet};

use bevy::asset::RenderAssetUsages;
use bevy::camera::visibility::RenderLayers;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::pbr::ExtendedMaterial;
use bevy::prelude::*;

use super::query::{wet_footprint, FoamPatch, LiquidSource, WmoPool};
use super::{ripple, WaterStyle};
use crate::lighting::WATER_SHININESS;
use benilla_assets::coords::wow_to_bevy;
use benilla_assets::materials::{LiquidExt, LiquidMaterial};
use benilla_assets::LockRecover;
use benilla_assets::{liquid_frame_array, RenderConfig, WorldAssets};
use benilla_formats::{
    read_texture_mip_chain, terrain_height_at, BlpMipChain, ChunkMesh, LiquidKind, LiquidMesh,
};

/// The shared liquid materials, keyed by [`LiquidKey`]. Read by the terrain streamer (via [`spawn_liquids`] /
/// [`spawn_wmo_liquids`]) to material the per-chunk water meshes. Absent when the client has no data
/// (no `WorldAssets`).
#[derive(Resource, Default)]
pub(crate) struct LiquidAssets {
    materials: HashMap<LiquidKey, LiquidEntry>,
}

/// **Which of the reference's liquid renderers a surface belongs to.**
///
/// The reference has three, and until now benilla had one. `0x6b62e0` sends the type nibble to a
/// category, and category 0 (river/water — nibbles 0/4/8) splits again on the owning group's
/// `MOGP.flags & 0x48`:
///
/// * [`Adt`](LiquidPath::Adt) — the MCLQ queues `0x6851b0` (river) / `0x685010` (ocean). **Two**
///   texture stages: a static depth-ramp on stage 0 addressed by `tc0 = (0.5, LUT[depthByte])`, the
///   animated sheet on stage 1 at `tc1 = (col·¼, row·¼)`, combined by `ocean0_s.bls`. The only path
///   with a depth ramp, and its vertex carries no colour at all.
/// * [`WmoExterior`](LiquidPath::WmoExterior) — `0x6b6630`. **One** stage (the sheet), a 9-float
///   vertex carrying an up normal *and* a colour dword, and the pixel program `MapObjExtWater0.bls`
///   bound at `0x6b6654`.
/// * [`WmoInterior`](LiquidPath::WmoInterior) — `0x6b6420`. One stage, a 6-float vertex with **no
///   normal**, lighting forced OFF, and no pixel program at all.
///
/// Magma and slime take none of these: they share `0x6b68f0` (WMO) / `0x68dca0` (ADT) whatever their
/// group's flags say, so for the fullbright kinds this enum records only which *spawner* made the
/// surface — which is still what picks the fog block below.
///
/// This also subsumes the old `interior: bool` lane, which existed for the **fog block**: the
/// reference decides fog per *pass*, not per liquid type, and a WMO group's own pool is drawn by the
/// WMO liquid pass, which re-submits the smoothed interior fog block (`0x6b6323`–`0x6b6342`) under
/// the same `[0xca7f00]` gate as the WMO *geometry* pass — so a pool and the walls around it always
/// fog alike, while ADT liquid submits no fog and draws under the once-a-frame scene block. (VERIFIED
/// wow-re `fog-env-state` §5's 6-site submit census + `liquid-render-state-sided` §5; decision 0691.)
/// `WmoInterior` is exactly the old `interior == true`, so [`Self::interior_fog`] is the whole of
/// what that flag used to say — with the renderer identity now carried alongside it rather than
/// inferred from it.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum LiquidPath {
    /// An ADT chunk's MCLQ surface.
    Adt,
    /// A WMO group's MLIQ surface, the group being exterior/exterior-lit (`MOGP.flags & 0x48 != 0`).
    WmoExterior,
    /// A WMO group's MLIQ surface, the group being a true interior (`MOGP.flags & 0x48 == 0`).
    WmoInterior,
}

impl LiquidPath {
    /// A WMO group's own class, as the reference's `[owner+0x10] & 0x48` test reads it.
    pub(crate) fn wmo(interior: bool) -> Self {
        if interior {
            Self::WmoInterior
        } else {
            Self::WmoExterior
        }
    }

    /// Does this surface fog with the **interior** block rather than the scene block?
    fn interior_fog(self) -> bool {
        matches!(self, Self::WmoInterior)
    }

    /// The renderer selector `liquid.wgsl` branches on (`LiquidParams.path.x`).
    fn shader_id(self) -> f32 {
        match self {
            Self::Adt => 0.0,
            Self::WmoExterior => 1.0,
            Self::WmoInterior => 2.0,
        }
    }

    /// Does a pool on this arm take its body colour from its **own MOMT `diffColor`**, baked into the
    /// mesh's vertex colour? Only the interior arm: the exterior one is lit and reads a live
    /// `Light.dbc` band instead, and ADT liquid has no MOMT to index at all.
    fn takes_material_body_color(self) -> bool {
        matches!(self, Self::WmoInterior)
    }
}

/// Which shared material a surface takes: its kind, its renderer ([`LiquidPath`]), and whether it
/// takes the lava/slime UV scroll.
///
/// The scroll lane is a per-*nibble*, per-*path* property that [`LiquidKind`] deliberately collapses
/// (2 and 6 are both `Magma`), so it cannot be derived from the kind and has to key the material —
/// see [`scrolls`].
///
/// The variants share one decoded frame array — only the tiny uniform differs — so the extra
/// materials cost a handful of bytes, not a second copy of 30 × 256² textures.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct LiquidKey {
    kind: LiquidKind,
    path: LiquidPath,
    /// This surface takes the animated stage-0 texture matrix ([`scrolls`]).
    scroll: bool,
}

/// Does this surface take the reference's animated stage-0 texture matrix — the lava/slime scroll?
///
/// **Only WMO MLIQ surfaces whose type nibble is 6 or 7** (VERIFIED wow-re `liquid-uv-scroll-law.md`:
/// the WMO magma/slime kernel `0x6b68f0` gates on `and esi,0xc; cmp esi,4` over the `0x6ba970`
/// nibble, and builds a matrix that is identity except element 13 = the v-translate). Three
/// consequences that are easy to get wrong, all of them load-bearing:
///
/// * **Nibbles 2 and 3 do NOT scroll**, though they reach the same kernel and draw the same
///   `lava.blp`/`slime.blp`. In the shipped data that is the difference between Blackrock's lava
///   (nibble 6, scrolls) and the **Ironforge Great Forge's** (nibble 2, static) — the same texture,
///   two behaviours, and the reason "lava scrolls" is not a statement you can make about lava.
/// * **ADT liquid never scrolls, magma included** — the open-world Searing Gorge / Burning Steppes
///   lava is nibble 6 too, but its queue is a *different* kernel (`0x68dca0`) which pushes a
///   byte-exact identity with element 13 hard-zeroed at `0x68dda1`. So the gate is the nibble AND
///   the path; a nibble-only test would slide every lava pool in the open world.
/// * The phase is driven off the reference's uptime clock, so only its **rate and period** are
///   reproducible — never its absolute value. We drive ours off the same wall clock the frame
///   cycler uses.
fn scrolls(nibble: u8) -> bool {
    matches!(nibble, 6 | 7)
}

struct LiquidEntry {
    material: Handle<LiquidMaterial>,
}

impl LiquidAssets {
    /// The shared material for a liquid kind on a given renderer, scrolling or not, if its frames
    /// loaded. `scroll` is [`scrolls`]'s verdict; a `true` for a kind that has no scrolling variant
    /// simply misses, which is the same "no material" path a failed frame load takes.
    pub(crate) fn material(
        &self,
        kind: LiquidKind,
        path: LiquidPath,
        scroll: bool,
    ) -> Option<Handle<LiquidMaterial>> {
        self.materials
            .get(&LiquidKey { kind, path, scroll })
            .map(|e| e.material.clone())
    }
}

/// Marks a spawned water surface (one per liquid MCNK chunk), so it can be queried/culled as a group.
#[derive(Component)]
pub(crate) struct LiquidSurface;

/// Attribution kill-switch: `$WOW_NO_LIQUID=1` hides every liquid **surface**, and only the surface —
/// the swim grid, the submersion verdict, the foam and the ambient loops all stay live, because those
/// ride sibling components on the same entity ([`WaterChunkInfo`], [`FoamPatch`],
/// [`LiquidSoundSource`]) rather than its `Visibility`.
///
/// It exists because "is that thing occluding the NPC a liquid surface?" was otherwise unanswerable
/// from inside the game. A liquid surface is opaque where it is deep (`WATER_DEEP_ALPHA` = 1.0) and
/// fully opaque for the fullbright kinds, and it carries no silhouette of its own that reads as
/// *water* at a glancing angle — so a mis-placed one is indistinguishable by eye from a mis-placed
/// wall, and both look like a hard straight seam across the scene. One A/B now separates them; the
/// alternative was a screenshot argument. Same shape as `$WOW_NO_PARTICLES` / `$WOW_NO_FFX`.
///
/// **An ordered override, not a second authority** — the shape `apply_self_model_fade` already
/// uses against the model-`Visibility` authority (`model_render::ModelVisSet`). Every liquid
/// surface now has a real per-frame `Visibility` owner: an ADT surface is `ExteriorScene`, written
/// by `exterior_cull::apply_exterior_cull`; a WMO pool carries `WmoGroupVis`, written by
/// `model_render::visibility::apply_model_visibility`'s `group_only` query. So this runs in
/// `PostUpdate` **after** both (decision 1652) and wins the frame.
///
/// It used to run in `Update` on `Added`, writing `Hidden` exactly once when a surface streamed in.
/// That was already broken for WMO pools before this change — 0689 gave them `WmoGroupVis` in
/// `Update`, so the authority re-wrote `Inherited` over the kill-switch on the very next frame and
/// `$WOW_NO_LIQUID` silently stopped hiding Stormwind's canals and every dungeon pool. Nobody
/// noticed because the switch's usual subject is a lake. Tagging ADT liquid would have taken the
/// remaining half the same way; making it an override fixes both at once.
///
/// Change-gated, so the steady state is one compare per surface and no change-detection churn —
/// and the system is `run_if`'d off the env var, so an ordinary run never schedules it at all.
pub(super) fn hide_liquid_surfaces(mut surfaces: Query<&mut Visibility, With<LiquidSurface>>) {
    for mut vis in &mut surfaces {
        if *vis != Visibility::Hidden {
            *vis = Visibility::Hidden;
        }
    }
}

/// What the **above-water ambient-loop system** needs beyond the surface's geometry (wow-re
/// `liquid-ambience-loop.md`, decision 0506): the sound-class nibble the driver resolves through
/// `SoundWaterType.dbc`. Attached to **every** liquid surface, the fullbright kinds included (the
/// Ironforge lava rumble, Undercity slime).
///
/// It used to carry its own copy of the footprint — bounds + a surface height — because when 0506
/// wrote it, magma and slime carried no [`WaterChunkInfo`] to read them from. 0634 gave every kind
/// one (that is what made lava swimmable), so the copy became a third set of numbers describing the
/// same surface, and its height stayed the chunk maximum after the grid sample landed. The driver
/// queries `(&LiquidSoundSource, &WaterChunkInfo)` instead — the pairing both spawn paths already
/// guarantee — and reads the geometry from the one component that owns it.
#[derive(Component)]
pub(crate) struct LiquidSoundSource {
    /// The surface's sound-class nibble (`class = n & 3`, `FluidSpeed = n & 0xc`).
    pub(crate) nibble: u8,
}

/// Spawn a set of water surfaces — one flat mesh per [`LiquidMesh`], on its [`LiquidKind`]'s shared
/// animated material. Used by the `AdtTile` pipeline (`terrain_stream`). No-op when the client has no
/// data (`liquid_assets` absent) or a kind's frames didn't load. Spawned entities are pushed onto
/// `entities` so they despawn with their tile.
pub(crate) fn spawn_liquids<'a>(
    commands: &mut Commands,
    liquids: impl Iterator<Item = &'a LiquidMesh>,
    // The owning tile's terrain, for the shore foam: the waterline on a coast is where the ground
    // rises through the water plane, and nothing in the liquid data alone records it
    // ([`shore_distances`]).
    chunks: &[ChunkMesh],
    liquid_assets: Option<&LiquidAssets>,
    meshes: &mut Assets<Mesh>,
    entities: &mut Vec<Entity>,
) {
    let Some(liquid) = liquid_assets else {
        return;
    };
    // One lattice for the whole tile, built before anything is spawned: a chunk cannot tell where
    // its water ends by looking only at itself (see [`WetLattice`]).
    let batch: Vec<&LiquidMesh> = liquids.collect();
    let lattice = WetLattice::build(batch.iter().copied());
    let shoreline = Shoreline::build(&batch, chunks);
    for lq in batch {
        // ADT liquid always takes the SCENE fog: the ADT liquid passes submit no fog block of their
        // own, so they draw under the once-a-frame scene submit (wow-re `fog-env-state` §5).
        //
        // And it NEVER scrolls — `scroll: false` here is not a default, it is the finding. Shipped
        // ADT magma is nibble 6 throughout, the very nibble that scrolls on the WMO path, but the
        // ADT queue is a different kernel that hard-zeroes the v-translate ([`scrolls`]). Deriving
        // the flag from `lq.sound_nibble` here — the obvious-looking thing — would slide every
        // Searing Gorge and Burning Steppes lava pool in the game.
        let Some(material) = liquid.material(lq.kind, LiquidPath::Adt, false) else {
            continue; // this kind's frames failed to load (warned at setup)
        };
        // The world-space liquid grid (MCLQ positions are already absolute WoW, so the IDENTITY
        // transform is a no-op round-trip).
        let info = wet_footprint(lq, &Transform::IDENTITY, LiquidSource::AdtChunk);
        let foam = !lq.kind.is_fullbright(); // white surf is a water thing
        entities.push(
            commands
                .spawn((
                    Mesh3d(meshes.add(liquid_bevy_mesh(lq, None, lattice.as_ref(), shoreline.as_ref()))),
                    MeshMaterial3d(material),
                    Transform::IDENTITY,
                    LiquidSurface,
                    // Liquid rides its own render layer so the stylised look's mirrored camera
                    // can leave it out — see `reflect`. The world camera renders layer 0 AND
                    // this one ([`reflect::stamp_world_camera_layers`]); nothing else does.
                    RenderLayers::layer(super::WATER_RENDER_LAYER),
                    // **Open-world liquid is exterior scene** — the reference's ADT liquid
                    // producer `0x683ab0` is called only from the per-window populate
                    // `0x682fa0`, exactly like ADT terrain (`0x683bf0`) and doodads
                    // (`0x683700`), so from inside a cavern the lake overhead draws only where a
                    // doorway window admits it, and not at all from a sealed room (decision
                    // 1652). This was 0774's last knowingly-ungated bucket, deferred again by
                    // 0784 on the belief that liquid "has its own lane" — it does not: nothing
                    // wrote an ADT surface's `Visibility` at all, so `apply_exterior_cull` is its
                    // sole authority and the tag is the whole change.
                    //
                    // Granularity is already the reference's: one entity per MCNK liquid layer
                    // (33.333 yd), never the 533 yd tile — the half of the cull that decision
                    // 0780 is about. The `IDENTITY` transform is not incidental either: MCLQ
                    // positions are absolute, so the mesh `Aabb` Bevy derives is already the
                    // world-space box the window test wants.
                    crate::exterior_cull::ExteriorScene,
                    info,
                    LiquidSoundSource {
                        nibble: lq.sound_nibble,
                    },
                ))
                .id(),
        );
        // Foam is water-only; the cells it clips against ride `info`, not the marker.
        if foam {
            commands
                .entity(*entities.last().expect("just pushed"))
                .insert(FoamPatch);
        }
    }
}

/// Every liquid cell a batch of surfaces covers, on one shared lattice, so a surface can measure
/// its distance to the water's edge across chunk seams.
///
/// A [`LiquidMesh`] is one MCNK's 9x9 grid, 33 yards square. Asking a single grid where its water
/// ends gives the wrong answer at every chunk border, because a chunk whose water runs straight into
/// the next one looks, from the inside, exactly like a chunk whose water stops there — and drawing
/// foam on that reading would put a line across open sea every 33 yards. The tile streamer hands
/// [`spawn_liquids`] every chunk of a tile at once, so the lattice is built from all of them
/// together and each surface measures against the whole.
///
/// `known` is the union of the batch's grids — the ground this batch can actually speak for. A cell
/// outside it is *unknown*, not dry: at a tile seam the water continues into a tile whose own
/// surfaces are a different batch, and treating that seam as land would draw the same false line
/// 533 yards long. The cost is a chunk entirely covered by water whose neighbour carries no liquid
/// at all, where the true edge lies on the seam and goes unmarked; that needs the coastline to run
/// along a chunk border for its whole length, which real ones do not.
pub(crate) struct WetLattice {
    /// The wet/dry boundary as a **traced curve**, not as the set of cells it separates.
    ///
    /// Built over a whole TILE on the ADT path and a whole PLACEMENT on the WMO one, never over a
    /// single surface. Stormwind's canal is a couple of dozen small grids, one per group, and each
    /// carries a ring of dry cells around its own water where its group's data ends. Traced alone,
    /// every segment calls the join with its neighbour a shore, and the canal drew a foam line
    /// straight across itself at each of them — a line in the middle of the water, repeating down
    /// its whole length. Folded onto one lattice, a cell one segment calls dry and the next calls
    /// wet is simply wet, and the only boundary left is the one against the quay. (The groups'
    /// grids share a common 4.1667-yard lattice, checked against the shipped data, so folding them
    /// is exact rather than approximate.)
    ///
    /// Measuring straight to the dry cells is the obvious thing and it is what this did first. It
    /// is also wrong, and visibly so: a union of axis-aligned 4.17-yard squares IS a staircase, so
    /// however exactly you measure to it, a band of constant distance around it comes out as a
    /// staircase too — a bright zig-zag with square corners running along a shoreline that is
    /// actually a smooth diagonal. Reading the same cells as a *fractional wetness at their
    /// corners* and following its half-way contour cuts each corner cell across the diagonal
    /// instead, which is the line the cells were a coarse sampling of in the first place.
    ///
    /// The cost is features one cell across: a lone dry cell in open water has all four of its
    /// corners at three-quarters wet, so no contour passes through it and it gets no foam. On an
    /// ADT that is exactly the case the traced depth contour ([`Shoreline`]) already covers — a
    /// lone dry cell is dry because the ground there is out of the water — so the two sources are
    /// complementary. A WMO pool has no ground to trace against and would lose a one-cell island,
    /// which is a shape no authored pool has.
    edge: Option<Shoreline>,
}

impl WetLattice {
    /// Fold every mesh in a batch onto one lattice. `None` when the batch has no usable grid.
    ///
    /// Cells are keyed by their CENTRE, which sits half a pitch off the lattice lines and so lands
    /// unambiguously inside one cell however the floor rounds. Positions are taken as they come —
    /// absolute WoW yards for MCLQ, model-local for a WMO pool — which is consistent as long as one
    /// batch is all of one kind, and it is.
    pub(crate) fn build<'a>(meshes: impl Iterator<Item = &'a LiquidMesh>) -> Option<Self> {
        let mut cell = 0.0_f32;
        let mut wet = HashSet::new();
        let mut known = HashSet::new();
        for lq in meshes {
            let (cols, rows) = (lq.grid[0] as usize, lq.grid[1] as usize);
            if cols < 2 || rows < 2 {
                continue;
            }
            if cell <= 0.0 {
                let (a, b) = (lq.positions[0], lq.positions[1]);
                cell = ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt();
                if !(cell > 1e-3) {
                    return None;
                }
            }
            let (xt, yt) = (cols - 1, rows - 1);
            for j in 0..yt {
                for i in 0..xt {
                    // The cell's centre, as the mean of its four corners — right whichever way the
                    // grid axes happen to lie in the world.
                    let corners = [
                        lq.positions[j * cols + i],
                        lq.positions[j * cols + i + 1],
                        lq.positions[(j + 1) * cols + i],
                        lq.positions[(j + 1) * cols + i + 1],
                    ];
                    let cx = corners.iter().map(|p| p[0]).sum::<f32>() / 4.0;
                    let cy = corners.iter().map(|p| p[1]).sum::<f32>() / 4.0;
                    if !cx.is_finite() || !cy.is_finite() {
                        continue;
                    }
                    let key = ((cx / cell).floor() as i32, (cy / cell).floor() as i32);
                    known.insert(key);
                    if lq.wet[j * xt + i] {
                        wet.insert(key);
                    }
                }
            }
        }
        if !(cell > 0.0) || known.is_empty() {
            return None;
        }

        // Fractional wetness at a lattice CORNER: the share of the four cells meeting there that
        // carry liquid. Cells outside the batch are not counted at all rather than counted dry —
        // the same rule the cell set itself follows, and what keeps a tile's outer edge (where the
        // neighbouring tile's cells simply are not loaded) from reading as a shoreline five
        // hundred yards long.
        let corner = |i: i32, j: i32| -> Option<f32> {
            let (mut w, mut k) = (0u32, 0u32);
            for (dx, dy) in [(-1, -1), (0, -1), (-1, 0), (0, 0)] {
                let key = (i + dx, j + dy);
                if known.contains(&key) {
                    k += 1;
                    if wet.contains(&key) {
                        w += 1;
                    }
                }
            }
            (k > 0).then(|| w as f32 / k as f32)
        };

        // Marching squares over the lattice at the half-wet contour, cell by cell. Same shape as
        // [`Shoreline::build`]'s march: two crossings are one segment, four are a saddle.
        let mut raw: Vec<[f32; 4]> = Vec::new();
        for &(i, j) in &known {
            let q = [
                (i, j),
                (i + 1, j),
                (i + 1, j + 1),
                (i, j + 1),
            ]
            .map(|(ci, cj)| corner(ci, cj).map(|w| ([ci as f32 * cell, cj as f32 * cell], w - 0.5)));
            if q.iter().any(|c| c.is_none()) {
                continue;
            }
            let q: Vec<([f32; 2], f32)> = q.into_iter().flatten().collect();
            let mut cross: Vec<[f32; 2]> = Vec::with_capacity(4);
            for e in 0..4 {
                let (a, b) = (q[e], q[(e + 1) % 4]);
                if (a.1 > 0.0) != (b.1 > 0.0) {
                    let t = a.1 / (a.1 - b.1);
                    cross.push([
                        a.0[0] + (b.0[0] - a.0[0]) * t,
                        a.0[1] + (b.0[1] - a.0[1]) * t,
                    ]);
                }
            }
            if cross.len() == 2 {
                raw.push([cross[0][0], cross[0][1], cross[1][0], cross[1][1]]);
            } else if cross.len() == 4 {
                raw.push([cross[0][0], cross[0][1], cross[1][0], cross[1][1]]);
                raw.push([cross[2][0], cross[2][1], cross[3][0], cross[3][1]]);
            }
        }
        Some(Self {
            edge: Shoreline::from_segments(raw),
        })
    }

    /// Distance in yards from a point to the water's edge as the liquid GRID draws it — the traced
    /// contour, not the cells. [`FAR_FROM_SHORE`] when the batch had no boundary at all.
    fn distance_to_dry(&self, px: f32, py: f32) -> f32 {
        self.edge
            .as_ref()
            .map_or(FAR_FROM_SHORE, |e| e.distance(px, py))
    }
}

/// The waterline itself, traced as line segments, with a coarse spatial index.
///
/// Two earlier readings of where the water ends both failed, and for the same underlying reason:
/// they inferred the edge from the liquid data instead of finding it. The authored depth byte's
/// zero crossing works on a river bank, where the water really does taper out, and misses a coast
/// entirely. The liquid grid's own edge is right for a WMO pool and wrong for a coast too, because
/// the sea's surface carries straight on *underneath* the beach — the waterline you see is where the
/// terrain rises through that plane, an intersection which neither the depth byte nor the grid
/// records.
///
/// So it is traced directly: sample `surface height − ground height` over the tile, and follow its
/// zero contour with marching squares. That crossing is the waterline by construction, on a coast
/// and a river bank alike, and being a real curve it yields a real distance — where `f / |grad f|`
/// only extrapolated one, and inflated it over every flat spot in the sand.
struct Shoreline {
    /// Spatial index pitch, in yards.
    bucket: f32,
    /// Segments `[ax, ay, bx, by]`, filed under every bucket their extent touches.
    segs: HashMap<(i32, i32), Vec<[f32; 4]>>,
}

/// How finely the depth field is sampled when tracing, as a multiple of the liquid lattice. Four
/// puts the samples about a yard apart, and the traced polyline's segments with them.
///
/// This was two, on the argument that the terrain's own vertices are about two yards apart and
/// tracing finer would invent detail. The argument does not hold: what the player sees is not the
/// terrain's vertices but the SURFACE interpolated between them, and the waterline is where that
/// interpolated surface crosses the water — so sampling at a yard follows the drawn ground more
/// closely rather than inventing anything. At two yards the traced curve is a polyline with
/// two-yard segments, and a band a fifth of a yard wide drawn around it wears every one of those
/// facets as a visible kink.
const CONTOUR_SUB: usize = 4;

impl Shoreline {
    /// Trace every waterline in a batch of surfaces. `None` without terrain to measure against,
    /// which is every WMO pool (its coordinates are model-local and there is no ground under them).
    fn build(meshes: &[&LiquidMesh], chunks: &[ChunkMesh]) -> Option<Self> {
        if chunks.is_empty() {
            return None;
        }
        let mut raw: Vec<[f32; 4]> = Vec::new();
        for lq in meshes {
            let (cols, rows) = (lq.grid[0] as usize, lq.grid[1] as usize);
            if cols < 2 || rows < 2 || lq.positions.len() != cols * rows {
                continue;
            }
            // A representative surface height, standing in for the corners MCLQ leaves as a
            // sentinel under dry ground.
            let (mut sum, mut cnt) = (0.0_f32, 0_u32);
            for p in &lq.positions {
                if p[2].abs() < 1.0e8 {
                    sum += p[2];
                    cnt += 1;
                }
            }
            let Some(level) = (cnt > 0).then(|| sum / cnt as f32) else {
                continue;
            };
            // The chunk this surface sits on, found once. Every sample below lands inside it, so the
            // per-sample lookup is a single chunk's test rather than a walk over the tile's 256.
            let mid = surface_point(lq, level, (cols - 1) as f32 * 0.5, (rows - 1) as f32 * 0.5);
            let own = chunks.iter().find(|c| c.height_at(mid).is_some());
            let ground = |p: [f32; 3]| -> Option<f32> {
                own.and_then(|c| c.height_at(p))
                    .or_else(|| terrain_height_at(chunks, p))
            };

            let (sw, sh) = (CONTOUR_SUB * (cols - 1) + 1, CONTOUR_SUB * (rows - 1) + 1);
            let mut pts: Vec<Option<([f32; 2], f32)>> = Vec::with_capacity(sw * sh);
            for sj in 0..sh {
                for si in 0..sw {
                    let p = surface_point(
                        lq,
                        level,
                        si as f32 / CONTOUR_SUB as f32,
                        sj as f32 / CONTOUR_SUB as f32,
                    );
                    pts.push(ground(p).map(|gz| ([p[0], p[1]], p[2] - gz)));
                }
            }

            // Marching squares. A cell with two sign changes carries one piece of the waterline;
            // four is a saddle, where either pairing is as defensible as the other at this scale.
            for sj in 0..sh - 1 {
                for si in 0..sw - 1 {
                    let corner = [
                        pts[sj * sw + si],
                        pts[sj * sw + si + 1],
                        pts[(sj + 1) * sw + si + 1],
                        pts[(sj + 1) * sw + si],
                    ];
                    if corner.iter().any(|c| c.is_none()) {
                        continue;
                    }
                    let q: Vec<([f32; 2], f32)> = corner.into_iter().flatten().collect();
                    let mut cross: Vec<[f32; 2]> = Vec::with_capacity(4);
                    for e in 0..4 {
                        let (a, b) = (q[e], q[(e + 1) % 4]);
                        if (a.1 > 0.0) != (b.1 > 0.0) {
                            let t = a.1 / (a.1 - b.1);
                            cross.push([
                                a.0[0] + (b.0[0] - a.0[0]) * t,
                                a.0[1] + (b.0[1] - a.0[1]) * t,
                            ]);
                        }
                    }
                    if cross.len() == 2 {
                        raw.push([cross[0][0], cross[0][1], cross[1][0], cross[1][1]]);
                    } else if cross.len() == 4 {
                        raw.push([cross[0][0], cross[0][1], cross[1][0], cross[1][1]]);
                        raw.push([cross[2][0], cross[2][1], cross[3][0], cross[3][1]]);
                    }
                }
            }
        }
        Self::from_segments(raw)
    }

    /// File a set of traced segments into the spatial index. Shared with [`WetLattice`], which
    /// traces the grid's own boundary through the same marching squares and wants the same lookup.
    fn from_segments(raw: Vec<[f32; 4]>) -> Option<Self> {
        if raw.is_empty() {
            return None;
        }
        let bucket = 4.0_f32;
        let mut segs: HashMap<(i32, i32), Vec<[f32; 4]>> = HashMap::new();
        for s in raw {
            let (x0, x1) = (s[0].min(s[2]), s[0].max(s[2]));
            let (y0, y1) = (s[1].min(s[3]), s[1].max(s[3]));
            for by in (y0 / bucket).floor() as i32..=(y1 / bucket).floor() as i32 {
                for bx in (x0 / bucket).floor() as i32..=(x1 / bucket).floor() as i32 {
                    segs.entry((bx, by)).or_default().push(s);
                }
            }
        }
        Some(Self { bucket, segs })
    }

    /// The nearest POINT on the waterline to `(px, py)`, and how far it is — or `None` when no
    /// piece of waterline is within the searched ring of buckets, which is far further out than any
    /// foam reaches.
    ///
    /// The point matters as much as the distance: the mesh carries the OFFSET to it per vertex
    /// (`benilla_assets::materials::ATTRIBUTE_WOW_SHORE_OFFSET`) so the shader can take the length itself,
    /// after interpolation. See that attribute for why.
    fn nearest(&self, px: f32, py: f32) -> Option<([f32; 2], f32)> {
        let (bx, by) = (
            (px / self.bucket).floor() as i32,
            (py / self.bucket).floor() as i32,
        );
        let mut best: Option<([f32; 2], f32)> = None;
        for dy in -1..=1 {
            for dx in -1..=1 {
                let Some(list) = self.segs.get(&(bx + dx, by + dy)) else {
                    continue;
                };
                for s in list {
                    let (vx, vy) = (s[2] - s[0], s[3] - s[1]);
                    let len2 = vx * vx + vy * vy;
                    let t = if len2 > 1e-12 {
                        (((px - s[0]) * vx + (py - s[1]) * vy) / len2).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    let (cx, cy) = (s[0] + vx * t, s[1] + vy * t);
                    let d = ((px - cx).powi(2) + (py - cy).powi(2)).sqrt();
                    if best.is_none_or(|(_, bd)| d < bd) {
                        best = Some(([cx, cy], d));
                    }
                }
            }
        }
        best
    }

    /// Distance only — the far field, and every consumer that does not draw a band.
    fn distance(&self, px: f32, py: f32) -> f32 {
        self.nearest(px, py).map_or(FAR_FROM_SHORE, |(_, d)| d)
    }
}

/// A point on a liquid surface at grid parameter `(u, v)`, bilinear over the four corners around it.
/// Corners MCLQ left as a height sentinel under dry ground take `level` instead, so a cell that is
/// half under the bank still interpolates a sane surface.
fn surface_point(lq: &LiquidMesh, level: f32, u: f32, v: f32) -> [f32; 3] {
    let (cols, rows) = (lq.grid[0] as usize, lq.grid[1] as usize);
    let i = (u.max(0.0).floor() as usize).min(cols - 2);
    let j = (v.max(0.0).floor() as usize).min(rows - 2);
    let (fu, fv) = (u - i as f32, v - j as f32);
    let c = [
        lq.positions[j * cols + i],
        lq.positions[j * cols + i + 1],
        lq.positions[(j + 1) * cols + i],
        lq.positions[(j + 1) * cols + i + 1],
    ];
    let mix = |a: f32, b: f32, t: f32| a + (b - a) * t;
    let z = |k: usize| {
        if c[k][2].abs() < 1.0e8 {
            c[k][2]
        } else {
            level
        }
    };
    [
        mix(mix(c[0][0], c[1][0], fu), mix(c[2][0], c[3][0], fu), fv),
        mix(mix(c[0][1], c[1][1], fu), mix(c[2][1], c[3][1], fu), fv),
        mix(mix(z(0), z(1), fu), mix(z(2), z(3), fu), fv),
    ]
}

/// Stands in for "no waterline anywhere near this point" — most of any open water.
///
/// Deliberately only a few cells' worth, not a huge number. It is interpolated across the triangles
/// that touch the shore, so the value chosen sets how steeply the coordinate climbs there, and the
/// shader takes `fwidth` of it to anti-alias the foam edge. At 4096 that slope put `fwidth` in the
/// hundreds of yards, the anti-aliasing width swamped the band it was meant to soften, and the foam
/// smeared at half strength across the whole surface instead of drawing a line.
const FAR_FROM_SHORE: f32 = 24.0;

/// How finely a cell carrying the waterline is broken up for drawing.
///
/// This is what cures the "teeth". The foam is a band well under a yard across and the liquid
/// lattice is 4.17 yards, so a distance carried only at the lattice corners cannot describe it:
/// where the waterline passes between corners, none is near enough for any foam to appear at all,
/// and where one happens to fall close the band swells to fill the triangle. Gap, blob, gap, blob.
/// Splitting the cells the waterline actually crosses puts a corner every half yard along it, finer
/// than the line being drawn, and the band becomes even. Only those cells pay for it.
const SHORE_CELL_SUB: usize = 8;

/// How near the waterline a cell must come, in yards, to earn that subdivision — comfortably past
/// the foam's whole reach, so the band never ends up straddling a coarse cell.
const SHORE_CELL_REACH: f32 = 5.0;

/// Build the Bevy render mesh for one [`LiquidMesh`]: positions mapped WoW→Bevy (`lq.positions` are
/// raw WoW coords — absolute for MCLQ, WMO-model-local for WMO liquid), a flat up normal, the tiling
/// UVs, the per-vertex swatch `V` in UV1.x and the distance to the waterline in UV1.y. The caller
/// decides the surface's world placement via the spawned entity's `Transform` (`IDENTITY` for
/// absolute MCLQ water; the WMO placement transform for WMO liquid).
///
/// The triangles are emitted per wet cell rather than from `lq.indices`, because the cells along the
/// waterline are subdivided ([`SHORE_CELL_SUB`]) and the rest are not. Sharing no vertices between
/// cells costs a little memory and buys a uniform rule. It cannot crack: the water surface is
/// bilinear, so a split edge's new points lie exactly on the coarse edge its neighbour draws.
fn liquid_bevy_mesh(
    lq: &LiquidMesh,
    body_color: Option<[f32; 3]>,
    lattice: Option<&WetLattice>,
    shoreline: Option<&Shoreline>,
) -> Mesh {
    let (cols, rows) = (lq.grid[0] as usize, lq.grid[1] as usize);
    let (xt, yt) = (cols.saturating_sub(1), rows.saturating_sub(1));
    let (mut sum, mut cnt) = (0.0_f32, 0_u32);
    for p in &lq.positions {
        if p[2].abs() < 1.0e8 {
            sum += p[2];
            cnt += 1;
        }
    }
    let level = if cnt > 0 { sum / cnt as f32 } else { 0.0 };

    // Both sources answer for any point, not just a lattice corner, which is the whole reason the
    // subdivision below is worth anything.
    //
    // `shore_at` returns the distance AND, when the traced contour is the nearer of the two, the
    // offset to the point on it — see `benilla_assets::materials::ATTRIBUTE_WOW_SHORE_OFFSET`. The offset is
    // what the band is actually drawn from; the distance stays for the far field, where nothing is
    // drawn and a scalar is all anyone needs.
    let shore_at = |x: f32, y: f32| -> (f32, [f32; 2]) {
        let traced = shoreline.and_then(|s| s.nearest(x, y));
        let edge = lattice.map_or(FAR_FROM_SHORE, |l| l.distance_to_dry(x, y));
        match traced {
            Some((p, d)) if d <= edge => (d, [p[0] - x, p[1] - y]),
            // The grid's own edge won, or there is no traced contour here. It has no point to
            // offer, so the offset is left at the distance along +X: past the near field the
            // shader reads the scalar instead, and inside it the traced contour is what wins on
            // every surface that has one.
            _ => (traced.map_or(edge, |(_, d)| d.min(edge)), [edge, 0.0]),
        }
    };
    let dist = |x: f32, y: f32| -> f32 { shore_at(x, y).0 };

    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut uvs: Vec<[f32; 2]> = Vec::new();
    let mut uv1: Vec<[f32; 2]> = Vec::new();
    let mut shore_offsets: Vec<[f32; 2]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    let mix = |a: f32, b: f32, t: f32| a + (b - a) * t;

    for j in 0..yt {
        for i in 0..xt {
            if !lq.wet[j * xt + i] {
                continue;
            }
            let mid = surface_point(lq, level, i as f32 + 0.5, j as f32 + 0.5);
            let sub = if dist(mid[0], mid[1]) < SHORE_CELL_REACH {
                SHORE_CELL_SUB
            } else {
                1
            };
            let corner = [
                j * cols + i,
                j * cols + i + 1,
                (j + 1) * cols + i,
                (j + 1) * cols + i + 1,
            ];
            let base = positions.len() as u32;
            for sj in 0..=sub {
                for si in 0..=sub {
                    let (fu, fv) = (si as f32 / sub as f32, sj as f32 / sub as f32);
                    let p = surface_point(lq, level, i as f32 + fu, j as f32 + fv);
                    positions.push(wow_to_bevy(p).to_array());
                    let uv = |k: usize| lq.uvs[corner[k]];
                    uvs.push([
                        mix(mix(uv(0)[0], uv(1)[0], fu), mix(uv(2)[0], uv(3)[0], fu), fv),
                        mix(mix(uv(0)[1], uv(1)[1], fu), mix(uv(2)[1], uv(3)[1], fu), fv),
                    ]);
                    let d = |k: usize| lq.depths[corner[k]];
                    let (shore_d, shore_off) = shore_at(p[0], p[1]);
                    uv1.push([
                        mix(mix(d(0), d(1), fu), mix(d(2), d(3), fu), fv),
                        shore_d,
                    ]);
                    // WoW's +X/+Y is Bevy's −Z/−X (`wow_to_bevy`), and this is a DIRECTION, so it
                    // takes the same rotation without the translation: the shader adds it to a
                    // Bevy world XZ.
                    let off = wow_to_bevy([shore_off[0], shore_off[1], 0.0]);
                    shore_offsets.push([off.x, off.z]);
                }
            }
            let stride = (sub + 1) as u32;
            for sj in 0..sub as u32 {
                for si in 0..sub as u32 {
                    let tl = base + sj * stride + si;
                    let (tr, bl) = (tl + 1, tl + stride);
                    indices.extend_from_slice(&[tl, bl, bl + 1, tl, bl + 1, tr]);
                }
            }
        }
    }

    let n = positions.len();
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    // Flat surface: WoW up (0,0,1) → Bevy up (0,1,0). The shader lights against this (rotated into
    // world by the entity transform) + the sun.
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, vec![[0.0, 1.0, 0.0]; n]);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_1, uv1);
    // The offset to the nearest point on the waterline — what the foam band is drawn from. See
    // [`benilla_assets::materials::ATTRIBUTE_WOW_SHORE_OFFSET`].
    mesh.insert_attribute(
        benilla_assets::materials::ATTRIBUTE_WOW_SHORE_OFFSET,
        shore_offsets,
    );
    // An INTERIOR WMO pool's body colour is its own `MOMT.diffColor`, and it rides the vertex colour
    // because that is where the reference's interior water vertex carries it (a colour dword in its
    // 6-float record). Baking it here keeps ONE shared material per lane: a per-surface uniform would
    // have meant a material per pool. Absent on every other lane, which adds no attribute and takes
    // the shader's `#else` white.
    if let Some([red, green, blue]) = body_color {
        mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, vec![[red, green, blue, 1.0]; n]);
    }
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

/// Spawn a WMO group's embedded liquid surfaces (Stormwind's canals + fountains, the Ironforge lava,
/// dungeon pools) at the building's placement `transform`, on the shared per-kind liquid material —
/// the same animated water render as MCLQ, but its geometry is WMO-model-local (built by
/// `benilla_formats::wmo_group_liquid_mesh`) so the placement transform lifts it into the world.
///
/// No-op when the client has no data (`liquid_assets` absent) or a kind's frames didn't load. Each
/// WATER surface also carries a world-space [`WaterChunkInfo`] + [`FoamPatch`] (both built by baking the
/// placement transform into the raw liquid coords, [`world_wow`]) so the whole water-interaction stack
/// sees WMO liquid exactly like MCLQ: swimming ([`crate::player::swim`]), the underwater murk
/// ([`detect_submersion`]), the wading splash/footstep sounds, AND the `CWater0Ripple` wade wake /
/// standing ring ([`crate::water_fx`], which builds each foam decal from the wet-cell lattice). The
/// foam's world-axis texgen + per-triangle overlap consume the transformed cells fine, so a rotated
/// canal's ring is still correctly world-oriented. Spawned entities are pushed onto `entities` so they
/// despawn with the placement.
///
/// `interior` is the owning group's `MOGI & 0x48 == 0` class, and it selects the **fog block** the
/// surface draws under — the Great Forge's lava hazes with the forge's own fog, Stormwind's open
/// canals with the sky's (see [`LiquidKey`]).
///
/// `pool` is the surface's **scope** (see [`WmoPool`]): the room it belongs to, and that room's own
/// floor. A liquid footprint has no floor of its own, so an unscoped pool claims every position
/// under its XY forever — the Uldaman entrance read as submerged under a mushroom cave's water
/// 186 yd overhead (0696), and Undercity's upper slime submerged the rooms 115 yd below it (0701).
#[allow(clippy::too_many_arguments)] // one param per concern: assets, placement, fog block, scope
pub(crate) fn spawn_wmo_liquids<'a>(
    commands: &mut Commands,
    liquids: impl Iterator<Item = &'a LiquidMesh>,
    liquid_assets: Option<&LiquidAssets>,
    meshes: &mut Assets<Mesh>,
    transform: Transform,
    interior: bool,
    pool: WmoPool,
    // The owning root's MOMT `diffColor` table (`WmoModel::material_diff_color`). An INTERIOR
    // pool's whole body colour is its own entry; MOMT lives in the root, so the group file the
    // `LiquidMesh` came from could not resolve it and the lookup happens here.
    material_diff_color: &[[f32; 3]],
    // **The whole placement's** wet lattice, not this group's — see [`WetLattice`] and the note on
    // the caller. A pool needs a lattice at all because its builder fills a single flat depth for
    // the whole surface, which leaves the depth-field half of the shore distance with nothing to
    // find; it needs the WHOLE model's because a WMO's liquid is not one grid.
    lattice: Option<&WetLattice>,
    entities: &mut Vec<Entity>,
) {
    let Some(liquid) = liquid_assets else {
        return;
    };
    let path = LiquidPath::wmo(interior);
    let batch: Vec<&LiquidMesh> = liquids.collect();
    for lq in batch {
        // `interior` picks the fog block, not the look: an interior group's pool is drawn by the WMO
        // liquid pass, which re-submits the smoothed interior fog under the same `[0xca7f00]` gate as
        // the WMO geometry pass — so the pool fogs exactly like the walls around it (see [`LiquidKey`]).
        // This is the ONE path that can scroll, and the nibble — not the kind — is what decides it:
        // Blackrock's lava (6) creeps, the Ironforge Great Forge's (2) does not, and both are
        // `LiquidKind::Magma` drawing the same sheet ([`scrolls`]).
        let scroll = scrolls(lq.sound_nibble);
        // The interior arm's body colour, resolved through the pool's own MLIQ `materialId`. A
        // fullbright pool takes none (its sheet IS the body), and a missing/out-of-range index
        // simply yields no colour rather than a guessed one.
        let body_color = (path.takes_material_body_color() && !lq.kind.is_fullbright())
            .then(|| {
                lq.material_id
                    .and_then(|id| material_diff_color.get(usize::from(id)))
                    .copied()
            })
            .flatten();
        let Some(material) = liquid.material(lq.kind, path, scroll) else {
            continue; // this kind's frames failed to load (warned at setup)
        };
        if scroll {
            // Rare (only nibble 6/7 pools reach it) and the only outside view of a decision that is
            // otherwise invisible until you stand over the pool and watch it for ten seconds.
            debug!(
                "liquid: {:?} nibble {} takes the scroll lane",
                lq.kind, lq.sound_nibble
            );
        }
        let surface = commands
            .spawn((
                Mesh3d(meshes.add(liquid_bevy_mesh(lq, body_color, lattice, None))),
                MeshMaterial3d(material),
                transform,
                LiquidSurface,
                // The same layer as the ADT surfaces above, for the same reason.
                RenderLayers::layer(super::WATER_RENDER_LAYER),
                // The per-frame interior-fog lane rides `MeshTag` bit 30, written by the one
                // `Visibility` authority off this pool's own room (decision 1787; `liquid.wgsl`'s
                // `room_fog`). Spawned clear: a pool wears the room's fog only once the flood has
                // said the room is on the lane.
                bevy::mesh::MeshTag(0),
                // The ambient-loop source rides EVERY kind — the fullbright lava/slime hum too
                // (0506). It reads its geometry off the `WaterChunkInfo` inserted below.
                LiquidSoundSource {
                    nibble: lq.sound_nibble,
                },
            ))
            .id();
        // The swim/submersion grid rides EVERY kind, magma and slime included — that is what
        // makes Blackrock's lava and Undercity's slime swimmable instead of something you fall
        // through (decision 0634, bugs B24/B25). It used to be gated on `!is_fullbright()` because
        // `WaterChunkInfo` carried no kind, so tagging lava would have swum the player under a teal
        // *water* murk with white foam. The component carries [`LiquidKind`] now and the
        // water-flavoured consumers filter on it (`water_surface_at`, `detect_submersion`), so the
        // exclusion is no longer what keeps lava from looking like a lake.
        //
        // Lava/slime **damage** is still not modelled — a named gap, not a reason to keep the
        // geometry non-solid.
        commands.entity(surface).insert(wet_footprint(
            lq,
            &transform,
            LiquidSource::WmoGroup(pool),
        ));
        // Foam stays water-only: it is white surf, and there is no such thing on magma.
        if !lq.kind.is_fullbright() {
            commands.entity(surface).insert(FoamPatch);
        }
        entities.push(surface);
    }
}

/// Each kind's animated frame set: `(kind, XTextures subdir, file stem, frame count on disk)`.
/// Frames are `XTextures\<dir>\<stem>.<1..=count>.blp` (256² RGBA, RGB dark + alpha ripple).
const FRAME_SETS: &[(LiquidKind, &str, &str, u32)] = &[
    (LiquidKind::Still, "river", "lake_a", 30),
    (LiquidKind::Rapids, "river", "fast_a", 16),
    (LiquidKind::Ocean, "ocean", "ocean_h", 30),
    // The fullbright kinds: opaque + unlit + fogged, the animated texture IS the body colour, there
    // being no vertex colour or depth LUT to modulate it by (VERIFIED wow-re
    // `liquid-render-state-sided` §5). **Magma reaches here from BOTH liquid systems** — the WMO
    // MLIQ pools *and* the ADT MCLQ magma queue (B21: the 128 open-world lava chunks, Burning
    // Steppes/Searing Gorge/Un'Goro); only slime is WMO-only, the reference having no ADT queue for
    // it at all. See [`benilla_formats::LiquidKind`].
    (LiquidKind::Magma, "lava", "lava", 30),
    (LiquidKind::Slime, "slime", "slime", 30),
];

/// The ripple map's seed. Arbitrary, and fixed: the map is generated at every startup, so a seed
/// that moved would make two runs of the same scene two different pictures.
const RIPPLE_SEED: u32 = 0xB0A7_5EA5;

#[allow(clippy::too_many_arguments)]
pub(super) fn setup_liquid(
    mut commands: Commands,
    config: Option<Res<RenderConfig>>,
    world_assets: Option<ResMut<WorldAssets>>,
    style: Res<WaterStyle>,
    reflect: Res<super::reflect::WaterReflect>,
    reflect_buf: Res<super::reflect::WaterReflectBuffer>,
    scene_color: Res<super::scene_color::WaterSceneColor>,
    sim: Res<super::ripple_sim::RippleSim>,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<LiquidMaterial>>,
) {
    let (Some(_config), Some(mut world_assets)) = (config, world_assets) else {
        return; // no client data → no terrain, so no water either
    };
    // No light seed and no per-frame push: light, fog and both water swatches come off the shared
    // global-light buffer (`lighting::global_light`), which `build_light_data` has already packed by
    // the time anything draws — the same path terrain and the models take.
    let mut assets = LiquidAssets::default();
    // The stylised look's ripple map: ONE generated 256² texture, shared by every liquid material
    // and sampled only when `waterStyle` selects that look. Built here rather than lazily on the
    // first flip because the binding is unconditional — a material cannot hold an empty texture
    // slot — and because generating it costs a few milliseconds once, beside 150 BLP decodes.
    let ripples = images.add(ripple::ripple_map(RIPPLE_SEED));
    // The wave simulation's live field — one shared image too, and bound on every liquid material
    // for the same reason the map above is: the binding cannot be conditional, so the toggle stays
    // a uniform write rather than a material rebuild.
    let wake = sim.image.clone();
    for &(kind, dir, stem, count) in FRAME_SETS {
        let Some((frames, frame_count)) =
            load_frame_array(&mut world_assets, &mut images, kind, dir, stem, count)
        else {
            warn!("liquid: no frames for {stem} — {kind:?} water will not render");
            continue;
        };
        // Blend state is per KIND, and this is where it is decided (VERIFIED wow-re
        // `liquid-render-state-sided` — the device render-state *defaults* at `0x593bf0` are the baseline
        // a liquid batch draws under, because every setter in the reference is Push/Pop-scoped):
        //
        //   * water / ocean — the reference sets EGxBlend 2 (SRC_ALPHA / INV_SRC_ALPHA) and depth-write
        //     OFF, which is exactly Bevy's `AlphaMode::Blend` transparent pass. Unchanged. (Both are
        //     gated on the fancy-water cvar `[0xc9a324]`, which we do not model — a separate lane.)
        //   * magma / slime — blend state stays at the default **0**, i.e. `glDisable(GL_BLEND)`, with
        //     the baseline depth test *and* depth write both ON (ids `0x10`/`0x12` = 1). The ADT lava
        //     pass says so explicitly (`0x6855e2 (0x07, 0)`) and the WMO magma/slime arm never touches
        //     blend at all. So they are genuinely opaque and belong in the OPAQUE pass: riding the
        //     transparent pass with depth-write off is how a lava sheet fails to occlude what is behind
        //     it. Foam already carries `depth_bias: 1.0` for the coplanar tie, so it still wins.
        //
        // Two-sided is universal — all four liquid passes force GL_CULL_FACE off at entry against a
        // cull-ON baseline, and `glFrontFace` is never imported, so winding is moot (§6).
        let alpha_mode = if kind.is_fullbright() {
            AlphaMode::Opaque
        } else {
            AlphaMode::Blend
        };
        // One material per (RENDERER × SCROLL) combination the kind can actually take (see
        // [`LiquidKey`]), all sharing the one decoded frame array (`frames` is a handle; the clone is
        // a refcount, not a second 30 × 256² decode). Every other input is identical, so this is the
        // whole cost of giving each of the reference's three liquid renderers its own material — and
        // of letting Blackrock's lava creep while Ironforge's sits still.
        //
        // The scroll variant exists only for the fullbright kinds: [`scrolls`] can only ever answer
        // true for nibbles 6/7, which map to `Magma`/`Slime` and nothing else, so building one for
        // water would be a material nothing can ever look up.
        let scroll_lanes: &[bool] = if kind.is_fullbright() {
            &[false, true]
        } else {
            &[false]
        };
        for (path, &scroll) in [
            LiquidPath::Adt,
            LiquidPath::WmoExterior,
            LiquidPath::WmoInterior,
        ]
        .into_iter()
        .flat_map(|p| scroll_lanes.iter().map(move |s| (p, s)))
        {
            let material = materials.add(ExtendedMaterial {
                base: StandardMaterial {
                    // We do our own (WoW) lighting in the shader.
                    unlit: true,
                    alpha_mode,
                    cull_mode: None,
                    double_sided: true,
                    // The transparent water kinds take the fixed water-pass slot: below every
                    // unclassified world transparent, above the far-side effects (the reference's
                    // 0x483460 interleave — `sky_order::WATER_BIAS`, where the byte story lives).
                    // Opaque kinds (magma/slime) draw in the opaque pass and take no rung.
                    depth_bias: if kind.is_fullbright() {
                        0.0
                    } else {
                        crate::sky_order::WATER_BIAS
                    },
                    ..default()
                },
                extension: LiquidExt {
                    frames: frames.clone(),
                    ripples: ripples.clone(),
                    reflection: reflect.image.clone(),
                    scene_color: scene_color.image.clone(),
                    wake: wake.clone(),
                    reflect_buf: reflect_buf.0.clone(),
                    // x = fullbright (magma/slime: the animated texture is the opaque body, skipping
                    // the swatch and N·L — but NOT the fog, which every liquid kind takes); y = read
                    // the ocean swatch rows rather than the river/lake ones; z = fog with the WMO
                    // INTERIOR block; w = water's own sun-sheen exponent.
                    kind: Vec4::new(
                        if kind.is_fullbright() { 1.0 } else { 0.0 },
                        if kind == LiquidKind::Ocean { 1.0 } else { 0.0 },
                        if path.interior_fog() { 1.0 } else { 0.0 },
                        WATER_SHININESS,
                    ),
                    // x = which of the reference's three liquid renderers `liquid.wgsl` runs;
                    // y = the stylised-look lane, seeded from the resource here and rewritten in
                    // place by [`apply_water_style`] whenever the player changes it.
                    path: Vec4::new(path.shader_id(), style.shader_flag(), 0.0, 0.0),
                    // x = reserved (frame 0; the shader derives the live index from its own
                    // clock); y = frame count; z = the SCROLL FLAG (1 = this lane takes the
                    // stage-0 v-scroll — [`scrolls`]' nibble-6/7 WMO lane); w = the clock
                    // enable (0 on a deterministic run bakes the 0600 capture freeze — frame 0,
                    // scroll 0 — with no tick left to skip).
                    anim: Vec4::new(
                        0.0,
                        frame_count as f32,
                        if scroll { 1.0 } else { 0.0 },
                        if crate::dev_state::deterministic_run() {
                            0.0
                        } else {
                            1.0
                        },
                    ),
                    light_buf: world_assets.shared_light.clone(),
                },
            });
            assets
                .materials
                .insert(LiquidKey { kind, path, scroll }, LiquidEntry { material });
        }
    }
    // Frame SETS, not materials — a set backs every renderer/scroll variant of its kind.
    info!(
        "liquid: loaded {} water frame set(s)",
        FRAME_SETS
            .iter()
            .filter(|(k, ..)| assets.material(*k, LiquidPath::Adt, false).is_some())
            .count()
    );
    commands.insert_resource(assets);
}

/// Push the current [`WaterStyle`] onto every liquid material — the whole of what the Water Style
/// row does.
///
/// The look is one uniform lane (`path.y`), so a flip is a handful of uniform writes rather than a
/// material rebuild, a shader recompile or a reload of anything. `iter_mut` marks each material
/// Modified, which is what re-uploads the uniform; the system is change-gated on the resource so
/// that costs nothing until the player actually moves the dropdown.
///
/// Every liquid material takes the write, magma and slime included, and their `path.y` is then read
/// by nobody: the fullbright kinds return before the stylised branch, having no water to restyle.
/// Skipping them here would be a second place that has to agree with the shader about which kinds
/// are water.
pub(super) fn apply_water_style(
    style: Res<WaterStyle>,
    mut materials: ResMut<Assets<LiquidMaterial>>,
) {
    let flag = style.shader_flag();
    for (_, material) in materials.iter_mut() {
        material.extension.path.y = flag;
    }
}

/// Decode frames `1..=count` for a kind — each with its BLP **authored mip chain** — into one
/// repeating, mipmapped + anisotropic `texture_2d_array` (`assets::liquid_frame_array`; mips are what
/// stop the ripple aliasing into sparkle at distance). Stops at the first missing/non-square/
/// size-mismatched frame (the on-disk sets are contiguous 256² runs). Returns the image handle + the
/// number of frames actually loaded, or `None` if none decoded.
fn load_frame_array(
    world_assets: &mut WorldAssets,
    images: &mut Assets<Image>,
    kind: LiquidKind,
    dir: &str,
    stem: &str,
    count: u32,
) -> Option<(Handle<Image>, u32)> {
    let mut frames: Vec<BlpMipChain> = Vec::new();
    let mut size = 0u32;
    for i in 1..=count {
        let path = format!("XTextures\\{dir}\\{stem}.{i}.blp");
        let Ok(chain) = read_texture_mip_chain(&mut world_assets.chain.lock_recover(), &path)
        else {
            break;
        };
        if chain.width != chain.height {
            break; // water frames are square; bail rather than build a ragged array
        }
        if size == 0 {
            size = chain.width;
        } else if chain.width != size {
            break; // a frame at a different resolution can't share the array
        }
        frames.push(chain);
    }
    if frames.is_empty() {
        return None;
    }
    let loaded = frames.len() as u32;
    // Flatten the per-frame DC for the WATER kinds only: mipping turns the shipped frames' drifting
    // means into a whole-sheet brightness breath once per animation loop, which reads as a flicker on
    // distant water. Magma/slime draw the sheet AS their body colour, where that swing is the intended
    // pulse — so they keep theirs (`assets::flatten_frame_dc`).
    let normalize_dc = !kind.is_fullbright();
    Some((images.add(liquid_frame_array(frames, normalize_dc)), loaded))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The scroll gate, nibble by nibble — every trap in [`scrolls`] as an assertion.
    ///
    /// The failure this guards is not "lava doesn't move"; it is lava moving in the **wrong
    /// places**, which is far harder to see and impossible to unsee once shipped.
    #[test]
    fn only_nibbles_six_and_seven_scroll() {
        // The two that do: Blackrock's lava and Stratholme's slime.
        assert!(scrolls(6), "magma variant 6 takes the animated matrix");
        assert!(scrolls(7), "slime variant 7 takes the animated matrix");

        // The two that reach the SAME kernel and draw the SAME sheet, and still do not scroll —
        // the Ironforge Great Forge (2) and the Stratholme raid / Ahn'Qiraj slime (3).
        assert!(!scrolls(2), "the Great Forge's lava is static");
        assert!(!scrolls(3), "nibble-3 slime is static");

        // The near-miss that a `nibble & 0xc == 4` test would wrongly admit: 4 satisfies it, but it
        // is a *river* variant that never reaches the magma/slime kernel at all.
        assert!(!scrolls(4), "nibble 4 is lake_a, not a hazard liquid");

        // Everything else in the table, including the hole nibble.
        for n in [0u8, 1, 5, 8, 0xf] {
            assert!(!scrolls(n), "nibble {n} must not scroll");
        }
    }

    /// A scrolling material exists for the fullbright kinds and **only** for them — a scroll lane
    /// built for water would be a material no lookup can ever reach ([`setup_liquid`]).
    #[test]
    fn only_fullbright_kinds_can_take_a_scroll_lane() {
        for &(kind, ..) in FRAME_SETS {
            let wants_lane = kind.is_fullbright();
            assert_eq!(
                wants_lane,
                [2u8, 3, 6, 7]
                    .iter()
                    .any(|&n| scrolls(n) && LiquidKind::from_nibble(n) == Some(kind)),
                "{kind:?}: scroll lane and the nibbles that can ask for one must agree"
            );
        }
    }

    /// One 2×2-vertex MCLQ sheet at `at` — the smallest thing `spawn_liquids` will actually build.
    fn one_sheet(at: [f32; 3]) -> LiquidMesh {
        let [x, y, z] = at;
        LiquidMesh {
            grid: [2, 2],
            wet: vec![true],
            shared: vec![false],
            positions: vec![
                [x, y, z],
                [x + 8.0, y, z],
                [x, y + 8.0, z],
                [x + 8.0, y + 8.0, z],
            ],
            uvs: vec![[0.0, 0.0]; 4],
            depths: vec![1.0; 4],
            indices: vec![0, 1, 2, 1, 3, 2],
            sound_nibble: 0,
            material_id: None,
            kind: LiquidKind::Still,
        }
    }

    /// A [`LiquidAssets`] whose only entry is the one the ADT arm asks for — `spawn_liquids`
    /// `continue`s past a kind whose frames failed to load, so without this the spawn is a no-op
    /// and the test below would pass on an empty world.
    fn adt_assets() -> LiquidAssets {
        LiquidAssets {
            materials: HashMap::from([(
                LiquidKey {
                    kind: LiquidKind::Still,
                    path: LiquidPath::Adt,
                    scroll: false,
                },
                LiquidEntry {
                    material: Handle::default(),
                },
            )]),
        }
    }

    /// **The spawner's half of the indoor water cull** (decision 1652) — and the half no test in
    /// `exterior_cull` can cover, because that module's harness spawns its own entities and tags
    /// them itself. `apply_exterior_cull` decides nothing about a surface it never queries, and
    /// what puts a surface in its query is this tag, applied here. 0780 left exactly this note on
    /// the terrain cull ("nothing in `apply_exterior_cull` can fix that — it is decided by what the
    /// spawner tags") and terrain still has no test for it; liquid gets one.
    ///
    /// The three companion assertions are the filter `UnownedSceneFilter` actually applies: a
    /// surface that picked up `ModelPart`, `WmoGroupVis` or `WorldUnit` would be silently dropped
    /// from the walk and go back to drawing through the ceiling with this test still green.
    #[test]
    fn an_adt_liquid_surface_is_tagged_exterior_scene() {
        use bevy::ecs::system::RunSystemOnce;
        let mut app = App::new();
        app.add_plugins(bevy::asset::AssetPlugin::default())
            .init_asset::<Mesh>()
            .init_asset::<LiquidMaterial>();
        let sheets = [one_sheet([100.0, 100.0, 5.0])];
        let assets = adt_assets();
        let mut spawned = Vec::new();
        app.world_mut()
            .run_system_once(
                move |mut commands: Commands, mut meshes: ResMut<Assets<Mesh>>| {
                    let mut ents = Vec::new();
                    spawn_liquids(
                        &mut commands,
                        sheets.iter(),
                        &[],
                        Some(&assets),
                        &mut meshes,
                        &mut ents,
                    );
                    ents
                },
            )
            .map(|ents| spawned = ents)
            .expect("the spawn system ran");
        assert_eq!(spawned.len(), 1, "one sheet in, one surface out");
        let e = app.world().entity(spawned[0]);
        assert!(
            e.contains::<crate::exterior_cull::ExteriorScene>(),
            "an open-world liquid surface is exterior scene — untagged, the window cull never \
             queries it and the lake draws through a sealed ceiling (the director's report)"
        );
        assert!(
            e.contains::<LiquidSurface>(),
            "…and is still a liquid surface"
        );
        // The three exclusions in `UnownedSceneFilter`. An ADT surface must satisfy all of them or
        // the tag buys nothing.
        assert!(!e.contains::<crate::model_render::ModelPart>());
        assert!(!e.contains::<crate::wmo_portal::WmoGroupVis>());
        assert!(!e.contains::<crate::world_unit::WorldUnit>());
    }
}

/// Which fog block a WMO pool takes, against the **real client data**. Skips when the 1.12.1 client
/// isn't present (the repo never carries Blizzard data).
#[cfg(test)]
mod real_data {
    use super::LiquidKind;
    use benilla_formats::{parse_wmo_root, wmo_group_liquid_mesh};
    use std::collections::HashMap;

    /// **Which of the shipped hazard pools actually scroll** — [`super::scrolls`] against the real
    /// data, so the gate is pinned to content rather than to a reading of the binary.
    ///
    /// The whole reason this needs a test: "does lava scroll?" has no answer, only a per-pool one.
    /// The nibble is authored per WMO group, `LiquidKind` collapses it (2 and 6 are both `Magma`,
    /// drawing the same `lava.blp`), and the two behaviours sit **inside one building** — so a gate
    /// that quietly regressed to "all magma" or "no magma" would look plausible everywhere and be
    /// wrong nearly everywhere. These counts are the shipped 1.12.1 numbers, measured:
    ///
    /// * **Ironforge is the two-sided case, and it is why the director's report needed a location.**
    ///   Eleven magma groups: **9 at nibble 2 (static)** and **2 at nibble 6 (scrolling)**, plus 2
    ///   still-water groups. Lava that moves and lava that does not, in the same forge.
    /// * **Blackrock Mountain — the pin behind `blackrock_lava_is_below_the_feet_not_above_it`** —
    ///   is nibble 6 throughout, in the outer mountain and both instance WMOs. All of it creeps.
    /// * **Slime splits the same way**: Undercity's 38 canals are nibble 3 and stand still;
    ///   Stratholme's are nibble 7 and flow.
    #[test]
    fn only_the_nibble_six_and_seven_pools_scroll_in_the_shipped_data() {
        let data = benilla_formats::wow_data_or_skip!();
        let mut chain = benilla_formats::open_chain(&data).expect("open chain");
        // Every liquid-bearing group of a WMO, as `(nibble, kind)` → count.
        let mut census = |root_path: &str| -> HashMap<(u8, LiquidKind), usize> {
            let bytes = chain.read_file(root_path).expect("root readable");
            let root = parse_wmo_root(&bytes).expect("parse root");
            let stem = root_path
                .strip_suffix(".wmo")
                .unwrap_or(root_path)
                .to_string();
            let mut tally = HashMap::new();
            for gi in 0..root.group_count() as usize {
                let Ok(gb) = chain.read_file(&format!("{stem}_{gi:03}.wmo")) else {
                    continue;
                };
                if let Some(m) = wmo_group_liquid_mesh(&gb) {
                    *tally.entry((m.sound_nibble, m.kind)).or_default() += 1;
                }
            }
            tally
        };

        let ironforge = census("world\\wmo\\khazmodan\\cities\\ironforge\\ironforge.wmo");
        assert_eq!(
            ironforge,
            HashMap::from([
                ((2, LiquidKind::Magma), 9),
                ((4, LiquidKind::Still), 2),
                ((6, LiquidKind::Magma), 2),
            ]),
            "Ironforge's liquid census moved",
        );

        let blackrock = census("world\\wmo\\dungeon\\az_blackrock\\blackrock.wmo");
        assert_eq!(
            blackrock,
            HashMap::from([((6, LiquidKind::Magma), 2)]),
            "Blackrock Mountain's lava — the `.go xyz -7531.21 -1123.64 172.58` pin — is all nibble 6",
        );

        let undercity = census("world\\wmo\\lorderon\\undercity\\undercity.wmo");
        assert_eq!(
            undercity,
            HashMap::from([((3, LiquidKind::Slime), 38)]),
            "Undercity's slime is nibble 3 throughout",
        );

        let stratholme = census("world\\wmo\\dungeon\\ld_stratholme\\stratholme.wmo");
        assert_eq!(
            stratholme,
            HashMap::from([((7, LiquidKind::Slime), 3)]),
            "Stratholme's slime is nibble 7 throughout",
        );

        // And the gate's verdict over all of it — the same sheet, two behaviours.
        assert_eq!(
            ironforge
                .iter()
                .filter(|((n, _), _)| super::scrolls(*n))
                .map(|(_, c)| c)
                .sum::<usize>(),
            2,
            "exactly 2 of Ironforge's 11 magma pools scroll",
        );
        for (label, tally) in [("blackrock", &blackrock), ("stratholme", &stratholme)] {
            assert!(
                tally.keys().all(|(n, _)| super::scrolls(*n)),
                "{label}: every pool scrolls",
            );
        }
        assert!(
            undercity.keys().all(|(n, _)| !super::scrolls(*n)),
            "Undercity's slime stands still",
        );
    }

    /// **Which fog block each liquid-bearing WMO group takes**, against the real client files.
    ///
    /// This is decision 0691's follow-on lane. A WMO group's own pool is drawn by the pass that
    /// re-submits the smoothed INTERIOR fog block (`0x6b6323`-`0x6b6342`), gated on the same
    /// `[0xca7f00]` as the WMO *geometry* pass — so an indoor pool hazes with its room and an open-air
    /// one must not. `spawn_wmo_liquids` decides that from the group's `MOGI & 0x48 == 0`, and an
    /// inverted flag is invisible in any screenshot taken where the two blocks happen to agree (in dry
    /// Undercity both clamp to 1 within ~58 yd). So pin it on the two buildings that straddle the line,
    /// with the real numbers:
    ///
    /// * **Undercity** — 38 liquid groups, and all but ONE are interior. Its slime canals are the
    ///   director's original report; the lone exterior one (group 7) is the granularity working, not
    ///   noise, and is what makes this a real two-sided test rather than a constant.
    /// * **Stormwind** — 22 liquid groups (canals + fountains), every one EXTERIOR. Reading these as
    ///   interior would haze the open city's water with an indoor triple under an open sky.
    ///
    /// Skips when the 1.12.1 client isn't present (the repo never carries Blizzard data).
    #[test]
    fn a_wmo_pools_fog_block_follows_its_groups_interior_class() {
        let data = benilla_formats::wow_data_or_skip!();
        let mut chain = benilla_formats::open_chain(&data).expect("open chain");
        // Which groups of a WMO carry liquid, and whether each is an interior group.
        let liquid_groups = |chain: &mut benilla_formats::Chain, root_path: &str| {
            let bytes = chain.read_file(root_path).expect("root readable");
            let root = parse_wmo_root(&bytes).expect("parse root");
            let stem = root_path
                .strip_suffix(".wmo")
                .unwrap_or(root_path)
                .to_string();
            (0..root.group_count() as usize)
                .filter_map(|gi| {
                    let gb = chain.read_file(&format!("{stem}_{gi:03}.wmo")).ok()?;
                    wmo_group_liquid_mesh(&gb)?;
                    Some((gi, root.group_infos().get(gi).is_some_and(|g| g.interior)))
                })
                .collect::<Vec<_>>()
        };

        let uc = liquid_groups(&mut chain, "world\\wmo\\lorderon\\undercity\\undercity.wmo");
        let exterior: Vec<usize> = uc.iter().filter(|(_, i)| !i).map(|(g, _)| *g).collect();
        assert_eq!(uc.len(), 38, "Undercity's liquid group count moved: {uc:?}");
        assert_eq!(
            exterior,
            vec![7],
            "exactly one Undercity liquid group is exterior (the flag is per group, not per building)",
        );

        let sw = liquid_groups(
            &mut chain,
            "world\\wmo\\azeroth\\buildings\\stormwind\\stormwind.wmo",
        );
        assert_eq!(sw.len(), 22, "Stormwind's liquid group count moved: {sw:?}");
        assert!(
            sw.iter().all(|(_, interior)| !interior),
            "Stormwind's canals and fountains are open to the sky — none takes the interior fog: {sw:?}",
        );
    }
}
