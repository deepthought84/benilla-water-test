//! The animated liquid surfaces: one shared material per (kind, [`LiquidPath`], scroll) over a
//! `texture_2d_array` of the kind's frames, the two spawn paths (an ADT chunk's MCLQ, a WMO
//! group's MLIQ) and the flat mesh. The 24 fps frame flip and the lava scroll run in the shader
//! off the wall clock, so no CPU mutates a material; a deterministic run bakes the clock off
//! (frame 0, scroll 0). Day and night follow server time.

use std::collections::{HashMap, HashSet};

use bevy::asset::RenderAssetUsages;
use bevy::camera::visibility::RenderLayers;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::pbr::ExtendedMaterial;
use bevy::prelude::*;

use super::query::{wet_footprint, FoamPatch, LiquidSource, WmoPool};
use super::{ripple, WaterStyle};
use crate::collision::liquid_layers;
use crate::lighting::WATER_SHININESS;
use avian3d::prelude::{Collider, RigidBody};
use benilla_assets::coords::wow_to_bevy;
use benilla_assets::materials::{LiquidExt, LiquidMaterial};
use benilla_assets::LockRecover;
use benilla_assets::{liquid_frame_array, RenderConfig, WorldAssets};
use benilla_formats::{
    read_texture_mip_chain, terrain_height_at, BlpMipChain, ChunkMesh, LiquidKind, LiquidMesh,
    PlanarMap, NO_SPOT,
};

/// The shared liquid materials by [`LiquidKey`]; absent without client data.
#[derive(Resource, Default)]
pub(crate) struct LiquidAssets {
    materials: HashMap<LiquidKey, LiquidEntry>,
}

/// Which of the reference's three liquid renderers draws a surface: `0x6b62e0` sends the type
/// nibble to a category, and category 0 (nibbles 0/4/8) splits on the group's `MOGP.flags & 0x48`.
/// Magma and slime share `0x6b68f0` (WMO) / `0x68dca0` (ADT) whatever the flags, so for them the
/// path only picks the fog block. Fog is per pass: the WMO liquid pass re-submits the smoothed
/// interior fog block (`0x6b6323`–`0x6b6342`) under the same `[0xca7f00]` gate as the WMO geometry,
/// so a pool fogs like its walls, while ADT liquid submits none and draws under the scene block.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum LiquidPath {
    /// An ADT chunk's MCLQ (`0x6851b0` river, `0x685010` ocean): two stages, the depth swatch at
    /// `tc0 = (0.5, LUT[depthByte])` and the sheet at `tc1 = (col·¼, row·¼)`; no vertex colour.
    Adt,
    /// A WMO group's MLIQ, exterior or exterior-lit (`MOGP.flags & 0x48 != 0`), `0x6b6630`: one
    /// stage, a 9-float vertex with an up normal and a colour dword, `MapObjExtWater0.bls` bound at
    /// `0x6b6654`.
    WmoExterior,
    /// A WMO group's MLIQ, a true interior (`MOGP.flags & 0x48 == 0`), `0x6b6420`: one stage, a
    /// 6-float vertex with no normal, lighting off, no pixel program.
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

    /// Does this arm's pool take its body colour from its own MOMT `diffColor`? Only the interior
    /// one: the exterior reads a live `Light.dbc` band, and ADT liquid has no MOMT.
    fn takes_material_body_color(self) -> bool {
        matches!(self, Self::WmoInterior)
    }
}

/// Which shared material a surface takes. The scroll is per nibble and per path, which
/// [`LiquidKind`] collapses (2 and 6 are both `Magma`), so it keys the material too.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct LiquidKey {
    kind: LiquidKind,
    path: LiquidPath,
    /// This surface takes the animated stage-0 texture matrix ([`scrolls`]).
    scroll: bool,
}

/// Does a WMO MLIQ surface of this type nibble take the reference's animated stage-0 texture
/// matrix, the lava/slime scroll? Only 6 and 7: the WMO magma/slime kernel `0x6b68f0` gates on
/// `and esi,0xc; cmp esi,4` over the `0x6ba970` nibble and builds an identity but for element 13,
/// the v-translate. Nibbles 2 and 3 reach the same kernel and stay still (the Great Forge is 2,
/// Blackrock 6). ADT liquid never scrolls, though open-world lava is nibble 6: its kernel
/// `0x68dca0` zeroes element 13 at `0x68dda1`, so the caller gates the path. The phase runs off
/// the reference's uptime clock; only its rate and period are reproducible.
fn scrolls(nibble: u8) -> bool {
    matches!(nibble, 6 | 7)
}

struct LiquidEntry {
    material: Handle<LiquidMaterial>,
}

impl LiquidAssets {
    /// The shared material for a kind, renderer and scroll, if one was built and its frames loaded.
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

/// The map's water classification on an ADT surface: where the probe placement finds the fixed
/// probe spots.
#[derive(Component)]
pub(crate) struct WaterMapRef(pub(crate) std::sync::Arc<PlanarMap>);

/// Marks a spawned liquid surface: one per MCNK liquid layer or WMO group pool.
#[derive(Component)]
pub(crate) struct LiquidSurface;

/// `WOW_NO_LIQUID`: hide every liquid surface and only the surface, since the swim grid, foam and
/// sound ride sibling components. An override run after both per-frame `Visibility` owners (the
/// exterior cull for ADT surfaces, `apply_model_visibility` for WMO pools), so it wins the frame.
pub(super) fn hide_liquid_surfaces(mut surfaces: Query<&mut Visibility, With<LiquidSurface>>) {
    for mut vis in &mut surfaces {
        if *vis != Visibility::Hidden {
            *vis = Visibility::Hidden;
        }
    }
}

/// The ambient loop's sound-class nibble, resolved through `SoundWaterType.dbc` (`0x54e0a0`), on
/// every liquid surface; the driver reads the geometry off the same entity's `WaterChunkInfo`.
#[derive(Component)]
pub(crate) struct LiquidSoundSource {
    /// The surface's sound-class nibble (`class = n & 3`, `FluidSpeed = n & 0xc`).
    pub(crate) nibble: u8,
}

/// Spawn an ADT tile's liquid surfaces into `entities`, to despawn with the tile.
pub(crate) fn spawn_liquids<'a>(
    commands: &mut Commands,
    liquids: impl Iterator<Item = &'a LiquidMesh>,
    // The owning tile's terrain, for the shore foam: the waterline on a coast is where the ground
    // rises through the water plane, and nothing in the liquid data alone records it
    // ([`shore_distances`]).
    chunks: &[ChunkMesh],
    // The map's water classification: which water the mirrors serve, where the probes stand.
    planar: Option<&std::sync::Arc<PlanarMap>>,
    liquid_assets: Option<&LiquidAssets>,
    meshes: &mut Assets<Mesh>,
    entities: &mut Vec<Entity>,
) {
    let Some(liquid) = liquid_assets else {
        return;
    };
    // One lattice for the whole tile: a chunk cannot tell where its water ends from itself alone.
    let batch: Vec<&LiquidMesh> = liquids.collect();
    let lattice = WetLattice::build(batch.iter().copied());
    let shoreline = Shoreline::build(&batch, chunks);
    for lq in batch {
        // The scene fog (the ADT passes submit none and draw under the scene submit `0x66ff20`),
        // and never a scroll, though ADT magma is nibble 6: its kernel zeroes the v-translate.
        let Some(material) = liquid.material(lq.kind, LiquidPath::Adt, false) else {
            continue; // this kind's frames failed to load (warned at setup)
        };
        let info = wet_footprint(lq, &Transform::IDENTITY, LiquidSource::AdtChunk)
            .with_planes(planar.map(|p| p.planes(lq)));
        let foam = !lq.kind.is_fullbright(); // white surf is a water thing
        entities.push(
            commands
                .spawn((
                    Mesh3d(meshes.add(liquid_bevy_mesh(
                        lq,
                        None,
                        lattice.as_ref(),
                        planar.map(|p| &**p),
                        true,
                        shoreline.as_ref(),
                    ))),
                    MeshMaterial3d(material),
                    Transform::IDENTITY,
                    LiquidSurface,
                    // Liquid rides its own layer, so the stylised look's mirrors can leave it out.
                    RenderLayers::layer(super::WATER_RENDER_LAYER),
                    // Exterior scene: the ADT liquid producer `0x683ab0` is called only from the
                    // per-window populate `0x682fa0`, like ADT terrain (`0x683bf0`) and doodads
                    // (`0x683700`), one entity per MCNK layer. The exterior cull is its only
                    // `Visibility` writer.
                    crate::exterior_cull::ExteriorScene,
                    info,
                    LiquidSoundSource {
                        nibble: lq.sound_nibble,
                    },
                ))
                .id(),
        );
        if foam {
            commands
                .entity(*entities.last().expect("just pushed"))
                .insert(FoamPatch);
        }
        if let Some(p) = planar {
            commands
                .entity(*entities.last().expect("just pushed"))
                .insert(WaterMapRef(p.clone()));
        }
        // The waterline for the camera sweep under `cameraWaterCollision`; nothing else queries it.
        if let Some(collider) = liquid_collider(lq) {
            commands
                .entity(*entities.last().expect("just pushed"))
                .insert((collider, RigidBody::Static, liquid_layers()));
        }
    }
}

/// The collision trimesh for one [`LiquidMesh`]: the render mesh's own wet-cell triangles, `None`
/// when every cell is dry. Built inline, as a layer is at most 128 triangles.
fn liquid_collider(lq: &LiquidMesh) -> Option<Collider> {
    let tris: Vec<[u32; 3]> = lq
        .indices
        .as_chunks::<3>()
        .0
        .iter()
        .map(|c| [c[0], c[1], c[2]])
        .collect();
    if tris.is_empty() {
        return None;
    }
    let verts: Vec<Vec3> = lq.positions.iter().map(|p| wow_to_bevy(*p)).collect();
    Some(Collider::trimesh(verts, tris))
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
                if cell.is_nan() || cell <= 1e-3 {
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
        if cell.is_nan() || cell <= 0.0 || known.is_empty() {
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
            let q = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)].map(|(ci, cj)| {
                corner(ci, cj).map(|w| ([ci as f32 * cell, cj as f32 * cell], w - 0.5))
            });
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

/// Each lattice corner's normal, WoW axes: the cross of the central differences to its neighbours
/// along both grid axes (one-sided at the edge), turned up. Corners MCLQ left as a height sentinel
/// take `level`, as [`surface_point`] does.
fn corner_normals(lq: &LiquidMesh, level: f32) -> Vec<[f32; 3]> {
    let (cols, rows) = (lq.grid[0] as usize, lq.grid[1] as usize);
    let at = |i: usize, j: usize| {
        let p = lq.positions[j * cols + i];
        Vec3::new(p[0], p[1], if p[2].abs() < 1.0e8 { p[2] } else { level })
    };
    let mut out = Vec::with_capacity(cols * rows);
    for j in 0..rows {
        for i in 0..cols {
            let du = at((i + 1).min(cols - 1), j) - at(i.saturating_sub(1), j);
            let dv = at(i, (j + 1).min(rows - 1)) - at(i, j.saturating_sub(1));
            let n = du.cross(dv).normalize_or(Vec3::Z);
            out.push((if n.z < 0.0 { -n } else { n }).to_array());
        }
    }
    out
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
    planar: Option<&PlanarMap>,
    // Whether the spot ids are this batch's map's (ADT) and not a WMO placement's own.
    bind_spots: bool,
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
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut planar_w: Vec<[f32; 4]> = Vec::new();
    let cell_water = planar.map(|p| p.cells(lq));
    let mut indices: Vec<u32> = Vec::new();
    let corner_n = corner_normals(lq, level);
    let corner_w: Vec<f32> = lq
        .positions
        .iter()
        .map(|p| planar.map_or(1.0, |m| m.corner(p[0], p[1])))
        .collect();
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
                    uv1.push([mix(mix(d(0), d(1), fu), mix(d(2), d(3), fu), fv), shore_d]);
                    // WoW's +X/+Y is Bevy's −Z/−X (`wow_to_bevy`), and this is a DIRECTION, so it
                    // takes the same rotation without the translation: the shader adds it to a
                    // Bevy world XZ.
                    let off = wow_to_bevy([shore_off[0], shore_off[1], 0.0]);
                    shore_offsets.push([off.x, off.z]);
                    // The smooth normal: the four corners' normals, blended like the position.
                    let cn = |k: usize| corner_n[corner[k]];
                    let blend = Vec3::from(cn(0))
                        .lerp(Vec3::from(cn(1)), fu)
                        .lerp(Vec3::from(cn(2)).lerp(Vec3::from(cn(3)), fu), fv);
                    normals.push(wow_to_bevy(blend.normalize_or(Vec3::Z).to_array()).to_array());
                    let cw = |k: usize| corner_w[corner[k]];
                    let here = cell_water
                        .as_ref()
                        .and_then(|c| c.get(j * xt + i))
                        .filter(|_| bind_spots);
                    let (lo, hi, share) =
                        here.map_or((NO_SPOT, NO_SPOT, 0.0), |c| (c.spots[0], c.spots[1], c.mix));
                    planar_w.push([
                        mix(mix(cw(0), cw(1), fu), mix(cw(2), cw(3), fu), fv),
                        f32::from(lo),
                        f32::from(hi),
                        share,
                    ]);
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
    // WoW up (0, 0, 1) is Bevy up (0, 1, 0).
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, vec![[0.0, 1.0, 0.0]; n]);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_1, uv1);
    // The offset to the nearest point on the waterline, which the foam band is drawn from.
    mesh.insert_attribute(
        benilla_assets::materials::ATTRIBUTE_WOW_SHORE_OFFSET,
        shore_offsets,
    );
    // The heightfield's smooth normal, for the stylised lane's reflections.
    mesh.insert_attribute(
        benilla_assets::materials::ATTRIBUTE_WOW_SURFACE_NORMAL,
        normals,
    );
    // The mirrors' weight, 0 on probe water, and the probe spots read; see `ATTRIBUTE_WOW_PLANAR`.
    mesh.insert_attribute(benilla_assets::materials::ATTRIBUTE_WOW_PLANAR, planar_w);
    // An interior pool's `MOMT.diffColor` rides the vertex colour, where the reference's interior
    // vertex carries it, keeping one material per lane; other lanes take the shader's white.
    if let Some([red, green, blue]) = body_color {
        mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, vec![[red, green, blue, 1.0]; n]);
    }
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

/// Spawn a WMO group's liquid surfaces at the building's placement `transform`, pushed onto
/// `entities` to despawn with the placement. `interior` is the group's `MOGI & 0x48 == 0` class,
/// which picks the renderer and the fog block; `pool` scopes each surface to its room and floor.
pub(crate) fn spawn_wmo_liquids<'a>(
    commands: &mut Commands,
    liquids: impl Iterator<Item = &'a LiquidMesh>,
    liquid_assets: Option<&LiquidAssets>,
    meshes: &mut Assets<Mesh>,
    transform: Transform,
    interior: bool,
    pool: WmoPool,
    // The owning root's MOMT `diffColor` table: MOMT lives in the root, not the group file.
    material_diff_color: &[[f32; 3]],
    // **The whole placement's** wet lattice, not this group's — see [`WetLattice`] and the note on
    // the caller. A pool needs a lattice at all because its builder fills a single flat depth for
    // the whole surface, which leaves the depth-field half of the shore distance with nothing to
    // find; it needs the WHOLE model's because a WMO's liquid is not one grid.
    lattice: Option<&WetLattice>,
    // The whole placement's mirror classes, for the same reason as the lattice.
    planar: Option<&PlanarMap>,
    entities: &mut Vec<Entity>,
) {
    let Some(liquid) = liquid_assets else {
        return;
    };
    let path = LiquidPath::wmo(interior);
    let batch: Vec<&LiquidMesh> = liquids.collect();
    for lq in batch {
        // The one path that can scroll, decided by the nibble, not the kind (`scrolls`).
        let scroll = scrolls(lq.sound_nibble);
        // The interior arm's body colour via the pool's MLIQ `materialId`; none when fullbright.
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
            debug!(
                "liquid: {:?} nibble {} takes the scroll lane",
                lq.kind, lq.sound_nibble
            );
        }
        let surface = commands
            .spawn((
                Mesh3d(meshes.add(liquid_bevy_mesh(
                    lq, body_color, lattice, planar, false, None,
                ))),
                MeshMaterial3d(material),
                transform,
                LiquidSurface,
                RenderLayers::layer(super::WATER_RENDER_LAYER),
                // Bit 30 is the interior-fog lane (`liquid.wgsl`'s `room_fog`), written each frame
                // by the `Visibility` authority off the pool's room; spawned clear.
                bevy::mesh::MeshTag(0),
                // Every kind, the lava and slime hum too.
                LiquidSoundSource {
                    nibble: lq.sound_nibble,
                },
            ))
            .id();
        // Every kind carries the swim grid, so lava and slime swim; their damage is not modelled.
        commands.entity(surface).insert(
            wet_footprint(lq, &transform, LiquidSource::WmoGroup(pool))
                .with_planes(planar.map(|p| p.planes(lq))),
        );
        if !lq.kind.is_fullbright() {
            commands.entity(surface).insert(FoamPatch);
        }
        // The camera's waterline, model-local under the entity's placement `transform`.
        if let Some(collider) = liquid_collider(lq) {
            commands
                .entity(surface)
                .insert((collider, RigidBody::Static, liquid_layers()));
        }
        entities.push(surface);
    }
}

/// Each kind's frames, `XTextures\<dir>\<stem>.<1..=count>.blp`, as `(kind, dir, stem, count)`.
const FRAME_SETS: &[(LiquidKind, &str, &str, u32)] = &[
    (LiquidKind::Still, "river", "lake_a", 30),
    (LiquidKind::Rapids, "river", "fast_a", 16),
    (LiquidKind::Ocean, "ocean", "ocean_h", 30),
    // Fullbright: the sheet is the opaque, unlit, fogged body (`0x6b68f0`). Magma comes from both
    // the WMO pools and the ADT magma queue; the reference has no ADT queue for slime.
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
    probe: Res<super::probe::WaterProbe>,
    hiz: Res<super::hiz::WaterHiz>,
    ssr_pip: Res<super::ssr_pip::WaterSsrPip>,
    sim: Res<super::ripple_sim::RippleSim>,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<LiquidMaterial>>,
) {
    let (Some(_config), Some(mut world_assets)) = (config, world_assets) else {
        return; // no client data → no terrain, so no water either
    };
    // Light, fog and the water swatches come off the shared global-light buffer, as for terrain.
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
        // Blend per kind, over the device defaults (`0x593bf0`) every reference setter pushes and
        // pops against. Water and ocean: EGxBlend 2 (`SRC_ALPHA / INV_SRC_ALPHA`), depth write off
        // (both under the fancy-water CVar `[0xc9a324]`, which is not modelled). Magma and slime:
        // blend stays 0 (disabled) with depth test and write on (ids `0x10`/`0x12`), as the ADT
        // lava pass sets (`0x6855e2`) and the WMO arm leaves it, so they are opaque. All four
        // liquid passes turn culling off at entry (`0x59d7d8`); `glFrontFace` is never imported.
        let alpha_mode = if kind.is_fullbright() {
            AlphaMode::Opaque
        } else {
            AlphaMode::Blend
        };
        // One material per renderer and scroll the kind can take, all on one frame array; only
        // magma and slime (nibbles 6 and 7) can scroll.
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
                    // The shader does the WoW lighting.
                    unlit: true,
                    alpha_mode,
                    cull_mode: None,
                    double_sided: true,
                    // Water takes the water-pass slot of the reference's `0x483460` interleave
                    // (`sky_order::WATER_BIAS`); the opaque kinds take no rung.
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
                    reflection2: reflect.image2.clone(),
                    scene_color: scene_color.image.clone(),
                    probe: probe.cube.clone(),
                    hiz: hiz.image.clone(),
                    hiz_far: hiz.far.clone(),
                    ssr_pip: ssr_pip.image.clone(),
                    wake: wake.clone(),
                    reflect_buf: reflect_buf.0.clone(),
                    // x = fullbright (the sheet as body, still fogged); y = the ocean swatch;
                    // z = the interior fog block; w = water's sun-sheen exponent.
                    kind: Vec4::new(
                        if kind.is_fullbright() { 1.0 } else { 0.0 },
                        if kind == LiquidKind::Ocean { 1.0 } else { 0.0 },
                        if path.interior_fog() { 1.0 } else { 0.0 },
                        WATER_SHININESS,
                    ),
                    // Which of the reference's three liquid renderers `liquid.wgsl` runs; the
                    // stylised lane (rewritten by [`apply_water_style`]); magma rather than slime.
                    path: Vec4::new(
                        path.shader_id(),
                        style.shader_flag(),
                        if kind == LiquidKind::Magma { 1.0 } else { 0.0 },
                        0.0,
                    ),
                    // x reserved; y = frame count; z = scroll; w = clock enable (0 on a
                    // deterministic run: frame 0, scroll 0).
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
    // Frame sets, not materials.
    info!(
        "liquid: loaded {} water frame set(s)",
        FRAME_SETS
            .iter()
            .filter(|(k, ..)| assets.material(*k, LiquidPath::Adt, false).is_some())
            .count()
    );
    commands.insert_resource(assets);
}

/// Push the current [`WaterStyle`] onto every liquid material's `path.y`: a uniform write, not a
/// rebuild, change-gated on the resource.
pub(super) fn apply_water_style(
    style: Res<WaterStyle>,
    mut materials: ResMut<Assets<LiquidMaterial>>,
) {
    let flag = style.shader_flag();
    for (_, material) in materials.iter_mut() {
        material.extension.path.y = flag;
    }
}

/// Decode frames `1..=count` with their authored mips into one repeating, anisotropic array,
/// stopping at the first missing, non-square or mis-sized frame; returns it and the count loaded.
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
    // Deviation: water frames have their per-frame mean flattened (`flatten_frame_dc`), because
    // mipping turns the shipped frames' drifting means into a flicker on distant water; magma and
    // slime keep theirs, the body's intended pulse.
    let normalize_dc = !kind.is_fullbright();
    Some((images.add(liquid_frame_array(frames, normalize_dc)), loaded))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_nibbles_six_and_seven_scroll() {
        // Blackrock's lava and Stratholme's slime.
        assert!(scrolls(6), "magma variant 6 takes the animated matrix");
        assert!(scrolls(7), "slime variant 7 takes the animated matrix");

        // Same kernel, same sheet, still static: the Great Forge (2) and Undercity's slime (3).
        assert!(!scrolls(2), "the Great Forge's lava is static");
        assert!(!scrolls(3), "nibble-3 slime is static");

        // 4 passes a bare `nibble & 0xc == 4` test but is a river variant, never the kernel's.
        assert!(!scrolls(4), "nibble 4 is lake_a, not a hazard liquid");

        // Everything else in the table, including the hole nibble.
        for n in [0u8, 1, 5, 8, 0xf] {
            assert!(!scrolls(n), "nibble {n} must not scroll");
        }
    }

    /// Only the fullbright kinds have nibbles that scroll, so only they get a scroll lane.
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

    /// The smallest MCLQ sheet `spawn_liquids` builds: 2×2 vertices at `at`.
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

    /// Only the ADT still-water material, without which `spawn_liquids` spawns nothing.
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

    /// An ADT surface carries `ExteriorScene` and none of the components `UnownedSceneFilter`
    /// excludes, so the exterior cull reaches it.
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
                        None,
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
             queries it and the lake draws through a sealed ceiling"
        );
        assert!(
            e.contains::<LiquidSurface>(),
            "…and is still a liquid surface"
        );
        // The three exclusions in `UnownedSceneFilter`.
        assert!(!e.contains::<crate::model_render::ModelPart>());
        assert!(!e.contains::<crate::wmo_portal::WmoGroupVis>());
        assert!(!e.contains::<crate::world_unit::WorldUnit>());
    }

    /// The mesh is built per wet cell, so every attribute has one entry per emitted vertex and
    /// none is the lattice's own count; bevy truncates a mesh whose attributes disagree.
    #[test]
    fn every_liquid_mesh_attribute_has_one_entry_per_vertex() {
        let mut lq = one_sheet([0.0, 0.0, 5.0]);
        lq.grid = [3, 3];
        lq.wet = vec![true; 4];
        lq.shared = vec![false; 4];
        lq.positions = (0..9)
            .map(|k| [(k % 3) as f32 * 4.0, (k / 3) as f32 * 4.0, 5.0])
            .collect();
        lq.uvs = vec![[0.0, 0.0]; 9];
        lq.depths = vec![1.0; 9];
        let mesh = liquid_bevy_mesh(&lq, Some([1.0, 1.0, 1.0]), None, None, false, None);
        let n = mesh.count_vertices();
        assert_eq!(n, 16, "four wet cells, four corners each");
        for (attr, values) in mesh.attributes() {
            assert_eq!(
                values.len(),
                n,
                "{} has {} entries",
                attr.name,
                values.len()
            );
        }
    }
}

/// The scroll and fog-block rules against the real client files.
#[cfg(test)]
mod real_data {
    use super::LiquidKind;
    use benilla_formats::{parse_wmo_root, wmo_group_liquid_mesh};
    use std::collections::HashMap;

    /// The scroll gate over the shipped pools, where both behaviours share buildings: Ironforge has
    /// 9 static nibble-2 and 2 scrolling nibble-6 magma groups, Blackrock is nibble 6 throughout,
    /// Undercity's 38 slime canals are nibble 3 and still, and Stratholme's are nibble 7 and flow.
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

        // The gate's verdict over all of it.
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

    /// A WMO pool's fog block follows its group's `MOGI & 0x48 == 0` class: Undercity's 38 liquid
    /// groups are interior but for group 7, and Stormwind's 22 are all exterior.
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
