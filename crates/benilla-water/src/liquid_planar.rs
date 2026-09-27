//! Which water a planar mirror serves and which the cube probe does, per 4.17-yd liquid cell,
//! decided once over a whole map (or one WMO placement) so the answer never depends on what is
//! streamed in.
//!
//! - A **body** is connected flat water between real slopes: a cell whose corners drop more than
//!   [`SLOPE_YD`], or two neighbours stepping more than that, ends it. A body votes one mirror
//!   plane, halfway between its highest and lowest water, however gently it eases along the way.
//! - **Probe water** is the slopes themselves, bodies under [`MIN_AREA_YD2`], and the bodies a
//!   group of meeting waters cannot mirror: bodies meet when one's water lies within
//!   [`BOX_MARGIN_YD`] of the other's bounding box, and groups link through basins but not through
//!   big water ([`BIG_AREA_YD2`]). In a group the largest body keeps a mirror, then the largest at
//!   least [`GAP_YD`] from it; a body within that of either shares it, the rest give way.
//! - Each connected **probe region** gets fixed probe spots: one per [`PART_CELLS`] of its length,
//!   on its centre line, halfway down that part's drop. A cell reads the probe of its region's
//!   nearest spot, blending towards the second nearest where two parts meet.
//! - Mirror water within [`RAMP_CELLS`] of probe water ramps its mirror weight from 0 to 1 and
//!   reads the nearest probe region's spots, so the two tiers meet without an edge.
//!
//! Magma and slime are neither: they vote no plane and take no probe.

use std::collections::{HashMap, HashSet, VecDeque};

use std::sync::Arc;

use benilla_formats::{CellWater, LiquidMesh, ProbeSpot, WaterClasses, NO_SPOT};

/// A cell whose corners differ by more than this, or two flat neighbours stepping more than this,
/// is a real slope (a fall, rapids, a step at a chunk edge).
const SLOPE_YD: f32 = 0.5;

/// The least area a body needs to earn a mirror plane, in square yards: just under one MCLQ
/// chunk. It keeps Elwynn's 1,579-yd² pond and drops slivers of flat water caught in a fall.
const MIN_AREA_YD2: f32 = 1_000.0;

/// How far around a body's bounding box other water still counts as meeting it, in yards.
const BOX_MARGIN_YD: f32 = 25.0;

/// Water at least this large, in square yards (the sea, a big lake, a long river), belongs to the
/// groups it meets but does not link them, so the sea does not make one group of a whole coast.
const BIG_AREA_YD2: f32 = 50_000.0;

/// Two bodies are different planes when their levels are this far apart: the second mirror's
/// least gap from the first.
const GAP_YD: f32 = 3.0;

/// How many cells the mirror's weight takes to rise from probe water to full.
const RAMP_CELLS: u32 = 3;

/// A probe region's length per fixed spot, in cells (100 yd).
const PART_CELLS: u32 = 24;

/// A part with fewer cells than this gets no spot of its own; its cells read the nearest.
const PART_MIN_CELLS: usize = 8;

/// Bumped whenever the classification changes, so a cached map is rebuilt.
pub const PLANAR_VERSION: u32 = 1;

type Key = (i32, i32);

/// One MCNK's 8 × 8 cells: most are alike (open sea, a lake's interior) and share one record.
#[derive(Clone, Debug, PartialEq)]
enum ChunkWater {
    Uniform(CellWater),
    Cells(Box<[CellWater; 64]>),
}

/// The classification of one map or one WMO placement.
#[derive(Clone, Debug, PartialEq)]
pub struct PlanarMap {
    cell: f32,
    chunks: HashMap<Key, ChunkWater>,
    spots: Vec<ProbeSpot>,
}

fn key_of(p: [f32; 2], cell: f32) -> Key {
    ((p[0] / cell).floor() as i32, (p[1] / cell).floor() as i32)
}

fn chunk_of((a, b): Key) -> (Key, usize) {
    (
        (a.div_euclid(8), b.div_euclid(8)),
        (a.rem_euclid(8) * 8 + b.rem_euclid(8)) as usize,
    )
}

fn cell_centre(lq: &LiquidMesh, i: usize, j: usize) -> [f32; 2] {
    let cols = lq.grid[0] as usize;
    let c =
        [(i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1)].map(|(a, b)| lq.positions[b * cols + a]);
    [
        c.iter().map(|p| p[0]).sum::<f32>() / 4.0,
        c.iter().map(|p| p[1]).sum::<f32>() / 4.0,
    ]
}

/// The batch's wet cells, packed per chunk so a neighbour lookup costs one chunk hash.
struct Grid {
    cell: f32,
    chunk: HashMap<Key, u32>,
    keys: Vec<Key>,
    h: Vec<f32>,
    flat: Vec<bool>,
    water: Vec<bool>,
}

impl Grid {
    fn build<'a>(meshes: impl Iterator<Item = &'a LiquidMesh>) -> Option<Self> {
        let mut g = Grid {
            cell: 0.0,
            chunk: HashMap::new(),
            keys: Vec::new(),
            h: Vec::new(),
            flat: Vec::new(),
            water: Vec::new(),
        };
        for lq in meshes {
            let (cols, rows) = (lq.grid[0] as usize, lq.grid[1] as usize);
            if cols < 2 || rows < 2 || lq.positions.len() != cols * rows {
                continue;
            }
            if g.cell <= 0.0 {
                let (a, b) = (lq.positions[0], lq.positions[1]);
                g.cell = ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt();
                if g.cell.is_nan() || g.cell <= 1e-3 {
                    return None;
                }
            }
            let water = !lq.kind.is_fullbright();
            let xt = cols - 1;
            for (c, _) in lq.wet.iter().enumerate().filter(|(_, w)| **w) {
                let (i, j) = (c % xt, c / xt);
                let real: Vec<f32> = [(i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1)]
                    .iter()
                    .map(|&(a, b)| lq.positions[b * cols + a][2])
                    .filter(|z| z.abs() < 1.0e8)
                    .collect();
                if real.is_empty() {
                    continue;
                }
                let key = key_of(cell_centre(lq, i, j), g.cell);
                let (ck, slot) = chunk_of(key);
                let n = g.h.len() as u32;
                let base = *g.chunk.entry(ck).or_insert(n);
                if base == n {
                    g.h.extend([f32::NAN; 64]);
                    g.flat.extend([false; 64]);
                    g.water.extend([false; 64]);
                    g.keys
                        .extend((0..64).map(|s| (ck.0 * 8 + s / 8, ck.1 * 8 + s % 8)));
                }
                let k = base as usize + slot;
                if !g.h[k].is_nan() {
                    continue; // a second layer over the same cell: the first wins
                }
                let lo = real.iter().copied().fold(f32::MAX, f32::min);
                let hi = real.iter().copied().fold(f32::MIN, f32::max);
                g.h[k] = real.iter().sum::<f32>() / real.len() as f32;
                g.flat[k] = hi - lo <= SLOPE_YD;
                g.water[k] = water;
            }
        }
        (!g.h.is_empty()).then_some(g)
    }

    fn at(&self, key: Key) -> Option<usize> {
        let (ck, slot) = chunk_of(key);
        let k = *self.chunk.get(&ck)? as usize + slot;
        (!self.h[k].is_nan()).then_some(k)
    }

    fn neighbours(&self, k: usize) -> impl Iterator<Item = usize> + '_ {
        let (a, b) = self.keys[k];
        [(a + 1, b), (a - 1, b), (a, b + 1), (a, b - 1)]
            .into_iter()
            .filter_map(|n| self.at(n))
    }

    fn cells(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.h.len()).filter(|&k| !self.h[k].is_nan())
    }

    fn centre(&self, k: usize) -> [f32; 2] {
        [
            (self.keys[k].0 as f32 + 0.5) * self.cell,
            (self.keys[k].1 as f32 + 0.5) * self.cell,
        ]
    }
}

/// A body's mirror plane: halfway between its highest and lowest water, so neither end of a long
/// river is further off it than half its drop.
fn body_level(heights: impl Iterator<Item = f32>) -> f32 {
    let (lo, hi) = heights.fold((f32::MAX, f32::MIN), |(lo, hi), h| (lo.min(h), hi.max(h)));
    (lo + hi) * 0.5
}

/// Connected components of the cells `member` admits, joined where `link` allows; `u32::MAX`
/// outside any.
fn components(
    g: &Grid,
    member: impl Fn(usize) -> bool,
    link: impl Fn(usize, usize) -> bool,
) -> (Vec<u32>, Vec<Vec<usize>>) {
    let mut id = vec![u32::MAX; g.h.len()];
    let mut comps = Vec::new();
    for start in g.cells() {
        if id[start] != u32::MAX || !member(start) {
            continue;
        }
        let n = comps.len() as u32;
        let mut cells = vec![start];
        id[start] = n;
        let mut q = 0;
        while q < cells.len() {
            let k = cells[q];
            q += 1;
            for m in g.neighbours(k) {
                if id[m] == u32::MAX && member(m) && link(k, m) {
                    id[m] = n;
                    cells.push(m);
                }
            }
        }
        comps.push(cells);
    }
    (id, comps)
}

impl PlanarMap {
    /// Classify every cell of a map's MCLQ water, or one WMO placement's. `None` without a grid.
    pub fn build<'a>(meshes: impl Iterator<Item = &'a LiquidMesh>) -> Option<Self> {
        let g = Grid::build(meshes)?;
        let cell = g.cell;

        // Bodies: flat water, split by real slopes.
        let (body, bodies) = components(
            &g,
            |k| g.water[k] && g.flat[k],
            |a, b| (g.h[a] - g.h[b]).abs() <= SLOPE_YD,
        );
        let level: Vec<f32> = bodies
            .iter()
            .map(|c| body_level(c.iter().map(|&k| g.h[k])))
            .collect();
        let large: Vec<bool> = bodies
            .iter()
            .map(|c| c.len() as f32 * cell * cell >= MIN_AREA_YD2)
            .collect();
        let demoted = stacked(&g, &body, &bodies, &level, &large);
        let mirrored = |b: u32| b != u32::MAX && large[b as usize] && !demoted[b as usize];

        // Probe water: every water cell no mirrored body holds, in connected regions.
        let (region, regions) = components(&g, |k| g.water[k] && !mirrored(body[k]), |_, _| true);
        let shore = shore_distance(&g);
        let mut spots: Vec<ProbeSpot> = Vec::new();
        let mut answer = vec![CellWater::DRY; g.h.len()];
        for cells in &regions {
            let mut ids: Vec<u16> = Vec::new();
            for k in place_spots(&g, cells, &region, &shore) {
                if spots.len() >= NO_SPOT as usize {
                    break;
                }
                let [x, y] = g.centre(k);
                ids.push(spots.len() as u16);
                spots.push(ProbeSpot { at: [x, y, g.h[k]] });
            }
            for &k in cells {
                answer[k] = CellWater {
                    plane: f32::NAN,
                    weight: 0.0,
                    ..nearest_spots(g.centre(k), &ids, &spots)
                };
            }
        }

        // Mirror water: its body's plane, a ramp off probe water, and the probe the ramp reads.
        let mut dist = vec![u32::MAX; g.h.len()];
        let mut queue: VecDeque<usize> = VecDeque::new();
        for k in g.cells() {
            if region[k] != u32::MAX {
                dist[k] = 0;
                queue.push_back(k);
            }
        }
        while let Some(k) = queue.pop_front() {
            if dist[k] >= RAMP_CELLS {
                continue;
            }
            for m in g.neighbours(k) {
                if dist[m] == u32::MAX && g.water[m] {
                    dist[m] = dist[k] + 1;
                    answer[m].spots = answer[k].spots;
                    answer[m].mix = answer[k].mix;
                    queue.push_back(m);
                }
            }
        }
        for k in g.cells() {
            if g.water[k] && mirrored(body[k]) {
                answer[k].plane = level[body[k] as usize];
                answer[k].weight = dist[k].min(RAMP_CELLS) as f32 / RAMP_CELLS as f32;
            }
        }

        let mut chunks = HashMap::with_capacity(g.chunk.len());
        for (&ck, &base) in &g.chunk {
            let base = base as usize;
            let slots: [CellWater; 64] = std::array::from_fn(|s| answer[base + s]);
            let wet = |s: usize| !g.h[base + s].is_nan();
            let first = (0..64).find(|&s| wet(s)).map(|s| slots[s]);
            let uniform = first.filter(|f| (0..64).all(|s| !wet(s) || same(&slots[s], f)));
            chunks.insert(
                ck,
                match uniform {
                    Some(u) => ChunkWater::Uniform(u),
                    None => ChunkWater::Cells(Box::new(slots)),
                },
            );
        }
        Some(Self {
            cell,
            chunks,
            spots,
        })
    }

    /// The cell whose centre is WoW `(x, y)`, if the map holds water there.
    pub fn cell(&self, x: f32, y: f32) -> Option<CellWater> {
        let (ck, slot) = chunk_of(key_of([x, y], self.cell));
        match self.chunks.get(&ck)? {
            ChunkWater::Uniform(c) => Some(*c),
            ChunkWater::Cells(cells) => Some(cells[slot]),
        }
    }

    /// The mirror's weight at a lattice corner: the least of the cells around it, so a corner
    /// touching probe water is 0 and the ramp meets it continuously. 1 with no water there.
    pub fn corner(&self, x: f32, y: f32) -> f32 {
        let h = self.cell * 0.5;
        [(-h, -h), (h, -h), (-h, h), (h, h)]
            .iter()
            .filter_map(|(dx, dy)| self.cell(x + dx, y + dy))
            .map(|c| c.weight)
            .fold(1.0, f32::min)
    }

    /// Per cell of `lq`, its answer; [`CellWater::DRY`] where the map has none.
    pub fn cells(&self, lq: &LiquidMesh) -> Vec<CellWater> {
        let (cols, rows) = (lq.grid[0] as usize, lq.grid[1] as usize);
        if cols < 2 || rows < 2 || lq.positions.len() != cols * rows {
            return Vec::new();
        }
        let xt = cols - 1;
        (0..xt * (rows - 1))
            .map(|c| {
                let [x, y] = cell_centre(lq, c % xt, c / xt);
                self.cell(x, y).unwrap_or(CellWater::DRY)
            })
            .collect()
    }

    /// Per cell of `lq`, the mirror plane it votes, NaN where none.
    pub fn planes(&self, lq: &LiquidMesh) -> Vec<f32> {
        self.cells(lq).iter().map(|c| c.plane).collect()
    }

    /// The probe spots, indexed by the ids cells carry.
    pub fn spots(&self) -> &[ProbeSpot] {
        &self.spots
    }

    /// The whole map as bytes, for the on-disk cache.
    pub fn to_bytes(&self, fingerprint: u64) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"BNWM");
        out.extend_from_slice(&PLANAR_VERSION.to_le_bytes());
        out.extend_from_slice(&fingerprint.to_le_bytes());
        out.extend_from_slice(&self.cell.to_le_bytes());
        out.extend_from_slice(&(self.spots.len() as u32).to_le_bytes());
        for s in &self.spots {
            for v in s.at {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        out.extend_from_slice(&(self.chunks.len() as u32).to_le_bytes());
        let put = |out: &mut Vec<u8>, c: &CellWater| {
            out.extend_from_slice(&c.plane.to_le_bytes());
            out.extend_from_slice(&c.weight.to_le_bytes());
            out.extend_from_slice(&c.spots[0].to_le_bytes());
            out.extend_from_slice(&c.spots[1].to_le_bytes());
            out.extend_from_slice(&c.mix.to_le_bytes());
        };
        let mut order: Vec<&Key> = self.chunks.keys().collect();
        order.sort_unstable();
        for ck in order {
            let cw = &self.chunks[ck];
            out.extend_from_slice(&ck.0.to_le_bytes());
            out.extend_from_slice(&ck.1.to_le_bytes());
            match cw {
                ChunkWater::Uniform(c) => {
                    out.push(0);
                    put(&mut out, c);
                }
                ChunkWater::Cells(cells) => {
                    out.push(1);
                    for c in cells.iter() {
                        put(&mut out, c);
                    }
                }
            }
        }
        out
    }

    /// A map read back from [`Self::to_bytes`]; `None` when it is not one, or was made from a
    /// different install (`fingerprint`) or classification ([`PLANAR_VERSION`]).
    pub fn from_bytes(bytes: &[u8], fingerprint: u64) -> Option<Self> {
        let mut r = Bytes { b: bytes, at: 0 };
        if r.take(4)? != b"BNWM" || r.u32()? != PLANAR_VERSION || r.u64()? != fingerprint {
            return None;
        }
        let cell = r.f32()?;
        let n_spots = r.u32()? as usize;
        let mut spots = Vec::with_capacity(n_spots.min(1 << 16));
        for _ in 0..n_spots {
            spots.push(ProbeSpot {
                at: [r.f32()?, r.f32()?, r.f32()?],
            });
        }
        let n_chunks = r.u32()? as usize;
        let mut chunks = HashMap::with_capacity(n_chunks.min(1 << 20));
        for _ in 0..n_chunks {
            let ck = (r.u32()? as i32, r.u32()? as i32);
            let cw = match r.take(1)?[0] {
                0 => ChunkWater::Uniform(r.cell()?),
                _ => {
                    let mut cells = Box::new([CellWater::DRY; 64]);
                    for c in cells.iter_mut() {
                        *c = r.cell()?;
                    }
                    ChunkWater::Cells(cells)
                }
            };
            chunks.insert(ck, cw);
        }
        (r.at == bytes.len()).then_some(Self {
            cell,
            chunks,
            spots,
        })
    }
}

/// A little-endian reader over the cache bytes.
struct Bytes<'a> {
    b: &'a [u8],
    at: usize,
}

impl Bytes<'_> {
    fn take(&mut self, n: usize) -> Option<&[u8]> {
        let s = self.b.get(self.at..self.at + n)?;
        self.at += n;
        Some(s)
    }
    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }
    fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }
    fn u16(&mut self) -> Option<u16> {
        Some(u16::from_le_bytes(self.take(2)?.try_into().ok()?))
    }
    fn f32(&mut self) -> Option<f32> {
        Some(f32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }
    fn cell(&mut self) -> Option<CellWater> {
        Some(CellWater {
            plane: self.f32()?,
            weight: self.f32()?,
            spots: [self.u16()?, self.u16()?],
            mix: self.f32()?,
        })
    }
}

/// Two answers alike, NaN planes included.
fn same(a: &CellWater, b: &CellWater) -> bool {
    (a.plane == b.plane || (a.plane.is_nan() && b.plane.is_nan()))
        && a.weight == b.weight
        && a.spots == b.spots
        && a.mix == b.mix
}

/// Which large bodies give way: see the module doc's groups. Built from which bodies meet (water
/// of one within [`BOX_MARGIN_YD`] of the other's bounding box), linked through bodies under
/// [`BIG_AREA_YD2`] only.
fn stacked(
    g: &Grid,
    body: &[u32],
    bodies: &[Vec<usize>],
    level: &[f32],
    large: &[bool],
) -> Vec<bool> {
    let margin = (BOX_MARGIN_YD / g.cell).round() as i32;
    let area = |b: usize| bodies[b].len() as f32 * g.cell * g.cell;
    let big = |b: usize| area(b) >= BIG_AREA_YD2;
    // Which large bodies each large body's grown box holds water of.
    let mut meets: Vec<HashSet<u32>> = vec![HashSet::new(); bodies.len()];
    for (b, cells) in bodies.iter().enumerate() {
        if !large[b] || big(b) {
            continue; // a big body's box is mostly land and says nothing about what meets it
        }
        let [x0, x1, y0, y1] =
            cells
                .iter()
                .fold([i32::MAX, i32::MIN, i32::MAX, i32::MIN], |r, &k| {
                    let (x, y) = g.keys[k];
                    [r[0].min(x), r[1].max(x), r[2].min(y), r[3].max(y)]
                });
        let (x0, x1, y0, y1) = (x0 - margin, x1 + margin, y0 - margin, y1 + margin);
        for cx in x0.div_euclid(8)..=x1.div_euclid(8) {
            for cy in y0.div_euclid(8)..=y1.div_euclid(8) {
                let Some(&base) = g.chunk.get(&(cx, cy)) else {
                    continue;
                };
                let slots = base as usize..base as usize + 64;
                for (&(x, y), &a) in g.keys[slots.clone()].iter().zip(&body[slots]) {
                    if a != u32::MAX
                        && a as usize != b
                        && large[a as usize]
                        && (x0..=x1).contains(&x)
                        && (y0..=y1).contains(&y)
                    {
                        meets[b].insert(a);
                    }
                }
            }
        }
    }
    for b in 0..bodies.len() {
        for a in meets[b].clone() {
            meets[a as usize].insert(b as u32);
        }
    }
    // Groups: linked through bodies that are not big; big ones join every group they meet.
    let mut out = vec![false; bodies.len()];
    let mut seen = vec![false; bodies.len()];
    for start in 0..bodies.len() {
        if seen[start] || !large[start] || big(start) || meets[start].is_empty() {
            continue;
        }
        let mut group: Vec<usize> = Vec::new();
        let mut members: HashSet<usize> = HashSet::new();
        let mut queue = VecDeque::from([start]);
        seen[start] = true;
        while let Some(b) = queue.pop_front() {
            group.push(b);
            members.insert(b);
            for &a in &meets[b] {
                let a = a as usize;
                if big(a) {
                    members.insert(a);
                } else if !seen[a] {
                    seen[a] = true;
                    queue.push_back(a);
                }
            }
        }
        let mut by_size: Vec<usize> = members.into_iter().collect();
        by_size.sort_by(|&a, &b| bodies[b].len().cmp(&bodies[a].len()).then(a.cmp(&b)));
        let first = by_size[0];
        let second = by_size
            .iter()
            .copied()
            .find(|&b| (level[b] - level[first]).abs() >= GAP_YD);
        for b in group {
            let near = |k: usize| (level[b] - level[k]).abs() < GAP_YD;
            out[b] = !near(first) && second.is_none_or(|s| !near(s));
        }
    }
    out
}

/// Each cell's distance from dry land in cells: 1 beside a dry cell, rising towards the middle.
fn shore_distance(g: &Grid) -> Vec<u32> {
    let mut dist = vec![u32::MAX; g.h.len()];
    let mut queue = VecDeque::new();
    for k in g.cells() {
        if g.neighbours(k).count() < 4 {
            dist[k] = 1;
            queue.push_back(k);
        }
    }
    while let Some(k) = queue.pop_front() {
        for m in g.neighbours(k) {
            if dist[m] == u32::MAX {
                dist[m] = dist[k] + 1;
                queue.push_back(m);
            }
        }
    }
    dist
}

/// A probe region's spots: the region cut into parts of [`PART_CELLS`] along its length from its
/// highest cell, and in each part the cell on its centre line nearest the part's mid-height.
fn place_spots(g: &Grid, cells: &[usize], region: &[u32], shore: &[u32]) -> Vec<usize> {
    let me = region[cells[0]];
    let top = *cells
        .iter()
        .max_by(|a, b| g.h[**a].total_cmp(&g.h[**b]))
        .expect("a region has cells");
    let mut along: HashMap<usize, u32> = HashMap::from([(top, 0)]);
    let mut queue = VecDeque::from([top]);
    while let Some(k) = queue.pop_front() {
        let d = along[&k];
        for m in g.neighbours(k) {
            if region[m] == me && !along.contains_key(&m) {
                along.insert(m, d + 1);
                queue.push_back(m);
            }
        }
    }
    let mut parts: HashMap<u32, Vec<usize>> = HashMap::new();
    for &k in cells {
        parts.entry(along[&k] / PART_CELLS).or_default().push(k);
    }
    let mut order: Vec<u32> = parts.keys().copied().collect();
    order.sort_unstable();
    let mut out = Vec::new();
    for p in order {
        let part = &parts[&p];
        if part.len() < PART_MIN_CELLS && !out.is_empty() {
            continue;
        }
        let lo = part.iter().map(|&k| g.h[k]).fold(f32::MAX, f32::min);
        let hi = part.iter().map(|&k| g.h[k]).fold(f32::MIN, f32::max);
        let mid = (lo + hi) * 0.5;
        let deepest = part.iter().map(|&k| shore[k]).max().unwrap_or(1);
        let spot = part
            .iter()
            .filter(|&&k| shore[k] + 1 >= deepest)
            .min_by(|&&a, &&b| {
                (g.h[a] - mid)
                    .abs()
                    .total_cmp(&(g.h[b] - mid).abs())
                    .then(shore[b].cmp(&shore[a]))
            });
        out.extend(spot.copied());
    }
    out
}

/// A probe cell's two nearest spots among `ids`, lower id first, with the higher id's share: 0
/// or 1 deep inside one part, a half where two are equally near.
fn nearest_spots(at: [f32; 2], ids: &[u16], spots: &[ProbeSpot]) -> CellWater {
    let mut by: Vec<(f32, u16)> = ids
        .iter()
        .map(|&i| {
            let s = spots[i as usize].at;
            ((s[0] - at[0]).hypot(s[1] - at[1]), i)
        })
        .collect();
    by.sort_by(|a, b| a.0.total_cmp(&b.0));
    let (spots, mix) = match by.as_slice() {
        [] => ([NO_SPOT, NO_SPOT], 0.0),
        [(_, a)] => ([*a, NO_SPOT], 0.0),
        [(da, a), (db, b), ..] => {
            let second = 0.5 * smoothstep(0.3, 0.5, da / (da + db).max(1e-3));
            if a < b {
                ([*a, *b], second)
            } else {
                ([*b, *a], 1.0 - second)
            }
        }
    };
    CellWater {
        spots,
        mix,
        ..CellWater::DRY
    }
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

impl WaterClasses for PlanarMap {
    fn cells(&self, lq: &LiquidMesh) -> Vec<CellWater> {
        PlanarMap::cells(self, lq)
    }

    fn corner(&self, x: f32, y: f32) -> f32 {
        PlanarMap::corner(self, x, y)
    }

    fn spots(&self) -> &[ProbeSpot] {
        PlanarMap::spots(self)
    }
}

/// One batch of liquid classified on its own: a WMO placement's pools.
pub(crate) fn classify_batch(liquids: &[&LiquidMesh]) -> Option<Arc<dyn WaterClasses>> {
    PlanarMap::build(liquids.iter().copied()).map(|m| Arc::new(m) as Arc<dyn WaterClasses>)
}

#[cfg(test)]
mod tests {
    use super::*;
    use benilla_formats::LiquidKind;

    const U: f32 = 100.0 / 3.0 / 8.0;
    /// A lattice-aligned origin inside one ADT tile, as MCLQ grids are.
    const X0: f32 = -2051.0 * U;
    const Y0: f32 = 72.0 * U;

    /// A `cols × rows`-cell sheet with its first corner at `(x0, y0)` and corner heights `z(i, j)`,
    /// all wet.
    fn sheet_at(
        x0: f32,
        y0: f32,
        cols: usize,
        rows: usize,
        z: impl Fn(usize, usize) -> f32,
    ) -> LiquidMesh {
        let mut positions = Vec::new();
        for j in 0..=rows {
            for i in 0..=cols {
                positions.push([x0 - j as f32 * U, y0 - i as f32 * U, z(i, j)]);
            }
        }
        LiquidMesh {
            grid: [(cols + 1) as u32, (rows + 1) as u32],
            wet: vec![true; cols * rows],
            shared: vec![false; cols * rows],
            uvs: vec![[0.0; 2]; positions.len()],
            depths: vec![0.0; positions.len()],
            positions,
            indices: Vec::new(),
            sound_nibble: 0,
            material_id: None,
            kind: LiquidKind::Still,
        }
    }

    fn sheet(cols: usize, rows: usize, z: impl Fn(usize, usize) -> f32) -> LiquidMesh {
        sheet_at(X0, Y0, cols, rows, z)
    }

    fn at(map: &PlanarMap, lq: &LiquidMesh, i: usize, j: usize) -> CellWater {
        let [x, y] = cell_centre(lq, i, j);
        map.cell(x, y).expect("water")
    }

    #[test]
    fn a_large_flat_lake_is_one_plane() {
        let lq = sheet(30, 30, |_, _| 48.64);
        let map = PlanarMap::build([&lq].into_iter()).expect("a grid");
        assert!(map.cells(&lq).iter().all(|c| c.weight == 1.0));
        assert!(map.planes(&lq).iter().all(|z| (*z - 48.64).abs() < 1e-3));
        assert!(map.spots().is_empty());
    }

    /// A pond of 7 × 7 cells (851 yd²) is probe water with one spot in its middle; 10 × 10
    /// (1,736 yd²) keeps a mirror.
    #[test]
    fn a_small_pond_is_probe_water_with_its_probe_in_the_middle() {
        let pond = sheet(10, 10, |_, _| 55.91);
        let map = PlanarMap::build([&pond].into_iter()).expect("a grid");
        assert!(map.cells(&pond).iter().all(|c| c.weight == 1.0));
        let lq = sheet(7, 7, |_, _| 55.91);
        let map = PlanarMap::build([&lq].into_iter()).expect("a grid");
        assert!(map
            .cells(&lq)
            .iter()
            .all(|c| c.weight == 0.0 && c.plane.is_nan()));
        assert_eq!(map.spots().len(), 1);
        let [x, y] = cell_centre(&lq, 3, 3);
        let s = map.spots()[0].at;
        assert!((s[0] - x).abs() < 0.05 && (s[1] - y).abs() < 0.05, "{s:?}");
        assert!(map.cells(&lq).iter().all(|c| c.spots == [0, NO_SPOT]));
    }

    /// A lake with a fall at one end: the fall is probe water, and the lake ramps its mirror up
    /// over `RAMP_CELLS` cells from it, reading the fall's probe on the way.
    #[test]
    fn the_mirror_ramps_up_away_from_a_fall() {
        let lq = sheet(
            30,
            32,
            |_, j| if j <= 1 { 42.0 + 3.0 * j as f32 } else { 48.64 },
        );
        let map = PlanarMap::build([&lq].into_iter()).expect("a grid");
        assert_eq!(at(&map, &lq, 10, 1).weight, 0.0, "the fall itself");
        assert_eq!(at(&map, &lq, 10, 2).weight, 1.0 / 3.0, "one cell off it");
        assert_eq!(at(&map, &lq, 10, 4).weight, 1.0, "past the ramp");
        assert_eq!(
            at(&map, &lq, 10, 2).spots,
            at(&map, &lq, 10, 1).spots,
            "the ramp reads the fall's probe"
        );
        let edge = lq.positions[2 * 31 + 10];
        assert_eq!(map.corner(edge[0], edge[1]), 0.0);
    }

    /// A long river easing down 6 yd with no real slope is one body and one plane, halfway down:
    /// neither end is more than 3 yd off it, and it never switches.
    #[test]
    fn a_gently_descending_river_is_one_plane() {
        let lq = sheet(8, 60, |_, j| {
            if j < 30 {
                40.0
            } else {
                40.0 - 0.2 * (j - 30) as f32
            }
        });
        let map = PlanarMap::build([&lq].into_iter()).expect("a grid");
        let planes = map.planes(&lq);
        assert!(
            planes.iter().all(|z| (*z - 37.1).abs() < 0.05),
            "{planes:?}"
        );
    }

    /// Two flat sheets meeting 5 yd apart are two bodies: a step is a real slope even with no
    /// sloped cell in it.
    #[test]
    fn a_step_between_flat_cells_splits_bodies() {
        let upper = sheet(10, 16, |_, _| 50.0);
        let lower = sheet_at(X0 - 16.0 * U, Y0, 10, 16, |_, _| 45.0);
        let map = PlanarMap::build([&upper, &lower].into_iter()).expect("a grid");
        assert!((at(&map, &upper, 5, 8).plane - 50.0).abs() < 1e-3);
        assert!((at(&map, &lower, 5, 8).plane - 45.0).abs() < 1e-3);
    }

    /// Where three heights meet, the smaller gives way: a body whose box holds water of two larger
    /// bodies is probe water with its own spot, even when it is the highest. With one of the large
    /// bodies far off, it keeps its mirror.
    #[test]
    fn the_smallest_of_three_heights_gives_way() {
        let lake = |col: f32, z: f32| sheet_at(X0, Y0 - col * U, 20, 20, move |_, _| z);
        // 20 × 3 cells (1,042 yd²) between the two lakes, touching both their boxes.
        let small = |z: f32| sheet_at(X0 - 8.0 * U, Y0 - 20.0 * U, 20, 3, move |_, _| z);
        for z in [20.0, 60.0] {
            let three = [lake(0.0, 0.0), small(z), lake(40.0, 40.0)];
            let map = PlanarMap::build(three.iter()).expect("a grid");
            assert!(
                !at(&map, &three[0], 10, 10).plane.is_nan(),
                "a large lake keeps it"
            );
            assert!(
                at(&map, &three[1], 10, 1).plane.is_nan(),
                "the small body gives way at {z}"
            );
            assert!(
                !at(&map, &three[2], 10, 10).plane.is_nan(),
                "a large lake keeps it"
            );
            assert_eq!(map.spots().len(), 1, "one probe, in the small body");
        }
        let apart = [
            lake(0.0, 0.0),
            small(20.0),
            sheet_at(X0 - 200.0 * U, Y0 - 40.0 * U, 20, 20, |_, _| 40.0),
        ];
        let map = PlanarMap::build(apart.iter()).expect("a grid");
        assert!(
            !at(&map, &apart[1], 10, 1).plane.is_nan(),
            "only two heights meet it"
        );
    }

    /// A dam: a big lake above, big water below, two basins between. The two big waters keep the
    /// mirrors and both basins give way, though each basin meets only one big water itself.
    #[test]
    fn a_dams_basins_both_give_way() {
        // Lake (big, 0..120 rows at col 0), basins at cols 125.. and 150.., low water past them.
        let body = |col: f32, rows: usize, cols: usize, z: f32| {
            sheet_at(X0, Y0 - col * U, cols, rows, move |_, _| z)
        };
        let dam = [
            body(0.0, 150, 160, 297.6), // 24,000 cells: big
            body(162.0, 30, 30, 177.9), // a basin, touching the lake's box
            body(194.0, 30, 30, 123.5), // the next basin, touching the first
            body(226.0, 150, 160, 7.3), // big water below
        ];
        let map = PlanarMap::build(dam.iter()).expect("a grid");
        assert!(
            !at(&map, &dam[0], 10, 10).plane.is_nan(),
            "the lake keeps its mirror"
        );
        assert!(
            at(&map, &dam[1], 10, 10).plane.is_nan(),
            "the upper basin gives way"
        );
        assert!(
            at(&map, &dam[2], 10, 10).plane.is_nan(),
            "the lower basin gives way"
        );
        assert!(
            !at(&map, &dam[3], 10, 10).plane.is_nan(),
            "the water below keeps its mirror"
        );
    }

    /// 300 yd of rapids gets a spot per 100 yd, each about halfway down its part, and a cell
    /// between two spots reads both.
    #[test]
    fn long_rapids_get_a_spot_per_part_halfway_down() {
        let lq = sheet(3, 72, |_, j| 100.0 - 0.6 * j as f32);
        let map = PlanarMap::build([&lq].into_iter()).expect("a grid");
        assert_eq!(map.spots().len(), 3, "{:?}", map.spots());
        let s = map.spots()[0].at;
        assert!(
            (s[2] - (100.0 - 0.6 * 12.0)).abs() < 1.0,
            "halfway down the first part: {s:?}"
        );
        let between = at(&map, &lq, 1, 24);
        assert!(
            between.spots[1] != NO_SPOT && between.mix > 0.2 && between.mix < 0.8,
            "{between:?}"
        );
    }

    #[test]
    fn magma_votes_nothing_and_takes_no_probe() {
        let mut lq = sheet(10, 10, |_, _| 20.0);
        lq.kind = LiquidKind::Magma;
        let map = PlanarMap::build([&lq].into_iter()).expect("a grid");
        assert!(map
            .cells(&lq)
            .iter()
            .all(|c| c.plane.is_nan() && c.weight == 1.0 && c.spots == [NO_SPOT; 2]));
        assert!(map.spots().is_empty());
    }

    #[test]
    fn the_cache_round_trips_and_refuses_another_install() {
        let lake = sheet(
            30,
            32,
            |_, j| if j <= 1 { 42.0 + 3.0 * j as f32 } else { 48.64 },
        );
        let map = PlanarMap::build([&lake].into_iter()).expect("a grid");
        let bytes = map.to_bytes(7);
        let back = PlanarMap::from_bytes(&bytes, 7).expect("reads back");
        assert_eq!(back.to_bytes(7), bytes);
        assert!(PlanarMap::from_bytes(&bytes, 8).is_none());
    }
}
