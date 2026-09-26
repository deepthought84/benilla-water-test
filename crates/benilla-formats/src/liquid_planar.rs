//! Which water a planar mirror may serve, per 4.17-yd liquid cell: the flat sections large enough
//! to be worth a mirror plane. Falls, descending rivers and small pools are probe water; the mirror
//! stands down over them and the cube probe under it shows through.
//!
//! An ADT tile is judged over itself and its eight neighbours, so a stream crossing a tile border
//! is measured whole and the answer does not depend on which tiles are streamed in. A WMO
//! placement holds its whole pool and is judged alone.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::LiquidMesh;

/// A cell whose corners differ by more than this is sloped water (a fall or rapids).
const FLAT_YD: f32 = 0.5;

/// A section is the connected cells within this of its seed height — the mirror's own bucket.
const SECTION_YD: f32 = 0.5;

/// The least area a flat section needs to earn a mirror plane, in square yards: about nine MCLQ
/// chunks, or a pool 100 yd across. Measured over both continents, it leaves at most two planes
/// within 200 yd of all but 22 of 59,000 water patches.
const MIN_AREA_YD2: f32 = 10_000.0;

/// How many cells the mirror's weight takes to rise from a probe cell to full, so the two tiers
/// meet in a ramp and not at a cell edge.
const RAMP_CELLS: u32 = 3;

type Key = (i32, i32);

/// The mirror's weight per liquid cell, keyed by the cell's centre on the lattice.
pub struct PlanarMap {
    cell: f32,
    weight: HashMap<Key, f32>,
}

struct Cell {
    key: Key,
    h: f32,
    flat: bool,
}

impl PlanarMap {
    /// Classify a batch that holds its whole water, a WMO placement. `None` without a usable grid.
    pub fn build<'a>(batch: impl Iterator<Item = &'a LiquidMesh>) -> Option<Self> {
        let (cell, cells, index) = gather(batch)?;
        let keep: HashSet<Key> = cells.iter().map(|c| c.key).collect();
        Some(classify(cell, &cells, &index, &keep))
    }

    /// Classify one ADT tile's cells, measuring sections over the tile and `window`, its
    /// neighbours' liquids. Weights are kept for the tile's cells and the ring around them, so a
    /// corner on the tile border reads the same cells from either side.
    pub fn build_tile<'a>(
        own: impl Iterator<Item = &'a LiquidMesh>,
        window: impl Iterator<Item = &'a LiquidMesh>,
    ) -> Option<Self> {
        let own: Vec<&LiquidMesh> = own.collect();
        let (_, own_cells, _) = gather(own.iter().copied())?;
        let mut keep: HashSet<Key> = HashSet::new();
        for c in &own_cells {
            keep.insert(c.key);
            keep.extend(around(c.key));
        }
        let (cell, cells, index) = gather(own.into_iter().chain(window))?;
        Some(classify(cell, &cells, &index, &keep))
    }

    /// The mirror's weight at a lattice corner: the least of the wet cells around it, so a corner
    /// touching probe water is 0 and the ramp meets it continuously. 1 with no wet cell there.
    pub fn corner(&self, x: f32, y: f32) -> f32 {
        let h = self.cell * 0.5;
        [(-h, -h), (h, -h), (-h, h), (h, h)]
            .iter()
            .filter_map(|(dx, dy)| self.weight.get(&key_of([x + dx, y + dy], self.cell)))
            .copied()
            .fold(1.0, f32::min)
    }

    /// Per cell of `lq`, whether it may vote for a mirror plane: any weight above 0.
    pub fn votes(&self, lq: &LiquidMesh) -> Vec<bool> {
        let (cols, rows) = (lq.grid[0] as usize, lq.grid[1] as usize);
        if cols < 2 || rows < 2 || lq.positions.len() != cols * rows {
            return Vec::new();
        }
        let xt = cols - 1;
        (0..xt * (rows - 1))
            .map(|c| {
                let centre = cell_centre(lq, c % xt, c / xt);
                self.weight
                    .get(&key_of(centre, self.cell))
                    .is_none_or(|w| *w > 0.0)
            })
            .collect()
    }
}

fn key_of(p: [f32; 2], cell: f32) -> Key {
    ((p[0] / cell).floor() as i32, (p[1] / cell).floor() as i32)
}

fn neighbours((a, b): Key) -> [Key; 4] {
    [(a + 1, b), (a - 1, b), (a, b + 1), (a, b - 1)]
}

fn around((a, b): Key) -> impl Iterator<Item = Key> {
    (-1..=1).flat_map(move |da| (-1..=1).map(move |db| (a + da, b + db)))
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

/// Every wet cell of the meshes, first occurrence of a key winning, with the lattice pitch.
fn gather<'a>(
    meshes: impl Iterator<Item = &'a LiquidMesh>,
) -> Option<(f32, Vec<Cell>, HashMap<Key, usize>)> {
    let mut cell = 0.0_f32;
    let mut cells: Vec<Cell> = Vec::new();
    let mut index: HashMap<Key, usize> = HashMap::new();
    for lq in meshes {
        let (cols, rows) = (lq.grid[0] as usize, lq.grid[1] as usize);
        if cols < 2 || rows < 2 || lq.positions.len() != cols * rows {
            continue;
        }
        if cell <= 0.0 {
            let (a, b) = (lq.positions[0], lq.positions[1]);
            cell = ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt();
            if cell.is_nan() || cell <= 1e-3 {
                return None;
            }
        }
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
            let key = key_of(cell_centre(lq, i, j), cell);
            if index.contains_key(&key) {
                continue;
            }
            let lo = real.iter().copied().fold(f32::MAX, f32::min);
            let hi = real.iter().copied().fold(f32::MIN, f32::max);
            index.insert(key, cells.len());
            cells.push(Cell {
                key,
                h: real.iter().sum::<f32>() / real.len() as f32,
                flat: hi - lo <= FLAT_YD,
            });
        }
    }
    (!cells.is_empty()).then_some((cell, cells, index))
}

fn classify(
    cell: f32,
    cells: &[Cell],
    index: &HashMap<Key, usize>,
    keep: &HashSet<Key>,
) -> PlanarMap {
    let planar = sections(cells, index, cell);
    let weight = ramp(cells, index, &planar)
        .filter(|(k, _)| keep.contains(k))
        .collect();
    PlanarMap { cell, weight }
}

/// Mark each cell planar or not. Sections are seeded at the most common remaining height and take
/// the connected cells within [`SECTION_YD`] of it, so no section drifts: a descending river breaks
/// into short steps and each is judged on its own size.
fn sections(cells: &[Cell], index: &HashMap<Key, usize>, cell: f32) -> Vec<bool> {
    let mut planar = vec![false; cells.len()];
    let mut left: Vec<bool> = cells.iter().map(|c| c.flat).collect();
    let bin = |h: f32| (h / 0.25).round() as i64;
    loop {
        let mut hist: HashMap<i64, u32> = HashMap::new();
        for c in cells.iter().zip(&left).filter(|(_, l)| **l).map(|(c, _)| c) {
            *hist.entry(bin(c.h)).or_default() += 1;
        }
        let Some((&mode, _)) = hist.iter().max_by_key(|(b, n)| (**n, -**b)) else {
            break;
        };
        let seed = mode as f32 * 0.25;
        let member = |k: usize| left[k] && (cells[k].h - seed).abs() <= SECTION_YD;
        let group: Vec<usize> = (0..cells.len()).filter(|&k| member(k)).collect();
        let mut seen = vec![false; cells.len()];
        for &start in &group {
            if seen[start] {
                continue;
            }
            let mut comp = Vec::new();
            let mut queue = VecDeque::from([start]);
            seen[start] = true;
            while let Some(k) = queue.pop_front() {
                comp.push(k);
                for n in neighbours(cells[k].key) {
                    if let Some(&m) = index.get(&n) {
                        if !seen[m] && member(m) {
                            seen[m] = true;
                            queue.push_back(m);
                        }
                    }
                }
            }
            let keep = comp.len() as f32 * cell * cell >= MIN_AREA_YD2;
            for &k in &comp {
                planar[k] = keep;
            }
        }
        for k in group {
            left[k] = false;
        }
    }
    planar
}

/// Each cell's weight: 0 on probe water, rising by `1/RAMP_CELLS` per cell away from it.
fn ramp<'a>(
    cells: &'a [Cell],
    index: &HashMap<Key, usize>,
    planar: &[bool],
) -> impl Iterator<Item = (Key, f32)> + 'a {
    let mut dist = vec![u32::MAX; cells.len()];
    let mut queue = VecDeque::new();
    for (k, p) in planar.iter().enumerate() {
        if !p {
            dist[k] = 0;
            queue.push_back(k);
        }
    }
    while let Some(k) = queue.pop_front() {
        if dist[k] >= RAMP_CELLS {
            continue;
        }
        for n in neighbours(cells[k].key) {
            if let Some(&m) = index.get(&n) {
                if dist[m] == u32::MAX {
                    dist[m] = dist[k] + 1;
                    queue.push_back(m);
                }
            }
        }
    }
    cells
        .iter()
        .zip(dist)
        .map(|(c, d)| (c.key, d.min(RAMP_CELLS) as f32 / RAMP_CELLS as f32))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LiquidKind;

    const U: f32 = 100.0 / 3.0 / 8.0;

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
        sheet_at(-8545.0, 300.0, cols, rows, z)
    }

    fn weight_at(map: &PlanarMap, lq: &LiquidMesh, i: usize, j: usize) -> f32 {
        map.weight[&key_of(cell_centre(lq, i, j), map.cell)]
    }

    /// A lake of 30 × 30 cells (15,600 yd²) keeps its mirror everywhere.
    #[test]
    fn a_large_flat_lake_is_planar() {
        let lq = sheet(30, 30, |_, _| 48.64);
        let map = PlanarMap::build([&lq].into_iter()).expect("a grid");
        assert!(map.weight.values().all(|w| *w == 1.0));
    }

    /// A pond of 10 × 10 cells (1,736 yd²) is probe water.
    #[test]
    fn a_small_pond_is_probe_water() {
        let lq = sheet(10, 10, |_, _| 55.91);
        let map = PlanarMap::build([&lq].into_iter()).expect("a grid");
        assert!(map.weight.values().all(|w| *w == 0.0));
    }

    /// A lake with a fall at one end: the fall's cells are 0 and the lake ramps up to 1 over
    /// `RAMP_CELLS` cells from it.
    #[test]
    fn the_mirror_ramps_up_away_from_a_fall() {
        // Corner rows 0 and 1 drop 6 yd, so cell rows 0 and 1 are the fall.
        let lq = sheet(
            30,
            32,
            |_, j| if j <= 1 { 42.0 + 3.0 * j as f32 } else { 48.64 },
        );
        let map = PlanarMap::build([&lq].into_iter()).expect("a grid");
        assert_eq!(weight_at(&map, &lq, 10, 1), 0.0, "the fall itself");
        assert_eq!(weight_at(&map, &lq, 10, 2), 1.0 / 3.0, "one cell off it");
        assert_eq!(weight_at(&map, &lq, 10, 4), 1.0, "past the ramp");
        let edge = lq.positions[2 * 31 + 10];
        assert_eq!(
            map.corner(edge[0], edge[1]),
            0.0,
            "the corner the fall touches"
        );
    }

    /// A 375-yd river descending 0.2 yd per cell breaks into short steps, none big enough for a
    /// mirror, even though every neighbour is within the section tolerance of the next.
    #[test]
    fn a_descending_river_is_probe_water() {
        let lq = sheet(8, 90, |_, j| 60.0 - 0.2 * j as f32);
        let map = PlanarMap::build([&lq].into_iter()).expect("a grid");
        assert!(map.weight.values().all(|w| *w == 0.0));
    }

    /// A tile's small piece of a lake that continues into the next tile is measured whole, and
    /// keeps only its own cells and their ring.
    #[test]
    fn a_tile_measures_its_sections_across_its_neighbours() {
        let own = sheet(10, 10, |_, _| 20.0);
        // The rest of the lake, starting where `own` ends.
        let next = sheet_at(-8545.0 - 10.0 * U, 300.0, 10, 60, |_, _| 20.0);
        let alone = PlanarMap::build_tile([&own].into_iter(), std::iter::empty()).expect("a grid");
        assert!(
            alone.weight.values().all(|w| *w == 0.0),
            "a pond on its own"
        );
        let map = PlanarMap::build_tile([&own].into_iter(), [&next].into_iter()).expect("a grid");
        assert!(
            map.weight.values().all(|w| *w == 1.0),
            "a lake with its neighbour"
        );
        assert_eq!(
            map.weight.len(),
            10 * 11,
            "its own 100 cells and the neighbour's first row"
        );
    }
}
