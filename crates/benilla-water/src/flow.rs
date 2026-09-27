//! Which way a map's water runs, per liquid cell, and how fast: the current the stylised surface
//! carries its ripple along. 1.12 water has no flow data (MCLQ holds heights, depths and a kind), so
//! it is read off the water's own shape, once per map with the rest of the classification. The sea
//! does not run; only the water that is not sea (the rivers) is walked.
//!
//! - **Where water ends:** the sea's edge; the lowest water of any water that never meets the sea;
//!   and every basin with no lower way out — a stretch whose steps stay under [`FLAT_STEP_YD`] that
//!   no lower water or sea borders — that is either at least [`BASIN_MIN_CELLS`] or a dead end,
//!   water stopping at dry land below higher water (a river running off the end of its MCLQ). A
//!   basin smaller than that with water beyond it is a dip, and its water runs on over the lip.
//!   Where water ends in a band within [`LAKE_TOL_YD`] of its lowest level, it ends at the band's
//!   far end from where water comes in, so a flat last reach still runs to its end.
//! - **How far upstream each cell is:** the cheapest path from where its water ends, walked over the
//!   8-neighbour lattice. A step costs its length, more near a bank ([`CENTRE_BIAS`]) so the path
//!   keeps to the middle of a channel, and more for each yard it climbs down ([`DOWN_COST`]), so a
//!   river is upstream of the lower water it feeds and a dip in an authored river does not turn it.
//! - **The direction** is down that distance's gradient, turned downhill wherever the surface
//!   itself falls more than [`TRUST_GRADE`] the other way, averaged over the neighbourhood and
//!   weighted down where the neighbourhood disagrees (a watershed); at a bank it is turned along
//!   the bank rather than into it.
//! - **The speed** is [`RIVER_SPEED`] in a channel [`REF_HALF_WIDTH`] cells half-wide, faster in a
//!   narrow one and gone on water wider than [`STILL_HALF_WIDTH`] (the half-width is the deepest
//!   water the cell's own water rises to, so a lake's rim is as still as its middle, and water
//!   within [`CALM_REACH`] of still water at its own level is still, so a lake's bays and the water
//!   among its islands are too); faster down a slope along the current ([`GRADE_GAIN`]), slower at a bank than
//!   mid-channel, never over [`FLOW_MAX`].

use std::cmp::Reverse;
use std::collections::{BinaryHeap, VecDeque};

use crate::liquid_planar::Grid;

/// Neighbouring water stepping less than this, in yards, is one stretch when looking for basins:
/// 2.4% over a cell, above every flat reach and most river ramps.
const FLAT_STEP_YD: f32 = 0.1;

/// The most one water may step from the next and still be the same water when measuring how wide
/// it is: a lake's MCLQ chunks meet up to a few tenths apart, a ramp's rows step further.
const LEVEL_STEP_YD: f32 = 0.5;

/// How far above a basin's lowest water, in yards, is still the water the basin ends in.
const LAKE_TOL_YD: f32 = 0.15;

/// A basin smaller than this, in cells (about 1,000 yd²), with water beyond it is a dip in a river
/// and not where it ends.
const BASIN_MIN_CELLS: usize = 60;

/// The extra cost of a step, in cells, per cell of distance from the bank it lands on: 1 cell
/// from the bank a step costs twice its length, 4 cells out a quarter more.
const CENTRE_BIAS: f32 = 1.0;

/// The cost, in cells, of each yard a path upstream climbs down past [`LEVEL_NOISE_YD`]: dear
/// enough that water never runs up a hill to reach the sea sooner (the flat Elwynn reach, 3,000
/// yd long, is not upstream of Lake Everstill's far outlet), cheap enough that a dip in an
/// authored river is crossed when there is no other way.
const DOWN_COST: f32 = 1000.0;

/// How far flat water's authored level wanders from cell to cell, in yards: free to climb down.
const LEVEL_NOISE_YD: f32 = 0.05;

/// A surface falling more than this (rise over run) is authored running that way: the current
/// never runs up it.
const TRUST_GRADE: f32 = 0.03;

/// The current in a channel [`REF_HALF_WIDTH`] cells half-wide, yards a second.
const RIVER_SPEED: f32 = 2.0;

/// The half-width, in cells, that runs at [`RIVER_SPEED`]: a 25-yd river.
const REF_HALF_WIDTH: f32 = 3.0;

/// Narrow water runs at most this much faster than [`RIVER_SPEED`], wide water at least this
/// fraction of it before it stills.
const NARROW_GAIN_MAX: f32 = 1.5;
const WIDE_GAIN_MIN: f32 = 0.3;

/// Water this many cells half-wide starts to still, and at [`STILL_HALF_WIDTH`] it is still: a
/// lake has no current worth drawing.
const SLOW_HALF_WIDTH: f32 = 7.0;
const STILL_HALF_WIDTH: f32 = 12.0;

/// Water within this many cells (33 yd) of still water at its own level, over steps under
/// [`LEVEL_STEP_YD`], is still with it, and it gathers its full speed over as many again.
const CALM_REACH: f32 = 8.0;

/// How much faster the current runs per unit of surface grade along it (rise over run): a 12%
/// ramp runs at twice the flat speed.
const GRADE_GAIN: f32 = 8.0;

/// The fastest any water runs, yards a second.
const FLOW_MAX: f32 = 6.0;

/// The share of the mid-channel speed left at the bank.
const BANK_SPEED: f32 = 0.35;

/// Averaging passes over the direction.
const SMOOTH_PASSES: usize = 3;

/// The 8 neighbours as lattice steps, with their lengths in cells: the four axes first, each
/// followed by its opposite.
const EIGHT: [((i32, i32), f32); 8] = [
    ((1, 0), 1.0),
    ((-1, 0), 1.0),
    ((0, 1), 1.0),
    ((0, -1), 1.0),
    ((1, 1), std::f32::consts::SQRT_2),
    ((1, -1), std::f32::consts::SQRT_2),
    ((-1, 1), std::f32::consts::SQRT_2),
    ((-1, -1), std::f32::consts::SQRT_2),
];

/// No water there: a dry or magma neighbour, or none.
const DRY: u32 = u32::MAX;
/// The sea there.
const SEA: u32 = u32::MAX - 1;

/// The water cell one lattice step `d` from `k`: a diagonal through either side's water, so the
/// current never cuts a dry corner.
fn stepped(g: &Grid, links: &[[u32; 4]], k: usize, (da, db): (i32, i32)) -> Option<usize> {
    let n = if da == 0 || db == 0 {
        g.step(links, k, (da, db))
    } else {
        g.step(links, k, (da, 0))
            .and_then(|m| g.step(links, m, (0, db)))
            .or_else(|| {
                g.step(links, k, (0, db))
                    .and_then(|m| g.step(links, m, (da, 0)))
            })
    };
    n.filter(|&n| g.water[n])
}

/// The water that is not sea, packed: its grid cells, heights, bank distances and 8 neighbours
/// (a river index, [`SEA`] or [`DRY`], in [`EIGHT`]'s order).
struct Rivers {
    cells: Vec<usize>,
    h: Vec<f32>,
    shore: Vec<f32>,
    near: Vec<[u32; 8]>,
}

impl Rivers {
    fn build(g: &Grid, shore: &[u32]) -> Self {
        let links = g.links();
        let river = |k: usize| g.water[k] && !g.ocean[k];
        let cells: Vec<usize> = g.cells().filter(|&k| river(k)).collect();
        let mut of = std::collections::HashMap::with_capacity(cells.len());
        for (i, &k) in cells.iter().enumerate() {
            of.insert(k, i as u32);
        }
        let near = cells
            .iter()
            .map(|&k| {
                EIGHT.map(|(d, _)| match stepped(g, &links, k, d) {
                    Some(m) if g.ocean[m] => SEA,
                    Some(m) => of[&m],
                    None => DRY,
                })
            })
            .collect();
        Rivers {
            h: cells.iter().map(|&k| g.h[k]).collect(),
            shore: cells.iter().map(|&k| shore[k].max(1) as f32).collect(),
            cells,
            near,
        }
    }

    fn len(&self) -> usize {
        self.cells.len()
    }

    /// The river neighbours of `i`, the four axes only or all eight.
    fn rivers(&self, i: usize, eight: bool) -> impl Iterator<Item = u32> + '_ {
        self.near[i][..if eight { 8 } else { 4 }]
            .iter()
            .copied()
            .filter(|&m| m < SEA)
    }

    /// Connected pieces over the four axes where `link` allows: each cell's piece and the pieces.
    fn pieces(&self, link: impl Fn(usize, usize) -> bool) -> (Vec<u32>, Vec<Vec<u32>>) {
        let mut id = vec![u32::MAX; self.len()];
        let mut out: Vec<Vec<u32>> = Vec::new();
        for start in 0..self.len() {
            if id[start] != u32::MAX {
                continue;
            }
            let n = out.len() as u32;
            id[start] = n;
            let mut cells = vec![start as u32];
            let mut q = 0;
            while q < cells.len() {
                let k = cells[q] as usize;
                q += 1;
                for m in self.rivers(k, false) {
                    if id[m as usize] == u32::MAX && link(k, m as usize) {
                        id[m as usize] = n;
                        cells.push(m);
                    }
                }
            }
            out.push(cells);
        }
        (id, out)
    }

    /// Central differences of `v` over the four axes, one-sided at an edge: (d/dX, d/dY) per cell.
    fn gradient(&self, i: usize, v: impl Fn(u32) -> Option<f32>) -> [f32; 2] {
        let here = v(i as u32);
        let axis = |plus: usize, minus: usize| {
            let at = |s: usize| Some(self.near[i][s]).filter(|&m| m < SEA).and_then(&v);
            match (at(plus), at(minus), here) {
                (Some(p), Some(m), _) => (p - m) * 0.5,
                (Some(p), None, Some(c)) => p - c,
                (None, Some(m), Some(c)) => c - m,
                _ => 0.0,
            }
        };
        [axis(0, 1), axis(2, 3)]
    }
}

/// Per cell, the current in WoW X and Y, yards a second; zero on the sea, still water, magma and
/// dry slots. `shore` is each cell's distance from dry land in cells, 1 at the bank.
pub(crate) fn currents(g: &Grid, shore: &[u32]) -> Vec<[f32; 2]> {
    let r = Rivers::build(g, shore);
    let n = r.len();
    let step_cost = |i: usize, s: usize| EIGHT[s].1 * (1.0 + CENTRE_BIAS / r.shore[i]);

    // Where water ends.
    let mut dist = vec![f32::INFINITY; n];
    for (i, near) in r.near.iter().enumerate() {
        for (s, &m) in near.iter().enumerate() {
            if m == SEA {
                dist[i] = dist[i].min(step_cost(i, s));
            }
        }
    }
    let (stretch, stretches) = r.pieces(|a, b| (r.h[a] - r.h[b]).abs() <= FLAT_STEP_YD);
    let mut band = vec![u32::MAX; n];
    let mut end_in = |id: u32, cells: &[u32], dist: &mut [f32]| {
        let low = cells
            .iter()
            .map(|&k| r.h[k as usize])
            .fold(f32::MAX, f32::min);
        let inside: Vec<u32> = cells
            .iter()
            .copied()
            .filter(|&k| r.h[k as usize] <= low + LAKE_TOL_YD)
            .collect();
        for &k in &inside {
            band[k as usize] = id;
        }
        // Walk the band from where water comes into it; it ends at the far end.
        let mut far = vec![u32::MAX; inside.len()];
        let at: std::collections::HashMap<u32, usize> =
            inside.iter().enumerate().map(|(j, &k)| (k, j)).collect();
        let mut queue = VecDeque::new();
        for (j, &k) in inside.iter().enumerate() {
            if r.rivers(k as usize, false).any(|m| band[m as usize] != id) {
                far[j] = 0;
                queue.push_back(k);
            }
        }
        let mut reach = 0;
        while let Some(k) = queue.pop_front() {
            let d = far[at[&k]];
            reach = reach.max(d);
            for m in r.rivers(k as usize, false) {
                if let Some(&j) = at.get(&m) {
                    if far[j] == u32::MAX {
                        far[j] = d + 1;
                        queue.push_back(m);
                    }
                }
            }
        }
        for (j, &k) in inside.iter().enumerate() {
            if far[j] == u32::MAX || far[j] + 1 >= reach {
                dist[k as usize] = 0.0;
            }
        }
    };
    for (s, cells) in stretches.iter().enumerate() {
        let lower_way_out = cells.iter().any(|&k| {
            let k = k as usize;
            r.near[k].iter().any(|&m| {
                m == SEA
                    || (m < SEA
                        && stretch[m as usize] != s as u32
                        && r.h[m as usize] < r.h[k] - FLAT_STEP_YD)
            })
        });
        if lower_way_out {
            continue;
        }
        // A dead end: higher water on one side of a cell and dry land straight across from it.
        let dead_end = || {
            cells.iter().any(|&k| {
                let near = &r.near[k as usize];
                (0..4).any(|a| {
                    let m = near[a];
                    m < SEA
                        && r.h[m as usize] > r.h[k as usize] + FLAT_STEP_YD
                        && near[a ^ 1] == DRY
                })
            })
        };
        if cells.len() >= BASIN_MIN_CELLS || dead_end() {
            end_in(s as u32, cells, &mut dist);
        }
    }
    let (_, waters) = r.pieces(|_, _| true);
    for (w, cells) in waters.iter().enumerate() {
        let sea = cells
            .iter()
            .any(|&k| r.near[k as usize][..4].contains(&SEA));
        if !sea {
            end_in((stretches.len() + w) as u32, cells, &mut dist);
        }
    }

    // How far upstream: Dijkstra out of those, walking against the current. Distances are
    // non-negative, so their bits order as the floats do.
    let mut heap: BinaryHeap<Reverse<(u32, u32)>> = (0..n)
        .filter(|&i| dist[i].is_finite())
        .map(|i| Reverse((dist[i].to_bits(), i as u32)))
        .collect();
    while let Some(Reverse((bits, k))) = heap.pop() {
        let (d, k) = (f32::from_bits(bits), k as usize);
        if d > dist[k] {
            continue;
        }
        for (s, &m) in r.near[k].iter().enumerate() {
            if m >= SEA {
                continue;
            }
            let m = m as usize;
            let e = d + step_cost(m, s) + DOWN_COST * (r.h[k] - r.h[m] - LEVEL_NOISE_YD).max(0.0);
            if e < dist[m] {
                dist[m] = e;
                heap.push(Reverse((e.to_bits(), m as u32)));
            }
        }
    }

    // The direction: down the distance's gradient, in lattice (WoW X, Y) axes, never up a surface
    // that is authored falling the other way.
    let downhill = |i: usize| {
        let [hx, hy] = r.gradient(i, |m| Some(r.h[m as usize]));
        let grade = hx.hypot(hy) / g.cell;
        let unit = if grade > 1e-6 {
            [-hx / (grade * g.cell), -hy / (grade * g.cell)]
        } else {
            [0.0, 0.0]
        };
        (unit, grade)
    };
    let slope: Vec<([f32; 2], f32)> = (0..n).map(downhill).collect();
    let trusted = |i: usize, d: [f32; 2]| {
        let (down, grade) = slope[i];
        if grade > TRUST_GRADE && d[0] * down[0] + d[1] * down[1] < 0.0 {
            down
        } else {
            d
        }
    };
    let mut dir = vec![[0.0f32; 2]; n];
    for (i, d) in dir.iter_mut().enumerate() {
        if !dist[i].is_finite() {
            continue;
        }
        let [gx, gy] = r.gradient(i, |m| Some(dist[m as usize]).filter(|d| d.is_finite()));
        let len = gx.hypot(gy);
        if len > 1e-6 {
            *d = trusted(i, [-gx / len, -gy / len]);
        }
    }
    for _ in 0..SMOOTH_PASSES {
        let next = (0..n)
            .map(|i| {
                let mut s = [2.0 * dir[i][0], 2.0 * dir[i][1]];
                let mut w = 2.0;
                for m in r.rivers(i, true) {
                    s[0] += dir[m as usize][0];
                    s[1] += dir[m as usize][1];
                    w += 1.0;
                }
                [s[0] / w, s[1] / w]
            })
            .collect();
        dir = next;
    }

    // The half-width: the deepest water each cell's own water rises to, carried down the bank
    // distance from the middle of every channel and lake, over steps under [`LEVEL_STEP_YD`] only,
    // so a ramp out of a lake is as narrow as its own banks.
    let mut half = r.shore.clone();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by(|&a, &b| r.shore[b].total_cmp(&r.shore[a]));
    for &k in &order {
        for m in r.rivers(k, false) {
            let m = m as usize;
            if r.shore[m] < r.shore[k] && (r.h[m] - r.h[k]).abs() <= LEVEL_STEP_YD {
                half[m] = half[m].max(half[k]);
            }
        }
    }

    // How far each cell is from still water it is level with: a lake's bays and the water between
    // its islands are as still as the lake, however narrow, and a river leaving a lake gathers
    // speed over its first yards.
    let mut calm = vec![u32::MAX; n];
    let mut queue: VecDeque<usize> = (0..n).filter(|&i| half[i] >= STILL_HALF_WIDTH).collect();
    for &i in &queue {
        calm[i] = 0;
    }
    while let Some(k) = queue.pop_front() {
        if calm[k] >= 2 * CALM_REACH as u32 {
            continue;
        }
        for m in r.rivers(k, false) {
            let m = m as usize;
            if calm[m] == u32::MAX && (r.h[m] - r.h[k]).abs() <= LEVEL_STEP_YD {
                calm[m] = calm[k] + 1;
                queue.push_back(m);
            }
        }
    }

    let mut out = vec![[0.0f32; 2]; g.h.len()];
    for i in 0..n {
        let [dx, dy] = dir[i];
        // A disagreeing neighbourhood averages short: a watershed.
        let coherence = dx.hypot(dy);
        if coherence < 1e-3 || !dist[i].is_finite() {
            continue;
        }
        let hw = half[i];
        let width_gain = (REF_HALF_WIDTH / hw).clamp(WIDE_GAIN_MIN, NARROW_GAIN_MAX)
            * (1.0 - smoothstep(SLOW_HALF_WIDTH, STILL_HALF_WIDTH, hw))
            * smoothstep(CALM_REACH, 2.0 * CALM_REACH, calm[i] as f32);
        if width_gain <= 0.0 {
            continue;
        }
        let mut d = trusted(i, [dx / coherence, dy / coherence]);
        // Along the bank, not into it: the part across the bank goes, all of it at the bank and
        // none mid-channel.
        let across = (1.0 - (r.shore[i] - 0.5) / hw).clamp(0.0, 1.0);
        let [sx, sy] = r.gradient(i, |m| Some(r.shore[m as usize]));
        let sl = sx.hypot(sy);
        if sl > 1e-6 && across > 0.0 {
            let (nx, ny) = (sx / sl, sy / sl);
            let dot = d[0] * nx + d[1] * ny;
            let turned = [d[0] - across * dot * nx, d[1] - across * dot * ny];
            let tl = turned[0].hypot(turned[1]);
            if tl > 1e-3 {
                d = [turned[0] / tl, turned[1] / tl];
            }
        }
        let (down, grade) = slope[i];
        let along = (d[0] * down[0] + d[1] * down[1]).max(0.0) * grade;
        let bank = ((r.shore[i] - 0.5) / hw).clamp(0.0, 1.0);
        let profile = BANK_SPEED + (1.0 - BANK_SPEED) * (1.0 - (1.0 - bank) * (1.0 - bank));
        let speed = (RIVER_SPEED * width_gain * (1.0 + GRADE_GAIN * along) * profile).min(FLOW_MAX)
            * coherence;
        out[r.cells[i]] = [d[0] * speed, d[1] * speed];
    }
    out
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}
