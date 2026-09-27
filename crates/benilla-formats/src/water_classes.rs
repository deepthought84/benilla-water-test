//! Which water the planar mirrors serve and which the cube probes do, per 4.17-yd liquid cell: the
//! shape the improved water's classification hands the base liquid code, which bakes it into the
//! surface meshes and the mirror election without knowing how it was decided.

use crate::LiquidMesh;

/// No probe spot.
pub const NO_SPOT: u16 = u16::MAX;

/// One cell's answer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CellWater {
    /// The mirror plane it votes, NaN where it votes none (probe water, magma).
    pub plane: f32,
    /// How far the mirrors serve it, 0 (probe water) to 1.
    pub weight: f32,
    /// The probe spots it reads, lower id first, [`NO_SPOT`] where none.
    pub spots: [u16; 2],
    /// The second spot's share, 0 to 1.
    pub mix: f32,
}

impl CellWater {
    /// No water here, or none the classification touches.
    pub const DRY: CellWater = CellWater {
        plane: f32::NAN,
        weight: 1.0,
        spots: [NO_SPOT, NO_SPOT],
        mix: 0.0,
    };
}

/// A fixed probe spot: where a probe stands for its region.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProbeSpot {
    /// Position on the water, in the batch's coordinates (absolute WoW yards for a map).
    pub at: [f32; 3],
}

/// A map's (or a WMO placement's) water classification.
pub trait WaterClasses: Send + Sync + 'static {
    /// Per cell of `lq`, its answer; [`CellWater::DRY`] where there is none.
    fn cells(&self, lq: &LiquidMesh) -> Vec<CellWater>;

    /// The mirrors' weight at a lattice corner, 1 with no water there.
    fn corner(&self, x: f32, y: f32) -> f32;

    /// The probe spots, indexed by the ids cells carry.
    fn spots(&self) -> &[ProbeSpot];

    /// Per cell of `lq`, the mirror plane it votes, NaN where none.
    fn planes(&self, lq: &LiquidMesh) -> Vec<f32> {
        self.cells(lq).iter().map(|c| c.plane).collect()
    }
}
