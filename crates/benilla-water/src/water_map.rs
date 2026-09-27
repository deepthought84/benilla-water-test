//! A map's whole water classification ([`PlanarMap`]): which water the planar
//! mirrors serve and where the probes stand. Built once per map from every tile's MCLQ at its
//! first tile load, then read back from the state folder's cache, keyed on the install.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, LazyLock, Mutex, OnceLock};
use std::time::Instant;

use benilla_formats::{adt_liquids, Chain, LiquidMesh, WaterClasses};

use crate::PlanarMap;
use bevy::prelude::*;

static STATE_DIR: OnceLock<Option<PathBuf>> = OnceLock::new();

/// One cell per map; the first tile load of a map builds it and every other waits on it.
type MapCell = Arc<OnceLock<Option<Arc<PlanarMap>>>>;
static MAPS: LazyLock<Mutex<HashMap<String, MapCell>>> = LazyLock::new(Default::default);

/// Where the classification is cached: the state folder, or `None` to keep nothing (a capture).
pub(crate) fn set_state_dir(dir: Option<PathBuf>) {
    let _ = STATE_DIR.set(dir);
}

/// The classification of `map` (its `World\Maps` directory name), built or read on first use.
pub(crate) fn water_map(chain: &Chain, map: &str) -> Option<Arc<dyn WaterClasses>> {
    let cell = MAPS
        .lock()
        .ok()?
        .entry(map.to_ascii_lowercase())
        .or_default()
        .clone();
    cell.get_or_init(|| build(chain, map))
        .clone()
        .map(|m| m as Arc<dyn WaterClasses>)
}

fn cache_path(map: &str) -> Option<PathBuf> {
    STATE_DIR.get().cloned().flatten().map(|d| {
        d.join("Cache")
            .join(format!("water-{}.bin", map.to_ascii_lowercase()))
    })
}

fn build(chain: &Chain, map: &str) -> Option<Arc<PlanarMap>> {
    let fingerprint = chain.fingerprint();
    let path = cache_path(map);
    if let Some(bytes) = path.as_ref().and_then(|p| std::fs::read(p).ok()) {
        if let Some(m) = PlanarMap::from_bytes(&bytes, fingerprint) {
            info!(
                "water: {map} classification from cache, {} probe spots",
                m.spots().len()
            );
            return Some(Arc::new(m));
        }
    }
    let t0 = Instant::now();
    let mut water: Vec<LiquidMesh> = Vec::new();
    for tx in 0..64 {
        for ty in 0..64 {
            let file = format!("World\\Maps\\{map}\\{map}_{tx}_{ty}.adt");
            if !chain.contains(&file) {
                continue;
            }
            if let Ok(bytes) = chain.read(&file) {
                water.extend(adt_liquids(&bytes).unwrap_or_default());
            }
        }
    }
    let read = t0.elapsed().as_secs_f32();
    let m = PlanarMap::build(water.iter())?;
    info!(
        "water: classified {map} in {:.1} s (read {read:.1} s), {} probe spots",
        t0.elapsed().as_secs_f32(),
        m.spots().len()
    );
    if let Some(p) = path {
        let tmp = p.with_extension("tmp");
        let written = p
            .parent()
            .map_or(Ok(()), std::fs::create_dir_all)
            .and_then(|()| std::fs::write(&tmp, m.to_bytes(fingerprint)))
            .and_then(|()| std::fs::rename(&tmp, &p));
        if let Err(e) = written {
            warn!("water: could not cache {map} at {}: {e}", p.display());
        }
    }
    Some(Arc::new(m))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// No two of the Eastern Kingdoms' probe spots stand on neighbouring cells: Loch Modan's dam
    /// basin placed two 4 yd apart, one cube's worth of water taking two of the live slots.
    #[test]
    fn no_two_probe_spots_stand_on_neighbouring_cells() {
        let data = benilla_formats::wow_data_or_skip!();
        let chain = benilla_formats::open_chain(&data).expect("the install's patch chain");
        let map = water_map(&chain, "Azeroth").expect("Azeroth's water");
        let apart = crate::liquid_planar::SAME_SPOT_CELLS * 100.0 / 3.0 / 8.0;
        let spots = map.spots();
        for (i, a) in spots.iter().enumerate() {
            for b in &spots[i + 1..] {
                let d = [a.at[0] - b.at[0], a.at[1] - b.at[1], a.at[2] - b.at[2]];
                let d = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                assert!(
                    d >= apart,
                    "spots at {:?} and {:?} are {d:.1} yd apart",
                    a.at,
                    b.at
                );
            }
        }
    }
}
