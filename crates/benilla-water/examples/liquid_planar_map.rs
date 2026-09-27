//! Classify a whole map's water with `PlanarMap::build`, as the game does at first load, and
//! write one record per wet MCLQ cell: centre X, Y, mean height, corner range, plane (NaN: not a
//! mirror), mirror weight (little-endian f32 × 6), then both probe spot ids (u16 × 2); a second
//! file lists the spots (f32 × 3 each), a third each MCNK's north-west corner X, Y (f32) and top
//! zone id (u32) then its 8 × 8 terrain heights at the cell centres (f32 × 64, NaN without MCVT),
//! and a text file the zone names. Prints the timings and the cache size. Output is
//! Blizzard data: never commit it.
//! `cargo run --release -p benilla-water --example liquid_planar_map -- <map> <out-prefix>`

use std::io::Write;
use std::time::Instant;

use benilla_formats::{adt_liquids, LiquidMesh};
use benilla_water::PlanarMap;

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let (map, out) = (a[0].clone(), a[1].clone());
    let data = benilla_formats::wow_data().expect("no WoW install found (set $WOW_DATA)");
    let mut chain = benilla_formats::open_chain(&data)?;
    let tiles = benilla_formats::MapTiles::load(&mut chain, &map)?;

    let t0 = Instant::now();
    let mut water: Vec<LiquidMesh> = Vec::new();
    let mut chunk_zones: Vec<([f32; 2], u32, [f32; 64])> = Vec::new();
    let mut bytes_all = Vec::new();
    for (tx, ty) in tiles.existing_in_radius(0.0, 0.0, 32) {
        let path = format!("World\\Maps\\{map}\\{map}_{tx}_{ty}.adt");
        if let Ok(bytes) = chain.read_file(&path) {
            water.extend(adt_liquids(&bytes).unwrap_or_default());
            bytes_all.push(bytes);
        }
    }
    let t_read = t0.elapsed();
    let areas = benilla_formats::load_area_table_catalog(&mut chain)?;
    let mut names = std::collections::BTreeMap::new();
    for bytes in &bytes_all {
        let mut cur = std::io::Cursor::new(bytes.as_slice());
        let Ok(benilla_adt::ParsedAdt::Root(root)) = benilla_adt::parse_adt(&mut cur) else {
            continue;
        };
        for mcnk in &root.mcnk_chunks {
            let zone = areas
                .top_zone(mcnk.header.area_id)
                .unwrap_or(mcnk.header.area_id);
            names.insert(zone, areas.name(zone).unwrap_or("?").to_string());
            let [x, y, z] = mcnk.header.position;
            let mut ground = [f32::NAN; 64];
            if let Some(v) = mcnk.heights.as_ref().filter(|v| v.heights.len() >= 145) {
                for (c, g) in ground.iter_mut().enumerate() {
                    *g = z + v.heights[(c / 8) * 17 + 9 + c % 8];
                }
            }
            chunk_zones.push(([x, y], zone, ground));
        }
    }
    let t1 = Instant::now();
    let pm = PlanarMap::build(water.iter()).expect("water");
    let t_class = t1.elapsed();
    let cache = pm.to_bytes(0).len();

    let mut file = std::io::BufWriter::new(std::fs::File::create(format!("{out}.bin"))?);
    let mut cells = 0usize;
    for lq in &water {
        let (cols, rows) = (lq.grid[0] as usize, lq.grid[1] as usize);
        let answers = pm.cells(lq);
        if answers.len() != (cols - 1) * (rows - 1) {
            continue;
        }
        for (c, wet) in lq.wet.iter().enumerate() {
            if !*wet {
                continue;
            }
            let (i, j) = (c % (cols - 1), c / (cols - 1));
            let p = [(i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1)]
                .map(|(a, b)| lq.positions[b * cols + a]);
            let z: Vec<f32> = p.iter().map(|q| q[2]).filter(|z| z.abs() < 1e8).collect();
            if z.is_empty() {
                continue;
            }
            let x = p.iter().map(|q| q[0]).sum::<f32>() / 4.0;
            let y = p.iter().map(|q| q[1]).sum::<f32>() / 4.0;
            let h = z.iter().sum::<f32>() / z.len() as f32;
            let range = z.iter().copied().fold(f32::MIN, f32::max)
                - z.iter().copied().fold(f32::MAX, f32::min);
            let w = answers[c];
            for v in [x, y, h, range, w.plane, w.weight] {
                file.write_all(&v.to_le_bytes())?;
            }
            file.write_all(&w.spots[0].to_le_bytes())?;
            file.write_all(&w.spots[1].to_le_bytes())?;
            cells += 1;
        }
    }
    file.flush()?;
    let mut zones = std::io::BufWriter::new(std::fs::File::create(format!("{out}-zones.bin"))?);
    for ([x, y], z, ground) in &chunk_zones {
        zones.write_all(&x.to_le_bytes())?;
        zones.write_all(&y.to_le_bytes())?;
        zones.write_all(&z.to_le_bytes())?;
        for g in ground {
            zones.write_all(&g.to_le_bytes())?;
        }
    }
    zones.flush()?;
    std::fs::write(
        format!("{out}-zones.txt"),
        names
            .iter()
            .map(|(id, n)| format!("{id}\t{n}\n"))
            .collect::<String>(),
    )?;
    let mut spots = std::io::BufWriter::new(std::fs::File::create(format!("{out}-spots.bin"))?);
    for s in pm.spots() {
        for v in s.at {
            spots.write_all(&v.to_le_bytes())?;
        }
    }
    spots.flush()?;
    eprintln!(
        "{map}: {} liquid blocks, {cells} cells, {} spots; read+parse {:.1} s, classify {:.1} s, \
         cache {:.1} MB",
        water.len(),
        pm.spots().len(),
        t_read.as_secs_f32(),
        t_class.as_secs_f32(),
        cache as f32 / 1e6
    );
    Ok(())
}
