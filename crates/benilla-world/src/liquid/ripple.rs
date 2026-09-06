//! The stylised water's ripple map — a generated, seamlessly tiling slope texture.
//!
//! **This is not the reference's water and does not pretend to be.** Everything else under
//! `liquid/` reproduces what the 1.12 client does, byte by byte where the bytes are known; this
//! file exists to feed the *alternative* look the `waterStyle` CVar selects (see
//! [`super::WaterStyle`]), and its numbers answer to the eye rather than to `WoW.exe`. The
//! reference's own animated sheet (`lake_a`/`ocean_h`) is untouched and still drives the faithful
//! path — the two share a material and a draw, and nothing here can reach the reference look.
//!
//! The map holds the *slope* of a smooth noise height field in R/G and the height itself in B:
//!
//! - **R/G — slope.** The shader rebuilds a world-space normal as `vec3(r, 1, g)` and needs no
//!   tangent frame to do it, because a liquid surface is a flat, axis-aligned, never-rotating
//!   plane. Signed values live around 0.5 in an unorm texture, which is also what makes the mip
//!   chain correct: the average of encoded slopes is the encoding of the average slope, i.e.
//!   distant water flattens, which is exactly what it should do.
//! - **B — the height.** A second, uncorrelated scalar field the shore foam masks itself with,
//!   free: it is the field the slopes were differentiated from.
//!
//! Generated rather than shipped for the same reason the rest of benilla generates nothing: there
//! are no bundled art assets in this project, and a stylised look that needed one would have to
//! ship one. Perlin over a *wrapped lattice* is what makes it tile — a ripple map with a seam
//! draws a grid across the whole ocean, once every twelve yards.
//!
//! The mip chain is not optional. Without one the GPU point-samples a 256² map through pixels
//! that, out toward the horizon, each cover many tiles of it, and the beat between the two grids
//! is a moiré that crawls over distant water. (The reference's own sheet ships nine authored mips
//! whose alpha deliberately dies with distance, for the same reason — see `liquid.wgsl`.)

use bevy::asset::RenderAssetUsages;
use bevy::image::{ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor};
use bevy::math::Vec2;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};

/// Edge length of the generated map. Small on purpose: the shader samples it at three different
/// world scales, so its own texel grid never becomes the visible detail.
const SIZE: usize = 256;

/// The octave ladder: `(lattice period, amplitude)`. Every period divides [`SIZE`], so each layer
/// tiles on its own and the sum tiles with them.
const OCTAVES: &[(i32, f32)] = &[(4, 1.0), (8, 0.5), (16, 0.28), (32, 0.14)];

/// Perlin noise that tiles seamlessly with the given lattice period.
///
/// Ordinary fbm does not tile. Wrapping the *lattice coordinates* — rather than cross-fading the
/// edges, which softens the whole texture — makes the repeat exact: the gradient is looked up at
/// the wrapped lattice point while the distance vector uses the unwrapped one, and mixing those
/// two up is what tears the noise at the seam.
fn tileable_perlin(p: Vec2, period: i32, seed: u32) -> f32 {
    const R: f32 = std::f32::consts::FRAC_1_SQRT_2;
    const DIRS: [(f32, f32); 8] = [
        (1.0, 0.0),
        (-1.0, 0.0),
        (0.0, 1.0),
        (0.0, -1.0),
        (R, R),
        (-R, R),
        (R, -R),
        (-R, -R),
    ];

    let xi = p.x.floor();
    let zi = p.y.floor();
    let xf = p.x - xi;
    let zf = p.y - zi;
    let wrap = |v: i32| v.rem_euclid(period);
    let (x0, z0) = (wrap(xi as i32), wrap(zi as i32));
    let (x1, z1) = (wrap(x0 + 1), wrap(z0 + 1));

    let g = |gx: i32, gz: i32, dx: f32, dz: f32| {
        let mut h = (gx as u32)
            .wrapping_mul(0x9E37_79B9)
            .wrapping_add((gz as u32).wrapping_mul(0x85EB_CA6B))
            .wrapping_add(seed.wrapping_mul(0xC2B2_AE35));
        h ^= h >> 15;
        h = h.wrapping_mul(0x2545_F491);
        h ^= h >> 13;
        let (ux, uz) = DIRS[(h & 7) as usize];
        ux * dx + uz * dz
    };

    let fade = |t: f32| t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
    let (u, v) = (fade(xf), fade(zf));

    let n00 = g(x0, z0, xf, zf);
    let n10 = g(x1, z0, xf - 1.0, zf);
    let n01 = g(x0, z1, xf, zf - 1.0);
    let n11 = g(x1, z1, xf - 1.0, zf - 1.0);
    let a = n00 + u * (n10 - n00);
    let b = n01 + u * (n11 - n01);
    a + v * (b - a)
}

/// The summed octaves, normalised to ±0.5.
fn height_field(seed: u32) -> Vec<f32> {
    let mut height = vec![0.0f32; SIZE * SIZE];
    let mut peak = 0.0f32;
    for z in 0..SIZE {
        for x in 0..SIZE {
            let uv = Vec2::new(x as f32 / SIZE as f32, z as f32 / SIZE as f32);
            let mut h = 0.0;
            for &(period, amp) in OCTAVES {
                h += tileable_perlin(uv * period as f32, period, seed + period as u32) * amp;
            }
            height[z * SIZE + x] = h;
            peak = peak.max(h.abs());
        }
    }
    let scale = if peak > 0.0 { 0.5 / peak } else { 1.0 };
    for h in &mut height {
        *h *= scale;
    }
    height
}

/// Box-filter one level down to half its edge. A 2×2 box never crosses the edge, so the wrap the
/// generator went to such lengths for survives the chain untouched.
fn halve(level: &[u8], n: usize) -> Vec<u8> {
    let half = n / 2;
    let mut next = vec![0u8; half * half * 4];
    for z in 0..half {
        for x in 0..half {
            for c in 0..4 {
                let at =
                    |dx: usize, dz: usize| level[((z * 2 + dz) * n + x * 2 + dx) * 4 + c] as u32;
                next[(z * half + x) * 4 + c] =
                    ((at(0, 0) + at(1, 0) + at(0, 1) + at(1, 1) + 2) / 4) as u8;
            }
        }
    }
    next
}

/// Build the ripple map: R/G = wrapped central-difference slope, B = the height it came from,
/// with a complete box-filtered mip chain and a repeating trilinear sampler.
///
/// `Rgba8Unorm`, never `Srgb`: this holds geometry, not colour, and gamma-decoding a slope bends
/// it.
pub(super) fn ripple_map(seed: u32) -> Image {
    let height = height_field(seed);
    let at = |x: usize, z: usize| height[(z % SIZE) * SIZE + (x % SIZE)];

    let mut mip0 = vec![0u8; SIZE * SIZE * 4];
    for z in 0..SIZE {
        for x in 0..SIZE {
            // Central differences, wrapping — the slope has to tile as well as the height does.
            let dx = at(x + 1, z) - at((x + SIZE - 1) % SIZE, z);
            let dz = at(x, z + 1) - at(x, (z + SIZE - 1) % SIZE);
            let i = (z * SIZE + x) * 4;
            mip0[i] = ((dx * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u8;
            mip0[i + 1] = ((dz * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u8;
            mip0[i + 2] = ((at(x, z) + 0.5).clamp(0.0, 1.0) * 255.0) as u8;
            mip0[i + 3] = 255;
        }
    }

    let mut chain = mip0.clone();
    let mut level = mip0;
    let mut n = SIZE;
    let mut levels = 1;
    while n > 1 {
        level = halve(&level, n);
        n /= 2;
        chain.extend_from_slice(&level);
        levels += 1;
    }

    let mut image = Image::new_uninit(
        Extent3d {
            width: SIZE as u32,
            height: SIZE as u32,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        TextureFormat::Rgba8Unorm,
        // Nothing reads it back: it is generated once at liquid setup and handed to the material
        // as a handle, exactly like the decoded frame arrays beside it.
        RenderAssetUsages::RENDER_WORLD,
    );
    image.data = Some(chain);
    image.texture_descriptor.mip_level_count = levels;
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        address_mode_u: ImageAddressMode::Repeat,
        address_mode_v: ImageAddressMode::Repeat,
        mag_filter: ImageFilterMode::Linear,
        min_filter: ImageFilterMode::Linear,
        mipmap_filter: ImageFilterMode::Linear,
        // No anisotropy, deliberately, and unlike the sheet beside it: the reference's own filter
        // policy is a process global this map has no business reading, and the measurement that
        // settled it upstream (the sandbox this look was ported from) was 3 % of the pixels of a
        // waterside view changed by more than one level for a quarter of the frame rate.
        ..default()
    });
    image
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The wrapped lattice is the whole tiling mechanism: one period out has to land on the same
    /// value, or the repeat draws a seam across the water every twelve yards.
    #[test]
    fn perlin_tiles() {
        for i in 0..16 {
            let p = Vec2::new(i as f32 * 0.37, i as f32 * 0.11);
            let here = tileable_perlin(p, 8, 1234);
            for step in [Vec2::new(8.0, 0.0), Vec2::new(0.0, 8.0), Vec2::splat(8.0)] {
                let there = tileable_perlin(p + step, 8, 1234);
                assert!(
                    (here - there).abs() < 1e-5,
                    "seam at {p:?} + {step:?}: {here} vs {there}"
                );
            }
        }
    }

    /// A complete chain down to 1x1, and exactly the bytes the descriptor promises — a short
    /// chain is a validation error at bind time, and a long one is silently sampled garbage.
    #[test]
    fn mip_chain_is_complete() {
        let image = ripple_map(7);
        let levels = image.texture_descriptor.mip_level_count;
        assert_eq!(levels, 9, "256 down to 1 is nine levels");
        let expected: usize = (0..levels)
            .map(|l| {
                let n = SIZE >> l;
                n * n * 4
            })
            .sum();
        assert_eq!(image.data.as_ref().map(Vec::len), Some(expected));
    }
}
