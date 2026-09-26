// The reflection march's hierarchical depth — a pyramid of NEAREST depths, one mip per halving.
//
// **Why a pyramid at all.** A linear march samples the depth buffer at a fixed cadence and is
// therefore blind between its samples: geometry thinner than a step is jumped over, and at a
// grazing angle the steps grow long enough that this happens constantly. The march in
// `liquid.wgsl` gave up on 13.7% of a grazing water surface at the Stranglethorn camera, and a
// fragment that gives up falls back to the sky — which at sunset is far brighter than the bank it
// should have reflected, so the failures are painted as bright bands. That is the artefact.
//
// A hierarchy removes the blindness rather than reducing it. Each texel of mip `n` holds the
// nearest surface anywhere in its 2^n by 2^n tile, so a ray that passes in front of that value has
// provably missed everything in the tile and can skip the whole tile in one step. A ray that does
// not can descend and ask a finer question. Nothing is jumped over, and the cost is logarithmic in
// the distance travelled instead of linear.
//
// **NEAREST, which is `max` here, and this is the one thing easy to get backwards.** Bevy is
// reverse-Z: the far plane is 0 and nearer surfaces have larger depth. Bevy ships its own depth
// pyramid in `bevy_core_pipeline::mip_generation`, and it reduces with `min` — the FURTHEST surface
// in each tile — because it is built for occlusion culling, where the safe error is to draw
// something that turns out to be hidden. A reflection march needs the opposite safety: it must
// never skip a tile that contains something, so it wants the nearest. That is why this exists
// rather than binding Bevy's.
//
// Sky is depth 0, the far plane, so it loses every `max` against real geometry and a tile holding
// both reports the geometry. That is the conservative answer and it needs no special case.
//
// **And the FARTHEST, into a second pyramid** (`min` here — the other way round from the first).
// The nearest proves a tile empty in FRONT of a ray; the farthest is what proves a ray has passed
// BEHIND everything in a tile, once each surface is given a thickness — see `liquid::hiz`'s image
// doc and `ssr_trace`. Sky, at 0, wins every `min`, so a tile holding any sky can never be passed
// behind: the ray has to descend and find it.

#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

// Three entry points, three bindings, and deliberately no sampler: every read here is a
// `textureLoad` of an exact texel, because a filtered value between two depths is a depth where
// nothing stands. Each entry point declares only its own binding at its own index — two entry
// points sharing a running count is how one of them was handed the other's type.
@group(0) @binding(0) var src: texture_depth_2d;
@group(0) @binding(1) var lvl: texture_2d<f32>;
@group(0) @binding(2) var src_ms: texture_depth_multisampled_2d;
// The farthest-depth pyramid's level above, beside `lvl` (the nearest's) — see `liquid::hiz`.
@group(0) @binding(3) var lvl_far: texture_2d<f32>;

// Every pass writes both pyramids: location 0 the nearest, location 1 the farthest.
struct HizOut {
    @location(0) near: f32,
    @location(1) far: f32,
};

// How many source texels per axis one pyramid texel may cover and still be read in full. The base
// is the previous power of two of the view, so the true ratio is below 2 and a footprint spans at
// most three texels; the bound is generous so a view whose size and the pyramid's ever disagree by
// more still reads every texel it covers rather than a sample of them.
const SEED_SPAN_MAX: i32 = 8;

// The source texels one pyramid texel covers, as an inclusive integer rectangle.
//
// **All of them, not the two nearest its centre.** The first seed read a fixed 2x2 around the
// texel's centre, which covers the footprint only when the ratio is exactly 2. Below that — which
// is always, the base being the previous power of two — a footprint can straddle three source
// texels per axis and the 2x2 skipped one, so geometry one or two pixels wide (a trunk, a limb, a
// pole seen from across a lake) could drop out of the pyramid entirely and the march would walk
// straight through it. A nearest-depth pyramid is only conservative if its base is.
//
// The destination texel's size in UV comes from the derivative of the fullscreen UV, which is
// exactly one over the target's size on each axis.
fn seed_rect(uv: vec2<f32>, dims: vec2<f32>) -> vec4<i32> {
    let half_texel = 0.5 * vec2<f32>(abs(dpdx(uv.x)), abs(dpdy(uv.y)));
    let lo = vec2<i32>(floor((uv - half_texel) * dims));
    let hi = vec2<i32>(ceil((uv + half_texel) * dims)) - vec2<i32>(1);
    let top = vec2<i32>(dims) - vec2<i32>(1);
    let a = clamp(lo, vec2<i32>(0), top);
    let b = clamp(max(hi, lo), vec2<i32>(0), top);
    return vec4<i32>(a, min(b, a + vec2<i32>(SEED_SPAN_MAX - 1)));
}

/// Mip 0: the view's depth, resampled onto the pyramid's own grid as the NEAREST surface anywhere
/// in the texel's footprint — the same reduction as every level below it, for the same reason.
///
/// The pyramid is sized to the previous power of two so that every level is exactly half the one
/// above and the traversal's cell arithmetic is a shift rather than a division, which is what makes
/// this a resample rather than a copy.
@fragment
fn fs_seed(in: FullscreenVertexOutput) -> HizOut {
    let r = seed_rect(in.uv, vec2<f32>(textureDimensions(src)));
    // **Nine independent loads, not a loop.** The base is the previous power of two of the view,
    // so the ratio is below 2 and a footprint spans at most three texels per axis: a fixed 3x3
    // clamped into the rectangle covers it exactly (a texel read twice is harmless to a max or a
    // min). The loop below does the same work with data-dependent bounds, which serialised its
    // loads — it measured +0.17 ms of GPU at Mirror Lake against the 2x2 it replaced, and this is
    // back level with the 2x2 while reading the whole footprint.
    if (r.z - r.x <= 2 && r.w - r.y <= 2) {
        let x1 = min(r.x + 1, r.z);
        let x2 = min(r.x + 2, r.z);
        let y1 = min(r.y + 1, r.w);
        let y2 = min(r.y + 2, r.w);
        let a = textureLoad(src, vec2<i32>(r.x, r.y), 0);
        let b = textureLoad(src, vec2<i32>(x1, r.y), 0);
        let c = textureLoad(src, vec2<i32>(x2, r.y), 0);
        let e = textureLoad(src, vec2<i32>(r.x, y1), 0);
        let f = textureLoad(src, vec2<i32>(x1, y1), 0);
        let g = textureLoad(src, vec2<i32>(x2, y1), 0);
        let h = textureLoad(src, vec2<i32>(r.x, y2), 0);
        let k = textureLoad(src, vec2<i32>(x1, y2), 0);
        let m = textureLoad(src, vec2<i32>(x2, y2), 0);
        let near = max(max(max(a, b), max(c, e)), max(max(f, g), max(max(h, k), m)));
        let far = min(min(min(a, b), min(c, e)), min(min(f, g), min(min(h, k), m)));
        return HizOut(near, far);
    }
    // A view and pyramid whose sizes disagree by more than the power-of-two rule allows: still
    // every covered texel, just slower.
    var near = 0.0;
    var far = 1.0;
    for (var y = r.y; y <= r.w; y = y + 1) {
        for (var x = r.x; x <= r.z; x = x + 1) {
            let v = textureLoad(src, vec2<i32>(x, y), 0);
            near = max(near, v);
            far = min(far, v);
        }
    }
    return HizOut(near, far);
}

/// The same seed from a MULTISAMPLED depth, which is what the view's depth is when `gxMultisample`
/// is above one. Every sample of every covered texel: a sample is a piece of surface like any other.
@fragment
fn fs_seed_ms(in: FullscreenVertexOutput) -> HizOut {
    let r = seed_rect(in.uv, vec2<f32>(textureDimensions(src_ms)));
    let n = i32(textureNumSamples(src_ms));
    var near = 0.0;
    var far = 1.0;
    for (var y = r.y; y <= r.w; y = y + 1) {
        for (var x = r.x; x <= r.z; x = x + 1) {
            for (var i = 0; i < n; i = i + 1) {
                let v = textureLoad(src_ms, vec2<i32>(x, y), i);
                near = max(near, v);
                far = min(far, v);
            }
        }
    }
    return HizOut(near, far);
}

/// Every level below: the nearest of the four texels above, and the farthest.
@fragment
fn fs_reduce(in: FullscreenVertexOutput) -> HizOut {
    // `lvl` is a view of the level ABOVE, so its dimensions are the source's and the destination
    // is half of them. Taking the source's as the destination's — and then doubling on top — reads
    // off the end of the texture, which clamps to the edge and leaves every level below the base
    // holding one corner of the level above. That is what an empty pyramid looked like.
    let src_dims = vec2<i32>(textureDimensions(lvl));
    let dst_dims = max(src_dims / 2, vec2<i32>(1));
    let p = vec2<i32>(in.uv * vec2<f32>(dst_dims)) * 2;
    let hi = src_dims - vec2<i32>(1);
    let pa = clamp(p, vec2<i32>(0), hi);
    let pb = clamp(p + vec2<i32>(1, 0), vec2<i32>(0), hi);
    let pc = clamp(p + vec2<i32>(0, 1), vec2<i32>(0), hi);
    let pd = clamp(p + vec2<i32>(1, 1), vec2<i32>(0), hi);
    let near = max(
        max(textureLoad(lvl, pa, 0).r, textureLoad(lvl, pb, 0).r),
        max(textureLoad(lvl, pc, 0).r, textureLoad(lvl, pd, 0).r),
    );
    let far = min(
        min(textureLoad(lvl_far, pa, 0).r, textureLoad(lvl_far, pb, 0).r),
        min(textureLoad(lvl_far, pc, 0).r, textureLoad(lvl_far, pd, 0).r),
    );
    return HizOut(near, far);
}
