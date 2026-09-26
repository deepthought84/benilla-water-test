// The probe cube's prefilter: one mip level convolved from the one above it.
//
// **The probe stopped trying to be a mirror, and this is what it became instead.** A
// parallax-corrected cube is exact at its capture point and drifts with distance from it, and a
// long measurement pass established that the drift is inherent rather than unfinished — a sharper
// march, a bigger face and a world-space landing consensus each moved it barely or not at all. A
// reflection that is sharp and in the wrong place reads as a fault. The same reflection blurred
// reads as water. So the cube is filtered into a mip chain and the water picks a level by how far
// it stands from the probe, which is exactly where the error it is hiding comes from.
//
// This is a PROGRESSIVE filter: each level convolves the level above rather than the base. The
// level above is already half the resolution and already carries the previous level's blur, so a
// modest cone here compounds into a wide lobe over the chain, at a fraction of the taps a full
// convolution of mip 0 would need for the same width. It is the standard way to build this chain
// and the reason the whole thing costs a handful of small passes per captured slot.
//
// Bevy ships a true GGX importance-sampled prefilter in `bevy_pbr::light_probe`'s
// `environment_filter.wgsl`, and it is the better tool for a PBR specular response indexed by
// material roughness. It is not what this is: nothing here is asking "how rough is this surface",
// it is asking "how wrong is this probe here, and how much blur hides it". A cosine-weighted cone
// answers that and needs no blue-noise texture, no storage bindings and no compute pipeline — it
// renders into a mip with the same fullscreen blit the faces already use.

#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

@group(0) @binding(0) var cube: texture_cube_array<f32>;
@group(0) @binding(1) var cube_sampler: sampler;
@group(0) @binding(2) var<uniform> cfg: FilterCfg;

struct FilterCfg {
    // `x` = which of the six faces is being written, `y` = which probe slot, `z` = the cone's
    // half-angle in radians, `w` unused.
    face: vec4<f32>,
}

// Taps per texel. Sixteen is where the banding disappears at the widest cone in the chain; the
// levels below it are narrower and would settle for fewer, but they are also a quarter the pixels
// each, so the chain is dominated by its first level and a per-level count buys nothing.
const TAPS: i32 = 16;
const PI: f32 = 3.14159265;

/// The direction a cube-face texel looks, in the left-handed cube convention the spec uses — the
/// same table in GL, D3D and Vulkan, and the same one `probe_face.wgsl` undoes the camera's mirror
/// to satisfy. Getting this wrong does not produce a broken image, it produces a plausible one that
/// disagrees with its neighbours, which is why it is written out per face rather than derived.
fn face_dir(face: i32, uv: vec2<f32>) -> vec3<f32> {
    let u = uv.x * 2.0 - 1.0;
    let v = 1.0 - uv.y * 2.0;
    switch face {
        case 0: { return normalize(vec3<f32>( 1.0,    v,   -u)); }
        case 1: { return normalize(vec3<f32>(-1.0,    v,    u)); }
        case 2: { return normalize(vec3<f32>(   u,  1.0,   -v)); }
        case 3: { return normalize(vec3<f32>(   u, -1.0,    v)); }
        case 4: { return normalize(vec3<f32>(   u,    v,  1.0)); }
        default: { return normalize(vec3<f32>(-u,     v, -1.0)); }
    }
}

/// A tangent frame around `n`, picking whichever axis `n` is least aligned with so the cross
/// product never collapses. The usual branchless trick; the branch is clearer and this runs over a
/// few thousand texels, not a few million.
fn frame(n: vec3<f32>) -> mat3x3<f32> {
    var up = vec3<f32>(0.0, 1.0, 0.0);
    if (abs(n.y) > 0.99) {
        up = vec3<f32>(1.0, 0.0, 0.0);
    }
    let t = normalize(cross(up, n));
    let b = cross(n, t);
    return mat3x3<f32>(t, b, n);
}

@fragment
fn fs_filter(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let face = i32(cfg.face.x);
    let slot = i32(cfg.face.y);
    let cone = cfg.face.z;
    let n = face_dir(face, in.uv);
    let tb = frame(n);

    // A cosine-weighted cone, sampled on the golden-angle spiral so the taps are even without a
    // noise texture and without the clumping a naive polar grid gives at the centre.
    var sum = vec3<f32>(0.0);
    var wsum = 0.0;
    for (var i = 0; i < TAPS; i = i + 1) {
        let fi = (f32(i) + 0.5) / f32(TAPS);
        let r = sqrt(fi) * cone;
        let phi = f32(i) * 2.399963; // the golden angle
        let local = vec3<f32>(sin(r) * cos(phi), sin(r) * sin(phi), cos(r));
        let dir = normalize(tb * local);
        let w = max(dot(dir, n), 0.0);
        sum = sum + textureSampleLevel(cube, cube_sampler, dir, slot, 0.0).rgb * w;
        wsum = wsum + w;
    }

    // **Alpha is the centre tap, never the average.** The cube's alpha carries the distance to
    // whatever the face drew, and the mean of a foreground distance and a background one is a
    // distance where nothing stands. The parallax correction reads mip 0 regardless, so this is
    // belt and braces — but a blurred distance field is exactly the kind of thing that would sit
    // in the texture looking reasonable and be wrong wherever it was used.
    let centre = textureSampleLevel(cube, cube_sampler, n, slot, 0.0).a;
    return vec4<f32>(sum / max(wsum, 1e-6), centre);
}
