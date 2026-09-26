// The probe's cube faces, mirrored on their way into the cube.
//
// **A right-handed camera cannot render a cube face in the face's own basis.** The cube-face
// convention is left-handed: for +X the spec wants `sc = -z` across the face and `tc = -y` down it
// (the same table in GL, D3D and Vulkan). Bevy's camera puts screen-right at `dir x up`, so
// `up = (0, 1, 0)` gets the vertical right and the horizontal exactly backwards, and the other
// choice of `up` trades one for the other. No up-vector satisfies both, and mirroring the
// PROJECTION instead would invert triangle winding and cull the world inside out.
//
// So the mirror is undone here, once, as the face is written into its cube layer. Before this the
// cube held six individually mirrored faces: every face plausible on its own — trees are trees —
// and none of them agreeing with its neighbours, which reads as a lit seam down every face edge
// rather than as the mirror it is. Measured across the four vertical seams at Mirror Lake, the
// discontinuity was 19.98 before and 6.08 after (the remainder is resampling, not content).

// ---------------------------------------------------------------------------------------------
//
// **The alpha channel carries the DISTANCE to whatever the face drew, in world yards from the
// capture point.** It was unused — the cube is `Rgba16Float` and the water only ever read `.rgb` —
// and filling it turns the cube from a picture into a picture with geometry attached.
//
// That is Lagarde's third geometry proxy. His paper lists sphere volume, box volume and *cube depth
// buffer*, and names the failure of the first two: the proxy does not match the real world, so
// anything standing away from the proxy surface reflects at the wrong place and size. A box fitted
// to one lake's treeline is that lake's measurement wearing a parameter's name, and it is wrong at
// the next lake. A distance stored per texel is a property of the scene and needs no tuning at all.

#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

@group(0) @binding(0) var face: texture_2d<f32>;
@group(0) @binding(1) var face_sampler: sampler;
@group(0) @binding(2) var face_depth: texture_depth_2d;

// Must equal `liquid::probe::PROBE_NEAR`, which is what the face cameras are spawned with. The
// conversion below is only correct for that near plane, and nothing will complain if they drift.
const PROBE_NEAR: f32 = 0.1;

// What to report where the face drew nothing at all — sky, or past the far clip. Not infinity: the
// consumer treats it as "no geometry this way" and a finite number keeps it out of trouble in
// `Rgba16Float`, whose ceiling is 65504.
const PROBE_NO_HIT: f32 = 10000.0;

@fragment
fn fs_mirror(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let uv = vec2<f32>(1.0 - in.uv.x, in.uv.y);
    let rgb = textureSample(face, face_sampler, uv).rgb;

    // `textureLoad` rather than a sampler: the face and its cube layer are the same size, so this
    // is a one-to-one copy, and a depth texture has no meaningful filtered value anyway — averaging
    // a foreground and a background depth yields a distance where nothing stands.
    let dims = vec2<f32>(textureDimensions(face_depth));
    let px = vec2<i32>(clamp(uv * dims, vec2<f32>(0.0), dims - vec2<f32>(1.0)));
    let d = textureLoad(face_depth, px, 0);

    // Bevy's perspective is `perspective_infinite_reverse_rh`, so depth is reversed and the far
    // plane is at 0: view-space depth is `near / ndc`. The projection's `far` field does not enter
    // the matrix at all, which is why it cannot be used here.
    let view_z = PROBE_NEAR / max(d, 1e-6);

    // View depth is measured along the view axis; the cube wants distance along the RAY. A cube face
    // is exactly 90 degrees, so at one unit of view depth the face spans [-1, 1] in both axes and
    // the ray is `length(sx, sy, 1)` long per unit of depth.
    let s = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let dist = view_z * length(vec3<f32>(s, 1.0));

    return vec4<f32>(rgb, select(dist, PROBE_NO_HIT, d <= 0.0));
}
