// Liquid shader, one arm per reference liquid renderer:
//   ADT MCLQ river/ocean (`0x6851b0`/`0x685010`): `ocean0_s.bls`, the depth swatch on stage 0 and
//     the animated sheet on stage 1; the only arm with a depth ramp.
//   WMO MLIQ water (`0x6b62e0` category 0), split on `MOGP.flags & 0x48`: exterior `0x6b6630`
//     binds `MapObjExtWater0.bls`, interior `0x6b6420` is fixed-function and unlit.
//   Magma/slime (`0x6b68f0` WMO, `0x68dca0` ADT): the sheet is the opaque body.
//
// The ADT combine, `Shaders\Pixel\ocean0_s.bls` (0.25 is the program's own `PARAM`):
//   rgb   = primary·colorTex.rgb + detail.rgb + (secondary + 0.25)·detail.a
//   alpha = colorTex.a
// colorTex is the depth swatch off the zone's `Light.dbc` water bands (IntBand 16/17 river/lake,
// 14/15 ocean), rebuilt every frame (`0x680b90`, refill `0x58acd0`); detail is the `lake_a` or
// `ocean_h` frame, near-black RGB and the ripple in alpha, which its authored mips fade with
// distance, so the sampler's mips and anisotropy matter; primary is the lit default white vertex
// (`glColorMaterial`); secondary is the sun sheen.
// The ADT water alpha is the swatch's own: the `0xc7fbc0` LUT texture binds only behind
// `[0xc800ec]` (`0x685244`-`0x685257`), which never holds one (its one store, `0x68c7f8`, is 0).
// Deviation: this is the `specular`/`pixelShaders` = 1 leg the reference install runs; both CVars
// (`0x6886a0`/`0x688712`) default to 0, where water has no program, no specular, a plain ADD
// combine and no blend. An active ARB program bypasses the texture environment.
//
// Culling is off for every kind: all four reference liquid passes disable GL_CULL_FACE. Water
// blends with depth write off; magma and slime are opaque and write depth. Output is raw gamma.

#import bevy_pbr::{
    mesh_functions,
    forward_io::Vertex,
    view_transformations::{
        position_world_to_clip, position_world_to_ndc, direction_world_to_clip, ndc_to_uv,
        frag_coord_to_uv,
        depth_ndc_to_view_z, perspective_camera_near,
    },
    mesh_view_bindings::{view, globals},
}
#ifdef DEPTH_PREPASS
// The opaque scene's depth, written before anything transparent draws — what is BEHIND the water.
// Present only while the stylised look is on; see `benilla_world::liquid::depth`.
#import bevy_pbr::prepass_utils::prepass_depth
#endif

@group(#{MATERIAL_BIND_GROUP}) @binding(100) var frames: texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(101) var frames_samp: sampler;
// The STYLISED look's ripple map (see `stylised_water` at the bottom of this file, and
// `benilla_world::liquid::ripple`): R/G = a tiling slope field, B = the height it came from.
// Bound always, sampled only under `w.path.y`.
@group(#{MATERIAL_BIND_GROUP}) @binding(103) var ripples: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(104) var ripples_samp: sampler;
// The stylised look's PLANAR REFLECTION: the world as a camera mirrored through the water plane
// saw it (`benilla_world::liquid::reflect`), with **alpha as coverage**.
// The opaque scene, snapshotted before anything transparent drew — what a screen-space reflection
// reads once its ray has found a hit (`benilla_world::liquid::scene_color`).
@group(#{MATERIAL_BIND_GROUP}) @binding(111) var scene_tex: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(112) var scene_samp: sampler;
// Must equal `liquid::probe::PROBE_SLOT_MAX`: it is the length of the slot array in the reflection
// buffer, and a mismatch silently reads the wrong rows.
const PROBE_SLOT_MAX: i32 = 16;

// The most probes that may be blended into one fragment. Three is the recommendation and four is the
// ceiling the fixed-size arrays above are built for; the taps actually used come from
// More than this does not make the reflection more correct — averaging cubes
// whose errors point in different directions ghosts rather than converges, which is why Unity caps
// its own blend at two — it makes the HANDOVER smoother, which is the thing worth buying here.
const PROBE_TAPS_MAX: i32 = 3;

@group(#{MATERIAL_BIND_GROUP}) @binding(113) var probe_tex: texture_cube_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(114) var probe_samp: sampler;

// The hierarchical depth the march traverses — see `liquid::hiz` and `shaders/hiz_reduce.wgsl`.
// Each texel is the NEAREST surface in its tile, so a ray in front of it has provably missed
// everything the tile holds.
@group(#{MATERIAL_BIND_GROUP}) @binding(115) var hiz_tex: texture_2d<f32>;
// The same pyramid's FARTHEST-depth twin — what lets a ray pass behind a tile. Read only on the steps
// where the ray is behind a tile's nearest surface; see `liquid::hiz` for why it is a separate image.
@group(#{MATERIAL_BIND_GROUP}) @binding(117) var hiz_far_tex: texture_2d<f32>;


@group(#{MATERIAL_BIND_GROUP}) @binding(105) var reflection_tex: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(106) var reflection_samp: sampler;
// The second planar mirror's capture, for a second water height in view; `water_reflect.mirror2`.
@group(#{MATERIAL_BIND_GROUP}) @binding(110) var reflection_tex2: texture_2d<f32>;

// The live wave field (`benilla_world::liquid::ripple_sim`) — a window of simulated water carried
// along with the viewer. R/G = surface slope, B = the foam the disturbance has whipped up.
@group(#{MATERIAL_BIND_GROUP}) @binding(108) var wake_tex: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(109) var wake_samp: sampler;

// One cubemap reflection probe: where its cube was taken from, and the proxy the parallax
// correction intersects against.
//
// **Every probe carries its own box.** A box-projected cube is exact only where its proxy wall
// coincides with real geometry, so a proxy fitted to one probe's surroundings is meaningless at
// another's; sharing one box between probes would reintroduce the error the probes exist to remove.
struct ProbeSlot {
    // xyz = the capture point in world yards; w = 1 when this slot holds a cube FOR ITS CURRENT
    // CELL. Zero while it is being filled, and zero the moment it is reassigned — a cube of
    // somewhere else is still a plausible reflection, which is exactly why the check has to be
    // structural rather than something the eye is trusted to catch.
    at: vec4<f32>,
    // w spare.
    box_min: vec4<f32>,
    // w = the sphere proxy's radius, 0 meaning "use the box".
    box_max: vec4<f32>,
    // x = the fixed probe spot this slot holds (`benilla_formats::PlanarMap`), -1 for none; yzw
    // spare.
    bind: vec4<f32>,
};

// ---- dome law: begin (shared verbatim by `sky.wgsl` and `liquid.wgsl`) ----------------------
//
// The whole gradient as ONE function of the colours and a direction, so the water's march can ask
// what the sky looks like along a reflected ray without reading the colour buffer. Kept textually
// identical in both shaders — `sky.rs`'s `the_dome_law_is_shared_verbatim` fails on any drift.

struct DomeColors {
    sky0: vec4<f32>, // zenith (90°)
    sky1: vec4<f32>, // 16.8°
    sky2: vec4<f32>, // 9.8°
    sky3: vec4<f32>, // 3.7°
    sky4: vec4<f32>, // 1.8°
    fog: vec4<f32>,  // horizon (0°) and below: LightIntBand row 7
    warp: vec4<f32>, // x = dawn/dusk warp strength S (0 = off), y = sun azimuth (rad), zw reserved
};

// Glow `g` for a sun-relative azimuth phase, sampled linearly like `0x6d0f50` from the reference's
// six-keyframe wrap-around table at `0xce9af8` (written by `0x6ce210`); 0.125 is the sun bearing.
fn azimuth_glow(phase: f32) -> f32 {
    let p = fract(phase);
    if (p < 0.125) { return mix(0.0, 1.0, (p + 0.125) / 0.25); } // wrap 0.875(g0)→1.125(g1)
    else if (p < 0.375) { return mix(1.0, 0.0, (p - 0.125) / 0.25); }
    else if (p < 0.5) { return mix(0.0, -0.5, (p - 0.375) / 0.125); }
    else if (p < 0.625) { return mix(-0.5, -0.7, (p - 0.5) / 0.125); }
    else if (p < 0.75) { return mix(-0.7, -0.5, (p - 0.625) / 0.125); }
    else if (p < 0.875) { return mix(-0.5, 0.0, (p - 0.75) / 0.125); }
    else { return mix(0.0, 1.0, (p - 0.875) / 0.25); } // wrap 0.875(g0)→1.125(g1)
}

// One mid-ring's warped colour for glow `g` (`0x6d0f50`): `S^2` is the prepass S nested in the
// per-segment S, and 0.7 is the constant at `0x7ffd7c`.
fn warp_one(c: DomeColors, base: vec3<f32>, g: f32, s: f32) -> vec3<f32> {
    let s2 = s * s;
    if (g >= 0.0) {
        return mix(base, c.sky1.rgb, (1.0 - g) * s2);
    }
    return mix(mix(base, c.sky1.rgb, s), c.sky0.rgb, 0.7 * (-g) * s2);
}

// The dome's colour along `dir` (unit, world space) from the camera.
fn dome_color(c: DomeColors, dir: vec3<f32>) -> vec3<f32> {
    let elev = degrees(asin(clamp(dir.y, -1.0, 1.0))); // −90..90, 0 = horizon

    // Dawn/dusk warp (`0x6d0f50`): only the four mid rings, never the apex or the fog rim, and
    // never brighter. The reference bakes it per vertex at 24 azimuth segments and interpolates,
    // hence the lerp of two segments; `+ 0.125` puts the sun bearing at glow phase 0.125.
    var s1 = c.sky1.rgb;
    var s2c = c.sky2.rgb;
    var s3 = c.sky3.rgb;
    var s4 = c.sky4.rgb;
    let warp_s = c.warp.x;
    if (warp_s > 0.0) {
        let az = fract((atan2(dir.z, dir.x) - c.warp.y) / 6.2831853 + 0.125);
        let seg = az * 24.0; // 24 azimuth segments, like the dome vertices
        let s0 = floor(seg);
        let f = seg - s0;
        let g0 = azimuth_glow(s0 / 24.0);
        let g1 = azimuth_glow((s0 + 1.0) / 24.0);
        s1 = mix(warp_one(c, c.sky1.rgb, g0, warp_s), warp_one(c, c.sky1.rgb, g1, warp_s), f);
        s2c = mix(warp_one(c, c.sky2.rgb, g0, warp_s), warp_one(c, c.sky2.rgb, g1, warp_s), f);
        s3 = mix(warp_one(c, c.sky3.rgb, g0, warp_s), warp_one(c, c.sky3.rgb, g1, warp_s), f);
        s4 = mix(warp_one(c, c.sky4.rgb, g0, warp_s), warp_one(c, c.sky4.rgb, g1, warp_s), f);
    }

    // Elevation gradient, linear between rings like the reference's Gouraud-shaded dome.
    var col: vec3<f32>;
    if (elev <= 0.0) {
        col = c.fog.rgb; // horizon and below = fog colour (row 7), unwarped
    } else if (elev < 1.8) {
        col = mix(c.fog.rgb, s4, elev / 1.8);
    } else if (elev < 3.7) {
        col = mix(s4, s3, (elev - 1.8) / (3.7 - 1.8));
    } else if (elev < 9.8) {
        col = mix(s3, s2c, (elev - 3.7) / (9.8 - 3.7));
    } else if (elev < 16.8) {
        col = mix(s2c, s1, (elev - 9.8) / (16.8 - 9.8));
    } else {
        col = mix(s1, c.sky0.rgb, (elev - 16.8) / (90.0 - 16.8)); // warped ring1 to the raw apex
    }
    return col;
}

// ---- dome law: end ---------------------------------------------------------------------------------


struct WaterReflect {
    // x = the mirror plane's world Y; y = strength (0 = no reflection this frame — the look is off,
    // no water is near, or the eye is under the surface); z = UV distortion; w = the plane
    // tolerance a surface must be within to take the image.
    params: vec4<f32>,
    // The sun the PLAYER CAN SEE, which is not `wow_light.light_sun`: xyz = the celestial to-sun
    // direction, w = how much of it gets through (horizon x clouds x terrain — the lens flare's own
    // envelope, republished as `benilla_world::sun::SunVisibility`). Written every frame, including
    // the frames the mirror above is off.
    sun: vec4<f32>,
    // The sky dome's own gradient stops, zenith and horizon (`WowLighting.sky` rows 0 and 4) — the
    // colours the dome overhead is actually drawn with, so the water agrees with the sky above it
    // at every hour and in every zone. `w` unused on both.
    sky_zenith: vec4<f32>,
    sky_horizon: vec4<f32>,
    // The wave simulation's window: xy = its lower corner in world XZ yards, z = one over its side
    // length, w = strength — 0 on the reference lane, where the client's own painted splash decals
    // are the wake instead of this.
    sim: vec4<f32>,
    // The white moon, the same pair as `sun`: xyz = the to-moon direction, w = how much of it gets
    // through. Water reflects moonlight as readily as sunlight and the first pass had it reflecting
    // none.
    moon: vec4<f32>,
    /// `x` = the mirror capture's blur radius in capture texels (0 = off —
    /// see [`sample_mirror`]); `y` = paint the march's confidence instead of the water
    /// (off); `z` = the march's OWN strength — 0 unless the debug toggle asks for it, and
    /// pointedly independent of `params.y`, which is the mirror's; `w` = composite the tiers the
    /// old layered way (off).
    ///
    /// `x` once carried a march kill switch and was dead weight even then: the
    /// gate the shader actually reads is `z`. The row is full at four lanes and the params buffer
    /// is sized to it, so a dead lane is the cheapest place for the blur to live.
    flags: vec4<f32>,
    // The cubemap probe (`benilla_world::liquid::probe`): xyz = the probe's world position, w = the
    // radius of the sphere proxy its parallax correction intersects against.
    probe_at: vec4<f32>,
    // x = strength (0 until every face has been captured, and on the reference lane); y = the
    // horizontal reach its weight fades out over; z = the VERTICAL reach, which is much tighter and
    // is what stops a stream sixty yards uphill reading a lake's cube; w = the probe debug view.
    probe_cfg: vec4<f32>,
    // The projection box in world yards, `w` unused on both. Its walls are the water body's own
    // footprint and its ceiling is authored — see `liquid::probe`'s `PROBE_BOX_UP`.
    // `probe_box_min.w` is the surface ripple scale. `probe_box_max.x` is the probe tier's Fresnel
    // boost; the proxy boxes themselves are per-slot and live in `probes`.
    probe_box_min: vec4<f32>,
    probe_box_max: vec4<f32>,
    // x = how much of the water's ripple the CUBE lookup sees; y/z/w spare.
    probe_extra: vec4<f32>,
    // One entry per probe slot — see `liquid::probe::probe_slot_lanes`. Fixed at PROBE_SLOT_MAX
    // rather than at the live count, because a WGSL array length is a compile-time constant; the
    // unused tail carries `at.w = 0` and is skipped.
    probes: array<ProbeSlot, PROBE_SLOT_MAX>,
    // The sky dome's own colours (`sky::dome_uniforms`), for the march's sky hits: where the depth
    // it walks holds nothing, the answer is the sky along the reflected ray by the dome's own law —
    // never the colour buffer, which may hold something drawn there that the depth does not.
    // `warp.z` = read the colour buffer for the march's sky hits instead of the dome law (off).
    dome: DomeColors,
    // The march's own switches. `x` = rays may pass BEHIND a tile once they are behind the
    // thickness slab of its farthest surface (see
    // [`ssr_trace`]); `y` = the ripple's shift of the ray origin; `z` = the probe's softening; `w` = 1
    // when the probes stand on fixed spots and each fragment reads its own region's (the planar lane).
    march: vec4<f32>,
    // The second planar mirror: the same four as `params` (plane, strength, distortion,
    // tolerance), zero while it is off.
    mirror2: vec4<f32>,
};

@group(#{MATERIAL_BIND_GROUP}) @binding(107) var<storage, read> water_reflect: WaterReflect;

struct LiquidParams {
    // x = fullbright (magma/slime); y = ocean swatch; z = interior fog; w = sun-sheen shininess.
    kind: vec4<f32>,
    // x = renderer (`LiquidPath`): 0 = ADT MCLQ, 1 = WMO exterior, 2 = WMO interior; yzw reserved.
    path: vec4<f32>,
    // y = frame count; z = scroll flag (WMO magma/slime only, liquid nibbles 6/7); w = clock
    // enable (0 on a deterministic run).
    anim: vec4<f32>,
};
@group(#{MATERIAL_BIND_GROUP}) @binding(102) var<uniform> w: LiquidParams;

// The prefix of the shared global light (`lighting::global_light`); it must match row for row.
struct WowLight {
    light_ambient: vec4<f32>,      // 0  rgb = ambient; w = Mod2x scale
    light_diffuse: vec4<f32>,      // 1  rgb = sun diffuse; w = clamp flag
    light_sun: vec4<f32>,          // 2  xyz = sun travel direction (to-light = −xyz)
    light_spec: vec4<f32>,         // 3  rgb = row-9 specular colour; w = terrain shininess, unread
    fog_color: vec4<f32>,          // 4  rgb = scene fog (block 1, gamma 0..1); w = enable (>0.5)
    fog_params: vec4<f32>,         // 5  x = start yd; y = end yd; w = the farclip wall
    _sh: array<vec4<f32>, 6>,      // 6-11  model SH coefficients, unread here
    _sh_c16: vec4<f32>,            // 12
    water_river: array<vec4<f32>, 2>, // 13-14 river/lake shallow, deep (IntBand 16/17); w = alpha
    water_ocean: array<vec4<f32>, 2>, // 15-16 ocean shallow, deep (IntBand 14/15); w = alpha
    _grade: vec4<f32>,             // 17
    wmo_fog_color: vec4<f32>,      // 18 rgb = interior fog (block 2); w = enable
    wmo_fog_params: vec4<f32>,     // 19 x = start yd; y = end yd
};
@group(#{MATERIAL_BIND_GROUP}) @binding(90) var<storage, read> wow_light: WowLight;

struct LiquidVsOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) depth: f32,
    // The sun sheen, per vertex and interpolated, as the reference's fixed-function stage does.
    @location(4) secondary_vtx: vec3<f32>,
    // A WMO interior pool's `MOMT.diffColor`, as its reference vertex carries; white elsewhere.
    @location(5) vcolor: vec4<f32>,
    // Distance from this vertex to the waterline in yards (UV1.y), for the shore foam.
    @location(6) shore: f32,
    // Offset to the nearest point on the waterline, world XZ yards; its length is taken per fragment.
    @location(7) shore_offset: vec2<f32>,
    // `MeshTag` bit 30: the room's per-frame interior-fog gate, the reference's `[0xca7f00]`.
    @location(8) @interpolate(flat) room_fog: u32,
    // The heightfield's smooth normal, world space; zero on a mesh without it.
    @location(9) surface_normal: vec3<f32>,
    // The two reflection tiers' split (`ATTRIBUTE_WOW_PLANAR`): x the mirrors' weight, y z the two
    // probe spots read, w the second's share.
    @location(10) planar: vec4<f32>,
    // The current (`ATTRIBUTE_WOW_FLOW`), world XZ yards a second; zero on still water.
    @location(11) flow: vec2<f32>,
}

// Sun sheen (`secondary`): the Blinn highlight `light_spec.rgb · (N·H)^shininess`.
fn sun_sheen(world_normal: vec3<f32>, world_pos: vec3<f32>) -> vec3<f32> {
    let n = normalize(world_normal);
    let to_light = -normalize(wow_light.light_sun.xyz);
    // Local viewer, per vertex: the reference sets `GL_LIGHT_MODEL_LOCAL_VIEWER = 1` at `0x59cf89`.
    let to_view = normalize(view.world_position.xyz - world_pos);
    let half_v = normalize(to_light + to_view);
    let ndoth = max(dot(n, half_v), 0.0);
    // No `N·L > 0` specular gate: the sun stays between +20° and +37° (`DayNight::SetDirection`),
    // so N·L on the flat up normal is always > 0. Shininess is water's own (6.0, `[0x8102e8]`) and
    // material specular is white (`SetRenderState(3, 0xffffffff)`). The reference's specular light
    // colour (`CGLight+0x48` to `glLightfv(GL_SPECULAR)`) is not pinned; row 9 stands in for it.
    return wow_light.light_spec.rgb * pow(ndoth, max(w.kind.w, 1.0));
}

fn anim_time() -> f32 {
    return w.anim.w * globals.time;
}

// The 24 fps flip, 30 frames over 1.25 s (`0x68aac0`), floored to a whole frame.
fn frame_layer() -> i32 {
    return i32(floor(anim_time() * 24.0) % max(w.anim.y, 1.0));
}

fn apply_scroll(uv: vec2<f32>) -> vec2<f32> {
    // Magma/slime scroll (liquid nibbles 6/7): the reference's stage-0 texture matrix (`0x6b68f0`,
    // pushed at `0x6b6ae3`) translates v by `fmod(t, 10) · 0.1` (rate `[0x801620]`, period
    // `[0x80e5a0]`), continuously; its phase is machine uptime, so only rate and period match.
    // REPEAT wrapping hides the reset.
    return vec2<f32>(uv.x, uv.y + w.anim.z * fract(anim_time() / 10.0));
}

// Planar eye-Z GL_LINEAR fog in gamma space, as terrain.wgsl; GL_FOG defaults on (`0x593bf0`) and
// no liquid pass turns it off. Block 1 (`+0x70/74/78`) is the scene fog (`0x66ff20`); block 2
// (`+0x80/84/88`) is the interior haze, eased toward the MFOG or zone target over about 4 s
// (`0x6cf054`). Only the WMO geometry pass (`0x6b51d9`/`0x6b51ea`) and the WMO liquid pass
// (`0x6b6323` to `0x6b6342`) submit block 2, under the room gate `[0xca7f00]`: `w.kind.z` is its
// static half (`MOGI & 0x48`), `room_fog` the per-frame half.
fn apply_fog(rgb: vec3<f32>, world_pos: vec3<f32>, room_fog: u32) -> vec3<f32> {
    var fog_color = wow_light.fog_color;
    var fog_span = wow_light.fog_params.xy;
    if (w.kind.z > 0.5 && room_fog != 0u) {
        fog_color = wow_light.wmo_fog_color;
        fog_span = wow_light.wmo_fog_params.xy;
    }
    if (fog_color.w <= 0.5) {
        return rgb;
    }
    let eye_z = -(view.view_from_world * vec4<f32>(world_pos, 1.0)).z;
    let denom = max(fog_span.y - fog_span.x, 0.001);
    let factor = clamp((fog_span.y - eye_z) / denom, 0.0, 1.0);
    return mix(fog_color.xyz, rgb, factor);
}

// ── The stylised look (`waterStyle` = 1) ────────────────────────────────────────────────────────
//
// **Not the reference, and it does not pretend to be.** Everything above this line reproduces what
// 1.12 draws; this is an alternative the player picks in Options → Graphics → Water Style, ported
// from the `water-test` sandbox's "Basic" preset. It replaces the *treatment* of the surface, not
// the world's own water data: the zone's `Light.dbc` swatch still supplies both colours, the same
// depth `V` still drives colour and opacity, the same lighting, the same fog — every input the
// world already decides is kept, and only what is done with them changes.
//
// What it does that `ocean0_s.bls` does not:
//
//   * **An animated surface normal.** Three layers of the generated slope map at unrelated scales
//     and drift directions. Three rather than two matters: two scrolling copies of one texture beat
//     against each other at a visible period, and a third at an unrelated scale hides it.
//   * **A real specular off that normal.** The reference's `secondary` is a Blinn highlight on a
//     FLAT plane, so it is a broad wash that the ripple *alpha* modulates. Here the ripples are in
//     the normal itself, so the same highlight breaks into a glitter path — which is what actually
//     reads as water in motion.
//   * **A Fresnel sky mix.** Water reflects almost nothing looking straight down and almost
//     everything at a glancing angle, and that ramp is most of why a lake reads as a lake. There is
//     no environment map on this path, so the sky is stood in for by the scene fog colour — which
//     is the horizon's colour, tracks the zone and the clock, and is already on this buffer.
//   * **Shore foam.** The depth ramp the reference uses for colour doubles as a shoreline: a band
//     that concentrates at the waterline, broken up at two scales by the map's height channel so it
//     reads as surf gathering rather than as an outline drawn around the water.
//
// The wavelengths are in YARDS of world space (the sandbox's metres, taken across at face value —
// the two units differ by less than a ripple).

/// How far out from the waterline the solid part of the line reaches, in yards. Thin on purpose:
/// this is a line drawn where the water meets the land, not a surf zone.
const FOAM_LINE_YARDS: f32 = 0.15;

/// How white the line gets at its strongest. Under 1 so the water's own colour still shows through
/// even at the very edge — the line is meant to be read as foam lying ON water, and a fully opaque
/// white one reads as a stroke drawn around the lake in a paint program.
const FOAM_LINE_LEVEL: f32 = 0.50;

/// How much the line's own width wavers along the shore, as a fraction of that width. Small — the
/// line wants to read as an even edge, and the broken-up look belongs to the bubbles behind it, not
/// to the line itself.
const FOAM_EDGE_WARP: f32 = 0.22;

/// How far past the line the bubbles carry before there is only water, in yards.
const FOAM_BUBBLE_YARDS: f32 = 0.62;

/// **The swell** — how far the waterline runs up and back down the shore, in yards, and how long
/// one breath of it takes, in seconds.
///
/// The band was fixed in world space, which is the one thing a shoreline never is: the water's edge
/// is the most obviously *still* part of a frame that is otherwise moving. Advancing and retreating
/// the whole band along its own gradient costs nothing — it is the same distance field, read at a
/// moving threshold — and it is what makes a coast read as tidal rather than painted on.
///
/// Deliberately small against the band's own width: this is a swell lapping, not a tide coming in.
const TAU: f32 = 6.2831855;
const SHORE_RUNUP_YARDS: f32 = 0.16;
const SHORE_SWELL_SECS: f32 = 5.5;

/// How far apart two stretches of coast are before they breathe out of step, in yards. Without this
/// every shoreline on the map advances and retreats in unison, which reads as the whole world
/// pulsing rather than as water; a slow phase drift along the coast makes each bay its own.
const SHORE_SWELL_SPREAD: f32 = 70.0;

/// How bright the bubbles are against the solid line.
const FOAM_BUBBLE_LEVEL: f32 = 0.30;

/// Edge length of the generated ripple map, in texels — `benilla_world::liquid::ripple`'s `SIZE`.
/// Needed here to turn its stored central-difference slope into a real `d/d(uv)`; see
/// [`foam_noise`].
const RIPPLE_TEXELS: f32 = 256.0;

/// How far from the waterline the shore band is evaluated at all, in yards.
///
/// **The band was drawn on every water fragment in the world.** Out in open ocean, four hundred
/// yards from any shore, `line` and `bubbles` both resolve to zero — and the surface still paid two
/// texture taps for the edge, three more inside [`foam_noise`], and two `fwidth` chains to arrive
/// at nothing. Five of the roughly thirteen taps a stylised water fragment makes, spent on the
/// overwhelming majority of them for no pixel.
///
/// The reach is the band's own outer extent and is derived from the terms that set it rather than
/// picked, so it cannot drift out of step with them: the line at its highest swell and widest warp,
/// plus the bubble taper behind it, plus a quarter yard for the anti-aliasing that softens the far
/// end. Past it `1 - out` is identically zero and so is the line's smoothstep.
const FOAM_REACH: f32 = (FOAM_LINE_YARDS + SHORE_RUNUP_YARDS) * (1.0 + FOAM_EDGE_WARP)
    + FOAM_BUBBLE_YARDS
    + 0.25;

/// The foam's own colour — spray, so near-white with the faintest cool cast.
const FOAM_COLOR: vec3<f32> = vec3<f32>(0.88, 0.94, 0.96);

/// How far past the trusted band the correction is allowed to keep working, as a multiple of it.
///
/// The reprojection does not stop being right at any particular distance — it degrades, because the
/// lifted sample is placed at the water's own depth rather than at the depth of what is being
/// reflected, and that approximation loosens as the lift grows. So the reflection is carried at
/// full weight across the trusted band and then faded out over another band this much wider, which
/// is long enough that no edge of it is visible as an edge.
const PLANE_FADE: f32 = 2.5;

/// The surface tilt, in radians, up to which the planar capture is trusted completely, and the
/// tilt at which it is trusted not at all.
///
/// **A capture is rendered through a HORIZONTAL mirror.** Its plane error can be corrected — that
/// is the reprojection — but its *orientation* cannot: a surface tilted by `t` reflects its rays
/// `2t` away from where the capture looked, and no amount of moving the sample around an image
/// recovers a direction the image never contained. Elwynn's stream falls about 3.04 yd across a
/// single chunk, roughly 5.2 deg, so its reflected ray is out by better than ten.
///
/// So tilt is the second half of the trust: past a few degrees the planar image is a confidently
/// wrong picture, and the sky mix it fades into is an honest one. The trusted band is generous
/// enough that an ordinary river keeps its reflection, and the limit is where a chute or a
/// cataract stops pretending.
/// **Tightened once the march landed**, and the reason is the march. These were 0.10 and 0.32,
/// chosen when the capture was the only tier there was: fading a sloped surface out then did not
/// hand it to something better, it simply took its reflection away, so the band had to be generous
/// enough to let an ordinary river keep one. Elwynn's stream is 5.22 deg (3.044 yd of relief across
/// one 33.33 yd chunk, measured), which the old band trusted **completely** — at a 10.4 deg ray
/// error.
///
/// Screen-space reflection is exact at any orientation and, measured, is
/// confident over nearly all of that same stream. So the capture no longer has to cover for it, and
/// can be honest about where a horizontal mirror stops describing the surface: 5.22 deg now keeps
/// about 0.71 of it and the rest comes from the tiers that are right.
const SLOPE_TRUSTED: f32 = 0.03; // ~1.7 deg
const SLOPE_LIMIT: f32 = 0.20; // ~11.5 deg

/// How much the sample is *additionally* smeared as trust falls away.
///
/// The reference's own reflections are faint and broken up, and that is precisely why its
/// single-plane errors never read as errors. A badly-fitted surface here does the same: it dims
/// (the trust weight) and it blurs (this), so what survives is a suggestion of a reflection rather
/// than a sharp claim about geometry the capture got wrong.
const DISTORT_LIFT: f32 = 3.0;

/// The **geometric** tilt of the water under this fragment, in radians.
///
/// Taken from screen-space derivatives of the world position rather than from the mesh, because
/// the liquid mesh carries no slope at all — `surface.rs` writes `[0, 1, 0]` into every vertex
/// normal, so `world_normal` is a constant and says nothing about the heightfield. The cross of
/// the two derivatives is the true facet normal and costs nothing extra here.
///
/// Called in uniform control flow, before any branch: derivatives past a per-fragment `if` are
/// undefined, which is the same rule the shore band is written around.
fn surface_tilt(world_pos: vec3<f32>) -> f32 {
    let g = cross(dpdx(world_pos), dpdy(world_pos));
    let len = length(g);
    if (len < 1e-12) {
        return 0.0;
    }
    // Only the magnitude of the tilt matters, so the facet's winding is irrelevant.
    return acos(clamp(abs(g.y / len), 0.0, 1.0));
}

/// The **geometric** normal of the water facet under this fragment — the heightfield's own
/// orientation, with none of the ripple in it.
///
/// Same derivative cross as [`surface_tilt`] and the same uniform-control-flow rule: call it before
/// any per-fragment branch. The winding IS forced here, because this one is a direction rather than
/// a magnitude and a liquid surface is always seen from above.
fn surface_normal(world_pos: vec3<f32>) -> vec3<f32> {
    let g = cross(dpdx(world_pos), dpdy(world_pos));
    let len = length(g);
    if (len < 1e-12) {
        return vec3<f32>(0.0, 1.0, 0.0);
    }
    let n = g / len;
    return select(-n, n, n.y >= 0.0);
}

/// The mirror capture, read through a **4-tap box blur** — the reference pipeline's own shape.
///
/// Cataclysm's Ultra water does not sample its mirrored capture directly. It renders the mirror
/// into a downscaled target, runs a 4-tap box blur into a second target, and the water shader
/// samples *that* (`/data/scratch/done/water-ultra/UltraWater_Cata_4.3.4_Spec.md` §5.2). This pass
/// had the downscale and not the blur, and the difference is most of why the reflection read as
/// hard-edged debris rather than as a reflection: a capture is a sharp image of the world, and the
/// ripple offset below *displaces* the sample rather than softening it, so every artefact in the
/// capture arrives at full contrast with a crisp edge on it. Blurring is the step that makes a
/// wrong sample read as water instead of as a mistake — and it is also why the reference's own
/// reflections are faint and broken up, which the shore-band note above says in the other
/// direction.
///
/// Four bilinear taps on the diagonals, which is the whole kernel: at `radius` = 1 each tap already
/// averages the 2x2 it lands between, so the four together cover 3x3 with tent weights for the cost
/// of four fetches.
///
/// **The average is premultiplied, and it has to be.** The capture clears to *transparent black*
/// (alpha is coverage — see `benilla_world::liquid::reflect`), so a straight RGBA mean drags colour
/// toward black wherever a tap lands off the drawn geometry, and every silhouette in the reflection
/// would gain a dark fringe exactly where the blur was supposed to soften it. Dividing the summed
/// colour by the summed coverage is the mean of what was actually *drawn*, and leaves the coverage
/// itself to fade the tier out the way it already does.
///
/// `textureSampleLevel` rather than `textureSample`: this is called under a per-fragment branch,
/// where implicit derivatives are undefined, and the capture is a single-mip image so level 0 is
/// the only level there is.
fn sample_mirror(uv: vec2<f32>, radius: f32) -> vec4<f32> {
    if (radius <= 0.0) {
        return textureSampleLevel(reflection_tex, reflection_samp, uv, 0.0);
    }
    let o = radius / vec2<f32>(textureDimensions(reflection_tex));
    var acc = textureSampleLevel(reflection_tex, reflection_samp, uv + vec2<f32>(-o.x, -o.y), 0.0);
    acc += textureSampleLevel(reflection_tex, reflection_samp, uv + vec2<f32>(o.x, -o.y), 0.0);
    acc += textureSampleLevel(reflection_tex, reflection_samp, uv + vec2<f32>(-o.x, o.y), 0.0);
    acc += textureSampleLevel(reflection_tex, reflection_samp, uv + vec2<f32>(o.x, o.y), 0.0);
    acc *= 0.25;
    return vec4<f32>(acc.rgb / max(acc.a, 1.0e-4), acc.a);
}

/// [`sample_mirror`] on the second mirror's capture.
fn sample_mirror2(uv: vec2<f32>, radius: f32) -> vec4<f32> {
    if (radius <= 0.0) {
        return textureSampleLevel(reflection_tex2, reflection_samp, uv, 0.0);
    }
    let o = radius / vec2<f32>(textureDimensions(reflection_tex2));
    var acc = textureSampleLevel(reflection_tex2, reflection_samp, uv + vec2<f32>(-o.x, -o.y), 0.0);
    acc += textureSampleLevel(reflection_tex2, reflection_samp, uv + vec2<f32>(o.x, -o.y), 0.0);
    acc += textureSampleLevel(reflection_tex2, reflection_samp, uv + vec2<f32>(-o.x, o.y), 0.0);
    acc += textureSampleLevel(reflection_tex2, reflection_samp, uv + vec2<f32>(o.x, o.y), 0.0);
    acc *= 0.25;
    return vec4<f32>(acc.rgb / max(acc.a, 1.0e-4), acc.a);
}

/// How much of the planar reflection a surface `err` yards off the capture plane keeps.
///
/// **This replaced a hard cutoff, and the reason is the shot that broke it.** A capture serves one
/// plane exactly and its neighbours approximately (see the reprojection in `stylised_water`), and
/// the first cut simply refused any surface more than `trusted` yards away. But a frame regularly
/// holds several water bodies at once — a roadside pool at 10.07 and the sea at 0.00 is a real
/// Stranglethorn view — and under a threshold exactly one of them could reflect while the other
/// went flat. Which one flipped as the camera moved, because the capture plane is chosen from what
/// covers the screen.
///
/// A falloff serves them all instead: the body the capture was built for is exact, the others are
/// approximate and a little weaker, and nothing pops when the choice changes hands.
fn plane_trust(err: f32, trusted: f32, tilt: f32) -> f32 {
    let by_height = 1.0 - smoothstep(trusted, trusted * (1.0 + PLANE_FADE), err);
    let by_slope = 1.0 - smoothstep(SLOPE_TRUSTED, SLOPE_LIMIT, tilt);
    return by_height * by_slope;
}

/// How reflective the water is allowed to get at a grazing angle. Below Schlick's 1.0 on purpose —
/// see its use.
const REFLECT_MAX: f32 = 0.44;

/// The most of the pixel the CUBE PROBE tier may take, however far its strength is pushed.
/// Above [`REFLECT_MAX`] because this knob exists to make the tier legible, but short of 1.0 so the
/// surface keeps some of its own body and does not read as a mirror.
const PROBE_REFLECT_CEIL: f32 = 0.85;

/// The specular lobe's exponent and weight — **the water's roughness, in the only two numbers this
/// shader has for it.**
///
/// These were 160 and 0.55, and together with a near-mirror reflection they are what made the
/// surface read as "someone poured oil into the water": a very tight lobe is a very smooth surface,
/// and a smooth surface returns the sun as a small hard disc sitting *on* the water rather than a
/// sheen scattered through it. Oil is smoother than water; that is the whole difference, and it is
/// this exponent.
///
/// Broadened and dialled back, the highlight spreads into the ripples instead of sliding over them.
/// A microfacet model would derive both from one roughness and from the same normal distribution
/// the reflection uses; this is the stylised path, so it is two numbers and an eye.
///
/// Broadened again, from 42/0.30, against the "oily" report: at 42 the lobe still resolved into
/// hard bright chips on the ripple crests, and a hard chip on a smooth crest is exactly the read of
/// a film on the surface rather than light coming off the water. The other half of that report is
/// answered by the reflection ceiling above and by the finest ripple layer's weight below — a
/// surface looks oily when it is smooth, mirror-like and finely crinkled all at once, and all three
/// had to come down together.
/// How hard the ambient ripples tilt the surface — see the normal's construction.
const NORMAL_STRENGTH: f32 = 1.35;

const SPEC_POWER: f32 = 24.0;
const SPEC_WEIGHT: f32 = 0.22;

/// The moon's own glitter, relative to the sun's. Moonlight IS sunlight at about a millionth the
/// intensity, but it lands on a scene lit at a millionth too — what actually differs on screen is
/// that the eye is scotopic, so the path reads dimmer, cooler and softer-edged than the sun's, and
/// that is what these three do.
const MOON_SPEC_WEIGHT: f32 = 0.55;
const MOON_SPEC_POWER: f32 = 34.0;
const MOON_SPEC_COLOR: vec3<f32> = vec3<f32>(0.62, 0.70, 0.86);

/// How far along the swatch's shallow→deep ramp the body is allowed to travel.
///
/// Under 1 on purpose. The 1.12 water palettes were authored for a surface with no reflection and
/// no glitter on it, and several zones' deep row is nearly black — Stranglethorn's is why the
/// report named Booty Bay. Multiplied by `lit` and then fogged, that row reads as tar rather than
/// as deep water. Capping the ramp keeps the lift inside the ZONE'S OWN pair of colours: deep water
/// is still the deep end of Booty Bay's green, one step back up its own ramp, and never a wash of
/// grey mixed in from outside the palette.
const BODY_DEPTH_MAX: f32 = 0.72;

/// How hard the ripples are carved into the water's own body, as a ± fraction of its colour.
///
/// The body is otherwise a constant, and a constant is exactly what you see when you look STEEPLY
/// down at water close to you: Fresnel goes to nothing at that angle, so the sky mix and the
/// mirrored image both fall away and the swatch is all that is left — a flat sheet of paint under a
/// surface that is visibly moving everywhere else in the frame. That is the near half of "not
/// enough activity in the water"; the far half is the long swell and the sky gradient.
///
/// The tilt is measured along the **sun's compass bearing**, which is fixed all day, rather than
/// along the sun vector itself: at noon the sun is nearly overhead, every facet faces it about
/// equally, and a term built on the full vector would go flat at exactly midday. Taken this way the
/// shading is zero on flat water — it darkens one face of each wave and lightens the other by the
/// same amount, so it carves the surface without changing the zone's colour.
const BODY_RIPPLE_SHADE: f32 = 0.20;

/// How far in from the wave window's edge the simulation is faded out, as a fraction of its side.
/// It matches the absorbing band on the CPU (`ripple_sim::EDGE_ABSORB`), so a wave is already down
/// to nothing by the time this has finished hiding it — the two together are what keep the window
/// from having a visible square edge in the water.
/// How near the waterline the exact per-fragment distance is used instead of the interpolated
/// scalar, in yards, and how far the two may disagree before the scalar is trusted instead. The
/// reach is comfortably past everything the band draws; the trust bound is what keeps a channel's
/// midline ridge from producing a false line. See the use.
/// Over how many yards of water column the surface fades out where something passes through it.
///
/// Without it every rock, pier and hull meets the water on a hard aliased line, because the water's
/// alpha knows nothing about what is behind it; with it the surface closes around the intersection
/// the way real water does. It is the cheapest thing depth buys and probably the most visible.
///
/// **Short on purpose.** This is a fade against an INTERSECTION, not a model of shallow water. At
/// half a yard it stopped being one: a knee-deep lagoon went transparent across its whole floor,
/// because a shallow bar seen at a grazing angle covers a great deal of screen, and the result was
/// a flat pale plateau rather than water. A hand's breadth closes the seam at a rock and leaves
/// water that is merely shallow looking like water.
const SOFT_EDGE_YARDS: f32 = 0.22;

/// **Beer–Lambert extinction per yard of water, per channel.** Red is gone within a couple of
/// yards, green carries further, blue furthest — which is the whole reason deep water is blue and
/// why a two-colour lerp of the zone swatch could never look like water at any depth but the one it
/// was tuned at.
///
/// The palette is still the zone's: these decide how fast the shallow row gives way to the deep
/// one, not what either colour is. What changes is that the answer now comes from the real distance
/// through the water rather than from the authored depth byte, so a hull sitting in six inches of it
/// reads as six inches.
const WATER_EXTINCTION: vec3<f32> = vec3<f32>(0.46, 0.16, 0.09);

/// How deep the water is allowed to count as, in yards, when the scene behind it is further away
/// than the far plane's honest answer — sky through a gap, or an unloaded chunk. Without a cap a
/// missing background reads as infinitely deep water and the shallows go black at tile edges.
const WATER_MAX_THICKNESS: f32 = 24.0;

const SHORE_EXACT_REACH: f32 = 3.0;
const SHORE_EXACT_TRUST: f32 = 0.75;

const WAKE_EDGE_FADE: f32 = 0.08;

/// How white a simulated wake's crest goes. Below the shoreline foam's own level: a wake is
/// aerated water, a breaking waterline is foam sitting on top of it.
const WAKE_FOAM_LEVEL: f32 = 0.14;

/// The wake slope at which foam starts, and how fast it arrives after that.
///
/// The floor is high and the level low because the reference is much subtler than it first seems:
/// in the 1.14.2 footage a swimmer's trail is almost entirely a TONAL streak — the water simply a
/// shade different along the track — with white only in a fleck at the body itself. Most of what
/// reads as a wake there is not foam at all, which is why the body-depth term below carries the
/// trail and this only tips the crest nearest the swimmer. Derived from the SLOPE
/// rather than carried in its own channel: a surface only aerates where it is steep, the slope is
/// the quantity with the range to say so (the curvature this first used peaks twenty times smaller
/// than the height and vanished into the bottom of a byte), and taking it from the same two
/// channels that bend the normal means the white always lands on the face the light does.
const WAKE_FOAM_FLOOR: f32 = 0.80;
const WAKE_FOAM_GAIN: f32 = 2.6;

/// How far the wake's own height moves the body along the depth ramp. A trough is less water
/// between the eye and the bed and a crest is more, and reading it that way is what puts a wake on
/// screen when you are looking STEEPLY DOWN at it — the angle at which Fresnel is nothing, the sky
/// mix and the reflection are gone, and the body colour is the only term left.
const WAKE_BODY_DEPTH: f32 = 0.22;
/// One octave of the wave slope, read off the tiling map on its **own rotated lattice**.
///
/// `rot` is `(cos, sin)` of the angle this layer's world plane is turned through before the lookup.
/// Without it every layer reads the same 256-texel tile on the same axes, so their seams land on
/// top of each other and the sum repeats on that one grid — a twelve-yard chequer that walks across
/// open water and was the "the noise is repeating" report. Turned against each other by angles that
/// share no common fraction of a turn, the layers' periods no longer line up and the sum has no
/// visible period at all.
///
/// The gradient comes back out of the rotated frame at the end. Skipping that is the trap: the map
/// holds a slope, and a slope sampled in a turned frame is a slope *in that frame*, so leaving it
/// there would light every wave as though its crests ran at an angle to the wave itself.
fn ripple_layer(
    world_xz: vec2<f32>,
    drift: vec2<f32>,
    inv_wavelength: f32,
    weight: f32,
    t: f32,
    rot: vec2<f32>,
) -> vec2<f32> {
    let r = mat2x2<f32>(rot.x, -rot.y, rot.y, rot.x);
    let uv = (r * world_xz) * inv_wavelength + drift * t;
    // **Fade the octave out once a pixel covers more than a texel of it.**
    //
    // This is the Nyquist limit, not a taste setting. A layer whose texels are smaller than the
    // pixel sampling it cannot be resolved: what comes back is not the wave, it is whichever part
    // of the wave happened to land under the sample point, and across a surface seen at a grazing
    // angle that is a moire — horizontal striation running along the lines of equal distance,
    // because that is where the footprint is constant. The sampler cannot save it either: a grazing
    // footprint is long in one axis and short in the other, and this map is deliberately sampled
    // without anisotropy (see `liquid::ripple`, where the cost of it was measured and declined), so
    // the mip it picks is wrong for one axis whichever it picks.
    //
    // Detail that cannot be resolved should be ABSENT rather than aliased, so the weight falls to
    // zero across the octave between "a pixel spans one texel" and "a pixel spans four". Distant
    // water loses its finest crinkle and keeps its larger structure, which is what distant water
    // actually looks like — the alternative is a shimmer that is not on the surface at all.
    let texels = f32(textureDimensions(ripples).x);
    let footprint = length(fwidth(uv)) * texels;
    let resolvable = 1.0 - smoothstep(1.0, 4.0, footprint);
    let g = (textureSample(ripples, ripples_samp, uv).rg * 2.0 - 1.0) * weight * resolvable;
    return transpose(r) * g;
}

// **What the linear march left behind, because it was expensive to learn.**
//
// This tier used a fixed-cadence march with geometric steps, a bisection and a thickness test, and
// all of it is gone — `ssr_trace` walks a hierarchy now. Three findings from it are not:
//
// * **The sky is a HIT, not a miss.** Reflecting off a surface below the eye sends the ray up and
//   forward, at sky that is on the screen, and the prepass draws no dome so the march read "no
//   depth" and gave up at exactly the pixels whose answer was in the snapshot. Treating it as a hit
//   took `water-noon` from 31.6% to 46.4% of water pixels answered. The hierarchy gets the same
//   thing for free: a cell reducing to zero is a tile of pure sky.
// * **The thickness test has to be asked at the crossing, not at the coarse step**, or it measures
//   the march's overshoot instead of the geometry and rejects solid hillsides in bands. That fix
//   took the same shot from 13.9% to 31.6%. The traversal honours it by construction: the test
//   runs once, at the level-zero texel the walk ended in.
// * **A finer march buys nothing.** 40 steps growing 1.12x against 28 growing 1.22 moved the hit
//   rate by 0.2 points. Sampling more often is not the answer to being blind between samples;
//   not being blind is.

/// The stylised lava look, in three numbers: how dark the crust goes, how hot the molten goes, and
/// how large the plates are in yards.
///
/// **Lava is the one liquid with no reflection to give it structure**, and on the faithful lane it
/// needs none — the reference draws magma as its animated sheet, fogged, and that sheet is the whole
/// surface. Ported unchanged onto the stylised lane it is the only liquid there that reads as a
/// scrolling texture rather than as a material, because every other kind got a normal, a Fresnel mix
/// and a reflection and magma got nothing.
///
/// What it wants instead of a reflection is **temperature**. Real lava is a dark basalt skin broken
/// into slow plates with molten seams between them, and the plates are what make it read as a crust
/// with something underneath rather than as a moving picture. So: a slow two-octave field off the
/// ripple map picks the plates, the sheet's own luminance says which parts are already hot, and the
/// surface is graded between a darkened crust and a molten seam that is pushed ABOVE 1.0 so the
/// glow pass blooms it. Nothing here invents colour — both ends are the zone's own animated texture
/// scaled — which is the same rule the water's tiers follow: change the treatment, never the
/// palette.
///
/// [`LAVA_GLOW`] above 1 is the point and not an accident. The pipeline is HDR with `ffx_glow`
/// downstream, so a seam at 2.6 blooms and a crust at 0.45 does not, and the difference between
/// them is most of what makes the plates read as solid.
const LAVA_CRUST: f32 = 0.45;
const LAVA_GLOW: f32 = 2.6;
/// Plate size in yards, and the finer break laid across it.
const LAVA_PLATE_YD: f32 = 19.0;
const LAVA_BREAK_YD: f32 = 5.5;

/// How far the march's READ is blurred, in **main-view texels** — the march's answer to the
/// downscale-and-blur the planar capture already gets.
///
/// **The march's hits are hard-edged, and that is information rather than texture.** Where the
/// planar capture is half resolution and then 4-tap blurred ([`sample_mirror`]), the snapshot is
/// full resolution and unfiltered, so the march's own discontinuities — the boundary between a
/// pixel that found something and one that did not, the quantisation of the refined crossing, the
/// edge-fade ramp — arrive at full contrast. Displaced per-pixel by the ripple, they read as crisp
/// striations across the surface, which is easy to mistake for wave detail and is nothing of the
/// kind: it is the shape of what the march does not know.
///
/// Two taps' worth of radius covers the same ~6 main-view texels the planar tier's kernel does (one
/// capture texel at [`REFLECT_DOWNSCALE`](benilla_world::liquid) 2, tented over 3x3), which is the
/// point — the two tiers should not be distinguishable by their grain. This is the cheap half of
/// "trace into a downscaled buffer and blur it": the *colour* is filtered here, for four fetches on
/// a path that was already about to make one. The march's hit-finding is untouched and stays exact,
/// because it walks the depth prepass at full resolution and never reads the snapshot to find a
/// crossing — only to colour one.
const SSR_READ_BLUR: f32 = 2.0;

/// How far the ripple slides a reflection's LOOKUP POINT across the surface, in yards —
/// shared by all three tiers, because the mistake it fixes was shared by all three.
///
/// **The ripple belongs to the surface, not to the ray and not to the read.** All three were
/// tried against a reproduction of the artefact Stefan reported — a reflected trunk broken into
/// repeated rungs — and the difference between them is not a matter of degree:
///
/// * **In the READ** (what this replaces): the hit's UV was displaced in ORIGINAL screen space,
///   which is not a reflection of anywhere. Shoving that image around samples unrelated geometry,
///   and on a vertical feature it walks up and down the trunk and shows the same bark repeatedly.
///   The planar tier can displace its read because its capture genuinely IS a reflected image; the
///   march has no such buffer, and borrowing the technique from the tier that does was the error.
/// * **In the RAY**: geometrically honest and unusable. Neighbouring pixels get meaningfully
///   different directions, and at a grazing angle a slight tilt swings the ray a long way
///   vertically, so hits scatter into vertical confetti — measured at a 2.10 vertical-to-horizontal
///   ratio against 1.32 for the read displacement, i.e. twice as bad as the bug. (Its hit rate is
///   fine now, 39.3% where the note that abandoned this approach recorded 11.8% — that measurement
///   predates the thickness-rejection reordering and no longer stands. The approach fails for a
///   different reason than the one on record.)
/// * **In the ORIGIN**, which is this: neighbours keep the SAME ray direction, so the march stays
///   coherent and the picture stays smooth, while every sample remains a true reflection of a real
///   point on the water — just a point a few inches away from the one under the pixel. Nothing can
///   repeat, because nothing is being shifted; the surface is simply being read where the wave has
///   moved it. Ratio 0.98 against a waves-off floor of 0.85, with the hit rate at 37.0%.
///
/// A wave displaces the water by a distance, so this is in yards and needs no conversion. Raising
/// it deepens the wobble without any risk of the repeats coming back — the failure mode it
/// replaces is not on this axis at all.
// Default 3, carried in `water_reflect.march.y` 
const RIPPLE_ORIGIN_YD: f32 = 3.0;


/// How far from the edge of the frame the march's confidence starts falling, in UV.
///
/// **The one artefact every screen-space reflection has**, and the only honest handling of it: the
/// ray can only find what the camera drew, so a reflection whose source is off-screen has to stop
/// existing — and it must stop *gradually*, or the boundary is a hard line across the water that
/// moves whenever the camera does. Where this fades out, the planar capture and the sky mix
/// underneath it are still there to take over.
const SSR_EDGE_FADE: f32 = 0.14;

/// A screen-space reflection hit: what was there, and how much to believe it.
struct SsrHit {
    rgb: vec3<f32>,
    conf: f32,
}

/// The sky a surface facing `dir` reflects, off the dome's own zenith→horizon gradient.
///
/// **This is what makes a wave visible when the sun is not behind it.** Mixing the reflection
/// toward one flat colour — which is what the scene fog was standing in for — cannot show a ripple:
/// at the grazing angles you look at water from, a tilted facet and a flat one both have Fresnel
/// near 1, so both come back the same colour and the whole surface reads as a sheet of glass except
/// along the sun's own glitter path. Reflecting the view ray and reading the sky at the reflected
/// ray's ELEVATION gives every facet a different colour instead: a wave's near face looks at the
/// horizon and its far face at the zenith, and those two are as far apart as the sky is.
///
/// It is also what carries the distance. Out toward the horizon the fine ripples have mipped away
/// by design (they would otherwise moiré) and only the long swell is left — and a long swell tilts
/// the surface by very little, which is invisible under a flat sky mix and plainly visible under a
/// gradient.
///
/// `sqrt` on the blend because the dome's own gradient is weighted toward its horizon stop, and a
/// linear ramp put the horizon colour only in the last few degrees above it.
fn sky_reflection(dir: vec3<f32>) -> vec3<f32> {
    let t = sqrt(saturate(dir.y));
    return mix(water_reflect.sky_horizon.rgb, water_reflect.sky_zenith.rgb, t);
}

/// Foam noise that does not repeat.
///
/// The ripple map is one 256-texel tile and the shader reads it in world yards, so a single tap of
/// it at the bubbles' own scale repeats about every ten inches — close enough together to read as a
/// printed pattern stamped along the shore rather than as foam, which is what the report meant by
/// "the noise is repeating". Three things break it, and it takes all three:
///
/// * **Two taps at scales with no common multiple**, so neither one's period is the pair's.
/// * **On lattices rotated against each other**, so their seams cross instead of stacking.
/// * **Domain-warped** by a third, much coarser tap: the lookups are displaced by a field that
///   itself varies over tens of yards, so what repeats is only the field being used to displace
///   itself. This is the one that does most of the work — a warp of a third of a tile moves the
///   repeat further than the eye can carry it.
///
/// ## Explicit gradients, and why the value brings its own footprint back
///
/// The band this feeds is drawn only within [`FOAM_REACH`] of the waterline, which is per-fragment
/// control flow — and there an implicit-LOD `textureSample` is undefined and naga rejects it, as
/// does any derivative builtin. So the caller takes `d(world xz)/d(pixel)` in uniform flow and
/// hands it down, and every tap here is a `textureSampleGrad`.
///
/// `fw` comes back for the same reason: the threshold that cuts these flecks softens itself by the
/// noise's own screen footprint, and `fwidth` of the result is no longer available to measure it
/// with. It does not have to be. **The map's R/G is the gradient of its B** — that is what it holds
/// and what `ripple_layer` reads it as (`benilla_world::liquid::ripple`) — so the same fetches that
/// give the value give its slope, and the footprint is that slope against the pixel's own step.
/// The stored slope is a central difference over two texels of a [`RIPPLE_TEXELS`]-wide map, so
/// `d(b)/d(uv)` is the decoded value times half the map's width.
///
/// The warp's contribution to the gradients is dropped: it varies over tens of yards, where the
/// terms it displaces vary over fractions of one.
struct FoamNoise {
    /// The field, 0..1.
    v: f32,
    /// How much `v` moves across one pixel — `fwidth(v)`, computed rather than sampled.
    fw: f32,
}

fn foam_noise(p: vec2<f32>, t: f32, ddx: vec2<f32>, ddy: vec2<f32>) -> FoamNoise {
    // cos/sin of ~33.9 degrees — a turn that is not a neat fraction of one.
    let rot = vec2<f32>(0.8305, 0.5570);
    let r = mat2x2<f32>(rot.x, -rot.y, rot.y, rot.x);
    let warp = textureSampleGrad(
        ripples,
        ripples_samp,
        p * 0.031 + vec2<f32>(0.004, -0.003) * t,
        ddx * 0.031,
        ddy * 0.031,
    ).rg * 2.0 - 1.0;
    let a_dx = ddx * 0.83;
    let a_dy = ddy * 0.83;
    let a = textureSampleGrad(
        ripples,
        ripples_samp,
        p * 0.83 + warp * 0.35 + vec2<f32>(0.010, 0.014) * t,
        a_dx,
        a_dy,
    );
    let b_dx = (r * ddx) * 1.97;
    let b_dy = (r * ddy) * 1.97;
    let b = textureSampleGrad(
        ripples,
        ripples_samp,
        (r * p) * 1.97 + warp * 0.20 + vec2<f32>(-0.021, 0.008) * t,
        b_dx,
        b_dy,
    );
    // d(b)/d(uv), out of the same texels — see the header.
    let ga = (a.rg * 2.0 - 1.0) * (RIPPLE_TEXELS * 0.5);
    let gb = (b.rg * 2.0 - 1.0) * (RIPPLE_TEXELS * 0.5);
    let fw_a = abs(dot(ga, a_dx)) + abs(dot(ga, a_dy));
    let fw_b = abs(dot(gb, b_dx)) + abs(dot(gb, b_dy));
    var out: FoamNoise;
    out.v = a.b * 0.55 + b.b * 0.45;
    out.fw = fw_a * 0.55 + fw_b * 0.45;
    return out;
}

/// The probe's lookup direction, **box-projected** — the whole of what makes a cubemap anchored to
/// the water worth more than one anchored to the eye.
///
/// A cube is a picture taken from ONE point, so reading it in the raw reflected direction is right
/// only for a fragment standing at that point; everywhere else the reflection slides as the camera
/// moves. The correction decides where the ray would actually LAND and looks up the direction from
/// the probe to that landing instead.
///
/// **Against a box, not a sphere, and that is the difference between this working and not.** This
/// is Unity's box projection and Unreal's box capture, and the rule both state is that the proxy
/// has to match where the reflected scenery stands, because the scene is reprojected onto it. A
/// lake is a floor with walls: bank at the shoreline in every direction, sky overhead. A box has
/// that shape, so a shallow ray meets a WALL and finds the bank and its trees while a steep one
/// meets the CEILING and finds sky. A sphere curves away uniformly — at any radius big enough to
/// clear the near bank it also arcs over the far treeline — so every lookup landed in open sky and
/// the tier returned a blue wash at 110 yards and again at 45.
///
/// The arithmetic is the standard one: for each axis, how far along the ray the relevant face lies;
/// the smallest of the three is the face actually hit; then rebase onto the probe.
fn probe_dir(p: vec3<f32>, d: vec3<f32>, centre: vec3<f32>, lo: vec3<f32>, hi: vec3<f32>) -> vec3<f32> {
    // A zero component gives an infinite factor, which `min` discards — that axis is never hit.
    let factors = (select(lo, hi, d > vec3<f32>(0.0)) - p) / d;
    let scalar = min(min(factors.x, factors.y), factors.z);
    return d * scalar + (p - centre);
}

/// The same correction against a **sphere** — the environment painted on a shell at one distance,
/// with the water showing that shell's inside.
///
/// Where the box assumes flat walls at a different distance per bearing, this assumes one distance
/// in every direction, which is the better fit when a body of water is ringed at a roughly even
/// remove. The far root is the one wanted: a reflection looks at what the ray meets on its way OUT
/// of the shell, and the near root lies behind the fragment.
fn probe_dir_sphere(p: vec3<f32>, d: vec3<f32>, centre: vec3<f32>, radius: f32) -> vec3<f32> {
    let oc = p - centre;
    let b = dot(oc, d);
    let c = dot(oc, oc) - radius * radius;
    let disc = b * b - c;
    if (disc <= 0.0) {
        return d;
    }
    let t = -b + sqrt(disc);
    if (t <= 0.0) {
        return d;
    }
    return p + d * t - centre;
}

/// How much this fragment believes the probe, by where it stands — **in three dimensions**.
///
/// The horizontal term is the obvious one: a cube of this lake should not be reflecting in the next
/// valley. The vertical term is the one that matters, and it is much tighter. Water bodies stack:
/// the box around this probe holds a lake, a second pool seven yards up and a stream running sixty
/// to eighty above that. They are tens of yards apart in height and hundreds in plan, so height is
/// the axis that actually tells them apart. Without this term one probe reproduces the "cube of a
/// pond behind a bank" failure on its own, with no second probe and no selector required — and it
/// does so invisibly, because a cube of somewhere else is still a plausible reflection.

/// One probe's contribution: correct the reflected direction against THAT probe's proxy, then read
/// its own slice of the cube array. `idx` is the slot, which is also the array index — the capture
/// writes six layers starting at `slot * 6`, and a cube-array lookup addresses them as one cube.
/// The positive `t` where the ray `o + t*d` (with `o` measured from the capture point) is `dist`
/// away from that point — one step of the depth correction, in closed form.
///
/// `|o + t*d|^2 = dist^2` with `|d| = 1` is `t^2 + 2t(o.d) + |o|^2 - dist^2 = 0`. A negative
/// discriminant means the ray never reaches that distance, which happens at silhouettes where the
/// sampled distance belongs to something the ray passes beside; the nearest approach is the least
/// wrong answer available and keeps the iteration from producing a NaN.
fn probe_t_for(o: vec3<f32>, d: vec3<f32>, dist: f32) -> f32 {
    let b = dot(o, d);
    let disc = b * b - dot(o, o) + dist * dist;
    if (disc < 0.0) {
        return max(-b, 0.0);
    }
    return max(-b + sqrt(disc), 0.0);
}

/// Correct the reflected direction against the cube's own stored DISTANCES rather than against a
/// proxy shape.
///
/// **This is what removes the tuned box.** A box proxy asserts that the whole world stands on one
/// rectangle, so it can be right for the far treeline (which is what a grazing ray hits) or for the
/// cottage ten yards past the bank, but not both — and whichever distance the constant is set to is
/// some other lake's error. The capture already knows where everything actually is: `probe_face.wgsl`
/// writes the distance from the capture point into the cube's alpha, so the geometry is per-texel
/// and per-direction instead of one number fitted at a development site.
///
/// The iteration is the obvious fixed point. Guess a direction, ask the cube how far away the
/// geometry is that way, move along the reflected ray to exactly that distance, and re-derive the
/// direction from the capture point. Each round makes the direction and the distance more nearly
/// agree; four is past the point where the picture changes at these face sizes.
fn probe_dir_depth(p: vec3<f32>, d: vec3<f32>, c: vec3<f32>, idx: i32, steps: i32) -> vec3<f32> {
    let o = p - c;
    var dir = d;
    for (var k = 0; k < steps; k = k + 1) {
        let dist = textureSampleLevel(probe_tex, probe_samp, dir, idx, 0.0).a;
        // Nothing drawn this way — sky, or past the far clip. Sky IS at infinity, and the
        // uncorrected direction is exactly the right answer for it.
        if (dist > 9000.0) {
            return d;
        }
        dir = normalize(o + probe_t_for(o, d, dist) * d);
    }
    return dir;
}

/// Steps of [`probe_dir_march`]'s search, the nearest and furthest it looks along the reflected
/// ray (yards), and its bisection steps.
const PROBE_MARCH_STEPS: i32 = 16;
const PROBE_MARCH_NEAR: f32 = 0.5;
const PROBE_MARCH_FAR: f32 = 400.0;
const PROBE_MARCH_REFINE: i32 = 4;

/// Correct the reflected direction by **marching the reflected ray** against the cube's stored
/// distances: step outward from the water point, and at each point ask the cube how far the
/// geometry is in that point's direction from the capture point; the first point at or just past
/// that distance is the hit, refined by bisection. Sky is the answer only when the whole ray passes
/// nothing.
///
/// The fixed point [`probe_dir_depth`] starts from the uncorrected direction and gives up if that
/// sees sky — and from the capture point, the direction to a crown's reflection often passes above
/// the crown, so tree tops read as sky and reflected trees came out short.
///
/// A point further than `dt` behind the stored surface is the ray passing behind something the
/// probe sees in front of it, not a hit, and the march goes on.
fn probe_dir_march(p: vec3<f32>, d: vec3<f32>, c: vec3<f32>, idx: i32) -> vec3<f32> {
    let o = p - c;
    let g = pow(PROBE_MARCH_FAR / PROBE_MARCH_NEAR, 1.0 / f32(PROBE_MARCH_STEPS - 1));
    var t_prev = 0.0;
    var t = PROBE_MARCH_NEAR;
    for (var k = 0; k < PROBE_MARCH_STEPS; k = k + 1) {
        let q = o + t * d;
        let dist = textureSampleLevel(probe_tex, probe_samp, q, idx, 0.0).a;
        let over = length(q) - dist;
        if (dist < 9000.0 && over >= 0.0 && over <= max(t - t_prev, 1.0)) {
            var lo = t_prev;
            var hi = t;
            for (var j = 0; j < PROBE_MARCH_REFINE; j = j + 1) {
                let mid = 0.5 * (lo + hi);
                let qm = o + mid * d;
                if (length(qm) >= textureSampleLevel(probe_tex, probe_samp, qm, idx, 0.0).a) {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            return normalize(o + hi * d);
        }
        t_prev = t;
        t = t * g;
    }
    return d;
}

/// **How blurred this probe should be read, derived rather than dialled.**
///
/// The probe no longer tries to put reflections in the right place — a long measurement pass
/// established that a parallax-corrected cube's drift away from its capture point is inherent, and
/// that a sharper march, a bigger face and a world-space landing consensus each moved it barely or
/// not at all. What is left is to stop the error being READABLE, which is what the prefilter chain
/// in `probe_filter.wgsl` is for and what this indexes.
///
/// The amount of blur is not a taste setting. Standing `d_probe` from the capture point, geometry
/// `d_geom` away reflects displaced by roughly `d_probe / d_geom` radians — small angles, and the
/// ratio is the whole of it. A mip-0 texel subtends `(pi/2) / face` radians, and each level down
/// doubles that. So the level whose texel is the size of the error is
/// `log2((d_probe / d_geom) / texel0)`, and at that level the mistake is smaller than the thing it
/// is written on, which is the definition of not being able to see it.
///
/// That derivation is why there is no tuned constant here and why it needs none at the next lake:
/// both distances are measured per fragment, and the texel angle is a property of the cube. The
/// FLOOR is the one authored number, and it is a look decision rather than a correction — a probe
/// read perfectly sharp invites the comparison with the planar mirror that it loses, so the tier is
/// never allowed to be a mirror even standing on top of a probe.
fn probe_lod(d_probe: f32, d_geom: f32) -> f32 {
    let texel0 = water_reflect.probe_box_min.y;
    let lo = water_reflect.probe_box_min.z;
    let hi = water_reflect.probe_box_max.z;
    if (texel0 <= 0.0) {
        return lo;
    }
    // `d_geom` is `PROBE_NO_HIT` for sky, which is correct and needs no special case: sky at ten
    // thousand yards has no parallax to hide and the ratio drives the level to the floor by itself.
    let err = d_probe / max(d_geom, 1.0);
    return clamp(lo + log2(max(err / texel0, 1.0)), lo, hi);
}

fn sample_probe(sl: ProbeSlot, idx: i32, p: vec3<f32>, r: vec3<f32>, steps: i32) -> vec3<f32> {
    // `probe_extra.z` picks the proxy: the depth correction by default, the old analytic shapes
    // so the two can be compared in one build rather than across two.
    var dir: vec3<f32>;
    if (water_reflect.probe_extra.z > 1.5) {
        dir = probe_dir_march(p, r, sl.at.xyz, idx);
    } else if (water_reflect.probe_extra.z > 0.5) {
        dir = probe_dir_depth(p, r, sl.at.xyz, idx, steps);
    } else {
        dir = select(
            probe_dir(p, r, sl.at.xyz, sl.box_min.xyz, sl.box_max.xyz),
            probe_dir_sphere(p, r, sl.at.xyz, sl.box_max.w),
            sl.box_max.w > 0.5,
        );
    }
    // Level 0 for the DISTANCE and a chosen level for the colour. The chain's alpha carries the
    // centre tap's distance rather than a blurred one, but reading it at level 0 keeps the geometry
    // exact regardless of how soft the picture is being read.
    let d_geom = textureSampleLevel(probe_tex, probe_samp, dir, idx, 0.0).a;
    let lod = probe_lod(distance(p, sl.at.xyz), d_geom);
    let centre = textureSampleLevel(probe_tex, probe_samp, dir, idx, lod).rgb;
    // A small softening (`march.z`, in cube texels): four more reads a fixed
    // angle around the direction, averaged with it, so the capture's texels do not read as pixels
    // on the water.
    let spread = water_reflect.march.z * water_reflect.probe_box_min.y;
    if (spread <= 0.0) {
        return centre;
    }
    let helper = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(dir.y) > 0.9);
    let t1 = normalize(cross(dir, helper)) * spread;
    let t2 = cross(dir, t1);
    var sum = centre;
    sum = sum + textureSampleLevel(probe_tex, probe_samp, dir + t1, idx, lod).rgb;
    sum = sum + textureSampleLevel(probe_tex, probe_samp, dir - t1, idx, lod).rgb;
    sum = sum + textureSampleLevel(probe_tex, probe_samp, dir + t2, idx, lod).rgb;
    sum = sum + textureSampleLevel(probe_tex, probe_samp, dir - t2, idx, lod).rgb;
    return sum * 0.2;
}

fn probe_weight(p: vec3<f32>, centre: vec3<f32>, reach: f32, lift: f32) -> f32 {
    let flat = length(p.xz - centre.xz);
    let rise = abs(p.y - centre.y);
    return (1.0 - smoothstep(reach * 0.6, reach, flat))
        * (1.0 - smoothstep(lift * 0.5, lift, rise));
}

// ---- lava ------------------------------------------------------------------------------------
//
// **Outside the `DEPTH_PREPASS` guard below, and that placement is load-bearing.** The march and
// everything it needs only exist when the depth prepass does, and on the faithful lane it does not
// (`liquid::depth` arms the prepass for the stylised look only). Magma is drawn on BOTH lanes, so a
// lava helper defined inside that guard is missing exactly where the reference needs the shader to
// compile — and a liquid shader that fails to compile does not fall back, it drops the draw: every
// magma surface in the zone stops being drawn at all. That is what happened. It was invisible from
// the stylised lane, which has the prepass and compiled fine, and the only sign was one
// `failed to process shader` line in the log under a pile of unrelated asset warnings.
/// One slow scalar octave of the ripple map, read in world yards.
///
/// `textureSampleLevel` rather than `textureSample`: this is called under the kind branch, and an
/// implicit-LOD fetch there needs a uniformity proof naga will not give it. Level 0 is the honest
/// level anyway — the features are tens of yards across, so there is nothing for a mip to prefilter.
fn lava_field(world_xz: vec2<f32>, wavelength_yd: f32, drift: vec2<f32>, t: f32) -> f32 {
    let uv = world_xz / wavelength_yd + drift * t;
    return textureSampleLevel(ripples, ripples_samp, uv, 0.0).b;
}

/// Grade the kind's animated sheet into crust and molten seam — see [`LAVA_CRUST`].
///
/// Takes the sheet's colour as it comes and only scales it, so a zone that authored dull red magma
/// keeps dull red magma and one that authored orange keeps orange. The drifts are deliberately an
/// order below the water's: lava moves, but it moves like something with a skin on it.
fn stylised_lava(sheet: vec3<f32>, world_xz: vec2<f32>, t: f32) -> vec3<f32> {
    let plates = lava_field(world_xz, LAVA_PLATE_YD, vec2<f32>(0.0041, 0.0024), t);
    let breaks = lava_field(world_xz, LAVA_BREAK_YD, vec2<f32>(-0.0063, 0.0038), t);
    // Thin crust = hot seam. The two octaves are weighted so the plates decide and the breaks only
    // roughen their edges; an even mix reads as noise rather than as plates.
    let crust = saturate(plates * 0.74 + breaks * 0.26);
    let seam = 1.0 - smoothstep(0.34, 0.62, crust);
    // The sheet is already painted with hot and cool regions; take them rather than fight them, so
    // the seams land where the artist put the bright parts and the plates where they did not.
    let lum = dot(sheet, vec3<f32>(0.299, 0.587, 0.114));
    let molten = saturate(max(seam, smoothstep(0.35, 0.85, lum)));
    return sheet * mix(LAVA_CRUST, LAVA_GLOW, molten);
}

#ifdef DEPTH_PREPASS



/// Read the scene snapshot where the ray landed, and score how much to believe it.
///
/// The ripple lands on the ray's ORIGIN, not here and not on the ray — see [`RIPPLE_ORIGIN_YD`].
/// Clamped inside the frame
/// because the snapshot has nothing outside it, and the confidence is measured on the DISPLACED UV
/// so a wobble that walks off the edge fades out with everything else rather than clamping to a
/// smeared border pixel.
///
/// The ripple's displacement of the march's read, in UV — a fixed distance in YARDS, converted per
/// fragment.
///
/// **A constant UV offset is the wrong shape and it is why the far water smeared.** The reflection
/// is read by displacing the hit's UV, and at a grazing angle the water is compressed hard into the
/// frame: near the horizon a fifth of a screen-width of UV spans hundreds of yards of surface, so a
/// fixed `0.055` there does not crinkle the reflection, it tears it and samples something
/// unrelated. Close to the camera the same number is a gentle wobble. That difference is the
/// artefact — long stretched bands of repeated foliage where the compression is worst, reported as
/// "ssr has repetetive patterns in the reflection like a longstretched repeated texture tree crown"
/// — and it is made worse by the ripple being a TILING pattern, so the tearing repeats with it.
///
/// A wave displaces what you see by some distance across the water, not by some fraction of the
/// screen, so the amplitude is authored in yards and converted here. `fwidth(world_pos.xz)` is how
/// much world a pixel covers at this fragment, which is exactly the conversion factor and costs two
/// derivatives. Near water keeps its wobble, far water stops tearing, and neither needed a distance
/// threshold to be chosen.
///
/// **This deliberately diverges from `REFLECT_DISTORT`**, which the constant above was matched to so
/// the two tiers would wobble alike. They still do where it can be seen — the yard figure is set to
/// match what `0.055` meant at close range — but the planar tier carries the same compression bug
/// untouched here, because it was not what was reported and its capture is a coherent mirror that
/// tears far less visibly when over-displaced.

/// Shared by both of the march's terminations — opaque geometry and sky — because the two differ
/// only in how the UV was arrived at, never in what is done with it.
fn ssr_read(uv: vec2<f32>) -> SsrHit {
    var out: SsrHit;
    let read_uv = clamp(uv, vec2<f32>(0.002), vec2<f32>(0.998));
    // Four bilinear taps on the diagonals — [`sample_mirror`]'s kernel, and deliberately the same
    // one. No premultiplied divide here: the snapshot is a copy of an opaque frame, so every tap
    // has something in it and there is no coverage to weight by. See [`SSR_READ_BLUR`].
    let o = SSR_READ_BLUR / vec2<f32>(textureDimensions(scene_tex));
    var acc = textureSampleLevel(scene_tex, scene_samp, read_uv + vec2<f32>(-o.x, -o.y), 0.0).rgb;
    acc += textureSampleLevel(scene_tex, scene_samp, read_uv + vec2<f32>(o.x, -o.y), 0.0).rgb;
    acc += textureSampleLevel(scene_tex, scene_samp, read_uv + vec2<f32>(-o.x, o.y), 0.0).rgb;
    acc += textureSampleLevel(scene_tex, scene_samp, read_uv + vec2<f32>(o.x, o.y), 0.0).rgb;
    out.rgb = acc * 0.25;
    out.conf = ssr_edge_conf(read_uv);
    return out;
}

/// How much a march that ended at `uv` is believed on account of where on the screen that is: full
/// in the frame, fading to nothing over [`SSR_EDGE_FADE`] at its border, where the tier hands over to
/// whatever is under it.
fn ssr_edge_conf(uv: vec2<f32>) -> f32 {
    let read_uv = clamp(uv, vec2<f32>(0.002), vec2<f32>(0.998));
    let edge = min(
        min(read_uv.x, 1.0 - read_uv.x),
        min(read_uv.y, 1.0 - read_uv.y),
    );
    return smoothstep(0.0, SSR_EDGE_FADE, edge);
}

/// How many cells the traversal may visit before giving up — AMD's FidelityFX SSSR sample runs its
/// `FFX_SSSR_HierarchicalRaymarch` with 128, and this is that figure.
const HIZ_MAX_STEPS: i32 = 128;

/// How far past a cell wall the ray is placed when it leaves the cell, as a fraction of a
/// level-zero texel — FFX SSSR's `uv_offset`, and its value.
///
/// **In SCREEN units, and that is the fix.** The first version of this traversal parametrised the
/// ray in world yards and stepped `1e-4` of a yard past each wall. For a ray running toward the
/// horizon a ten-thousandth of a yard is a vanishing fraction of a texel, so the step often failed
/// to leave the cell at all and the ray sat on one wall until its budget ran out: at the
/// Stranglethorn camera 23% of water fragments exhausted it, 58% of the far field, and every one of
/// them had stalled. Measured in texels the overshoot is the same size everywhere on the screen.
const HIZ_CROSS: f32 = 0.005;

/// How thick every surface in the depth buffer is taken to be, in yards — McGuire & Mara's
/// `zThickness` ("Efficient GPU Screen-Space Ray Tracing", JCGT 2014): each depth sample is a slab
/// from its surface to this far behind it, and a ray hits only inside a slab.
///
/// A depth buffer records front faces and nothing behind them, so some thickness has to be
/// assumed. Too little and a ray that meets a surface at a grazing angle slips past it between
/// samples; too much and a ray that passed behind something thin — a trunk, a branch — is taken to
/// have hit it, and paints it where it does not reach. The old march's constant, kept.
const SSR_THICKNESS: f32 = 6.0;

/// Is a ray at reverse-Z depth `ray` BEHIND the slab of the surface at `depth` — farther than the
/// point [`SSR_THICKNESS`] yards behind that surface?
///
/// Bevy's view is an infinite reverse-Z perspective, where depth is `near / distance`, so the back
/// of the slab sits at `near·d / (near + T·d)` and the test is `ray < near·d / (near + T·d)`. Every
/// quantity is positive, so it is asked as `ray·(near + T·d) < near·d` instead — two fused
/// multiply-adds and no divide. That form is not cosmetic: the compiler hoists this arithmetic out
/// of the branch it sits in and runs it on every step of the march, taken or not, and with the
/// divide there (a reciprocal) merely compiling the pass-behind in cost 0.26 ms of the water's pass
/// at Mirror Lake with the branch never taken. Depth zero (sky) is never passed behind: the right
/// side is zero.
fn behind_slab(ray: f32, depth: f32, near: f32) -> bool {
    return ray * (near + SSR_THICKNESS * depth) < near * depth;
}

/// Project a world point to the traversal's space: `xy` the screen UV, `z` reverse-Z device depth.
fn hiz_screen(p: vec3<f32>) -> vec3<f32> {
    let ndc = position_world_to_ndc(p);
    return vec3<f32>(ndc_to_uv(ndc.xy), ndc.z);
}

/// Walk the reflected ray through `liquid::hiz`'s pyramid and read the scene snapshot where it
/// lands.
///
/// **This is AMD FidelityFX SSSR's `FFX_SSSR_HierarchicalRaymarch`**, ported rather than derived
/// (GPUOpen-Effects/FidelityFX-SSSR, `ffx_sssr.h`, built with `FFX_SSSR_INVERTED_DEPTH_RANGE`
/// because Bevy is reverse-Z). Its three load-bearing choices, each of which the first version of
/// this function got differently and paid for:
///
/// * **The ray lives in screen space** — UV and device depth. A perspective projection maps lines
///   to lines, so a world ray is a straight line in that space too, and every cell wall and every
///   depth plane is an axis-aligned plane the ray meets at a closed-form parameter.
/// * **Leaving a cell overshoots by a fraction of a texel** ([`HIZ_CROSS`]), so it always leaves.
/// * **A ray that is still in front of the tile's nearest surface but would pass it inside the
///   tile advances to that depth before descending**, rather than descending where it stands.
///
/// The pyramid holds the NEAREST depth per tile, so the rule is: in front of the tile's nearest
/// surface and leaving through a wall, the whole tile is provably empty along the ray — skip it and
/// go up a level; otherwise go down one. Falling below level zero is a hit.
///
/// **This is the half of the water's reflection that a plane cannot do.** The direction traced is
/// `reflect(-V, N)` on the fragment's GEOMETRIC normal ([`surface_normal`]) — the heightfield's own
/// facet, so a sloped stream is traced at its real orientation and needs no mirror built for it.
///
/// The fine ripple is deliberately NOT in that direction: it slides the ray's ORIGIN across the
/// surface instead, so neighbouring pixels keep one direction and the march stays coherent while
/// every sample is still a true reflection of a real point. [`RIPPLE_ORIGIN_YD`] carries the
/// measurements and why the other two placements fail. What it cannot do is see off the screen,
/// which is why it returns a confidence rather than just a colour: where the ray leaves the frame
/// the planar capture and the sky mix behind it take over. Under [`WaterStyle::StylisedSsr`] there is no capture behind it and
/// the sky mix alone catches what it misses, which is the trade that lane is offered for.
///
/// The water itself is not in the prepass (Bevy excludes alpha-blended materials, and
/// `liquid::depth`'s doc keeps it that way deliberately), so there is no self-intersection to bias
/// against — the first step off the surface is already reading opaque geometry only.
fn ssr_trace(origin: vec3<f32>, dir: vec3<f32>) -> SsrHit {
    var out: SsrHit;
    out.rgb = vec3<f32>(0.0);
    out.conf = 0.0;

    // The direction as the difference of two projected points. The second point has to be in
    // front of the eye for the difference to mean anything; `clip.w` is view depth, so a ray
    // heading back toward the camera is cut to half the distance to the eye plane.
    let clip_o = position_world_to_clip(origin);
    let clip_d = direction_world_to_clip(dir);
    var reach = clip_o.w;
    if (clip_d.w < 0.0) {
        reach = min(reach, 0.5 * clip_o.w / -clip_d.w);
    }
    let o = hiz_screen(origin);
    let d = hiz_screen(origin + dir * reach) - o;
    let big = 3.4e38;
    let inv_d = vec3<f32>(
        select(big, 1.0 / d.x, d.x != 0.0),
        select(big, 1.0 / d.y, d.y != 0.0),
        select(big, 1.0 / d.z, d.z != 0.0),
    );

    let top = i32(textureNumLevels(hiz_tex)) - 1;
    let base_res = vec2<f32>(textureDimensions(hiz_tex, 0));
    let cross = select(vec2<f32>(HIZ_CROSS), vec2<f32>(-HIZ_CROSS), d.xy < vec2<f32>(0.0)) / base_res;
    let wall = select(vec2<f32>(1.0), vec2<f32>(0.0), d.xy < vec2<f32>(0.0));

    // Step off the origin's own cell first, so the walk does not open by intersecting it.
    var level = 0;
    var t: f32;
    {
        let planes = (floor(o.xy * base_res) + wall) / base_res + cross;
        let tw = (planes - o.xy) * inv_d.xy;
        t = min(tw.x, tw.y);
    }
    var p = o + t * d;

    // **Loop invariants, read once.** Both come from buffers — the lever from the water's storage
    // buffer, the near plane from the view uniform — and a load the compiler does not hoist is a
    // memory round trip on every step of a loop whose whole cost is its latency chain. Read inside
    // the loop, merely compiling the pass-behind in (branch never taken) cost 0.28 ms of the
    // water's pass at Mirror Lake against 0.09 for the pass-behind work itself.
    let pass_behind = water_reflect.march.x > 0.5;
    let near_plane = perspective_camera_near();

    var i = 0;
    for (; i < HIZ_MAX_STEPS && level >= 0; i = i + 1) {
        if (any(p.xy < vec2<f32>(0.0)) || any(p.xy >= vec2<f32>(1.0))) {
            return out; // off the frame: the screen cannot say
        }
        if (p.z <= 0.0) {
            return out; // past the far plane
        }
        let res = vec2<f32>(textureDimensions(hiz_tex, level));
        let cell = floor(p.xy * res);
        // The nearest surface anywhere in the tile. The farthest is fetched below, only if needed.
        let surface = textureLoad(hiz_tex, vec2<i32>(cell), level).r;

        // **A tile of pure sky, at any level.** The pyramid reduces with `max` and sky is the far
        // plane, so a cell reporting zero has nothing in it anywhere — the ray is over open sky.
        // That is a HIT: reflecting a surface below the eye sends the ray up and forward, at sky
        // that is on the screen, and treating it as a miss was most of why this mode fell back to
        // a flat colour over open water. Getting it from the hierarchy costs nothing extra.
        //
        // **The colour is the dome's along the reflected ray, not the colour buffer's here.** It
        // used to be `ssr_read(p.xy)`, which trusts that a pixel with no depth shows sky — and the
        // colour buffer can hold things the depth does not: anything drawn without writing it.
        // Every such pixel then became a sky hit that read the object's colour, the same few pixels
        // for every water row whose ray climbed that column, which is how a palm missing from the
        // depth came out as a stalk (Stefan's Stranglethorn report). The dome is a pure function
        // of direction, so the sky can be asked for instead of read: an object without depth then
        // reads as the sky behind it rather than being smeared down the water.
        //
        // It is also the more correct sky. Seen in the water, the sky is what lies along the
        // reflected ray — its own vanishing point — not whatever the screen shows where the ray
        // happened to leave the geometry. Reading at the entry point quantised that to the tile
        // the walk was on, which is what drew the blocky light rectangles in sky reflections.
        //
        // The edge fade stays keyed to where the ray left the frame's geometry, so the hand-over
        // to the tier underneath at the screen border is exactly what it was.
        if (surface <= 0.0) {
            if (water_reflect.dome.warp.z > 0.5) {
                return ssr_read(p.xy);
            }
            var sky_hit: SsrHit;
            sky_hit.rgb = dome_color(water_reflect.dome, normalize(dir));
            sky_hit.conf = ssr_edge_conf(p.xy);
            return sky_hit;
        }

        // FFX_SSSR_AdvanceRay. The cell's two exit walls and its nearest-depth plane; the depth
        // plane only counts for a ray moving away from the eye, which under reverse-Z is `d.z < 0`.
        let planes = vec3<f32>((cell + wall) / res + cross, surface);
        var tp = (planes - o) * inv_d;
        tp.z = select(big, tp.z, d.z < 0.0);
        let t_wall = min(tp.x, tp.y);
        let t_min = min(t_wall, tp.z);
        // Larger depth is nearer under reverse-Z: the ray is in front of everything in the tile.
        let in_front = surface < p.z;
        // **Or behind everything in it** — the half FFX SSSR does not have, and the reason there is
        // a farthest-depth pyramid beside the nearest. Every surface is a slab [`SSR_THICKNESS`] deep, so a
        // ray that stays behind the back of the FARTHEST surface's slab for its whole way across
        // the tile has passed behind all of them, and the tile can be skipped whole exactly as an
        // empty one in front is. Its nearest point in the tile is whichever end is nearer.
        //
        // Without it the ray could only descend onto the first surface it went behind, and a ray
        // going under a tree crown — which the water sees and the camera does not — met the
        // crown's front leaves 24 yards and more in front of it and had to be thrown away: the
        // black fragments in reflected canopies (Stefan's Stranglethorn report; 74% of those
        // rejects were that far behind). Now it walks on behind the crown to whatever it really
        // meets. What it cannot know is anything hidden behind the crown that the screen never
        // held; there it finds the next thing that IS on screen, which is the known cost of the
        // method and why Bevy's own marcher keeps it off for colour.
        //
        // **Only a ray that is NOT in front needs the farthest surface**, so only then is it read:
        // most steps cross empty space in front of everything, and reading both pyramids on every
        // step cost the water's pass nearly a millisecond at Mirror Lake as one 64-bit texel.
        var behind = false;
        // **The lever first, on its own.** `pass_behind` is the same for every lane, so this is a
        // scalar branch the compiler does not speculate past. Folded into one condition with the
        // per-lane `in_front`, it hoisted the whole test onto every step instead, and merely having
        // it compiled in with the lever OFF cost 0.24 ms of the water's pass at Mirror Lake; apart,
        // off measures level with the shader that never had it.
        if (pass_behind) {
        if (!in_front) {
            let p_exit_z = o.z + t_wall * d.z;
            let seg_near = max(p.z, p_exit_z);
            let seg_far = min(p.z, p_exit_z);
            // **Only then, and only if the ray is past the NEAREST surface's slab too.** A ray
            // still within it cannot be behind the farthest's (that slab is never nearer) nor
            // between the two, so the fetch below would be for nothing — and that is the common
            // case: a ray closing in on a hit. Measured at Mirror Lake before this gate, the pass
            // cost 0.44 ms of the water's pass over the lever off.
            if (behind_slab(seg_near, surface, near_plane)) {
                let farthest = textureLoad(hiz_far_tex, vec2<i32>(cell), level).r;
                let behind_all = behind_slab(seg_near, farthest, near_plane);
                // **At the finest level a texel is two layers, and the ray may be between them.**
                // A level-zero texel covers one and a half to three and a half screen pixels, so
                // at the porous edge of a crown most of them hold a leaf AND whatever shows through
                // beside it. A ray behind the leaf's slab but still in front of the far layer over
                // its whole way across is in the empty space between the two and walks on; without
                // this, one mixed texel on the path stopped the ray and it was rejected — measured:
                // with only the whole-tile skip the canopy's rejects barely moved (14.9% to 13.6%).
                // A single-layer texel is untouched (its two slabs are one).
                let between = level == 0 && seg_far > farthest;
                behind = behind_all || between;
            }
        }
        }
        let skipped = (t_min != tp.z && in_front) || behind;
        // **One position update per step.** The march loop is where this shader spends its time,
        // and its cost is instructions per step — written as two branches each recomputing the
        // position, the compiler ran both and merged them with selects.
        let t_next = select(select(t, t_wall, behind), t_min, in_front);
        t = t_next;
        p = o + t * d;
        level = select(level - 1, min(level + 1, top), skipped);
    }
    if (level >= 0) {
        return out; // budget
    }

    // FFX_SSSR_ValidateHit, the parts that apply. Off the frame is no hit; neither is a ray that
    // never really left its own texel.
    if (any(p.xy < vec2<f32>(0.0)) || any(p.xy >= vec2<f32>(1.0))) {
        return out;
    }
    // Against the texel's NEAREST surface when rays cannot pass behind — the test as it was. When
    // they can, a finest texel may hold two layers (a leaf and the bank behind it), the walk only
    // stops inside one of their slabs, and the gap is measured to the layer it stopped in.
    let texel0 = vec2<i32>(floor(p.xy * base_res));
    let near0 = textureLoad(hiz_tex, texel0, 0).r;
    var gap = depth_ndc_to_view_z(near0) - depth_ndc_to_view_z(p.z);
    if (pass_behind && gap > SSR_THICKNESS) {
        // Past the nearest layer's slab: the walk stopped in the farthest's.
        let far0 = textureLoad(hiz_far_tex, texel0, 0).r;
        gap = depth_ndc_to_view_z(far0) - depth_ndc_to_view_z(p.z);
    }
    if (abs(gap) > SSR_THICKNESS) {
        return out;
    }
    return ssr_read(p.xy);
}
#endif

/// One planar mirror's read for this fragment: its reprojected colour, and in `a` its weight —
/// strength times coverage times [`plane_trust`] — zero where it does not serve this water.
fn planar_read(
    params: vec4<f32>,
    second: bool,
    world_pos: vec3<f32>,
    n: vec3<f32>,
    frag_coord: vec2<f32>,
    tilt: f32,
) -> vec4<f32> {
    let plane_error = abs(world_pos.y - params.x);
    let trust = plane_trust(plane_error, params.w, tilt);
    if (params.y <= 0.0 || trust <= 0.0 || view.world_position.y <= world_pos.y) {
        return vec4<f32>(0.0);
    }
    {
        // ---- the plane-error reprojection ----
        //
        // **A capture is exact only for fragments lying ON the plane it was made for**, and almost
        // no fragment does: a liquid grid is a heightfield, so an Elwynn stream slopes away from
        // its own capture plane along its whole length, and the pond above it is a second plane
        // entirely. Read at the fragment's own screen UV, the image those fragments get is mirrored
        // about the wrong height — and because the error varies with the height, it *slides* along
        // the surface as the water drops, which is what reads as a smeared or swimming reflection
        // on a stream and as a plainly wrong one on a second body of water.
        //
        // The correction is exact and costs one projection. Let `p` be the capture plane and `h`
        // this fragment's surface height. Mirroring a point about `h` and then back about `p`
        // translates it by `2(p − h)` in Y — so the texel holding what this fragment should reflect
        // is the one the capture drew for the world point `world_pos` lifted by that much, and its
        // screen UV is where the MAIN camera projects the lifted point. At `h == p` the lift is zero
        // and this reduces to `frag_coord_to_uv(frag_coord)` exactly, which is what it replaces.
        //
        // The one approximation: the lifted point is placed at the water's own depth rather than at
        // the depth of whatever is being reflected. That makes the correction exact for reflections
        // of things at the waterline — the bank, the trees on it, a hull alongside, which is what
        // the eye judges a reflection by — and an overcorrection for distant mountains, where the
        // true offset tends to zero and the residual is a fraction of a pixel through a wobble.
        // **The ripple moves the point the mirror is read FOR, not the UV it is read AT.** The
        // capture is a reflected image, so displacing its UV is far safer here than it was in the
        // march — but it is still shifting a picture rather than reflecting a different place, and
        // at a grazing angle it shifts it far enough to show the same content twice. Reprojecting a
        // rippled position instead asks the capture what is mirrored a few inches away, which is a
        // question with a true answer. Horizontal only: the wave slides the surface, the plane the
        // mirror was built for does not move, and `lift` below depends on a height this must not
        // change.
        let mirror_pos = world_pos + vec3<f32>(n.x, 0.0, n.z) * water_reflect.march.y;
        let lift = 2.0 * (params.x - mirror_pos.y);
        let lifted = position_world_to_ndc(mirror_pos + vec3<f32>(0.0, lift, 0.0));
        // Behind the eye (reverse-Z puts `w < 0` there, and the divide flips `z` negative with it)
        // or thrown well off the side, the reprojection means nothing; the uncorrected UV is the
        // image this drew before the correction existed and is the honest fallback.
        let usable = lifted.z > 0.0 && all(abs(lifted.xy) < vec2<f32>(1.5));
        let uv = select(frag_coord_to_uv(frag_coord), ndc_to_uv(lifted.xy), usable);
        let ruv = clamp(vec2<f32>(1.0 - uv.x, uv.y), vec2<f32>(0.002), vec2<f32>(0.998));
        var mirrored: vec4<f32>;
        if (second) {
            mirrored = sample_mirror2(ruv, water_reflect.flags.x);
        } else {
            mirrored = sample_mirror(ruv, water_reflect.flags.x);
        }
        return vec4<f32>(mirrored.rgb, saturate(min(params.y, 1.0) * mirrored.a * trust));
    }
}

/// **The current**, Valve's two-phase flow map (Vlachos, "Water Flow in Portal 2", 2010): the ripple
/// is carried along the current for one period and restarted, twice, half a period apart, each
/// weighted by a triangle that is zero at its own restart, so neither reset is ever seen. The
/// current is the map's (`ATTRIBUTE_WOW_FLOW`); on still water both phases read the same point and
/// this is the unflowed layer exactly.
const FLOW_PERIOD_S: f32 = 1.6;

/// [`ripple_layer`], carried along `flow` (world XZ yards a second).
fn ripple_flowing(
    world_xz: vec2<f32>,
    drift: vec2<f32>,
    inv_wavelength: f32,
    weight: f32,
    t: f32,
    rot: vec2<f32>,
    flow: vec2<f32>,
) -> vec2<f32> {
    let ph0 = fract(t / FLOW_PERIOD_S);
    let ph1 = fract(ph0 + 0.5);
    let w0 = 1.0 - abs(2.0 * ph0 - 1.0);
    // The second phase reads another stretch of the map, or the two restarts would show one
    // pattern pulsing; scaled by the current, so still water keeps its one pattern.
    let apart = vec2<f32>(0.37, 0.61) * saturate(length(flow)) / inv_wavelength;
    let a = ripple_layer(world_xz - flow * ph0 * FLOW_PERIOD_S, drift, inv_wavelength, weight, t, rot);
    let b = ripple_layer(
        world_xz - flow * ph1 * FLOW_PERIOD_S + apart,
        drift,
        inv_wavelength,
        weight,
        t,
        rot,
    );
    return a * w0 + b * (1.0 - w0);
}

fn stylised_water(
    world_pos: vec3<f32>,
    frag_coord: vec2<f32>,
    /// Which prepass sample the thickness lane reads. Always 0 now — see the fragment entry point
    /// for why this shader no longer takes `@builtin(sample_index)` — but kept as a parameter
    /// rather than inlined so the one place that decides it stays the one place that decides it.
    sample_index: u32,
    // Yards of water between this fragment and whatever opaque surface is behind it, or a negative
    // number where the scene depth is unavailable (the reference lane, no prepass) — in which case
    // every term below falls back to the authored depth byte, exactly as it did before.
    thickness: f32,
    depth: f32,
    shore: f32,
    shore_offset: vec2<f32>,
    // The heightfield's smooth normal (zero where the mesh has none).
    smooth_normal: vec3<f32>,
    /// The mirrors' weight, then the two probe spots and the second's share — see
    /// `ATTRIBUTE_WOW_PLANAR`.
    planar: vec4<f32>,
    /// The current, world XZ yards a second — see `ATTRIBUTE_WOW_FLOW`.
    flow: vec2<f32>,
    shallow: vec4<f32>,
    deep: vec4<f32>,
    lit: vec3<f32>,
    room_fog: u32,
) -> vec4<f32> {
    let t = anim_time();
    let xz = world_pos.xz;
    // In uniform control flow, before anything branches — see [`surface_tilt`].
    let tilt = surface_tilt(world_pos);
    // Beside the tilt, and for the same reason: both read screen-space derivatives, which are only
    // defined before the per-fragment branches below. This is the direction the march is cast on.
    // The heightfield's smooth normal where the mesh carries one, so the reflection bends across a
    // slope instead of stepping at each triangle; the facet's own normal otherwise. Both read in
    // uniform control flow.
    let facet_n = surface_normal(world_pos);
    let flat_n = select(facet_n, normalize(smooth_normal), dot(smooth_normal, smooth_normal) > 0.25);
    // One tile every 12 yd at scale 1; the three layers run at 3.4x, 1x and 0.3x of it.
    let inv_tile = 1.0 / 12.0;
    // The fourth layer is the smallest and does the most: at ~1.4 yd it is the only one whose
    // features are smaller than the sun's specular lobe, so it is what breaks the highlight into a
    // glitter path instead of a single blown-out disc on a nearly flat plane. It carries the least
    // weight of the four, and the mip chain retires it first with distance.
    //
    // Five octaves now, from ~110 yd down to ~1.4 yd, each on its own rotated lattice (see
    // [`ripple_layer`]). The **long swell at the head is new**, and it is there for the distance:
    // everything shorter than it either mips away toward the horizon or subtends too little of a
    // pixel to be seen there, which is what left the far water looking like a painted plate. A
    // 110-yard swell is still several pixels of tilt at the far clip, and it is the only layer that
    // is, so it is the one carrying the far field — read through the sky gradient below, which is
    // what turns half a degree of tilt into a visible colour.
    //
    // The drifts on the two longest layers are the other half of that: at the old rates the far
    // water moved a third of a yard a second, which over a horizon-sized wave is no motion at all.
    let slope =
        ripple_layer(xz, vec2<f32>(0.012, 0.007), inv_tile / 9.2, 0.85, t, vec2<f32>(1.0, 0.0))
        + ripple_layer(
            xz,
            vec2<f32>(0.016, 0.021),
            inv_tile / 3.4,
            1.00,
            t,
            vec2<f32>(0.8572, 0.5150),
        )
        // The two middle layers are what the broken highlight is MADE of: at twelve and at
        // three-and-a-half yards their features are the size of the bright patches in the
        // reference's sun column, so lifting them is what turns a smooth sheet into a mottle.
        + ripple_flowing(
            xz,
            vec2<f32>(-0.021, 0.010),
            inv_tile,
            1.00,
            t,
            vec2<f32>(0.4540, 0.8910),
            flow,
        )
        + ripple_flowing(
            xz,
            vec2<f32>(0.014, -0.026),
            inv_tile / 0.30,
            0.70,
            t,
            vec2<f32>(-0.2924, 0.9563),
            flow,
        )
        // Dialled back from 0.30 with the specular: the finest layer is the crinkle, and a fine
        // crinkle under a tight highlight is the texture of oil on water.
        + ripple_flowing(
            xz,
            vec2<f32>(0.041, 0.033),
            inv_tile / 0.12,
            0.20,
            t,
            vec2<f32>(-0.8572, 0.5150),
            flow,
        );
    // ---- the live wave field ----------------------------------------------------------------
    //
    // Everything above is a texture of waves. This is water that has actually been pushed on: a
    // height field integrated on the CPU under the 2-D wave equation, with every swimmer in range a
    // moving source in it (`benilla_world::liquid::ripple_sim`). What arrives here is its slope,
    // and it is simply added to the ambient ripple's — a wake is not a decal drawn over the water,
    // it is the water being a different shape.
    //
    // Sampled UNCONDITIONALLY, with the window handled afterwards: a `textureSample` inside an `if`
    // is non-uniform control flow, its implicit derivatives are undefined there, and WGSL rejects
    // it. The sampler clamps, so a lookup from outside the window returns its rim, which the CPU's
    // absorbing band has already brought to zero.
    let sim_uv = (xz - water_reflect.sim.xy) * water_reflect.sim.z;
    let sim_tex = textureSample(
        wake_tex,
        wake_samp,
        clamp(sim_uv, vec2<f32>(0.0), vec2<f32>(1.0)),
    );
    let inside = select(
        0.0,
        1.0,
        all(sim_uv > vec2<f32>(0.0)) && all(sim_uv < vec2<f32>(1.0)),
    );
    // Distance to the nearest edge of the window, in window fractions — the fade the far side of
    // [`WAKE_EDGE_FADE`] describes.
    let sim_edge = min(min(sim_uv.x, 1.0 - sim_uv.x), min(sim_uv.y, 1.0 - sim_uv.y));
    let sim_w = min(water_reflect.sim.w, 1.0)
        * inside
        * smoothstep(0.0, WAKE_EDGE_FADE, sim_edge);
    let wake_slope = (sim_tex.rg * 2.0 - 1.0) * sim_w;
    // The wave's own height, signed — how much water the disturbance has put under this pixel.
    let wake_height = (sim_tex.b * 2.0 - 1.0) * sim_w;
    let wake_foam = saturate((length(wake_slope) - WAKE_FOAM_FLOOR) * WAKE_FOAM_GAIN);

    // The map holds slope, so the normal is rebuilt with Y up — no tangent frame, because a liquid
    // surface is a flat axis-aligned plane that never rotates.
    //
    // **The strength is what breaks the light.** In the reference the sun's column on the water is
    // not a wash, it is shattered — a mottle of bright patches and dark gaps several yards across,
    // moving. That look comes from the surface having enough tilt to swing the reflected ray right
    // off the highlight and back onto it again across a single wave, and at the sandbox's 0.75 it
    // simply does not: every facet stays near enough to flat that the highlight slides over the
    // whole surface as one smooth sheet.
    //
    // Note this is the opposite lever from the one that fixed "oily". A tighter specular lobe would
    // also break the highlight up, into hard little chips — which is exactly the film-on-water read
    // that complaint was about. Breaking it with the SURFACE instead keeps the lobe broad and the
    // water rough, which is the same thing real water does.
    // `probe_box_min.w` scales the whole ripple — 1, and 0 for a
    // dead-flat surface. Flat water is the only way to compare two REFLECTIONS against each other:
    // with waves in, the planar tier's ripple is a small bounded offset into an already-correct
    // image while the cube's goes through `reflect()`, which at a grazing eye swings across a huge
    // arc for the same tilt — so the two tiers are being asked different questions and the
    // comparison says nothing. Flattening removes the wave from both sides at once.
    let ripple_scale = water_reflect.probe_box_min.w;
    let n = normalize(vec3<f32>(
        (slope.x * NORMAL_STRENGTH + wake_slope.x) * ripple_scale,
        1.0,
        (slope.y * NORMAL_STRENGTH + wake_slope.y) * ripple_scale,
    ));

    // Body: the zone's own swatch, deepening with V. sqrt, not linear — absorption in real water is
    // exponential, and the square root keeps the shallows a distinct band instead of a thin gradient.
    // **How fast the water becomes water.** The depth coordinate runs 0 at the waterline to 1 at
    // about five yards, and reading the body straight off it makes a stream a different substance
    // from a pond: knee-deep water sits a fifth of the way along the ramp, so it keeps the swatch's
    // shallow row — which in the shipped data is the muddy olive of a riverbed, not water — and its
    // opacity stays near the shallow end, letting the bed through. The result is a wet path where
    // there should be a stream.
    //
    // The exponents below pull both ramps forward so that anything more than ankle-deep reads as
    // the same water a pond is made of, while the last hand's breadth still lightens into the
    // shore. It is a look choice, not a depth model: the world's own numbers still decide, they are
    // just read on a curve that treats shallow water as water.
    //
    // The ramp is also CAPPED short of the deep row — see [`BODY_DEPTH_MAX`], which is the "too
    // dark, look at Booty Bay" report.
    // The wake rides the depth ramp with the world's own bathymetry — see [`WAKE_BODY_DEPTH`].
    // **How far along the swatch the body has travelled**, and there are two answers.
    //
    // Where the scene behind the water is known, the water column is a real distance in yards and
    // the shallow row gives way to the deep one by Beer–Lambert extinction, PER CHANNEL: red is
    // gone within a couple of yards, green carries further, blue furthest. That is the whole reason
    // deep water is blue rather than "the deep colour", and it is a thing a single lerp of two
    // swatch rows cannot express at any depth but the one it was tuned at — which is why the deep
    // end had to be capped by hand to stop several zones going to tar.
    //
    // Where it is not known — the reference lane, which has no prepass, or the first frame after a
    // style flip — this is the authored depth byte on a curve, exactly as before.
    let wake_push = wake_height * WAKE_BODY_DEPTH;
    var body_t = saturate(pow(depth, 0.35) + wake_push) * BODY_DEPTH_MAX;
    var body_rgb = mix(shallow.rgb, deep.rgb, body_t);
    if (thickness >= 0.0) {
        let t = min(thickness, WATER_MAX_THICKNESS);
        // [`BODY_DEPTH_MAX`] applies here too, and leaving it off was the other half of the slab.
        // The cap is not a fudge around the authored byte's units — it is a statement about the
        // 1.12 palettes themselves, several of whose deep rows are nearly black because they were
        // authored for a surface with no reflection and no glitter on it. A physically-correct
        // extinction curve run all the way to that row is still a run to tar.
        let absorbed =
            saturate(vec3<f32>(1.0) - exp(-WATER_EXTINCTION * t) + wake_push) * BODY_DEPTH_MAX;
        body_rgb = mix(shallow.rgb, deep.rgb, absorbed);
    }
    // The sun's bearing, taken flat — see [`BODY_RIPPLE_SHADE`]. The epsilon is for the frame at
    // startup before the lanes are written, and for the instant the sun crosses the zenith.
    let sun_xz = normalize(water_reflect.sun.xz + vec2<f32>(1e-4, 1e-4));
    let wave_tilt = dot(n.xz, sun_xz);
    let body = body_rgb * lit * (1.0 + BODY_RIPPLE_SHADE * wave_tilt);

    // Fresnel toward the horizon. Schlick's 5th power, floored at the 2 % water reflects head-on.
    let to_view = normalize(view.world_position.xyz - world_pos);
    let fresnel = pow(1.0 - saturate(dot(n, to_view)), 5.0);
    // What the surface reflects when the mirrored pass has nothing for it: the sky, read at the
    // reflected view ray's own elevation rather than as one flat colour — see [`sky_reflection`],
    // which is the answer to "waves are only visible in the line of the sun".
    let sky = sky_reflection(reflect(-to_view, n));
    let sky_lit = sky * lit;
    // **A surface has one reflection, and Fresnel is how big its share of the fragment is.** This
    // is that share, and the tiers below spend it: the sky, the capture and the march are three
    // answers to *what* is reflected, never three separate reflections to be layered one over the
    // other.
    //
    // They used to be layered, and it cost the tiers about half their contrast. `rgb` took its full
    // `s` of sky here; a tier then mixed itself in at its own `s`, which displaces only `s` of the
    // sky and leaves `s(1 - s)` standing — so a reflection of a dark gorge arrived with a wash of
    // bright sky still painted over it, the body was robbed to pay for it, and a river read as flat
    // while a coast, where the tier is looking at the sky anyway, read as correct.
    let fres = mix(0.02, 0.45, fresnel);
    // What is reflected. The sky is the FLOOR, not a layer: it is the answer for a direction no
    // tier has a better one for, and each tier below replaces it over the coverage it actually has.
    var refl = sky_lit;
    let layered = water_reflect.flags.w > 0.5;
    // How much of the final `refl` came from the CUBE PROBE, tracked down the tier chain so the
    // Fresnel weight can be lifted where — and only where — the probe is the tier answering. Each
    // tier that mixes over `refl` takes the same share out of this. See the final mix.
    var probe_share = 0.0;
    var rgb = mix(body, sky_lit, fres);

    // ---- the cubemap probe, tier four ---------------------------------------------------------
    //
    // **Directly over the sky and under everything else**, which is its confidence order: it knows
    // the neighbourhood but not this fragment, so anything that knows this fragment outranks it. It
    // answers the one thing none of the other three can — off-screen content at the RIGHT
    // ORIENTATION. The march sees only what is on the frame; the capture sees off the frame but at
    // one plane's orientation; the sky knows nothing. A cube anchored to the water is a picture of
    // what is actually around it, read in the fragment's own reflected direction.
    //
    // The strength lane is zero until every face has been captured, so the frames while the six
    // cameras take their turns show the sky mix rather than a half-built cube.
    // **Pick the nearest few live probes.** Not a convenience: a box-projected cube's error is
    // identically zero at its own capture point — there the ray-box hit rebased onto the centre
    // collapses to the true reflected direction, whatever the proxy is — and grows with the
    // distance from it. So "which probe" IS the error bound, and nearest is the whole reason there
    // is more than one.
    //
    // **Considering a probe is free; SAMPLING one is not.** The loop below is distance arithmetic
    // over every slot and costs no memory traffic, while each tap is a colour fetch plus its depth
    // march — five cube reads at the default march depth. That is why the answer to "how many
    // probes blend" is "look at all of them, read the best few": `PROBE_TAPS + 1` are tracked, the
    // extra one never sampled, existing only to supply the weight every tap is measured above.
    // `probe_box_min.x` pins every fragment to one slot, so two renders can be differenced to
    // measure what the blend between those probes actually has to hide. See `probe::probe_force`.
    let forced = i32(water_reflect.probe_box_min.x);
    let taps = select(
        min(i32(water_reflect.probe_box_max.y), PROBE_TAPS_MAX),
        1,
        forced >= 0,
    );
    // **Ranked by weight, not by distance.** A probe's weight is its availability over its
    // distance, and the taps are the heaviest few; the next one is the cutoff each tap's weight is
    // measured above. A probe fading in starts at weight zero, so it cannot push a contributing
    // probe out of the taps, and it takes a tap only at the weight of the probe it replaces, which
    // is where both sit at zero above the cutoff. Ranked by distance, a probe arriving at nearly
    // zero strength displaced a contributing one on its first frame, and the water blinked.
    // `march.w`: the probes stand on fixed spots and each fragment reads its own.
    let bound = water_reflect.march.w > 0.5;
    var idx = array<i32, 4>(-1, -1, -1, -1);
    var wt = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
    for (var i = 0; i < PROBE_SLOT_MAX; i = i + 1) {
        let sl = water_reflect.probes[i];
        if (sl.at.w <= 0.0 || (forced >= 0 && i != forced)) {
            continue;
        }
        // Horizontal distance: probes stand on the water they answer for, so the vertical component
        // is noise from the lift.
        // Availability times share (`box_min.w`, 1 except in the live probe's crossfade).
        var u = sl.at.w * sl.box_min.w / max(distance(world_pos.xz, sl.at.xz), 1e-4);
        // Bound to fixed spots (the planar lane): a fragment reads only its own region's probes, by
        // the share its cell gives each — never the nearest probe of some other water.
        if (bound) {
            let id = sl.bind.x;
            let own = select(0.0, 1.0 - planar.w, abs(id - planar.y) < 0.5)
                + select(0.0, planar.w, abs(id - planar.z) < 0.5);
            u = sl.at.w * sl.box_min.w * own;
        }
        // Insertion into the short list, heaviest first, shifting the tail down.
        for (var k = 0; k <= taps; k = k + 1) {
            if (u > wt[k]) {
                for (var m = taps; m > k; m = m - 1) {
                    wt[m] = wt[m - 1];
                    idx[m] = idx[m - 1];
                }
                wt[k] = u;
                idx[k] = i;
                break;
            }
        }
    }

    var probe_w = 0.0;
    var probe_rgb = vec3<f32>(0.0);
    let cut = wt[taps];
    var wsum = 0.0;
    var w = array<f32, 3>(0.0, 0.0, 0.0);
    for (var k = 0; k < taps; k = k + 1) {
        w[k] = max(wt[k] - cut, 0.0);
        wsum = wsum + w[k];
    }
    if (wsum > 1e-8) {
        // The cube is read through a FLATTER normal than the water draws with — see
        // `liquid::probe::probe_normal` for the measurement and for why a prefiltered cube is the
        // real answer.
        // **Flat direction, rippled lookup point** — the same correction the march and the mirror
        // got. Perturbing the DIRECTION swings the sample across the cube, and at a grazing angle a
        // slight tilt swings it far: measured at the Stranglethorn camera this tier scored 3.69 on
        // the repeat signature against a waves-off 1.26, the worst of the three. The cube is
        // parallax-corrected against its own capture point, so moving the point the correction is
        // solved for is a real change of viewpoint on a real place, and cannot repeat.
        // About the facet's geometric normal, as the march does: on a sloping stream straight up
        // is off by the slope, and the reflected ray twice that.
        let r = reflect(-to_view, flat_n);
        let probe_pos = world_pos + vec3<f32>(n.x, 0.0, n.z) * water_reflect.march.y;
        // The tier's strength is the weighted mean, over the taps, of each probe's availability
        // and its own distance falloff (`probe_weight`), so it does not depend on which tap ranks
        // first. The taps are blended by weight and nothing cleverer: the probe is a soft term
        // under the screen-space march, not an attempt to place reflections exactly.
        var strength = 0.0;
        for (var k = 0; k < taps; k = k + 1) {
            if (w[k] <= 0.0) {
                continue;
            }
            let sl = water_reflect.probes[idx[k]];
            let share = w[k] / wsum;
            // A bound probe serves its whole region, so it does not fade with distance from it.
            let falloff = select(
                probe_weight(
                    world_pos,
                    sl.at.xyz,
                    water_reflect.probe_cfg.y,
                    water_reflect.probe_cfg.z,
                ),
                1.0,
                bound,
            );
            strength = strength + share * sl.at.w * falloff;
            probe_rgb = probe_rgb
                + sample_probe(sl, idx[k], probe_pos, r, i32(water_reflect.probe_extra.w)) * share;
        }
        probe_w = water_reflect.probe_cfg.x * strength;
    }

    // `$WOW_PROBE_SHOW` — the cube as the water reads it, black where the tier is not contributing.
    // It exists because a cubemap of the WRONG PLACE is still a plausible reflection: a face
    // captured at the wrong orientation, or a cube that never received its copy, yields water that
    // looks like water and reflects somewhere else. The final image cannot show that; this can.
    if (water_reflect.probe_cfg.w > 0.5 && water_reflect.probe_cfg.w < 1.5) {
        return vec4<f32>(probe_rgb, 1.0);
    }
    // `$WOW_PROBE_CUBE` — the cube unwrapped across the water as a lat-long map: screen X is the
    // bearing, screen Y the elevation, horizon across the middle. No reflection, no ripple, no
    // Fresnel; just what the six cameras captured. See `liquid::probe`'s `probe_cube_show`.
    if (water_reflect.probe_cfg.w > 2.5) {
        // **The latitude sweep is packed into the bottom half of the screen, deliberately.** This
        // debug can only paint WATER fragments, and water occupies roughly the lower half of a
        // frame at any normal camera — so mapping screen Y straight to latitude showed only the
        // cube's lower hemisphere, horizon down to nadir, while the upper half of the image was
        // the real world above the waterline. That is an easy thing to misread as "the cube has
        // trees in it" when the trees being looked at are the actual trees. `fract(y * 2)` sweeps
        // zenith to nadir across whichever half the water is in, so the whole sphere is visible.
        let uv = frag_coord_to_uv(frag_coord);
        // Longitude across the screen, latitude down the LOWER HALF, linearly and with no wrap.
        //
        // The first version used `fract(uv.y * 2)` to pack a full sphere into the half of the frame
        // water occupies, and `fract` wraps: the bottom row returned to the zenith instead of
        // reaching the nadir, so the map carried a seam and every latitude read off it was suspect.
        // Two wrong readings of this image came from that. `uv.y = 0.5` is straight up, `uv.y = 1`
        // is straight down, and the arithmetic between them is a straight line — which is what
        // makes the horizon land exactly halfway down the band and a mis-oriented face obvious.
        let lon = (uv.x * 2.0 - 1.0) * 3.14159265;
        let lat = (1.5 - 2.0 * uv.y) * 3.14159265;
        let dir = vec3<f32>(cos(lat) * sin(lon), sin(lat), cos(lat) * cos(lon));
        // Slot 0's cube — the debug unwraps ONE probe, because a lat-long of "whichever probe is
        // nearest this fragment" would be a mosaic of several and unreadable as a check on any of
        // them. `probe_extra.y` picks which.
        let texel = textureSampleLevel(
            probe_tex,
            probe_samp,
            dir,
            i32(water_reflect.probe_extra.y),
            0.0,
        );
        return vec4<f32>(texel.rgb, 1.0);
    }

    // ---- the planar reflection ----------------------------------------------------------------
    //
    // Replaces the sky mix above wherever the mirrored camera actually drew something. Three gates,
    // and each is a case where the image is not this surface's reflection: strength 0 (the pass did
    // not run), a surface further from the mirror plane than the tolerance (one capture is right
    // for one plane, and a world of terraced pools has many), and an eye below the surface (the
    // reflection is on the other side of it).
    //
    // `1 - u` is the mirrored camera's negated right vector undone — see `reflect`'s module doc; the
    // normal's XZ displaces the sample, which is what makes it read as a reflection in water rather
    // than in glass. Alpha is coverage, so where the mirrored view saw nothing the sky mix stands.
    // **Every water body in the frame reflects, not just the one the capture was built for.**
    // `params.w` is no longer a yes/no threshold on the plane error — it is the distance over which
    // the correction below is trusted, and past it the surface fades back to its sky mix instead of
    // switching off. See [`plane_trust`] for why a hard edge was the wrong shape: a pool and the
    // ocean are routinely in one shot, they are ten yards apart, and whichever one lost the coin
    // toss had no reflection at all.
    // ---- screen-space reflection, over the planar capture, over the sky ------------------------
    //
    // The layering is the whole design, and each tier covers the one below's blind spot. SSR is
    // right for any surface orientation because it traces the fragment's real normal, and blind to
    // everything off the screen. The planar capture sees off-screen and behind the camera, and is
    // right for one plane at one orientation. The sky is always available and always plausible.
    //
    // They fail in complementary places, which is why this is a hybrid rather than a compromise:
    // open water is flat (where the plane is exact) and grazing (where the march runs off the
    // frame), while a stream is sloped (where the plane is wrong by twice its tilt) and enclosed by
    // banks a few yards away (where the march has plenty to hit).
    var ssr: SsrHit;
    ssr.rgb = vec3<f32>(0.0);
    ssr.conf = 0.0;
#ifdef DEPTH_PREPASS
    if (water_reflect.flags.z > 0.0 && view.world_position.y > world_pos.y) {
        ssr = ssr_trace(
            world_pos + vec3<f32>(n.x, 0.0, n.z) * water_reflect.march.y,
            reflect(-to_view, flat_n),
        );
    }
#endif

    // ---- the probe's contribution ------------------------------------------------------------
    //
    // **Composited here rather than where it is sampled, so that it can stand down for the march.**
    // Every other tier is attenuated by `1 - ssr.conf` — where the screen found the answer itself,
    // nothing else averages into it, because two reflections of one surface blended together is a
    // double image and not a better one. The probe was missing that term purely because it was
    // mixed in before `ssr` had been traced, which made it the one tier that painted at full weight
    // over a march that already had the fragment. Against the planar tier's
    // `fresnel * params.y * mirrored.a * trust * (1 - ssr.conf)` it carried a single attenuation,
    // and it read as visibly heavier water than the mirror gives.
    //
    // The order is unchanged: still directly over the sky and under the plane and the march. Only
    // the point at which it is applied has moved.
    if (probe_w > 0.0) {
        refl = mix(refl, probe_rgb, saturate(probe_w));
        probe_share = saturate(probe_w);
        if (layered) {
            // **Saturated**, because `probe_w` carries a strength that is allowed above 1. WGSL's
            // `mix` extrapolates rather than clamping, so an unsaturated weight past 1.0 does not
            // merely show more reflection — it runs past the reflection and out the other side,
            // subtracting the water's own body colour.
            rgb = mix(
                rgb,
                probe_rgb,
                saturate(
                    mix(0.02, REFLECT_MAX, fresnel)
                        * probe_w
                        * max(water_reflect.probe_box_max.x, 0.0)
                        * (1.0 - ssr.conf),
                ),
            );
        }
    }

    // Each mirror serves the water near its own plane: its colour and its weight here — strength
    // times coverage times trust. Where both reach a fragment they blend by weight.
    // Off probe water the mirrors stand down, so the probe beneath them shows through.
    let serve = vec4<f32>(1.0, 1.0, 1.0, smoothstep(0.0, 1.0, planar.x));
    let m1 = planar_read(water_reflect.params, false, world_pos, n, frag_coord, tilt) * serve;
    let m2 = planar_read(water_reflect.mirror2, true, world_pos, n, frag_coord, tilt) * serve;
    // The colour favours the mirror whose plane is nearer this water — both are trusted across
    // their whole tolerance, so without this a body between the two planes took an even mix of a
    // right image and a wrong one. The overall weight below is untouched.
    let k1 = m1.a / (1.0 + abs(world_pos.y - water_reflect.params.x));
    let k2 = m2.a / (1.0 + abs(world_pos.y - water_reflect.mirror2.x));
    let msum = k1 + k2;
    if (msum > 0.0) {
        let mirrored = vec4<f32>((m1.rgb * k1 + m2.rgb * k2) / msum, 1.0);
        // `$WOW_MIRROR_SHOW` — the capture as the water sees it.
        if (water_reflect.params.y > 1.5) {
            return vec4<f32>(mirrored.rgb, 1.0);
        }
        // Either mirror's weight counts; together they are an "either", not a sum past one.
        let w_planar = saturate(m1.a + m2.a - m1.a * m2.a);
        refl = mix(refl, mirrored.rgb, w_planar);
        probe_share = probe_share * (1.0 - w_planar);
        if (layered) {
            let amount = mix(0.02, REFLECT_MAX, fresnel) * w_planar * (1.0 - ssr.conf);
            rgb = mix(rgb, mirrored.rgb, amount);
        }
    }

    // …and the march's own contribution, on the same Fresnel weight the tiers below it use, so the
    // water does not change how reflective it is depending on which tier answered.
    if (ssr.conf > 0.0) {
        let w_ssr = saturate(water_reflect.flags.z * ssr.conf);
        refl = mix(refl, ssr.rgb, w_ssr);
        probe_share = probe_share * (1.0 - w_ssr);
        if (layered) {
            rgb = mix(
                rgb,
                ssr.rgb,
                mix(0.02, REFLECT_MAX, fresnel) * water_reflect.flags.z * ssr.conf,
            );
        }
    }

    // **The probe lane draws its reflection and nothing else** (`probe_at.w`). Not a debug switch:
    // it is what that lane is while the tier is being made correct, because a reflection arriving
    // at a few per cent over a lit green lake hides every error it contains. See `liquid::probe`.
    if (water_reflect.probe_at.w > 0.5) {
        return vec4<f32>(refl, 1.0);
    }

    // The one mix. With both tiers off this is exactly the sky mix `rgb` already holds, which is
    // what keeps a world with no reflection pass byte-identical.
    //
    // **The probe's share gets its own Fresnel weight**, and that is a deliberate departure from the
    // rule stated above it — that the water should not change how reflective it looks depending on
    // which tier answered. It is here because the probe could not otherwise be made more visible at
    // all: on this path `refl` is one composited colour and the probe ALREADY owns it outright
    // wherever it contributes, so raising the tier's own weight past 1.0 changed literally nothing
    // (four strengths from 1.2 to 3.0 rendered byte-identical). The only remaining lever on how
    // much of it reaches the pixel is the Fresnel mix, so that is the lever.
    //
    // `probe_share` is the fraction of `refl` still coming from the cube after the plane and the
    // march have taken their turns, so the lift lands only where the probe actually answered and
    // the other tiers keep the shared weight exactly as before. At `probe_boost == 1.0` this is
    // arithmetically the old line.
    if (!layered) {
        // **What the boost multiplies is small, which is why it needs range.** `fres` is
        // `mix(0.02, REFLECT_MAX, fresnel)`, and looking down at water from standing height the
        // Fresnel term is near its floor — around 0.04. A 1.5x lift of that is still four per cent
        // of reflection, which reads as no reflection at all. The useful range of this knob is
        // therefore several times larger than a "strength" normally wants to be.
        //
        // Capped below 1 rather than at it: a fully mirrored surface is what [`REFLECT_MAX`] exists
        // to prevent, and holding a little of the water's own body at every angle is what stops it
        // reading as polished glass instead of a lake.
        let probe_boost = max(water_reflect.probe_box_max.x, 0.0);
        let lifted = min(fres * probe_boost, PROBE_REFLECT_CEIL);
        rgb = mix(body, refl, saturate(mix(fres, lifted, probe_share)));
    }

    // The glitter path: the same Blinn highlight the reference computes, evaluated against the
    // rippled normal per fragment. The fine ripple layer above breaks it into pieces; the exponent
    // and the weight decide how hard and how bright those pieces are — which is to say, how ROUGH
    // the water is. See [`SPEC_POWER`].
    //
    // **Against the VISIBLE sun, not the lighting one.** The engine carries two (see
    // `lighting::daynight`): `light_sun` is pinned at compass 225 deg and never sets — it exists so
    // MCSH shadows can be baked once — while the disc in the sky rides `celestial_sun_direction`
    // at compass 45 deg and genuinely rises and sets. They measure 25-32 deg apart, so a highlight
    // aimed by `light_sun` lands in a different part of the water than the sun it is supposed to be
    // a reflection of, and it lands there at midnight too. `water_reflect.sun` is the one you can
    // point at.
    // Guarded: the lanes are zero for one frame at startup, before the sun pass has run, and
    // `normalize` of a zero vector is a NaN that would survive the multiply by a zero reach.
    let sun_len = length(water_reflect.sun.xyz);
    let to_light = select(vec3<f32>(0.0, 1.0, 0.0), water_reflect.sun.xyz / sun_len, sun_len > 1e-4);
    // ...and only as far as the sun actually reaches: below the horizon, behind a cloud, or behind
    // the ridge you are standing under, there is no glitter to have. The scalar is slewed on the
    // CPU, so this fades over about a second rather than switching.
    let sun_reach = water_reflect.sun.w;
    let half_v = normalize(to_light + to_view);
    rgb += wow_light.light_spec.rgb
        * pow(max(dot(n, half_v), 0.0), SPEC_POWER)
        * (SPEC_WEIGHT * sun_reach);

    // ...and the moon's, which is the same computation against the other body in the sky. Water
    // does not care which one is up; it was only ever the sun here because the sun was the only
    // direction this shader had been handed.
    let moon_len = length(water_reflect.moon.xyz);
    let to_moon = select(
        vec3<f32>(0.0, 1.0, 0.0),
        water_reflect.moon.xyz / moon_len,
        moon_len > 1e-4,
    );
    let half_m = normalize(to_moon + to_view);
    rgb += MOON_SPEC_COLOR
        * pow(max(dot(n, half_m), 0.0), MOON_SPEC_POWER)
        * (SPEC_WEIGHT * MOON_SPEC_WEIGHT * water_reflect.moon.w);

    // The white line where the water meets the land, and the bubbles behind it.
    //
    // `to_shore` is the distance to the waterline in yards, traced on the CPU from where the water
    // surface crosses the ground and carried per vertex (`benilla_world::liquid::surface`). The
    // cells it runs through are subdivided so that distance is described finely enough to draw a
    // band under a yard wide; at the bare 4.17 yd lattice the band appeared only where a corner
    // happened to fall near the water's edge, which is what made it look like teeth.
    // **The distance, measured here rather than interpolated.** `shore` is the same quantity carried
    // per vertex, and it is what the band used to be drawn from — but distance to a curve creases
    // along the curve, and a linear interpolation across that crease cuts its corner: the
    // reconstructed band lands off by up to half a vertex spacing, so at half a yard between
    // vertices and a fifth of a yard of band it visibly jogged, thinned and doubled as the camera
    // moved. The offset has no crease, so it interpolates honestly and the length is exact.
    //
    // The interpolated scalar still wins beyond the near field, for two reasons: past a couple of
    // yards nothing is drawn either way, and an offset is only meaningful where a nearest point
    // exists — on the ridge halfway between two facing banks the nearest point flips sides, and
    // interpolating across that flip would collapse the offset to nothing and draw a foam line down
    // the middle of a channel. Inside the near field the two agree to within the mesh's own error;
    // where they do not, the scalar is the conservative answer and is taken.
    let exact = length(shore_offset);
    var to_shore = shore;
    if (shore < SHORE_EXACT_REACH && abs(exact - shore) < SHORE_EXACT_TRUST) {
        to_shore = exact;
    }

    // The derivatives the band needs, taken HERE, where the flow is still uniform. Everything below
    // sits behind [`FOAM_REACH`], and past that gate neither `fwidth` nor an implicit-LOD
    // `textureSample` is defined — the same rule the wake field's unconditional tap is written
    // around, answered the other way: hoist the derivative rather than the sample.
    //
    // Anti-aliasing against how much distance one pixel covers, bounded at both ends: below so a
    // near-view edge stays an edge, above because past a fraction of the band's own width this
    // stops being anti-aliasing and becomes a smear.
    let aa = clamp(fwidth(to_shore), 0.015, 0.20);
    // World yards per pixel, for the taps inside the gate.
    let d_xz_dx = dpdx(xz);
    let d_xz_dy = dpdy(xz);

    var line = 0.0;
    var bubbles = 0.0;
    if (to_shore < FOAM_REACH) {
        // 1. The line. Its outer edge wavers only slightly, so it reads as an even edge rather than
        //    as something ragged; the raggedness belongs to the bubbles. No time in this term —
        //    foam gathers where the shore's shape makes it gather, and an edge that crawls reads as
        //    a bug.
        let warp = textureSampleGrad(
            ripples,
            ripples_samp,
            xz * 0.05,
            d_xz_dx * 0.05,
            d_xz_dy * 0.05,
        ).b * 2.0 - 1.0;
        // The swell: the band's outer edge advances up the shore and slides back down, out of step
        // from bay to bay (see [`SHORE_RUNUP_YARDS`]). The phase comes from a very coarse read of
        // the same map, so it drifts along a coastline instead of switching at some boundary.
        let swell_phase = textureSampleGrad(
            ripples,
            ripples_samp,
            xz / SHORE_SWELL_SPREAD,
            d_xz_dx / SHORE_SWELL_SPREAD,
            d_xz_dy / SHORE_SWELL_SPREAD,
        ).b * TAU;
        let swell = sin(t * (TAU / SHORE_SWELL_SECS) + swell_phase);
        let edge = (FOAM_LINE_YARDS + SHORE_RUNUP_YARDS * swell) * (1.0 + FOAM_EDGE_WARP * warp);
        line = 1.0 - smoothstep(edge - aa, edge + aa, to_shore);

        // 2. Behind it, bubbles: the same white, but broken into flecks that thin out with distance
        //    until there is only water. `out` runs 0 at the line to 1 where they stop, and it is
        //    the THRESHOLD the noise is cut at — so the coverage falls away on its own, and there
        //    is no second hard edge anywhere out in the water to alias against.
        let out = saturate((to_shore - edge) / FOAM_BUBBLE_YARDS);
        // Fine on purpose: the band is barely a yard, so the flecks have to be a good deal smaller
        // than that or only one spans it and the break-up never reads as bubbles at all — and
        // non-repeating, which at that size the map is not on its own (see [`foam_noise`]).
        let raw = foam_noise(xz, t, d_xz_dx, d_xz_dy);
        // Spread before cutting. The map's height channel is a sum of Perlin octaves normalised on
        // its single largest texel, so its values crowd hard around 0.5 and only the extremes ever
        // approach 0 or 1 (`benilla_world::liquid::ripple`). Cut at a threshold sweeping the whole
        // 0..1 it behaves as a step — everything passes, then nothing — which collapsed the bubbles
        // into a narrow ring and made the line's outer edge read as hard. Widened about its middle,
        // the field spans the range the threshold actually travels.
        let bub = saturate((raw.v - 0.5) * 3.5 + 0.5);
        // The cut is softened by the noise's own pixel footprint, so at distance the flecks resolve
        // to their average instead of sparkling. The footprint is carried out of `foam_noise` on
        // the same fetches that gave the value — `fwidth` is not available on this side of the gate
        // — and the `* 3.5` is the spread above, which multiplies the derivative with the value.
        let bw = max(0.07, raw.fw * 3.5 * 1.5);
        // The taper must reach exactly zero where the band ends, not merely dim. `out` saturates at
        // 1 out in open water, and a threshold sitting at 1 still admits the top of the spread
        // field — which put flecks across the whole surface the first time this was tried.
        bubbles = smoothstep(out - bw, out + bw, bub) * (1.0 - out);
    }

    // ...and the wake's own aeration alongside them. `max`, not a sum: these are three ways for the
    // same surface to be white, and adding them blows out where a swimmer crosses a waterline —
    // which is precisely where a player spends their time in the water.
    let foam = max(
        max(line * FOAM_LINE_LEVEL, bubbles * FOAM_BUBBLE_LEVEL),
        wake_foam * WAKE_FOAM_LEVEL,
    );
    rgb = mix(rgb, FOAM_COLOR * lit, foam);

    // Opacity is still the world's: the swatch ramp the reference uses, plus the foam, which is
    // spray and hides what is under it.
    // ...and the BODY thins to nothing where something passes through the surface: a sheet of water
    // meeting a rock along a hard line is the giveaway that it is a sheet, while real water fades
    // over the last hand's breadth because by then there is barely any water left to be opaque
    // with. Needs to know what is behind the surface, so it is another thing the prepass buys.
    //
    // **The foam is taken after the fade, not before it.** Foam sits ON the surface — it is spray,
    // not water column — so softening it against the very intersection it gathers at erases the
    // shoreline exactly where the line belongs, which is what the first attempt did.
    var soft = 1.0;
    if (thickness >= 0.0) {
        soft = saturate(thickness / SOFT_EDGE_YARDS);
    }
    let alpha = max(mix(shallow.w, deep.w, sqrt(depth)) * soft, foam);
    return vec4<f32>(apply_fog(rgb, world_pos, room_fog), alpha);
}

// The vertex input — bevy 0.18's `forward_io::Vertex` fields at bevy's own shader locations, plus
// the shore offset at 10 under `LIQUID_SHORE_OFFSET` (`LiquidExt::specialize` sets the def and
// appends the attribute to the buffer layout when the mesh carries it). Declared here rather than
// imported for the one reason the model shader declares its own: a struct you did not write cannot
// gain a field.
struct LiquidVertex {
    @builtin(instance_index) instance_index: u32,
#ifdef VERTEX_POSITIONS
    @location(0) position: vec3<f32>,
#endif
#ifdef VERTEX_NORMALS
    @location(1) normal: vec3<f32>,
#endif
#ifdef VERTEX_UVS_A
    @location(2) uv: vec2<f32>,
#endif
#ifdef VERTEX_UVS_B
    @location(3) uv_b: vec2<f32>,
#endif
#ifdef VERTEX_COLORS
    @location(5) color: vec4<f32>,
#endif
#ifdef LIQUID_SHORE_OFFSET
    // Vertex → nearest point on the waterline, in mesh-local XZ yards.
    @location(10) shore_offset: vec2<f32>,
#endif
#ifdef LIQUID_SURFACE_NORMAL
    @location(11) surface_normal: vec3<f32>,
#endif
#ifdef LIQUID_PLANAR
    @location(12) planar: vec4<f32>,
#endif
#ifdef LIQUID_FLOW
    // The current, mesh-local XZ yards a second.
    @location(13) flow: vec2<f32>,
#endif
}

@vertex
fn vertex(in: LiquidVertex) -> LiquidVsOut {
    var out: LiquidVsOut;
    let world_from_local = mesh_functions::get_world_from_local(in.instance_index);
    out.world_position =
        mesh_functions::mesh_position_local_to_world(world_from_local, vec4<f32>(in.position, 1.0));
    out.clip_position = position_world_to_clip(out.world_position.xyz);
    out.world_normal = mesh_functions::mesh_normal_local_to_world(in.normal, in.instance_index);
#ifdef LIQUID_SURFACE_NORMAL
    out.surface_normal =
        mesh_functions::mesh_normal_local_to_world(in.surface_normal, in.instance_index);
#else
    out.surface_normal = vec3<f32>(0.0);
#endif
#ifdef LIQUID_PLANAR
    out.planar = in.planar;
#else
    out.planar = vec4<f32>(1.0, 65535.0, 65535.0, 0.0);
#endif
    out.uv = in.uv;
#ifdef VERTEX_COLORS
    out.vcolor = in.color;
#else
    out.vcolor = vec4<f32>(1.0);
#endif
    // Depth coordinate V (0..1) in UV1.x: the swatch row on ADT water, the alpha ramp on WMO.
    out.depth = in.uv_b.x;
    // UV1.y is the distance to the waterline in yards, for the far field and for the shore foam on
    // a mesh without the offset attribute.
    out.shore = in.uv_b.y;
#ifdef LIQUID_SHORE_OFFSET
    // A direction: the placement's rotation and scale, not its translation.
    let off_local = vec3<f32>(in.shore_offset.x, 0.0, in.shore_offset.y);
    let off_world = mat3x3<f32>(
        world_from_local[0].xyz,
        world_from_local[1].xyz,
        world_from_local[2].xyz,
    ) * off_local;
    out.shore_offset = off_world.xz;
#else
    out.shore_offset = vec2<f32>(in.uv_b.y, 0.0);
#endif
#ifdef LIQUID_FLOW
    // The current, a direction like the offset: the placement's rotation and scale only.
    out.flow = (mat3x3<f32>(
        world_from_local[0].xyz,
        world_from_local[1].xyz,
        world_from_local[2].xyz,
    ) * vec3<f32>(in.flow.x, 0.0, in.flow.y)).xz;
#else
    out.flow = vec2<f32>(0.0);
#endif
    out.secondary_vtx = sun_sheen(out.world_normal, out.world_position.xyz);
    // ADT surfaces carry no `MeshTag`, so they take the scene fog.
    out.room_fog = mesh_functions::get_tag(in.instance_index) & 0x40000000u;
    return out;
}

// ── The ADT depth swatch ─────────────────────────────────────────────────────────────────────
//
// `0x68a830` fills an 8×64 texture, each row the same across its 8 columns, with an exact
// byte-space integer accumulator, `row(i) = c0 + floor(i * (c1 - c0) / 64)` for i = 0..63, so
// row 63 stops short of the deep endpoint. On the ocean (selector 0) the last row's HSV value is
// scaled by 0.9 (`0x68aa13`, `[0x8102ec]`), which `floor(0.9 * byte)` reproduces within 1/255,
// and its alpha is forced to 255 (`0x7bbec0`/`0x7bbec8`). Sampling is LINEAR, no mip, clamped
// (flags `0x201`), so V maps to texel `V*64 - 0.5` and the ocean darkening ramps over the last
// 1/64 of V. The WMO arms use the 256-entry ramp `0xca7f10` instead, which a plain lerp
// reproduces.
fn swatch_row(shallow: vec4<f32>, deep: vec4<f32>, i: f32, ocean: bool) -> vec4<f32> {
    // RGB endpoints are bytes already (`0x68a8fb`/`0x68a902`), so they round back exactly; alpha
    // endpoints are `LightParams` floats the reference quantizes with `floor(v*255)`.
    let c0 = vec4<f32>(round(shallow.rgb * 255.0), floor(shallow.w * 255.0));
    let c1 = vec4<f32>(round(deep.rgb * 255.0), floor(deep.w * 255.0));
    let row = c0 + floor(i * (c1 - c0) / 64.0);
    if ocean && i >= 63.0 {
        return vec4<f32>(floor(row.rgb * 0.9), 255.0) / 255.0;
    }
    return row / 255.0;
}

/// The swatch sampled at depth coord `v`, LINEAR across the two rows it falls between.
fn swatch_at(shallow: vec4<f32>, deep: vec4<f32>, v: f32, ocean: bool) -> vec4<f32> {
    let t = clamp(v * 64.0 - 0.5, 0.0, 63.0);
    let i0 = floor(t);
    return mix(
        swatch_row(shallow, deep, i0, ocean),
        swatch_row(shallow, deep, min(i0 + 1.0, 63.0), ocean),
        t - i0,
    );
}

@fragment
fn fragment(in: LiquidVsOut) -> @location(0) vec4<f32> {
    // Prepass sample 0, and no `@builtin(sample_index)`: that builtin turns on sample-rate shading,
    // four times the water shader's work under MSAA 4 for no visible difference.
    let sample_index = 0u;
    // The far-clip wall, as terrain and models: discard beyond `fog_params.w` (0 disables it).
    if (wow_light.fog_params.w > 0.0) {
        let clip_z = -(view.view_from_world * vec4<f32>(in.world_position.xyz, 1.0)).z;
        if (clip_z > wow_light.fog_params.w) {
            discard;
        }
    }

    // The animated frame; `view.mip_bias` is the render-scale LOD compensation, 0 at native.
    let detail = textureSampleBias(
        frames,
        frames_samp,
        apply_scroll(in.uv),
        frame_layer(),
        view.mip_bias,
    );

    // Magma/slime: the sheet is the opaque body, unmodulated (the ADT vertex has no colour, the WMO
    // one is `0xffffffff`) and unlit (lighting off on both paths), but fogged.
    if (w.kind.x > 0.5) {
        // The stylised lane grades magma into crust and seam ([`stylised_lava`]); slime keeps the
        // sheet, having no heat to render, and the faithful lane keeps it for both. Fog applies to
        // all four cases — see the module note on why skipping it was wrong.
        //
        // **`select`, not an `if`, and that is a correctness fix rather than a style choice.**
        // `detail` above is a `textureSampleBias`, which needs implicit derivatives, and wrapping
        // the code after it in a branch was enough to cost it a mip level: the FAITHFUL lane's lava
        // came back duller and greyer (−76 red, +9 green, +13 blue over the magma silhouette — the
        // signature of a blurrier mip, not of a grade), on a path where none of this code runs.
        // Caught by capturing `WOW_WATER_STYLE=0` before and after and diffing; it would have been
        // invisible in any stylised shot. A `select` has no control flow for the sample to be
        // reachable-from differently, so the derivatives stay put.
        //
        // The cost of evaluating the grade on surfaces that discard it is two `textureSampleLevel`
        // taps on magma and slime fragments only, which is a rounding error next to being wrong.
        let graded = stylised_lava(detail.rgb, in.world_position.xz, anim_time());
        let full = select(detail.rgb, graded, w.path.y > 0.5 && w.path.z > 0.5);
        return vec4<f32>(apply_fog(full, in.world_position.xyz, in.room_fog), 1.0);
    }

    // V, from the authored depth byte CPU-side: clamp(byte/42) on river/lake (LUT `0xc81768`,
    // `0x68d790`, saturating near 5 yd), clamp(byte/255) on ocean (LUT `0xc7fcd8`,
    // `0x68d690`), both built in `0x68c4c0`. One V indexes colour and alpha alike.
    let depth = clamp(in.depth, 0.0, 1.0);
    var shallow = wow_light.water_river[0];
    var deep = wow_light.water_river[1];
    if (w.kind.y > 0.5) {
        shallow = wow_light.water_ocean[0];
        deep = wow_light.water_ocean[1];
    }

    // ---- The stylised look ------------------------------------------------------------------
    //
    // It replaces all three water arms at once, and it runs HERE — after the world's own swatch and
    // depth are resolved, before any of the reference's three combines — because those two are the
    // only world inputs it takes.
    //
    // Each arm keeps its own colour source and its own lighting, so what changes is the treatment
    // and never the palette: the ADT ramp lerps the zone swatch by depth; a WMO exterior canal
    // takes the deep river band flat, having no bathymetry to lerp over; and a WMO interior pool
    // keeps its authored `MOMT.diffColor` body, unlit, because that arm has no normal to light with
    // and its rooms are not lit by the scene's sun.
    if (w.path.y > 0.5) {
        // **The ocean's depth coordinate is a different ramp from the river's**, and this path has
        // to reconcile them. Both come off the same per-vertex MCLQ depth byte (~8.5 byte/yd), but
        // the reference divides the river's by 42 — saturating at ~5 yd — and the ocean's by 255,
        // which `liquid.rs` records as a placeholder pending its own RE. The faithful path above
        // takes each as it finds it, because each indexes its own swatch and that IS the reference's
        // behaviour. The stylised look cannot: it reads this number as *how deep the water is*, and
        // a sixth of a river's ramp read as a sixth of the way to deep water is what put a
        // shoreline's worth of foam across the whole of Booty Bay and left the sea the colour of a
        // shallow. Rescaled into the river's units here, and here only.
        var style_depth = depth;
        if (w.kind.y > 0.5) {
            style_depth = min(depth * (255.0 / 42.0), 1.0);
        }
        var body_shallow = shallow;
        var body_deep = deep;
        var lit = vec3<f32>(1.0);
        if (w.path.x > 1.5) {
            body_shallow = vec4<f32>(in.vcolor.rgb, shallow.w);
            body_deep = vec4<f32>(in.vcolor.rgb, deep.w);
        } else {
            if (w.path.x > 0.5) {
                body_shallow = deep;
            }
            let n_lit = normalize(in.world_normal);
            lit = clamp(
                wow_light.light_ambient.rgb
                    + wow_light.light_diffuse.rgb
                        * max(dot(n_lit, -normalize(wow_light.light_sun.xyz)), 0.0),
                vec3<f32>(0.0),
                vec3<f32>(1.0),
            );
        }
        // **How much water stands between this fragment and whatever is behind it**, in yards.
        //
        // The opaque scene's depth comes from the prepass, which exists only while this look is on
        // (`benilla_world::liquid::depth`); both depths are converted out of reverse-Z into view
        // space, where the difference is a distance rather than a ratio. Negative means "not
        // known", and every consumer falls back to the authored depth byte.
        var thickness = -1.0;
#ifdef DEPTH_PREPASS
        let scene_depth = prepass_depth(in.clip_position, sample_index);
        // **A pixel with nothing in the prepass is UNKNOWN, not infinitely deep.** Reverse-Z clears
        // to zero at the far plane, so that is what "nothing was drawn here" reads as — and taking
        // it at face value makes the water column enormous, which drives the absorption below
        // straight to the deep row and paints a flat slab.
        //
        // It happens over more of the world than it sounds. Terrain is in the prepass, but the
        // static-gx pass draws on its own pipeline and is in none, and models opt out
        // (`WowModelExt::enable_prepass`) — so every WMO floor under water is a hole, which is
        // exactly what Booty Bay's harbour is. Falling back to the authored byte there gives the
        // look this had before the prepass existed, which is the right answer for a pixel whose
        // depth we genuinely do not know.
        if (scene_depth > 0.0) {
            let scene_z = depth_ndc_to_view_z(scene_depth);
            let water_z = depth_ndc_to_view_z(in.clip_position.z);
            // View Z runs negative into the screen, so the surface is the larger of the two and the
            // column is their difference.
            thickness = max(water_z - scene_z, 0.0);
        }
#endif
        return stylised_water(
            in.world_position.xyz,
            in.clip_position.xy,
            sample_index,
            thickness,
            style_depth,
            in.shore,
            in.shore_offset,
            in.surface_normal,
            in.planar,
            in.flow,
            body_shallow,
            body_deep,
            lit,
            in.room_fog,
        );
    }

    // ---- The WMO water arms: opacity is the per-vertex authored byte through the zone's linear
    // alpha ramp, carried in `in.depth`.
    let vtx_alpha = mix(shallow.w, deep.w, depth);
    if (w.path.x > 1.5) {
        // ---- WMO interior (`0x6b6420`): fixed-function whatever the CVars (`[0xc9607c]` unread),
        // lighting off (`0x0e = 0`), fog on (`0x0f = 1`), combine preset `(0x1f, 3)`:
        //     rgb = clamp(Cf + Ct)      alpha = clamp(Af + At)
        // Cf is the pool's raw `MOMT.diffColor`; the vertex has no normal, so no sun and no sheen.
        // Preset 3 is `GL_ADD` through `GL_COMBINE` on both channels (`COMBINE_ALPHA` at
        // `0x85c2fc`), so the ripple adds to the pool's opacity rather than multiplying it.
        let body = clamp(in.vcolor.rgb + detail.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
        return vec4<f32>(
            apply_fog(body, in.world_position.xyz, in.room_fog),
            clamp(vtx_alpha + detail.a, 0.0, 1.0),
        );
    }
    if (w.path.x > 0.5) {
        // ---- WMO exterior (`0x6b6630`): `Shaders\Pixel\MapObjExtWater0.bls` (bound `0x6b6654`),
        // the shader leg of the `[0xc9607c]` gate:
        //     rgb = primary.rgb + detail.rgb + secondary·detail.a      alpha = primary.a
        // No `+0.25`, which is the ADT program's own. Lighting is on (`0x0e` never set), so
        // primary = band · clamp(ambient + diffuse·max(N·L, 0)), the band one flat colour for
        // every nibble: the deep river row, `LightIntBand` sub-17 (`water_river[1]`), an immediate
        // at `0x6b66be`. A DayNight slot is not a sub: `0x6d64d0` moves sub-8 out to `+0x4c`, so
        // the kernel's slot 16 is sub-17.
        let n_ext = normalize(in.world_normal);
        let to_light_ext = -normalize(wow_light.light_sun.xyz);
        let primary_ext = clamp(
            wow_light.light_ambient.rgb + wow_light.light_diffuse.rgb
                * max(dot(n_ext, to_light_ext), 0.0),
            vec3<f32>(0.0),
            vec3<f32>(1.0),
        ) * deep.rgb;
        let rgb_ext = primary_ext + detail.rgb + in.secondary_vtx * detail.a;
        // Alpha is `fragment.color.primary` alone: the bound program bypasses the texture
        // environment, so the interior arm's `+ At` does not apply.
        return vec4<f32>(apply_fog(rgb_ext, in.world_position.xyz, in.room_fog), vtx_alpha);
    }

    // The ADT arm. `primary`: the lit white vertex.
    let n = normalize(in.world_normal);
    let to_light = -normalize(wow_light.light_sun.xyz);
    let ndotl = max(dot(n, to_light), 0.0);
    let primary = clamp(
        wow_light.light_ambient.rgb + wow_light.light_diffuse.rgb * ndotl,
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    );

    let secondary = in.secondary_vtx;

    // colorTex: the stage-0 depth swatch.
    let swatch = swatch_at(shallow, deep, depth, w.kind.y > 0.5);

    // The `ocean0_s.bls` combine.
    var rgb = primary * swatch.rgb + detail.rgb + (secondary + vec3<f32>(0.25)) * detail.a;

    // `colorTex.a`, over the same V as the colour: deeper water is more opaque.
    let alpha = swatch.w;

    rgb = apply_fog(rgb, in.world_position.xyz, in.room_fog);

    // Raw gamma out; alpha blends in gamma space like the reference's bytes.
    return vec4<f32>(rgb, alpha);
}
