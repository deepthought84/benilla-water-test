// Liquid shader — a port of the reference's **ADT** water path (`ocean0_s.bls`) to WGSL.
//
// ## SCOPE — this file implements ONE of the reference's three liquid renderers
//
// This is the ADT MCLQ river/ocean path and nothing else. It is also, today, what benilla runs on
// WMO-embedded liquid, and that is **wrong** — see `liquid/surface.rs`'s `spawn_wmo_liquids` and the
// `wmo_liquid_arms` census. The reference has three:
//
//   * **ADT MCLQ river/ocean** — passes `0x6851b0`/`0x685010`. Two texture stages: a static depth-ramp
//     on stage 0, the animated sheet on stage 1, combined by the pixel program `ocean0_s.bls`. The
//     only liquid path with a depth ramp, and the only one whose vertex carries NO colour. **This
//     file.**
//   * **WMO MLIQ water** — `0x6b62e0` category 0, split again on the group's `MOGP.flags & 0x48`:
//     exterior `0x6b6630` (binds `MapObjExtWater0.bls`), interior `0x6b6420` (no shader, lighting
//     forced off). One texture stage, no depth ramp at all — zero references to the ADT ramp globals
//     `0xc7fbc0`/`0xc81768`/`0xc7fcd8` anywhere in `[0x6b0000, 0x6c4000)` — and alpha comes from a
//     per-vertex authored byte, not from a depth swatch. 164 groups; 90% of them interior.
//   * **magma/slime** — `0x6b68f0` (WMO) / `0x68dca0` (ADT), arm-blind, the sheet IS the body.
//
// ## The ADT combine (VERIFIED — the program is an asset, extracted and read)
//
// `Shaders\Pixel\ocean0_s.bls` out of `patch.MPQ`, verbatim ARB:
//
//   PARAM c[1] = { { 0.25 } };
//   TEX R0, fragment.texcoord[0], texture[0], 2D;   # colorTex  = the depth ramp
//   TEX R1, fragment.texcoord[1], texture[1], 2D;   # detailTex = the animated sheet
//   MAD R1.xyz, fragment.color.primary, R0, R1;
//   ADD R0.xyz, fragment.color.secondary, c[0].x;
//   MAD result.color.xyz, R0, R1.w, R1;
//   MOV result.color.w, R0;                          # R0.w still holds colorTex.a
//
//   ⇒ rgb = primary*colorTex.rgb + detailTex.rgb + (secondary + 0.25)*detailTex.a
//     alpha = colorTex.a
//
// The `+0.25` is the program's own scalar `PARAM`, not an FFP env colour (`glTexEnvfv` has zero call
// sites image-wide) and not a material. **The formula this file has always carried is right, verbatim
// — but its provenance was fiction**: the header used to cite "apitrace WoW.17 program 159" and a
// `docs/knowledge/terrain.md`, neither of which exists, and wow-re had the program attributed to a
// *character* draw (`Model2.bls` ships 32 ARBfp permutations, none containing `0.25` or
// `fragment.color.secondary`). Corrected and recorded: wow-re `terrain/scratch/water-shading-law.md`.
//
//   * colorTex (unit 0) is the depth swatch — a 2-endpoint linear lerp of the zone's dedicated
//     `Light.dbc` water rows, RAW (no ×0.711): `water_shallow.rgb` = IntBand row 16 (river/lake) / 14
//     (ocean), `water_deep.rgb` = row 17 / 15. Rebuilt **per world frame**, not baked once: the dirty
//     flags `[0xc8117c]`/`[0xc81b70]` clear at `0x680b90`/`0x680b97` and refill via `0x58acd0`, so the
//     colour and the opacity track the zone and the clock. (The earlier "reflected sky × 0.711 via
//     `FUN_0068c250`" model fingered the WRONG builder — a separate grey edge texture never bound on
//     the water unit. Rows 14–17 were right all along.)
//   * detailTex (unit 1) is the animated `lake_a`/`ocean_h` frame: RGB near-black, ALPHA = the ripple.
//     MEASURED off the shipped BLPs (DXT3, 9 authored mips): lake_a RGB mean 0.0140 and achromatic to
//     ±1 LSB, alpha mean 0.21 / p50 51 / p99 255. So it adds a faint flat lift + an achromatic shimmer
//     on the crests — NOT the body. The authored mip chain deliberately kills the shimmer with
//     distance (per-mip alpha max 255, 255, 255, 136, 68, then flat 51), which is why the sampler's
//     mips and 16× aniso are load-bearing rather than a nicety.
//   * primary = the vertex's lit colour `clamp(ambient + N·L·sun)`. The ADT liquid vertex has no
//     colour element, so `glColor` is the device default `(1,1,1,1)` tracked into material
//     ambient+diffuse by `glColorMaterial(FRONT_AND_BACK, AMBIENT_AND_DIFFUSE)`, and `GL_LIGHTING` is
//     ON at both water draws (lava explicitly turns it off; water does not).
//   * secondary = the specular sheen — see `sun_sheen` for what is verified in it and what is not.
//   * alpha = swatch.a over the SAME V as the colour. LightParams endpoints: river 0.5→1.0, ocean
//     0.75→1.0. Deeper water = more opaque. **Open**: wow-re reads the byte-verified ADT water alpha
//     as the `0xc7fbc0` LUT's `1.6·(i/63)^8` curve rather than the linear `127+2·row` this file
//     applies — a much later-breaking ramp. Not changed here; it is a look change to every ADT water
//     surface in the game and it belongs in the same A/B as the WMO arms.
//
// **Both `ocean0_s.bls` and `MapObjExtWater0.bls` are CVar-gated** — `specular` and `pixelShaders`
// (registered `0x6886a0`/`0x688712`) default to `"0"`, and with them off there is no program, no
// specular, the stage-1 combine is a plain ADD, and blend is never set so water draws OPAQUE. Our
// reference install's `Config.wtf` sets both to `"1"`, so every capture and every director comparison
// is against the shader leg — which is the leg we implement. The two do not compose: an active ARB
// program bypasses the texture environment entirely.
//
// A SINGLE swatch row (V) indexes both the colour and the alpha — they track together. Each kind reads
// its OWN verified LUT, built side by side in `FUN_0068c4c0`: `clamp(byte/42)` for river/lake
// (`c81768`, `FUN_0068d790`, saturates ~5 yd → the channel middle hits the deep teal row) and
// `clamp(byte/255)` for ocean (`c7fcd8`, `FUN_0068d690`, saturates ~148 yd). The divisors differ
// because the authored bytes do — the sea ramps 1.72 byte/yd against a river's 8.96, and 83 % of ocean
// vertices are pinned at 255 outright (decision 2069, `examples/liquid_depth_census`). (Earlier cuts:
// ripple-as-colour → black; ×8 over-saturated; FLAT colour killed the gradient; sky×0.711 was the wrong
// builder; `byte/255` on the RIVER was the wrong LUT → river middle never went teal. Corrected to
// rows 14–17 raw lerp + the /42 V on rivers, 2026-05-31.)
//
// Two-sided comes from the material (cull off) and is right for EVERY kind: all four reference liquid
// passes force GL_CULL_FACE off at pass entry against a cull-ON device baseline, and `glFrontFace` is
// not even imported (VERIFIED wow-re `liquid-render-state-sided` §6). Blending is per KIND, decided in
// liquid.rs: water/ocean blend with depth-write off; magma/slime are opaque and depth-write. Fog + gamma
// mirror terrain.wgsl (planar eye-Z GL_LINEAR fog in gamma space; raw gamma out — GAMMA LANE, 0161), fog
// applies to every kind — magma and slime included — and WHICH fog block a surface takes is per-surface:
// a WMO interior group's own pool fogs with the interior block, everything else with the scene block
// (see `apply_fog`). Light, fog and both water swatches all come off the ONE shared global-light buffer.

#import bevy_pbr::{
    mesh_functions,
    forward_io::Vertex,
    view_transformations::{
        position_world_to_clip, position_world_to_ndc, ndc_to_uv, frag_coord_to_uv,
        depth_ndc_to_view_z,
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

@group(#{MATERIAL_BIND_GROUP}) @binding(105) var reflection_tex: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(106) var reflection_samp: sampler;

// The live wave field (`benilla_world::liquid::ripple_sim`) — a window of simulated water carried
// along with the viewer. R/G = surface slope, B = the foam the disturbance has whipped up.
@group(#{MATERIAL_BIND_GROUP}) @binding(108) var wake_tex: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(109) var wake_samp: sampler;

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
    /// `x` = the screen-space march is switched off (`$WOW_NO_SSR`); `y` = paint its confidence
    /// instead of the water (`$WOW_SSR_SHOW`); `zw` spare.
    flags: vec4<f32>,
};
@group(#{MATERIAL_BIND_GROUP}) @binding(107) var<storage, read> water_reflect: WaterReflect;

struct LiquidParams {
    // x = fullbright (magma/slime); y = ocean swatch; z = interior fog; w = sun-sheen shininess.
    kind: vec4<f32>,
    // x = WHICH RENDERER (see `LiquidPath`): 0 = ADT MCLQ, 1 = WMO exterior, 2 = WMO interior.
    // y/z/w reserved.
    path: vec4<f32>,
    // x = reserved (frame 0); y = frame count; z = the SCROLL FLAG (1 only on the nibble-6/7
    // WMO magma/slime lane — the reference's animated stage-0 texture matrix; see
    // `liquid/surface.rs`'s `scrolls`, and `apply_scroll` below); w = the clock enable (0 on a
    // deterministic run — the whole animation freezes at frame 0 / scroll 0, the 0600 capture
    // pin, baked at material build).
    anim: vec4<f32>,
};
@group(#{MATERIAL_BIND_GROUP}) @binding(102) var<uniform> w: LiquidParams;

// The shared global light (`lighting::global_light`) — the SAME buffer terrain and the models read,
// mirrored here as its canonical row prefix. Liquid used to carry its own copy of every one of these
// values, re-pushed per material by `apply_wow_lighting`; that copy is what left it with only the
// scene fog and no way to see the interior block (decision 0691).
struct WowLight {
    light_ambient: vec4<f32>,      // 0  rgb = ambient; w = Mod2x scale
    light_diffuse: vec4<f32>,      // 1  rgb = sun diffuse; w = clamp flag
    light_sun: vec4<f32>,          // 2  xyz = sun TRAVEL dir (to-light = −xyz)
    light_spec: vec4<f32>,         // 3  rgb = row-9 specular colour; w = TERRAIN shininess (liquid uses w.kind.w)
    fog_color: vec4<f32>,          // 4  rgb = scene fog (block 1, gamma 0..1); w = enable (>0.5)
    fog_params: vec4<f32>,         // 5  x = start yd; y = end yd; w = the farclip wall
    _sh: array<vec4<f32>, 6>,      // 6-11  model SH coeffs — unread here
    _sh_c16: vec4<f32>,            // 12
    water_river: array<vec4<f32>, 2>, // 13-14 shallow/deep river-lake swatch (IntBand 16/17); w = alpha
    water_ocean: array<vec4<f32>, 2>, // 15-16 shallow/deep ocean swatch    (IntBand 14/15); w = alpha
    _grade: vec4<f32>,             // 17
    wmo_fog_color: vec4<f32>,      // 18 rgb = INTERIOR fog (block 2); w = enable
    wmo_fog_params: vec4<f32>,     // 19 x = start yd; y = end yd
};
@group(#{MATERIAL_BIND_GROUP}) @binding(90) var<storage, read> wow_light: WowLight;

struct LiquidVsOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) depth: f32,
    // The sun-sheen `secondary` evaluated PER-VERTEX (the faithful 1.12 path: the real client computes
    // the Blinn highlight in its FFP vertex shader and interpolates it across the coarse water mesh).
    // The fragment shader uses this interpolated value directly.
    @location(4) secondary_vtx: vec3<f32>,
    // The mesh's vertex COLOUR, which on a WMO INTERIOR pool carries its `MOMT.diffColor` body
    // colour — the reference's own interior water vertex carries a colour dword for exactly this.
    // White on every other lane (and on any mesh with no colour attribute), where nothing reads it.
    @location(5) vcolor: vec4<f32>,
    /// Distance from this vertex to the waterline in yards, for the shore foam. Packed into UV1.y
    /// by the mesh builder; see the foam block in `stylised_water`.
    @location(6) shore: f32,
    /// Offset from this fragment to the nearest point on the waterline, world XZ yards. Its LENGTH
    /// is the distance the foam band is drawn from — taken here rather than interpolated, because
    /// distance to a curve has a crease along the curve and an interpolated crease is a wrong
    /// crease. See `benilla_assets::materials::ATTRIBUTE_WOW_SHORE_OFFSET`.
    @location(7) shore_offset: vec2<f32>,
    // The per-frame INTERIOR FOG lane for this surface's own room (decision 1787): the client's
    // `[0xca7f00]`, which gates the WMO liquid pass's block-2 submit (`0x6b6323`–`0x6b6342`)
    // exactly as it gates the geometry pass — so a pool and the walls around it can never
    // disagree about which fog they wear. Carried on `MeshTag` bit 30, read once in the vertex
    // stage and flat-interpolated (the whole surface is one instance).
    @location(8) @interpolate(flat) room_fog: u32,
}

// Sun sheen (`secondary`): a Blinn highlight of the sun on the flat water surface — the glint that's
// strongest at grazing (sunrise/sunset) sun. `secondary = light_spec.rgb · (N·H)^shininess`. Shared by
// both stages so the per-vertex (faithful) and per-pixel (current) paths run IDENTICAL math — only the
// EVALUATION DOMAIN differs (interpolated vs evaluated per fragment).
fn sun_sheen(world_normal: vec3<f32>, world_pos: vec3<f32>) -> vec3<f32> {
    let n = normalize(world_normal);
    let to_light = -normalize(wow_light.light_sun.xyz);
    // LOCAL viewer, not the infinite-viewer constant: the reference sets
    // `GL_LIGHT_MODEL_LOCAL_VIEWER = 1` at `0x59cf89`, so `H = normalize(L + normalize(eye − vertex))`
    // and the eye vector is recomputed per vertex (VERIFIED wow-re `water-shading-law.md`). It is
    // the difference between a highlight and a flood: with the infinite viewer `N·H` is very nearly
    // CONSTANT over a flat water plane, so at high sun the whole sheet saturates at once instead of
    // carrying a glint that moves with the camera.
    let to_view = normalize(view.world_position.xyz - world_pos);
    let half_v = normalize(to_light + to_view);
    let ndoth = max(dot(n, half_v), 0.0);
    // The reference's specular is additionally gated on `N·L > 0` (the fixed-function rule: a light
    // behind the surface contributes no specular). **Deliberately not ported**, because it cannot
    // fire here: `DayNight::SetDirection` holds the LIGHTING sun's azimuth constant and wobbles its
    // elevation only between +20° and +37° — it is always above the horizon, night being the colours
    // going dark rather than the sun setting (`lighting::daynight::sun_direction`, and the separate
    // *visible* sun is the one that rises and sets). Against a liquid surface's flat up normal `N·L`
    // is therefore positive at every minute of the day, so the gate is a branch that can only ever
    // take one arm. Named rather than added: it is real, it is verified, and it is inapplicable.
    //
    // Shininess is WATER's own (`w.kind.w`), not the shared row-3 terrain exponent. Material
    // specular is white (`SetRenderState(3, 0xffffffff)`) and the exponent is 6.0 (`[0x8102e8]`),
    // both VERIFIED — so `light_spec.rgb` is the whole scale, and it is the one input here that is
    // still INFERRED: wow-re pinned the mechanism (`CGLight+0x48` → `collector+0x6c` → `0x589d80(0)`
    // → `glLightfv(GL_SPECULAR)`) but not the number, and reads it as a warm ≈(1.0, 0.91, 0.76)
    // against the row-9 feed we use. Left on row 9 until that lands rather than swapped for an
    // estimate.
    return wow_light.light_spec.rgb * pow(ndoth, max(w.kind.w, 1.0));
}

// The lava/slime **surface scroll**: the reference's animated stage-0 texture matrix, which for
// liquid-type nibbles 6 and 7 is the identity with element 13 — the **v translate** — set to
// `fmod(uptime_s, 10.0) · 0.1` (VERIFIED wow-re `liquid-uv-scroll-law.md`, six-agent §5, matrix
// built at `0x6b68f0` and pushed at stage 0 by `0x6b6ae3`). A texture matrix times `(s, t, 0, 1)`
// with only element 13 non-identity is exactly `t += phase`, so the whole mechanism is this add —
// no matrix, no second sampler, no cost on the paths that do not scroll (`anim.z` is a hard 0
// there, which the CPU side guarantees rather than the shader branching on it).
//
// A full repeat every 10 s, so `REPEAT` wrapping makes the sawtooth's reset invisible. Only the
// rate and the period are reproducible — the reference's phase comes off `GetTickCount`, i.e. the
// machine's uptime, so its absolute value is not a thing to match.
// The liquid clock: `globals.time` (the same wall-elapsed seconds the CPU cycler used to read)
// under the build-time enable. Both animations below are pure functions of it — the CPU-side
// 24 Hz `Assets::get_mut` tick this replaces mutated ~14 materials a tick and its Modified
// fallout (uniform re-uploads, bind-group rebuilds, whole-population `AssetChanged` arming)
// measured 0.28 cpu_ms/frame at the SW pin (2026-08-18 bracket).
fn anim_time() -> f32 {
    return w.anim.w * globals.time;
}

// The 24 fps frame flip — 30 frames over 1.25 s (VERIFIED `FUN_0068aac0`), floor-quantized to
// the tick exactly as the reference's integer frame index is.
fn frame_layer() -> i32 {
    return i32(floor(anim_time() * 24.0) % max(w.anim.y, 1.0));
}

fn apply_scroll(uv: vec2<f32>) -> vec2<f32> {
    // v += (t mod 10) · 0.1 — repeats/s `[0x801620]` = 0.1, period `[0x80e5a0]` = 10.0 (VERIFIED
    // wow-re `liquid-uv-scroll-law.md` §5): a sawtooth over exactly one repeat, invisible under
    // REPEAT wrapping. CONTINUOUS now, where the CPU tick quantized it to 1/24 s — the reference
    // itself rebuilds the matrix per draw off a millisecond clock, so this is the more faithful
    // reading, not a new liberty. anim.z is the flag: hard 0 on every non-scrolling lane.
    return vec2<f32>(uv.x, uv.y + w.anim.z * fract(anim_time() / 10.0));
}

// Distance fog — planar eye-Z, GL_LINEAR, gamma space (mirrors terrain.wgsl). Applied to EVERY liquid
// kind, because the reference never disables fog for a liquid batch: the device default for GL_FOG is
// **ON** (`0x593bf0` writes state id `0x0f` = 1) and all 42 fog-enable setters in the binary are
// Push/Pop-scoped, so what a batch inherits at its draw is that default. The ADT lava pass sets only
// cull/lighting/blend (`0x6855ca`/`0x6855d6`/`0x6855e2`) and the WMO magma/slime arm sets only lighting
// (`0x6b6afe`) — neither touches fog — while the WMO *river* arm goes out of its way to re-assert
// `(0x0f, 1)`, which only makes sense in a world where fog-on is liquid's intended state. (VERIFIED
// wow-re `liquid-render-state-sided` §1–§3, §5.)
//
// WHICH fog is a per-surface choice, and it is the reference's own (VERIFIED wow-re `fog-env-state`
// §5, the complete 6-site submit census). The device holds two fog blocks: **block 1** (`+0x70/74/78`)
// is the scene fog, submitted once a frame from `WorldFrame::Render` (`0x66ff20`), and **block 2**
// (`+0x80/84/88`) is block 1 smoothed toward the MFOG/zone target over ~4 s (`0x6cf054`+) — the
// interior haze. Only two call sites in the whole binary re-submit block 2, and they are the WMO
// *geometry* pass (`0x6b51d9`/`0x6b51ea`) and the WMO *liquid* pass (`0x6b6323`–`0x6b6342`), both under
// the same `[0xca7f00]` gate. So an interior room's pool takes the room's fog, in lockstep with the
// walls around it; ADT liquid submits nothing and draws under the scene block. `w.kind.z` is the STATIC
// half of that gate, resolved at spawn from the group's `MOGI & 0x48`; the per-frame half is the
// room's own `[0xca7f00]` bit on `MeshTag` bit 30 (decision 1787 — the flood decides it, and the
// WMO geometry lane reads the same answer, which is what keeps the two in step).
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
/// Screen-space reflection is exact at any orientation and, measured with `$WOW_SSR_SHOW`, is
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
    let g = (textureSample(ripples, ripples_samp, uv).rg * 2.0 - 1.0) * weight;
    return transpose(r) * g;
}

/// How many steps the screen-space march takes before giving up, and how the step grows.
///
/// Geometric, not uniform: what a reflection needs resolved finely is the first few yards off the
/// surface — the bank, the trunk base, the hull at the waterline — while the far end of the ray is
/// a tree canopy that one long step lands on just as well. A uniform march fine enough for the near
/// field would need hundreds of steps to reach the far one.
const SSR_STEPS: i32 = 28;
const SSR_FIRST_STEP: f32 = 0.35;
const SSR_GROWTH: f32 = 1.22;

/// How many times the crossing is bisected once the march has stepped past a surface.
///
/// The march finds the interval; this finds the point. Five halvings take the last step's error
/// down by 32x, which is what turns a stair-stepped reflection edge into a clean one.
const SSR_REFINE: i32 = 5;

// **A finer march was tried and buys nothing.** 40 steps growing 1.12x from 0.25 yd — roughly twice
// the near-field resolution — moved the hit rate over the Elwynn reach from 80.2 to 80.0, i.e. not
// at all. The red speckle inside that stream's confidence is therefore NOT undersampling; it is
// the thickness rejection and the genuine gaps between its rocks, and more steps only cost more.
// Anyone reaching for these numbers to clean that speckle should look at [`SSR_THICKNESS`] instead.

/// How far behind a surface a crossing may be and still count as a hit, in yards.
///
/// A depth buffer records a front face and says nothing about what is behind it, so a ray that
/// passes *well* behind a thin object has not hit it — it has gone past it, through space the
/// buffer cannot describe. Accepting those is what paints a tree's silhouette onto water the tree
/// does not overhang. Six yards is generous enough for real geometry and mean enough to reject a
/// ray that has left the visible world.
const SSR_THICKNESS: f32 = 6.0;

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

#ifdef DEPTH_PREPASS
/// The scene depth under a world point, and the point's own depth, both in view space.
///
/// View Z runs negative into the screen, so "the ray is behind the surface" reads as `ray < scene`.
/// A pixel the prepass never wrote clears to reverse-Z zero, which is the far plane — returned as
/// a sentinel rather than taken at face value, exactly as the thickness lane does with it.
fn ssr_probe(p: vec3<f32>, sample_index: u32) -> vec3<f32> {
    let ndc = position_world_to_ndc(p);
    if (ndc.z <= 0.0 || any(abs(ndc.xy) > vec2<f32>(1.0))) {
        return vec3<f32>(0.0, 0.0, -1.0); // off screen or behind the eye
    }
    let uv = ndc_to_uv(ndc.xy);
    let px = uv * view.viewport.zw + view.viewport.xy;
    let d = prepass_depth(vec4<f32>(px, 0.0, 0.0), sample_index);
    if (d <= 0.0) {
        return vec3<f32>(0.0, 0.0, -1.0); // nothing was drawn here — unknown, not infinitely far
    }
    // x = the scene's view z, y = the ray's, z = the smaller UV distance to the frame's edge.
    let edge = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
    return vec3<f32>(depth_ndc_to_view_z(d), depth_ndc_to_view_z(ndc.z), edge);
}

/// March the reflected ray through the prepass depth and read the scene snapshot where it lands.
///
/// **This is the half of the water's reflection that a plane cannot do.** The direction traced is
/// `reflect(-V, N)` on the fragment's OWN normal — ripple, wave sim and macro surface slope already
/// summed — so a sloped stream and a rippled lake are the same code path, and neither needs a
/// mirror to be built for it. What it cannot do is see off the screen, which is why it returns a
/// confidence rather than just a colour: where the ray leaves the frame the planar capture and the
/// sky mix behind it take over.
///
/// The water itself is not in the prepass (Bevy excludes alpha-blended materials, and
/// `liquid::depth`'s doc keeps it that way deliberately), so there is no self-intersection to bias
/// against — the first step off the surface is already reading opaque geometry only.
fn ssr_trace(origin: vec3<f32>, dir: vec3<f32>, sample_index: u32) -> SsrHit {
    var out: SsrHit;
    out.rgb = vec3<f32>(0.0);
    out.conf = 0.0;

    var step = SSR_FIRST_STEP;
    var t = SSR_FIRST_STEP;
    var prev_t = 0.0;
    for (var i = 0; i < SSR_STEPS; i = i + 1) {
        let probe = ssr_probe(origin + dir * t, sample_index);
        if (probe.z < 0.0) {
            return out; // ran off the frame or into a pixel with no depth: no hit, no confidence
        }
        // Crossed behind the surface, and not so far behind that the ray has left the world the
        // depth buffer can describe — see [`SSR_THICKNESS`].
        if (probe.y < probe.x && probe.x - probe.y < SSR_THICKNESS) {
            // Bisect the interval the march just stepped over.
            var lo = prev_t;
            var hi = t;
            for (var r = 0; r < SSR_REFINE; r = r + 1) {
                let mid = 0.5 * (lo + hi);
                let m = ssr_probe(origin + dir * mid, sample_index);
                if (m.z < 0.0) {
                    break;
                }
                if (m.y < m.x) {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            let hit = origin + dir * hi;
            let ndc = position_world_to_ndc(hit);
            let uv = ndc_to_uv(ndc.xy);
            out.rgb = textureSampleLevel(scene_tex, scene_samp, uv, 0.0).rgb;
            let edge = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
            out.conf = smoothstep(0.0, SSR_EDGE_FADE, edge);
            return out;
        }
        prev_t = t;
        step = step * SSR_GROWTH;
        t = t + step;
    }
    return out;
}
#endif

fn stylised_water(
    world_pos: vec3<f32>,
    frag_coord: vec2<f32>,
    /// Which MSAA sample this fragment is — the prepass depth is multisampled and the march has to
    /// read the same one the thickness lane does.
    sample_index: u32,
    // Yards of water between this fragment and whatever opaque surface is behind it, or a negative
    // number where the scene depth is unavailable (the reference lane, no prepass) — in which case
    // every term below falls back to the authored depth byte, exactly as it did before.
    thickness: f32,
    depth: f32,
    shore: f32,
    shore_offset: vec2<f32>,
    shallow: vec4<f32>,
    deep: vec4<f32>,
    lit: vec3<f32>,
    room_fog: u32,
) -> vec4<f32> {
    let t = anim_time();
    let xz = world_pos.xz;
    // In uniform control flow, before anything branches — see [`surface_tilt`].
    let tilt = surface_tilt(world_pos);
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
        + ripple_layer(
            xz,
            vec2<f32>(-0.021, 0.010),
            inv_tile,
            1.00,
            t,
            vec2<f32>(0.4540, 0.8910),
        )
        + ripple_layer(
            xz,
            vec2<f32>(0.014, -0.026),
            inv_tile / 0.30,
            0.70,
            t,
            vec2<f32>(-0.2924, 0.9563),
        )
        // Dialled back from 0.30 with the specular: the finest layer is the crinkle, and a fine
        // crinkle under a tight highlight is the texture of oil on water.
        + ripple_layer(
            xz,
            vec2<f32>(0.041, 0.033),
            inv_tile / 0.12,
            0.20,
            t,
            vec2<f32>(-0.8572, 0.5150),
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
    // Clamped, because the lane doubles as the `$WOW_WAKE_SHOW` debug switch below.
    let sim_w = min(water_reflect.sim.w, 1.0)
        * inside
        * smoothstep(0.0, WAKE_EDGE_FADE, sim_edge);
    let wake_slope = (sim_tex.rg * 2.0 - 1.0) * sim_w;
    // The wave's own height, signed — how much water the disturbance has put under this pixel.
    let wake_height = (sim_tex.b * 2.0 - 1.0) * sim_w;
    let wake_foam = saturate((length(wake_slope) - WAKE_FOAM_FLOOR) * WAKE_FOAM_GAIN);
    // `$WOW_WAKE_SHOW` — the field, painted flat, with the window's own extent as the black border.
    // Nothing to read into: either there are waves on the screen or the simulation is not arriving.
    if (water_reflect.sim.w > 1.5) {
        return vec4<f32>(sim_tex.rgb * inside, 1.0);
    }

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
    let n = normalize(vec3<f32>(
        slope.x * NORMAL_STRENGTH + wake_slope.x,
        1.0,
        slope.y * NORMAL_STRENGTH + wake_slope.y,
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
    var rgb = mix(body, sky * lit, mix(0.02, 0.45, fresnel));

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
    if (water_reflect.params.y > 0.0
        && water_reflect.flags.x < 0.5
        && view.world_position.y > world_pos.y) {
        ssr = ssr_trace(world_pos, reflect(-to_view, n), sample_index);
    }
#endif

    // `$WOW_SSR_SHOW` — green where the march found a hit and how much it is believed, red where
    // it found nothing. A reflection that is merely faint and one that was never traced look
    // identical in the final image, which is the confusion this ends.
    if (water_reflect.flags.y > 0.5) {
        return vec4<f32>(1.0 - ssr.conf, ssr.conf, 0.0, 1.0);
    }

    let plane_error = abs(world_pos.y - water_reflect.params.x);
    let trust = plane_trust(plane_error, water_reflect.params.w, tilt);
    if (water_reflect.params.y > 0.0 && trust > 0.0 && view.world_position.y > world_pos.y) {
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
        let lift = 2.0 * (water_reflect.params.x - world_pos.y);
        let lifted = position_world_to_ndc(world_pos + vec3<f32>(0.0, lift, 0.0));
        // Behind the eye (reverse-Z puts `w < 0` there, and the divide flips `z` negative with it)
        // or thrown well off the side, the reprojection means nothing; the uncorrected UV is the
        // image this drew before the correction existed and is the honest fallback.
        let usable = lifted.z > 0.0 && all(abs(lifted.xy) < vec2<f32>(1.5));
        let uv = select(frag_coord_to_uv(frag_coord), ndc_to_uv(lifted.xy), usable);
        let ruv = clamp(
            vec2<f32>(1.0 - uv.x, uv.y)
                + n.xz * water_reflect.params.z * (1.0 + DISTORT_LIFT * (1.0 - trust)),
            vec2<f32>(0.002),
            vec2<f32>(0.998),
        );
        let mirrored = textureSample(reflection_tex, reflection_samp, ruv);
        // Schlick again, on the same normal: water reflects almost nothing straight down and almost
        // everything at a glancing angle, which is most of why a lake reads as a lake.
        //
        // The ceiling is a **look choice against the physics**, and named as one. Schlick's own top
        // end is 1.0 — at a grazing angle real water is a mirror — and a mirror is what this drew:
        // too clean, too complete, more polished glass than a lake with a breeze on it. Holding the
        // top at [`REFLECT_MAX`] keeps a quarter of the water's own body in the picture at every
        // angle, which is what stops it reading as perfect. It is a dimmer, not a distorter: the
        // knob for a *broken* reflection rather than a fainter one is `REFLECT_DISTORT` on the CPU
        // side.
        // `1 - ssr.conf`: where the march found the answer itself, the capture stands down rather
        // than averaging with it. Two reflections of one surface blended together is a double
        // image, not a better one.
        let amount = mix(0.02, REFLECT_MAX, fresnel)
            * water_reflect.params.y
            * mirrored.a
            * trust
            * (1.0 - ssr.conf);
        rgb = mix(rgb, mirrored.rgb, amount);
    }

    // …and the march's own contribution, on the same Fresnel weight the tiers below it use, so the
    // water does not change how reflective it is depending on which tier answered.
    if (ssr.conf > 0.0) {
        rgb = mix(
            rgb,
            ssr.rgb,
            mix(0.02, REFLECT_MAX, fresnel) * water_reflect.params.y * ssr.conf,
        );
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
    // `$WOW_WATER_DEPTH_SHOW` — the water column in yards, as greyscale (black 0, white 8).
    //
    // Kept, not scaffolding. Everything below now depends on a quantity that is invisible in the
    // final image and wrong in ways that look like art problems: too little and the water is flat,
    // too much and the shallows vanish. Being able to look at the number directly is what separated
    // "the soft edge is too wide" from "the absorption is wrong" the first time they were confused
    // for each other.
    if (water_reflect.sky_zenith.w > 0.5) {
        let t = max(thickness, 0.0) / 8.0;
        return vec4<f32>(vec3<f32>(saturate(t)), 1.0);
    }
    // `$WOW_SCENE_SHOW` — paint the scene snapshot straight onto the water at this fragment's own
    // screen position. If the copy is landing, the water becomes a window showing the world behind
    // the camera's own view of it, seamlessly continuous with the frame around it; if the node
    // never ran, it is black. Nothing downstream can distinguish those two, which is why this
    // exists before the march that will depend on it.
    if (water_reflect.moon.w > 1.5) {
        let suv = frag_coord_to_uv(frag_coord);
        return vec4<f32>(textureSample(scene_tex, scene_samp, suv).rgb, 1.0);
    }
    // `$WOW_WATER_TILT_SHOW` — the geometric tilt this fragment thinks it has, as greyscale, black
    // flat and white at [`SLOPE_LIMIT`]. The trust term reads this number and nothing in the final
    // image shows it directly, so a tilt that is wrong looks exactly like a reflection that is
    // wrong — which is the confusion this exists to end.
    if (water_reflect.sky_horizon.w > 0.5) {
        return vec4<f32>(vec3<f32>(saturate(tilt / SLOPE_LIMIT)), 1.0);
    }
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
}

@vertex
fn vertex(in: LiquidVertex) -> LiquidVsOut {
    var out: LiquidVsOut;
    let world_from_local = mesh_functions::get_world_from_local(in.instance_index);
    out.world_position =
        mesh_functions::mesh_position_local_to_world(world_from_local, vec4<f32>(in.position, 1.0));
    out.clip_position = position_world_to_clip(out.world_position.xyz);
    out.world_normal = mesh_functions::mesh_normal_local_to_world(in.normal, in.instance_index);
    out.uv = in.uv;
#ifdef VERTEX_COLORS
    out.vcolor = in.color;
#else
    out.vcolor = vec4<f32>(1.0);
#endif
    // Per-vertex MCLQ depth (0..1) packed into UV1.x; drives the opacity ramp.
    out.depth = in.uv_b.x;
    // UV1.y is the distance to the waterline in yards; drives the far field, and the shore foam on
    // any mesh without the offset attribute below.
    out.shore = in.uv_b.y;
#ifdef LIQUID_SHORE_OFFSET
    // A DIRECTION, so it takes the placement's rotation and scale but not its translation — which
    // is what `mat3(world_from_local)` is. Interpolating the offset is equivalent to interpolating
    // the nearest point and the position separately, because both are linear; the length is taken
    // in the fragment stage, where the crease belongs.
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
    // The faithful per-vertex sun sheen — interpolated across the coarse mesh by the fragment stage.
    out.secondary_vtx = sun_sheen(out.world_normal, out.world_position.xyz);
    // `MeshTag` bit 30 — see `LiquidVsOut::room_fog`. ADT surfaces carry no tag (0 ⇒ scene fog,
    // which is their only lane anyway).
    out.room_fog = mesh_functions::get_tag(in.instance_index) & 0x40000000u;
    return out;
}

// ── The ADT depth swatch, as the reference actually BUILDS and SAMPLES it ────────────────────
//
// `FUN_0068a830` fills an 8×64 texture (U inert — each row is `rep stosd`-replicated across all 8
// columns, matching the vertex fill's pinned `u = 0.5`). Its rows are an exact 32-bit integer
// accumulator in **byte space**, not a float lerp:
//
//     step   = ((c1 - c0) << 8) >> 6     ; == 4*(c1 - c0) EXACTLY (six zero low bits, so the sar
//     acc    = c0 << 8 ; acc += step     ;  cannot truncate -> no accumulator drift over 64 rows)
//     row(i) = (acc >> 8) & 0xff         ; == c0 + floor(i*(c1 - c0) / 64),  i = 0..63
//
// Two things that costs us, both missed until wow-re's ocean §5 round (decision 2074):
//
//   * **the ramp never reaches the deep endpoint.** Row 63 is `c0 + floor(63*d/64)`, ≈98.4 % of the
//     way, not `c1`. Our old `mix(shallow, deep, V)` ran the last 1/64 of the ramp that does not
//     exist.
//   * **the ocean's last row alone is darkened.** The tail `[0x68a9c0, 0x68aa36)` is gated on *last
//     row* AND *selector == 0* (ocean; river is selector 1 and gets neither): RGB→HSV,
//     `0x68aa13 fmul [0x8102ec]` — **V *= 0.9** (`0x3f666666`, the f32 nearest 0.9) — HSV→RGB, then
//     `0x7bbec0`/`0x7bbec8` forcing that row's alpha to 255. `0x7bbd60`'s HSV→RGB writes every
//     channel as a product with V and the tail touches neither H nor S (and `S == 0` returns
//     `(V,V,V)` without reading H, so achromatic input takes no hue shift), so the colour half is
//     exactly `floor(0.9 * byte)` per channel — transcribed at f32 and run over all 2^24 byte
//     triples: 99.03 % bit-exact, **max deviation 1/255**.
//
// It is not a corner case: ~80 % of the ocean vertices in the shipped world carry depth byte 255,
// so `V = 1.0` and this row IS the open sea (decision 2069's census).
//
// Sampling is **LINEAR/LINEAR, no mip, D3DTADDRESS_CLAMP** — flags word `0x201`, `0x5a2a18 and 7`
// -> `0x85c7d8` row 1 `{MAG, MIN, MIP} = {2, 2, 0}`, `0x5a2a62 shr 3` -> `0x80a254[0] = 3`;
// corroborated by a GL capture of this exact texture (8x64, levels 1, CLAMP_TO_EDGE, LINEAR/LINEAR).
// So V maps to the texel coordinate `V*64 - 0.5` and blends across neighbouring rows: the darkening
// **ramps in over the final 1/64 of V**, it does not step. That band is the only place any of this
// is visible on a shore.
//
// The WMO arms do NOT come through here — their opacity is a different, 256-entry ramp
// (`0xca7f10`), whose 1/256 granularity the plain lerp above reproduces.
fn swatch_row(shallow: vec4<f32>, deep: vec4<f32>, i: f32, ocean: bool) -> vec4<f32> {
    // RGB endpoints arrive already BYTES (`0x68a8fb`/`0x68a902` read two packed dwords straight out
    // of DayNight state — there is no quantization step for RGB at all); the ALPHA endpoints are
    // `LightParams` floats the reference quantizes `floor(v*255)` first.
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
fn fragment(
    in: LiquidVsOut,
#ifdef MULTISAMPLED
    // `prepass_depth` reads a multisampled texture and needs to know which sample this is. The
    // world camera runs at four, so this is the live path.
    @builtin(sample_index) sample_index: u32,
#endif
) -> @location(0) vec4<f32> {
#ifndef MULTISAMPLED
    let sample_index = 0u;
#endif
    // HARD FAR-CLIP WALL (same as terrain/models, see terrain.wgsl): discard water beyond the
    // projection far plane so lakes/rivers don't render past the wall. `fog_params.w` = farclip
    // (0 ⇒ disabled).
    if (wow_light.fog_params.w > 0.0) {
        let clip_z = -(view.view_from_world * vec4<f32>(in.world_position.xyz, 1.0)).z;
        if (clip_z > wow_light.fog_params.w) {
            discard;
        }
    }

    // Animated frame. For water/ocean this is the DETAIL ripple (RGB ≈ near-black, ALPHA = ripple);
    // for magma/slime it is the OPAQUE BODY texture.
    // `view.mip_bias`: the render-scale LOD compensation (1639), 0.0 at native and above.
    let detail = textureSampleBias(
        frames,
        frames_samp,
        apply_scroll(in.uv),
        frame_layer(),
        view.mip_bias,
    );

    // Magma / slime (kind.x > 0.5): the animated texture IS the opaque body colour — no depth swatch
    // (the ADT liquid vertex format carries no colour element at all, and the WMO one is a hard
    // `0xffffffff`, so there is nothing to modulate the sheet by) and no N·L (lighting state 0 on both
    // paths). It IS fogged, like every other liquid batch.
    //
    // The earlier "emissive / no-darken / no fog" reading here was WRONG, and wrong twice over: it came
    // from the ADT-lava row of `rf-water-liquid-type-texture-material`, which read GX state `0x37` as an
    // emissive path when `0x37` is the per-stage TEXTURE-MATRIX enable pushing an identity — a texgen
    // *reset*; and that row is the ADT queue, which never dispatches slime at all (Undercity's slime is
    // WMO liquid). Skipping fog is what made a submerged slime surface a flat unshaded sheet at any
    // depth instead of one that recedes into the murk. (VERIFIED wow-re `liquid-render-state-sided`
    // §3/§3.1/§5, which corrects that row.)
    if (w.kind.x > 0.5) {
        return vec4<f32>(apply_fog(detail.rgb, in.world_position.xyz, in.room_fog), 1.0);
    }

    // Per-vertex swatch coord V (in `in.depth`, computed CPU-side in wow-formats/liquid.rs): river/lake
    // = `clamp(byte/42)` (VERIFIED WoW.exe `c81768` LUT / `FUN_0068d790`, saturating ~5 yd so the channel
    // middle reaches the deep/teal row), ocean = `clamp(byte/255)` (VERIFIED `c7fcd8` / `FUN_0068d690`,
    // its own LUT on its own authored byte scale — decision 2069). The depth swatch
    // is a plain 2-endpoint lerp (`FUN_0068a830`), so a SINGLE V indexes BOTH the colour and the alpha
    // row: colour `shallow→deep` and opacity `shallow_α→deep_α` track together. (Earlier `×4` colour
    // compression + the gentle `byte/255` V were band-aids for a wrong "V tops at 0.31" belief — removed.)
    let depth = clamp(in.depth, 0.0, 1.0);
    // The kind's swatch endpoints, off the shared light: ocean reads rows 15/16 (IntBand 14/15),
    // river/lake rows 13/14 (IntBand 16/17). Both are packed every frame by `build_light_data`.
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
            body_shallow,
            body_deep,
            lit,
            in.room_fog,
        );
    }

    // ---- The two WMO water arms ------------------------------------------------------------
    //
    // Neither is the ADT combine below. `0x6b62e0`'s category 0 splits on the owning group's
    // `MOGP.flags & 0x48`, and both halves bind ONE texture (the animated sheet) with no depth ramp
    // anywhere — there is not a single reference to the ADT ramp globals `0xc7fbc0`/`0xc81768`/
    // `0xc7fcd8` in all of `[0x6b0000, 0x6c4000)`. Opacity on both is the per-vertex authored byte
    // through the zone's linear alpha ramp, which `in.depth` carries and this lerp reproduces
    // (`wmo_water_alpha_v`). VERIFIED wow-re `terrain/scratch/water-shading-law.md` §11.
    let vtx_alpha = mix(shallow.w, deep.w, depth);
    if (w.path.x > 1.5) {
        // ---- WMO INTERIOR (`0x6b6420`) — 134 of the game's 164 water groups, Blackfathom included.
        //
        // Fixed-function, always: the kernel body contains no `mov ecx,0x3f` at all (a positive
        // finding — the same scan on the exterior kernel finds two), and `[0xc9607c]`, the
        // specular/pixelShaders gate, is never read on this path. So it ignores those CVars, runs
        // with lighting OFF (`0x0e = 0`) and fog ON (`0x0f = 1`), and its whole output is the
        // combine preset `(0x1f, 3)` over one texture stage:
        //
        //     rgb = clamp(Cf + Ct)      alpha = clamp(Af + At)
        //
        // `Cf` is the pool's `MOMT[materialId].diffColor` taken RAW — baked into the mesh's vertex
        // colour, which is where the reference's own 6-float vertex carries it. NO sun term and no
        // sheen: that vertex has no normal to compute one from.
        //
        // The ALPHA op is the part worth stating. Preset 3 is `GL_ADD` on both channels via
        // `GL_COMBINE` (`COMBINE_ALPHA = GL_ADD` @`0x85c2fc`, operands left at the GL default), NOT
        // the legacy `GL_TEXTURE_ENV_MODE = GL_ADD` whose alpha would be `Af · At`. The client takes
        // the COMBINE path because `GL_ARB_texture_env_combine` is GL 1.3 core. So the ripple **adds**
        // to the pool's opacity instead of multiplying it away — a pool is at its authored opacity in
        // the troughs and saturates opaque on the crests, which is the opposite of what the legacy
        // reading would have drawn.
        let body = clamp(in.vcolor.rgb + detail.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
        return vec4<f32>(
            apply_fog(body, in.world_position.xyz, in.room_fog),
            clamp(vtx_alpha + detail.a, 0.0, 1.0),
        );
    }
    if (w.path.x > 0.5) {
        // ---- WMO EXTERIOR (`0x6b6630`) — Stormwind's canals and fountains.
        //
        // This arm DOES bind a pixel program, `Shaders\Pixel\MapObjExtWater0.bls` (bound at
        // `0x6b6654`, unbound `0x6b689c`), under the `[0xc9607c]` specular/pixelShaders gate. Both
        // CVars default to "0", but our reference install's `Config.wtf` sets both to "1", so the
        // shader leg is what every director comparison is against and it is the leg we implement.
        // Decoded verbatim from the asset:
        //
        //     rgb = primary.rgb + detail.rgb + secondary·detail.a      alpha = primary.a
        //
        // **No `+0.25`.** That constant is the ADT program's own `PARAM` and has no counterpart
        // here, so carrying it over — which is what we did — added a flat achromatic lift to every
        // canal pixel at all times, sun or no sun. Against the sheet's real texel distribution that
        // is ~3.5x the ripple contrast off the glint, and it is why the reference's canal shimmers
        // only where the sun is while ours sparkled everywhere.
        //
        // `primary` is FFP-lit over a constant up-normal with the vertex colour tracked into
        // ambient+diffuse (`glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)`), so the band
        // is the MATERIAL colour and `primary = band · clamp(ambient + diffuse·max(N·L, 0))`.
        // Lighting really is on here: neither this kernel nor its dispatch touches render-state id
        // `0x0e`, and the control that such a call would be findable is the interior kernel, which
        // does exactly that at `0x6b65bf`.
        //
        // The band is a SINGLE one — there is no bathymetry to lerp by — and it is the **deep** river
        // row, `LightIntBand` sub-17, i.e. `water_river[1]`. Read as a hard immediate
        // (`0x6b66be add edi, 0xec`), so nibbles 0, 4 and 8 all take it; exterior ocean cannot arise
        // (category 2 falls to a bare epilogue with no draw).
        //
        // **Sub-17, not sub-16, and the distinction cost a round trip.** A DayNight band *slot* is not
        // a `LightIntBand` *sub*: `0x6d64d0` displaces sub-8 out to record `+0x4c`, so `sub = slot + 1`
        // across slots 8–16, and the kernel's slot 16 is sub-17. The shipped data says the same thing
        // twice over — across all 367 LightParams rows carrying river bands, sub-16 is browns, olives
        // and muddy yellows (G > B in 71%: the shallow colour of water over a riverbed) while sub-17 is
        // blues and teals. At Stormwind sub-16 is `(79, 93, 20)`, which renders the canals olive-green;
        // sub-17 is `(51, 82, 85)`. The ADT ramp runs sub-16 → sub-17 across its 64 texels; WMO
        // exterior water takes the deep end alone, flat.
        let n_ext = normalize(in.world_normal);
        let to_light_ext = -normalize(wow_light.light_sun.xyz);
        let primary_ext = clamp(
            wow_light.light_ambient.rgb + wow_light.light_diffuse.rgb
                * max(dot(n_ext, to_light_ext), 0.0),
            vec3<f32>(0.0),
            vec3<f32>(1.0),
        ) * deep.rgb;
        let rgb_ext = primary_ext + detail.rgb + in.secondary_vtx * detail.a;
        // `result.color.w = fragment.color.primary` — the vertex alpha ALONE. The bound program
        // bypasses the texture environment entirely, so the interior arm's `+ At` does not apply here.
        return vec4<f32>(apply_fog(rgb_ext, in.world_position.xyz, in.room_fog), vtx_alpha);
    }

    // Body colour: lit vertex colour × the depth-lerped water-row swatch colour (`primary·colorTex`).
    let n = normalize(in.world_normal);
    let to_light = -normalize(wow_light.light_sun.xyz);
    let ndotl = max(dot(n, to_light), 0.0);
    let primary = clamp(
        wow_light.light_ambient.rgb + wow_light.light_diffuse.rgb * ndotl,
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    );

    // Sun sheen (`secondary`): the `ocean0_s.bls` Blinn highlight, computed PER-VERTEX in `fn vertex`
    // and interpolated across the coarse ~4 yd MCLQ mesh — the faithful 1.12 path (the real client
    // evaluates it in its FFP vertex stage). Per-pixel evaluation of the sharply-peaked `pow(N·H,6)`
    // would fill its broad lobe at full value (a brighter, denser sheen); interpolating from the
    // vertices flattens the peak to match the reference. (A per-pixel/per-vertex A/B toggle proved the
    // two visually identical on our mesh — we keep per-vertex as the faithful mechanism; RE:
    // `docs/knowledge/scratch/liquid-depth/fleck-deep.md`.)
    let secondary = in.secondary_vtx;

    // The stage-0 depth swatch, built and sampled as the reference does (see `swatch_row`): a
    // byte-space 64-row ramp that stops short of the deep endpoint, LINEAR across rows, with the
    // ocean's last row darkened `floor(0.9*byte)` and its alpha forced opaque.
    let swatch = swatch_at(shallow, deep, depth, w.kind.y > 0.5);

    // primary·colorTex.rgb  +  detail.rgb  +  (secondary + 0.25)·detail.a   (the ocean0_s.bls math)
    var rgb = primary * swatch.rgb + detail.rgb + (secondary + vec3<f32>(0.25)) * detail.a;

    // Opacity: depth ramp between the shallow/deep LightParams water alphas, over the SAME V as the
    // colour. Deeper = more opaque, up to α=1.0 where V saturates (river/lake byte 42 ≈ 5 yd), so the
    // channel middle is opaque + teal while the shore stays semi-transparent (V→0, α≈0.5) and the bottom
    // shows through (faithful — the pale edge band). One steep V drives both colour and opacity together.
    let alpha = swatch.w;

    // Distance fog (see `apply_fog`) — the water fog colour is also teal, so far water converges on the
    // haze.
    rgb = apply_fog(rgb, in.world_position.xyz, in.room_fog);

    // GAMMA LANE (0161): raw gamma out; alpha blends in gamma like the reference's bytes.
    return vec4<f32>(rgb, alpha);
}
