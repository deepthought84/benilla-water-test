// `$WOW_SSR_PIP` — the reflection march's raw output, drawn as an inset over the normal frame.
//
// The water writes what `ssr_trace` came back with into `liquid::ssr_pip`'s image, one texel per
// two screen pixels, before any Fresnel, tint or compositing — the same colour `$WOW_SSR_SHOW=2`
// paints, but beside the finished picture instead of in place of it, so a feature in the water can
// be traced to what the march actually returned for it. Black is a miss; magenta is water whose
// look runs no march at all; grey is not water.

#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

@group(0) @binding(0) var pip: texture_2d<f32>;
// `liquid::hiz`'s pyramid — shown instead of `pip` under `$WOW_SSR_PIP=2`.
@group(0) @binding(1) var hiz: texture_2d<f32>;

@fragment
fn fs_pip(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let dims = vec2<f32>(textureDimensions(pip));
    let texel = textureLoad(pip, vec2<i32>(min(in.uv * dims, dims - vec2<f32>(1.0))), 0);
    // A one-texel frame so the inset reads as an inset against any scene.
    let edge = min(min(in.uv.x, 1.0 - in.uv.x) * dims.x, min(in.uv.y, 1.0 - in.uv.y) * dims.y);
    if (edge < 2.0) {
        return vec4<f32>(1.0, 1.0, 1.0, 1.0);
    }
#ifdef PIP_DEPTH
    // `$WOW_SSR_PIP=2` — the pyramid's finest level: the depth the march actually traverses.
    // Brighter is nearer (log scale over reverse-Z); dark blue is NO depth at all, which the march
    // treats as open sky whatever the colour buffer shows there.
    let hd = vec2<f32>(textureDimensions(hiz, 0));
    let d = textureLoad(hiz, vec2<i32>(min(in.uv * hd, hd - vec2<f32>(1.0))), 0).r;
    if (d <= 0.0) {
        return vec4<f32>(0.02, 0.03, 0.2, 1.0);
    }
    let v = saturate(1.0 + log2(d) / 14.0);
    return vec4<f32>(v, v, v, 1.0);
#endif
    // Cleared to zero every frame, so alpha zero is "no water fragment wrote here".
    if (texel.a <= 0.0) {
        return vec4<f32>(0.12, 0.12, 0.12, 1.0);
    }
    return vec4<f32>(texel.rgb, 1.0);
}
