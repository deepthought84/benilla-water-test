//! The render materials: the four `ExtendedMaterial`s the world is drawn with (terrain splat,
//! M2/WMO model, WDL far band, liquid surface) and the WGSL each one binds.
//!
//! The WGSL is embedded (`embedded://benilla_assets/shaders/…`), not served: a relative path would
//! resolve against the host binary's `AssetPlugin::file_path` and render nothing elsewhere.

use bevy::image::Image;
use bevy::mesh::MeshVertexBufferLayoutRef;
use bevy::pbr::{
    ExtendedMaterial, MaterialExtension, MaterialExtensionKey, MaterialExtensionPipeline,
    MeshPipelineKey,
};
use bevy::prelude::*;
use bevy::render::render_resource::{
    AsBindGroup, BlendComponent, BlendFactor, BlendOperation, BlendState, Buffer, ColorWrites,
    CompareFunction, RenderPipelineDescriptor, SpecializedMeshPipelineError,
};
use bevy::shader::ShaderRef;

/// Compile the four WGSL files into the binary under `embedded://benilla_assets/shaders/…`. Call
/// after Bevy's `AssetPlugin`, whose registry this fills; [`crate::register_asset_loaders`] does.
pub fn register_shaders(app: &mut App) {
    bevy::asset::embedded_asset!(app, "shaders/terrain.wgsl");
    bevy::asset::embedded_asset!(app, "shaders/wow_model.wgsl");
    bevy::asset::embedded_asset!(app, "shaders/wdl.wgsl");
    bevy::asset::embedded_asset!(app, "shaders/liquid.wgsl");
}

/// The WDL far-band shader's source, for the tests that live beside the renderer (`wdl.rs`).
pub const WDL_WGSL: &str = include_str!("shaders/wdl.wgsl");

/// Alpha-test reference for blend mode 1 (`Blend_AlphaKey`): 224 in the reference's
/// per-blend-mode table at `0x85ad20`, `{0, 224, 1, 1, 1, 1, 1, 0, 0, 0, 0}`. Must stay in sync
/// with `VANILLA_ALPHA_KEY` in `shaders/wow_model.wgsl`.
pub const VANILLA_ALPHA_KEY_REF: f32 = 224.0 / 255.0;

/// `StandardMaterial` plus the per-tile layer-blend extension.
pub type TerrainMaterial = ExtendedMaterial<StandardMaterial, TerrainExtension>;

/// World models (doodads, WMOs, creatures, GameObjects) lit in gamma space like terrain, not by
/// PBR; `StandardMaterial` carries only the texture, alpha mode and culling.
pub type WowModelMaterial = ExtendedMaterial<StandardMaterial, WowModelExt>;

/// Pipeline-specialization key for [`WowModelExt`]: the depth and blend states Bevy's `AlphaMode`
/// cannot express. `fade` (`model_flags.y`) is the M2 doodad fade twin and `clutter`
/// (`clutter_fade.w`) ground clutter; both fade opacity with depth-write on, as the reference does.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct WowModelKey {
    fade: bool,
    clutter: bool,
    /// Additive glow card (`clutter_fade.z` bit 2).
    additive: bool,
    /// M2 render flag 0x10, no depth write (`clutter_fade.z` bit 0).
    no_depth_write: bool,
    /// M2 render flag 0x08, no depth test (`clutter_fade.z` bit 1).
    no_depth_test: bool,
    /// The multiply blends, Mod and Mod2x (`clutter_fade.z` bits 7/8).
    modulate: bool,
    modulate2x: bool,
    /// The depth-prime twin (`clutter_fade.z` bit 9, `model_render::zfill_material`).
    zfill: bool,
    /// Far side of the water plane (`clutter_fade.z` bit 11).
    far_side: bool,
    /// The WMO-skybox lane (`clutter_fade.z` bit 13, `model_render::SKY_DEPTH_MARKER`), a key axis
    /// because only this model's depth is pinned.
    sky_depth: bool,
    // The WMO batch order is deliberately not a key axis (a pipeline per batch index stalls a
    // city's first sight): the file-order layering (`0x6b4f10`/`0x6b5190`) rides `sun_scale.y`.
}

impl From<&WowModelExt> for WowModelKey {
    fn from(e: &WowModelExt) -> Self {
        let markers = e.clutter_fade.z as u32;
        Self {
            fade: e.model_flags.y > 0.5,
            clutter: e.clutter_fade.w > 0.5,
            additive: markers & 4 != 0,
            no_depth_write: markers & 1 != 0,
            no_depth_test: markers & 2 != 0,
            modulate: markers & 0x80 != 0,
            modulate2x: markers & 0x100 != 0,
            zfill: markers & 0x200 != 0,
            far_side: markers & 0x800 != 0,
            sky_depth: markers & 0x2000 != 0,
        }
    }
}

/// The model material's extension. Every `Vec4` uniform packs into binding 100, one buffer entry,
/// to stay under Metal's 16-buffer vertex-stage cap (Bevy 0.18 ignores `visibility` on uniforms);
/// `ModelParams` in `wow_model.wgsl` must match the field order.
#[derive(Asset, AsBindGroup, Clone, TypePath)]
#[bind_group_data(WowModelKey)]
pub struct WowModelExt {
    /// Clutter fade: `x`, `y` = full-opacity and fully-faded view depth (yd; the ~70 yd horizon),
    /// `w` = clutter, all zero off clutter; `z` = the batch marker bits.
    #[uniform(100)]
    pub clutter_fade: Vec4,
    /// `x` = WMO, `y` = the fade blend twin, `z` = interior, `w` = unlit fullbright.
    #[uniform(100)]
    pub model_flags: Vec4,
    /// `x` = the terrain-shade selector, `y` = the WMO batch order (the clip-z layering nudge),
    /// `zw` = the UV-animation offset.
    #[uniform(100)]
    pub sun_scale: Vec4,
    /// `xyz` = the M2Color tint of a batch whose colour track animates, identity otherwise; `w` =
    /// the WMO interior batch class (`0x6b5190`): `0` exterior, `1` INT, `2` TRANS.
    #[uniform(100)]
    pub tint: Vec4,
    /// WMO glass, `0` on M2: `xyz` = the MOMT SIDN night emissive (ramp `0x6b4090`), `w` = the MOMT
    /// WINDOW flag, the brighter interior light (`0x6d37e0`, ambient +16/255).
    #[uniform(100)]
    pub sidn: Vec4,
    /// Rows of the shared light buffer's `matanim` region, `0` = identity: `x` = UV scroll, `y` =
    /// tint, `z` = the texture-transform affine, `w` = the UI tile's cell clip.
    #[uniform(100)]
    pub anim_slots: Vec4,
    /// The shared global light (`lighting::global_light`), updated in place once a frame; the
    /// vertex stage reads its point-light table, since Bevy's clusterable lights are fragment-only.
    #[storage(90, read_only, buffer, visibility(vertex, fragment))]
    pub light_buf: Buffer,
}

impl MaterialExtension for WowModelExt {
    /// Out of the depth prepass: `specialize` rebuilds the vertex layout around attributes Bevy's
    /// prepass shader does not declare.
    fn enable_prepass() -> bool {
        false
    }

    /// Bevy's mesh vertex plus the point-light term, per vertex like the reference's FFP.
    fn vertex_shader() -> ShaderRef {
        "embedded://benilla_assets/shaders/wow_model.wgsl".into()
    }

    fn fragment_shader() -> ShaderRef {
        "embedded://benilla_assets/shaders/wow_model.wgsl".into()
    }

    /// The reference writes depth for every M2 batch, transparent ones too, and tests `LEQUAL`,
    /// unless render flag 0x10 (no write) or 0x08 (no test) clears it (`0x70c190`).
    fn specialize(
        _pipeline: &MaterialExtensionPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        layout: &MeshVertexBufferLayoutRef,
        key: MaterialExtensionKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        // `WOW_RIG_SKIN` skins from the shared buffer's palette. The base layout lacks our joint
        // attributes: rebuild it with Bevy's conditionals (mesh.rs, locations 0-5) plus 10/11.
        if layout.0.contains(crate::ATTRIBUTE_WOW_JOINT_INDEX) {
            descriptor.vertex.shader_defs.push("WOW_RIG_SKIN".into());
            let mut attrs = Vec::with_capacity(7);
            for (attr, loc) in [
                (Mesh::ATTRIBUTE_POSITION, 0),
                (Mesh::ATTRIBUTE_NORMAL, 1),
                (Mesh::ATTRIBUTE_UV_0, 2),
                (Mesh::ATTRIBUTE_UV_1, 3),
                (Mesh::ATTRIBUTE_TANGENT, 4),
                (Mesh::ATTRIBUTE_COLOR, 5),
            ] {
                if layout.0.contains(attr) {
                    attrs.push(attr.at_shader_location(loc));
                }
            }
            attrs.push(crate::ATTRIBUTE_WOW_JOINT_INDEX.at_shader_location(10));
            attrs.push(crate::ATTRIBUTE_WOW_JOINT_WEIGHT.at_shader_location(11));
            descriptor.vertex.buffers = vec![layout.0.get_layout(&attrs)?];
        }
        // A merged blob computes the doodad fade per vertex from its fade sphere; same rebuild,
        // and merged blobs never skin.
        if layout.0.contains(crate::ATTRIBUTE_WOW_FADE_SPHERE) {
            descriptor.vertex.shader_defs.push("WOW_MERGED_FADE".into());
            if let Some(fragment) = descriptor.fragment.as_mut() {
                fragment.shader_defs.push("WOW_MERGED_FADE".into());
            }
            let mut attrs = Vec::with_capacity(6);
            for (attr, loc) in [
                (Mesh::ATTRIBUTE_POSITION, 0),
                (Mesh::ATTRIBUTE_NORMAL, 1),
                (Mesh::ATTRIBUTE_UV_0, 2),
                (Mesh::ATTRIBUTE_UV_1, 3),
                (Mesh::ATTRIBUTE_TANGENT, 4),
                (Mesh::ATTRIBUTE_COLOR, 5),
            ] {
                if layout.0.contains(attr) {
                    attrs.push(attr.at_shader_location(loc));
                }
            }
            attrs.push(crate::ATTRIBUTE_WOW_FADE_SPHERE.at_shader_location(12));
            // An interior-prop blob's baked SH-probe slot replaces the per-entity MeshTag payload.
            if layout.0.contains(crate::ATTRIBUTE_WOW_MERGED_SLOT) {
                descriptor.vertex.shader_defs.push("WOW_MERGED_SLOT".into());
                if let Some(fragment) = descriptor.fragment.as_mut() {
                    fragment.shader_defs.push("WOW_MERGED_SLOT".into());
                }
                attrs.push(crate::ATTRIBUTE_WOW_MERGED_SLOT.at_shader_location(13));
            }
            descriptor.vertex.buffers = vec![layout.0.get_layout(&attrs)?];
        }
        if let Some(ds) = descriptor.depth_stencil.as_mut() {
            ds.depth_write_enabled = !key.bind_group_data.no_depth_write;
            if key.bind_group_data.no_depth_test {
                ds.depth_compare = CompareFunction::Always;
            }
        }
        // A sort bias is not a raster bias: the base `StandardMaterial::specialize` also packs it
        // into `depth_stencil.bias.constant`, where the far side's −4e4 would clip coplanar layers.
        if key.bind_group_data.far_side {
            if let Some(ds) = descriptor.depth_stencil.as_mut() {
                ds.bias.constant = 0;
            }
        }
        // The skybox pins clip z to 0, reverse-Z far, in the vertex stage; its bias is sort-only.
        if key.bind_group_data.sky_depth {
            descriptor.vertex.shader_defs.push("WOW_SKY_DEPTH".into());
            if let Some(fragment) = descriptor.fragment.as_mut() {
                fragment.shader_defs.push("WOW_SKY_DEPTH".into());
            }
            if let Some(ds) = descriptor.depth_stencil.as_mut() {
                ds.bias.constant = 0;
            }
        }
        if key.bind_group_data.fade && !key.bind_group_data.sky_depth {
            // A fading model's near cards occlude its far ones; the sky never writes depth.
            if let Some(ds) = descriptor.depth_stencil.as_mut() {
                ds.depth_write_enabled = true;
            }
        }
        if key.bind_group_data.clutter {
            // Clutter blends in the alpha-mask pass, as the reference does, so the ~70 yd ramp
            // fades opacity; the shader keeps the 128/255 discard.
            if let Some(target) = descriptor
                .fragment
                .as_mut()
                .and_then(|f| f.targets.get_mut(0))
                .and_then(|t| t.as_mut())
            {
                target.blend = Some(BlendState::ALPHA_BLENDING);
            }
        }
        if key.bind_group_data.additive {
            // A pure (ONE, ONE) add: the shader weights by alpha in gamma space, where a `SrcAlpha`
            // factor or `AlphaMode::Add` would weight after linearising and fatten soft edges.
            if let Some(target) = descriptor
                .fragment
                .as_mut()
                .and_then(|f| f.targets.get_mut(0))
                .and_then(|t| t.as_mut())
            {
                target.blend = Some(BlendState {
                    color: BlendComponent {
                        src_factor: BlendFactor::One,
                        dst_factor: BlendFactor::One,
                        operation: BlendOperation::Add,
                    },
                    alpha: BlendComponent {
                        src_factor: BlendFactor::Zero,
                        dst_factor: BlendFactor::One,
                        operation: BlendOperation::Add,
                    },
                });
            }
        }
        if key.bind_group_data.zfill {
            // The reference's `M2UseZFill` pre-pass (`0x707f7d`-`0x708072`): depth only, so a
            // translucent model shows one blended layer. The colour is computed and masked, not
            // skipped by an early return, which naga's MSL backend miscompiles.
            if let Some(frag) = descriptor.fragment.as_mut() {
                if let Some(target) = frag.targets.get_mut(0).and_then(|t| t.as_mut()) {
                    target.blend = None;
                    target.write_mask = ColorWrites::empty();
                }
            }
            if let Some(ds) = descriptor.depth_stencil.as_mut() {
                ds.depth_write_enabled = true;
            }
        }
        // The waterline clip (`benilla_world::straddle`) keys on the blend pass: no new pipelines.
        if key
            .mesh_key
            .intersection(MeshPipelineKey::BLEND_RESERVED_BITS)
            == MeshPipelineKey::BLEND_ALPHA
        {
            if let Some(fragment) = descriptor.fragment.as_mut() {
                fragment.shader_defs.push("WOW_WATER_CLIP".into());
            }
        }
        let key = &key.bind_group_data;
        if key.modulate || key.modulate2x {
            // `0x70c190`: Mod (M2 mode 5, WMO 4) is `DST_COLOR/ZERO`, Mod2x (M2 6, WMO 5)
            // `DST_COLOR/SRC_COLOR`; on a gamma framebuffer this is the reference's byte math.
            if let Some(target) = descriptor
                .fragment
                .as_mut()
                .and_then(|f| f.targets.get_mut(0))
                .and_then(|t| t.as_mut())
            {
                target.blend = Some(BlendState {
                    color: BlendComponent {
                        src_factor: BlendFactor::Dst,
                        dst_factor: if key.modulate2x {
                            BlendFactor::Src
                        } else {
                            BlendFactor::Zero
                        },
                        operation: BlendOperation::Add,
                    },
                    alpha: BlendComponent {
                        src_factor: BlendFactor::Zero,
                        dst_factor: BlendFactor::One,
                        operation: BlendOperation::Add,
                    },
                });
            }
        }
        Ok(())
    }
}

/// Distant low-detail terrain (WDL), the horizon hills beyond the streamed tiles: opaque white
/// geometry under the scene fog colour at the hull pass's own start 0 and end 1.0.
pub type WdlMaterial = ExtendedMaterial<StandardMaterial, WdlExt>;

/// WDL has no per-material input: its whole colour is the scene fog, off the shared global light.
#[derive(Asset, AsBindGroup, Clone, TypePath)]
pub struct WdlExt {
    /// The shared global light, rows 4/5 only (scene fog colour, farclip wall).
    #[storage(90, read_only, buffer)]
    pub light_buf: Buffer,
}

impl MaterialExtension for WdlExt {
    /// **Out of the depth prepass**, for the reason the model lane is out of it: a pass that cannot
    /// reproduce this material's discards writes depth for fragments the main pass throws away.
    ///
    /// The WDL hull is the coarse horizon, and its fragment stage cuts everything NEARER than
    /// `farclip − 33` so the fine terrain owns the near field. Bevy's stock prepass has no such cut,
    /// so an armed prepass wrote the whole hull's depth — including the near part that never draws.
    /// The coarse surface sits above the fine one wherever the ground is flat, which is exactly the
    /// river valleys and lake beds, so the detailed terrain there failed `GreaterEqual` against a
    /// horizon that was not drawn either, and the sky came through the hole. That is the
    /// "terrain vanishes wherever a river or lake runs over it" this whole prepass has been off for.
    ///
    /// Opting out costs the water nothing: it measures its column against the real terrain, and the
    /// coarse hull only exists past the far-clip wall where there is no water to measure.
    fn enable_prepass() -> bool {
        false
    }

    fn vertex_shader() -> ShaderRef {
        "embedded://benilla_assets/shaders/wdl.wgsl".into()
    }
    fn fragment_shader() -> ShaderRef {
        "embedded://benilla_assets/shaders/wdl.wgsl".into()
    }
}

/// Liquid surfaces, one `liquid.wgsl` arm per reference liquid renderer: ADT MCLQ (the
/// `ocean0_s.bls` combine), WMO exterior and interior water, and magma and slime.
pub type LiquidMaterial = ExtendedMaterial<StandardMaterial, LiquidExt>;

/// Liquid inputs: the animated frames and the per-material lanes; all light comes off the shared
/// global light. The `Vec4` uniforms pack into binding 102, in the WGSL struct's field order.
#[derive(Asset, AsBindGroup, Clone, TypePath)]
pub struct LiquidExt {
    /// The kind's animated frames (`lake_a`, `fast_a`, `ocean_h`): RGB near-black, alpha ripple.
    #[texture(100, dimension = "2d_array", visibility(fragment))]
    #[sampler(101, visibility(fragment))]
    pub frames: Handle<Image>,
    /// The stylised look's generated ripple map (`benilla_world::liquid::ripple`): R/G a tiling
    /// slope field, B its height.
    #[texture(103, visibility(fragment))]
    #[sampler(104, visibility(fragment))]
    pub ripples: Handle<Image>,
    /// The first planar mirror's capture (`benilla_world::liquid::reflect`), alpha as coverage.
    #[texture(105, visibility(fragment))]
    #[sampler(106, visibility(fragment))]
    pub reflection: Handle<Image>,
    /// The second planar mirror's capture, for a second water height in view; read through the
    /// first one's sampler.
    #[texture(110, visibility(fragment))]
    pub reflection2: Handle<Image>,
    /// The stylised look's live wave field (`benilla_world::liquid::ripple_sim`): R/G slope, B foam.
    #[texture(108, visibility(fragment))]
    #[sampler(109, visibility(fragment))]
    pub wake: Handle<Image>,
    /// The opaque scene before transparents (`benilla_world::liquid::scene_color`), which the
    /// screen-space march reads its hits from.
    #[texture(111, visibility(fragment))]
    #[sampler(112, visibility(fragment))]
    pub scene_color: Handle<Image>,
    /// The cubemap reflection probes (`benilla_world::liquid::probe`), one cube per array slice.
    #[texture(113, dimension = "cube_array", visibility(fragment))]
    #[sampler(114, visibility(fragment))]
    pub probe: Handle<Image>,
    /// The march's nearest-depth pyramid (`benilla_world::liquid::hiz`), read by `textureLoad`.
    #[texture(115, sample_type = "float", filterable = false, visibility(fragment))]
    pub hiz: Handle<Image>,
    /// The farthest-depth twin of [`Self::hiz`], which lets the march pass behind a tile.
    #[texture(117, sample_type = "float", filterable = false, visibility(fragment))]
    pub hiz_far: Handle<Image>,
    /// `$WOW_SSR_PIP`'s image (`benilla_world::liquid::ssr_pip`); 1x1 while it is off.
    #[storage_texture(116, image_format = Rgba16Float, access = WriteOnly, visibility(fragment))]
    pub ssr_pip: Handle<Image>,
    /// The reflection's per-frame parameters (`benilla_world::liquid::reflect`), a buffer so the
    /// per-frame write does not rebuild every liquid material's bind group.
    #[storage(107, read_only, buffer, visibility(fragment))]
    pub reflect_buf: Buffer,
    /// Which lanes of the shared light this surface reads:
    /// - `x` = fullbright (magma, slime): the sheet is the opaque body.
    /// - `y` = ocean: rows 15/16 (`Light.dbc` IntBand 14/15), else rows 13/14 (IntBand 16/17).
    /// - `z` = a WMO interior group's liquid, fogged with the interior block (rows 18/19): the
    ///   reference gates its fog (`0x6b6323`-`0x6b6342`) on the same `[0xca7f00]` as the group's
    ///   geometry (`0x6b51d9`/`0x6b51ea`).
    /// - `w` = the sun-sheen shininess (`lighting::WATER_SHININESS`).
    #[uniform(102)]
    pub kind: Vec4,
    /// `x` = the renderer (`liquid::surface::LiquidPath`): `0` ADT MCLQ, `1` WMO exterior, `2` WMO
    /// interior.
    /// `y` = the stylised look (`benilla_world::liquid::WaterStyle`), rewritten on toggle. `z` =
    /// magma rather than slime, read by the stylised lane only.
    #[uniform(102)]
    pub path: Vec4,
    /// `y` = frame count, `z` = scroll flag, `w` = clock enable.
    #[uniform(102)]
    pub anim: Vec4,
    /// The shared global light, read in both stages (the vertex stage evaluates the sun sheen).
    #[storage(90, read_only, buffer, visibility(vertex, fragment))]
    pub light_buf: Buffer,
}

/// A liquid vertex's **offset to the nearest point on the waterline**, in world XZ yards
/// (`benilla_world::liquid::surface`).
///
/// It replaces carrying the shore DISTANCE and interpolating that, and the difference is the whole
/// point. Distance to a curve is not a linear function — it has a V-shaped crease along the curve
/// itself — so interpolating it across a triangle cuts that crease's corner, and where the crease
/// falls inside a triangle the reconstructed zero lands in the wrong place by up to half a vertex
/// spacing. With a foam band a fifth of a yard wide drawn from vertices half a yard apart, that is
/// most of a band, and it is why the line jogged, thinned and doubled as the camera moved along it.
///
/// The offset has no crease: `nearest − position` is smooth wherever the nearest point is, and both
/// terms interpolate linearly, so the interpolated offset is exactly the offset of the interpolated
/// point. Taking its LENGTH in the fragment shader puts the non-linearity where it belongs, after
/// interpolation, and the band comes out the same width everywhere at any mesh density.
///
/// Own attribute id and shader location 10, for the same reason the rig's attributes have theirs —
/// Bevy's builtin set ends at 7 and a mesh must not accidentally look like one of its lanes.
pub const ATTRIBUTE_WOW_SHORE_OFFSET: bevy::mesh::MeshVertexAttribute =
    bevy::mesh::MeshVertexAttribute::new(
        "Wow_ShoreOffset",
        988_540_921,
        bevy::render::render_resource::VertexFormat::Float32x2,
    );

/// The water surface's **smooth normal**, mesh-local: each lattice corner's normal from its
/// neighbours' heights, blended bilinearly to the vertex — so the reflection direction bends
/// smoothly across a sloping stream instead of stepping at every triangle edge. Read by the
/// stylised lane only; `ATTRIBUTE_NORMAL` stays flat for the reference lane's lighting.
///
/// Own attribute id and shader location 11, beside [`ATTRIBUTE_WOW_SHORE_OFFSET`].
pub const ATTRIBUTE_WOW_SURFACE_NORMAL: bevy::mesh::MeshVertexAttribute =
    bevy::mesh::MeshVertexAttribute::new(
        "Wow_SurfaceNormal",
        988_540_922,
        bevy::render::render_resource::VertexFormat::Float32x3,
    );

/// How far a planar mirror may serve this water, 0 to 1 per vertex: 0 on falls, descending rivers
/// and small pools (`benilla_world::liquid::planar`), ramping to 1 over a few cells, so the mirror
/// fades into the cube probe beneath it. Read by the stylised lane only; 1 on a mesh without it.
///
/// Own attribute id and shader location 12, beside [`ATTRIBUTE_WOW_SURFACE_NORMAL`].
pub const ATTRIBUTE_WOW_PLANAR: bevy::mesh::MeshVertexAttribute =
    bevy::mesh::MeshVertexAttribute::new(
        "Wow_Planar",
        988_540_923,
        bevy::render::render_resource::VertexFormat::Float32,
    );

impl MaterialExtension for LiquidExt {
    fn vertex_shader() -> ShaderRef {
        "embedded://benilla_assets/shaders/liquid.wgsl".into()
    }
    fn fragment_shader() -> ShaderRef {
        "embedded://benilla_assets/shaders/liquid.wgsl".into()
    }

    /// Admits [`ATTRIBUTE_WOW_SHORE_OFFSET`], [`ATTRIBUTE_WOW_SURFACE_NORMAL`] and
    /// [`ATTRIBUTE_WOW_PLANAR`] into the vertex
    /// layout, each behind its shader def, since Bevy builds the layout from its own attributes only.
    ///
    /// `sky_order::WATER_BIAS` (−2e4) is a sort rung, kept out of the rasterizer: as a depth-bias
    /// constant it would move the waterline, by an amount that doubles at every float exponent
    /// boundary, so neighbouring triangles' shorelines would disagree.
    fn specialize(
        _pipeline: &MaterialExtensionPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        layout: &MeshVertexBufferLayoutRef,
        _key: MaterialExtensionKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        // Clear the depth bias that StandardMaterial would have added
        if let Some(ds) = descriptor.depth_stencil.as_mut() {
            ds.bias.constant = 0;
        }

        // Our own attributes, where the mesh carries them, each behind its shader def.
        let ours = [
            (ATTRIBUTE_WOW_SHORE_OFFSET, 10, "LIQUID_SHORE_OFFSET"),
            (ATTRIBUTE_WOW_SURFACE_NORMAL, 11, "LIQUID_SURFACE_NORMAL"),
            (ATTRIBUTE_WOW_PLANAR, 12, "LIQUID_PLANAR"),
        ];
        if !ours.iter().any(|(a, _, _)| layout.0.contains(*a)) {
            return Ok(());
        }
        let mut attrs = Vec::with_capacity(8);
        for (attr, loc) in [
            (Mesh::ATTRIBUTE_POSITION, 0),
            (Mesh::ATTRIBUTE_NORMAL, 1),
            (Mesh::ATTRIBUTE_UV_0, 2),
            (Mesh::ATTRIBUTE_UV_1, 3),
            (Mesh::ATTRIBUTE_TANGENT, 4),
            (Mesh::ATTRIBUTE_COLOR, 5),
        ] {
            if layout.0.contains(attr) {
                attrs.push(attr.at_shader_location(loc));
            }
        }
        for (attr, loc, def) in ours {
            if !layout.0.contains(attr) {
                continue;
            }
            attrs.push(attr.at_shader_location(loc));
            descriptor.vertex.shader_defs.push(def.into());
            if let Some(f) = descriptor.fragment.as_mut() {
                f.shader_defs.push(def.into());
            }
        }
        descriptor.vertex.buffers = vec![layout.0.get_layout(&attrs)?];
        Ok(())
    }
}

/// Per-tile terrain splat, one material per ADT tile: each merged-mesh vertex carries its chunk's
/// four layer indices (`COLOR`) and alpha layer (`UV1.x`), so a tile is one draw.
///
/// One repeating sampler (105) serves every array, to stay under Metal's 16 fragment samplers;
/// `terrain.wgsl` insets the alpha UVs by half a texel so they do not wrap at a chunk edge.
#[derive(Asset, AsBindGroup, Clone, TypePath)]
pub struct TerrainExtension {
    #[texture(100, dimension = "2d_array", visibility(fragment))]
    #[sampler(105, visibility(fragment))]
    pub layer_array: Handle<Image>,
    #[texture(104, dimension = "2d_array", visibility(fragment))]
    pub alpha_array: Handle<Image>,
    /// Per-chunk MCSH shadow maps (`R` = shadowed); `UV1.y` is the layer, `-1` for none.
    #[texture(110, dimension = "2d_array", visibility(fragment))]
    pub shadow_array: Handle<Image>,

    // Binding 106's uniforms, in the field order of the WGSL `TerrainParams`.
    /// `x` = texture tiling factor (repeats per chunk); other lanes unused.
    #[uniform(106)]
    pub params: Vec4,

    /// The shared global light, rows 0-5 (light, fog, farclip).
    #[storage(90, read_only, buffer)]
    pub light_buf: Buffer,
}

impl MaterialExtension for TerrainExtension {
    // The sun specular is per vertex (the reference's light flush `0x59c820`).
    fn vertex_shader() -> ShaderRef {
        "embedded://benilla_assets/shaders/terrain.wgsl".into()
    }
    fn fragment_shader() -> ShaderRef {
        "embedded://benilla_assets/shaders/terrain.wgsl".into()
    }
}

#[cfg(test)]
mod tests {
    /// The sky depth law for the WMO skybox, the one sky element on the model lane.
    #[test]
    fn the_sky_lane_pins_the_far_depth_at_the_vertex() {
        let src = include_str!("shaders/wow_model.wgsl");
        // The pin, matched behind its ifdef: ungated, every model draw sits at the far plane.
        let pin = src
            .find("#ifdef WOW_SKY_DEPTH")
            .expect("the sky-depth branch is gone");
        let branch = &src[pin..src[pin..].find("#endif").map_or(src.len(), |e| pin + e)];
        assert!(
            branch.contains("out.position.z = 0.0;"),
            "the model lane's sky branch no longer pins the far depth at the vertex — a WMO \
             skybox's shell radius is deciding occlusion again (benilla_world::sky_order, \"The \
             depth law\")"
        );
        assert!(
            !src.contains("@builtin(frag_depth)"),
            "the model lane writes a fragment depth again — the sky pin is the vertex stage's, \
             and a fragment write costs every draw on this lane its early-Z"
        );
    }

    /// The WGSL carries the reference's swatch row arithmetic, and a Rust mirror of it reproduces
    /// the row fill `0x68a830` on `Light.dbc` id 4, map 0, t = 1440, ocean `LightIntBand` 14 to 15.
    #[test]
    fn liquid_swatch_reproduces_the_reference_row_ramp() {
        let src = include_str!("shaders/liquid.wgsl");
        // The row accumulator `c0 + floor(i*(c1 - c0)/64)`, not a lerp to the deep endpoint.
        assert!(
            src.contains("let row = c0 + floor(i * (c1 - c0) / 64.0);"),
            "the swatch stopped building its rows the way `0x68a830` does: a plain lerp to the \
             deep endpoint runs a 64th of a ramp that does not exist"
        );
        // The ocean-only tail: floor(0.9*byte) on the last row, alpha forced opaque.
        assert!(
            src.contains("if ocean && i >= 63.0 {")
                && src.contains("vec4<f32>(floor(row.rgb * 0.9), 255.0)"),
            "the ocean's last-row darkening is gone — ~80% of the world's ocean vertices sample \
             that row, so this is the open sea's colour"
        );
        // Linear sampling across the two rows V falls between, at texel `V*64 - 0.5`.
        assert!(
            src.contains("let t = clamp(v * 64.0 - 0.5, 0.0, 63.0);"),
            "the swatch stopped sampling as an 8x64 LINEAR/CLAMP texture — the ocean darkening \
             would step instead of ramping across the final 1/64 of V"
        );

        /// `row(i)` for one channel, `swatch_row`'s byte arithmetic.
        fn row(c0: i32, c1: i32, i: i32) -> i32 {
            c0 + (i * (c1 - c0)).div_euclid(64)
        }
        /// The ocean tail's colour half.
        fn tail(b: i32) -> i32 {
            (f64::from(b) * 0.9).floor() as i32
        }

        // Ocean shallow (sub-14) 0x00457b63 and deep (sub-15) 0x000b2228, as R/G/B.
        let (shallow, deep) = ([69, 123, 99], [11, 34, 40]);
        let at = |i| [0, 1, 2].map(|k| row(shallow[k], deep[k], i));

        // Row 63 is one short of the endpoint on G: the ramp never arrives.
        assert_eq!(at(63), [11, 35, 40], "row 63 before the ocean tail");
        assert_ne!(
            at(63)[1],
            deep[1],
            "the ramp must NOT reach the deep endpoint"
        );
        assert_eq!(
            at(62),
            [12, 36, 41],
            "row 62, the other half of the last blend"
        );
        assert_eq!(
            at(63).map(tail),
            [9, 31, 36],
            "row 63 after the ocean tail (HSV V *= 0.9, which is floor(0.9*byte) per channel)"
        );
        // The tail and the shortfall are deep-end only.
        assert_eq!(at(0), shallow, "row 0 is the shallow endpoint verbatim");
    }
}
