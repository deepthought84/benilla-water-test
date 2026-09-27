//! **Improved Water**, benilla's own water look, as a plugin on the engine's liquid seam
//! (`benilla_world::liquid::hooks`): the whole-map classification of which water the planar
//! mirrors serve and where the fixed cube probes stand (`liquid_planar`, cached by `water_map`),
//! the two planar mirrors (`reflect`), the probes (`probe`), the depth pyramid and scene colour
//! the screen-space march reads (`depth`, `hiz`, `scene_color`) and the wave simulation
//! (`ripple_sim`). The surface shading is `liquid.wgsl`'s stylised lane, in the engine.
//!
//! Every pass is inert on the reference lane (`WaterStyle::Reference`); the plugin only makes
//! the Improved Water option draw what it names.

use std::path::PathBuf;

use bevy::prelude::*;

mod depth;
mod flow;
mod hiz;
mod liquid_planar;
mod probe;
mod reflect;
mod ripple_sim;
mod scene_color;
mod water_map;

pub use liquid_planar::PlanarMap;

// The engine items the moved passes name by their old paths.
use benilla_world::liquid::{
    WaterChunkInfo, WaterIndex, WaterMapRef, WaterPassImages, WaterPassSet, WaterStyle,
    UNMIRRORED_RENDER_LAYER,
};
use benilla_world::{dev_state, lighting, sky, sun, view, water_fx, world_unit};

/// The Improved Water plugin. Add it beside the engine's `WorldPlugins`; `cache_dir` is where each
/// map's classification is cached (the state folder), `None` to keep nothing.
pub struct ImprovedWaterPlugin {
    /// The classification cache's folder.
    pub cache_dir: Option<PathBuf>,
}

impl Plugin for ImprovedWaterPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(benilla_world::liquid::ImprovedWaterPresent);
        water_map::set_state_dir(self.cache_dir.clone());
        benilla_assets::set_water_classifier(benilla_assets::WaterClassifier {
            map: water_map::water_map,
            batch: liquid_planar::classify_batch,
        });
        bevy::asset::embedded_asset!(app, "shaders/probe_face.wgsl");
        bevy::asset::embedded_asset!(app, "shaders/probe_filter.wgsl");
        bevy::asset::embedded_asset!(app, "shaders/hiz_reduce.wgsl");
        // The mirrors' block and targets, then the wave simulation, the depth passes and the
        // probes, each inert unless the Improved Water look is selected.
        reflect::register(app);
        ripple_sim::register(app);
        hiz::register(app);
        scene_color::register(app);
        depth::register(app);
        probe::register(app);
        app.add_systems(
            Startup,
            publish_pass_images
                .in_set(WaterPassSet)
                .after(benilla_assets::AssetSet::Open),
        );
    }
}

/// Hand the engine's liquid materials this plugin's targets; each pass re-points them itself when
/// it rebuilds one.
fn publish_pass_images(
    mut commands: Commands,
    reflect: Option<Res<reflect::WaterReflect>>,
    reflect_buf: Option<Res<reflect::WaterReflectBuffer>>,
    scene_color: Option<Res<scene_color::WaterSceneColor>>,
    probe: Option<Res<probe::WaterProbe>>,
    hiz: Option<Res<hiz::WaterHiz>>,
    sim: Option<Res<ripple_sim::RippleSim>>,
) {
    let (Some(reflect), Some(reflect_buf), Some(scene_color), Some(probe), Some(hiz), Some(sim)) =
        (reflect, reflect_buf, scene_color, probe, hiz, sim)
    else {
        return;
    };
    commands.insert_resource(WaterPassImages {
        reflection: reflect.image.clone(),
        reflection2: reflect.image2.clone(),
        scene_color: scene_color.image.clone(),
        probe: probe.cube.clone(),
        hiz: hiz.image.clone(),
        hiz_far: hiz.far.clone(),
        wake: sim.image.clone(),
        reflect_buf: reflect_buf.0.clone(),
    });
}
