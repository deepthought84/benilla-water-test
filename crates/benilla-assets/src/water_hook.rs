//! The seam a water classification plugs into: a map's [`WaterClasses`], handed to every tile
//! that has water, and a WMO placement's, built from its pools. Without a registered classifier
//! there is none and the water is drawn as the reference draws it.

use std::sync::{Arc, OnceLock};

use benilla_formats::{Chain, LiquidMesh, WaterClasses};

/// The two classifiers a water plugin registers.
#[derive(Clone, Copy)]
pub struct WaterClassifier {
    /// Builds or reads the classification of a map (its `World\Maps` directory name) off the chain.
    pub map: fn(&Chain, &str) -> Option<Arc<dyn WaterClasses>>,
    /// Classifies one batch of liquid on its own: a WMO placement's pools.
    pub batch: fn(&[&LiquidMesh]) -> Option<Arc<dyn WaterClasses>>,
}

static CHAIN: OnceLock<Arc<Chain>> = OnceLock::new();
static CLASSIFIER: OnceLock<WaterClassifier> = OnceLock::new();

/// The patch chain the classifier reads tiles from; set with the `mpq://` source.
pub(crate) fn set_chain(chain: Arc<Chain>) {
    let _ = CHAIN.set(chain);
}

/// Register the classifier; the first registration holds.
pub fn set_water_classifier(f: WaterClassifier) {
    let _ = CLASSIFIER.set(f);
}

/// The classification of `map`, if a classifier is registered and the chain is open.
pub(crate) fn water_classes(map: &str) -> Option<Arc<dyn WaterClasses>> {
    (CLASSIFIER.get()?.map)(CHAIN.get()?, map)
}

/// The classification of one batch of liquid, if a classifier is registered.
pub fn classify_liquids(liquids: &[&LiquidMesh]) -> Option<Arc<dyn WaterClasses>> {
    (CLASSIFIER.get()?.batch)(liquids)
}
