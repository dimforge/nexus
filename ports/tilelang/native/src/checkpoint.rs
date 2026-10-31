//! Vendor-neutral frame state. Kernels and restored words remain trusted.
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct FrameCheckpoint {
    pub schema: u32,
    pub steps: u64,
    pub capacities: BTreeMap<String, u32>,
    pub regions: BTreeMap<String, Vec<u32>>,
}
impl FrameCheckpoint {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema != 1
            || self.capacities.is_empty()
            || self.regions.is_empty()
            || self.regions.values().any(Vec::is_empty)
        {
            return Err("Invalid frame checkpoint".into());
        }
        Ok(())
    }
}
