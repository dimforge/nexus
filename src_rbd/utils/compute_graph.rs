//! Caching of captured compute graphs.

use khal::backend::GpuGraph;

/// What a frame should do with the work described by a key, as decided by
/// [`ComputeGraphCache::plan`].
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum ComputeGraphPlan {
    /// Run the work directly: the key changed since the previous frame (or
    /// its capture failed).
    Direct,
    /// Record the work into a new graph, then launch that graph.
    Capture,
    /// Launch the cached graph.
    Replay,
}

/// A cached compute graph of some recorded GPU work, keyed by a value
/// describing everything that shapes that work: buffer identities, dispatch
/// sizes, host-side loop counts, solver paths.
///
/// A graph is captured only once the same key has been seen on two
/// consecutive frames: one-time work (buffer growth, uploads of changed
/// counts) happens on the first frame with a new key and must not be
/// recorded. A failed capture is remembered and not retried until the key
/// changes.
pub struct ComputeGraphCache<K> {
    graph: Option<(K, GpuGraph)>,
    /// Key of the last [`Self::plan`] call.
    current: Option<K>,
    /// Key whose capture failed.
    failed: Option<K>,
}

impl<K> Default for ComputeGraphCache<K> {
    fn default() -> Self {
        Self {
            graph: None,
            current: None,
            failed: None,
        }
    }
}

impl<K: Clone + PartialEq> ComputeGraphCache<K> {
    /// Drops the cached graph and the key history, e.g. when graph replay is
    /// switched off.
    pub fn invalidate(&mut self) {
        self.graph = None;
        self.current = None;
    }

    /// Decides what this frame should do for the work described by `key`.
    pub fn plan(&mut self, key: K) -> ComputeGraphPlan {
        let plan = match &self.graph {
            Some((cached, _)) if *cached == key => ComputeGraphPlan::Replay,
            _ if self.current.as_ref() == Some(&key) && self.failed.as_ref() != Some(&key) => {
                ComputeGraphPlan::Capture
            }
            _ => ComputeGraphPlan::Direct,
        };
        if plan != ComputeGraphPlan::Replay {
            self.graph = None;
        }
        self.current = Some(key);
        plan
    }

    /// The cached graph, if [`Self::plan`] just returned
    /// [`ComputeGraphPlan::Replay`].
    pub fn graph(&self) -> Option<&GpuGraph> {
        self.graph.as_ref().map(|(_, graph)| graph)
    }

    /// Stores the graph captured for the key of the last [`Self::plan`] call.
    pub fn captured(&mut self, graph: GpuGraph) {
        let key = self.current.clone().unwrap_or_else(|| unreachable!());
        self.graph = Some((key, graph));
    }

    /// Records that capturing the work of the last [`Self::plan`] call failed,
    /// so it runs directly until its key changes.
    pub fn capture_failed(&mut self) {
        self.failed = self.current.clone();
    }
}
