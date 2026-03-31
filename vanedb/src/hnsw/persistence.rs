use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use parking_lot::RwLock;
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

use super::{HnswIndex, Inner};
use crate::distance::{distance_fn, DistanceMetric};
use crate::error::{Result, VaneError};

#[derive(Serialize, Deserialize)]
struct HnswData {
    dim: usize,
    metric: u32,
    max_elements: usize,
    m: usize,
    m_max: usize,
    m_max0: usize,
    ef_construction: usize,
    ef_search: usize,
    mult: f64,
    count: usize,
    entry_point: Option<usize>,
    max_level: i32,
    vectors: Vec<f32>,
    ext_ids: Vec<u64>,
    levels: Vec<i32>,
    neighbors: Vec<Vec<Vec<usize>>>,
    id_map: HashMap<u64, usize>,
}

fn metric_to_u32(m: DistanceMetric) -> u32 {
    match m {
        DistanceMetric::L2 => 0,
        DistanceMetric::Cosine => 1,
        DistanceMetric::Dot => 2,
    }
}

fn u32_to_metric(v: u32) -> Result<DistanceMetric> {
    match v {
        0 => Ok(DistanceMetric::L2),
        1 => Ok(DistanceMetric::Cosine),
        2 => Ok(DistanceMetric::Dot),
        _ => Err(VaneError::Io("invalid metric in file".to_string())),
    }
}

impl HnswIndex {
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let inner = self.inner.read();
        let data = HnswData {
            dim: self.dim,
            metric: metric_to_u32(self.metric),
            max_elements: self.max_elements,
            m: self.m,
            m_max: self.m_max,
            m_max0: self.m_max0,
            ef_construction: self.ef_construction,
            ef_search: self.ef_search.load(Ordering::Relaxed),
            mult: self.mult,
            count: inner.count,
            entry_point: inner.entry_point,
            max_level: inner.max_level,
            vectors: inner.vectors.clone(),
            ext_ids: inner.ext_ids.clone(),
            levels: inner.levels.clone(),
            neighbors: inner.neighbors.clone(),
            id_map: inner.id_map.clone(),
        };

        let bytes =
            bincode::serialize(&data).map_err(|e| VaneError::Io(format!("serialize: {e}")))?;

        let path = path.as_ref();
        let tmp = path.with_extension("tmp");
        fs::write(&tmp, &bytes).map_err(|e| VaneError::Io(format!("write: {e}")))?;
        fs::rename(&tmp, path).map_err(|e| VaneError::Io(format!("rename: {e}")))?;
        Ok(())
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let bytes = fs::read(path.as_ref()).map_err(|e| VaneError::Io(format!("read: {e}")))?;
        let data: HnswData =
            bincode::deserialize(&bytes).map_err(|e| VaneError::Io(format!("deserialize: {e}")))?;

        let metric = u32_to_metric(data.metric)?;

        Ok(HnswIndex {
            dim: data.dim,
            metric,
            dist_fn: distance_fn(metric),
            max_elements: data.max_elements,
            m: data.m,
            m_max: data.m_max,
            m_max0: data.m_max0,
            ef_construction: data.ef_construction,
            ef_search: AtomicUsize::new(data.ef_search),
            mult: data.mult,
            inner: RwLock::new(Inner {
                vectors: data.vectors,
                ext_ids: data.ext_ids,
                id_map: data.id_map,
                levels: data.levels,
                neighbors: data.neighbors,
                entry_point: data.entry_point,
                max_level: data.max_level,
                count: data.count,
                rng: StdRng::seed_from_u64(data.count as u64),
            }),
        })
    }
}
