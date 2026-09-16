//! hnsw_rs adapter.

use std::path::Path;

use hnsw_rs::prelude::*;

use super::{BuildParams, BuildStats, Engine, MetricKind};
use crate::measure::{current_rss_bytes, InstantTimer};

enum IndexKind {
    L2(Hnsw<'static, f32, DistL2>),
    Cosine(Hnsw<'static, f32, DistCosine>),
}

pub struct HnswRsEngine {
    index: Option<IndexKind>,
    /// Keep owned vector storage alive for the graph's lifetime.
    _storage: Vec<Vec<f32>>,
}

impl HnswRsEngine {
    pub fn new() -> Self {
        Self {
            index: None,
            _storage: Vec::new(),
        }
    }
}

impl Default for HnswRsEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl Engine for HnswRsEngine {
    fn name(&self) -> &'static str {
        "hnsw_rs"
    }

    fn version(&self) -> String {
        "0.3.4".into()
    }

    fn supports_delete(&self) -> bool {
        false
    }

    fn supports_save(&self) -> bool {
        // hnsw_rs can dump via hnswio; not wired here for a fair path yet.
        false
    }

    fn build(
        &mut self,
        vectors: &[f32],
        ids: &[u64],
        dim: usize,
        metric: MetricKind,
        params: &BuildParams,
    ) -> Result<BuildStats, String> {
        let n = ids.len();
        let max_layer = 16.min((n.max(2) as f32).ln().trunc() as usize);
        let mut storage: Vec<Vec<f32>> = Vec::with_capacity(n);
        for i in 0..n {
            storage.push(vectors[i * dim..(i + 1) * dim].to_vec());
        }

        let rss_before = current_rss_bytes();
        let timer = InstantTimer::start();
        let index = match metric {
            MetricKind::L2 => {
                let h = Hnsw::<f32, DistL2>::new(
                    params.m,
                    n.max(1),
                    max_layer,
                    params.ef_construction,
                    DistL2 {},
                );
                for (i, &id) in ids.iter().enumerate() {
                    h.insert((storage[i].as_slice(), id as usize));
                }
                IndexKind::L2(h)
            }
            MetricKind::Cosine => {
                let h = Hnsw::<f32, DistCosine>::new(
                    params.m,
                    n.max(1),
                    max_layer,
                    params.ef_construction,
                    DistCosine {},
                );
                for (i, &id) in ids.iter().enumerate() {
                    h.insert((storage[i].as_slice(), id as usize));
                }
                IndexKind::Cosine(h)
            }
        };
        let build_secs = timer.elapsed_secs();
        let rss_after = current_rss_bytes();
        self._storage = storage;
        self.index = Some(index);
        Ok(BuildStats {
            build_secs,
            peak_rss_bytes: rss_delta(rss_before, rss_after),
            notes: vec![format!(
                "max_nb_connection(M)={} efC={} max_layer={}",
                params.m, params.ef_construction, max_layer
            )],
        })
    }

    fn search(&self, query: &[f32], k: usize, ef: usize) -> Result<Vec<u64>, String> {
        let index = self.index.as_ref().ok_or("hnsw_rs: not built")?;
        let neighbours = match index {
            IndexKind::L2(h) => h.search(query, k, ef),
            IndexKind::Cosine(h) => h.search(query, k, ef),
        };
        Ok(neighbours
            .into_iter()
            .map(|n| n.get_origin_id() as u64)
            .collect())
    }

    fn save(&self, _path: &Path) -> Result<u64, String> {
        Err("hnsw_rs: save not wired in this harness".into())
    }

    fn remove(&mut self, _id: u64) -> Result<(), String> {
        Err("hnsw_rs: no delete API".into())
    }
}

fn rss_delta(before: Option<u64>, after: Option<u64>) -> Option<u64> {
    match (before, after) {
        (Some(b), Some(a)) if a >= b => Some(a - b),
        (_, after) => after,
    }
}
