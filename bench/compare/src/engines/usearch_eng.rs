use std::path::Path;

use usearch::{Index, IndexOptions, MetricKind as UMetric, ScalarKind};

use super::{BuildParams, BuildStats, Engine, MetricKind};
use crate::measure::{current_rss_bytes, InstantTimer};

pub struct UsearchEngine {
    index: Option<Index>,
}

impl UsearchEngine {
    pub fn new() -> Self {
        Self { index: None }
    }
}

impl Default for UsearchEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl Engine for UsearchEngine {
    fn name(&self) -> &'static str {
        "usearch"
    }

    fn version(&self) -> String {
        usearch::version().to_string()
    }

    fn supports_delete(&self) -> bool {
        true
    }

    fn supports_save(&self) -> bool {
        true
    }

    fn build(
        &mut self,
        vectors: &[f32],
        ids: &[u64],
        dim: usize,
        metric: MetricKind,
        params: &BuildParams,
    ) -> Result<BuildStats, String> {
        let options = IndexOptions {
            dimensions: dim,
            metric: match metric {
                MetricKind::L2 => UMetric::L2sq,
                MetricKind::Cosine => UMetric::Cos,
            },
            quantization: ScalarKind::F32,
            connectivity: params.m,
            expansion_add: params.ef_construction,
            expansion_search: params.ef_search,
            multi: false,
        };

        let rss_before = current_rss_bytes();
        let timer = InstantTimer::start();
        let index = Index::new(&options).map_err(|e| e.to_string())?;
        index.reserve(ids.len()).map_err(|e| e.to_string())?;
        for (i, &id) in ids.iter().enumerate() {
            let row = &vectors[i * dim..(i + 1) * dim];
            index.add(id, row).map_err(|e| e.to_string())?;
        }
        let build_secs = timer.elapsed_secs();
        let rss_after = current_rss_bytes();
        self.index = Some(index);
        Ok(BuildStats {
            build_secs,
            peak_rss_bytes: rss_delta(rss_before, rss_after),
            notes: vec![format!(
                "connectivity(M)={} expansion_add(efC)={}",
                params.m, params.ef_construction
            )],
        })
    }

    fn search(&self, query: &[f32], k: usize, ef: usize) -> Result<Vec<u64>, String> {
        let index = self.index.as_ref().ok_or("usearch: not built")?;
        index.change_expansion_search(ef);
        let matches = index.search(query, k).map_err(|e| e.to_string())?;
        Ok(matches.keys)
    }

    fn save(&self, path: &Path) -> Result<u64, String> {
        let index = self.index.as_ref().ok_or("usearch: not built")?;
        let path_str = path.to_str().ok_or("usearch: non-utf8 path")?;
        index.save(path_str).map_err(|e| e.to_string())?;
        std::fs::metadata(path)
            .map(|m| m.len())
            .map_err(|e| e.to_string())
    }

    fn remove(&mut self, id: u64) -> Result<(), String> {
        let index = self.index.as_ref().ok_or("usearch: not built")?;
        index.remove(id).map_err(|e| e.to_string())?;
        Ok(())
    }
}

fn rss_delta(before: Option<u64>, after: Option<u64>) -> Option<u64> {
    match (before, after) {
        (Some(b), Some(a)) if a >= b => Some(a - b),
        (_, after) => after,
    }
}
