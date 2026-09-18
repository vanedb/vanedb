use std::path::Path;

use vanedb::{ApproxIndex, Metric, SearchParams};

use super::{BuildParams, BuildStats, Engine, MetricKind};
use crate::measure::{current_rss_bytes, InstantTimer};

pub struct VanedbEngine {
    index: Option<ApproxIndex>,
}

impl VanedbEngine {
    pub fn new() -> Self {
        Self { index: None }
    }
}

impl Default for VanedbEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl Engine for VanedbEngine {
    fn name(&self) -> &'static str {
        "vanedb"
    }

    fn version(&self) -> String {
        option_env!("VANEDB_VERSION").unwrap_or("path").to_string()
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
        let metric = match metric {
            MetricKind::L2 => Metric::L2,
            MetricKind::Cosine => Metric::Cosine,
        };
        let rss_before = current_rss_bytes();
        let timer = InstantTimer::start();
        let index = ApproxIndex::builder(dim, metric)
            .m(params.m)
            .ef_construction(params.ef_construction)
            .capacity(ids.len())
            .seed(params.seed)
            .build()
            .map_err(|e| e.to_string())?;
        index.add_batch(ids, vectors).map_err(|e| e.to_string())?;
        index.set_ef_search(params.ef_search);
        let build_secs = timer.elapsed_secs();
        let rss_after = current_rss_bytes();
        self.index = Some(index);
        Ok(BuildStats {
            build_secs,
            peak_rss_bytes: rss_delta(rss_before, rss_after),
            notes: vec![format!(
                "vanedb crate {}; ApproxIndex M={} efC={} seed={}; build uses add_batch \
                 (same graph topology as serial add; wall time is not sequential-insert peer)",
                option_env!("VANEDB_VERSION").unwrap_or("path"),
                params.m,
                params.ef_construction,
                params.seed
            )],
        })
    }

    fn search(&self, query: &[f32], k: usize, ef: usize) -> Result<Vec<u64>, String> {
        let index = self.index.as_ref().ok_or("vanedb: not built")?;
        let hits = index
            .search_with(query, k, &SearchParams::new().ef_search(ef))
            .map_err(|e| e.to_string())?;
        Ok(hits.into_iter().map(|h| h.id).collect())
    }

    fn save(&self, path: &Path) -> Result<u64, String> {
        let index = self.index.as_ref().ok_or("vanedb: not built")?;
        index.save(path).map_err(|e| e.to_string())?;
        file_size(path)
    }

    fn remove(&mut self, id: u64) -> Result<(), String> {
        let index = self.index.as_mut().ok_or("vanedb: not built")?;
        index.remove(id).map_err(|e| e.to_string())
    }
}

fn file_size(path: &Path) -> Result<u64, String> {
    std::fs::metadata(path)
        .map(|m| m.len())
        .map_err(|e| e.to_string())
}

fn rss_delta(before: Option<u64>, after: Option<u64>) -> Option<u64> {
    match (before, after) {
        (Some(b), Some(a)) if a >= b => Some(a - b),
        (_, after) => after,
    }
}
