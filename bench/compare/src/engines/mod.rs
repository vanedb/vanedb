//! Shared engine interface and adapters.

mod hnsw_rs_eng;
mod hnswlib;
mod instant;
mod sqlite_vec;
mod usearch_eng;
mod vanedb_eng;

use std::path::Path;

pub use hnsw_rs_eng::HnswRsEngine;
pub use hnswlib::HnswlibEngine;
pub use instant::InstantDistanceEngine;
pub use sqlite_vec::SqliteVecEngine;
pub use usearch_eng::UsearchEngine;
pub use vanedb_eng::VanedbEngine;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MetricKind {
    L2,
    Cosine,
}

impl MetricKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::L2 => "l2",
            Self::Cosine => "cosine",
        }
    }
}

/// HNSW construction knobs shared across engines that expose them.
#[derive(Clone, Debug)]
pub struct BuildParams {
    pub m: usize,
    pub ef_construction: usize,
    /// Default search ef used when an engine does not take ef per call.
    pub ef_search: usize,
    pub seed: u64,
}

impl Default for BuildParams {
    fn default() -> Self {
        Self {
            m: 16,
            ef_construction: 200,
            ef_search: 50,
            seed: 42,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct BuildStats {
    pub build_secs: f64,
    /// Peak resident set during build, when the host can report it.
    pub peak_rss_bytes: Option<u64>,
    /// Notes recorded for engines whose parameters could not match the shared set.
    pub notes: Vec<String>,
}

pub trait Engine: Send {
    fn name(&self) -> &'static str;
    fn version(&self) -> String;
    fn supports_delete(&self) -> bool;
    fn supports_save(&self) -> bool;

    fn build(
        &mut self,
        vectors: &[f32],
        ids: &[u64],
        dim: usize,
        metric: MetricKind,
        params: &BuildParams,
    ) -> Result<BuildStats, String>;

    fn search(&self, query: &[f32], k: usize, ef: usize) -> Result<Vec<u64>, String>;

    /// Persist the index and return the on-disk byte size.
    fn save(&self, path: &Path) -> Result<u64, String>;

    fn remove(&mut self, id: u64) -> Result<(), String>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum EngineKind {
    Vanedb,
    Usearch,
    Hnswlib,
    InstantDistance,
    HnswRs,
    SqliteVec,
}

impl EngineKind {
    pub fn all() -> &'static [EngineKind] {
        &[
            EngineKind::Vanedb,
            EngineKind::Usearch,
            EngineKind::Hnswlib,
            EngineKind::InstantDistance,
            EngineKind::HnswRs,
            EngineKind::SqliteVec,
        ]
    }

    pub fn make(self) -> Box<dyn Engine> {
        match self {
            EngineKind::Vanedb => Box::new(VanedbEngine::new()),
            EngineKind::Usearch => Box::new(UsearchEngine::new()),
            EngineKind::Hnswlib => Box::new(HnswlibEngine::new()),
            EngineKind::InstantDistance => Box::new(InstantDistanceEngine::new()),
            EngineKind::HnswRs => Box::new(HnswRsEngine::new()),
            EngineKind::SqliteVec => Box::new(SqliteVecEngine::new()),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            EngineKind::Vanedb => "vanedb",
            EngineKind::Usearch => "usearch",
            EngineKind::Hnswlib => "hnswlib",
            EngineKind::InstantDistance => "instant-distance",
            EngineKind::HnswRs => "hnsw_rs",
            EngineKind::SqliteVec => "sqlite-vec",
        }
    }
}
