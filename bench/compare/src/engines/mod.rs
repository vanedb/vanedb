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

#[cfg(test)]
mod version_pin_tests {
    use super::*;

    fn exact_dep(cargo: &str, name: &str) -> String {
        let prefix = format!("{name} = \"=");
        for line in cargo.lines() {
            let line = line.trim();
            if let Some(rest) = line.strip_prefix(&prefix) {
                return rest.trim_end_matches('"').to_string();
            }
        }
        panic!("Cargo.toml missing exact pin for {name}");
    }

    #[test]
    fn competitor_versions_match_cargo_vendor_and_comparison() {
        let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let cargo = std::fs::read_to_string(dir.join("Cargo.toml")).unwrap();
        let comparison = std::fs::read_to_string(dir.join("../COMPARISON.md")).unwrap();
        let vendor = std::fs::read_to_string(dir.join("vendor/README.md")).unwrap();
        let sqlite_h = std::fs::read_to_string(dir.join("vendor/sqlite-vec.h")).unwrap();
        let pending =
            std::fs::read_to_string(dir.join("scripts/testdata/COMPARISON.pending.md")).unwrap();

        let usearch = exact_dep(&cargo, "usearch");
        let instant = exact_dep(&cargo, "instant-distance");
        let hnsw_rs_v = exact_dep(&cargo, "hnsw_rs");
        assert_eq!(UsearchEngine::new().version(), usearch);
        assert_eq!(InstantDistanceEngine::new().version(), instant);
        assert_eq!(HnswRsEngine::new().version(), hnsw_rs_v);

        let hnswlib_v = HnswlibEngine::new().version();
        assert!(
            vendor.contains(&format!("| v{hnswlib_v} |")),
            "vendor README missing hnswlib v{hnswlib_v}"
        );
        let sqlite_v = SqliteVecEngine::new().version();
        assert!(
            sqlite_h.contains(&format!("#define SQLITE_VEC_VERSION \"v{sqlite_v}\"")),
            "sqlite-vec.h version != engine {sqlite_v}"
        );
        assert!(
            vendor.contains(&format!("| v{sqlite_v} |")),
            "vendor README missing sqlite-vec v{sqlite_v}"
        );

        let engines_row = format!(
            "vanedb, usearch {usearch}, hnswlib {hnswlib_v}, instant-distance {instant}, hnsw_rs {hnsw_rs_v}, sqlite-vec {sqlite_v} (L2)"
        );
        for doc in [&comparison, &pending] {
            assert!(
                doc.contains(&engines_row),
                "missing methodology engines row:\n{engines_row}"
            );
            assert!(doc.contains(&format!("| USearch | `usearch` crate | {usearch} |")));
            assert!(doc.contains(&format!(
                "| hnswlib | vendored C++ + thin FFI | {hnswlib_v} |"
            )));
            assert!(doc.contains(&format!(
                "| instant-distance | crate | {instant} (`M=32` fixed) |"
            )));
            assert!(doc.contains(&format!("| hnsw_rs | crate | {hnsw_rs_v} |")));
            assert!(doc.contains(&format!(
                "| sqlite-vec | vendored amalgamation via rusqlite | {sqlite_v} |"
            )));
        }
    }
}
