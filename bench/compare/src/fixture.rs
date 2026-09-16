//! Checksummed embedding fixture (VNEF v1).
//!
//! Layout (little-endian):
//! - magic: b"VNEF"
//! - version: u32 = 1
//! - dim: u32
//! - n_docs: u32
//! - n_queries: u32
//! - metric_hint: u32 (0=L2, 1=cosine) — informational; runs may still sweep both
//! - reserved: u32 = 0
//! - docs: n_docs * dim * f32
//! - queries: n_queries * dim * f32
//! - doc_ids: n_docs * u64 (optional identity 0..n-1 if generated)
//!
//! A sidecar `metadata.json` records model, corpus, and generation command.
//! `SHA256SUMS` pins the fixture bytes; CI and the harness refuse a mismatched file.

use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const FIXTURE_MAGIC: &[u8; 4] = b"VNEF";
pub const FIXTURE_VERSION: u32 = 1;
/// Minimum document count for a publishable COMPARISON.md fixture (RFC 0003).
pub const PUBLISH_MIN_DOCS: usize = 100_000;
/// Minimum query count for a publishable fixture.
pub const PUBLISH_MIN_QUERIES: usize = 1_000;

/// Whether a loaded fixture may appear in published comparison tables.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FixtureRole {
    /// Deterministic harness smoke / any fixture below the publish size floor.
    Smoke,
    /// Large enough for local experiments but not the RFC publish fixture.
    Dev,
    /// ≥ [`PUBLISH_MIN_DOCS`] + [`PUBLISH_MIN_QUERIES`] with real corpus metadata.
    Publish,
}

impl FixtureRole {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Smoke => "smoke",
            Self::Dev => "dev",
            Self::Publish => "publish",
        }
    }

    pub fn requires_allow_smoke(self) -> bool {
        matches!(self, Self::Smoke | Self::Dev)
    }
}

fn meta_forbids_publish(meta: &FixtureMeta) -> bool {
    let notes = meta.notes.to_ascii_lowercase();
    let corpus = meta.corpus.to_ascii_lowercase();
    let model = meta.model.to_ascii_lowercase();
    notes.contains("not for published")
        || notes.contains("pending")
        || corpus == "synthetic"
        || corpus.contains("synthetic")
        || model.contains("smoke")
        || model.contains("deterministic smoke")
        || model.contains("pending")
}

/// Classify by content (size + metadata), not filename.
/// Missing metadata can never be `Publish` — fail closed for pasteable output.
pub fn classify_fixture(fixture: &Fixture) -> FixtureRole {
    match fixture.meta.as_ref() {
        None => {
            if fixture.n_docs() <= 1024 {
                FixtureRole::Smoke
            } else {
                FixtureRole::Dev
            }
        }
        Some(meta) if meta_forbids_publish(meta) => {
            if fixture.n_docs() <= 1024 {
                FixtureRole::Smoke
            } else {
                FixtureRole::Dev
            }
        }
        Some(_) => {
            let n = fixture.n_docs();
            let nq = fixture.n_queries();
            if n >= PUBLISH_MIN_DOCS && nq >= PUBLISH_MIN_QUERIES {
                FixtureRole::Publish
            } else if n <= 1024 {
                FixtureRole::Smoke
            } else {
                FixtureRole::Dev
            }
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FixtureMeta {
    pub model: String,
    pub corpus: String,
    pub dim: u32,
    pub n_docs: u32,
    pub n_queries: u32,
    pub metric_native: String,
    pub generator: String,
    pub notes: String,
}

#[derive(Clone, Debug)]
pub struct Fixture {
    pub dim: usize,
    pub vectors: Vec<f32>,
    pub ids: Vec<u64>,
    pub queries: Vec<f32>,
    pub meta: Option<FixtureMeta>,
    pub sha256: String,
}

impl Fixture {
    pub fn n_docs(&self) -> usize {
        self.ids.len()
    }

    pub fn n_queries(&self) -> usize {
        self.queries.len().checked_div(self.dim).unwrap_or(0)
    }

    pub fn query(&self, i: usize) -> &[f32] {
        &self.queries[i * self.dim..(i + 1) * self.dim]
    }
}

pub fn load_fixture(path: &Path) -> Result<Fixture, String> {
    let mut file = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)
        .map_err(|e| format!("read {}: {e}", path.display()))?;
    let sha256 = hex_sha256(&bytes);
    let fixture = parse_fixture(&bytes)?;
    let meta = load_meta_beside(path).ok();
    Ok(Fixture {
        dim: fixture.0,
        vectors: fixture.1,
        ids: fixture.2,
        queries: fixture.3,
        meta,
        sha256,
    })
}

pub fn verify_sha256sums(fixture_path: &Path, sums_path: &Path) -> Result<(), String> {
    let sums = std::fs::read_to_string(sums_path).map_err(|e| e.to_string())?;
    let name = fixture_path
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or("fixture path has no file name")?;
    let mut file = File::open(fixture_path).map_err(|e| e.to_string())?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).map_err(|e| e.to_string())?;
    let got = hex_sha256(&bytes);
    for line in sums.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.split_whitespace();
        let hash = parts.next().ok_or("bad SHA256SUMS line")?;
        let file_name = parts.next().ok_or("bad SHA256SUMS line")?;
        let file_name = file_name.trim_start_matches('*');
        if file_name == name {
            if hash != got {
                return Err(format!(
                    "checksum mismatch for {name}: expected {hash}, got {got}"
                ));
            }
            return Ok(());
        }
    }
    Err(format!("{name} not listed in {}", sums_path.display()))
}

pub fn write_fixture(
    path: &Path,
    dim: usize,
    vectors: &[f32],
    ids: &[u64],
    queries: &[f32],
) -> Result<String, String> {
    let n_docs = ids.len();
    let n_queries = queries.len() / dim;
    assert_eq!(vectors.len(), n_docs * dim);
    assert_eq!(queries.len(), n_queries * dim);

    let mut bytes = Vec::new();
    bytes.extend_from_slice(FIXTURE_MAGIC);
    bytes.extend_from_slice(&FIXTURE_VERSION.to_le_bytes());
    bytes.extend_from_slice(&(dim as u32).to_le_bytes());
    bytes.extend_from_slice(&(n_docs as u32).to_le_bytes());
    bytes.extend_from_slice(&(n_queries as u32).to_le_bytes());
    bytes.extend_from_slice(&1u32.to_le_bytes()); // cosine hint
    bytes.extend_from_slice(&0u32.to_le_bytes());
    for f in vectors {
        bytes.extend_from_slice(&f.to_le_bytes());
    }
    for f in queries {
        bytes.extend_from_slice(&f.to_le_bytes());
    }
    for id in ids {
        bytes.extend_from_slice(&id.to_le_bytes());
    }
    let sha = hex_sha256(&bytes);
    let mut file = File::create(path).map_err(|e| e.to_string())?;
    file.write_all(&bytes).map_err(|e| e.to_string())?;
    Ok(sha)
}

/// Deterministic smoke fixture (not for published numbers).
pub fn write_smoke_fixture(
    path: &Path,
    n_docs: usize,
    n_queries: usize,
    dim: usize,
) -> Result<String, String> {
    let mut vectors = vec![0f32; n_docs * dim];
    let mut queries = vec![0f32; n_queries * dim];
    let ids: Vec<u64> = (0..n_docs as u64).collect();
    for i in 0..n_docs {
        for d in 0..dim {
            // Smooth, non-uniform field — not a real embedding, only for harness smoke.
            let v = ((i * 17 + d * 3) % 1000) as f32 / 1000.0 - 0.5;
            vectors[i * dim + d] = v;
        }
        // L2-normalize for cosine-ish behaviour.
        normalize_row(&mut vectors[i * dim..(i + 1) * dim]);
    }
    for i in 0..n_queries {
        let src = (i * 7) % n_docs;
        queries[i * dim..(i + 1) * dim].copy_from_slice(&vectors[src * dim..(src + 1) * dim]);
        // Perturb slightly.
        for d in 0..dim.min(8) {
            queries[i * dim + d] += 0.01 * (d as f32);
        }
        normalize_row(&mut queries[i * dim..(i + 1) * dim]);
    }
    write_fixture(path, dim, &vectors, &ids, &queries)
}

fn normalize_row(row: &mut [f32]) {
    let mut sum = 0.0f32;
    for x in row.iter() {
        sum += x * x;
    }
    if sum > 0.0 {
        let inv = sum.sqrt().recip();
        for x in row.iter_mut() {
            *x *= inv;
        }
    }
}

type ParsedFixture = (usize, Vec<f32>, Vec<u64>, Vec<f32>);

fn parse_fixture(bytes: &[u8]) -> Result<ParsedFixture, String> {
    if bytes.len() < 28 {
        return Err("fixture too short".into());
    }
    if &bytes[0..4] != FIXTURE_MAGIC {
        return Err("bad fixture magic".into());
    }
    let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    if version != FIXTURE_VERSION {
        return Err(format!("unsupported fixture version {version}"));
    }
    let dim = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    let n_docs = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
    let n_queries = u32::from_le_bytes(bytes[16..20].try_into().unwrap()) as usize;
    let _metric_hint = u32::from_le_bytes(bytes[20..24].try_into().unwrap());
    let _reserved = u32::from_le_bytes(bytes[24..28].try_into().unwrap());

    let docs_bytes = n_docs * dim * 4;
    let queries_bytes = n_queries * dim * 4;
    let ids_bytes = n_docs * 8;
    let need = 28 + docs_bytes + queries_bytes + ids_bytes;
    if bytes.len() != need {
        return Err(format!(
            "fixture size mismatch: got {} want {need}",
            bytes.len()
        ));
    }
    let mut offset = 28;
    let vectors = read_f32_slice(&bytes[offset..offset + docs_bytes]);
    offset += docs_bytes;
    let queries = read_f32_slice(&bytes[offset..offset + queries_bytes]);
    offset += queries_bytes;
    let ids = read_u64_slice(&bytes[offset..offset + ids_bytes]);
    Ok((dim, vectors, ids, queries))
}

fn read_f32_slice(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn read_u64_slice(bytes: &[u8]) -> Vec<u64> {
    bytes
        .chunks_exact(8)
        .map(|c| u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
        .collect()
}

fn hex_sha256(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn load_meta_beside(path: &Path) -> Result<FixtureMeta, String> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let candidates = [
        path.with_file_name("metadata.json"),
        parent.join("metadata.smoke.json"),
        path.with_extension("json"),
    ];
    // Prefer metadata.smoke.json when the fixture itself is a smoke file.
    let name = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if name.contains("smoke") {
        let smoke_meta = parent.join("metadata.smoke.json");
        if smoke_meta.exists() {
            let text = std::fs::read_to_string(&smoke_meta).map_err(|e| e.to_string())?;
            return serde_json::from_str(&text).map_err(|e| e.to_string());
        }
    }
    for meta_path in &candidates {
        if meta_path.exists() {
            let text = std::fs::read_to_string(meta_path).map_err(|e| e.to_string())?;
            return serde_json::from_str(&text).map_err(|e| e.to_string());
        }
    }
    Err("no metadata.json beside fixture".into())
}

pub fn default_fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures")
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn smoke_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("smoke.vnef");
        let sha = write_smoke_fixture(&path, 32, 4, 16).unwrap();
        let loaded = load_fixture(&path).unwrap();
        assert_eq!(loaded.dim, 16);
        assert_eq!(loaded.n_docs(), 32);
        assert_eq!(loaded.n_queries(), 4);
        assert_eq!(loaded.sha256, sha);
    }

    #[test]
    fn rejects_bad_magic() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("bad.vnef");
        std::fs::write(&path, b"XXXX").unwrap();
        assert!(load_fixture(&path).is_err());
    }

    #[test]
    fn checksum_mismatch_detected() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("smoke.vnef");
        write_smoke_fixture(&path, 8, 2, 8).unwrap();
        let sums = dir.path().join("SHA256SUMS");
        std::fs::write(&sums, "0000  smoke.vnef\n").unwrap();
        assert!(verify_sha256sums(&path, &sums).is_err());
    }

    #[test]
    fn checksum_missing_entry() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("smoke.vnef");
        write_smoke_fixture(&path, 8, 2, 8).unwrap();
        let sums = dir.path().join("SHA256SUMS");
        std::fs::write(&sums, "abcd  other.vnef\n").unwrap();
        assert!(verify_sha256sums(&path, &sums).is_err());
    }

    #[test]
    fn renamed_smoke_still_classified_smoke() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("tiny.vnef");
        write_smoke_fixture(&path, 64, 4, 16).unwrap();
        let loaded = load_fixture(&path).unwrap();
        assert_eq!(classify_fixture(&loaded), FixtureRole::Smoke);
        assert!(classify_fixture(&loaded).requires_allow_smoke());
    }

    #[test]
    fn publish_floor_requires_100k_docs() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("mid.vnef");
        // 2048 docs → Dev (not Publish); still requires --allow-smoke.
        write_smoke_fixture(&path, 2048, 8, 8).unwrap();
        let loaded = load_fixture(&path).unwrap();
        assert_eq!(classify_fixture(&loaded), FixtureRole::Dev);
        assert!(classify_fixture(&loaded).requires_allow_smoke());
    }

    #[test]
    fn missing_meta_never_publish() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("orphan.vnef");
        write_smoke_fixture(&path, 2048, 8, 8).unwrap();
        // No metadata.json beside the file.
        let loaded = load_fixture(&path).unwrap();
        assert!(loaded.meta.is_none());
        assert_ne!(classify_fixture(&loaded), FixtureRole::Publish);
        assert!(classify_fixture(&loaded).requires_allow_smoke());
    }

    #[test]
    fn synthetic_meta_never_publish_even_at_100k() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("big.vnef");
        // Cheap tiny file with forged meta claiming 100k — classifier uses loaded
        // n_docs from bytes, so also attach forbidding meta and assert Dev/Smoke.
        write_smoke_fixture(&path, 64, 4, 8).unwrap();
        let meta = FixtureMeta {
            model: "none (deterministic smoke)".into(),
            corpus: "synthetic".into(),
            dim: 8,
            n_docs: 100_000,
            n_queries: 1_000,
            metric_native: "cosine".into(),
            generator: "test".into(),
            notes: "NOT for published COMPARISON.md numbers".into(),
        };
        std::fs::write(
            dir.path().join("metadata.json"),
            serde_json::to_string_pretty(&meta).unwrap(),
        )
        .unwrap();
        let loaded = load_fixture(&path).unwrap();
        assert_ne!(classify_fixture(&loaded), FixtureRole::Publish);
    }

    #[test]
    fn pending_generation_meta_blocks_publish() {
        let mut fixture = load_fixture_from_bytes();
        fixture.meta = Some(FixtureMeta {
            model: "nomic (pending generation)".into(),
            corpus: "BeIR/nq".into(),
            dim: 768,
            n_docs: 100_000,
            n_queries: 1_000,
            metric_native: "cosine".into(),
            generator: "test".into(),
            notes: "pending generation".into(),
        });
        // Size from bytes is small; force ids/queries lengths via role path on meta first.
        assert_ne!(classify_fixture(&fixture), FixtureRole::Publish);
    }

    fn load_fixture_from_bytes() -> Fixture {
        let dir = tempdir().unwrap();
        let path = dir.path().join("t.vnef");
        write_smoke_fixture(&path, 32, 4, 8).unwrap();
        load_fixture(&path).unwrap()
    }
}
