//! sqlite-vec (brute-force) adapter via rusqlite + statically linked amalgamation.

use std::path::Path;
use std::sync::Once;

use rusqlite::{Connection, OptionalExtension};

use super::{BuildParams, BuildStats, Engine, MetricKind};
use crate::measure::{current_rss_bytes, InstantTimer};

static INIT: Once = Once::new();

extern "C" {
    fn sqlite3_auto_extension(
        xEntryPoint: Option<
            unsafe extern "C" fn(
                *mut libsqlite3_sys::sqlite3,
                *mut *mut i8,
                *const libsqlite3_sys::sqlite3_api_routines,
            ) -> i32,
        >,
    ) -> i32;
    fn sqlite3_vec_init(
        db: *mut libsqlite3_sys::sqlite3,
        pz_err_msg: *mut *mut i8,
        p_api: *const libsqlite3_sys::sqlite3_api_routines,
    ) -> i32;
}

fn ensure_vec_extension() {
    INIT.call_once(|| {
        let rc = unsafe { sqlite3_auto_extension(Some(sqlite3_vec_init)) };
        assert_eq!(rc, 0, "sqlite3_auto_extension(sqlite3_vec_init) failed");
    });
}

pub struct SqliteVecEngine {
    conn: Option<Connection>,
    db_path: Option<std::path::PathBuf>,
    metric: MetricKind,
    dim: usize,
}

impl SqliteVecEngine {
    pub fn new() -> Self {
        Self {
            conn: None,
            db_path: None,
            metric: MetricKind::L2,
            dim: 0,
        }
    }
}

impl Default for SqliteVecEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl Engine for SqliteVecEngine {
    fn name(&self) -> &'static str {
        "sqlite-vec"
    }

    fn version(&self) -> String {
        "0.1.6".into()
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
        _params: &BuildParams,
    ) -> Result<BuildStats, String> {
        ensure_vec_extension();
        let path = std::env::temp_dir().join(format!(
            "vanedb-compare-sqlite-vec-{}-{}.db",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let _ = std::fs::remove_file(&path);

        let rss_before = current_rss_bytes();
        let timer = InstantTimer::start();
        let conn = Connection::open(&path).map_err(|e| e.to_string())?;
        // vec0 virtual table; type float[dim].
        conn.execute_batch(&format!(
            "CREATE VIRTUAL TABLE vecs USING vec0(id INTEGER PRIMARY KEY, embedding float[{dim}]);"
        ))
        .map_err(|e| e.to_string())?;

        let mut stmt = conn
            .prepare("INSERT INTO vecs(id, embedding) VALUES (?1, ?2)")
            .map_err(|e| e.to_string())?;
        for (i, &id) in ids.iter().enumerate() {
            let row = &vectors[i * dim..(i + 1) * dim];
            let blob = float_slice_to_le_bytes(row);
            stmt.execute(rusqlite::params![id as i64, blob])
                .map_err(|e| e.to_string())?;
        }
        drop(stmt);
        let build_secs = timer.elapsed_secs();
        let rss_after = current_rss_bytes();

        self.conn = Some(conn);
        self.db_path = Some(path);
        self.metric = metric;
        self.dim = dim;
        Ok(BuildStats {
            build_secs,
            peak_rss_bytes: rss_delta(rss_before, rss_after),
            notes: vec![
                "brute-force KNN via vec0 for L2 (no ANN; sqlite-vec #25)".into(),
                "ef unused — single construction-ef latency row only".into(),
                format!(
                    "metric {}: {}",
                    metric.as_str(),
                    match metric {
                        MetricKind::L2 => "native vec0 MATCH",
                        MetricKind::Cosine => {
                            "harness-side f32 brute-force over blobs (not sqlite-vec)"
                        }
                    }
                ),
            ],
        })
    }

    fn search(&self, query: &[f32], k: usize, _ef: usize) -> Result<Vec<u64>, String> {
        let conn = self.conn.as_ref().ok_or("sqlite-vec: not built")?;
        let blob = float_slice_to_le_bytes(query);
        // vec0 MATCH returns nearest by L2; cosine is not a native vec0 metric.
        // For cosine we fall back to a full scan with a SQL expression.
        match self.metric {
            MetricKind::L2 => {
                let mut stmt = conn
                    .prepare(
                        "SELECT id FROM vecs WHERE embedding MATCH ?1 AND k = ?2 ORDER BY distance",
                    )
                    .map_err(|e| e.to_string())?;
                let rows = stmt
                    .query_map(rusqlite::params![blob, k as i64], |row| {
                        row.get::<_, i64>(0)
                    })
                    .map_err(|e| e.to_string())?;
                let mut out = Vec::new();
                for r in rows {
                    out.push(r.map_err(|e| e.to_string())? as u64);
                }
                Ok(out)
            }
            MetricKind::Cosine => {
                // vec0 has no cosine metric. This path is a harness-side
                // brute-force scan over stored blobs — not sqlite-vec ANN.
                let mut stmt = conn
                    .prepare("SELECT id, embedding FROM vecs")
                    .map_err(|e| e.to_string())?;
                let rows = stmt
                    .query_map([], |row| {
                        let id: i64 = row.get(0)?;
                        let emb: Vec<u8> = row.get(1)?;
                        Ok((id, emb))
                    })
                    .map_err(|e| e.to_string())?;
                let mut scored = Vec::new();
                for r in rows {
                    let (id, emb) = r.map_err(|e| e.to_string())?;
                    let v = le_bytes_to_floats(&emb);
                    scored.push((cosine_distance(query, &v), id as u64));
                }
                scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                Ok(scored.into_iter().take(k).map(|(_, id)| id).collect())
            }
        }
    }

    fn save(&self, path: &Path) -> Result<u64, String> {
        let src = self.db_path.as_ref().ok_or("sqlite-vec: not built")?;
        // Checkpoint so the file is self-contained.
        if let Some(conn) = &self.conn {
            let _ = conn
                .query_row("SELECT vec_version()", [], |row| row.get::<_, String>(0))
                .optional();
            conn.execute_batch("PRAGMA wal_checkpoint(TRUNCATE);")
                .map_err(|e| e.to_string())?;
        }
        std::fs::copy(src, path).map_err(|e| e.to_string())?;
        std::fs::metadata(path)
            .map(|m| m.len())
            .map_err(|e| e.to_string())
    }

    fn remove(&mut self, id: u64) -> Result<(), String> {
        let conn = self.conn.as_ref().ok_or("sqlite-vec: not built")?;
        conn.execute(
            "DELETE FROM vecs WHERE id = ?1",
            rusqlite::params![id as i64],
        )
        .map_err(|e| e.to_string())?;
        Ok(())
    }
}

fn float_slice_to_le_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for f in v {
        out.extend_from_slice(&f.to_le_bytes());
    }
    out
}

fn le_bytes_to_floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let (mut dot, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
    for (x, y) in a.iter().zip(b) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = na.sqrt() * nb.sqrt();
    if !(denom > 0.0 && denom.is_finite()) {
        return 1.0;
    }
    1.0 - (dot / denom).clamp(-1.0, 1.0)
}

fn rss_delta(before: Option<u64>, after: Option<u64>) -> Option<u64> {
    match (before, after) {
        (Some(b), Some(a)) if a >= b => Some(a - b),
        (_, after) => after,
    }
}
