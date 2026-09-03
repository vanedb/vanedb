//! Runtime configuration for the `report` binary.
//!
//! The defaults reproduce the published snapshot. The environment overrides
//! exist so CI can execute the same code path at sizes that finish in seconds
//! (see `.github/workflows/integration-ci.yml`).

use std::path::PathBuf;

/// Workload and output settings for one `report` run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReportConfig {
    pub dim: usize,
    pub n: usize,
    pub k: usize,
    pub queries: usize,
    pub out: PathBuf,
}

impl ReportConfig {
    /// Reads the `VANEDB_BENCH_*` variables from the process environment.
    pub fn from_env() -> Result<Self, String> {
        Self::from_lookup(|name| std::env::var(name).ok())
    }

    /// Same, against an arbitrary lookup — the process environment is global
    /// mutable state, so tests supply their own.
    pub fn from_lookup(lookup: impl Fn(&str) -> Option<String>) -> Result<Self, String> {
        let dim = size(&lookup, "VANEDB_BENCH_DIM", 128)?;
        let n = size(&lookup, "VANEDB_BENCH_N", 10_000)?;
        let k = size(&lookup, "VANEDB_BENCH_K", 10)?;
        let queries = size(&lookup, "VANEDB_BENCH_QUERIES", 100)?;

        // The distance comparison comes from the first two corpus vectors.
        if n < 2 {
            return Err(format!("VANEDB_BENCH_N: must be at least 2, got {n}"));
        }
        if k > n {
            return Err(format!(
                "VANEDB_BENCH_K: {k} exceeds VANEDB_BENCH_N ({n}); no engine can return that many neighbours"
            ));
        }

        // Anchored to the crate, not the cwd: the snapshot belongs beside the
        // README that quotes it however the binary was invoked.
        let out = lookup("VANEDB_BENCH_OUT")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("RESULTS.md"));

        Ok(Self {
            dim,
            n,
            k,
            queries,
            out,
        })
    }
}

fn size(
    lookup: &impl Fn(&str) -> Option<String>,
    name: &str,
    default: usize,
) -> Result<usize, String> {
    let Some(raw) = lookup(name) else {
        return Ok(default);
    };
    let value: usize = raw
        .trim()
        .parse()
        .map_err(|_| format!("{name}: expected a positive integer, got {raw:?}"))?;
    if value == 0 {
        return Err(format!("{name}: must be at least 1"));
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(vars: &[(&str, &str)]) -> Result<ReportConfig, String> {
        ReportConfig::from_lookup(|name| {
            vars.iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| (*value).to_string())
        })
    }

    #[test]
    fn defaults_reproduce_the_published_snapshot() {
        let cfg = config(&[]).unwrap();
        assert_eq!(cfg.dim, 128);
        assert_eq!(cfg.n, 10_000);
        assert_eq!(cfg.k, 10);
        assert_eq!(cfg.queries, 100);
    }

    #[test]
    fn results_default_beside_the_crate_not_the_working_directory() {
        let cfg = config(&[]).unwrap();
        assert_eq!(
            cfg.out,
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("RESULTS.md")
        );
        assert!(
            cfg.out.is_absolute(),
            "{} must not depend on the cwd",
            cfg.out.display()
        );
    }

    #[test]
    fn every_dimension_is_overridable() {
        let cfg = config(&[
            ("VANEDB_BENCH_DIM", "32"),
            ("VANEDB_BENCH_N", "500"),
            ("VANEDB_BENCH_K", "5"),
            ("VANEDB_BENCH_QUERIES", "3"),
            ("VANEDB_BENCH_OUT", "/tmp/smoke.md"),
        ])
        .unwrap();
        assert_eq!(
            cfg,
            ReportConfig {
                dim: 32,
                n: 500,
                k: 5,
                queries: 3,
                out: PathBuf::from("/tmp/smoke.md"),
            }
        );
    }

    #[test]
    fn a_non_numeric_value_names_the_variable_it_came_from() {
        let err = config(&[("VANEDB_BENCH_N", "ten thousand")]).unwrap_err();
        assert!(err.contains("VANEDB_BENCH_N"), "unhelpful message: {err}");
    }

    #[test]
    fn zero_sized_workloads_are_rejected() {
        for var in [
            "VANEDB_BENCH_DIM",
            "VANEDB_BENCH_N",
            "VANEDB_BENCH_K",
            "VANEDB_BENCH_QUERIES",
        ] {
            let err = config(&[(var, "0")]).unwrap_err();
            assert!(err.contains(var), "unhelpful message for {var}: {err}");
        }
    }

    #[test]
    fn k_may_not_exceed_the_corpus() {
        // Otherwise the search asserts deep inside the run: every engine
        // returns fewer than k neighbours and the failure names neither.
        let err = config(&[("VANEDB_BENCH_N", "8"), ("VANEDB_BENCH_K", "10")]).unwrap_err();
        assert!(err.contains("VANEDB_BENCH_K"), "unhelpful message: {err}");
    }

    #[test]
    fn the_corpus_holds_the_two_vectors_the_distance_bench_needs() {
        let err = config(&[("VANEDB_BENCH_N", "1"), ("VANEDB_BENCH_K", "1")]).unwrap_err();
        assert!(err.contains("VANEDB_BENCH_N"), "unhelpful message: {err}");
    }
}
