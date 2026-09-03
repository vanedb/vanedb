//! The approved scope, as data.
//!
//! `docs/superpowers/specs/2026-05-28-vanedb-bench-design.md` lists the
//! operations this harness must measure. The implementation plan quietly
//! covered a third of them while its self-review claimed the scope was
//! complete (#63), so the claim is now checked mechanically rather than
//! asserted in prose.

/// Criterion group names. The benches build their group names from these, so
/// deleting a group here breaks compilation and deleting one from a bench
/// breaks [`SCOPE`]'s test — a scope claim cannot drift from the code silently.
pub mod groups {
    pub const L2_SQ: &str = "l2_sq";
    pub const COSINE: &str = "cosine";
    pub const DOT: &str = "dot";
    pub const STORE_ADD: &str = "store_add";
    pub const STORE_SEARCH: &str = "store_search";
    pub const HNSW_BUILD: &str = "hnsw_build";
    pub const HNSW_SEARCH: &str = "hnsw_search";
    pub const MMAP_BUILD: &str = "mmap_build";
    pub const MMAP_OPEN: &str = "mmap_open";
    pub const MMAP_SEARCH: &str = "mmap_search";
}

/// One operation the approved scope promises, and where it is measured.
pub struct Coverage {
    /// The operation as the spec words it.
    pub operation: &'static str,
    /// Criterion group, or the report binary, that reports it.
    pub measured_by: &'static str,
    /// Crate-relative source that must implement the measurement.
    pub source: &'static str,
    /// Identifier that source must reference — the mechanical link.
    pub must_reference: &'static str,
}

/// Every operation the approved design promises to measure.
pub const SCOPE: &[Coverage] = &[
    Coverage {
        operation: "L2 distance latency",
        measured_by: "l2_sq/dim={128,768}",
        source: "benches/distance.rs",
        must_reference: "groups::L2_SQ",
    },
    Coverage {
        operation: "cosine distance latency",
        measured_by: "cosine/dim={128,768}",
        source: "benches/distance.rs",
        must_reference: "groups::COSINE",
    },
    Coverage {
        operation: "dot distance latency",
        measured_by: "dot/dim={128,768}",
        source: "benches/distance.rs",
        must_reference: "groups::DOT",
    },
    Coverage {
        operation: "VectorStore add throughput",
        measured_by: "store_add/n=10000",
        source: "benches/store.rs",
        must_reference: "groups::STORE_ADD",
    },
    Coverage {
        operation: "VectorStore search latency",
        measured_by: "store_search/n={1000,10000}",
        source: "benches/store.rs",
        must_reference: "groups::STORE_SEARCH",
    },
    Coverage {
        operation: "HNSW build latency",
        measured_by: "hnsw_build",
        source: "benches/hnsw.rs",
        must_reference: "groups::HNSW_BUILD",
    },
    Coverage {
        operation: "HNSW search latency",
        measured_by: "hnsw_search",
        source: "benches/hnsw.rs",
        must_reference: "groups::HNSW_SEARCH",
    },
    Coverage {
        operation: "HNSW recall@k",
        measured_by: "report binary",
        source: "src/bin/report.rs",
        must_reference: "ground_truth::recall_at_k",
    },
    Coverage {
        operation: "mmap build latency",
        measured_by: "mmap_build",
        source: "benches/mmap.rs",
        must_reference: "groups::MMAP_BUILD",
    },
    Coverage {
        operation: "mmap open latency",
        measured_by: "mmap_open",
        source: "benches/mmap.rs",
        must_reference: "groups::MMAP_OPEN",
    },
    Coverage {
        operation: "mmap search latency",
        measured_by: "mmap_search",
        source: "benches/mmap.rs",
        must_reference: "groups::MMAP_SEARCH",
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    /// Sources are embedded at compile time: the test must not depend on the
    /// working directory it is run from.
    fn source(path: &str) -> &'static str {
        match path {
            "benches/distance.rs" => include_str!("../benches/distance.rs"),
            "benches/store.rs" => include_str!("../benches/store.rs"),
            "benches/hnsw.rs" => include_str!("../benches/hnsw.rs"),
            "benches/mmap.rs" => include_str!("../benches/mmap.rs"),
            "src/bin/report.rs" => include_str!("bin/report.rs"),
            other => panic!("SCOPE names {other}, which the test cannot read"),
        }
    }

    /// Proves the named source registers the measurement. It cannot prove the
    /// measurement is reached at runtime — that is what the CI smoke run and
    /// `cargo bench --no-run` are for.
    #[test]
    fn every_promised_operation_is_measured() {
        for entry in SCOPE {
            let src = source(entry.source);
            assert!(
                src.lines()
                    .filter(|line| !line.trim_start().starts_with("//"))
                    .any(|line| line.contains(entry.must_reference)),
                "{}: {} does not reference {}",
                entry.operation,
                entry.source,
                entry.must_reference
            );
        }
    }

    #[test]
    fn the_table_promises_each_operation_once() {
        for (i, entry) in SCOPE.iter().enumerate() {
            assert!(
                !SCOPE[..i].iter().any(|e| e.operation == entry.operation),
                "{} is listed twice",
                entry.operation
            );
        }
    }
}
