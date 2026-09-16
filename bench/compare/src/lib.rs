//! Competitor benchmark harness (RFC 0003 / #198).

pub mod engines;
pub mod fixture;
pub mod ground_truth;
pub mod measure;
pub mod publish;
pub mod report;
pub mod run;

pub use engines::{BuildParams, Engine, EngineKind, MetricKind};
pub use fixture::{
    Fixture, FixtureMeta, FixtureRole, FIXTURE_MAGIC, PUBLISH_MIN_DOCS, PUBLISH_MIN_QUERIES,
};
pub use publish::{refuse_incomplete_save_rows, refuse_markdown_flags, MarkdownFlagGate};
pub use run::{run_comparison, RunConfig};
