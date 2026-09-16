//! Competitor benchmark harness (RFC 0003 / #198).

pub mod engines;
pub mod fixture;
pub mod ground_truth;
pub mod measure;
pub mod report;
pub mod run;

pub use engines::{BuildParams, Engine, EngineKind, MetricKind};
pub use fixture::{Fixture, FixtureMeta, FixtureRole, FIXTURE_MAGIC, PUBLISH_MIN_DOCS};
pub use run::{run_comparison, RunConfig};
