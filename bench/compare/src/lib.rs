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
pub use publish::{
    dedicated_hw_attested, refuse_bad_hw_label, refuse_ci_env_for_markdown,
    refuse_incomplete_delete_rows, refuse_incomplete_engine_set, refuse_incomplete_save_rows,
    refuse_markdown_flags, refuse_noncanonical_params, refuse_unattested_dedicated_hw,
    shared_runner_env, CanonicalParamsGate, MarkdownFlagGate, PUBLISH_EF_CONSTRUCTION,
    PUBLISH_EF_SWEEP, PUBLISH_ENGINES_COSINE, PUBLISH_ENGINES_L2, PUBLISH_K, PUBLISH_M,
    PUBLISH_SEED,
};
pub use run::{run_comparison, RunConfig};
