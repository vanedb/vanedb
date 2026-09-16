//! Interleaved multi-engine comparison runner.

use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::engines::{BuildParams, EngineKind, MetricKind};
use crate::fixture::{Fixture, FixtureRole};
use crate::ground_truth::{brute_force_topk_f64, recall_at_k, GtMetric};
use crate::measure::{median_f64, spread_f64, InstantTimer};

/// Local noise floor from AGENTS.md / existing bench abtest.
pub const NOISE_FLOOR: f64 = 0.03;

#[derive(Clone, Debug)]
pub struct RunConfig {
    pub engines: Vec<EngineKind>,
    pub metric: MetricKind,
    pub k: usize,
    pub ef_sweep: Vec<usize>,
    pub params: BuildParams,
    pub rounds: usize,
    pub out_dir: PathBuf,
    pub max_queries: Option<usize>,
    pub skip_save: bool,
    pub skip_delete: bool,
    pub fixture_role: FixtureRole,
}

#[derive(Clone, Debug, Serialize)]
pub struct EngineResult {
    pub engine: String,
    pub version: String,
    pub metric: String,
    pub build_secs_median: f64,
    pub build_secs_spread: f64,
    pub peak_rss_bytes: Option<u64>,
    pub file_size_bytes: Option<u64>,
    pub latency_ns_by_ef: Vec<LatencyRow>,
    pub recall_at_k_by_ef: Vec<RecallRow>,
    pub delete_ok: Option<bool>,
    pub notes: Vec<String>,
    pub rounds: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct LatencyRow {
    pub ef: usize,
    pub median_ns: f64,
    pub spread: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct RecallRow {
    pub ef: usize,
    pub mean_recall: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct ComparisonReport {
    pub commit: String,
    pub hostname: String,
    pub hardware_label: String,
    pub recorded_at_utc: String,
    pub fixture_role: FixtureRole,
    pub fixture_sha256: String,
    pub fixture_n_docs: usize,
    pub fixture_n_queries: usize,
    pub fixture_dim: usize,
    pub metric: String,
    pub k: usize,
    pub params: ParamsOut,
    pub results: Vec<EngineResult>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ParamsOut {
    pub m: usize,
    pub ef_construction: usize,
    pub seed: u64,
}

pub fn run_comparison(fixture: &Fixture, cfg: &RunConfig) -> Result<ComparisonReport, String> {
    let n_queries = cfg
        .max_queries
        .unwrap_or(fixture.n_queries())
        .min(fixture.n_queries());
    if n_queries == 0 {
        return Err("fixture has no queries".into());
    }

    let gt_metric = match cfg.metric {
        MetricKind::L2 => GtMetric::L2,
        MetricKind::Cosine => GtMetric::Cosine,
    };

    // Precompute exact neighbours once (f64).
    let mut truth: Vec<Vec<u64>> = Vec::with_capacity(n_queries);
    for qi in 0..n_queries {
        truth.push(brute_force_topk_f64(
            &fixture.vectors,
            &fixture.ids,
            fixture.dim,
            fixture.query(qi),
            cfg.k,
            gt_metric,
        ));
    }

    // Interleaved rounds: for each round, run every engine in order (A-B-C-… then
    // repeat), matching the existing harness discipline.
    let mut per_engine: Vec<EngineAccum> = cfg
        .engines
        .iter()
        .map(|&kind| EngineAccum::new(kind))
        .collect();

    for round in 0..cfg.rounds {
        eprintln!("round {}/{}", round + 1, cfg.rounds);
        for accum in &mut per_engine {
            let mut engine = accum.kind.make();
            // instant-distance ignores per-query ef; emit a single construction-ef row.
            let ef_for_engine: Vec<usize> = if accum.kind == EngineKind::InstantDistance {
                vec![cfg.params.ef_search]
            } else {
                cfg.ef_sweep.clone()
            };
            let stats = engine.build(
                &fixture.vectors,
                &fixture.ids,
                fixture.dim,
                cfg.metric,
                &cfg.params,
            )?;
            accum.build_secs.push(stats.build_secs);
            if let Some(rss) = stats.peak_rss_bytes {
                accum.peak_rss = Some(accum.peak_rss.map_or(rss, |p| p.max(rss)));
            }
            accum.notes.extend(stats.notes);

            if !cfg.skip_save && engine.supports_save() {
                let path = cfg.out_dir.join(format!(
                    "{}-{}-r{}.idx",
                    engine.name(),
                    cfg.metric.as_str(),
                    round
                ));
                match engine.save(&path) {
                    Ok(sz) => accum.file_size = Some(sz),
                    Err(e) => {
                        // Soft-skip only on the non-publish path; --markdown refuses
                        // incomplete file_size rows after the run.
                        accum.notes.push(format!("save failed: {e}"));
                    }
                }
            }

            for &ef in &ef_for_engine {
                let mut latencies = Vec::with_capacity(n_queries);
                let mut recalls = Vec::with_capacity(n_queries);
                for (qi, truth_ids) in truth.iter().enumerate().take(n_queries) {
                    let q = fixture.query(qi);
                    let timer = InstantTimer::start();
                    let got = engine.search(q, cfg.k, ef)?;
                    latencies.push(timer.elapsed_nanos() as f64);
                    recalls.push(recall_at_k(&got, truth_ids));
                }
                accum
                    .latency
                    .entry(ef)
                    .or_default()
                    .push(median_f64(&latencies));
                accum
                    .recall
                    .entry(ef)
                    .or_default()
                    .push(recalls.iter().sum::<f64>() / recalls.len() as f64);
            }

            if !cfg.skip_delete && engine.supports_delete() && !fixture.ids.is_empty() {
                let victim = fixture.ids[0];
                let victim_vec = &fixture.vectors[..fixture.dim];
                // Prefer the victim's own vector so pre-delete membership is meaningful.
                let before = engine.search(victim_vec, cfg.k, cfg.params.ef_search)?;
                if !before.contains(&victim) {
                    accum.notes.push(format!(
                        "delete oracle failed: id {victim} not in top-{} of its own vector before delete",
                        cfg.k
                    ));
                    accum.delete_ok = Some(false);
                } else {
                    engine.remove(victim)?;
                    let after = engine.search(victim_vec, cfg.k, cfg.params.ef_search)?;
                    let ok = !after.contains(&victim);
                    accum.delete_ok = Some(accum.delete_ok.unwrap_or(true) && ok);
                }
            }
        }
    }

    let results = per_engine.into_iter().map(|a| a.into_result(cfg)).collect();

    Ok(ComparisonReport {
        commit: git_commit(),
        hostname: hostname(),
        hardware_label: std::env::var("VANEDB_COMPARE_HW").unwrap_or_else(|_| "unlabelled".into()),
        recorded_at_utc: utc_now(),
        fixture_role: cfg.fixture_role,
        fixture_sha256: fixture.sha256.clone(),
        fixture_n_docs: fixture.n_docs(),
        fixture_n_queries: n_queries,
        fixture_dim: fixture.dim,
        metric: cfg.metric.as_str().into(),
        k: cfg.k,
        params: ParamsOut {
            m: cfg.params.m,
            ef_construction: cfg.params.ef_construction,
            seed: cfg.params.seed,
        },
        results,
    })
}

struct EngineAccum {
    kind: EngineKind,
    build_secs: Vec<f64>,
    peak_rss: Option<u64>,
    file_size: Option<u64>,
    latency: std::collections::BTreeMap<usize, Vec<f64>>,
    recall: std::collections::BTreeMap<usize, Vec<f64>>,
    delete_ok: Option<bool>,
    notes: Vec<String>,
}

impl EngineAccum {
    fn new(kind: EngineKind) -> Self {
        Self {
            kind,
            build_secs: Vec::new(),
            peak_rss: None,
            file_size: None,
            latency: std::collections::BTreeMap::new(),
            recall: std::collections::BTreeMap::new(),
            delete_ok: None,
            notes: Vec::new(),
        }
    }

    fn into_result(self, cfg: &RunConfig) -> EngineResult {
        let engine = self.kind.make();
        let version = engine.version();
        let name = engine.name().to_string();
        drop(engine);

        let mut latency_ns_by_ef = Vec::new();
        for (ef, samples) in &self.latency {
            latency_ns_by_ef.push(LatencyRow {
                ef: *ef,
                median_ns: median_f64(samples),
                spread: spread_f64(samples),
            });
        }
        let mut recall_at_k_by_ef = Vec::new();
        for (ef, samples) in &self.recall {
            recall_at_k_by_ef.push(RecallRow {
                ef: *ef,
                mean_recall: median_f64(samples),
            });
        }

        EngineResult {
            engine: name,
            version,
            metric: cfg.metric.as_str().into(),
            build_secs_median: median_f64(&self.build_secs),
            build_secs_spread: spread_f64(&self.build_secs),
            peak_rss_bytes: self.peak_rss,
            file_size_bytes: self.file_size,
            latency_ns_by_ef,
            recall_at_k_by_ef,
            delete_ok: self.delete_ok,
            notes: dedupe_notes(self.notes),
            rounds: cfg.rounds,
        }
    }
}

fn dedupe_notes(notes: Vec<String>) -> Vec<String> {
    let mut out = Vec::new();
    for n in notes {
        if !out.iter().any(|e: &String| e == &n) {
            out.push(n);
        }
    }
    out
}

fn git_commit() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                Some(String::from_utf8_lossy(&o.stdout).trim().to_string())
            } else {
                None
            }
        })
        .unwrap_or_else(|| "unknown".into())
}

fn hostname() -> String {
    std::env::var("HOSTNAME")
        .or_else(|_| std::env::var("HOST"))
        .unwrap_or_else(|_| {
            std::fs::read_to_string("/etc/hostname")
                .map(|s| s.trim().to_string())
                .unwrap_or_else(|_| "unknown".into())
        })
}

fn utc_now() -> String {
    // Prefer GNU date; fall back to a coarse local stamp if unavailable.
    std::process::Command::new("date")
        .args(["-u", "+%Y-%m-%dT%H:%M:%SZ"])
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                Some(String::from_utf8_lossy(&o.stdout).trim().to_string())
            } else {
                None
            }
        })
        .unwrap_or_else(|| "unknown".into())
}

pub fn write_json_report(report: &ComparisonReport, path: &Path) -> Result<(), String> {
    let text = serde_json::to_string_pretty(report).map_err(|e| e.to_string())?;
    std::fs::write(path, text).map_err(|e| e.to_string())
}
