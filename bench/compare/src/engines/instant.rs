//! instant-distance adapter.
//!
//! `M` is a compile-time constant (32) in the crate; the shared harness M is
//! recorded as a note when it differs. No public delete API; file persistence
//! is not enabled in this harness (`with-serde` left off).

use std::path::Path;

use instant_distance::{Builder, HnswMap, Search};

use super::{BuildParams, BuildStats, Engine, MetricKind};
use crate::measure::{current_rss_bytes, InstantTimer};

#[derive(Clone)]
struct VecPoint {
    data: Vec<f32>,
    metric: MetricKind,
}

impl instant_distance::Point for VecPoint {
    fn distance(&self, other: &Self) -> f32 {
        match self.metric {
            MetricKind::L2 => l2_sq(&self.data, &other.data),
            MetricKind::Cosine => cosine_distance(&self.data, &other.data),
        }
    }
}

fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
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

pub struct InstantDistanceEngine {
    map: Option<HnswMap<VecPoint, u64>>,
    metric: MetricKind,
}

impl InstantDistanceEngine {
    pub fn new() -> Self {
        Self {
            map: None,
            metric: MetricKind::L2,
        }
    }
}

impl Default for InstantDistanceEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl Engine for InstantDistanceEngine {
    fn name(&self) -> &'static str {
        "instant-distance"
    }

    fn version(&self) -> String {
        "0.6.1".into()
    }

    fn supports_delete(&self) -> bool {
        false
    }

    fn supports_save(&self) -> bool {
        false
    }

    fn build(
        &mut self,
        vectors: &[f32],
        ids: &[u64],
        dim: usize,
        metric: MetricKind,
        params: &BuildParams,
    ) -> Result<BuildStats, String> {
        let mut points = Vec::with_capacity(ids.len());
        let mut values = Vec::with_capacity(ids.len());
        for (i, &id) in ids.iter().enumerate() {
            let row = vectors[i * dim..(i + 1) * dim].to_vec();
            points.push(VecPoint { data: row, metric });
            values.push(id);
        }
        let mut notes = vec![
            "M is a crate constant (32); cannot set harness M".into(),
            format!(
                "ef_construction={} ef_search={} seed={}",
                params.ef_construction, params.ef_search, params.seed
            ),
        ];
        if params.m != 32 {
            notes.push(format!(
                "requested M={} ignored; instant-distance uses M=32",
                params.m
            ));
        }
        notes.push(
            "per-query ef is ignored; ef-sweep latency/recall rows duplicate construction ef_search"
                .into(),
        );

        let rss_before = current_rss_bytes();
        let timer = InstantTimer::start();
        let map = Builder::default()
            .ef_construction(params.ef_construction)
            .ef_search(params.ef_search)
            .seed(params.seed)
            .build(points, values);
        let build_secs = timer.elapsed_secs();
        let rss_after = current_rss_bytes();
        self.metric = metric;
        self.map = Some(map);
        Ok(BuildStats {
            build_secs,
            peak_rss_bytes: rss_delta(rss_before, rss_after),
            notes,
        })
    }

    fn search(&self, query: &[f32], k: usize, ef: usize) -> Result<Vec<u64>, String> {
        let map = self.map.as_ref().ok_or("instant-distance: not built")?;
        // instant-distance fixes ef at construction; per-query ef is ignored.
        // Callers should treat ef-sweep rows for this engine as duplicates of
        // the construction ef_search (recorded in build notes).
        let _ = ef;
        let mut search = Search::default();
        let q = VecPoint {
            data: query.to_vec(),
            metric: self.metric,
        };
        Ok(map
            .search(&q, &mut search)
            .take(k)
            .map(|item| *item.value)
            .collect())
    }

    fn save(&self, _path: &Path) -> Result<u64, String> {
        Err("instant-distance: persistence not enabled in this harness".into())
    }

    fn remove(&mut self, _id: u64) -> Result<(), String> {
        Err("instant-distance: no delete API".into())
    }
}

fn rss_delta(before: Option<u64>, after: Option<u64>) -> Option<u64> {
    match (before, after) {
        (Some(b), Some(a)) if a >= b => Some(a - b),
        (_, after) => after,
    }
}
