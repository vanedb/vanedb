//! Markdown helpers for COMPARISON.md scaffolding (not for inventing timings).

use crate::run::ComparisonReport;

/// Render a single machine's JSON report as a markdown section.
pub fn render_machine_section(report: &ComparisonReport) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "### {} ({}) — metric `{}`\n\n",
        report.hardware_label, report.hostname, report.metric
    ));
    out.push_str(&format!(
        "- Date (UTC) / commit: `{}` / `{}`\n",
        report.recorded_at_utc, report.commit
    ));
    out.push_str(&format!(
        "- Fixture: role={}, dim={}, n_docs={}, n_queries={} (measured {}), sha256=`{}`\n",
        report.fixture_role.as_str(),
        report.fixture_dim,
        report.fixture_n_docs,
        report.fixture_n_queries,
        report.queries_measured,
        report.fixture_sha256
    ));
    out.push_str(&format!(
        "- Params: M={}, ef_construction={}, k={}, seed={}\n\n",
        report.params.m, report.params.ef_construction, report.k, report.params.seed
    ));

    out.push_str(
        "| Engine | Version | Build (s, median) | Spread | Peak RSS | File size | Delete OK |\n",
    );
    out.push_str("|---|---|---:|---:|---:|---:|:---:|\n");
    for r in &report.results {
        out.push_str(&format!(
            "| {} | {} | {:.4} | {:.1}% | {} | {} | {} |\n",
            r.engine,
            r.version,
            r.build_secs_median,
            r.build_secs_spread * 100.0,
            fmt_bytes(r.peak_rss_bytes),
            fmt_bytes(r.file_size_bytes),
            match r.delete_ok {
                Some(true) => "yes",
                Some(false) => "FAIL",
                None => "n/a",
            }
        ));
    }
    out.push('\n');

    out.push_str("Recall@10 and latency by ef (median across rounds):\n\n");
    for r in &report.results {
        out.push_str(&format!("**{}**\n\n", r.engine));
        out.push_str("| ef | latency/query | spread | recall@10 (median) |\n");
        out.push_str("|---:|---:|---:|---:|\n");
        for (lat, rec) in r.latency_ns_by_ef.iter().zip(r.recall_at_k_by_ef.iter()) {
            out.push_str(&format!(
                "| {} | {} | {:.1}% | {:.3} |\n",
                lat.ef,
                fmt_ns(lat.median_ns),
                lat.spread * 100.0,
                rec.recall_median
            ));
        }
        out.push('\n');
        if !r.notes.is_empty() {
            out.push_str("Notes:\n");
            for n in &r.notes {
                out.push_str(&format!("- {n}\n"));
            }
            out.push('\n');
        }
    }
    out
}

fn fmt_bytes(v: Option<u64>) -> String {
    match v {
        None => "n/a".into(),
        Some(b) if b >= 1 << 30 => format!("{:.2} GiB", b as f64 / (1u64 << 30) as f64),
        Some(b) if b >= 1 << 20 => format!("{:.2} MiB", b as f64 / (1u64 << 20) as f64),
        Some(b) if b >= 1 << 10 => format!("{:.2} KiB", b as f64 / (1u64 << 10) as f64),
        Some(b) => format!("{b} B"),
    }
}

fn fmt_ns(ns: f64) -> String {
    if ns >= 1_000_000.0 {
        format!("{:.2} ms", ns / 1_000_000.0)
    } else if ns >= 1_000.0 {
        format!("{:.2} µs", ns / 1_000.0)
    } else {
        format!("{ns:.0} ns")
    }
}
